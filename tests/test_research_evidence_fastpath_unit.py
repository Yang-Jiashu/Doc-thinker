"""Live, lossless evidence-index fast paths; all fixtures are synthetic/local."""
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import sqlite3
import threading

import pytest

from docthinker.server import research_evidence_index as evidence_module
from docthinker.server.research_evidence_index import INDEX_VERSION, ResearchEvidenceIndex
from docthinker.server.research_store import ResearchStore, ResearchStoreError
from tests.test_research_store_unit import ready


SID = '#00001'


@pytest.fixture
def indexed(tmp_path):
    store = ResearchStore(tmp_path)
    doc, sources = ready(store, content=b'Observation: grid ready. If grid ready then supply grows.')
    index = ResearchEvidenceIndex(store)
    index.ensure_project(SID)
    return store, index, doc, sources


def _legacy_ensure(index, sid):
    """The former warm ensure path, with identical schema and parser version."""
    docs = [row for row in index.store.list_documents(sid) if row['status'] == 'ready']
    with index.store._write() as db:
        index._schema(db)
        done = {row[0] for row in db.execute(
            'SELECT document_id FROM research_claim_documents WHERE session_id=? AND parser=?',
            (sid, INDEX_VERSION))}
    for doc in docs:
        if doc['id'] not in done:
            index.index_document(sid, doc['id'])


def _trace_connections(monkeypatch):
    original = sqlite3.connect
    connections, statements = [], []
    def connect(*args, **kwargs):
        db = original(*args, **kwargs)
        connections.append(db)
        db.set_trace_callback(statements.append)
        return db
    monkeypatch.setattr(sqlite3, 'connect', connect)
    return connections, statements


def test_warm_ensure_is_live_read_only_one_connection(indexed, monkeypatch):
    store, index, _, _ = indexed
    before = store.path.read_bytes()
    connections, statements = _trace_connections(monkeypatch)
    index.ensure_project(SID)
    assert len(connections) == 1
    assert len(statements) == 4
    assert all(statement.lstrip().upper().startswith('SELECT') for statement in statements)
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute('SELECT 1')
    assert store.path.read_bytes() == before
    def forbidden():
        raise AssertionError('A complete live index must not acquire a writer')
    monkeypatch.setattr(store, '_write', forbidden)
    index.ensure_project(SID)


def test_same_hot_output_with_less_sql_than_former_ensure(indexed, monkeypatch):
    store, index, _, _ = indexed
    connections, statements = _trace_connections(monkeypatch)
    _legacy_ensure(index, SID)
    old_connections, old_sql = len(connections), len(statements)
    connections.clear()
    statements.clear()
    index.ensure_project(SID)
    assert old_connections == 2 and len(connections) == 1
    assert len(statements) == 4 and old_sql >= 35
    with store._read() as db:
        assert db.execute('SELECT count(*) FROM research_claim_index').fetchone()[0] == 2


def test_hot_ensure_does_not_wait_for_another_reserved_writer(indexed):
    store, index, _, _ = indexed
    started, finished = threading.Event(), threading.Event()
    def former():
        started.set()
        _legacy_ensure(index, SID)
        finished.set()
    holder = sqlite3.connect(store.path)
    holder.execute('BEGIN IMMEDIATE')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            blocked = pool.submit(former)
            assert started.wait(1)
            # SQLite's reserved writer lock still permits current read-only
            # readers; the former BEGIN IMMEDIATE path cannot reserve it too.
            pool.submit(index.ensure_project, SID).result(timeout=1)
            assert not finished.wait(0.05)
            holder.rollback()
            blocked.result(timeout=2)
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.parametrize('name,kind', [
    ('research_claim_cases', 'INDEX'), ('research_claim_relations', 'INDEX'),
    ('research_claim_sources', 'INDEX'), ('research_atoms_lookup', 'INDEX'),
    ('research_atom_revoke', 'TRIGGER'), ('research_claim_revoke', 'TRIGGER'),
    ('research_answer_evidence', 'TABLE'), ('research_claim_documents', 'TABLE'),
    ('research_claim_atoms', 'TABLE'), ('research_claim_index', 'TABLE'),
])
def test_live_schema_loss_is_repaired_not_hidden_by_an_initialized_flag(indexed, name, kind):
    store, index, _, sources = indexed
    expected = index.candidates(SID, sources, 'deduction')
    with store._write() as db:
        db.execute('DROP ' + kind + ' ' + name)
    index.ensure_project(SID)
    assert index.candidates(SID, sources, 'deduction') == expected
    with store._read() as db:
        assert index._schema_ready(db)


def test_content_index_loss_invalidates_other_project_receipts_without_using_its_sources(indexed):
    store, index, _, _ = indexed
    _, other_sources = ready(store, session='#00002', content=b'Observation: private grid ready.')
    index.ensure_project('#00002')
    with store._write() as db:
        db.execute('DROP TABLE research_claim_atoms')
    index.ensure_project(SID)
    with store._read() as db:
        assert not db.execute('SELECT 1 FROM research_claim_documents WHERE session_id=?', ('#00002',)).fetchone()
    claims, _ = index.candidates('#00002', other_sources, 'deduction')
    assert claims and all(row['quote'] == 'Observation: private grid ready.' for row in claims)


def test_same_store_object_detects_database_replacement(indexed, tmp_path):
    store, index, _, _ = indexed
    other_root = tmp_path / 'replacement'
    other_root.mkdir(mode=0o700)
    replacement = ResearchStore(other_root)
    doc, sources = ready(replacement, content=b'Observation: replacement ready.')
    store.path.unlink()
    shutil.copyfile(replacement.path, store.path)
    os.chmod(store.path, 0o600)
    index.ensure_project(SID)
    claims, _ = index.candidates(SID, sources, 'deduction')
    assert {row['document_id'] for row in claims} == {doc['id']}
    assert all('replacement' in row['quote'] for row in claims)


def test_schema_failure_rolls_back_and_next_call_retries(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path)
    _, sources = ready(store, content=b'Observation: grid ready.')
    index = ResearchEvidenceIndex(store)
    original = ResearchEvidenceIndex._schema
    def failed(db):
        original(db)
        raise OSError('synthetic migration interruption')
    with monkeypatch.context() as patch:
        patch.setattr(ResearchEvidenceIndex, '_schema', staticmethod(failed))
        with pytest.raises(OSError, match='synthetic migration'):
            index.ensure_project(SID)
    with store._read() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='research_claim_documents'").fetchone()
    index.ensure_project(SID)
    assert index.candidates(SID, sources, 'deduction')[0]


def test_concurrent_cold_initialization_remains_idempotent(tmp_path):
    store = ResearchStore(tmp_path)
    ready(store, content=b'Observation: grid ready. If grid ready then supply grows.')
    index = ResearchEvidenceIndex(store)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(index.ensure_project, [SID] * 12))
    with store._read() as db:
        assert index._schema_ready(db)
        assert db.execute('SELECT count(*) FROM research_claim_index').fetchone()[0] == 2
        assert db.execute('SELECT count(*) FROM research_claim_documents').fetchone()[0] == 1


def test_parser_version_receipt_is_checked_on_every_hot_read(indexed, monkeypatch):
    store, index, doc, _ = indexed
    with store._write() as db:
        db.execute('UPDATE research_claim_documents SET parser=?', ('obsolete',))
    calls = []
    original = index.index_document
    def counted(*args):
        calls.append(args)
        original(*args)
    monkeypatch.setattr(index, 'index_document', counted)
    index.ensure_project(SID)
    assert calls == [(SID, doc['id'])]


def test_hot_path_still_validates_session_and_private_layout(indexed):
    store, index, _, _ = indexed
    with pytest.raises(ResearchStoreError):
        index.ensure_project('../#00001')
    os.chmod(store.directory, 0o755)
    try:
        with pytest.raises(ResearchStoreError):
            index.ensure_project(SID)
    finally:
        os.chmod(store.directory, 0o700)


def _uncached_atom_cases(db, sid, key, _cache):
    return db.execute(
        'SELECT DISTINCT i.case_id FROM research_claim_atoms a JOIN research_claim_index i '
        'ON i.claim_id=a.claim_id WHERE a.session_id=? AND a.atom=? LIMIT 13',
        (sid, key)).fetchall()


def _legacy_dependency_cases(db, sid, cases):
    """Frozen former FIFO/list/uncached-atom closure, including old budgets."""
    frontier, visited = list(cases), set()
    while frontier:
        case = frontier.pop(0)
        if case in visited:
            continue
        visited.add(case)
        rows = db.execute('SELECT payload FROM research_claim_index WHERE session_id=? AND case_id=? LIMIT 193', (sid, case)).fetchall()
        if len(rows) > 192 or len(visited) > 12:
            return None
        for row in rows:
            claim = evidence_module.json.loads(row['payload'])
            for atom in evidence_module._atoms(claim):
                variants = [atom]
                opposite = {'increases': 'reduces', 'reduces': 'increases', 'state_increase': 'state_decrease', 'state_decrease': 'state_increase'}.get(atom['relation'])
                if opposite:
                    variants.append({**atom, 'relation': opposite})
                for variant in variants:
                    related = _uncached_atom_cases(db, sid, evidence_module._atom_key(claim, variant), None)
                    frontier.extend(item[0] for item in related if item[0] not in visited and item[0] not in frontier)
    return sorted(visited)


def test_repeated_atom_join_is_reused_without_changing_whole_candidate_result(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path)
    _, sources = ready(store, content=('Solar reduces imports.\n' * 100).encode())
    index = ResearchEvidenceIndex(store)
    index.ensure_project(SID)
    _, statements = _trace_connections(monkeypatch)
    actual = index.candidates(SID, sources, 'deduction')
    memo_joins = sum('SELECT DISTINCT i.case_id FROM research_claim_atoms' in sql for sql in statements)
    memo_versions = sum(sql == 'PRAGMA data_version' for sql in statements)
    assert memo_joins == 2 and memo_versions == 200
    statements.clear()
    with monkeypatch.context() as patch:
        patch.setattr(evidence_module, '_dependency_cases', _legacy_dependency_cases)
        expected = index.candidates(SID, sources, 'deduction')
    old_joins = sum('SELECT DISTINCT i.case_id FROM research_claim_atoms' in sql for sql in statements)
    assert old_joins == 200
    assert actual == expected
    assert len(actual[0]) == 100
    assert not any(sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'BEGIN')) for sql in statements)


@pytest.mark.parametrize('mutation', ['delete', 'add'])
def test_atom_memo_invalidates_after_other_connection_commit(indexed, mutation):
    store, index, doc, _ = indexed
    with store._read() as db:
        claim = evidence_module.json.loads(db.execute(
            'SELECT payload FROM research_claim_index WHERE document_id=? ORDER BY claim_id LIMIT 1',
            (doc['id'],)).fetchone()['payload'])
        atom = evidence_module._atoms(claim)[0]
        key = evidence_module._atom_key(claim, atom)
        cache = {'data_version': None, 'cases': {}}
        first = evidence_module._related_atom_cases(db, SID, key, cache)
        assert first
        # data_version is connection-local; do not compare numbers from two
        # connections. An external commit invalidates the same reader's memo.
        if mutation == 'delete':
            with store._write() as writer:
                writer.execute('DELETE FROM research_claim_index WHERE document_id=?', (doc['id'],))
            assert evidence_module._related_atom_cases(db, SID, key, cache) == []
        else:
            added, _ = ready(store, content=b'Observation: grid ready.', filename='new.txt')
            index.index_document(SID, added['id'])
            assert len(evidence_module._related_atom_cases(db, SID, key, cache)) == len(first) + 1
        assert not db.in_transaction


def test_separate_candidate_request_never_reuses_revoked_atom_results(indexed):
    store, index, doc, sources = indexed
    assert index.candidates(SID, sources, 'deduction')[0]
    store.delete_document(SID, doc['id'])
    assert index.candidates(SID, sources, 'deduction')[0] == []


@pytest.mark.parametrize('mode', ['deduction', 'analogy', 'induction'])
def test_candidate_outputs_equal_uncached_for_scoped_and_opposing_sources(tmp_path, monkeypatch, mode):
    store = ResearchStore(tmp_path)
    _, seeds = ready(store, content=b'If solar reduces imports then security improves.', filename='rule.txt')
    for i, text in enumerate([
        b'Solar reduces imports.', b'Solar increases imports.',
        b'Case: elsewhere\nSolar reduces imports.', b'Supplier diversity reduces concentration risk.',
    ]):
        ready(store, content=text, filename=f'case-{i}.txt')
    ready(store, session='#00002', content=b'Solar reduces imports.', filename='other-project.txt')
    index = ResearchEvidenceIndex(store)
    actual = index.candidates(SID, seeds, mode)
    with monkeypatch.context() as patch:
        patch.setattr(evidence_module, '_dependency_cases', _legacy_dependency_cases)
        expected = index.candidates(SID, seeds, mode)
    assert actual == expected
    assert all('elsewhere' == row['scope'] or 'project' == row['scope'] for row in actual[0])
    assert all(row['document_id'] in {doc['id'] for doc in store.list_documents(SID)} for row in actual[0])


def test_dependency_budget_is_not_relaxed_by_atom_memo(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path)
    sources = []
    for i in range(13):
        _, source = ready(store, content=f'Observation: grid ready.\n{i}'.encode(), filename=f'{i}.txt')
        sources.extend(source)
    index = ResearchEvidenceIndex(store)
    actual = index.candidates(SID, sources[:1], 'deduction')
    with monkeypatch.context() as patch:
        patch.setattr(evidence_module, '_dependency_cases', _legacy_dependency_cases)
        expected = index.candidates(SID, sources[:1], 'deduction')
    assert actual == expected
    assert actual[0] == [] and actual[1][0]['code'] == 'dependency_budget_reached'


class _Rows(list):
    def fetchall(self):
        return self
    def fetchone(self):
        return self[0] if self else None


class _ClosureDB:
    """Deterministic SQL result fixture to expose FIFO ordering/cycles/budgets."""
    def __init__(self, edges, counts):
        self.edges, self.counts, self.visits = edges, counts, []
    def execute(self, sql, args=()):
        if sql == 'PRAGMA data_version':
            return _Rows([(1,)])
        if sql.startswith('SELECT payload'):
            case = args[1]
            self.visits.append(case)
            atom = {'scope': 'project', 'kind': 'observation', 'subject': case,
                    'relation': 'requires', 'object': 'ready', 'conditions': []}
            return _Rows([{'payload': evidence_module.json.dumps(atom)}] * self.counts.get(case, 1))
        if sql.startswith('SELECT DISTINCT i.case_id'):
            case = evidence_module.json.loads(args[1])[1]
            return _Rows([(item,) for item in dict.fromkeys(self.edges.get(case, []))][:13])
        raise AssertionError('Unexpected closure operation')


@pytest.mark.parametrize('edges,counts', [
    ({'root': ['left', 'right'], 'left': ['goal'], 'right': ['goal'], 'goal': ['root']}, {}),
    ({'root': ['left', 'left', 'right'], 'left': ['right', 'root']}, {'root': 100}),
    ({'root': [str(i) for i in range(13)]}, {}),
    ({'root': ['large']}, {'large': 193}),
    ({'root': ['missing']}, {'missing': 0}),
])
def test_deque_pending_set_preserves_original_fifo_order_and_budget(edges, counts):
    old, new = _ClosureDB(edges, counts), _ClosureDB(edges, counts)
    expected = _legacy_dependency_cases(old, SID, ['root'])
    assert evidence_module._dependency_cases(new, SID, ['root']) == expected
    assert new.visits == old.visits
