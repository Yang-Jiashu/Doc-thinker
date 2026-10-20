"""Derived, revocable claim index and answer proof metadata in the research DB.

No model call and no second memory database. Relations help discover cases whose
entity names differ; each selected case is loaded whole or omitted. Query and
proof windows are bounded, never a claim that the corpus was exhausted.
"""
from __future__ import annotations

from collections import deque
import json

from .research_provenance import hydrate_audit
from .research_reasoning import PARSER_VERSION, extract_document_claims
from .research_reasoning_context import proof_claim_ids
from .research_store import (ANSWER_READ_BATCH, MAX_ANSWER_LOOKUP, MAX_DOCUMENTS_PER_SESSION,
                             ResearchLimitError, _session, _identifier)
from .research_source_contract import MAX_ANSWER_SOURCES

INDEX_VERSION = PARSER_VERSION + '-document-atoms-v2'
_SCHEMA_OBJECTS = frozenset({
    ('research_claim_index', 'table', 'research_claim_index'),
    ('research_claim_cases', 'index', 'research_claim_index'),
    ('research_claim_relations', 'index', 'research_claim_index'),
    ('research_claim_sources', 'index', 'research_claim_index'),
    ('research_claim_atoms', 'table', 'research_claim_atoms'),
    ('research_atoms_lookup', 'index', 'research_claim_atoms'),
    ('research_atom_revoke', 'trigger', 'research_claim_index'),
    ('research_claim_documents', 'table', 'research_claim_documents'),
    ('research_answer_evidence', 'table', 'research_answer_evidence'),
    ('research_claim_revoke', 'trigger', 'research_chunks'),
})


def _atoms(claim):
    return [claim] if claim['kind'] == 'observation' else [*claim['premises'], claim['consequence']]


def _atom_key(claim, atom):
    return json.dumps([claim['scope'], atom['subject'], atom['relation'], atom['object'], atom['conditions']], ensure_ascii=False, separators=(',', ':'))


def _related_atom_cases(db, sid, key, cache):
    # candidates() itself performs SELECTs only. SQLite data_version changes on
    # this same connection when another connection commits, so a concurrent
    # upload/revocation invalidates this request-local derived lookup. Never
    # cache across connections, projects, or candidate requests.
    version = db.execute('PRAGMA data_version').fetchone()[0]
    if cache['data_version'] != version:
        cache['cases'].clear()
        cache['data_version'] = version
    scoped_key = (sid, key)
    if scoped_key not in cache['cases']:
        cache['cases'][scoped_key] = db.execute(
            "SELECT DISTINCT i.case_id FROM research_claim_atoms a JOIN research_claim_index i ON i.claim_id=a.claim_id WHERE a.session_id=? AND a.atom=? LIMIT 13",
            (sid, key)).fetchall()
    return cache['cases'][scoped_key]


def _dependency_cases(db, sid, cases):
    """Original bounded FIFO closure, with O(1) pending/visited membership."""
    frontier, pending, visited = deque(cases), set(cases), set()
    atom_cases = {'data_version': None, 'cases': {}}
    while frontier:
        case = frontier.popleft()
        pending.discard(case)
        if case in visited:
            continue
        visited.add(case)
        rows = db.execute("SELECT payload FROM research_claim_index WHERE session_id=? AND case_id=? LIMIT 193", (sid, case)).fetchall()
        if len(rows) > 192 or len(visited) > 12:
            return None
        for row in rows:
            claim = json.loads(row['payload'])
            for atom in _atoms(claim):
                variants = [atom]
                opposite = {'increases': 'reduces', 'reduces': 'increases', 'state_increase': 'state_decrease', 'state_decrease': 'state_increase'}.get(atom['relation'])
                if opposite:
                    variants.append({**atom, 'relation': opposite})
                for variant in variants:
                    related = _related_atom_cases(db, sid, _atom_key(claim, variant), atom_cases)
                    for item in related:
                        other = item[0]
                        if other not in visited and other not in pending:
                            frontier.append(other)
                            pending.add(other)
    return sorted(visited)


class ResearchEvidenceIndex:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _schema_objects(db):
        # Inspect the live connection, not a process-local "initialized" flag.
        # A replaced database or a dropped revocation trigger must be detected
        # on the next call, including after a failed/concurrent first migration.
        marks = ','.join('?' for _ in _SCHEMA_OBJECTS)
        return {tuple(row) for row in db.execute(
            'SELECT name,type,tbl_name FROM sqlite_master WHERE name IN (' + marks + ')',
            tuple(item[0] for item in _SCHEMA_OBJECTS))}

    @classmethod
    def _schema_ready(cls, db):
        return cls._schema_objects(db) == _SCHEMA_OBJECTS

    @classmethod
    def _ensure_schema(cls, db):
        objects = cls._schema_objects(db)
        if objects != _SCHEMA_OBJECTS:
            cls._schema(db)
            # Lost derived content cannot retain successful parser markers.
            # Invalidate only the derived completion receipts, transactionally;
            # authoritative documents/answer proofs remain untouched. Other
            # projects will rebuild from their own live sources on demand.
            if not {('research_claim_index', 'table', 'research_claim_index'),
                    ('research_claim_atoms', 'table', 'research_claim_atoms')} <= objects:
                db.execute('DELETE FROM research_claim_documents')

    @staticmethod
    def _schema(db):
        db.execute("CREATE TABLE IF NOT EXISTS research_claim_index (claim_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_id TEXT NOT NULL, document_id TEXT NOT NULL, case_id TEXT NOT NULL, relation TEXT NOT NULL, payload TEXT NOT NULL)")
        db.execute("CREATE INDEX IF NOT EXISTS research_claim_cases ON research_claim_index(session_id,case_id)")
        db.execute("CREATE INDEX IF NOT EXISTS research_claim_relations ON research_claim_index(session_id,relation,case_id)")
        db.execute("CREATE INDEX IF NOT EXISTS research_claim_sources ON research_claim_index(source_id)")
        db.execute("CREATE TABLE IF NOT EXISTS research_claim_atoms (claim_id TEXT NOT NULL, session_id TEXT NOT NULL, atom TEXT NOT NULL, PRIMARY KEY(claim_id,atom))")
        db.execute("CREATE INDEX IF NOT EXISTS research_atoms_lookup ON research_claim_atoms(session_id,atom)")
        db.execute("CREATE TRIGGER IF NOT EXISTS research_atom_revoke AFTER DELETE ON research_claim_index BEGIN DELETE FROM research_claim_atoms WHERE claim_id=OLD.claim_id; END")
        db.execute("CREATE TABLE IF NOT EXISTS research_claim_documents (document_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, parser TEXT NOT NULL, limited INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS research_answer_evidence (answer_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, payload TEXT NOT NULL)")
        db.execute("CREATE TRIGGER IF NOT EXISTS research_claim_revoke AFTER DELETE ON research_chunks BEGIN DELETE FROM research_claim_index WHERE source_id=OLD.chunk_id; DELETE FROM research_claim_documents WHERE document_id=OLD.document_id; END")

    def ensure_project(self, sid):
        _session(sid)
        # Preserve list_documents' ordering/window before filtering ready rows.
        # A warm complete project needs neither DDL nor a BEGIN IMMEDIATE writer
        # reservation; this is also one live connection instead of two.
        with self.store._read() as db:
            documents = [] if db is None else [row['document_id'] for row in db.execute(
                "SELECT document_id,state FROM research_documents WHERE session_id=? "
                "AND state!='deleted' ORDER BY created_at,document_id LIMIT ?",
                (sid, MAX_DOCUMENTS_PER_SESSION)) if row['state'] == 'ready']
            indexed = None if db is None or not self._schema_ready(db) else {
                row[0] for row in db.execute(
                    "SELECT document_id FROM research_claim_documents WHERE session_id=? AND parser=?",
                    (sid, INDEX_VERSION))}
        if indexed is None:
            # First initialization/migration remains transactional and is
            # rechecked under the existing store write lock for other callers.
            with self.store._write() as db:
                self._ensure_schema(db)
                indexed = {row[0] for row in db.execute(
                    "SELECT document_id FROM research_claim_documents WHERE session_id=? AND parser=?",
                    (sid, INDEX_VERSION))}
        for document_id in documents:
            if document_id not in indexed:
                self.index_document(sid, document_id)

    def index_document(self, sid, document_id):
        chunks = self.store.source_chunks(sid, document_id=document_id)
        extracted = extract_document_claims(chunks, max_claims=1536)
        claims, limited = extracted['claims'], extracted['limited']
        with self.store._write() as db:
            self._ensure_schema(db)
            self.store._require_storage(db)
            live = db.execute("SELECT 1 FROM research_documents WHERE session_id=? AND document_id=? AND state='ready'", (sid, document_id)).fetchone()
            if not live:
                return
            db.execute("DELETE FROM research_claim_index WHERE document_id=? AND session_id=?", (document_id, sid))
            for claim in claims:
                payload = {k: v for k, v in claim.items() if k != 'quote'}
                relation = claim.get('relation') or claim.get('consequence', {}).get('relation', '')
                db.execute("INSERT INTO research_claim_index VALUES (?,?,?,?,?,?,?)", (claim['id'], sid, claim['source_ids'][0], document_id, claim['case_id'], relation, json.dumps(payload, ensure_ascii=False)))
                for atom in _atoms(claim):
                    db.execute("INSERT OR IGNORE INTO research_claim_atoms VALUES (?,?,?)", (claim['id'], sid, _atom_key(claim, atom)))
            db.execute("INSERT OR REPLACE INTO research_claim_documents VALUES (?,?,?,?)", (document_id, sid, INDEX_VERSION, int(limited)))
            self.store._require_storage(db)

    def candidates(self, sid, seed_sources, mode):
        self.ensure_project(sid)
        seed_ids = [row['id'] for row in seed_sources][:8]
        if not seed_ids:
            return [], [{'code': 'no_reasoning_seed', 'message': '没有找到问题对应的原文起点，未任意选择其他案例推理。'}]
        with self.store._read() as db:
            marks = ','.join('?' for _ in seed_ids)
            seeds = db.execute(f"SELECT case_id,relation FROM research_claim_index WHERE session_id=? AND source_id IN ({marks}) ORDER BY case_id LIMIT 192", (sid, *seed_ids)).fetchall()
            cases = list(dict.fromkeys(row['case_id'] for row in seeds))
            relations = sorted({row['relation'] for row in seeds})
            if mode == 'induction':
                relations = sorted(set(relations) | ({'increases', 'reduces'} if {'increases', 'reduces'} & set(relations) else set()))
            if mode in {'analogy', 'induction'} and relations:
                marks = ','.join('?' for _ in relations)
                related = db.execute(f"SELECT case_id,COUNT(DISTINCT relation) AS matches FROM research_claim_index WHERE session_id=? AND relation IN ({marks}) GROUP BY case_id ORDER BY matches DESC,case_id LIMIT 24", (sid, *relations)).fetchall()
                cases += [row['case_id'] for row in related if row['case_id'] not in cases]
            claims, gaps = [], []
            if mode == 'deduction':
                # Exact scoped atom closure retrieves rules, missing premises,
                # and contradictory observations even without query-word overlap.
                cases = _dependency_cases(db, sid, cases)
                if cases is None:
                    return [], [{'code': 'dependency_budget_reached', 'message': '完整前提闭包超过预算，本次未使用截断前提推导。'}]
            if len(cases) > 12:
                gaps.append({'code': 'case_budget_reached', 'message': '仅比较本次候选案例窗口，并未穷尽项目。'})
            for case in cases[:12]:
                rows = db.execute("SELECT i.payload,c.text FROM research_claim_index i JOIN research_chunks c ON c.chunk_id=i.source_id JOIN research_documents d ON d.document_id=c.document_id WHERE i.session_id=? AND c.session_id=? AND d.session_id=? AND d.state='ready' AND i.case_id=? ORDER BY i.claim_id LIMIT 193", (sid, sid, sid, case)).fetchall()
                if len(rows) + len(claims) > 192:
                    if mode == 'deduction':
                        return [], [{'code': 'dependency_budget_reached', 'message': '完整前提闭包超过预算，本次未使用截断前提推导。'}]
                    gaps.append({'code': 'case_omitted_as_whole', 'message': '某案例超过推理预算，已整体省略，未把截断案例当成完整证据。'})
                    continue
                for row in rows:
                    claim = json.loads(row['payload'])
                    claim['quote'] = row['text'][claim['excerpt_start']:claim['excerpt_end']]
                    if claim['quote']:
                        claims.append(claim)
            if db.execute("SELECT 1 FROM research_claim_documents WHERE session_id=? AND limited=1 LIMIT 1", (sid,)).fetchone():
                gaps.append({'code': 'index_window_reached', 'message': '部分文档超过结构化索引窗口；不能将未找到当作不存在。'})
        return claims, gaps[:8]

    def record(self, sid, answer_id, *, reasoning, audit, memory_as_of, reasoning_mode, knowledge_skill=None):
        clean_audit = {**audit, 'claims': [{**claim, 'source_refs': [{k: v for k, v in ref.items() if k != 'quote'} for ref in claim['source_refs']]} for claim in audit['claims']]}
        clean_reasoning = {**reasoning, 'claims': [{k: v for k, v in claim.items() if k != 'quote'} for claim in reasoning.get('claims', [])]}
        clean_reasoning['conclusions'] = [{**item, 'proof': [{k: v for k, v in node.items() if k != 'text' or node.get('kind') == 'deduced'} for node in item.get('proof', [])]} for item in reasoning.get('conclusions', [])]
        payload = json.dumps({'reasoning': clean_reasoning, 'claim_audit': clean_audit,
                              'memory_as_of': memory_as_of, 'reasoning_mode': reasoning_mode,
                              # Versioned public method snapshot, not copied
                              # into user documents, memory, or the claim index.
                              **({'knowledge_skill': knowledge_skill} if knowledge_skill else {})}, ensure_ascii=False)
        if len(payload.encode()) > 128000:
            raise ValueError('Proof metadata budget exceeded')
        with self.store._write() as db:
            self._ensure_schema(db)
            self.store._require_storage(db)
            if not db.execute("SELECT 1 FROM research_answers WHERE answer_id=? AND session_id=?", (answer_id, sid)).fetchone():
                raise ValueError('Answer unavailable')
            db.execute("INSERT OR REPLACE INTO research_answer_evidence VALUES (?,?,?)", (answer_id, sid, payload))

    def hydrate(self, sid, answer_id, sources):
        with self.store._read() as db:
            if db is None or not db.execute("SELECT 1 FROM sqlite_master WHERE name='research_answer_evidence'").fetchone():
                return {}
            row = db.execute("SELECT payload FROM research_answer_evidence WHERE session_id=? AND answer_id=?", (sid, answer_id)).fetchone()
        if not row:
            return {}
        return self._hydrate_payload(row['payload'], sources)

    def hydrate_many(self, sid, sources_by_answer):
        """Rehydrate only current requested proofs; never cache across requests.

        The caller supplies each answer's actual live excerpts, not the union
        of all historical references. Existing offset/hash/revocation checks
        therefore remain local to that answer's own labelled source packet.
        """
        _session(sid)
        if not isinstance(sources_by_answer, dict) or len(sources_by_answer) > MAX_ANSWER_LOOKUP:
            raise ResearchLimitError('Too many answer references')
        for answer_id, sources in sources_by_answer.items():
            _identifier(answer_id)
            if not isinstance(sources, list) or len(sources) > MAX_ANSWER_SOURCES:
                raise ResearchLimitError('Too many answer source references')
        identifiers = list(sources_by_answer)
        if not identifiers:
            return {}
        with self.store._read() as db:
            if db is None or not db.execute("SELECT 1 FROM sqlite_master WHERE name='research_answer_evidence'").fetchone():
                return {}
            payloads = {}
            for offset in range(0, len(identifiers), ANSWER_READ_BATCH):
                batch = identifiers[offset:offset + ANSWER_READ_BATCH]
                rows = db.execute('SELECT answer_id,payload FROM research_answer_evidence WHERE session_id=? '
                                  'AND answer_id IN (' + ','.join('?' for _ in batch) + ')', [sid, *batch])
                payloads.update((row['answer_id'], row['payload']) for row in rows)
        return {answer_id: self._hydrate_payload(payload, sources_by_answer[answer_id])
                for answer_id, payload in payloads.items()}

    @staticmethod
    def _hydrate_payload(payload, sources):
        result = json.loads(payload)
        result['claim_audit'] = hydrate_audit(result['claim_audit'], sources)
        result['reasoning'] = hydrate_reasoning(result['reasoning'], sources)
        return result


def hydrate_reasoning(reasoning, sources):
    available = {row['id']: row for row in sources}
    conclusions, claims, missing = [], [], False
    for item in reasoning.get('claims', []):
        source = available.get(item['source_ids'][0])
        if source:
            left = item['excerpt_start'] - source.get('excerpt_start', 0)
            right = item['excerpt_end'] - source.get('excerpt_start', 0)
            if 0 <= left < right <= len(source['content']):
                claims.append({**item, 'quote': source['content'][left:right]})
    by_id = {claim['id']: claim for claim in claims}
    for item in reasoning.get('conclusions', []):
        if (item.get('source_ids') and all(source in available for source in item['source_ids'])
                and proof_claim_ids(item, by_id) is not None):
            conclusions.append({**item, 'proof': [{**node, 'text': by_id[node['id']]['quote']} if node['id'] in by_id else node for node in item.get('proof', [])]})
        else:
            missing = True
    trace = dict(reasoning.get('trace', {}))
    # Exploration scaffolds and graph paths are dependencies too, even when a
    # model (rather than the strict grammar) composed the final hypothesis.
    # Never retain old source-derived trace text after a source is withdrawn.
    for name, field in (('graph_retrieval', 'kept_source_ids'), ('hypothesis_support', 'source_ids'),
                        ('insight_runtime', 'source_ids'), ('trial_delivery', 'source_ids')):
        receipt = trace.get(name)
        statuses = {'valid', 'draft'} if name == 'trial_delivery' else {'ready', 'expanded', 'unchanged'}
        if isinstance(receipt, dict) and receipt.get('status') in statuses:
            expected = receipt.get(field, [])
            if any(identifier not in available for identifier in expected):
                missing = True
                if name == 'trial_delivery':
                    from .research_trial_delivery import unavailable_trial_receipt
                    trace[name] = unavailable_trial_receipt()
                else:
                    trace[name] = {'status': 'source_unavailable', 'model_calls': 0,
                                   'semantic_entailment_verified': False}
    result = {**reasoning, 'conclusions': conclusions, 'claims': claims, 'trace': trace}
    if missing:
        result.update(status='source_unavailable', gaps=[{'code': 'source_revoked', 'message': '部分前提来源已撤回，相关推理结论不再展示。'}, *reasoning.get('gaps', [])][:8])
    return result
