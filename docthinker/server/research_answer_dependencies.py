"""Reference-only lineage for reused analyses, including cross-phase handoffs.

No answer text or inferred fact is copied here. A current source/confirmed
memory check propagates through the actual parent answers, not just citations
that happened to survive the child's final packet.
"""
from __future__ import annotations

import json
from pathlib import Path
from contextlib import closing
import sqlite3
import time

from .research_store import ResearchStore, ResearchStoreError, _identifier, _session

MAX_PARENTS = 12


def answer_parents_live(db, sid, answer_id, *, at, seen):
    if db is None:
        return False
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='research_answer_parents'").fetchone()
    proof_exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='research_answer_evidence'").fetchone()
    deadline, inspected_bytes = time.monotonic() + .5, 0

    def node(identifier, *, is_parent):
        nonlocal inspected_bytes
        if time.monotonic() > deadline:
            return None
        metadata = db.execute('SELECT context_metadata_json FROM research_answers WHERE session_id=? AND answer_id=?',
                              (sid, identifier)).fetchone()
        if metadata is None:
            return None
        context = json.loads(metadata[0])
        if type(context) is not dict:
            return None
        row = db.execute('SELECT * FROM research_answer_parents WHERE session_id=? AND answer_id=?',
                         (sid, identifier)).fetchone() if exists else None
        inspected_bytes += len(metadata[0]) + (len(row['parents']) + len(row['task_refs']) if row and 'task_refs' in row.keys() else 0)
        if inspected_bytes > 8 * 1024 * 1024:
            return None
        if (row is not None or is_parent) and (not proof_exists or not db.execute(
                'SELECT 1 FROM research_answer_evidence WHERE session_id=? AND answer_id=?', (sid, identifier)).fetchone()):
            return None
        sources = ResearchStore._memory_answer_sources(db, sid, identifier)
        if sources and db.execute('SELECT count(*) FROM research_chunks c JOIN research_documents d '
                'ON c.document_id=d.document_id AND c.session_id=d.session_id '
                "WHERE c.session_id=? AND d.state='ready' AND c.chunk_id IN (" + ','.join('?' for _ in sources) + ')',
                [sid, *sources]).fetchone()[0] != len(sources):
            return None
        if row is None:
            return [] if context.get('answer_dependency_required') is not True else None
        if 'task_refs' in row.keys() and not _task_refs_live(db, sid, json.loads(row['task_refs'])):
            return None
        return AnswerDependencies.decode(row['parents'])

    parents = node(answer_id, is_parent=False)
    if parents is None:
        return False
    # Iterative post-order DFS avoids Python recursion and a 128-stage lifetime
    # wall. Bounds apply to actual inspected bytes/time, not task age/depth.
    # The shared request-local memo also avoids rechecking diamond ancestors.
    from .research_memory_dependencies import _source_answer_memory_live
    stack, owned = [(parent, False) for parent in reversed(parents)], set()
    try:
        while stack:
            parent, exiting = stack.pop()
            key = (sid, parent, at)
            if exiting:
                seen['memo'][key] = True; seen['active'].discard(key); owned.discard(key)
                continue
            if key in seen['memo']:
                if not seen['memo'][key] or node(parent, is_parent=True) is None:
                    return False
                continue
            if key in seen['active'] or time.monotonic() > deadline:
                return False
            seen['active'].add(key); owned.add(key)
            if not _source_answer_memory_live(db, sid, parent, at=at, seen=seen):
                return False
            children = node(parent, is_parent=True)
            if children is None:
                return False
            stack.append((parent, True))
            stack.extend((child, False) for child in reversed(children))
        return True
    finally:
        seen['active'].difference_update(owned)


def _task_refs_live(db, sid, refs):
    if type(refs) is not list:
        return False
    if not refs:
        return True
    from .project_task_store import ProjectTaskStore
    if len(refs) > 6:
        return False
    location = next((row[2] for row in db.execute('PRAGMA database_list') if row[1] == 'main'), '')
    if not location:
        return False
    system_directory = Path(location).parent.parent
    for ref in refs:
        required = {'instance_id', 'task_id', 'step_id', 'version', 'answer_id'}
        if (type(ref) is not dict or not required <= set(ref) or set(ref) - required - {'storage_layout'}
                or type(ref.get('storage_layout', 'research_root_system')) is not str
                or ref.get('storage_layout', 'research_root_system') not in {'research_root_system', 'worker_data_system'}):
            return False
        if type(ref['version']) is not int or not 1 <= ref['version'] <= 32:
            return False
        # Exactly the layout bound by the real adapter. Never search another
        # owned directory after a failed lookup or migrate historical refs.
        directory = (system_directory.parent / 'data' / '_system'
                     if ref.get('storage_layout') == 'worker_data_system' else system_directory)
        task_store = ProjectTaskStore(directory, session_id=sid, instance_id=ref['instance_id'])
        if not task_store._file():
            return False
        with closing(sqlite3.connect(task_store.path.as_uri() + '?mode=ro', uri=True, timeout=.05)) as tasks:
            tasks.row_factory = sqlite3.Row
            tasks.execute('BEGIN')
            task_store._task(tasks, ref['task_id'])
            step = next((row for row in task_store._steps(tasks, ref['task_id']) if row['id'] == ref['step_id']), None)
            artifact = tasks.execute("SELECT payload FROM project_task_artifacts WHERE task_id=? AND step_id=? "
                "AND version=? AND status='current'", (ref['task_id'], ref['step_id'], ref['version'])).fetchone()
            if (step is None or step['status'] != 'completed' or step['artifact_version'] != ref['version']
                    or artifact is None or json.loads(artifact['payload']).get('answer_id') != ref['answer_id']):
                return False
    return True


class AnswerDependencies:
    def __init__(self, store):
        self.store = store

    def task_origins_live(self, sid, refs):
        _session(sid)
        with self.store._read() as db:
            if db is None:
                return not refs
            db.execute('BEGIN')
            try:
                return _task_refs_live(db, sid, refs)
            except (ValueError, sqlite3.Error, KeyError, TypeError):
                return False

    @staticmethod
    def decode(raw):
        if type(raw) is not str or len(raw) > 2048:
            raise ResearchStoreError('Analysis lineage unavailable')
        values = json.loads(raw)
        if type(values) is not list or not 1 <= len(values) <= MAX_PARENTS or len(set(values)) != len(values):
            raise ResearchStoreError('Invalid analysis lineage')
        for value in values:
            _identifier(value)
        return values

    def parents_live(self, sid, parents):
        _session(sid)
        if not parents:
            return True
        parents = self.decode(json.dumps(parents))
        from .research_memory_dependencies import source_answer_memory_live
        with self.store._read() as db:
            if db is None:
                return False
            db.execute('BEGIN')
            now, seen = self.store.clock(), {'active': set(), 'memo': {}}
            for parent in parents:
                if not source_answer_memory_live(db, sid, parent, at=now, seen=seen):
                    return False
                sources = self.store._memory_answer_sources(db, sid, parent)
                if sources and db.execute('SELECT count(*) FROM research_chunks c JOIN research_documents d '
                    'ON d.document_id=c.document_id AND d.session_id=c.session_id '
                    "WHERE c.session_id=? AND d.state='ready' AND c.chunk_id IN (" + ','.join('?' for _ in sources) + ')',
                    [sid, *sources]).fetchone()[0] != len(sources):
                    return False
            return True

    def record(self, sid, answer_id, parents, *, task_refs=None):
        if not parents:
            return
        _session(sid); _identifier(answer_id)
        parents = self.decode(json.dumps(parents))
        task_refs = [] if task_refs is None else task_refs
        required = {'instance_id', 'task_id', 'step_id', 'version', 'answer_id'}
        if (type(task_refs) is not list or len(task_refs) > 6
                or any(type(ref) is not dict or not required <= set(ref) or set(ref) - required - {'storage_layout'}
                       or type(ref.get('storage_layout', 'research_root_system')) is not str
                       or ref.get('storage_layout', 'research_root_system') not in {'research_root_system', 'worker_data_system'}
                       or ref.get('answer_id') not in parents for ref in task_refs)
                or len(json.dumps(task_refs).encode()) > 4096):
            raise ResearchStoreError('Task origins differ from analysis parents')
        if answer_id in parents:
            raise ResearchStoreError('Cyclic analysis lineage')
        with self.store._write() as db:
            db.execute('CREATE TABLE IF NOT EXISTS research_answer_parents '
                       '(answer_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,parents TEXT NOT NULL)')
            if 'task_refs' not in {row[1] for row in db.execute('PRAGMA table_info(research_answer_parents)')}:
                db.execute("ALTER TABLE research_answer_parents ADD COLUMN task_refs TEXT NOT NULL DEFAULT '[]'")
            db.execute('CREATE INDEX IF NOT EXISTS research_answer_parent_scope ON research_answer_parents(session_id,answer_id)')
            for identifier in [answer_id, *parents]:
                if not db.execute('SELECT 1 FROM research_answers WHERE session_id=? AND answer_id=?', (sid, identifier)).fetchone():
                    raise ResearchStoreError('Analysis parent unavailable')
            raw = json.dumps(parents, separators=(',', ':'))
            task_raw = json.dumps(task_refs, separators=(',', ':'), sort_keys=True)
            old = db.execute('SELECT session_id,parents,task_refs FROM research_answer_parents WHERE answer_id=?', (answer_id,)).fetchone()
            if old is not None and tuple(old) != (sid, raw, task_raw):
                raise ResearchStoreError('Analysis parent binding changed')
            db.execute('INSERT OR IGNORE INTO research_answer_parents VALUES(?,?,?,?)', (answer_id, sid, raw, task_raw))

    def states(self, sid, answer_ids):
        _session(sid)
        if len(answer_ids) > 200:
            raise ResearchStoreError('Analysis lineage read limit reached')
        from .research_memory_dependencies import source_answer_memory_live
        with self.store._read() as db:
            if db is None:
                return {}
            db.execute('BEGIN')
            now, seen = self.store.clock(), {'active': set(), 'memo': {}}
            return {identifier: source_answer_memory_live(db, sid, identifier, at=now, seen=seen)
                    for identifier in answer_ids}
