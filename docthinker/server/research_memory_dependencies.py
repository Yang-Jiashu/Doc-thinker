"""Version-bound memory lineage for answers; no copied private text or model I/O.

Flatten a bounded set of direct and conversational dependencies. Every read
rechecks current revisions, event-time validity and live source dependencies;
deleting a note cannot reintroduce it through an older assistant response.
"""
from __future__ import annotations

import json
import math
import time
import sqlite3

from .research_store import ResearchLimitError, ResearchStore, ResearchStoreError, _identifier, _session

MAX_BINDINGS = 128
MAX_ANSWERS = 200


def source_answer_memory_live(db, session_id, answer_id, *, at=None, seen=None):
    """Connection-local transitive check, shared by memory and answer readers."""
    at = time.time() if at is None else at
    seen = seen if seen is not None else {'active': set(), 'memo': {}}
    key = (session_id, answer_id, at)
    if key in seen['memo']:
        return seen['memo'][key]
    # A request-local memo can contain many independent answers, while the
    # iterative answer-parent walker can have a long nonrecursive active path.
    # Neither is a memory dependency packet or Python recursion depth. Meter
    # only nested memory -> source-answer calls; retain the exact cycle guard
    # and the separate MAX_BINDINGS bound in each decoded memory packet.
    depth = seen.get('memory_depth', 0)
    if key in seen['active'] or depth >= MAX_BINDINGS:
        return False
    seen['active'].add(key)
    seen['memory_depth'] = depth + 1
    try:
        result = _source_answer_memory_live(db, session_id, answer_id, at=at, seen=seen)
        if result:
            from .research_answer_dependencies import answer_parents_live
            try:
                result = answer_parents_live(db, session_id, answer_id, at=at, seen=seen)
            except (ValueError, TypeError, KeyError, OverflowError, AttributeError, sqlite3.Error):
                result = False
        seen['memo'][key] = result
        return result
    finally:
        seen['active'].discard(key)
        seen['memory_depth'] = depth


def _source_answer_memory_live(db, session_id, answer_id, *, at, seen):
    if not MemoryDependencies._exists(db):
        row = None
    else:
        row = db.execute('SELECT bindings FROM research_answer_memory_dependencies '
                         'WHERE session_id=? AND answer_id=?', (session_id, answer_id)).fetchone()
    if row is None:
        legacy = db.execute('SELECT context_metadata_json FROM research_answers '
                            'WHERE session_id=? AND answer_id=?', (session_id, answer_id)).fetchone()
        if legacy is None:
            return False
        try:
            metadata = json.loads(legacy['context_metadata_json'])
            return metadata.get('used_memory_count', 0) == 0 and metadata.get('memory_dependency_required') is not True
        except (ValueError, TypeError, AttributeError):
            return False
    try:
        return _bindings_live(db, session_id, MemoryDependencies._decode(row['bindings']),
                              now=at, seen=seen)
    except (ValueError, TypeError, OverflowError):
        return False


def _bindings_live(db, sid, bindings, *, now, seen=None):
    if not bindings:
        return True
    if db is None:
        return False
    seen = seen if seen is not None else {'active': set(), 'memo': {}}
    wanted = list(dict.fromkeys((mid, revision, now if at is None else at)
                               for mid, revision, at in bindings))
    found, checked_sources = set(), {}
    # Exact composite-key lookup, <=271 SQL variables per bounded batch.
    for start in range(0, len(wanted), 90):
        batch = wanted[start:start + 90]
        sql = ('WITH wanted(memory_id,revision,at) AS (VALUES ' + ','.join('(?,?,?)' for _ in batch) + ') '
            'SELECT v.*,w.at AS checked_at FROM wanted w '
            'JOIN research_memory_versions v ON v.memory_id=w.memory_id AND v.revision=w.revision '
            'JOIN research_memories m ON m.memory_id=v.memory_id '
            "WHERE v.session_id=? AND m.session_id=v.session_id AND m.status IN ('active','archived') "
            "AND v.status NOT IN ('deleted','invalidated') AND v.valid_from<=w.at "
            'AND (v.valid_until IS NULL OR w.at<v.valid_until) '
            'AND NOT EXISTS (SELECT 1 FROM research_memory_versions newer '
            'WHERE newer.memory_id=v.memory_id AND newer.revision>v.revision '
            "AND newer.status NOT IN ('deleted','invalidated') AND newer.valid_from<=w.at)")
        rows = db.execute(sql, [value for row in batch for value in row] + [sid]).fetchall()
        for row in rows:
            key = (row['memory_id'], row['revision'], row['checked_at'])
            if key not in checked_sources:
                checked_sources[key] = ResearchStore._memory_sources_live(
                    db, row, at=row['checked_at'], seen=seen)
            if checked_sources[key]:
                found.add(key)
    return len(found) == len(wanted)


class MemoryDependencies:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _exists(db):
        return db is not None and db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='research_answer_memory_dependencies'").fetchone() is not None

    @staticmethod
    def _decode(raw):
        if type(raw) is not str or len(raw) > 32768:
            raise ResearchStoreError('Memory lineage unavailable')
        rows = json.loads(raw)
        if type(rows) is not list or not 1 <= len(rows) <= MAX_BINDINGS:
            raise ResearchStoreError('Memory lineage unavailable')
        result = []
        for row in rows:
            if type(row) is not list or len(row) != 3:
                raise ResearchStoreError('Memory lineage unavailable')
            mid, revision, at = row
            _identifier(mid)
            if type(revision) is not int or revision < 1 or (at is not None and (
                    type(at) not in (int, float) or not math.isfinite(at) or not 0 <= at <= 253402300799)):
                raise ResearchStoreError('Memory lineage unavailable')
            result.append((mid, revision, at))
        return result

    def collect(self, sid, memories, *, as_of=None, history_ids=()):
        _session(sid)
        if len(memories) > 20 or len(history_ids) > 6:
            raise ResearchLimitError('Memory lineage limit reached')
        bindings = []
        for memory in memories:
            bindings.append((memory['id'], memory['revision'], as_of))
        with self.store._read() as db:
            if self._exists(db) and history_ids:
                for identifier in history_ids:
                    _identifier(identifier)
                rows = db.execute('SELECT bindings FROM research_answer_memory_dependencies '
                    'WHERE session_id=? AND answer_id IN (' + ','.join('?' for _ in history_ids) + ')',
                    [sid, *history_ids]).fetchall()
                for row in rows:
                    bindings.extend(self._decode(row['bindings']))
        bindings = sorted(set(bindings), key=lambda row: (row[0], row[1], -1 if row[2] is None else row[2]))
        if len(bindings) > MAX_BINDINGS:
            # Never silently omit an old dependency or make another model call.
            raise ResearchLimitError('Memory lineage limit reached')
        if bindings:
            self._decode(json.dumps(bindings))
        return bindings

    def _live(self, db, sid, bindings):
        return _bindings_live(db, sid, bindings, now=self.store.clock())

    def live(self, sid, bindings):
        _session(sid)
        if bindings:
            self._decode(json.dumps(bindings))
        with self.store._read() as db:
            return self._live(db, sid, bindings)

    def record(self, sid, answer_id, bindings):
        _session(sid); _identifier(answer_id)
        if not bindings:
            return
        self._decode(json.dumps(bindings))
        with self.store._write() as db:
            db.execute('CREATE TABLE IF NOT EXISTS research_answer_memory_dependencies ('
                'answer_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, bindings TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS research_answer_memory_scope '
                'ON research_answer_memory_dependencies(session_id,answer_id)')
            if not db.execute('SELECT 1 FROM research_answers WHERE session_id=? AND answer_id=?',
                              (sid, answer_id)).fetchone():
                raise ResearchStoreError('Answer reference unavailable')
            existing = db.execute('SELECT session_id,bindings FROM research_answer_memory_dependencies '
                                  'WHERE answer_id=?', (answer_id,)).fetchone()
            raw = json.dumps(bindings, separators=(',', ':'))
            if existing and (existing['session_id'] != sid or existing['bindings'] != raw):
                raise ResearchStoreError('Memory lineage changed')
            db.execute('INSERT OR IGNORE INTO research_answer_memory_dependencies VALUES(?,?,?)',
                       (answer_id, sid, raw))

    def states(self, sid, answer_ids):
        _session(sid)
        if len(answer_ids) > MAX_ANSWERS:
            raise ResearchLimitError('Memory lineage read limit reached')
        if not answer_ids:
            return {}
        for identifier in answer_ids:
            _identifier(identifier)
        with self.store._read() as db:
            if not self._exists(db):
                return {}
            rows = db.execute('SELECT answer_id,bindings FROM research_answer_memory_dependencies '
                'WHERE session_id=? AND answer_id IN (' + ','.join('?' for _ in answer_ids) + ')',
                [sid, *answer_ids]).fetchall()
            result = {}
            for row in rows:
                try:
                    result[row['answer_id']] = self._live(db, sid, self._decode(row['bindings']))
                except (ValueError, TypeError, OverflowError):
                    result[row['answer_id']] = False
            return result
