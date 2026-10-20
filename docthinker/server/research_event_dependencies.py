"""Current event/relation lineage; reference-only and connection-local.

No event text is copied into the dependency ledger. Event context is explicit
user-provided data, not proof of facts, causality or authority to execute.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time

from .research_store import ResearchStoreError, _identifier, _session


def _raw(bindings):
    return json.dumps(bindings, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _checked(bindings):
    if type(bindings) is not dict or set(bindings) != {'events', 'relations', 'source_scope_sha256', 'memory_scope_sha256'}:
        raise ResearchStoreError('Invalid event dependency packet')
    events, relations = bindings['events'], bindings['relations']
    if type(events) is not list or len(events) > 8 or type(relations) is not list or len(relations) > 12:
        raise ResearchStoreError('Event dependency limit reached')
    digest = lambda value: type(value) is str and re.fullmatch('[0-9a-f]{64}', value)
    for key in ('source_scope_sha256', 'memory_scope_sha256'):
        if bindings[key] is not None and not digest(bindings[key]):
            raise ResearchStoreError('Invalid event scope binding')
    found = {}
    for event in events:
        if (type(event) is not dict or set(event) != {'id', 'revision', 'sha256'}
                or type(event['revision']) is not int or event['revision'] < 1 or not digest(event['sha256'])):
            raise ResearchStoreError('Invalid event revision binding')
        _identifier(event['id'])
        if event['id'] in found:
            raise ResearchStoreError('Duplicate event binding')
        found[event['id']] = event['revision']
    ids = set()
    for relation in relations:
        if (type(relation) is not dict or set(relation) != {'id', 'revision', 'source_id', 'source_revision',
                'target_id', 'target_revision', 'sha256'} or type(relation['revision']) is not int
                or relation['revision'] != 1 or not digest(relation['sha256'])
                or type(relation['source_id']) is not str or type(relation['target_id']) is not str
                or type(relation['source_revision']) is not int or type(relation['target_revision']) is not int
                or found.get(relation['source_id']) != relation['source_revision']
                or found.get(relation['target_id']) != relation['target_revision']):
            raise ResearchStoreError('Invalid event relation binding')
        _identifier(relation['id']); _identifier(relation['source_id']); _identifier(relation['target_id'])
        if relation['id'] in ids:
            raise ResearchStoreError('Duplicate relation binding')
        ids.add(relation['id'])
    raw = _raw(bindings)
    if len(raw.encode()) > 6000:
        raise ResearchStoreError('Event dependency byte limit reached')
    return json.loads(raw)


def event_bindings_live(db, sid, bindings, *, store=None, at=None, seen=None):
    try:
        _session(sid)
        checked = _checked(bindings)
        from .personal_events import PersonalEvents
        return PersonalEvents._event_bindings_live(db, sid, checked, store=store,
            at=(store.clock() if store is not None else time.time()) if at is None else at, seen=seen)
    except (ValueError, TypeError, KeyError, OverflowError, sqlite3.Error):
        return False


def answer_events_live(db, sid, answer_id, *, at=None, seen=None):
    """Missing required ledger is unavailable, not permission to reuse prose."""
    if db is None:
        return False
    row = db.execute('SELECT context_metadata_json FROM research_answers WHERE session_id=? AND answer_id=?',
                     (sid, answer_id)).fetchone()
    if row is None:
        return False
    try:
        metadata = json.loads(row['context_metadata_json'])
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='research_answer_event_dependencies'").fetchone()
        saved = db.execute('SELECT bindings FROM research_answer_event_dependencies WHERE session_id=? AND answer_id=?',
                           (sid, answer_id)).fetchone() if exists else None
        if saved is None:
            return (metadata.get('event_dependency_required') is not True
                    and metadata.get('used_event_count', 0) == 0 and metadata.get('used_event_relation_count', 0) == 0)
        bindings = _checked(json.loads(saved['bindings']))
        expected = EventDependencies.context_metadata(bindings)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        return event_bindings_live(db, sid, bindings, at=at, seen=seen)
    except (ValueError, TypeError, KeyError, AttributeError, sqlite3.Error):
        return False


class EventDependencies:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def context_metadata(bindings):
        bindings = _checked(bindings)
        count, relations = len(bindings['events']), len(bindings['relations'])
        return {'event_dependency_required': bool(count or relations),
            'event_bindings_sha256': hashlib.sha256(_raw(bindings).encode()).hexdigest(),
            'used_event_count': count, 'used_event_relation_count': relations}

    def live(self, sid, bindings):
        _session(sid)
        with self.store._read(timeout=.05) as db:
            if db is not None:
                db.execute('BEGIN')
            return event_bindings_live(db, sid, bindings, store=self.store)

    def record(self, sid, answer_id, bindings):
        _session(sid); _identifier(answer_id)
        bindings = _checked(bindings)
        if not bindings['events'] and not bindings['relations']:
            return
        raw = _raw(bindings)
        with self.store._write() as db:
            self.store._require_storage(db)
            answer = db.execute('SELECT context_metadata_json FROM research_answers WHERE session_id=? AND answer_id=?', (sid, answer_id)).fetchone()
            if answer is None:
                raise ResearchStoreError('Event answer provenance unavailable')
            metadata = json.loads(answer['context_metadata_json'])
            if any(metadata.get(key) != value for key, value in self.context_metadata(bindings).items()):
                raise ResearchStoreError('Event answer metadata differs from its bindings')
            if not event_bindings_live(db, sid, bindings, store=self.store):
                raise ResearchStoreError('Event dependencies changed before publication')
            db.execute('CREATE TABLE IF NOT EXISTS research_answer_event_dependencies '
                '(answer_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,bindings TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS research_answer_event_scope ON research_answer_event_dependencies(session_id,answer_id)')
            previous = db.execute('SELECT session_id,bindings FROM research_answer_event_dependencies WHERE answer_id=?', (answer_id,)).fetchone()
            if previous is not None and tuple(previous) != (sid, raw):
                raise ResearchStoreError('Event answer binding changed')
            db.execute('INSERT OR IGNORE INTO research_answer_event_dependencies VALUES(?,?,?)', (answer_id, sid, raw))
            self.store._require_storage(db)

    def states(self, sid, answer_ids):
        _session(sid)
        if type(answer_ids) is not list or len(answer_ids) > 200:
            raise ResearchStoreError('Event lineage lookup limit reached')
        for answer_id in answer_ids:
            _identifier(answer_id)
        with self.store._read(timeout=.05) as db:
            if db is None:
                return {}
            db.execute('BEGIN')
            now, seen = self.store.clock(), {'active': set(), 'memo': {}}
            return {answer_id: answer_events_live(db, sid, answer_id, at=now, seen=seen) for answer_id in answer_ids}
