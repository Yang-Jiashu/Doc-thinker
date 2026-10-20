"""Private, source-bound domain events, not Harness execution logs.

This projection uses the existing ResearchStore connection and live-memory
checks. Events and relationships are explicit user assertions, not verified
facts, inferred causality, an authorization token, or a new execution engine.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import uuid

from .research_memory_dependencies import MemoryDependencies
from .research_store import ResearchLimitError, ResearchNotFoundError, ResearchRevisionConflictError, ResearchStoreError, _identifier, _session

EVENT_TYPES = frozenset({'decision', 'observation', 'task', 'meeting', 'milestone', 'change', 'incident', 'preference'})
RELATIONS = frozenset({'before', 'supersedes', 'depends_on', 'contradicts', 'related_to', 'reported_cause', 'hypothesis'})
MAX_ORIGINS = 16


def _text(value, cap):
    if type(value) is not str or not value.strip() or len(value) > cap or '\x00' in value:
        raise ResearchStoreError('Invalid personal event text')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise ResearchStoreError('Invalid personal event text') from None
    return value


def _normal(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _strings(values, cap=16):
    if type(values) not in (tuple, list) or len(values) > cap:
        raise ResearchLimitError('Personal event list limit reached')
    return list(dict.fromkeys(_text(value, 256) for value in values))


def _time(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not -62135596800 <= value <= 253402300799:
        raise ResearchStoreError('Invalid event occurrence time')
    return float(value)


class PersonalEvents:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _schema(db):
        # execute, not executescript: remain inside ResearchStore's transaction.
        statements = (
            'CREATE TABLE IF NOT EXISTS personal_entities (entity_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, identity_sha256 TEXT NOT NULL, UNIQUE(session_id,kind,identity_sha256))',
            'CREATE TABLE IF NOT EXISTS personal_entity_aliases (session_id TEXT NOT NULL, kind TEXT NOT NULL, alias TEXT NOT NULL, entity_id TEXT NOT NULL, PRIMARY KEY(session_id,kind,alias,entity_id))',
            'CREATE TABLE IF NOT EXISTS personal_events (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, revision INTEGER NOT NULL, actor_id TEXT, object_id TEXT, occurred_from REAL, occurred_until REAL, recorded_at REAL NOT NULL, payload TEXT NOT NULL)',
            'CREATE INDEX IF NOT EXISTS personal_event_actor_time ON personal_events(session_id,actor_id,occurred_from,event_id)',
            'CREATE INDEX IF NOT EXISTS personal_event_object_time ON personal_events(session_id,object_id,occurred_from,event_id)',
            'CREATE INDEX IF NOT EXISTS personal_event_recorded ON personal_events(session_id,recorded_at DESC,event_id)',
            'CREATE TABLE IF NOT EXISTS personal_event_versions (event_id TEXT NOT NULL, session_id TEXT NOT NULL, revision INTEGER NOT NULL, recorded_at REAL NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(event_id,revision))',
            'CREATE TABLE IF NOT EXISTS personal_event_relations (relation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_id TEXT NOT NULL, source_revision INTEGER NOT NULL, target_id TEXT NOT NULL, target_revision INTEGER NOT NULL, relation TEXT NOT NULL, recorded_at REAL NOT NULL, payload TEXT NOT NULL)',
            'CREATE INDEX IF NOT EXISTS personal_relation_source ON personal_event_relations(session_id,source_id,relation,target_id)',
            'CREATE INDEX IF NOT EXISTS personal_relation_target ON personal_event_relations(session_id,target_id,relation,source_id)',
        )
        for statement in statements:
            db.execute(statement)

    @staticmethod
    def _exists(db):
        return db is not None and db.execute("SELECT 1 FROM sqlite_master WHERE name='personal_events'").fetchone() is not None

    @staticmethod
    def _entity(db, sid, identifier):
        if identifier is None:
            return None
        _identifier(identifier)
        row = db.execute('SELECT entity_id,kind,name FROM personal_entities WHERE session_id=? AND entity_id=?', (sid, identifier)).fetchone()
        if row is None:
            raise ResearchNotFoundError('Entity not found in this project')
        return {'id': row['entity_id'], 'kind': row['kind'], 'name': row['name']}

    def register_entity(self, session_id, kind, name, *, aliases=(), identity_key=None):
        _session(session_id); _text(kind, 64); _text(name, 160)
        aliases = _strings(aliases)
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', kind):
            raise ResearchStoreError('Invalid entity kind')
        if identity_key is not None:
            _text(identity_key, 256)
        identity = hashlib.sha256(json.dumps([_normal(name), identity_key], ensure_ascii=False).encode()).hexdigest()
        with self.store._write() as db:
            self._schema(db)
            row = db.execute('SELECT entity_id FROM personal_entities WHERE session_id=? AND kind=? AND identity_sha256=?', (session_id, kind, identity)).fetchone()
            identifier = row[0] if row else str(uuid.uuid4())
            if row is None:
                self.store._require_storage(db)
                db.execute('INSERT INTO personal_entities VALUES(?,?,?,?,?)', (identifier, session_id, kind, name, identity))
            db.executemany('INSERT OR IGNORE INTO personal_entity_aliases VALUES(?,?,?,?)',
                [(session_id, kind, _normal(alias), identifier) for alias in [name, *aliases]])
            self.store._require_storage(db)
            return self._entity(db, session_id, identifier)

    def resolve_entity(self, session_id, kind, name):
        _session(session_id); _text(kind, 64); _text(name, 160)
        with self.store._read() as db:
            if not self._exists(db):
                rows = []
            else:
                rows = db.execute('SELECT e.entity_id,e.kind,e.name FROM personal_entity_aliases a JOIN personal_entities e ON e.entity_id=a.entity_id AND e.session_id=a.session_id WHERE a.session_id=? AND a.kind=? AND a.alias=? ORDER BY e.entity_id LIMIT 17', (session_id, kind, _normal(name))).fetchall()
        if len(rows) > 16:
            raise ResearchLimitError('Ambiguous entity candidate limit reached')
        candidates = [{'id': row['entity_id'], 'kind': row['kind'], 'name': row['name']} for row in rows]
        return {'status': 'resolved' if len(rows) == 1 else 'ambiguous' if rows else 'not_found', 'candidates': candidates, 'auto_merged': False}

    def _origins(self, db, sid, origins, *, validate=True, maximum=MAX_ORIGINS):
        if type(origins) not in (tuple, list) or not 1 <= len(origins) <= maximum:
            raise ResearchStoreError('An event requires bounded explicit provenance')
        normalized = []
        for origin in origins:
            if type(origin) is not dict:
                raise ResearchStoreError('Invalid event provenance')
            if origin.get('kind') == 'memory' and set(origin) == {'kind', 'memory_id', 'revision'}:
                _identifier(origin['memory_id'])
                if type(origin['revision']) is not int or origin['revision'] < 1:
                    raise ResearchStoreError('Invalid event memory revision')
                if validate and not MemoryDependencies(self.store)._live(db, sid, [(origin['memory_id'], origin['revision'], None)]):
                    raise ResearchStoreError('Event memory provenance unavailable')
                normalized.append(dict(origin))
            elif origin.get('kind') == 'source' and set(origin) == {'kind', 'source_id', 'span', 'sha256'}:
                _identifier(origin['source_id'])
                span = origin['span']
                if (type(span) is not list or len(span) != 2 or any(type(value) is not int for value in span)
                        or not 0 <= span[0] < span[1] <= 8000 or type(origin['sha256']) is not str
                        or not re.fullmatch('[0-9a-f]{64}', origin['sha256'])):
                    raise ResearchStoreError('Invalid event source coordinates')
                if validate:
                    row = db.execute("SELECT c.text FROM research_chunks c JOIN research_documents d ON d.document_id=c.document_id AND d.session_id=c.session_id WHERE c.session_id=? AND c.chunk_id=? AND d.state='ready'", (sid, origin['source_id'])).fetchone()
                    if (row is None or span[1] > len(row['text']) or hashlib.sha256(row['text'][span[0]:span[1]].encode()).hexdigest() != origin['sha256']):
                        raise ResearchStoreError('Event source provenance unavailable')
                normalized.append({'kind': 'source', 'source_id': origin['source_id'], 'span': span[:], 'sha256': origin['sha256']})
            else:
                raise ResearchStoreError('Invalid event provenance')
        return list({json.dumps(row, sort_keys=True): row for row in normalized}.values())

    def _live(self, db, sid, payload):
        try:
            self._origins(db, sid, payload['origins'], maximum=32)
            return True
        except (ResearchStoreError, KeyError, TypeError, ValueError):
            return False

    def _payload(self, db, sid, *, event_type, action, origins, actor_id=None, object_id=None, occurred_from=None, occurred_until=None, conditions=()):
        if type(event_type) is not str or event_type not in EVENT_TYPES:
            raise ResearchStoreError('Invalid domain event type')
        _text(action, 1000)
        start, end = _time(occurred_from), _time(occurred_until)
        if start is not None and end is not None and end < start:
            raise ResearchStoreError('Event occurrence end precedes its start')
        return {'event_type': event_type, 'action': action,
                'actor': self._entity(db, sid, actor_id), 'object': self._entity(db, sid, object_id),
                'occurred_from': start, 'occurred_until': end, 'conditions': _strings(conditions),
                'origins': self._origins(db, sid, origins)}

    @staticmethod
    def _required_event(db, sid, event_id):
        _identifier(event_id)
        row = db.execute('SELECT * FROM personal_events WHERE session_id=? AND event_id=?', (sid, event_id)).fetchone() if db else None
        if row is None:
            raise ResearchNotFoundError('Event not found in this project')
        return row

    def _view(self, db, row):
        payload = json.loads(row['payload'])
        result = {'id': row['event_id'], 'session_id': row['session_id'], 'revision': row['revision'],
                  'recorded_at': row['recorded_at'], 'semantic_verified': False, 'causality_verified': False, 'execution_authorized': False}
        if not self._live(db, row['session_id'], payload):
            return {**result, 'status': 'source_unavailable', 'redacted': True}
        return {**result, **payload, 'status': 'active', 'redacted': False}

    def record_event(self, session_id, **fields):
        _session(session_id)
        with self.store._write() as db:
            self._schema(db)
            payload = self._payload(db, session_id, **fields)
            identifier, recorded = str(uuid.uuid4()), self.store._timestamp(self.store.clock())
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            self.store._require_storage(db)
            db.execute('INSERT INTO personal_events VALUES(?,?,1,?,?,?,?,?,?)',
                (identifier, session_id, payload['actor']['id'] if payload['actor'] else None, payload['object']['id'] if payload['object'] else None,
                 payload['occurred_from'], payload['occurred_until'], recorded, raw))
            db.execute('INSERT INTO personal_event_versions VALUES(?,?,1,?,?)', (identifier, session_id, recorded, raw))
            self.store._require_storage(db)
            return self._view(db, self._required_event(db, session_id, identifier))

    def revise_event(self, session_id, event_id, *, expected_revision, **fields):
        _session(session_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ResearchStoreError('Invalid event revision')
        with self.store._write() as db:
            self._schema(db)
            previous = self._required_event(db, session_id, event_id)
            if previous['revision'] != expected_revision:
                raise ResearchRevisionConflictError('Event changed; refresh before correcting it')
            payload = self._payload(db, session_id, **fields)
            recorded, raw = self.store._timestamp(self.store.clock()), json.dumps(payload, ensure_ascii=False, sort_keys=True)
            db.execute('UPDATE personal_events SET revision=revision+1,actor_id=?,object_id=?,occurred_from=?,occurred_until=?,recorded_at=?,payload=? WHERE event_id=?',
                (payload['actor']['id'] if payload['actor'] else None, payload['object']['id'] if payload['object'] else None,
                 payload['occurred_from'], payload['occurred_until'], recorded, raw, event_id))
            db.execute('INSERT INTO personal_event_versions VALUES(?,?,?,?,?)', (event_id, session_id, expected_revision + 1, recorded, raw))
            self.store._require_storage(db)
            return self._view(db, self._required_event(db, session_id, event_id))

    def get_event(self, session_id, event_id):
        _session(session_id)
        with self.store._read() as db:
            return self._view(db, self._required_event(db if self._exists(db) else None, session_id, event_id))

    def snapshots(self, session_id, event_ids):
        """Explicit bounded current views for a caller-owned task context.

        No implicit event selection or task dispatch occurs. Callers must keep
        the returned revision and revalidate it before later delivery/action.
        """
        _session(session_id)
        if type(event_ids) is not list or len(event_ids) > 100:
            raise ResearchLimitError('Invalid event snapshot limit')
        identifiers = list(dict.fromkeys(_identifier(value) for value in event_ids))
        if not identifiers:
            return {}
        with self.store._read() as db:
            if not self._exists(db):
                raise ResearchNotFoundError('Event not found in this project')
            placeholders = ','.join('?' for _ in identifiers)
            rows = db.execute('SELECT * FROM personal_events WHERE session_id=? AND event_id IN (' + placeholders + ')', [session_id, *identifiers]).fetchall()
            if len(rows) != len(identifiers):
                raise ResearchNotFoundError('Event not found in this project')
            return {row['event_id']: self._view(db, row) for row in rows}

    def history(self, session_id, event_id, *, limit=20):
        """Revision audit only; no historical row is a current authorization."""
        _session(session_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ResearchLimitError('Invalid event history page size')
        with self.store._read() as db:
            current = self._required_event(db if self._exists(db) else None, session_id, event_id)
            rows = db.execute('SELECT * FROM personal_event_versions WHERE session_id=? AND event_id=? ORDER BY revision DESC LIMIT ?', (session_id, event_id, limit)).fetchall()
            return [{**self._view(db, row), 'historical': row['revision'] != current['revision']} for row in rows]

    def list_events(self, session_id, *, entity_id=None, limit=50):
        _session(session_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ResearchLimitError('Invalid event page size')
        with self.store._read() as db:
            if not self._exists(db):
                return []
            where, params = 'session_id=?', [session_id]
            if entity_id is not None:
                self._entity(db, session_id, entity_id)
                where += ' AND (actor_id=? OR object_id=?)'
                params.extend([entity_id, entity_id])
            rows = db.execute('SELECT * FROM personal_events WHERE ' + where + ' ORDER BY recorded_at DESC,event_id LIMIT ?', [*params, limit]).fetchall()
            return [self._view(db, row) for row in rows]

    def relate(self, session_id, source_id, target_id, relation, *, conditions=(), origins=()):
        _session(session_id)
        if type(relation) is not str or relation not in RELATIONS or source_id == target_id:
            raise ResearchStoreError('Invalid event relation')
        conditions = _strings(conditions)
        with self.store._write() as db:
            self._schema(db)
            source, target = self._required_event(db, session_id, source_id), self._required_event(db, session_id, target_id)
            left, right = self._view(db, source), self._view(db, target)
            if left['status'] != 'active' or right['status'] != 'active':
                raise ResearchStoreError('Event endpoint provenance unavailable')
            inherited = [*left['origins'], *right['origins']]
            provenance = self._origins(db, session_id, list(origins) if origins else inherited, maximum=32)
            if relation == 'before' and left['occurred_until'] is not None and right['occurred_from'] is not None and left['occurred_until'] > right['occurred_from']:
                raise ResearchStoreError('Before relation contradicts known event times')
            if relation in {'before', 'supersedes', 'depends_on'}:
                queue, visited, checked_edges = [target_id], set(), 0
                while queue:
                    current = queue.pop()
                    if current == source_id:
                        raise ResearchStoreError('Cyclic event dependency')
                    if current in visited:
                        continue
                    visited.add(current)
                    if len(visited) > 256:
                        raise ResearchLimitError('Event dependency traversal limit reached')
                    rows = db.execute('SELECT r.* FROM personal_event_relations r '
                        'JOIN personal_events s ON s.event_id=r.source_id AND s.session_id=r.session_id AND s.revision=r.source_revision '
                        'JOIN personal_events t ON t.event_id=r.target_id AND t.session_id=r.session_id AND t.revision=r.target_revision '
                        'WHERE r.session_id=? AND r.source_id=? AND r.relation=? LIMIT 257', (session_id, current, relation)).fetchall()
                    checked_edges += len(rows)
                    if checked_edges > 256:
                        raise ResearchLimitError('Event dependency traversal limit reached')
                    queue.extend(row['target_id'] for row in rows if self._relation_view(db, row)['status'] == 'active')
            identifier, recorded = str(uuid.uuid4()), self.store._timestamp(self.store.clock())
            payload = {'conditions': conditions, 'origins': provenance}
            db.execute('INSERT INTO personal_event_relations VALUES(?,?,?,?,?,?,?,?,?)',
                (identifier, session_id, source_id, source['revision'], target_id, target['revision'], relation, recorded, json.dumps(payload, ensure_ascii=False, sort_keys=True)))
            self.store._require_storage(db)
            return self._relation_view(db, db.execute('SELECT * FROM personal_event_relations WHERE relation_id=?', (identifier,)).fetchone())

    def _relation_view(self, db, row):
        result = {'id': row['relation_id'], 'session_id': row['session_id'], 'recorded_at': row['recorded_at'],
                  'semantic_verified': False, 'causality_verified': False, 'execution_authorized': False}
        left, right = self._required_event(db, row['session_id'], row['source_id']), self._required_event(db, row['session_id'], row['target_id'])
        payload = json.loads(row['payload'])
        if left['revision'] != row['source_revision'] or right['revision'] != row['target_revision']:
            return {**result, 'status': 'stale_endpoint', 'redacted': True}
        if not self._live(db, row['session_id'], payload) or self._view(db, left)['status'] != 'active' or self._view(db, right)['status'] != 'active':
            return {**result, 'status': 'source_unavailable', 'redacted': True}
        return {**result, **payload, 'source_id': row['source_id'], 'source_revision': row['source_revision'],
                'target_id': row['target_id'], 'target_revision': row['target_revision'], 'relation': row['relation'],
                'status': 'active', 'redacted': False}

    def list_relations(self, session_id, *, event_id=None, limit=100):
        _session(session_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ResearchLimitError('Invalid event relation page size')
        with self.store._read() as db:
            if not self._exists(db):
                return []
            where, params = 'session_id=?', [session_id]
            if event_id is not None:
                self._required_event(db, session_id, event_id)
                where += ' AND (source_id=? OR target_id=?)'
                params.extend([event_id, event_id])
            rows = db.execute('SELECT * FROM personal_event_relations WHERE ' + where + ' ORDER BY recorded_at DESC,relation_id LIMIT ?', [*params, limit]).fetchall()
            return [self._relation_view(db, row) for row in rows]
