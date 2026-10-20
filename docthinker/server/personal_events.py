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
import time
import unicodedata
import uuid
from collections import deque

from .research_memory_dependencies import MemoryDependencies, _bindings_live
from .research_retrieval import lexical_tokens
from .research_store import ResearchLimitError, ResearchNotFoundError, ResearchRevisionConflictError, ResearchStoreError, _identifier, _session

EVENT_TYPES = frozenset({'decision', 'observation', 'task', 'meeting', 'milestone', 'change', 'incident', 'preference'})
RELATIONS = frozenset({'before', 'supersedes', 'depends_on', 'contradicts', 'related_to', 'reported_cause', 'hypothesis'})
MAX_ORIGINS = 16
MAX_CONTEXT_EVENTS = 8
MAX_CONTEXT_RELATIONS = 12
MAX_CONTEXT_BYTES = 6000
TARGET_CONTEXT_BYTES = 5500
MAX_CONTEXT_BINDING_BYTES = 8192
MAX_BRIDGE_STATES = 128
MAX_BRIDGE_EDGES = 192


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _event_binding(row):
    return {'id': row['event_id'], 'revision': row['revision'],
            'sha256': _digest([row['event_id'], row['revision'], row['recorded_at'], row['payload']])}


def _relation_binding(row):
    # Relations are immutable in the existing API. Revision 1 plus the complete
    # typed-row digest does not silently retag a changed edge as the same claim.
    return {'id': row['relation_id'], 'revision': 1, 'source_id': row['source_id'],
        'source_revision': row['source_revision'], 'target_id': row['target_id'],
        'target_revision': row['target_revision'], 'sha256': _digest([row['relation_id'],
            row['source_id'], row['source_revision'], row['target_id'], row['target_revision'],
            row['relation'], row['recorded_at'], row['payload']])}


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
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='personal_event_search'").fetchone():
            db.execute('CREATE VIRTUAL TABLE personal_event_search USING fts5(event_id UNINDEXED,scope,tokens)')
            # One upgrade backfill, never a last-100 recall window or a scan on
            # every question. Revisions update this current-event projection.
            for row in db.execute('SELECT event_id,session_id,payload FROM personal_events').fetchall():
                PersonalEvents._index_event(db, row['event_id'], row['session_id'], json.loads(row['payload']))

    @staticmethod
    def _index_event(db, event_id, sid, payload):
        fields = [payload['event_type'], payload['action'], *payload['conditions']]
        fields.extend(payload[key]['name'] for key in ('actor', 'object') if payload.get(key))
        db.execute('DELETE FROM personal_event_search WHERE event_id=?', (event_id,))
        db.execute('INSERT INTO personal_event_search VALUES(?,?,?)',
                   (event_id, 's' + sid[1:], ' '.join(lexical_tokens(' '.join(fields)))))

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

    def _origins(self, db, sid, origins, *, validate=True, maximum=MAX_ORIGINS,
                 at=None, seen=None, current_only=False):
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
                if validate:
                    if current_only:
                        current = db.execute('SELECT revision,status FROM research_memories WHERE session_id=? AND memory_id=?',
                                             (sid, origin['memory_id'])).fetchone()
                        if current is None or current['status'] != 'active' or current['revision'] != origin['revision']:
                            raise ResearchStoreError('Event memory provenance unavailable')
                    if not _bindings_live(db, sid, [(origin['memory_id'], origin['revision'], None)],
                            now=self.store.clock() if at is None else at, seen=seen):
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
            self._index_event(db, identifier, session_id, payload)
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
            self._index_event(db, event_id, session_id, payload)
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

    @staticmethod
    def _scope(values, maximum):
        if values is None:
            return None
        if (type(values) is not list or len(values) > maximum or any(type(value) is not str for value in values)
                or len(set(values)) != len(values)):
            raise ResearchLimitError('Invalid event context scope')
        return frozenset(_identifier(value) for value in values)

    def _context_origins_live(self, db, sid, origins, *, at, seen, sources=None, memories=None):
        if any((origin.get('kind') == 'source' and sources is not None and origin.get('source_id') not in sources)
               or (origin.get('kind') == 'memory' and memories is not None and origin.get('memory_id') not in memories)
               for origin in origins):
            return False
        try:
            self._origins(db, sid, origins, maximum=32, at=at, seen=seen, current_only=True)
            return True
        except (ValueError, TypeError, KeyError):
            return False

    @staticmethod
    def _event_bindings_live(db, sid, bindings, *, at=None, seen=None, store=None):
        """Connection-local exact reference check; never resolve names anew."""
        from types import SimpleNamespace
        at = time.time() if at is None else at
        seen = seen if seen is not None else {'active': set(), 'memo': {}}
        owner = PersonalEvents(store or SimpleNamespace(clock=lambda: at))
        if not bindings['events'] and not bindings['relations']:
            return True
        if not owner._exists(db):
            return False
        versions = {}
        for binding in bindings['events']:
            row = db.execute('SELECT * FROM personal_events WHERE session_id=? AND event_id=?', (sid, binding['id'])).fetchone()
            if row is None or _event_binding(row) != binding:
                return False
            payload = json.loads(row['payload'])
            if not owner._context_origins_live(db, sid, payload['origins'], at=at, seen=seen):
                return False
            versions[binding['id']] = binding['revision']
        for binding in bindings['relations']:
            row = db.execute('SELECT * FROM personal_event_relations WHERE session_id=? AND relation_id=?', (sid, binding['id'])).fetchone()
            if (row is None or _relation_binding(row) != binding
                    or versions.get(row['source_id']) != row['source_revision']
                    or versions.get(row['target_id']) != row['target_revision']):
                return False
            if not owner._context_origins_live(db, sid, json.loads(row['payload'])['origins'], at=at, seen=seen):
                return False
        return True

    def retrieve_context(self, session_id, question, *, allowed_source_ids=None,
                         allowed_memory_ids=None, event_ids=None):
        """Relevant current seeds, then bounded directed adjacency.

        Scopes constrain every origin, not just one convenient citation. An
        explicit event_ids list is a hard subgraph allowlist, not permission to
        borrow neighboring events. No relation type is promoted into proof.
        """
        _session(session_id)
        # This is an optional index query, not a new public question contract.
        # The caller retains the complete question (including any retrieval-only
        # history suffix); only lexical search work is bounded locally.
        if type(question) is not str:
            raise ResearchStoreError('Invalid personal event text')
        _text(question, len(question))
        sources = self._scope(allowed_source_ids, 20000)
        memories = self._scope(allowed_memory_ids, 128)
        explicit = self._scope(event_ids, 100)
        bindings = {'events': [], 'relations': [],
            'source_scope_sha256': None if sources is None else _digest(sorted(sources)),
            'memory_scope_sha256': None if memories is None else _digest(sorted(memories))}
        trace = {'strategy': 'current_fts_seeds_bounded_adjacency', 'seed_candidates': 0,
            'ambiguous_aliases': 0, 'truncation': [], 'semantic_verified': False, 'causality_verified': False}
        packet = {'events': [], 'relations': [], 'bindings': bindings, 'trace': trace}
        # Authorization belongs before the relevance candidate limit. Otherwise
        # many higher-ranked records from another selected-document/memory
        # scope can crowd out the first permitted event before live checks run.
        # JSON parameters avoid one SQL variable per possible source ID.
        origin_scope_sql, origin_scope_params = '', []
        for kind, identifier, allowed in (('source', 'source_id', sources), ('memory', 'memory_id', memories)):
            if allowed is not None:
                origin_scope_sql += (
                    " AND NOT EXISTS (SELECT 1 FROM json_each(e.payload,'$.origins') scoped_origin "
                    "WHERE json_extract(scoped_origin.value,'$.kind')='" + kind + "' "
                    "AND NOT EXISTS (SELECT 1 FROM json_each(?) allowed_origin "
                    "WHERE allowed_origin.value=json_extract(scoped_origin.value,'$." + identifier + "')))"
                )
                origin_scope_params.append(json.dumps(sorted(allowed), separators=(',', ':')))
        def gap(code):
            if code not in trace['truncation']:
                trace['truncation'].append(code)
        def size(value):
            return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        def evidence_size():
            # Host-only revision/hash guards do not consume model context.
            # They remain complete under their own constant memory bound.
            return size({'events': packet['events'], 'relations': packet['relations']})
        with self.store._read() as db:
            if not self._exists(db) or explicit == frozenset():
                return packet
            indexed = db.execute("SELECT 1 FROM sqlite_master WHERE name='personal_event_search'").fetchone() is not None
        if not indexed:
            with self.store._write() as db:
                self._schema(db)
                self.store._require_storage(db)
        with self.store._read(timeout=.05) as db:
            if not self._exists(db):
                return packet
            db.execute('BEGIN')
            at, seen, live_rows = self.store.clock(), {'active': set(), 'memo': {}}, {}
            tokens = list(dict.fromkeys(lexical_tokens(question[:16384])))[:24]
            if len(question) > 16384:
                gap('search_text_budget')
            if explicit is not None:
                rows = db.execute('SELECT e.* FROM personal_events e WHERE e.session_id=? AND e.event_id IN ('
                    + ','.join('?' for _ in explicit) + ')' + origin_scope_sql
                    + ' ORDER BY e.recorded_at DESC,e.event_id LIMIT 33',
                    [session_id, *sorted(explicit), *origin_scope_params]).fetchall()
            elif tokens:
                expression = 'scope : "s' + session_id[1:] + '" AND tokens : (' + ' OR '.join(
                    '"' + token.replace('"', '""') + '"' for token in tokens) + ')'
                rows = db.execute('SELECT e.* FROM personal_event_search f JOIN personal_events e ON e.event_id=f.event_id '
                    'WHERE personal_event_search MATCH ? AND e.session_id=?' + origin_scope_sql
                    + ' ORDER BY bm25(personal_event_search),e.event_id LIMIT 33',
                    [expression, session_id, *origin_scope_params]).fetchall()
            else:
                rows = []
            trace['seed_candidates'] = min(len(rows), 32)
            if len(rows) > 32:
                gap('seed_candidate_budget')
            def read_event(identifier, row=None):
                if identifier in live_rows:
                    return live_rows[identifier]
                if explicit is not None and identifier not in explicit:
                    return None
                row = row or db.execute('SELECT * FROM personal_events WHERE session_id=? AND event_id=?', (session_id, identifier)).fetchone()
                if row is None:
                    return None
                payload = json.loads(row['payload'])
                if not self._context_origins_live(db, session_id, payload['origins'], at=at, seen=seen, sources=sources, memories=memories):
                    return None
                event = {'id': identifier, 'revision': row['revision'], 'event_type': payload['event_type'],
                    'action': payload['action'], 'actor_id': payload['actor']['id'] if payload.get('actor') else None,
                    'object_id': payload['object']['id'] if payload.get('object') else None,
                    'actor': payload.get('actor'), 'object': payload.get('object'),
                    'occurred_from': payload['occurred_from'], 'occurred_until': payload['occurred_until'],
                    'recorded_at': row['recorded_at'], 'conditions': payload['conditions'], 'origins': payload['origins']}
                live_rows[identifier] = (event, _event_binding(row))
                return live_rows[identifier]
            selected = set()
            def admit_event(value, *, bundle_limit=None):
                event, binding = value
                if event['id'] in selected:
                    return True
                if len(selected) >= MAX_CONTEXT_EVENTS:
                    gap('event_limit'); return False
                packet['events'].append(event); bindings['events'].append(binding)
                limit = bundle_limit if bundle_limit is not None else (MAX_CONTEXT_BYTES - 256 if not selected else TARGET_CONTEXT_BYTES - 256)
                if size(bindings) > MAX_CONTEXT_BINDING_BYTES:
                    packet['events'].pop(); bindings['events'].pop(); gap('binding_bytes'); return False
                if evidence_size() > limit:
                    packet['events'].pop(); bindings['events'].pop(); gap('context_bytes'); return False
                selected.add(event['id'])
                return True
            relation_seen, checked = set(), 0

            def edge_view(row):
                left, right = read_event(row['source_id']), read_event(row['target_id'])
                payload = json.loads(row['payload'])
                if (not left or not right or left[0]['revision'] != row['source_revision']
                        or right[0]['revision'] != row['target_revision'] or not self._context_origins_live(
                            db, session_id, payload['origins'], at=at, seen=seen, sources=sources, memories=memories)):
                    return None
                return {'id': row['relation_id'], 'revision': 1, 'source_id': row['source_id'],
                        'source_revision': row['source_revision'], 'target_id': row['target_id'],
                        'target_revision': row['target_revision'], 'relation': row['relation'],
                        'conditions': payload['conditions'], 'origins': payload['origins']}

            def admit_path(path):
                # A mechanism bridge is one evidence bundle. Do not keep its
                # endpoints while silently dropping the critical middle edge.
                before = (len(packet['events']), len(packet['relations']), len(bindings['events']),
                          len(bindings['relations']), set(selected))
                for row in path:
                    edge = edge_view(row)
                    if (edge is None or not admit_event(read_event(row['source_id']), bundle_limit=MAX_CONTEXT_BYTES - 256)
                            or not admit_event(read_event(row['target_id']), bundle_limit=MAX_CONTEXT_BYTES - 256)):
                        break
                    if row['relation_id'] not in {value['id'] for value in packet['relations']}:
                        if len(packet['relations']) >= MAX_CONTEXT_RELATIONS:
                            gap('relation_limit'); break
                        packet['relations'].append(edge)
                        bindings['relations'].append(_relation_binding(row))
                    if size(bindings) > MAX_CONTEXT_BINDING_BYTES:
                        gap('binding_bytes'); break
                    if evidence_size() > MAX_CONTEXT_BYTES - 256:
                        gap('context_bytes'); break
                else:
                    relation_seen.update(row['relation_id'] for row in path)
                    return True
                packet['events'][before[0]:] = []; packet['relations'][before[1]:] = []
                bindings['events'][before[2]:] = []; bindings['relations'][before[3]:] = []
                selected.clear(); selected.update(before[4])
                gap('bridge_capacity')
                return False

            # Find directed bridges BETWEEN query seeds before a local
            # neighborhood can fill all slots. Reusing one adjacency cache
            # bounds database work; no top-k expansion or invented edge occurs.
            # depends_on is traversed prerequisite -> effect, while its stored
            # direction and type remain unchanged in the returned bundle.
            seeds = [value for row in rows[:32] if (value := read_event(row['event_id'], row))]
            goal_rank = {value[0]['id']: index for index, value in enumerate(seeds)}
            # Root selection covers distinct query terms and prefers actual
            # explanatory endpoints over isolated repeated-keyword matches.
            # All authorized lexical seeds remain goals, even if not roots.
            root_candidates = []
            for index, (value, _) in enumerate(seeds):
                connected = db.execute('SELECT 1 FROM personal_event_relations WHERE session_id=? '
                    "AND (source_id=? OR target_id=?) AND relation IN ('reported_cause','hypothesis','depends_on') LIMIT 1",
                    (session_id, value['id'], value['id'])).fetchone() is not None
                root_candidates.append((connected, index, value['id'], set(lexical_tokens(value['action'])) & set(tokens)))
            terminals, covered_terms = [], set()
            while root_candidates and len(terminals) < MAX_CONTEXT_EVENTS:
                root_candidates.sort(key=lambda value: (-int(value[0]), -len(value[3] - covered_terms), value[1]))
                _, _, identifier, terms = root_candidates.pop(0)
                terminals.append(identifier); covered_terms.update(terms)
            bridge_queue = deque((identifier, identifier, direction, [], frozenset({identifier}))
                for identifier in terminals for direction in ('out', 'in'))
            bridge_seen, adjacency, bridges = set(), {}, {}
            bridge_states, bridge_edges = 0, 0
            while bridge_queue and bridge_states < MAX_BRIDGE_STATES:
                root, current, direction, path, visited = bridge_queue.popleft()
                # Different directed edge paths are different mechanism
                # candidates (including distinct source-bound parallel edges).
                # A speculative shortcut must not hide a recorded mechanism.
                state_key = root, current, direction, tuple(row['relation_id'] for row in path)
                if state_key in bridge_seen:
                    continue
                bridge_seen.add(state_key); bridge_states += 1
                if current != root and current in goal_rank:
                    directed_path = path if direction == 'out' else list(reversed(path))
                    key = tuple(row['relation_id'] for row in directed_path)
                    bridges[key] = (goal_rank[root] + goal_rank[current], directed_path)
                    continue
                if len(path) >= MAX_CONTEXT_EVENTS - 1:
                    continue
                cache_key = current, direction
                if cache_key not in adjacency:
                    remaining_edges = MAX_BRIDGE_EDGES - bridge_edges
                    if remaining_edges <= 0:
                        gap('bridge_search_budget'); break
                    causal_column, prerequisite_column = ('source_id', 'target_id') if direction == 'out' else ('target_id', 'source_id')
                    raw_links = db.execute('SELECT * FROM personal_event_relations WHERE session_id=? AND '
                        f"(({causal_column}=? AND relation IN ('reported_cause','hypothesis')) OR ({prerequisite_column}=? AND relation='depends_on')) "
                        'ORDER BY relation_id LIMIT ?',
                        (session_id, current, current, min(48, remaining_edges) + 1)).fetchall()
                    if len(raw_links) > min(48, remaining_edges):
                        gap('bridge_search_budget')
                    raw_links = raw_links[:min(48, remaining_edges)]
                    bridge_edges += len(raw_links)
                    adjacency[cache_key] = [row for row in raw_links if edge_view(row) is not None]
                for row in adjacency[cache_key]:
                    reverse = (row['relation'] == 'depends_on') != (direction == 'in')
                    other = row['source_id'] if reverse else row['target_id']
                    if other not in visited:
                        bridge_queue.append((root, other, direction, [*path, row], visited | {other}))
            if bridge_queue:
                gap('bridge_search_budget')
            trace['bridge_search'] = {'states': bridge_states, 'edges': bridge_edges,
                                      'candidate_paths': len(bridges), 'retained_paths': 0}
            # Prefer complete recorded chains over speculative shortcuts;
            # this orders candidate retrieval, never verifies a causal claim.
            for _, path in sorted(bridges.values(), key=lambda value: (value[0],
                                        sum(row['relation'] == 'hypothesis' for row in value[1]), len(value[1]),
                                        tuple(row['relation_id'] for row in value[1]))):
                if admit_path(path):
                    trace['bridge_search']['retained_paths'] += 1

            def expand_bundle(seed):
                nonlocal selected, checked
                queued = [(seed, 0)]
                while queued:
                    identifier, depth = queued.pop(0)
                    if depth >= 2:
                        continue
                    links = db.execute('SELECT * FROM personal_event_relations WHERE session_id=? '
                        'AND (source_id=? OR target_id=?) ORDER BY recorded_at DESC,relation_id LIMIT 49',
                        (session_id, identifier, identifier)).fetchall()
                    if len(links) > 48:
                        gap('adjacency_budget')
                    for row in links[:48]:
                        if row['relation_id'] in relation_seen:
                            continue
                        relation_seen.add(row['relation_id']); checked += 1
                        if checked > 48:
                            gap('adjacency_budget'); queued.clear(); return
                        if len(packet['relations']) >= MAX_CONTEXT_RELATIONS:
                            gap('relation_limit'); queued.clear(); return
                        left, right = read_event(row['source_id']), read_event(row['target_id'])
                        payload = json.loads(row['payload'])
                        if (not left or not right or left[0]['revision'] != row['source_revision']
                                or right[0]['revision'] != row['target_revision'] or not self._context_origins_live(
                                    db, session_id, payload['origins'], at=at, seen=seen, sources=sources, memories=memories)):
                            continue
                        before = (len(packet['events']), len(bindings['events']), set(selected))
                        if not admit_event(left) or not admit_event(right):
                            packet['events'][before[0]:] = []; bindings['events'][before[1]:] = []; selected = before[2]
                            continue
                        edge = {'id': row['relation_id'], 'revision': 1, 'source_id': row['source_id'],
                            'source_revision': row['source_revision'], 'target_id': row['target_id'],
                            'target_revision': row['target_revision'], 'relation': row['relation'],
                            'conditions': payload['conditions'], 'origins': payload['origins']}
                        packet['relations'].append(edge); bindings['relations'].append(_relation_binding(row))
                        if evidence_size() > TARGET_CONTEXT_BYTES - 256 or size(bindings) > MAX_CONTEXT_BINDING_BYTES:
                            overflow = 'binding_bytes' if size(bindings) > MAX_CONTEXT_BINDING_BYTES else 'context_bytes'
                            packet['relations'].pop(); bindings['relations'].pop()
                            packet['events'][before[0]:] = []; bindings['events'][before[1]:] = []; selected = before[2]
                            gap(overflow); continue
                        for endpoint in (row['source_id'], row['target_id']):
                            if endpoint not in before[2]:
                                queued.append((endpoint, depth + 1))
            # Finish a relevant seed's bounded two-hop evidence bundle before
            # unrelated lexical hits can consume all event/byte slots.
            for row in rows[:32]:
                value = read_event(row['event_id'], row)
                if value and admit_event(value):
                    expand_bundle(row['event_id'])
            entity_ids = {value[key]['id'] for value in packet['events'] for key in ('actor', 'object') if value[key]}
            if entity_ids:
                for token in tokens[:16]:
                    trace['ambiguous_aliases'] += len(db.execute('SELECT kind FROM personal_entity_aliases '
                        'WHERE session_id=? AND alias=? AND entity_id IN (' + ','.join('?' for _ in entity_ids)
                        + ') GROUP BY kind HAVING count(DISTINCT entity_id)>1 LIMIT 8',
                        [session_id, _normal(token), *sorted(entity_ids)]).fetchall())
            trace['evidence_bytes'], trace['binding_bytes'] = evidence_size(), size(bindings)
            trace['evidence_byte_limit'], trace['binding_byte_limit'] = MAX_CONTEXT_BYTES, MAX_CONTEXT_BINDING_BYTES
            if evidence_size() > MAX_CONTEXT_BYTES or size(bindings) > MAX_CONTEXT_BINDING_BYTES:
                raise ResearchLimitError('Complete event context exceeds its evidence or host-binding byte budget')
            return packet
