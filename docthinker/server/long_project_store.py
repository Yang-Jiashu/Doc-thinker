"""Durable phase manifests over the existing task database and executor.

Phases contain immutable task/report references, not copied answers or sources.
Customer acceptance is an explicit version-bound review, never semantic proof.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import math
import uuid

from .project_task_store import (
    ProjectTaskStore, ProjectTaskError, ProjectTaskConflictError,
    ProjectTaskNotFoundError, _json, _uuid, _text, _revision, _questions,
)


MAX_PAGE = 100
MAX_PARENT_STEPS = 6


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _criteria(value):
    if type(value) is not list or not 1 <= len(value) <= 12:
        raise ProjectTaskError('One to twelve explicit acceptance criteria are required')
    rows = [_text(item, 1000, required=True) for item in value]
    if len(_json(rows).encode('utf-8')) > 8192 or len(set(rows)) != len(rows):
        raise ProjectTaskError('Acceptance criteria exceed their budget or repeat')
    return rows


class LongProjectStore:
    """Same owner, session, instance, file, transaction and byte quota as tasks."""

    def __init__(self, task_store):
        if not isinstance(task_store, ProjectTaskStore):
            raise ProjectTaskError('Long projects require the existing scoped task store')
        self.tasks = task_store
        self.path, self.session_id, self.instance_id = task_store.path, task_store.session_id, task_store.instance_id

    @staticmethod
    def _schema(db):
        db.execute('CREATE TABLE IF NOT EXISTS long_projects ('
            'id TEXT PRIMARY KEY,session_id TEXT NOT NULL,instance_id TEXT NOT NULL,'
            'goal TEXT NOT NULL,acceptance_criteria TEXT NOT NULL,revision INTEGER NOT NULL,'
            'status TEXT NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS long_projects_scope ON long_projects(session_id,instance_id,created_at,id)')
        db.execute('CREATE TABLE IF NOT EXISTS long_project_phases ('
            'id TEXT PRIMARY KEY,project_id TEXT NOT NULL,ordinal INTEGER NOT NULL,task_id TEXT UNIQUE NOT NULL,'
            'title TEXT NOT NULL,acceptance_criteria TEXT NOT NULL,parent_refs TEXT NOT NULL,'
            'revision INTEGER NOT NULL,created_at REAL NOT NULL,UNIQUE(project_id,ordinal))')
        db.execute('CREATE INDEX IF NOT EXISTS long_project_phase_directory ON long_project_phases(project_id,ordinal)')
        db.execute('CREATE TABLE IF NOT EXISTS long_project_phase_parents ('
            'child_phase_id TEXT NOT NULL,parent_phase_id TEXT NOT NULL,parent_task_id TEXT NOT NULL,'
            'report_version INTEGER NOT NULL,step_id TEXT NOT NULL,artifact_version INTEGER NOT NULL,'
            'PRIMARY KEY(child_phase_id,parent_phase_id,step_id))')
        db.execute('CREATE INDEX IF NOT EXISTS long_project_parent_reverse ON long_project_phase_parents(parent_task_id,child_phase_id)')
        db.execute('CREATE TABLE IF NOT EXISTS long_project_requests ('
            'request_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,instance_id TEXT NOT NULL,'
            'project_id TEXT NOT NULL,operation TEXT NOT NULL,fingerprint TEXT NOT NULL,'
            'result_ref TEXT NOT NULL,created_at REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS long_project_reviews ('
            'id TEXT PRIMARY KEY,project_id TEXT NOT NULL,phase_id TEXT,decision TEXT NOT NULL,'
            'binding TEXT NOT NULL,binding_sha256 TEXT NOT NULL,created_at REAL NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS long_project_review_latest ON long_project_reviews(project_id,phase_id)')

    @contextmanager
    def _db(self, *, write=False):
        with self.tasks._db(write=write) as db:
            if db is not None:
                if write:
                    self._schema(db)
                elif not db.execute("SELECT 1 FROM sqlite_master WHERE name='long_projects'").fetchone():
                    db = None
            yield db

    def _now(self):
        value = self.tasks.clock()
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 253402300799:
            raise ProjectTaskError('Invalid project clock')
        return value

    def _project(self, db, project_id):
        _uuid(project_id)
        row = db.execute('SELECT * FROM long_projects WHERE id=? AND session_id=? AND instance_id=?',
            (project_id, self.session_id, self.instance_id)).fetchone() if db else None
        if row is None:
            raise ProjectTaskNotFoundError('Long project is unavailable')
        return row

    def _phase(self, db, project_id, phase_id):
        self._project(db, project_id)
        _uuid(phase_id)
        row = db.execute('SELECT * FROM long_project_phases WHERE project_id=? AND id=?',
                         (project_id, phase_id)).fetchone()
        if row is None:
            raise ProjectTaskNotFoundError('Project phase is unavailable')
        return row

    @staticmethod
    def _cas(project, expected_revision):
        if project['revision'] != _revision(expected_revision):
            raise ProjectTaskConflictError('Long project changed; reload its manifest')

    def _prior(self, db, request_id, operation, fingerprint):
        _uuid(request_id)
        row = db.execute('SELECT * FROM long_project_requests WHERE request_id=?', (request_id,)).fetchone()
        if row is not None and (row['session_id'] != self.session_id or row['instance_id'] != self.instance_id
                               or row['operation'] != operation or row['fingerprint'] != fingerprint):
            raise ProjectTaskConflictError('Request ID already belongs to different project input')
        return row

    def _record(self, db, request_id, project_id, operation, fingerprint, result_ref):
        db.execute('INSERT INTO long_project_requests VALUES(?,?,?,?,?,?,?,?)',
            (request_id, self.session_id, self.instance_id, project_id, operation, fingerprint, result_ref, self._now()))

    def _touch(self, db, project_id):
        db.execute("UPDATE long_projects SET revision=revision+1,status='active',updated_at=? WHERE id=?",
                   (self._now(), project_id))

    def _report_binding(self, db, phase, version=None):
        task = self.tasks._task(db, phase['task_id'])
        report = next(row for row in self.tasks._steps(db, task['id']) if row['kind'] == 'report')
        if (task['status'] != 'completed' or self.tasks._unresolved_attempt(db, task['id'])
                or report['status'] != 'completed' or not report['artifact_version']
                or (version is not None and report['artifact_version'] != _revision(version))):
            raise ProjectTaskConflictError('A current completed parent report is required')
        artifact = db.execute("SELECT * FROM project_task_artifacts WHERE task_id=? AND step_id='report' "
            "AND version=? AND status='current'", (task['id'], report['artifact_version'])).fetchone()
        if artifact is None or artifact['generation'] != report['generation']:
            raise ProjectTaskConflictError('Report version is unavailable')
        payload = json.loads(artifact['payload'])
        steps = {row['id']: row for row in self.tasks._steps(db, task['id']) if row['kind'] == 'analysis'}
        if {ref['step_id'] for ref in payload['step_refs']} != set(steps):
            raise ProjectTaskConflictError('Report dependencies are incomplete')
        inputs = []
        for ref in payload['step_refs']:
            row = steps[ref['step_id']]
            saved = db.execute("SELECT * FROM project_task_artifacts WHERE task_id=? AND step_id=? "
                "AND version=? AND status='current'", (task['id'], row['id'], ref['version'])).fetchone()
            if (row['status'] != 'completed' or row['artifact_version'] != ref['version'] or saved is None
                    or saved['generation'] != row['generation']):
                raise ProjectTaskConflictError('Report analysis version is unavailable')
            inputs.append({'step_id': row['id'], 'artifact_version': ref['version'], 'generation': row['generation'],
                'input_snapshot': json.loads(row['input_snapshot']),
                'input_kind': row['input_kind'], 'context_snapshot': json.loads(row['context_snapshot']),
                'references_sha256': _digest(json.loads(saved['payload']))})
        return {'task_id': task['id'], 'task_generation': task['generation'],
            'report_version': report['artifact_version'], 'report_generation': report['generation'],
            'report_sha256': _digest(payload), 'inputs': inputs,
            'criteria_sha256': _digest(json.loads(phase['acceptance_criteria']))}

    @staticmethod
    def _review_public(row, *, current):
        if row is None:
            return None
        return {'id': row['id'], 'decision': row['decision'], 'status': row['decision'] if current else 'stale',
            'binding_sha256': row['binding_sha256'], 'binding': json.loads(row['binding']),
            'created_at': row['created_at'], 'reviewer': 'user', 'semantic_verified': False,
            'goal_verified': False}

    def _phase_public(self, db, project, phase):
        task = self.tasks._task(db, phase['task_id'])
        steps = self.tasks._steps(db, task['id'])
        try:
            binding = self._report_binding(db, phase)
        except ProjectTaskConflictError:
            binding = None
        review = db.execute('SELECT * FROM long_project_reviews WHERE project_id=? AND phase_id=? '
            'ORDER BY rowid DESC LIMIT 1', (project['id'], phase['id'])).fetchone()
        reviewed = self._review_public(review, current=bool(binding and review and _digest(binding) == review['binding_sha256']))
        return {'id': phase['id'], 'project_id': project['id'], 'session_id': self.session_id,
            'ordinal': phase['ordinal'], 'task_id': phase['task_id'], 'title': phase['title'],
            'acceptance_criteria': json.loads(phase['acceptance_criteria']), 'revision': phase['revision'],
            'project_revision': project['revision'], 'parent_refs': json.loads(phase['parent_refs']),
            'task_status': task['status'], 'report_version': binding['report_version'] if binding else None,
            'analysis_steps': [{'id': row['id'], 'title': row['title'], 'status': row['status']}
                               for row in steps if row['kind'] == 'analysis'],
            'review_status': reviewed['status'] if reviewed else 'not_reviewed', 'review': reviewed,
            'handoff_status': 'references_only_not_delivered', 'goal_verified': False,
            'created_at': phase['created_at']}

    def _project_binding(self, db, project):
        phases = db.execute('SELECT * FROM long_project_phases WHERE project_id=? ORDER BY ordinal', (project['id'],)).fetchall()
        if not phases:
            raise ProjectTaskConflictError('A project needs current phase reports before review')
        return {'definition_sha256': _digest([project['goal'], json.loads(project['acceptance_criteria'])]),
            'phases': [{'phase_id': row['id'], 'ordinal': row['ordinal'],
                        'report': self._report_binding(db, row)} for row in phases]}

    def _public(self, db, project):
        result = dict(project)
        result.pop('instance_id')
        result['acceptance_criteria'] = json.loads(result['acceptance_criteria'])
        result['phase_count'] = db.execute('SELECT count(*) FROM long_project_phases WHERE project_id=?', (project['id'],)).fetchone()[0]
        review = db.execute('SELECT * FROM long_project_reviews WHERE project_id=? AND phase_id IS NULL '
            'ORDER BY rowid DESC LIMIT 1', (project['id'],)).fetchone()
        current = False
        if review is not None:
            try:
                current = _digest(self._project_binding(db, project)) == review['binding_sha256']
            except ProjectTaskConflictError:
                pass
        result['review'] = self._review_public(review, current=current)
        result['review_status'] = result['review']['status'] if review else 'not_reviewed'
        result['status'] = review['decision'] if current else 'active'
        result.update(semantic_verified=False, goal_verified=False)
        return result

    def create_project(self, *, goal, acceptance_criteria, request_id):
        goal, criteria = _text(goal, 1200, required=True), _criteria(acceptance_criteria)
        fingerprint = _digest([goal, criteria])
        with self._db(write=True) as db:
            previous = self._prior(db, request_id, 'create', fingerprint)
            if previous:
                return {**self._public(db, self._project(db, previous['project_id'])), 'request_replayed': True}
            key, now = str(uuid.uuid4()), self._now()
            db.execute('INSERT INTO long_projects VALUES(?,?,?,?,?,?,?,?,?)',
                (key, self.session_id, self.instance_id, goal, _json(criteria), 1, 'active', now, now))
            self._record(db, request_id, key, 'create', fingerprint, key)
            return self._public(db, self._project(db, key))

    def get_project(self, project_id):
        with self._db() as db:
            return self._public(db, self._project(db, project_id))

    def revise_project(self, project_id, *, expected_revision, goal, acceptance_criteria, request_id):
        goal, criteria = _text(goal, 1200, required=True), _criteria(acceptance_criteria)
        fingerprint = _digest([project_id, expected_revision, goal, criteria])
        with self._db(write=True) as db:
            project = self._project(db, project_id)
            if self._prior(db, request_id, 'revise', fingerprint):
                return {**self._public(db, project), 'request_replayed': True}
            self._cas(project, expected_revision)
            db.execute('UPDATE long_projects SET goal=?,acceptance_criteria=? WHERE id=?', (goal, _json(criteria), project_id))
            self._touch(db, project_id)
            self._record(db, request_id, project_id, 'revise', fingerprint, project_id)
            return self._public(db, self._project(db, project_id))

    def _parents(self, db, project_id, refs):
        if type(refs) is not list or len(refs) > MAX_PARENT_STEPS:
            raise ProjectTaskError('Invalid parent report selection')
        selected, count, seen = [], 0, set()
        for ref in refs:
            if type(ref) is not dict or set(ref) - {'phase_id', 'report_version', 'step_ids'}:
                raise ProjectTaskError('Invalid parent report reference')
            phase = self._phase(db, project_id, ref.get('phase_id'))
            if phase['id'] in seen:
                raise ProjectTaskError('Duplicate parent phase')
            seen.add(phase['id'])
            binding = self._report_binding(db, phase, ref.get('report_version'))
            if 'report_version' not in ref:
                raise ProjectTaskError('Parent report version must be explicit')
            steps = {row['step_id']: row['artifact_version'] for row in binding['inputs']}
            ids = ref.get('step_ids', list(steps))
            if (type(ids) is not list or not ids or any(type(key) is not str for key in ids)
                    or len(ids) != len(set(ids)) or any(key not in steps for key in ids)):
                raise ProjectTaskError('Invalid explicit parent analysis selection')
            count += len(ids)
            if count > MAX_PARENT_STEPS:
                raise ProjectTaskError('Parent analysis selection exceeds six steps')
            selected.append({'phase_id': phase['id'], 'task_id': phase['task_id'],
                'report_version': binding['report_version'], 'step_ids': list(ids),
                'step_versions': {key: steps[key] for key in ids}, 'binding_sha256': _digest(binding),
                'selected_binding_sha256': _digest([row for row in binding['inputs'] if row['step_id'] in ids])})
        return selected

    def add_phase(self, project_id, *, expected_revision, request_id, title, acceptance_criteria,
                  questions, parent_refs=None, constraints=''):
        title, criteria = _text(title, 120, required=True), _criteria(acceptance_criteria)
        constraints, questions = _text(constraints, 1200), _questions(questions)
        refs = [] if parent_refs is None else parent_refs
        fingerprint = _digest([project_id, expected_revision, title, criteria, questions, refs, constraints])
        with self._db(write=True) as db:
            project = self._project(db, project_id)
            previous = self._prior(db, request_id, 'add_phase', fingerprint)
            if previous:
                return {**self._phase_public(db, project, self._phase(db, project_id, previous['result_ref'])), 'request_replayed': True}
            self._cas(project, expected_revision)
            # A new phase must not launder an unresolved accepted attempt into
            # a fresh request identity. Recovery never changes this condition.
            if db.execute("SELECT 1 FROM long_project_phases p JOIN project_task_attempts a ON a.task_id=p.task_id "
                "WHERE p.project_id=? AND a.status IN ('running','provider_outcome_unknown') LIMIT 1", (project_id,)).fetchone():
                raise ProjectTaskConflictError('An existing phase has an unresolved model outcome')
            parents = self._parents(db, project_id, refs)
            phase_id = str(uuid.uuid4())
            ordinal = db.execute('SELECT COALESCE(MAX(ordinal),0)+1 FROM long_project_phases WHERE project_id=?', (project_id,)).fetchone()[0]
            # Parent scope and CAS are settled in this very transaction. The
            # existing outer _db byte quota applies to manifest AND task rows.
            task = self.tasks._insert_task(db, goal=project['goal'], constraints=constraints,
                questions=questions, enforce_task_limits=False)
            db.execute('INSERT INTO long_project_phases VALUES(?,?,?,?,?,?,?,?,?)',
                (phase_id, project_id, ordinal, task['id'], title, _json(criteria), _json(parents), 1, self._now()))
            for parent in parents:
                db.executemany('INSERT INTO long_project_phase_parents VALUES(?,?,?,?,?,?)',
                    [(phase_id, parent['phase_id'], parent['task_id'], parent['report_version'], key, version)
                     for key, version in parent['step_versions'].items()])
            self._touch(db, project_id)
            self._record(db, request_id, project_id, 'add_phase', fingerprint, phase_id)
            return self._phase_public(db, self._project(db, project_id), self._phase(db, project_id, phase_id))

    def get_phase(self, project_id, phase_id):
        with self._db() as db:
            return self._phase_public(db, self._project(db, project_id), self._phase(db, project_id, phase_id))

    def phase_for_task(self, task_id):
        with self.tasks._db() as db:
            self.tasks._task(db, task_id)
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='long_project_phases'").fetchone():
                return None
            row = db.execute('SELECT p.* FROM long_project_phases p JOIN long_projects r ON r.id=p.project_id '
                'WHERE p.task_id=? AND r.session_id=? AND r.instance_id=?', (task_id, self.session_id, self.instance_id)).fetchone() if db else None
            return self._phase_public(db, self._project(db, row['project_id']), row) if row else None

    def parent_dependencies(self, task_id):
        with self.tasks._db() as db:
            self.tasks._task(db, task_id)
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='long_project_phases'").fetchone():
                return []
            phase = db.execute('SELECT p.* FROM long_project_phases p JOIN long_projects r ON r.id=p.project_id '
                'WHERE p.task_id=? AND r.session_id=? AND r.instance_id=?', (task_id, self.session_id, self.instance_id)).fetchone() if db else None
            if phase is None:
                return []
            refs = json.loads(phase['parent_refs'])
            for ref in refs:
                parent = self._phase(db, phase['project_id'], ref['phase_id'])
                self._selected_parent_live(db, parent, ref)
            return refs

    def _selected_parent_live(self, db, phase, ref):
        """Pinned report authorizes selection; only selected steps stay live.

        An unrelated parent branch may be revised without erasing an unchanged
        analysis checkpoint. Historical report prose is never exported here.
        """
        task = self.tasks._task(db, phase['task_id'])
        if self.tasks._unresolved_attempt(db, task['id']):
            raise ProjectTaskConflictError('Parent outcome is unresolved')
        report = db.execute("SELECT payload FROM project_task_artifacts WHERE task_id=? AND step_id='report' AND version=?",
                            (task['id'], ref['report_version'])).fetchone()
        if report is None:
            raise ProjectTaskConflictError('Pinned parent manifest is unavailable')
        authorized = {row['step_id']: row['version'] for row in json.loads(report['payload'])['step_refs']}
        if any(authorized.get(key) != version for key, version in ref['step_versions'].items()):
            raise ProjectTaskConflictError('Parent selection differs from its pinned manifest')
        rows = {row['id']: row for row in self.tasks._steps(db, task['id'])}
        inputs = []
        for key in authorized:
            if key not in ref['step_versions']:
                continue
            row = rows.get(key)
            saved = db.execute("SELECT * FROM project_task_artifacts WHERE task_id=? AND step_id=? AND version=? AND status='current'",
                (task['id'], key, ref['step_versions'][key])).fetchone()
            if (row is None or row['status'] != 'completed' or row['artifact_version'] != ref['step_versions'][key]
                    or saved is None or saved['generation'] != row['generation']):
                raise ProjectTaskConflictError('Selected parent analysis changed')
            inputs.append({'step_id': key, 'artifact_version': row['artifact_version'], 'generation': row['generation'],
                'input_snapshot': json.loads(row['input_snapshot']), 'input_kind': row['input_kind'],
                'context_snapshot': json.loads(row['context_snapshot']),
                'references_sha256': _digest(json.loads(saved['payload']))})
        if _digest(inputs) != ref['selected_binding_sha256']:
            raise ProjectTaskConflictError('Selected parent inputs changed')

    def _cursor(self, kind, project_id, value):
        packet = [kind, self.session_id, self.instance_id, project_id, value]
        return base64.urlsafe_b64encode(_json(packet).encode()).decode().rstrip('=')

    def _after(self, cursor, kind, project_id, default):
        if cursor is None:
            return default
        try:
            if type(cursor) is not str or len(cursor) > 2048:
                raise ValueError
            packet = json.loads(base64.b64decode(cursor + '=' * (-len(cursor) % 4), altchars=b'-_', validate=True))
            if type(packet) is not list or len(packet) != 5 or packet[:4] != [kind, self.session_id, self.instance_id, project_id]:
                raise ValueError
            return packet[4]
        except (ValueError, TypeError, UnicodeError):
            raise ProjectTaskError('Invalid scoped project cursor') from None

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
            raise ProjectTaskError('Invalid project page size')
        return limit

    def list_phases(self, project_id, *, limit=20, cursor=None):
        limit = self._limit(limit)
        after = self._after(cursor, 'phases', project_id, 0)
        if type(after) is not int or after < 0:
            raise ProjectTaskError('Invalid phase cursor')
        with self._db() as db:
            project = self._project(db, project_id)
            rows = db.execute('SELECT * FROM long_project_phases WHERE project_id=? AND ordinal>? ORDER BY ordinal LIMIT ?',
                              (project_id, after, limit + 1)).fetchall()
            return {'items': [self._phase_public(db, project, row) for row in rows[:limit]],
                'next_cursor': self._cursor('phases', project_id, rows[limit-1]['ordinal']) if len(rows) > limit else None}

    def list_projects(self, *, limit=20, cursor=None):
        limit = self._limit(limit)
        after = self._after(cursor, 'projects', None, [-1, ''])
        if (type(after) is not list or len(after) != 2 or type(after[0]) not in (int, float)
                or not math.isfinite(after[0]) or type(after[1]) is not str):
            raise ProjectTaskError('Invalid project cursor')
        with self._db() as db:
            rows = db.execute('SELECT * FROM long_projects WHERE session_id=? AND instance_id=? '
                'AND (created_at>? OR (created_at=? AND id>?)) ORDER BY created_at,id LIMIT ?',
                (self.session_id, self.instance_id, after[0], after[0], after[1], limit + 1)).fetchall() if db else []
            return {'items': [self._public(db, row) for row in rows[:limit]],
                'next_cursor': self._cursor('projects', None, [rows[limit-1]['created_at'], rows[limit-1]['id']]) if len(rows) > limit else None}

    def dependent_tasks(self, task_id, *, limit=100, cursor=None):
        limit = self._limit(limit)
        after = self._after(cursor, 'dependents', task_id, 0)
        if type(after) is not int or after < 0:
            raise ProjectTaskError('Invalid dependency cursor')
        with self.tasks._db() as db:
            self.tasks._task(db, task_id)
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='long_project_phase_parents'").fetchone():
                return {'items': [], 'next_cursor': None}
            rows = db.execute('SELECT DISTINCT p.* FROM long_project_phase_parents e '
                'JOIN long_project_phases p ON p.id=e.child_phase_id JOIN long_projects r ON r.id=p.project_id '
                'WHERE e.parent_task_id=? AND r.session_id=? AND r.instance_id=? AND p.ordinal>? ORDER BY p.ordinal LIMIT ?',
                (task_id, self.session_id, self.instance_id, after, limit + 1)).fetchall() if db else []
            return {'items': [{'phase_id': row['id'], 'task_id': row['task_id'], 'project_id': row['project_id'], 'ordinal': row['ordinal']} for row in rows[:limit]],
                'next_cursor': self._cursor('dependents', task_id, rows[limit-1]['ordinal']) if len(rows) > limit else None}

    def review_phase(self, project_id, phase_id, *, expected_revision, request_id, report_version, decision):
        if type(decision) is not str or decision not in {'accepted', 'rejected'}:
            raise ProjectTaskError('Review must explicitly accept or reject')
        fingerprint = _digest([project_id, phase_id, expected_revision, report_version, decision])
        with self._db(write=True) as db:
            project, phase = self._project(db, project_id), self._phase(db, project_id, phase_id)
            if self._prior(db, request_id, 'review_phase', fingerprint):
                return {**self._phase_public(db, project, phase), 'request_replayed': True}
            self._cas(project, expected_revision)
            binding = self._report_binding(db, phase, report_version)
            db.execute('INSERT INTO long_project_reviews VALUES(?,?,?,?,?,?,?)',
                (str(uuid.uuid4()), project_id, phase_id, decision, _json(binding), _digest(binding), self._now()))
            db.execute('UPDATE long_project_phases SET revision=revision+1 WHERE id=?', (phase_id,))
            self._touch(db, project_id)
            self._record(db, request_id, project_id, 'review_phase', fingerprint, phase_id)
            return self._phase_public(db, self._project(db, project_id), self._phase(db, project_id, phase_id))

    def review_project(self, project_id, *, expected_revision, request_id, phase_reports, decision):
        if type(decision) is not str or decision not in {'accepted', 'rejected'} or type(phase_reports) is not list:
            raise ProjectTaskError('Review requires explicit phase reports and decision')
        fingerprint = _digest([project_id, expected_revision, phase_reports, decision])
        with self._db(write=True) as db:
            project = self._project(db, project_id)
            if self._prior(db, request_id, 'review_project', fingerprint):
                return {**self._public(db, project), 'request_replayed': True}
            self._cas(project, expected_revision)
            binding = self._project_binding(db, project)
            expected = [{'phase_id': row['phase_id'], 'report_version': row['report']['report_version']} for row in binding['phases']]
            if phase_reports != expected:
                raise ProjectTaskConflictError('Review must bind every current phase report version')
            db.execute('INSERT INTO long_project_reviews VALUES(?,?,?,?,?,?,?)',
                (str(uuid.uuid4()), project_id, None, decision, _json(binding), _digest(binding), self._now()))
            self._touch(db, project_id)
            db.execute('UPDATE long_projects SET status=? WHERE id=?', (decision, project_id))
            self._record(db, request_id, project_id, 'review_project', fingerprint, project_id)
            return self._public(db, self._project(db, project_id))
