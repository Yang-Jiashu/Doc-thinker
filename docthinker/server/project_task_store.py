"""Bounded durable project work, with reference-only immutable artifacts.

The host owns authorization, document access and provenance hydration. A stored
artifact is NOT a second copy of an answer or source. Unknown paid outcomes are
never retried by recovery, resume, or a repeated run request.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
import uuid

MAX_ANALYSIS_STEPS = 6
MAX_TASKS = 100
MAX_PROJECT_TASKS = 20
MAX_ARTIFACT_VERSIONS = 32
MAX_RUNS = 128
MAX_STORAGE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES = 16384
STATUSES = frozenset({'ready', 'running', 'completed', 'blocked', 'paused', 'cancelled', 'stale'})
REASONS = frozenset({'provider_outcome_unknown', 'provider_failed', 'source_changed', 'source_revoked',
                     'source_unavailable', 'provenance_unavailable', 'budget_exhausted', 'process_restarted',
                     'invalid_result', 'dependency_blocked', 'user_paused', 'user_cancelled'})
_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z')
_IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')


class ProjectTaskError(ValueError):
    pass


class ProjectTaskConflictError(ProjectTaskError):
    pass


class ProjectTaskNotFoundError(ProjectTaskError):
    pass


class ProjectTaskLimitError(ProjectTaskError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _uuid(value):
    if type(value) is not str or not _UUID.fullmatch(value):
        raise ProjectTaskError('Invalid task request reference')
    return value


def _identifier(value):
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ProjectTaskError('Invalid record reference')
    return value


def _text(value, limit, *, required=False):
    if (type(value) is not str or len(value) > limit or (required and not value.strip())
            or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', value)):
        raise ProjectTaskError('Invalid project task text')
    try:
        value.encode('utf-8', errors='strict')
    except UnicodeError:
        raise ProjectTaskError('Invalid project task text') from None
    return value


def _revision(value):
    if type(value) is not int or not 1 <= value < 2147483647:
        raise ProjectTaskError('Invalid task revision')
    return value


def _ids(values, *, limit=100, allow_empty=True):
    if type(values) is not list or len(values) > limit or (not allow_empty and not values):
        raise ProjectTaskError('Invalid document scope')
    result = [_identifier(value) for value in values]
    if len(set(result)) != len(result):
        raise ProjectTaskError('Duplicate record reference')
    return result


def _snapshot(value):
    if type(value) is not dict or len(value) > 100:
        raise ProjectTaskError('Invalid source snapshot')
    result = {}
    for key, digest in value.items():
        _identifier(key)
        if type(digest) is not str or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ProjectTaskError('Invalid source fingerprint')
        result[key] = digest
    return result


def _scoped(snapshot, scope):
    if scope is None:
        return dict(snapshot)
    if any(key not in snapshot for key in scope):
        raise ProjectTaskConflictError('A scoped document is unavailable')
    return {key: snapshot[key] for key in scope}


def _contexts(value):
    if value is None:
        return {}
    if type(value) is not dict or len(value) > MAX_ANALYSIS_STEPS * 8:
        raise ProjectTaskError('Invalid personal context snapshot')
    result = {}
    for key, revision in value.items():
        _uuid(key)
        result[key] = _revision(revision)
    return result


def _personal(row):
    return 'input_kind' in row.keys() and row['input_kind'] == 'personal'


def _selected_context(row, snapshots):
    ids = json.loads(row['memory_ids']) if _personal(row) else []
    if any(key not in snapshots for key in ids):
        raise ProjectTaskConflictError('Selected personal context is unavailable')
    return {key: snapshots[key] for key in ids}


def _handoff(value, required):
    keys = {'required', 'used', 'full', 'excerpted', 'omitted'}
    if (type(value) is not dict or set(value) != keys
            or any(type(count) is not int or not 0 <= count <= MAX_ANALYSIS_STEPS
                   for count in value.values())
            or value['required'] != required
            or value['required'] != value['used'] + value['omitted']
            or value['used'] != value['full'] + value['excerpted']):
        raise ProjectTaskError('Invalid dependency handoff counts')
    return dict(value)


def _phase_handoff(value):
    """Bounded receipt metadata, never authority to read another phase."""
    if type(value) is not dict or set(value) != {'required', 'used', 'full', 'excerpted', 'omitted', 'refs'}:
        raise ProjectTaskError('Invalid phase dependency handoff')
    refs = value['refs']
    if type(refs) is not list or len(refs) > MAX_ANALYSIS_STEPS:
        raise ProjectTaskError('Invalid phase dependency references')
    counts = _handoff({key: value[key] for key in value if key != 'refs'}, len(refs))
    if counts['used'] != len(refs) or counts['omitted']:
        raise ProjectTaskError('Every selected phase dependency must be delivered')
    result, seen = [], set()
    for ref in refs:
        if type(ref) is not dict or set(ref) != {'phase_id', 'task_id', 'report_version', 'step_id', 'version'}:
            raise ProjectTaskError('Invalid phase dependency reference')
        item = {'phase_id': _uuid(ref['phase_id']), 'task_id': _uuid(ref['task_id']),
                'report_version': ref['report_version'], 'step_id': ref['step_id'], 'version': ref['version']}
        if (type(item['step_id']) is not str or not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', item['step_id'])
                or item['step_id'] in {'inventory', 'report'}
                or any(type(item[key]) is not int or not 1 <= item[key] <= MAX_ARTIFACT_VERSIONS
                       for key in ('report_version', 'version'))):
            raise ProjectTaskError('Invalid phase dependency version')
        identity = (item['phase_id'], item['task_id'], item['step_id'])
        if identity in seen:
            raise ProjectTaskError('Duplicate phase dependency reference')
        seen.add(identity)
        result.append(item)
    return {**counts, 'refs': result}


def _questions(rows):
    if type(rows) is not list or not 1 <= len(rows) <= MAX_ANALYSIS_STEPS:
        raise ProjectTaskLimitError('A task needs one to six analysis questions')
    result, seen = [], set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) - {'id', 'title', 'question', 'document_ids', 'depends_on', 'input_kind', 'memory_ids'}:
            raise ProjectTaskError('Invalid analysis step')
        key = row.get('id', 'q' + str(index + 1))
        if type(key) is not str or not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', key) or key in {'inventory', 'report'} or key in seen:
            raise ProjectTaskError('Invalid analysis identifier')
        question = _text(row.get('question'), 2000, required=True)
        title = _text(row.get('title', question[:80]), 120, required=True)
        scope = row.get('document_ids')
        input_kind = row.get('input_kind', 'documents')
        if type(input_kind) is not str or input_kind not in {'documents', 'personal'}:
            raise ProjectTaskError('Invalid task input kind')
        memory_ids = row.get('memory_ids', [])
        if (type(memory_ids) is not list or len(memory_ids) > 8 or
                any(type(value) is not str for value in memory_ids) or len(set(memory_ids)) != len(memory_ids)):
            raise ProjectTaskError('Invalid personal memory scope')
        for identifier in memory_ids:
            _uuid(identifier)
        if (input_kind == 'personal' and scope is not None) or (input_kind == 'documents' and memory_ids):
            raise ProjectTaskError('Task input scopes cannot be mixed')
        if scope is not None:
            scope = _ids(scope, allow_empty=False)
        deps = row.get('depends_on', [])
        if type(deps) is not list or len(deps) > MAX_ANALYSIS_STEPS:
            raise ProjectTaskError('Invalid step dependencies')
        if any(type(dep) is not str for dep in deps) or len(set(deps)) != len(deps):
            raise ProjectTaskError('Invalid step dependencies')
        result.append({'id': key, 'title': title, 'question': question, 'document_ids': scope, 'depends_on': deps})
        if input_kind == 'personal':
            result[-1].update(input_kind='personal', memory_ids=list(memory_ids))
        seen.add(key)
    deps = {row['id']: row['depends_on'] for row in result}
    for key, values in deps.items():
        if any(dep not in seen or dep == key for dep in values):
            raise ProjectTaskError('Unknown or circular dependency')
    visited, visiting = set(), set()
    def visit(key):
        if key in visiting:
            raise ProjectTaskError('Circular dependency')
        if key in visited:
            return
        visiting.add(key)
        for dep in deps[key]:
            visit(dep)
        visiting.remove(key)
        visited.add(key)
    for key in deps:
        visit(key)
    return result


class ProjectTaskStore:
    def __init__(self, directory: Path, *, session_id, instance_id, boot_id=None, clock=time.time):
        self.directory = Path(directory)
        if (not self.directory.is_absolute() or self.directory.is_symlink()
                or self.directory.resolve(strict=True) != self.directory or not self.directory.is_dir()
                or self.directory.stat().st_uid != os.getuid()):
            raise ProjectTaskError('Task storage requires an existing owned worker directory')
        if type(session_id) is not str or not re.fullmatch(r'#[0-9]{5}', session_id):
            raise ProjectTaskError('Invalid project reference')
        self.session_id, self.instance_id = session_id, _identifier(instance_id)
        self.boot_id = _identifier(boot_id or str(uuid.uuid4()))
        self.path, self.clock = self.directory / 'project_tasks.db', clock

    def _file(self, create=False):
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            if not create:
                return False
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
            info = self.path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ProjectTaskError('Task storage must be an owner-only regular file')
        return True

    @staticmethod
    def _schema(db):
        db.execute('CREATE TABLE IF NOT EXISTS project_tasks (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, instance_id TEXT NOT NULL, goal TEXT NOT NULL, constraints_text TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, generation INTEGER NOT NULL, run_id TEXT, reason TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS project_tasks_scope ON project_tasks(session_id,instance_id,created_at)')
        db.execute('CREATE TABLE IF NOT EXISTS project_task_steps (task_id TEXT NOT NULL, id TEXT NOT NULL, ordinal INTEGER NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, question TEXT NOT NULL, document_ids TEXT, dependencies TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, generation INTEGER NOT NULL, request_id TEXT, input_snapshot TEXT NOT NULL, claim_token TEXT, claim_boot TEXT, claim_run TEXT, reason TEXT, artifact_version INTEGER NOT NULL, PRIMARY KEY(task_id,id))')
        columns = {row[1] for row in db.execute('PRAGMA table_info(project_task_steps)')}
        for name, declaration in (('input_kind', "TEXT NOT NULL DEFAULT 'documents'"),
                                  ('memory_ids', "TEXT NOT NULL DEFAULT '[]'"),
                                  ('context_snapshot', "TEXT NOT NULL DEFAULT '{}'"),
                                  ('retired', 'INTEGER NOT NULL DEFAULT 0')):
            if name not in columns:
                db.execute(f'ALTER TABLE project_task_steps ADD COLUMN {name} {declaration}')
        db.execute('CREATE TABLE IF NOT EXISTS project_task_runs (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, fingerprint TEXT NOT NULL, max_calls INTEGER NOT NULL, attempts INTEGER NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL)')
        # A durable approval belongs to one process incarnation, including the
        # gap after a checkpoint commits and before the next step is claimed.
        # Legacy approvals have unknown ownership and must be paused on repair.
        if 'owner_boot' not in {row[1] for row in db.execute('PRAGMA table_info(project_task_runs)')}:
            db.execute('ALTER TABLE project_task_runs ADD COLUMN owner_boot TEXT')
        db.execute('CREATE TABLE IF NOT EXISTS project_task_artifacts (task_id TEXT NOT NULL, step_id TEXT NOT NULL, version INTEGER NOT NULL, generation INTEGER NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(task_id,step_id,version))')
        db.execute('CREATE TABLE IF NOT EXISTS project_task_definitions (task_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(task_id,version))')
        db.execute('CREATE TABLE IF NOT EXISTS project_task_attempts (request_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, step_id TEXT NOT NULL, generation INTEGER NOT NULL, run_id TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS project_task_attempts_pending ON project_task_attempts(task_id,status)')

    @contextmanager
    def _db(self, *, write=False):
        if not self._file(create=write):
            yield None
            return
        db = sqlite3.connect(self.path if write else self.path.as_uri() + '?mode=ro', timeout=2, uri=not write)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute('PRAGMA journal_mode=DELETE')
                db.execute('BEGIN IMMEDIATE')
                self._schema(db)
            elif not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_tasks'").fetchone():
                # A rejected first write may leave an empty owned SQLite file,
                # but it must not create phantom task state or break listing.
                yield None
                return
            yield db
            if write:
                size = db.execute('PRAGMA page_count').fetchone()[0] * db.execute('PRAGMA page_size').fetchone()[0]
                if size > MAX_STORAGE_BYTES:
                    raise ProjectTaskLimitError('Project task storage capacity reached')
                db.commit()
        finally:
            db.close()

    def _task(self, db, task_id):
        _uuid(task_id)
        row = db.execute('SELECT * FROM project_tasks WHERE id=? AND session_id=? AND instance_id=?', (task_id, self.session_id, self.instance_id)).fetchone() if db else None
        if row is None:
            raise ProjectTaskNotFoundError('Project task is unavailable')
        return row

    @staticmethod
    def _steps(db, task_id):
        rows = db.execute('SELECT * FROM project_task_steps WHERE task_id=? ORDER BY ordinal', (task_id,)).fetchall()
        return [row for row in rows if 'retired' not in row.keys() or not row['retired']]

    @staticmethod
    def _unresolved_attempt(db, task_id):
        # Step reason/status may change on a source withdrawal or user edit.
        # The immutable request identity, not that UI state, owns uncertainty.
        return db.execute("SELECT 1 FROM project_task_attempts WHERE task_id=? AND status IN ('running','provider_outcome_unknown') LIMIT 1", (task_id,)).fetchone() is not None

    def _public(self, db, task):
        result = dict(task)
        result['constraints'] = result.pop('constraints_text')
        run = db.execute('SELECT * FROM project_task_runs WHERE id=?', (task['run_id'],)).fetchone() if task['run_id'] else None
        result.update(run_remaining=max(0, run['max_calls'] - run['attempts']) if run else 0,
                      run_max_calls=run['max_calls'] if run else 0)
        result['steps'] = []
        for row in self._steps(db, task['id']):
            item = dict(row)
            for key in ('claim_token', 'claim_boot', 'claim_run', 'task_id', 'ordinal', 'retired'):
                item.pop(key, None)
            item['document_ids'] = json.loads(item['document_ids']) if item['document_ids'] is not None else None
            item['depends_on'] = json.loads(item.pop('dependencies'))
            item['input_snapshot'] = json.loads(item['input_snapshot'])
            if item.get('input_kind') == 'personal':
                item['memory_ids'] = json.loads(item['memory_ids'])
                item['context_snapshot'] = json.loads(item['context_snapshot'])
            else:
                for key in ('input_kind', 'memory_ids', 'context_snapshot'):
                    item.pop(key, None)
            if item['kind'] == 'analysis' and item['status'] == 'completed' and item['artifact_version']:
                artifact = db.execute("SELECT payload FROM project_task_artifacts WHERE task_id=? AND step_id=? AND version=? AND status='current'",
                                      (task['id'], item['id'], item['artifact_version'])).fetchone()
                payload = json.loads(artifact['payload']) if artifact else {}
                if 'handoff' in payload:
                    # The host records only bounded use counts. Neither these
                    # counts nor successful parsing verify a model's reasoning.
                    item['handoff'] = _handoff(payload['handoff'], sum(dep != 'inventory' for dep in item['depends_on']))
                if 'phase_handoff' in payload:
                    phase = _phase_handoff(payload['phase_handoff'])
                    item['phase_handoff'] = {key: value for key, value in phase.items() if key != 'refs'}
            result['steps'].append(item)
        return result

    def _touch(self, db, task_id, *, status=None, reason=None):
        if status is None:
            task = self._task(db, task_id)
            if task['status'] in {'paused', 'cancelled'}:
                status, reason = task['status'], task['reason']
            else:
                states = {row['status'] for row in self._steps(db, task_id)}
                run = db.execute('SELECT status FROM project_task_runs WHERE id=?', (task['run_id'],)).fetchone() if task['run_id'] else None
                status = ('completed' if states == {'completed'} else 'running' if 'running' in states else
                          'stale' if 'stale' in states else 'blocked' if 'blocked' in states else
                          'running' if run is not None and run['status'] == 'running' else 'ready')
        if status == 'completed':
            db.execute("UPDATE project_task_runs SET status='completed' WHERE task_id=? AND status='running'", (task_id,))
        db.execute('UPDATE project_tasks SET status=?,reason=?,revision=revision+1,updated_at=? WHERE id=?', (status, reason, self.clock(), task_id))
        return self._public(db, self._task(db, task_id))

    @staticmethod
    def _cas(task, expected_revision):
        if task['revision'] != _revision(expected_revision):
            raise ProjectTaskConflictError('Task changed; reload before modifying it')

    def create_task(self, *, goal, constraints='', questions):
        # Preserve legacy rejection-before-file-creation semantics. The shared
        # helper validates too because transaction callers are a separate edge.
        _text(goal, 1200, required=True)
        _text(constraints, 1200)
        questions = _questions(questions)
        with self._db(write=True) as db:
            return self._insert_task(db, goal=goal, constraints=constraints, questions=questions)

    def _insert_task(self, db, *, goal, constraints='', questions, task_id=None,
                     created_at=None, enforce_task_limits=True):
        """Insert into THIS store's existing transaction; never open/commit.

        A long-project manifest may disable legacy task-count limits only after
        checking its own parent scope/phase quotas in this same transaction.
        The enclosing _db always enforces the database byte quota on commit.
        """
        if (not isinstance(db, sqlite3.Connection) or not db.in_transaction or
                Path(db.execute('PRAGMA database_list').fetchone()[2]).resolve() != self.path or
                type(enforce_task_limits) is not bool):
            raise ProjectTaskError('Task insertion requires its owned transaction')
        _text(goal, 1200, required=True)
        _text(constraints, 1200)
        questions = _questions(questions)
        now = self.clock() if created_at is None else created_at
        if type(now) not in (int, float) or not math.isfinite(now) or not 0 <= now <= 253402300799:
            raise ProjectTaskError('Invalid task creation time')
        key = str(uuid.uuid4()) if task_id is None else _uuid(task_id)
        if enforce_task_limits:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            standalone = ('NOT EXISTS (SELECT 1 FROM long_project_phases p JOIN long_projects l '
                          'ON l.id=p.project_id WHERE p.task_id=t.id AND l.session_id=t.session_id '
                          'AND l.instance_id=t.instance_id)') if {'long_projects', 'long_project_phases'} <= tables else '1'
            if (db.execute('SELECT count(*) FROM project_tasks t WHERE ' + standalone).fetchone()[0] >= MAX_TASKS or
                    db.execute('SELECT count(*) FROM project_tasks t WHERE session_id=? AND instance_id=? AND ' + standalone,
                               (self.session_id, self.instance_id)).fetchone()[0] >= MAX_PROJECT_TASKS):
                raise ProjectTaskLimitError('Project task capacity reached')
        db.execute('INSERT INTO project_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (key, self.session_id, self.instance_id, goal, constraints, 'ready', 1, 1, None, None, now, now))
        rows = [{'id': 'inventory', 'title': 'Material inventory', 'kind': 'inventory', 'question': '', 'document_ids': None, 'depends_on': []}]
        rows.extend({**row, 'kind': 'analysis', 'depends_on': ['inventory', *row['depends_on']]} for row in questions)
        rows.append({'id': 'report', 'title': 'Evidence report', 'kind': 'report', 'question': '', 'document_ids': None, 'depends_on': [row['id'] for row in questions]})
        if all(row.get('input_kind') == 'personal' for row in questions):
            union = list(dict.fromkeys(identifier for row in questions for identifier in row['memory_ids']))
            for wrapper in (rows[0], rows[-1]):
                wrapper.update(input_kind='personal', memory_ids=union)
        for ordinal, row in enumerate(rows):
            self._insert_step(db, key, row, ordinal)
        db.execute('INSERT INTO project_task_definitions VALUES (?,?,?,?)', (key, 1, _json({'goal': goal, 'constraints': constraints, 'questions': questions}), now))
        return self._public(db, self._task(db, key))

    def _insert_step(self, db, task_id, row, ordinal):
        db.execute('INSERT INTO project_task_steps '
            '(task_id,id,ordinal,kind,title,question,document_ids,dependencies,status,revision,generation,request_id,input_snapshot,claim_token,claim_boot,claim_run,reason,artifact_version,input_kind,memory_ids,context_snapshot,retired) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)',
            (task_id, row['id'], ordinal, row['kind'], row['title'], row['question'],
             _json(row['document_ids']) if row['document_ids'] is not None else None,
             _json(row['depends_on']), 'ready', 1, 1, str(uuid.uuid4()) if row['kind'] == 'analysis' else None,
             '{}', None, None, None, None, 0, row.get('input_kind', 'documents'), _json(row.get('memory_ids', [])), '{}'))

    def get_task(self, task_id):
        with self._db() as db:
            return self._public(db, self._task(db, task_id))

    def list_tasks(self):
        with self._db() as db:
            if db is None:
                return []
            rows = db.execute('SELECT * FROM project_tasks WHERE session_id=? AND instance_id=? ORDER BY created_at DESC,id LIMIT ?', (self.session_id, self.instance_id, MAX_PROJECT_TASKS)).fetchall()
            return [self._public(db, row) for row in rows]

    def begin_run(self, task_id, *, expected_revision, max_calls, source_snapshots, request_id=None, context_snapshots=None):
        if type(max_calls) is not int or not 1 <= max_calls <= MAX_ANALYSIS_STEPS:
            raise ProjectTaskError('Run allows one to six paid attempts')
        _revision(expected_revision)
        snapshots = _snapshot(source_snapshots)
        contexts = _contexts(context_snapshots)
        run_id = _uuid(request_id) if request_id is not None else str(uuid.uuid4())
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            rows = self._steps(db, task_id)
            personal = any(_personal(row) for row in rows)
            data = [task_id, expected_revision, max_calls, snapshots]
            if personal:
                data.append(contexts)
            fingerprint = hashlib.sha256(_json(data).encode()).hexdigest()
            previous = db.execute('SELECT * FROM project_task_runs WHERE id=?', (run_id,)).fetchone()
            if previous:
                if previous['task_id'] != task_id or previous['fingerprint'] != fingerprint:
                    raise ProjectTaskConflictError('Run ID belongs to different approved input')
                # Replaying an old approval is a status read, not permission
                # to dispatch work funded by a later, different run.
                result = self._public(db, task)
                result.update(run_id=run_id, run_replayed=True,
                              run_max_calls=previous['max_calls'],
                              run_remaining=max(0, previous['max_calls'] - previous['attempts']))
                return result
            self._cas(task, expected_revision)
            if task['status'] in {'cancelled', 'completed', 'stale', 'running'}:
                raise ProjectTaskConflictError('Task cannot start in its current state')
            if db.execute('SELECT count(*) FROM project_task_runs WHERE task_id=?', (task_id,)).fetchone()[0] >= MAX_RUNS:
                raise ProjectTaskLimitError('Task run history capacity reached')
            if self._unresolved_attempt(db, task_id) or any(row['status'] == 'running' or row['reason'] == 'provider_outcome_unknown' for row in rows):
                raise ProjectTaskConflictError('An earlier paid outcome requires reconciliation')
            for row in rows:
                scope = json.loads(row['document_ids']) if row['document_ids'] is not None else None
                current = {} if _personal(row) else _scoped(snapshots, scope)
                context = _selected_context(row, contexts)
                old = json.loads(row['input_snapshot'])
                if row['status'] == 'completed' and (old != current or (_personal(row) and json.loads(row['context_snapshot']) != context)):
                    raise ProjectTaskConflictError('Source changes require explicit refresh')
                if row['status'] == 'ready':
                    db.execute('UPDATE project_task_steps SET input_snapshot=?,context_snapshot=? WHERE task_id=? AND id=?', (_json(current), _json(context), task_id, row['id']))
            db.execute('INSERT INTO project_task_runs (id,task_id,fingerprint,max_calls,attempts,status,created_at,owner_boot) VALUES (?,?,?,?,?,?,?,?)', (run_id, task_id, fingerprint, max_calls, 0, 'running', self.clock(), self.boot_id))
            db.execute('UPDATE project_tasks SET run_id=? WHERE id=?', (run_id, task_id))
            return self._touch(db, task_id, status='ready')

    def claim_next(self, task_id, run_id, *, context_snapshots=None):
        _uuid(run_id)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            run = db.execute('SELECT * FROM project_task_runs WHERE id=? AND task_id=?', (run_id, task_id)).fetchone()
            if task['run_id'] != run_id or run is None or run['status'] != 'running' or run['owner_boot'] != self.boot_id or task['status'] in {'paused', 'cancelled', 'stale'}:
                return None
            rows = self._steps(db, task_id)
            if self._unresolved_attempt(db, task_id) or any(row['status'] == 'running' for row in rows):
                return None
            completed = {row['id'] for row in rows if row['status'] == 'completed'}
            row = next((row for row in rows if row['status'] == 'ready' and set(json.loads(row['dependencies'])) <= completed), None)
            if row is None:
                return None
            if _personal(row) and _selected_context(row, _contexts(context_snapshots)) != json.loads(row['context_snapshot']):
                self._invalidate(db, task, {row['id']}, 'source_changed', include_inventory=False)
                return None
            if row['kind'] == 'analysis' and run['attempts'] >= run['max_calls']:
                db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (run_id,))
                self._touch(db, task_id, status='paused', reason='budget_exhausted')
                return None
            token = str(uuid.uuid4())
            db.execute("UPDATE project_task_steps SET status='running',claim_token=?,claim_boot=?,claim_run=?,revision=revision+1,reason=NULL WHERE task_id=? AND id=? AND status='ready'", (token, self.boot_id, run_id, task_id, row['id']))
            if row['kind'] == 'analysis':
                db.execute('UPDATE project_task_runs SET attempts=attempts+1 WHERE id=?', (run_id,))
                db.execute('INSERT INTO project_task_attempts VALUES (?,?,?,?,?,?,?)', (row['request_id'], task_id, row['id'], row['generation'], run_id, 'running', self.clock()))
            result = self._touch(db, task_id)
            step = next(item for item in result['steps'] if item['id'] == row['id'])
            return {**step, 'task_id': task_id, 'run_id': run_id, 'claim_token': token}

    def _artifact_payload(self, row, artifact):
        if type(artifact) is not dict:
            raise ProjectTaskError('Invalid artifact references')
        kind = row['kind']
        if kind == 'analysis':
            allowed = {'answer_id', 'source_ids', 'document_ids', 'source_refs', 'proof', 'handoff', 'phase_handoff'}
            if _personal(row):
                allowed.update({'input_kind', 'context_refs'})
            if set(artifact) - allowed:
                raise ProjectTaskError('Artifacts store references, not answer or source copies')
            result = {'answer_id': _uuid(artifact.get('answer_id')), 'source_ids': _ids(artifact.get('source_ids', []), limit=10), 'document_ids': _ids(artifact.get('document_ids', []))}
            if any(value not in json.loads(row['input_snapshot']) for value in result['document_ids']):
                raise ProjectTaskError('Artifact documents exceed approved snapshot')
            if _personal(row):
                if (artifact.get('input_kind') != 'personal' or
                        type(artifact.get('context_refs')) is not dict or
                        _contexts(artifact.get('context_refs')) != json.loads(row['context_snapshot']) or
                        set(artifact.get('context_refs', {})) != set(json.loads(row['memory_ids'])) or
                        result['document_ids'] or result['source_ids']):
                    raise ProjectTaskError('Artifact personal context differs from approved snapshot')
                result.update(input_kind='personal', context_refs=dict(artifact['context_refs']))
            if 'handoff' in artifact:
                result['handoff'] = _handoff(artifact['handoff'],
                    sum(dep != 'inventory' for dep in json.loads(row['dependencies'])))
            if 'phase_handoff' in artifact:
                result['phase_handoff'] = _phase_handoff(artifact['phase_handoff'])
            # Proof content is always rehydrated from answer_id. Do not persist
            # a model-authored free-form "proof" or a second evidence body.
            if 'source_refs' in artifact:
                refs = artifact['source_refs']
                if type(refs) is not list or len(refs) > 10:
                    raise ProjectTaskError('Invalid source references')
                result['source_refs'] = []
                for ref in refs:
                    if type(ref) is not dict or set(ref) - {'id', 'label', 'document_id', 'excerpt_start', 'excerpt_end'}:
                        raise ProjectTaskError('Invalid source reference metadata')
                    source = {'id': _identifier(ref.get('id'))}
                    if source['id'] not in result['source_ids']:
                        raise ProjectTaskError('Source reference is not in artifact sources')
                    if 'label' in ref:
                        if type(ref['label']) is not str or not re.fullmatch(r'S(?:[1-9]|10)', ref['label']):
                            raise ProjectTaskError('Invalid source label')
                        source['label'] = ref['label']
                    if 'document_id' in ref:
                        source['document_id'] = _identifier(ref['document_id'])
                        if source['document_id'] not in result['document_ids']:
                            raise ProjectTaskError('Source document is not in artifact documents')
                    if 'excerpt_start' in ref or 'excerpt_end' in ref:
                        a, b = ref.get('excerpt_start'), ref.get('excerpt_end')
                        if type(a) is not int or type(b) is not int or not 0 <= a < b <= 2_000_000:
                            raise ProjectTaskError('Invalid excerpt coordinates')
                        source.update(excerpt_start=a, excerpt_end=b)
                    result['source_refs'].append(source)
        elif kind == 'inventory':
            if set(artifact) != {'documents'} or type(artifact['documents']) is not list or len(artifact['documents']) > 100:
                raise ProjectTaskError('Invalid inventory references')
            documents = artifact['documents']
            if any(type(item) is not dict or set(item) != {'id', 'digest'} for item in documents):
                raise ProjectTaskError('Invalid inventory reference')
            snapshot = _snapshot({item['id']: item['digest'] for item in documents})
            if len(snapshot) != len(documents) or snapshot != json.loads(row['input_snapshot']):
                raise ProjectTaskConflictError('Inventory differs from approved snapshot')
            result = {'documents': documents}
        else:
            if set(artifact) != {'step_refs'} or type(artifact['step_refs']) is not list or len(artifact['step_refs']) > MAX_ANALYSIS_STEPS:
                raise ProjectTaskError('Invalid report references')
            refs = artifact['step_refs']
            if any(type(ref) is not dict or set(ref) != {'step_id', 'version'} or type(ref['version']) is not int or not 1 <= ref['version'] <= MAX_ARTIFACT_VERSIONS for ref in refs):
                raise ProjectTaskError('Invalid report version references')
            keys = [ref['step_id'] for ref in refs]
            if len(set(keys)) != len(keys) or set(keys) != set(json.loads(row['dependencies'])):
                raise ProjectTaskError('Report must reference every analysis dependency')
            result = {'step_refs': refs}
        if len(_json(result).encode()) > MAX_ARTIFACT_BYTES:
            raise ProjectTaskLimitError('Artifact metadata exceeds its budget')
        return result

    def _complete(self, db, task_id, row, artifact):
        payload = self._artifact_payload(row, artifact)
        version = row['artifact_version'] + 1
        if version > MAX_ARTIFACT_VERSIONS:
            raise ProjectTaskLimitError('Artifact version capacity reached')
        if row['kind'] == 'report':
            steps = {item['id']: item for item in self._steps(db, task_id)}
            if any(steps[ref['step_id']]['status'] != 'completed' or steps[ref['step_id']]['artifact_version'] != ref['version'] for ref in payload['step_refs']):
                raise ProjectTaskConflictError('Report dependency changed')
        db.execute('INSERT INTO project_task_artifacts VALUES (?,?,?,?,?,?,?)', (task_id, row['id'], version, row['generation'], 'current', _json(payload), self.clock()))
        db.execute("UPDATE project_task_steps SET status='completed',artifact_version=?,claim_token=NULL,claim_boot=NULL,claim_run=NULL,reason=NULL,revision=revision+1 WHERE task_id=? AND id=?", (version, task_id, row['id']))
        if row['request_id']:
            db.execute("UPDATE project_task_attempts SET status='completed' WHERE request_id=?", (row['request_id'],))
        return self._touch(db, task_id)

    def complete_step(self, task_id, step_id, claim_token, artifact, *, context_snapshots=None):
        _identifier(step_id)
        _uuid(claim_token)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            row = db.execute('SELECT * FROM project_task_steps WHERE task_id=? AND id=?', (task_id, step_id)).fetchone()
            if row is None or row['status'] != 'running' or row['claim_token'] != claim_token or row['claim_boot'] != self.boot_id or task['status'] == 'cancelled':
                raise ProjectTaskConflictError('Step claim is no longer current')
            if _personal(row) and _selected_context(row, _contexts(context_snapshots)) != json.loads(row['context_snapshot']):
                raise ProjectTaskConflictError('Personal context changed during execution')
            return self._complete(db, task_id, row, artifact)

    def block_step(self, task_id, step_id, claim_token, *, reason):
        reason = {'unknown': 'provider_outcome_unknown', 'cancelled': 'provider_outcome_unknown',
                  'dependency_unavailable': 'dependency_blocked'}.get(reason, reason)
        if reason not in REASONS:
            raise ProjectTaskError('Invalid step failure reason')
        _uuid(claim_token)
        with self._db(write=True) as db:
            self._task(db, task_id)
            changed = db.execute("UPDATE project_task_steps SET status='blocked',reason=?,claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND id=? AND status='running' AND claim_token=? AND claim_boot=?", (reason, task_id, step_id, claim_token, self.boot_id)).rowcount
            if not changed:
                raise ProjectTaskConflictError('Step claim is no longer current')
            db.execute('UPDATE project_task_attempts SET status=? WHERE task_id=? AND step_id=? AND status=\'running\'', (reason, task_id, step_id))
            return self._touch(db, task_id, reason=reason)

    def _control(self, task_id, expected_revision, action):
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            self._cas(task, expected_revision)
            if task['status'] == 'cancelled':
                raise ProjectTaskConflictError('Cancelled tasks cannot resume')
            if action == 'cancelled':
                db.execute("UPDATE project_task_steps SET status='cancelled',claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND status!='completed'", (task_id,))
                db.execute("UPDATE project_task_attempts SET status='provider_outcome_unknown' WHERE task_id=? AND status='running'", (task_id,))
            if action in {'cancelled', 'paused'} and task['run_id']:
                db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (task['run_id'],))
            if action == 'ready':
                if task['status'] != 'paused':
                    raise ProjectTaskConflictError('Only a paused task can resume')
                # This is a state transition only; a fresh explicit run still
                # owns its allowance. No failed or ambiguous step is reset.
                db.execute("UPDATE project_tasks SET status='ready' WHERE id=?", (task_id,))
                return self._touch(db, task_id)
            return self._touch(db, task_id, status=action, reason='user_' + action)

    def pause_task(self, task_id, *, expected_revision):
        return self._control(task_id, expected_revision, 'paused')

    def resume_task(self, task_id, *, expected_revision):
        return self._control(task_id, expected_revision, 'ready')

    def cancel_task(self, task_id, *, expected_revision):
        return self._control(task_id, expected_revision, 'cancelled')

    def reset_failed_task(self, task_id, *, expected_revision):
        """Explicitly prepare known failed steps; never dispatch or erase cost."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            self._cas(task, expected_revision)
            rows = self._steps(db, task_id)
            if (task['status'] in {'cancelled', 'stale', 'completed'} or self._unresolved_attempt(db, task_id) or
                    any(row['status'] == 'running' or row['reason'] == 'provider_outcome_unknown' for row in rows)):
                raise ProjectTaskConflictError('Unknown or active work cannot be reset')
            eligible = [row for row in rows if row['status'] == 'blocked' and row['reason'] in
                        {'provider_failed', 'source_unavailable', 'invalid_result', 'dependency_blocked', 'provenance_unavailable'}]
            if not eligible:
                raise ProjectTaskConflictError('No known failed step can be reset')
            for row in eligible:
                db.execute("UPDATE project_task_steps SET status='ready',reason=NULL,revision=revision+1,generation=generation+1,request_id=? WHERE task_id=? AND id=?", (str(uuid.uuid4()) if row['kind'] == 'analysis' else None, task_id, row['id']))
            if task['run_id']:
                db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (task['run_id'],))
            db.execute("UPDATE project_tasks SET status='ready',run_id=NULL,generation=generation+1 WHERE id=?", (task_id,))
            return self._touch(db, task_id)

    def recover(self):
        """Startup-only repair; local work may be reclaimed, paid work cannot."""
        with self._db(write=True) as db:
            tasks = db.execute('SELECT * FROM project_tasks WHERE session_id=? AND instance_id=?', (self.session_id, self.instance_id)).fetchall()
            changed = []
            for task in tasks:
                rows = [row for row in self._steps(db, task['id']) if row['status'] == 'running' and row['claim_boot'] != self.boot_id]
                run = db.execute('SELECT * FROM project_task_runs WHERE id=? AND task_id=?', (task['run_id'], task['id'])).fetchone() if task['run_id'] else None
                lost_grant = run is not None and run['status'] == 'running' and run['owner_boot'] != self.boot_id
                if not rows and not lost_grant:
                    continue
                for row in rows:
                    state, reason = ('blocked', 'provider_outcome_unknown') if row['kind'] == 'analysis' else ('ready', 'process_restarted')
                    db.execute('UPDATE project_task_steps SET status=?,reason=?,claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND id=?', (state, reason, task['id'], row['id']))
                    if row['request_id']:
                        db.execute("UPDATE project_task_attempts SET status='provider_outcome_unknown' WHERE request_id=? AND status='running'", (row['request_id'],))
                if task['run_id']:
                    db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (task['run_id'],))
                changed.append(self._touch(db, task['id'], status='paused', reason='process_restarted'))
            return changed

    def _invalidate(self, db, task, affected, reason, *, include_inventory=True):
        rows = self._steps(db, task['id'])
        # Inventory is a scheduling dependency, not a semantic dependency of
        # every scoped question. Propagate only actual analysis dependencies.
        for _ in range(MAX_ANALYSIS_STEPS + 1):
            more = {row['id'] for row in rows if affected.intersection(json.loads(row['dependencies']))}
            if more <= affected:
                break
            affected.update(more)
        if include_inventory:
            affected.add('inventory')
        for row in rows:
            if row['id'] not in affected:
                continue
            db.execute("UPDATE project_task_steps SET status='stale',reason=?,claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND id=?", (reason, task['id'], row['id']))
            if row['request_id']:
                db.execute("UPDATE project_task_attempts SET status='provider_outcome_unknown' WHERE request_id=? AND status='running'", (row['request_id'],))
            db.execute("UPDATE project_task_artifacts SET status=? WHERE task_id=? AND step_id=? AND status='current'", ('withdrawn' if reason == 'source_revoked' else 'stale', task['id'], row['id']))
        if task['run_id']:
            db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (task['run_id'],))
        return self._touch(db, task['id'], status='cancelled' if task['status'] == 'cancelled' else 'stale', reason=reason)

    def invalidate_sources(self, changed_document_ids, *, reason='source_changed'):
        changed_ids = set(_ids(changed_document_ids))
        if reason not in {'source_changed', 'source_revoked', 'source_unavailable', 'provenance_unavailable'}:
            raise ProjectTaskError('Invalid source invalidation reason')
        if not changed_ids:
            return []
        result = []
        with self._db(write=True) as db:
            tasks = db.execute('SELECT * FROM project_tasks WHERE session_id=? AND instance_id=?', (self.session_id, self.instance_id)).fetchall()
            for task in tasks:
                rows = self._steps(db, task['id'])
                affected = {row['id'] for row in rows if row['kind'] == 'analysis' and not _personal(row) and (row['document_ids'] is None or changed_ids.intersection(json.loads(row['document_ids'])))}
                if not affected:
                    continue
                result.append(self._invalidate(db, task, affected, reason))
        return result

    def invalidate_memories(self, changed_memory_ids, *, reason='source_changed'):
        if type(changed_memory_ids) is not list or len(changed_memory_ids) > 100:
            raise ProjectTaskError('Invalid personal context changes')
        changed = {_uuid(identifier) for identifier in changed_memory_ids}
        if reason not in {'source_changed', 'source_revoked', 'source_unavailable', 'provenance_unavailable'}:
            raise ProjectTaskError('Invalid source invalidation reason')
        if not changed:
            return []
        result = []
        with self._db(write=True) as db:
            tasks = db.execute('SELECT * FROM project_tasks WHERE session_id=? AND instance_id=?', (self.session_id, self.instance_id)).fetchall()
            for task in tasks:
                affected = {row['id'] for row in self._steps(db, task['id']) if row['kind'] == 'analysis' and _personal(row)
                            and changed.intersection(json.loads(row['memory_ids']))}
                if affected:
                    result.append(self._invalidate(db, task, affected, reason, include_inventory=False))
        return result

    def refresh_task(self, task_id, *, expected_revision, source_snapshots, context_snapshots=None):
        snapshot = _snapshot(source_snapshots)
        contexts = _contexts(context_snapshots)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            self._cas(task, expected_revision)
            rows = self._steps(db, task_id)
            if task['status'] == 'cancelled' or any(row['status'] == 'running' for row in rows):
                raise ProjectTaskConflictError('Only an inactive task can refresh')
            if self._unresolved_attempt(db, task_id):
                raise ProjectTaskConflictError('Unknown paid outcomes require reconciliation')
            # Refresh is explicitly authorized and compares current source
            # identities itself; no earlier status GET is needed for safety.
            affected = set()
            for row in rows:
                if row['kind'] != 'analysis' or row['status'] == 'ready':
                    continue
                scope = json.loads(row['document_ids']) if row['document_ids'] is not None else None
                old = json.loads(row['input_snapshot'])
                current = {} if _personal(row) else snapshot if scope is None else {key: snapshot.get(key) for key in scope}
                # Missing/deleted context invalidates the old result but cannot
                # be silently removed from a new authorized input set.
                context = {key: contexts.get(key) for key in json.loads(row['memory_ids'])} if _personal(row) else {}
                if current != old or (_personal(row) and context != json.loads(row['context_snapshot'])):
                    affected.add(row['id'])
            if affected:
                self._invalidate(db, task, affected, 'source_changed')
                rows = self._steps(db, task_id)
            inventory = next(row for row in rows if row['kind'] == 'inventory')
            inventory_documents = {} if _personal(inventory) else snapshot
            inventory_context = _selected_context(inventory, contexts)
            if inventory['status'] == 'completed' and (json.loads(inventory['input_snapshot']) != inventory_documents or
                    (_personal(inventory) and json.loads(inventory['context_snapshot']) != inventory_context)):
                # An unrelated document can leave scoped analyses valid while
                # making the project inventory obsolete. Refresh that local
                # checkpoint alone; its scheduling edge is not a reason to
                # repeat already-completed questions on unchanged documents.
                if any(row['reason'] == 'provider_outcome_unknown' for row in rows):
                    raise ProjectTaskConflictError('Unknown paid outcomes require reconciliation')
                self._invalidate(db, task, set(), 'source_changed')
                rows = self._steps(db, task_id)
            if not any(row['status'] == 'stale' for row in rows):
                if any(row['reason'] == 'provider_outcome_unknown' for row in rows):
                    raise ProjectTaskConflictError('Unknown paid outcomes require reconciliation')
                return self._public(db, self._task(db, task_id))
            for row in rows:
                if row['status'] != 'stale':
                    continue
                scope = json.loads(row['document_ids']) if row['document_ids'] is not None else None
                current = {} if _personal(row) else _scoped(snapshot, scope)
                context = _selected_context(row, contexts)
                db.execute("UPDATE project_task_steps SET status='ready',revision=revision+1,generation=generation+1,request_id=?,input_snapshot=?,context_snapshot=?,reason=NULL WHERE task_id=? AND id=?", (str(uuid.uuid4()) if row['kind'] == 'analysis' else None, _json(current), _json(context), task_id, row['id']))
            db.execute('UPDATE project_tasks SET generation=generation+1,run_id=NULL WHERE id=?', (task_id,))
            return self._touch(db, task_id)

    def revise_task(self, task_id, *, expected_revision, goal, constraints, questions):
        """User revision, never an automatic rewrite or permission expansion.

        Version one preserves node identity/topology; editing a question or its
        explicit document scope invalidates that node and its actual dependents.
        """
        _text(goal, 1200, required=True)
        _text(constraints, 1200)
        questions = _questions(questions)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            self._cas(task, expected_revision)
            if task['status'] == 'cancelled':
                raise ProjectTaskConflictError('Cancelled tasks cannot be revised')
            rows = self._steps(db, task_id)
            existing = {row['id']: row for row in rows if row['kind'] == 'analysis'}
            if set(existing) != {row['id'] for row in questions}:
                raise ProjectTaskError('Revision must preserve existing question identifiers')
            changed = set()
            if goal != task['goal'] or constraints != task['constraints_text']:
                changed.update(existing)
            metadata_changed = bool(changed)
            for item in questions:
                old = existing[item['id']]
                if ['inventory', *item['depends_on']] != json.loads(old['dependencies']):
                    raise ProjectTaskError('Revision must preserve dependency topology')
                old_scope = json.loads(old['document_ids']) if old['document_ids'] is not None else None
                if (item['question'] != old['question'] or item['document_ids'] != old_scope or
                        item.get('input_kind', 'documents') != old['input_kind'] or
                        item.get('memory_ids', []) != json.loads(old['memory_ids'])):
                    changed.add(item['id'])
                if item['title'] != old['title']:
                    # A rendered report contains section titles. Changing a
                    # title must not mutate an already published report version.
                    changed.add('report')
                metadata_changed |= item['title'] != old['title'] or item['id'] in changed
            wrapper_kind = 'personal' if all(item.get('input_kind') == 'personal' for item in questions) else 'documents'
            wrapper_ids = list(dict.fromkeys(identifier for item in questions for identifier in item.get('memory_ids', []))) if wrapper_kind == 'personal' else []
            wrapper_changed = any(row['input_kind'] != wrapper_kind or json.loads(row['memory_ids']) != wrapper_ids
                                  for row in rows if row['kind'] in {'inventory', 'report'})
            metadata_changed |= wrapper_changed
            if not metadata_changed:
                return self._public(db, task)
            count = db.execute('SELECT count(*) FROM project_task_definitions WHERE task_id=?', (task_id,)).fetchone()[0]
            if count >= MAX_ARTIFACT_VERSIONS:
                raise ProjectTaskLimitError('Task definition history capacity reached')
            for _ in range(MAX_ANALYSIS_STEPS + 1):
                new = {row['id'] for row in rows if changed.intersection(json.loads(row['dependencies']))}
                if new <= changed:
                    break
                changed.update(new)
            # The inventory edge is scheduling only. Changing its context
            # union must not re-execute unrelated completed analysis nodes.
            if wrapper_changed:
                changed.update({'inventory', 'report'})
                db.execute("UPDATE project_task_steps SET input_kind=?,memory_ids=? WHERE task_id=? AND kind IN ('inventory','report')",
                           (wrapper_kind, _json(wrapper_ids), task_id))
            for item in questions:
                db.execute('UPDATE project_task_steps SET title=?,question=?,document_ids=?,input_kind=?,memory_ids=? WHERE task_id=? AND id=?', (item['title'], item['question'], _json(item['document_ids']) if item['document_ids'] is not None else None,
                           item.get('input_kind', 'documents'), _json(item.get('memory_ids', [])), task_id, item['id']))
            for key in changed:
                db.execute("UPDATE project_task_steps SET status='stale',reason='source_changed',claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND id=?", (task_id, key))
                db.execute("UPDATE project_task_artifacts SET status='stale' WHERE task_id=? AND step_id=? AND status='current'", (task_id, key))
                db.execute("UPDATE project_task_attempts SET status='provider_outcome_unknown' WHERE task_id=? AND step_id=? AND status='running'", (task_id, key))
            db.execute('UPDATE project_tasks SET goal=?,constraints_text=? WHERE id=?', (goal, constraints, task_id))
            db.execute('INSERT INTO project_task_definitions VALUES (?,?,?,?)', (task_id, count + 1, _json({'goal': goal, 'constraints': constraints, 'questions': questions}), self.clock()))
            if changed:
                db.execute("UPDATE project_task_runs SET status='paused' WHERE task_id=? AND status='running'", (task_id,))
            return self._touch(db, task_id, status='stale' if changed else task['status'], reason='source_changed' if changed else task['reason'])

    def _replan_view(self, db, task, *, goal, constraints, questions):
        _text(goal, 1200, required=True)
        _text(constraints, 1200)
        questions = _questions(questions)
        rows = self._steps(db, task['id'])
        previous = []
        for row in rows:
            if row['kind'] != 'analysis':
                continue
            item = {'id': row['id'], 'title': row['title'], 'question': row['question'],
                    'document_ids': json.loads(row['document_ids']) if row['document_ids'] is not None else None,
                    'depends_on': [key for key in json.loads(row['dependencies']) if key != 'inventory']}
            if _personal(row):
                item.update(input_kind='personal', memory_ids=json.loads(row['memory_ids']))
            previous.append(item)
        old, new = {row['id']: row for row in previous}, {row['id']: row for row in questions}
        all_rows = db.execute('SELECT * FROM project_task_steps WHERE task_id=?', (task['id'],)).fetchall()
        retired_ids = {row['id'] for row in all_rows if 'retired' in row.keys() and row['retired']}
        if retired_ids.intersection(new):
            raise ProjectTaskConflictError('Retired task identifiers cannot be reused')
        added, retired = set(new) - set(old), set(old) - set(new)
        changed = {key for key in old.keys() & new.keys()
                   if any(old[key].get(field) != new[key].get(field) for field in
                          ('question', 'document_ids', 'depends_on', 'input_kind', 'memory_ids'))}
        if goal != task['goal'] or constraints != task['constraints_text']:
            changed.update(old.keys() | new.keys())
        affected = changed | added | retired
        # Unioning both DAGs also invalidates descendants whose old ancestor
        # is retired or whose old edge is removed in the proposed definition.
        for _ in range(2 * MAX_ANALYSIS_STEPS + 1):
            more = {row['id'] for row in [*previous, *questions] if affected.intersection(row['depends_on'])}
            if more <= affected:
                break
            affected.update(more)
        modified = goal != task['goal'] or constraints != task['constraints_text'] or previous != questions
        if modified:
            affected.add('report')
        wrapper_kind = 'personal' if all(row.get('input_kind') == 'personal' for row in questions) else 'documents'
        wrapper_ids = list(dict.fromkeys(mid for row in questions for mid in row.get('memory_ids', []))) if wrapper_kind == 'personal' else []
        wrapper_changed = any(('personal' if _personal(row) else 'documents') != wrapper_kind or
                              (json.loads(row['memory_ids']) if _personal(row) else []) != wrapper_ids
                              for row in rows if row['kind'] in {'inventory', 'report'})
        if wrapper_changed:
            affected.add('inventory')  # Local scheduling checkpoint, not a semantic parent.
        version = db.execute('SELECT count(*) FROM project_task_definitions WHERE task_id=?', (task['id'],)).fetchone()[0]
        run = db.execute('SELECT status FROM project_task_runs WHERE id=?', (task['run_id'],)).fetchone() if task['run_id'] else None
        blocked = ('task_cancelled' if task['status'] == 'cancelled' else
                   'active_step' if any(row['status'] == 'running' for row in rows) else
                   'unresolved_outcome' if self._unresolved_attempt(db, task['id']) else
                   'active_run' if task['status'] in {'ready', 'running'} and run and run['status'] == 'running' else
                   'definition_capacity' if modified and version >= MAX_ARTIFACT_VERSIONS else None)
        payload = {'task_id': task['id'], 'session_id': self.session_id, 'instance_id': self.instance_id,
                   'base_revision': task['revision'], 'definition_version': version,
                   'goal': goal, 'constraints': constraints, 'questions': questions}
        view = {'task_id': task['id'], 'base_revision': task['revision'], 'definition_version': version,
                'proposal_sha256': hashlib.sha256(_json(payload).encode()).hexdigest(),
                'added': sorted(added), 'retired': sorted(retired), 'changed': sorted(changed),
                'affected': sorted(affected), 'unaffected_completed': sorted(row['id'] for row in rows
                    if row['kind'] == 'analysis' and row['status'] == 'completed' and row['id'] not in affected),
                'blocked': blocked is not None, 'blocked_reason': blocked, 'modified': modified,
                'active_analysis_count': len(questions), 'max_analysis_count': MAX_ANALYSIS_STEPS,
                'execution_authorized': False}
        return view, questions, wrapper_kind, wrapper_ids

    def preview_replan(self, task_id, *, expected_revision, goal, constraints, questions):
        """Read-only affected closure; no request IDs, grants or planner calls."""
        with self._db() as db:
            if db is not None:
                db.execute('BEGIN')
            task = self._task(db, task_id)
            self._cas(task, expected_revision)
            return self._replan_view(db, task, goal=goal, constraints=constraints, questions=questions)[0]

    def replan_task(self, task_id, *, expected_revision, proposal_sha256, goal, constraints, questions):
        """Apply one explicitly reviewed bounded DAG revision atomically.

        Completed unaffected answers and every old request identity survive.
        Changed work is only made stale; explicit refresh and a new approval
        still own any future model call. Retired IDs are durable tombstones.
        """
        if type(proposal_sha256) is not str or not re.fullmatch(r'[0-9a-f]{64}', proposal_sha256):
            raise ProjectTaskError('Invalid task proposal reference')
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            self._cas(task, expected_revision)
            view, proposed, wrapper_kind, wrapper_ids = self._replan_view(
                db, task, goal=goal, constraints=constraints, questions=questions)
            if view['proposal_sha256'] != proposal_sha256:
                raise ProjectTaskConflictError('Task proposal differs from the reviewed input')
            if view['blocked']:
                raise ProjectTaskConflictError('Active, unresolved or exhausted tasks cannot be replanned')
            if not view['modified']:
                return self._public(db, task)
            existing = {row['id']: row for row in self._steps(db, task_id)}
            for key in view['retired']:
                db.execute("UPDATE project_task_steps SET retired=1,status='cancelled',reason='source_changed',claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND id=?",
                           (task_id, key))
            for ordinal, item in enumerate(proposed, 1):
                dependencies = ['inventory', *item['depends_on']]
                if item['id'] in existing:
                    db.execute('UPDATE project_task_steps SET ordinal=?,title=?,question=?,document_ids=?,dependencies=?,input_kind=?,memory_ids=? WHERE task_id=? AND id=?',
                        (ordinal, item['title'], item['question'], _json(item['document_ids']) if item['document_ids'] is not None else None,
                         _json(dependencies), item.get('input_kind', 'documents'), _json(item.get('memory_ids', [])), task_id, item['id']))
                else:
                    self._insert_step(db, task_id, {**item, 'kind': 'analysis', 'depends_on': dependencies}, ordinal)
            db.execute("UPDATE project_task_steps SET ordinal=?,dependencies=? WHERE task_id=? AND id='report'",
                       (len(proposed) + 1, _json([row['id'] for row in proposed]), task_id))
            db.execute("UPDATE project_task_steps SET input_kind=?,memory_ids=? WHERE task_id=? AND kind IN ('inventory','report')",
                       (wrapper_kind, _json(wrapper_ids), task_id))
            for key in view['affected']:
                if key not in view['retired'] and key in existing:
                    db.execute("UPDATE project_task_steps SET status='stale',reason='source_changed',claim_token=NULL,claim_boot=NULL,claim_run=NULL,revision=revision+1 WHERE task_id=? AND id=?",
                               (task_id, key))
                db.execute("UPDATE project_task_artifacts SET status='stale' WHERE task_id=? AND step_id=? AND status='current'", (task_id, key))
            db.execute('UPDATE project_tasks SET goal=?,constraints_text=?,run_id=NULL WHERE id=?', (goal, constraints, task_id))
            db.execute('INSERT INTO project_task_definitions VALUES(?,?,?,?)',
                (task_id, view['definition_version'] + 1, _json({'goal': goal, 'constraints': constraints, 'questions': proposed}), self.clock()))
            db.execute("UPDATE project_task_runs SET status='paused' WHERE task_id=? AND status='running'", (task_id,))
            return self._touch(db, task_id, status='stale', reason='source_changed')

    def definition_history(self, task_id):
        with self._db() as db:
            self._task(db, task_id)
            return [{'version': row['version'], 'created_at': row['created_at'], **json.loads(row['payload'])} for row in db.execute('SELECT * FROM project_task_definitions WHERE task_id=? ORDER BY version DESC', (task_id,)).fetchall()]

    def reconcile_step(self, task_id, step_id, *, request_id, artifact, source_snapshots, context_snapshots=None):
        _uuid(request_id)
        snapshot = _snapshot(source_snapshots)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            row = db.execute('SELECT * FROM project_task_steps WHERE task_id=? AND id=?', (task_id, step_id)).fetchone()
            if task['status'] == 'cancelled' or row is None or row['status'] != 'blocked' or row['reason'] != 'provider_outcome_unknown' or row['request_id'] != request_id:
                raise ProjectTaskConflictError('Step has no reconcilable paid outcome')
            scope = json.loads(row['document_ids']) if row['document_ids'] is not None else None
            current = {} if _personal(row) else _scoped(snapshot, scope)
            if current != json.loads(row['input_snapshot']) or (_personal(row) and _selected_context(row, _contexts(context_snapshots)) != json.loads(row['context_snapshot'])):
                raise ProjectTaskConflictError('Reconciled answer belongs to a changed snapshot')
            self._complete(db, task_id, row, artifact)
            if task['run_id']:
                db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (task['run_id'],))
            return self._touch(db, task_id, status='paused', reason='process_restarted')

    def unresolved_requests(self, task_id):
        """Scoped journal lookup IDs, never an approval to repeat a request."""
        with self._db() as db:
            self._task(db, task_id)
            return {row['request_id'] for row in db.execute(
                "SELECT request_id FROM project_task_attempts WHERE task_id=? AND status='provider_outcome_unknown'", (task_id,))}

    def settle_invalidated_request(self, task_id, step_id, *, request_id, outcome):
        """Record a journal terminal result without publishing outdated work.

        Source/goal revisions cannot cancel an accepted call. Once its exact
        journal request is terminal, a fresh explicit refresh/run may continue;
        neither its old answer nor an old approval is made current here.
        """
        _uuid(request_id)
        _identifier(step_id)
        if outcome not in {'completed', 'failed'}:
            raise ProjectTaskError('Request has no terminal outcome')
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            row = db.execute('SELECT * FROM project_task_steps WHERE task_id=? AND id=?', (task_id, step_id)).fetchone()
            if task['status'] == 'cancelled' or row is None or row['status'] != 'stale' or row['request_id'] != request_id:
                raise ProjectTaskConflictError('Invalidated request is no longer current')
            changed = db.execute("UPDATE project_task_attempts SET status=? WHERE request_id=? AND task_id=? AND step_id=? AND generation=? AND status='provider_outcome_unknown'",
                (outcome, request_id, task_id, step_id, row['generation'])).rowcount
            if not changed:
                raise ProjectTaskConflictError('Request has no unresolved outcome')
            return self._touch(db, task_id, status='stale', reason=task['reason'])

    def reconcile_failed_step(self, task_id, step_id, *, request_id, source_snapshots, context_snapshots=None):
        """A trusted journal can resolve unknown to known failure, never retry."""
        _uuid(request_id)
        snapshot = _snapshot(source_snapshots)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            row = db.execute('SELECT * FROM project_task_steps WHERE task_id=? AND id=?', (task_id, step_id)).fetchone()
            if task['status'] == 'cancelled' or row is None or row['status'] != 'blocked' or row['reason'] != 'provider_outcome_unknown' or row['request_id'] != request_id:
                raise ProjectTaskConflictError('Step has no reconcilable paid outcome')
            scope = json.loads(row['document_ids']) if row['document_ids'] is not None else None
            current = {} if _personal(row) else _scoped(snapshot, scope)
            if current != json.loads(row['input_snapshot']) or (_personal(row) and _selected_context(row, _contexts(context_snapshots)) != json.loads(row['context_snapshot'])):
                raise ProjectTaskConflictError('Reconciled failure belongs to a changed snapshot')
            db.execute("UPDATE project_task_steps SET reason='provider_failed',revision=revision+1 WHERE task_id=? AND id=?", (task_id, step_id))
            db.execute("UPDATE project_task_attempts SET status='provider_failed' WHERE request_id=?", (request_id,))
            if task['run_id']:
                db.execute("UPDATE project_task_runs SET status='paused' WHERE id=?", (task['run_id'],))
            return self._touch(db, task_id, status='paused', reason='provider_failed')

    def get_artifact(self, task_id, step_id, version=None):
        _identifier(step_id)
        if version is not None and (type(version) is not int or not 1 <= version <= MAX_ARTIFACT_VERSIONS):
            raise ProjectTaskError('Invalid artifact version')
        with self._db() as db:
            self._task(db, task_id)
            row = db.execute('SELECT * FROM project_task_artifacts WHERE task_id=? AND step_id=? ' + ('AND version=?' if version is not None else 'ORDER BY version DESC LIMIT 1'), (task_id, step_id, version) if version is not None else (task_id, step_id)).fetchone()
            if row is None:
                raise ProjectTaskNotFoundError('Artifact is unavailable')
            result = {key: row[key] for key in ('task_id', 'step_id', 'version', 'generation', 'status', 'created_at')}
            result['available'] = row['status'] == 'current'
            result['references'] = json.loads(row['payload']) if result['available'] else None
            return result

    def artifact_history(self, task_id, step_id):
        _identifier(step_id)
        with self._db() as db:
            self._task(db, task_id)
            return [{**dict(row), 'available': row['status'] == 'current'} for row in db.execute('SELECT task_id,step_id,version,generation,status,created_at FROM project_task_artifacts WHERE task_id=? AND step_id=? ORDER BY version DESC', (task_id, step_id)).fetchall()]

    def delete_session_metadata(self, session_id):
        """Parent deletion clears all reused-number generations atomically.

        No empty file is created. Deletion must complete before the session is
        removed; old claim completion then fails because its task no longer
        exists. This does not delete the separate paid-request journal.
        """
        if type(session_id) is not str or not re.fullmatch(r'#[0-9]{5}', session_id) or session_id != self.session_id:
            raise ProjectTaskError('Invalid cleanup project reference')
        if not self._file():
            return
        with self._db(write=True) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {'long_projects', 'long_project_phases'} <= tables:
                phase_scope = ('SELECT p.id FROM long_project_phases p JOIN long_projects r '
                               'ON r.id=p.project_id WHERE r.session_id=?')
                if 'long_project_phase_parents' in tables:
                    db.execute('DELETE FROM long_project_phase_parents WHERE child_phase_id IN (' + phase_scope +
                               ') OR parent_phase_id IN (' + phase_scope + ')', (session_id, session_id))
                if 'long_project_reviews' in tables:
                    db.execute('DELETE FROM long_project_reviews WHERE project_id IN '
                               '(SELECT id FROM long_projects WHERE session_id=?)', (session_id,))
                if 'long_project_requests' in tables:
                    db.execute('DELETE FROM long_project_requests WHERE session_id=?', (session_id,))
                db.execute('DELETE FROM long_project_phases WHERE project_id IN '
                           '(SELECT id FROM long_projects WHERE session_id=?)', (session_id,))
                db.execute('DELETE FROM long_projects WHERE session_id=?', (session_id,))
            keys = [row[0] for row in db.execute('SELECT id FROM project_tasks WHERE session_id=?', (session_id,)).fetchall()]
            for key in keys:
                for table in ('project_task_steps', 'project_task_artifacts', 'project_task_runs',
                              'project_task_definitions', 'project_task_attempts'):
                    db.execute('DELETE FROM ' + table + ' WHERE task_id=?', (key,))
                db.execute('DELETE FROM project_tasks WHERE id=?', (key,))
