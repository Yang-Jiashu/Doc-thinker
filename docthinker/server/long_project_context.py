"""On-demand phase dependency resolution over the existing durable task DAG.

Only selected current parent analyses are handed off. Ancestors are checked by
reference, never concatenated into a growing transcript or summarized by a model.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import json
import sqlite3
import time

from .long_project_store import LongProjectStore
from .project_task_store import ProjectTaskConflictError

# Inspection is bounded by actual work, not the project's retained lifetime.
MAX_CHECK_BYTES = 4 * 1024 * 1024
MAX_CHECK_SECONDS = 2.0


class _InspectionLimit(ProjectTaskConflictError):
    pass


class _Budget:
    def __init__(self):
        self.deadline = time.monotonic() + MAX_CHECK_SECONDS
        self.bytes = 0

    def check(self, size=0):
        self.bytes += size
        if self.bytes > MAX_CHECK_BYTES:
            raise _InspectionLimit('Phase dependency inspection byte budget exceeded; no ancestor was skipped')
        if time.monotonic() >= self.deadline:
            raise _InspectionLimit('Phase dependency inspection deadline exceeded; no ancestor was skipped')


class LongProjectContext:
    def __init__(self, tasks, bridge):
        self.tasks, self.bridge = tasks, bridge
        self.projects = LongProjectStore(tasks)

    def execution_context(self, *, task_id):
        with self.tasks._db() as db:
            self.tasks._task(db, task_id)
            db.execute('BEGIN')
            phase = self._phase_for_task(db, task_id)
            if phase is None:
                return None
            project = self.projects._project(db, phase['project_id'])
            return {'project_goal': project['goal'],
                    'project_acceptance_criteria': json.loads(project['acceptance_criteria']),
                    'phase_id': phase['id'], 'phase_title': phase['title'],
                    'phase_acceptance_criteria': json.loads(phase['acceptance_criteria'])}

    def _phase_for_task(self, db, task_id):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='long_project_phases'").fetchone():
            return None
        return db.execute('SELECT p.* FROM long_project_phases p JOIN long_projects r ON r.id=p.project_id '
            'WHERE p.task_id=? AND r.session_id=? AND r.instance_id=?',
            (task_id, self.tasks.session_id, self.tasks.instance_id)).fetchone()

    def dependencies(self, *, task_id, step=None):
        return self._inspect(task_id, include_dependencies=True)

    def _session_live(self):
        session = self.bridge.manager.get_session(self.bridge.session_id)
        return bool(session and session.get('instance_id') == getattr(self.bridge, 'instance_id', None)
                    and self.bridge.session_id == self.tasks.session_id)

    @contextmanager
    def _snapshot(self, budget):
        try:
            with self.tasks._db() as db:
                if db is not None:
                    db.execute('PRAGMA busy_timeout=50')
                    db.execute('BEGIN')
                    db.set_progress_handler(lambda: int(time.monotonic() >= budget.deadline), 256)
                yield db
        except sqlite3.Error:
            budget.check()  # Turn deadline interruption into an explicit gap.
            raise

    def _inspect(self, task_id, *, include_dependencies=False):
        """One task snapshot plus one current-input snapshot, all request-local."""
        budget, groups, result = _Budget(), {}, []
        with self._snapshot(budget) as db:
            self.tasks._task(db, task_id)
            queue, pending, visited = deque([task_id]), {task_id}, set()
            while queue:
                budget.check()
                current = queue.popleft(); pending.discard(current)
                if current in visited:
                    continue
                visited.add(current)
                phase = self._phase_for_task(db, current)
                if phase is None:
                    continue
                budget.check(len(phase['parent_refs'].encode('utf-8')))
                refs = json.loads(phase['parent_refs'])
                if type(refs) is not list or len(refs) > 6:
                    raise ProjectTaskConflictError('Invalid phase dependency scope')
                for ref in refs:
                    budget.check()
                    parent = self.projects._phase(db, phase['project_id'], ref['phase_id'])
                    if parent['task_id'] != ref['task_id'] or parent['ordinal'] >= phase['ordinal']:
                        raise ProjectTaskConflictError('Parent scope or phase order changed')
                    versions = ref['step_versions']
                    if type(versions) is not dict or not 1 <= len(versions) <= 6:
                        raise ProjectTaskConflictError('Invalid parent step selection')
                    # Meter actual pinned payload sizes before parsing the
                    # existing binding validator; no giant opaque row bypass.
                    conditions = ["(step_id='report' AND version=?)"]
                    parameters = [ref['task_id'], ref['report_version']]
                    for key, version in versions.items():
                        conditions.append('(step_id=? AND version=?)')
                        parameters.extend((key, version))
                    size = db.execute('SELECT coalesce(sum(length(cast(payload AS BLOB))),0) '
                        'FROM project_task_artifacts WHERE task_id=? AND (' + ' OR '.join(conditions) + ')',
                        parameters).fetchone()[0]
                    budget.check(size)
                    self.projects._selected_parent_live(db, parent, ref)
                    selected = [row for row in self.tasks._steps(db, parent['task_id']) if row['id'] in versions]
                    if len(selected) != len(versions):
                        raise ProjectTaskConflictError('Selected parent analysis is unavailable')
                    group = []
                    selected_payloads = {}
                    for row in selected:
                        packet = {'input_kind': row['input_kind'],
                                  'document_ids': json.loads(row['document_ids']) if row['document_ids'] is not None else None,
                                  'input_snapshot': json.loads(row['input_snapshot']),
                                  'context_snapshot': json.loads(row['context_snapshot'])}
                        saved = db.execute("SELECT payload FROM project_task_artifacts WHERE task_id=? "
                            "AND step_id=? AND version=? AND status='current'",
                            (parent['task_id'], row['id'], versions[row['id']])).fetchone()
                        if saved is None:
                            raise ProjectTaskConflictError('Selected parent analysis is unavailable')
                        payload = json.loads(saved['payload'])
                        selected_payloads[row['id']] = payload
                        # Legacy callback artifacts lack review provenance.
                        # Modern reviews must retain the exact originating
                        # answer, including event-only revision invalidation.
                        if any(key in payload for key in ('event_reasoning', 'task_review', 'analysis_review')):
                            packet['answer_id'] = payload.get('answer_id')
                        encoded = json.dumps(packet, sort_keys=True, separators=(',', ':'), allow_nan=False)
                        budget.check(len(encoded.encode('utf-8')))
                        group.append(packet)
                    encoded_group = json.dumps(group, sort_keys=True, separators=(',', ':'), allow_nan=False)
                    groups[encoded_group] = group
                    if include_dependencies and current == task_id:
                        for key, version in versions.items():
                            result.append({'step_id': 'p:' + ref['phase_id'] + ':' + key,
                                'version': version, 'references': selected_payloads[key],
                                'phase_ref': {'phase_id': ref['phase_id'], 'task_id': ref['task_id'],
                                    'report_version': ref['report_version'], 'step_id': key, 'version': version}})
                    if ref['task_id'] not in pending and ref['task_id'] not in visited:
                        pending.add(ref['task_id']); queue.append(ref['task_id'])
            if len(result) > 6:
                raise ProjectTaskConflictError('Too many selected parent analyses')
            budget.check()
            if groups:
                if not self._session_live():
                    raise ProjectTaskConflictError('Phase owner instance changed')
                inputs_live = self.bridge.service.task_input_groups_live(
                    self.bridge.session_id, list(groups.values()), deadline=budget.deadline)
                budget.check()
                if not inputs_live or not self._session_live():
                    raise ProjectTaskConflictError('Phase input provenance is unavailable')
            budget.check()
            return result

    def valid(self, task_id):
        """No fixed stage-lifetime wall, skipped ancestors or freshness cache."""
        try:
            self._inspect(task_id)
            return True
        except (ValueError, sqlite3.Error, KeyError, TypeError, AttributeError):
            return False

    def invalidate_if_stale(self, task_id):
        try:
            self._inspect(task_id)
        except _InspectionLimit:
            # Capacity is not evidence that an old source changed. Refuse this
            # attempt explicitly rather than permanently rewriting its state.
            raise
        except (ValueError, sqlite3.Error, KeyError, TypeError, AttributeError):
            pass
        else:
            return self.tasks.get_task(task_id)
        with self.tasks._db(write=True) as db:
            task = self.tasks._task(db, task_id)
            rows = self.tasks._steps(db, task_id)
            affected = {row['id'] for row in rows if row['kind'] == 'analysis' and row['status'] != 'stale'}
            if not affected:
                return self.tasks._public(db, task)
            return self.tasks._invalidate(db, task, affected, 'source_changed', include_inventory=False)
