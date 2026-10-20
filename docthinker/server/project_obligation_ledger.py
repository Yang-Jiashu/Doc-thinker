"""A version-bound work ledger over the existing project task database.

This is a projection of current, live-checked artifact receipts, not another
agent, an authorization, or a truth database. It never stores source/answer
bodies and never promotes successful execution or candidate retrieval to proof.
"""
from __future__ import annotations

import hashlib
import json

from .project_evidence_feedback import feedback_view
from .project_task_store import (ProjectTaskStore, ProjectTaskError,
                                ProjectTaskConflictError, ProjectTaskLimitError)
from .research_task_review import build_task_requirements, task_requirement_rows
from .project_analysis_lineage import superseded_step_ids

MAX_OBLIGATIONS = 600
MAX_LEDGER_BYTES = 2 * 1024 * 1024
LOOKUP_POLICY_VERSION = 'goal_evidence_lookup_v2'
LOOKUP_OUTCOMES = ('partially_searched', 'exhausted', 'candidates_found')
_STATES = ('ready_for_review', 'unresolved', 'contradicted', 'not_assessed')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value, maximum=2500):
    if (type(value) is not str or not value.strip() or len(value) > maximum
            or any(ord(c) < 32 and c not in '\n\t' for c in value)):
        raise ProjectTaskError('Invalid obligation ledger text')
    return value


def _lookup_hint(value):
    """Bound an optional search expression, never a goal or source witness.

    Model-provided missing queries already have a 400-byte validated bound.
    A longer host requirement is only a fallback retrieval hint; the complete
    requirement remains in ``query`` and the original approved task definition.
    """
    encoded = value.encode('utf-8')
    return value if len(encoded) <= 400 else encoded[:397].decode('utf-8', errors='ignore') + '…'


def _state(row, *, event=False, personal=False):
    """A review-ready candidate is explicitly not verified semantic truth."""
    if row is None:
        return 'not_assessed'
    if not event and row.get('status') == 'unknown' and row.get('gaps') == ['requirement_unassessed']:
        return 'not_assessed'
    if event and row.get('relation') == 'contradicts':
        # Refuting an alleged conflict removes that obstacle; supporting the
        # conflict is adverse to the proposed chain. Preserve typed polarity.
        if row['status'] == 'model_assessed_support':
            return 'contradicted'
        if row['status'] == 'model_assessed_counterevidence' and row.get('evidence') and not row.get('gaps'):
            return 'ready_for_review'
        return 'unresolved'
    if row['status'] in {'model_assessed_counterevidence'}:
        return 'contradicted'
    good = 'model_assessed_support' if event else 'model_assessed_addressed'
    if (personal and not event and row['status'] == good and not row.get('gaps')
            and row.get('basis') == 'answer_fulfilment_only' and row.get('answer_witness')):
        # A user-requested private draft can be delivered without pretending
        # that a document externally proves it. This never increments the
        # receipt's factual-support count or verifies a real-world action.
        return 'ready_for_review'
    if (row['status'] == good and row.get('evidence') and not row.get('gaps')
            and (event or row.get('basis') == 'source_and_answer')):
        return 'ready_for_review'
    return 'unresolved'


def _binding(task, step, payload, review_binding=None):
    # The originating answer is the reference whose event/memory/source
    # dependencies the caller checks at the live barrier. No event L# alias is
    # ever used as a cross-round identity.
    return {'task_id': task['id'], 'step_id': step['id'],
            'artifact_version': step.get('artifact_version', 0),
            'step_generation': step.get('generation', 1),
            'answer_id': payload.get('answer_id'),
            'input_sha256': _hash({'documents': step.get('input_snapshot', {}),
                                   'memory': step.get('context_snapshot', {})}),
            'receipt_sha256': review_binding.get('receipt_sha256') if review_binding else None}


def _view(task, step, payload, kind):
    name = 'task_review' if kind == 'task' else 'event_reasoning'
    binding = payload.get('task_review_binding' if kind == 'task' else 'evidence_feedback_binding')
    if name not in payload:
        return None
    return feedback_view(payload[name], binding, task_id=task['id'], step_id=step['id'],
                         artifact_version=step['artifact_version'],
                         current_task_revision=task['revision'], review_kind=kind)


def build_obligation_progress(task, artifacts, *, execution_context=None, eligible_step_ids=None):
    """Pure complete host-requirement projection; eligibility must be live-checked.

    ``artifacts`` maps step IDs to current reference payloads. Missing eligibility
    is intentionally conservative, not an implicit trust in completed steps.
    Unknown/malformed/stale receipts cannot make a requirement review-ready.
    """
    if type(task) is not dict or type(artifacts) is not dict:
        raise ProjectTaskError('Invalid obligation ledger input')
    steps = task.get('steps')
    if type(steps) is not list:
        raise ProjectTaskError('Invalid obligation ledger steps')
    analysis = [row for row in steps if row.get('kind') == 'analysis']
    if not 1 <= len(analysis) <= 12:
        raise ProjectTaskError('Invalid obligation ledger step capacity')
    eligible = [] if eligible_step_ids is None else eligible_step_ids
    if (type(eligible) not in (list, tuple, set, frozenset)
            or any(type(i) is not str for i in eligible)
            or len(set(eligible)) != len(eligible)
            or not set(eligible) <= {row['id'] for row in analysis}):
        raise ProjectTaskError('Invalid live obligation step scope')
    context = execution_context or {}
    if type(context) is not dict:
        raise ProjectTaskError('Invalid obligation project context')
    project_goal = context.get('project_goal', task['goal'])
    project_criteria = context.get('project_acceptance_criteria', [])
    phase_criteria = context.get('phase_acceptance_criteria', [])
    obligations = []
    fingerprints = []
    by_step = {row['id']: row for row in analysis}
    # The approved definition, not matching text or a successful review, owns
    # replacement. A pending/revoked replacement stays unknown; it must not
    # reactivate an earlier draft's positive proof as an automatic fallback.
    replaced = superseded_step_ids(analysis)
    lineage_cache = {}

    def lineage(step_id, visiting=()):
        """Only an explicit, current checked dependency may reuse a proof."""
        if step_id in lineage_cache:
            return lineage_cache[step_id]
        step, payload = by_step.get(step_id), artifacts.get(step_id, {})
        value = None
        try:
            from .research_review_stage import normalize_review_lineage
            if (step is None or step_id in visiting or step_id not in eligible
                    or step.get('status') != 'completed' or type(payload) is not dict):
                raise ValueError('Review lineage is not currently eligible')
            value = normalize_review_lineage(payload.get('analysis_review'))
            if value['body_sha256'] != value['root_body_sha256']:
                raise ValueError('Review draft changed')
            if step.get('purpose', 'compose') == 'compose':
                if (value['purpose'] != 'compose' or value['root_answer_id'] != payload.get('answer_id')
                        or value['target_answer_id'] is not None):
                    raise ValueError('Writer lineage does not own its body')
            else:
                target_id = step.get('review_of')
                target = lineage(target_id, (*visiting, step_id))
                prior = artifacts.get(target_id, {})
                if (value['purpose'] != 'verify' or target is None
                        or target_id not in step.get('depends_on', [])
                        or step['question'] != by_step[target_id]['question']
                        or value['target_answer_id'] != prior.get('answer_id')
                        or any(value[key] != target[key] for key in ('root_answer_id', 'root_body_sha256',
                            'body_sha256', 'context_sha256', 'source_scope_sha256'))):
                    raise ValueError('Review lineage differs from its current target')
        except (ValueError, TypeError, KeyError):
            value = None
        lineage_cache[step_id] = value
        return value

    def bind_lineage(binding, step_id):
        value = lineage(step_id)
        if value is None:
            return binding
        return {**binding, 'review_lineage_sha256': _hash({key: value[key] for key in
            ('root_answer_id', 'root_body_sha256', 'context_sha256', 'source_scope_sha256')})}

    def add(step, kind, requirement_id, semantic_id, state, query, gaps, binding,
            requirement_sha256=None, lookup_query=None):
        query = _text(query)
        identifier = _hash({'step_id': step['id'], 'kind': kind,
                            'semantic_id': semantic_id, 'binding': binding,
                            'requirement_sha256': requirement_sha256})
        obligations.append({'id': identifier, 'owner_step_id': step['id'],
                            'kind': kind, 'requirement_id': requirement_id,
                            'semantic_id': semantic_id, 'status': state,
                            'query': query, 'gaps': list(gaps), 'binding': binding,
                            'lookup_query': _lookup_hint(lookup_query or query),
                            'requirement_sha256': requirement_sha256})

    for step in analysis:
        if step['id'] in replaced:
            fingerprints.append({'step_id': step['id'], 'superseded': True,
                                 'artifact_sha256': _hash(artifacts.get(step['id'], {}))})
            continue
        question = _text(step['question'], 2000)
        # The complete original goal always exists in this catalog, including
        # legacy receipts which never assessed it. No model can omit a goal.
        packet = build_task_requirements(question, phase_criteria,
            input_kind=step.get('input_kind', 'documents'), project_goal=project_goal,
            project_acceptance_criteria=project_criteria)
        catalog = task_requirement_rows(packet)
        payload = artifacts.get(step['id'], {})
        live = step['id'] in eligible and step.get('status') == 'completed'
        if type(payload) is not dict:
            raise ProjectTaskError('Invalid current obligation artifact')
        task_view = event_view = None
        invalid_task = invalid_event = False
        if live:
            try:
                task_view = _view(task, step, payload, 'task')
            except (ValueError, TypeError, KeyError):
                invalid_task = True
            try:
                event_view = _view(task, step, payload, 'event')
            except (ValueError, TypeError, KeyError):
                invalid_event = True
        receipt = task_view['task_review'] if task_view else None
        expected_packet = packet
        if receipt and receipt.get('requirements_version', 1) == 1:
            expected_packet = build_task_requirements(question, phase_criteria,
                input_kind=step.get('input_kind', 'documents'))
        if receipt and receipt['requirements_sha256'] != expected_packet['requirements_sha256']:
            receipt, task_view, invalid_task = None, None, True
        assessed = {row['id']: row for row in receipt['obligations']} if receipt else {}
        task_binding = _binding(task, step, payload, task_view['binding'] if task_view else None)
        task_binding = bind_lineage(task_binding, step['id'])
        task_binding['requirements_sha256'] = packet['requirements_sha256']
        fingerprints.append({'step_id': step['id'], 'live': live, 'task_binding': task_binding,
                             'artifact_sha256': _hash(payload)})
        queries = receipt['suggested_queries'] if receipt else []
        query_by_hash = {hashlib.sha256(query.encode('utf-8')).hexdigest(): query for query in queries}
        for required in catalog:
            row = assessed.get(required['id'])
            if (row and receipt.get('requirements_version', 1) == 2
                    and row.get('requirement_sha256') != required['requirement_sha256']):
                row = None
                invalid_task = True
            gaps = (row.get('gaps', []) if row else
                    ['receipt_invalid' if invalid_task else 'not_assessed'])
            # The query is a bounded hint. The complete original requirement
            # remains host-owned and hash-bound even when a hint is available.
            hint = (row.get('lookup_query') or query_by_hash.get(row.get('lookup_query_sha256'))
                    if row and row.get('gaps') else None)
            query = required['text'] + '\nMissing-evidence lookup: ' + hint if hint else required['text']
            add(step, 'task_requirement', required['id'], required['requirement_sha256'],
                _state(row, personal=step.get('input_kind') == 'personal')
                if receipt and receipt['model_assessed'] else 'not_assessed',
                query, gaps, task_binding, required['requirement_sha256'],
                lookup_query=hint or required['text'])

        if event_view:
            event_receipt = event_view['event_reasoning']
            binding = bind_lineage(_binding(task, step, payload, event_view['binding']), step['id'])
            for row in event_receipt['obligations']:
                semantic_id = row.get('relation_id') or row.get('event_id')
                if type(semantic_id) is not str or not 1 <= len(semantic_id) <= 128:
                    semantic_id = None
                missing_identity = row['kind'] in {'relation', 'event'} and not semantic_id
                # Whole-path and alternatives identities bind to the complete
                # receipt, never to mutable local P1/A1 ordering aliases.
                semantic_id = semantic_id or _hash({'context': binding.get('review_lineage_sha256',
                    binding['receipt_sha256']), 'kind': row['kind']})
                gaps = list(row['gaps'])
                if missing_identity:
                    gaps.append('stable_event_identity_missing')
                state = _state(row, event=True) if event_receipt['model_assessed'] else 'not_assessed'
                if (row.get('status') == 'unknown' and 'assessed_need_ids' in event_receipt
                        and row['id'] not in event_receipt['assessed_need_ids']):
                    state = 'not_assessed'
                if missing_identity and state == 'ready_for_review':
                    state = 'unresolved'
                query = row.get('lookup_query') or next((gap['query'] for gap in event_receipt['gaps']
                              if gap['obligation_id'] == row['id'] and gap.get('query')), None)
                # Never substitute another obligation's first generic query.
                # Legacy receipts without a per-item hint retain the original
                # question as an explicitly unassessed fallback.
                query = query or question
                add(step, 'event_' + row['kind'], row['id'], semantic_id,
                    state, query, gaps, binding)
        elif invalid_event:
            binding = _binding(task, step, payload)
            add(step, 'event_review', 'event_review', _hash({'step': step['id']}),
                'not_assessed', question, ['receipt_invalid'], binding)

    # Repeated audits inspect one immutable writer body/context. Collapse only
    # that explicit live-checked lineage, never similarly named events from
    # another draft. Unknown writer rows do not permanently defeat later proof.
    review_groups, retained = {}, []
    for row in obligations:
        identity = row['binding'].get('review_lineage_sha256')
        if identity and (row['kind'].startswith('event_') or row['requirement_id'] == 'Q1'):
            key = (identity, row['kind'], row['semantic_id'])
            review_groups.setdefault(key, []).append(row)
        else:
            retained.append(row)
    for identity, members in review_groups.items():
        adverse = [row for row in members if row['status'] == 'contradicted']
        ready = [row for row in members if row['status'] == 'ready_for_review']
        unresolved = [row for row in members if row['status'] == 'unresolved']
        chosen = (adverse or ready or unresolved or members)[-1]
        group = _hash([{'binding': row['binding'], 'status': row['status']} for row in members])
        retained.append({**chosen, 'binding': {**chosen['binding'], 'review_group_sha256': group},
            'id': _hash({'review_lineage': identity, 'members': group}),
            'gaps': list(dict.fromkeys([*chosen['gaps'], *(['conflicting_reviews'] if adverse and ready else [])]))})
    obligations = retained
    # A project/phase criterion is a shared requirement, not a demand that
    # every small analysis independently solve the whole project. Keep every
    # original step question, but combine full-criterion reviews under their
    # host text hash. Any explicit contradiction wins over a positive claim.
    # A complete positive review remains only a review-ready candidate.
    shared, combined = {}, []
    for row in obligations:
        if row['kind'] == 'task_requirement' and row['requirement_id'] != 'Q1':
            shared.setdefault(row['requirement_sha256'], []).append(row)
        else:
            combined.append(row)
    for requirement_hash, members in shared.items():
        contradicted = [row for row in members if row['status'] == 'contradicted']
        ready = [row for row in members if row['status'] == 'ready_for_review']
        unresolved = [row for row in members if row['status'] == 'unresolved']
        candidates = contradicted or ready or unresolved or members
        chosen = candidates[-1]
        binding = {**chosen['binding'], 'review_group_sha256': _hash([
            {'binding': row['binding'], 'status': row['status']} for row in members])}
        combined.append({**chosen, 'binding': binding,
            'id': _hash({'kind': 'shared_requirement', 'requirement_sha256': requirement_hash,
                         'binding': binding}),
            'gaps': list(dict.fromkeys([*chosen['gaps'], *(['conflicting_reviews']
                         if contradicted and ready else [])]))})
    obligations = combined
    if len(obligations) > MAX_OBLIGATIONS:
        raise ProjectTaskLimitError('Obligation capacity exceeded; no requirement was omitted')
    work, by_key = [], {}
    for row in obligations:
        if row['status'] == 'ready_for_review':
            continue
        lookup_binding = {**row['binding'], 'lookup_policy': LOOKUP_POLICY_VERSION}
        key = _hash({'owner_step_id': row['owner_step_id'], 'kind': row['kind'],
                     'query': row['query'], 'binding': lookup_binding})
        if key in by_key:
            by_key[key]['obligation_ids'].append(row['id'])
            continue
        item = {'id': key, 'owner_step_id': row['owner_step_id'], 'kind': row['kind'],
                'query': row['query'], 'status': 'unresolved', 'attempts': 0, 'round': 0,
                'lookup_query': row['lookup_query'],
                'binding': lookup_binding, 'obligation_ids': [row['id']]}
        work.append(item)
        by_key[key] = item
    coverage = {state: sum(row['status'] == state for row in obligations) for state in _STATES}
    coverage.update(total=len(obligations), goal_verified=False)
    result = {'schema_version': 1, 'task_id': task['id'], 'task_revision': task['revision'],
              'fingerprint': _hash({'task_revision': task['revision'], 'goal': project_goal,
                                    'lookup_policy': LOOKUP_POLICY_VERSION,
                                    'project_criteria': project_criteria, 'phase_criteria': phase_criteria,
                                    'steps': fingerprints}),
              'obligations': obligations, 'work_items': work, 'coverage': coverage,
              'next_action': ('review' if not work else 'await_analysis' if coverage['not_assessed']
                              else 'local_lookup'),
              'semantic_verified': False, 'goal_verified': False}
    if len(_json(result).encode()) > MAX_LEDGER_BYTES:
        raise ProjectTaskLimitError('Obligation ledger byte capacity exceeded')
    return result


class ProjectObligationLedger:
    """Durable, scoped projection and idempotent local-lookup attempt journal."""
    def __init__(self, task_store):
        if not isinstance(task_store, ProjectTaskStore):
            raise ProjectTaskError('Obligations require the existing scoped task store')
        self.tasks = task_store

    @staticmethod
    def _schema(db):
        db.execute('CREATE TABLE IF NOT EXISTS project_obligation_ledgers ('
                   'task_id TEXT PRIMARY KEY,task_revision INTEGER NOT NULL,'
                   'fingerprint TEXT NOT NULL,payload TEXT NOT NULL,updated_at REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS project_obligation_attempts ('
                   'task_id TEXT NOT NULL,work_id TEXT NOT NULL,binding_sha256 TEXT NOT NULL,'
                   'outcome TEXT NOT NULL,attempts INTEGER NOT NULL,updated_at REAL NOT NULL,'
                   'PRIMARY KEY(task_id,work_id))')

    def _context(self, db, task_id):
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'long_projects', 'long_project_phases'} <= tables:
            return None
        row = db.execute('SELECT r.goal,r.acceptance_criteria AS project_criteria,'
                         'p.acceptance_criteria AS phase_criteria FROM long_project_phases p '
                         'JOIN long_projects r ON r.id=p.project_id WHERE p.task_id=? '
                         'AND r.session_id=? AND r.instance_id=?',
                         (task_id, self.tasks.session_id, self.tasks.instance_id)).fetchone()
        return ({'project_goal': row['goal'],
                 'project_acceptance_criteria': json.loads(row['project_criteria']),
                 'phase_acceptance_criteria': json.loads(row['phase_criteria'])} if row else None)

    def synchronize(self, task_id, *, expected_revision=None, task=None, eligible_step_ids=None):
        with self.tasks._db(write=True) as db:
            current = self.tasks._task(db, task_id)
            if expected_revision is not None:
                self.tasks._cas(current, expected_revision)
            if task is not None and (type(task) is not dict or task.get('id') != task_id
                                     or task.get('revision') != current['revision']):
                raise ProjectTaskConflictError('Live-checked task changed before obligation synchronization')
            # Never trust a caller's copied question/scope payload. All host
            # requirements and current artifact references come from this same
            # transaction, under the task store's owner/session/instance check.
            public = self.tasks._public(db, current)
            artifacts = {}
            for step in public['steps']:
                if step['kind'] != 'analysis' or step['status'] != 'completed' or not step['artifact_version']:
                    continue
                row = db.execute("SELECT payload FROM project_task_artifacts WHERE task_id=? AND step_id=? "
                                 "AND version=? AND generation=? AND status='current'",
                                 (task_id, step['id'], step['artifact_version'], step['generation'])).fetchone()
                if row:
                    artifacts[step['id']] = json.loads(row['payload'])
            result = build_obligation_progress(public, artifacts, execution_context=self._context(db, task_id),
                                                eligible_step_ids=eligible_step_ids)
            self._schema(db)
            for item in result['work_items']:
                saved = db.execute('SELECT * FROM project_obligation_attempts WHERE task_id=? AND work_id=?',
                                   (task_id, item['id'])).fetchone()
                if saved and saved['binding_sha256'] == _hash(item['binding']):
                    item.update(status=saved['outcome'], attempts=saved['attempts'], round=saved['attempts'])
            if result['work_items'] and all(row['status'] == 'exhausted' for row in result['work_items']):
                result['next_action'] = 'collect_evidence'
            old = db.execute('SELECT payload FROM project_obligation_ledgers WHERE task_id=?', (task_id,)).fetchone()
            encoded = _json(result)
            if old is None or old['payload'] != encoded:
                db.execute('INSERT INTO project_obligation_ledgers VALUES (?,?,?,?,?) '
                           'ON CONFLICT(task_id) DO UPDATE SET task_revision=excluded.task_revision,'
                           'fingerprint=excluded.fingerprint,payload=excluded.payload,updated_at=excluded.updated_at',
                           (task_id, current['revision'], result['fingerprint'], encoded, self.tasks.clock()))
            return result

    def read(self, task_id, *, expected_revision=None, eligible_step_ids=None):
        # A persisted badge is never an authority. Rebuild after the caller's
        # current provenance barrier; missing eligibility is fail-closed.
        return self.synchronize(task_id, expected_revision=expected_revision,
                                eligible_step_ids=eligible_step_ids)

    def record_attempt(self, task_id, work_id, *, expected_revision, outcome):
        if type(outcome) is not str or outcome not in LOOKUP_OUTCOMES:
            raise ProjectTaskError('A local lookup outcome is not evidence verification')
        if type(work_id) is not str or len(work_id) != 64:
            raise ProjectTaskError('Invalid obligation work reference')
        with self.tasks._db(write=True) as db:
            current = self.tasks._task(db, task_id)
            self.tasks._cas(current, expected_revision)
            self._schema(db)
            saved = db.execute('SELECT * FROM project_obligation_ledgers WHERE task_id=?', (task_id,)).fetchone()
            if saved is None or saved['task_revision'] != current['revision']:
                raise ProjectTaskConflictError('Obligation work is no longer current')
            result = json.loads(saved['payload'])
            item = next((row for row in result['work_items'] if row['id'] == work_id), None)
            if item is None:
                raise ProjectTaskConflictError('Obligation work is no longer current')
            previous = db.execute('SELECT * FROM project_obligation_attempts WHERE task_id=? AND work_id=?',
                                  (task_id, work_id)).fetchone()
            if previous:
                if previous['binding_sha256'] != _hash(item['binding']) or previous['outcome'] not in LOOKUP_OUTCOMES:
                    raise ProjectTaskConflictError('A lookup belongs to another input or policy')
                # Monotone observations, not proof: finding any candidate
                # wins over a miss; a delayed partial receipt cannot erase a
                # completed scan. An upgrade is still the same bound attempt.
                merged = max((previous['outcome'], outcome), key=LOOKUP_OUTCOMES.index)
                if merged != previous['outcome']:
                    db.execute('UPDATE project_obligation_attempts SET outcome=?,updated_at=? '
                        'WHERE task_id=? AND work_id=?', (merged, self.tasks.clock(), task_id, work_id))
                return dict(item, status=merged, attempts=previous['attempts'], round=previous['attempts'])
            db.execute('INSERT INTO project_obligation_attempts VALUES (?,?,?,?,?,?)',
                       (task_id, work_id, _hash(item['binding']), outcome, 1, self.tasks.clock()))
            # One bounded lookup per exact work/source/receipt binding. New
            # evidence or an explicitly revised task produces a new work ID.
            return dict(item, status=outcome, attempts=1, round=1)
