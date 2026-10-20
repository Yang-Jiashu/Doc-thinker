"""Goal-directed work selection over the existing Harness and task DAG.

This controller creates no grant, provider call or external action. Current
evidence gaps are input to an already approved successor; an additional pass
uses the existing version-bound replan preview and explicit run approval.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import re


def retrieval_worklist(event_obligations=None, *, followup_queries=(), requirement_queries=()):
    """Fairly select whole queries without permanently starving global goals.

    These strings are retrieval hints, never shortened substitute requirements.
    The original host requirements remain intact in the prompt and ledger.
    """
    pools = {}
    for kind, values in (
            ('parent_gap', followup_queries),
            ('event_gap', (event_obligations or {}).get('suggested_queries', [])),
            ('requirement', requirement_queries)):
        pools[kind] = deque(dict.fromkeys(value for value in values
            if type(value) is str and value.strip() and len(value.encode('utf-8')) <= 400))
    total = {kind: len(values) for kind, values in pools.items()}
    selected, counts, seen = [], dict.fromkeys(pools, 0), set()
    while len(selected) < 8 and any(pools.values()):
        for kind, pool in pools.items():
            while pool and pool[0] in seen:
                pool.popleft()
            if pool and len(selected) < 8:
                value = pool.popleft()
                selected.append(value); seen.add(value); counts[kind] += 1
    return ({**(event_obligations or {}), 'suggested_queries': selected}, {
        'algorithm': 'goal_evidence_worklist_v1', 'scheduled_queries': len(selected),
        'scheduled_by_kind': counts, 'candidate_queries_by_kind': total,
        'deferred_queries': sum(len(pool) for pool in pools.values()),
        'lookup_only': True, 'automatic_model_retry': False})


def repair_plan(task, progress):
    """Construct a real CAS-replan input, retaining all original scope/goals.

    Revisit the smallest set of original analyses that owns unresolved work.
    Existing dependency invalidation recalculates its affected descendants;
    independent completed branches remain untouched. Nothing is applied here.
    """
    if (type(progress) is not dict or progress.get('task_id') != task.get('id')
            or progress.get('task_revision') != task.get('revision')):
        raise ValueError('Evidence worklist belongs to a different task version')
    questions = []
    for step in task['steps']:
        if step['kind'] != 'analysis':
            continue
        row = {key: step[key] for key in ('id', 'title', 'question')}
        row['depends_on'] = [key for key in step['depends_on'] if key != 'inventory']
        if step.get('purpose') == 'verify':
            row.update(purpose='verify', review_of=step['review_of'])
        if step.get('refine_of') is not None:
            row['refine_of'] = step['refine_of']
        if step.get('input_kind') == 'personal':
            row.update(input_kind='personal', memory_ids=list(step.get('memory_ids', [])))
        elif step.get('document_ids') is not None:
            row['document_ids'] = list(step['document_ids'])
        questions.append(row)
    by_step = {row['id']: [] for row in questions}
    needs_refine = set()
    obligation_by_id = {row['id']: row for row in progress.get('obligations', [])}
    skipped = 0
    for work in progress.get('work_items', []):
        if work.get('status') == 'exhausted':
            # The same evidence/version has already been searched. A new
            # document or corrected input produces a fresh bound work item;
            # simply clicking repair must not purchase the same empty pass.
            skipped += 1
            continue
        owner, query = work.get('owner_step_id'), work.get('lookup_query', work.get('query'))
        if owner in by_step and type(query) is str and query.strip() and query not in by_step[owner]:
            # The ledger orders contradictions and unresolved bottlenecks
            # before generic delivery checks. Never use a query as an action.
            by_step[owner].append(query)
        if owner in by_step and (work.get('status') == 'candidates_found' or any(
                obligation_by_id.get(identifier, {}).get('status') in {'unresolved', 'contradicted'}
                for identifier in work.get('obligation_ids', []))):
            needs_refine.add(owner)
    changed, deferred, capacity = [], skipped, []
    states = {row['id']: row for row in task['steps'] if row['kind'] == 'analysis'}
    # Explicit review nodes are part of the visible plan/grant. Do not turn a
    # writer into a combined writer-and-proof task by appending every gap to
    # its question. Full hints remain on the work ledger and parent packet.
    for row in list(questions):
        available = by_step[row['id']]
        if not available:
            continue
        owner = states[row['id']]
        if owner.get('status', 'completed') != 'completed':
            deferred += len(available)
            continue
        existing = [item for item in questions if item.get('purpose') == 'verify'
                    and item.get('review_of') == row['id']]
        if existing:
            # A prepared or current audit already has the whole parent gap
            # packet. Its own remaining work will target that newer lineage.
            deferred += len(available)
            continue
        refine = row['id'] in needs_refine
        if (len(questions) + (2 if refine else 1) > 12 or
                sum(item.get('purpose') == 'verify' for item in questions) >= 6 or
                (refine and sum(item.get('purpose', 'compose') == 'compose' for item in questions) >= 6)):
            capacity.append(row['id']); deferred += len(available)
            continue
        identity = hashlib.sha256(json.dumps([task['id'], task['revision'], row['id']],
            separators=(',', ':')).encode()).hexdigest()[:20]
        target = row
        if refine:
            target = {**row, 'id': 'refine_' + identity, 'title': '补查更新 · ' + row['title'][:110],
                      'refine_of': row['id'], 'depends_on': [row['id']]}
            target.pop('purpose', None); target.pop('review_of', None)
            questions.append(target); changed.append(target['id'])
        audit = {**target, 'id': 'verify_' + identity, 'title': '核对证据 · ' + row['title'][:110],
                 'purpose': 'verify', 'review_of': target['id'], 'depends_on': [target['id']]}
        audit.pop('refine_of', None)
        questions.append(audit)
        changed.append(audit['id'])
    return {'plan': {'goal': task['goal'], 'constraints': task['constraints'], 'questions': questions},
            'changed_steps': changed, 'deferred_work_count': deferred, 'capacity_steps': capacity,
            'execution_authorized': False, 'automatic_execution': False,
            'next_action': 'preview_replan' if changed else 'collect_evidence' if progress.get('work_items') else 'review'}


def causal_delivery_state(receipt):
    """Admission status for a candidate chain, never a truth certificate."""
    obligations = receipt.get('obligations', []) if type(receipt) is dict else []
    paths = [row for row in obligations if row.get('kind') == 'path']
    causal = [row for row in obligations if row.get('causal_candidate')]
    if not paths and not causal:
        return None
    contradicted = any(
        (row.get('relation') == 'contradicts' and row.get('status') == 'model_assessed_support')
        or (row.get('relation') != 'contradicts'
            and (row.get('causal_candidate') or row.get('prerequisite_candidate')
                 or row.get('kind') in {'path', 'alternatives', 'event'} or row.get('relation') is None)
            and row.get('status') == 'model_assessed_counterevidence') for row in obligations)
    ready = bool(paths) and not contradicted and all(
        row.get('status') == 'model_assessed_support' and not row.get('gaps')
        for row in obligations if row.get('causal_candidate') or row.get('kind') in {'path', 'alternatives'})
    return {'status': 'counterevidence' if contradicted else 'ready_for_review' if ready else 'needs_evidence',
            'causality_verified': False, 'semantic_verified': False,
            'definitive_causal_conclusion_allowed': False,
            'unresolved_ids': [row['id'] for row in obligations if row.get('status') != 'model_assessed_support'],
            'policy': 'candidate_explanation_only' if ready else 'no_definitive_causal_conclusion'}


def record_parent_lookups(bridge, trace):
    """Consume actual local lookup receipts against current parent work only.

    This is best-effort diagnostic bookkeeping after a completed answer, not
    permission to retry it or a proof upgrade. Every replay remains idempotent;
    a changed parent/source/receipt cannot inherit a previous lookup outcome.
    """
    from .project_evidence_feedback import feedback_view
    from .project_obligation_ledger import ProjectObligationLedger, LOOKUP_POLICY_VERSION
    from .project_task_store import ProjectTaskStore
    from .research_task_review import validate_task_followup
    try:
        if type(trace) is not dict:
            return {'status': 'not_recorded', 'recorded_count': 0}
        if trace.get('lookup_policy') != LOOKUP_POLICY_VERSION:
            return {'status': 'lookup_policy_unavailable', 'recorded_count': 0}
        attempted = trace.get('attempted_query_sha256')
        candidates = trace.get('candidate_query_sha256')
        corpus_attempted = trace.get('corpus_attempted_query_sha256', [])
        corpus_limited = trace.get('corpus_scan_limited', True)
        scope_packet = trace.get('lookup_scope')
        lookup_scope = None
        if scope_packet is not None:
            if type(scope_packet) is not dict:
                return {'status': 'invalid_lookup_receipt', 'recorded_count': 0}
            if scope_packet.get('input_kind') == 'documents':
                lookup_scope = scope_packet.get('document_snapshot')
                if (set(scope_packet) != {'input_kind', 'document_snapshot'} or type(lookup_scope) is not dict
                        or len(lookup_scope) > 100 or any(type(key) is not str
                        or not re.fullmatch('[A-Za-z0-9_-]{1,128}', key) or type(digest) is not str
                        or not re.fullmatch('[0-9a-f]{64}', digest) for key, digest in lookup_scope.items())):
                    return {'status': 'invalid_lookup_receipt', 'recorded_count': 0}
            elif (scope_packet.get('input_kind') != 'personal' or
                    set(scope_packet) != {'input_kind', 'context_refs'} or
                    type(scope_packet['context_refs']) is not dict or len(scope_packet['context_refs']) > 8 or
                    any(type(key) is not str or not re.fullmatch('[0-9a-f-]{36}', key)
                        or type(revision) is not int or revision < 1 for key, revision in scope_packet['context_refs'].items())):
                return {'status': 'invalid_lookup_receipt', 'recorded_count': 0}
        dependencies = trace.get('dependencies')
        if (type(attempted) is not list or type(candidates) is not list
                or type(corpus_attempted) is not list or type(corpus_limited) is not bool
                or len(attempted) > 48 or len(candidates) > 48 or len(corpus_attempted) > 48
                or any(type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value)
                       for value in [*attempted, *candidates, *corpus_attempted])
                or len(set(attempted)) != len(attempted) or len(set(candidates)) != len(candidates)
                or len(set(corpus_attempted)) != len(corpus_attempted)
                or not (set(candidates) | set(corpus_attempted)) <= set(attempted)
                or type(dependencies) is not list or len(dependencies) > 12):
            return {'status': 'invalid_lookup_receipt', 'recorded_count': 0}
        if not attempted or not dependencies:
            return {'status': 'no_lookup_attempt', 'recorded_count': 0}
        keys = {'instance_id', 'task_id', 'step_id', 'artifact_version', 'answer_id',
                'review_kind', 'receipt_sha256'}
        if any(type(row) is not dict or set(row) != keys for row in dependencies):
            return {'status': 'invalid_lookup_receipt', 'recorded_count': 0}
        validate_task_followup({'version': 1, 'dependencies': [{**row,
            'priority': 'missing_evidence', 'queries': [], 'gaps': [],
            'omitted_query_count': 0, 'omitted_gap_count': 0} for row in dependencies]})
        instance = getattr(bridge, 'instance_id', None)
        session = bridge.manager.get_session(bridge.session_id)
        if (not session or session.get('instance_id') != instance
                or any(row['instance_id'] != instance for row in dependencies)):
            return {'status': 'parent_scope_unavailable', 'recorded_count': 0}
        store = ProjectTaskStore(Path(bridge.manager.base_storage_path),
                                 session_id=bridge.session_id, instance_id=instance)
        ledger = ProjectObligationLedger(store)
        groups = {}
        for dependency in dependencies:
            groups.setdefault(dependency['task_id'], []).append(dependency)
        recorded = 0
        for task_id, group in groups.items():
            try:
                task = store.get_task(task_id)
                packets, payloads = [], {}
                for step in task['steps']:
                    if step['kind'] != 'analysis' or step['status'] != 'completed':
                        continue
                    saved = store.get_artifact(task_id, step['id'], version=step['artifact_version'])
                    if not saved.get('available') or saved.get('status') != 'current':
                        raise ValueError('Current parent artifact is unavailable')
                    payload = saved['references']
                    packets.append({**step, 'answer_id': payload['answer_id']})
                    payloads[step['id']] = payload
                allowed = []
                for dependency in group:
                    step = next((row for row in packets if row['id'] == dependency['step_id']), None)
                    if (step is None or step['artifact_version'] != dependency['artifact_version']
                            or step['answer_id'] != dependency['answer_id']):
                        raise ValueError('Parent lookup belongs to a different artifact')
                    payload = payloads[step['id']]
                    kind = dependency['review_kind']
                    receipt_key, binding_key = (('task_review', 'task_review_binding') if kind == 'task'
                                                 else ('event_reasoning', 'evidence_feedback_binding'))
                    review = feedback_view(payload[receipt_key], payload[binding_key], task_id=task_id,
                        step_id=step['id'], artifact_version=step['artifact_version'],
                        current_task_revision=task['revision'], review_kind=kind)
                    if review['binding']['receipt_sha256'] != dependency['receipt_sha256']:
                        raise ValueError('Parent lookup receipt hash changed')
                    allowed.append(dependency)
                # One batch barrier validates every completed parent answer,
                # including its source, event and memory lineage. No awaits or
                # external actions occur between this check and scoped writes.
                if not packets or bridge.validate_inputs(packets) is not True:
                    continue
                if bridge.phase_guard is not None and bridge.phase_guard(task_id) is not True:
                    continue
                progress = ledger.synchronize(task_id, expected_revision=task['revision'], task=task,
                    eligible_step_ids=[row['id'] for row in packets])
                for work in progress['work_items']:
                    binding = work['binding']
                    kind = 'task' if work['kind'] == 'task_requirement' else 'event'
                    matches = any(row['step_id'] == work['owner_step_id']
                        and row['artifact_version'] == binding['artifact_version']
                        and row['answer_id'] == binding['answer_id'] and row['review_kind'] == kind
                        and row['receipt_sha256'] == binding['receipt_sha256'] for row in allowed)
                    query_hash = hashlib.sha256(work['lookup_query'].encode('utf-8')).hexdigest()
                    if not matches or query_hash not in attempted:
                        continue
                    owner = next(row for row in packets if row['id'] == work['owner_step_id'])
                    full_scope = (lookup_scope is not None and bool(owner.get('input_snapshot'))
                        and all(lookup_scope.get(key) == value for key, value in owner['input_snapshot'].items()))
                    outcome = ('candidates_found' if query_hash in candidates else 'exhausted'
                        if not corpus_limited and query_hash in corpus_attempted and full_scope else 'partially_searched')
                    ledger.record_attempt(task_id, work['id'], expected_revision=task['revision'],
                        outcome=outcome)
                    recorded += 1
            except (ValueError, TypeError, KeyError, OSError):
                # An obsolete/malformed individual parent cannot authorize new
                # work or make the already-paid child answer fail.
                continue
        return {'status': 'recorded' if recorded else 'no_current_matching_work', 'recorded_count': recorded}
    except Exception:
        # Storage unavailable, races and legacy traces are diagnostic gaps,
        # never automatic paid retries, successful proof or leaked exceptions.
        return {'status': 'not_recorded', 'recorded_count': 0}


def prioritize_parent_queries(bridge, dependencies):
    """Prefer untried hints from the same immutable parent work binding.

    This read-only cache affects scheduling order only. It cannot supply facts,
    alter scope, hide obligations, authorize execution, or upgrade proof. All
    complete queries remain in the originating receipt and this returned list.
    """
    from .project_task_store import ProjectTaskStore
    from .project_obligation_ledger import LOOKUP_POLICY_VERSION, LOOKUP_OUTCOMES
    try:
        store = ProjectTaskStore(Path(bridge.manager.base_storage_path),
            session_id=bridge.session_id, instance_id=bridge.instance_id)
        result = [{**row, 'queries': list(row['queries'])} for row in dependencies]
        with store._db() as db:
            if db is None or not {'project_obligation_ledgers', 'project_obligation_attempts'} <= {
                    row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
                return result
            for row in result:
                store._task(db, row['task_id'])  # same owner/session/instance
                saved = db.execute('SELECT payload FROM project_obligation_ledgers WHERE task_id=?',
                                   (row['task_id'],)).fetchone()
                if saved is None or len(saved['payload'].encode()) > 2 * 1024 * 1024:
                    continue
                work = json.loads(saved['payload']).get('work_items', [])
                attempted = set()
                for item in work[:600]:
                    binding = item.get('binding', {})
                    kind = 'task' if item.get('kind') == 'task_requirement' else 'event'
                    if (item.get('owner_step_id') != row['step_id'] or kind != row['review_kind']
                            or binding.get('artifact_version') != row['artifact_version']
                            or binding.get('answer_id') != row['answer_id']
                            or binding.get('lookup_policy') != LOOKUP_POLICY_VERSION
                            or binding.get('receipt_sha256') != row['receipt_sha256']):
                        continue
                    outcome = db.execute('SELECT binding_sha256,outcome FROM project_obligation_attempts '
                        'WHERE task_id=? AND work_id=?', (row['task_id'], item['id'])).fetchone()
                    digest = hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                    if (outcome and outcome['binding_sha256'] == digest
                            and outcome['outcome'] in LOOKUP_OUTCOMES):
                        attempted.add(item.get('lookup_query'))
                row['queries'].sort(key=lambda query: query in attempted)
        return result
    except Exception:
        return dependencies
