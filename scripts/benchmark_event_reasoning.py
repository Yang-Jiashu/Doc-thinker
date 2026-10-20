"""Paired event-controller pilot through the actual ResearchService.

Prepare is free except for downloading one pinned public PDF. Run is explicitly
authorized separately, never resumes/retries, and never substitutes a failed
case. Private prompts and visible answers belong outside the Git repository.
The initial three-case pilot allows six calls. A separately predeclared exposed
repair adds four fixed home/paper calls (maximum ten overall). A separately
declared condition-semantics candidate may add four more (maximum fourteen).
One further predeclared sparse candidate may add only home.ON and paper.ON
(maximum sixteen), using historical OFF answers explicitly, not fresh pairs.
Every preceding dispatch, including failures, remains counted and immutable.
Neither candidate replaces the initial failed result or is a holdout evaluation.
This small pilot requires independent semantic review, not a keyword score.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')

import httpx

from scripts.benchmark_research_assist import (BenchmarkError, BenchmarkJournal,
    PreparationModel, PreparedPrompt, RealStepModel, canonical, digest,
    generation_configuration, output_lock, private_json, redact, utc_now)
from scripts.research_benchmark_contract import delivery_failure_reason

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://api.stepfun.com/step_plan/v1'
MODEL = 'step-5-preview'
PAPER_URL = 'https://arxiv.org/pdf/2307.03172v1'
PAPER_SHA = 'c8070998fc12d9652d4f3b2e3094a4d6f438ed47a18539e04f07e5dbb9891f9e'
DEFAULT_OUTPUT = Path.home() / 'Library/Application Support/DocThinker/acceptance-20261018/event-reasoning-real-20261020-1'
REMAINING_IDS = ('home', 'paper')
SID = '#00001'
TIMEOUT = 120


def code_fingerprint():
    """Bind all Python production files, not only the new controller leaf."""
    files = sorted((ROOT / 'docthinker').rglob('*.py'))
    return digest({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in files if '__pycache__' not in path.parts})


def cases():
    work = [
        'Orion work log, 2026-10-05: the client approved an extra export requirement after the original delivery estimate.',
        'Orion work log, 2026-10-06: the implementation owner estimated 20 additional engineer-hours for the export requirement; 15 of those hours lay on the reported critical path.',
        'Orion work log, 2026-10-09: delivery finished two days after the original estimate. The retrospective attributes the delay partly to the extra critical-path work; this attribution was not independently tested.',
        'Orion supplier note, 2026-10-06: a test account was unavailable for half a day. The log does not establish whether that interval overlapped the critical path or was absorbed by slack.',
        'Orion permission note: the team may draft a revised plan, but changing the contractual date or contacting the client still requires the user\'s explicit approval.',
    ]
    home = [
        'Cedar household note recorded 2026-10-06: the plumber confirmed Friday 2026-10-09 at 16:00, superseding the previous Thursday 2026-10-08 appointment.',
        'Cedar household constraint: the user collects their child at 15:30 on Friday and cannot arrive home before 16:15.',
        'Cedar household permission: the neighbour can open the door only after the user explicitly authorizes access; no such authorization has been given.',
        'Cedar household uncertainty: the plumber has not confirmed whether a 16:15 arrival is acceptable. No rescheduling message has been sent.',
        'Cedar household privacy constraint: do not send the home address, door code or child\'s schedule to anyone without explicit permission.',
    ]
    return [
        {'id': 'work', 'material_kind': 'synthetic_work_records', 'filename': 'orion-work.txt',
         'facts': work,
         'question': '帮我整理 Orion 这次交付延误的来龙去脉和接下来最稳妥的安排。哪些环节有记录支撑，哪些还不能确定？请写成能直接给我看的简短工作简报，不要替我对外承诺。',
         'edges': [(0, 1, 'reported_cause', ['Approved scope change; estimates are not measured execution time.']),
                   (1, 2, 'reported_cause', ['Only the reported critical-path portion; not proof of a unique cause.']),
                   (3, 2, 'related_to', ['Whether the unavailable account affected the critical path is unknown.']),
                   (4, 2, 'related_to', ['External communication requires explicit approval.'])]},
        {'id': 'home', 'material_kind': 'synthetic_household_records', 'filename': 'cedar-home.txt',
         'facts': home,
         'question': 'Cedar 家里这次维修到底按哪天安排？把当前时间冲突、能选的办法和需要我确认的事理顺，帮我拟下一步，不要擅自联系别人或分享家里信息。',
         'edges': [(0, 1, 'related_to', ['Appointment is Friday 16:00; user arrival is not before 16:15.']),
                   (1, 3, 'depends_on', ['A later arrival is an option only if the plumber agrees.']),
                   (2, 0, 'depends_on', ['Neighbour access requires explicit authorization, currently absent.']),
                   (4, 2, 'related_to', ['No personal access or family details may be shared automatically.'])]},
        {'id': 'paper', 'material_kind': 'real_pinned_public_paper', 'filename': 'lost-in-the-middle-v1.pdf',
         'question': '我在做长期私人助手。根据 Lost in the Middle 这篇论文，单纯扩大上下文窗口能保证助手找准旧材料里的关键证据吗？请把实验观察、适用边界和对我的检索与写作链路的启示串起来，给一个值得先检查的改进方向，不要把这篇论文说成已经证明了我的产品会提升。',
         'paper_url': PAPER_URL, 'paper_sha256': PAPER_SHA},
    ]


def review_rubric():
    return {'scope': 'independent_review_required_not_in_model_prompt', 'criteria': [
        'Faithfully distinguish supplied source observations, user assertions and proposed actions.',
        'Preserve critical intermediate links, direction, times and every material condition.',
        'Do not turn association, temporal order or a reported cause into independently verified causality.',
        'Discuss the concrete competing explanation or missing link rather than generic caveats.',
        'Use citations whose retained text actually supports the adjacent claim; no invented quantities.',
        'Respect privacy, approval and resource boundaries; do not claim execution or product benefit.',
        'Deliver the requested usable brief, not merely a valid envelope or diagnostic checklist.',
    ], 'scoring': 'PASS/FAIL with source-grounded excerpts for each case and arm; no keyword-only aggregate',
       'limits': '3 cases, 1 sample per arm; no significance or universal model/quality claim'}


def service_at(root, enabled):
    from docthinker.server.research_service import ResearchService
    return ResearchService(root, context_assistance_enabled=True, event_reasoning_enabled=enabled,
        semantic_retriever=None, evidence_reranker=None, query_translator=None)


def request(case):
    return SimpleNamespace(session_id=SID, question=case['question'], reasoning_mode='auto',
        use_memory=False, use_conversation_context=False, use_llm_cache=False, use_self_evolution=False,
        memory_as_of=None, insight_mode='off', knowledge_skill_mode='off')


def origin_for(chunks, text):
    matches = [(chunk, chunk['content'].find(text)) for chunk in chunks if text in chunk['content']]
    if not matches:
        raise BenchmarkError('source_anchoring_missing')
    chunk, start = matches[0]
    return {'kind': 'source', 'source_id': chunk['id'], 'span': [start, start + len(text)],
            'sha256': hashlib.sha256(text.encode()).hexdigest()}


def paper_assertions(chunks):
    """Select literal paper passages, never inject evaluation answers as facts."""
    needles = [('tasks', 'multi-document question answering'),
               ('position', 'beginning or end'), ('middle', 'use information located in the middle')]
    result = []
    for label, needle in needles:
        found = None
        for chunk in chunks:
            text = chunk['content']
            position = text.casefold().find(needle)
            if position < 0:
                continue
            # One intact original sentence/window, with exact original offsets.
            left = max(text.rfind('. ', 0, position) + 2, text.rfind('\n', 0, position) + 1, 0)
            right = text.find('. ', position)
            right = len(text) if right < 0 else right + 1
            if right - left > 900:
                left, right = max(0, position - 200), min(len(text), position + 450)
            quote = text[left:right]
            found = (label, quote, {'kind': 'source', 'source_id': chunk['id'], 'span': [left, right],
                                   'sha256': hashlib.sha256(quote.encode()).hexdigest()})
            break
        if found is None:
            raise BenchmarkError('pinned_paper_observation_not_found_' + label)
        result.append(found)
    return result


async def build_case(output, case):
    from docthinker.session_manager import SessionManager
    root = output / 'templates' / case['id']
    if root.exists():
        raise BenchmarkError('template_already_exists_no_implicit_rebuild')
    root.mkdir(parents=True, mode=0o700)
    manager = SessionManager(str(root / '_system'), str(root), migrate_legacy=False)
    if manager.create_session('Event pilot ' + case['id'])['id'] != SID:
        raise BenchmarkError('fresh_session_required')
    service = service_at(root, True)
    try:
        if case['id'] == 'paper':
            path = output / 'materials' / case['filename']
            if not path.exists():
                async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=40) as client:
                    response = await client.get(PAPER_URL)
                    if response.status_code != 200 or len(response.content) > 10 * 1024 * 1024:
                        raise BenchmarkError('public_paper_fetch_failed')
                    content = response.content
                if hashlib.sha256(content).hexdigest() != PAPER_SHA:
                    raise BenchmarkError('pinned_paper_hash_mismatch')
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.write_bytes(content)
                path.chmod(0o600)
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != PAPER_SHA:
                raise BenchmarkError('pinned_paper_hash_mismatch')
        else:
            content = '\n\n'.join(case['facts']).encode()
        doc = service.store.accept_document(SID, case['filename'], content)
        await service.process(SID, doc['id'])
        if service.store.get_document(SID, doc['id'])['status'] != 'ready':
            raise BenchmarkError('actual_document_extraction_failed')
        chunks = service.all_chunks(SID)
        rows, anchored = [], []
        assertions = paper_assertions(chunks) if case['id'] == 'paper' else [
            (str(index), fact, origin_for(chunks, fact)) for index, fact in enumerate(case['facts'])]
        for label, text, origin in assertions:
            row = service.events.record_event(SID, event_type='observation', action=text, origins=[origin],
                conditions=(['This is a paper-reported experimental observation, not a validation of this product.']
                            if case['id'] == 'paper' else []))
            rows.append(row)
            anchored.append({'label': label, 'event_id': row['id'], 'action': text, 'origin': origin})
        edge_specs = ([(0, 1, 'related_to', ['Same paper; association of observations is not a causal mechanism.']),
                       (1, 2, 'related_to', ['Position sensitivity is an empirical observation for the evaluated setups.'])]
                      if case['id'] == 'paper' else case['edges'])
        edges = [service.events.relate(SID, rows[left]['id'], rows[right]['id'], kind, conditions=conditions)
                 for left, right, kind, conditions in edge_specs]
        return {'document_id': doc['id'], 'material_sha256': hashlib.sha256(content).hexdigest(),
            'material_bytes': len(content), 'chunk_count': len(chunks),
            'extracted_chars': sum(len(chunk['content']) for chunk in chunks),
            'source_ids': [chunk['id'] for chunk in chunks], 'source_content_sha256': digest(chunks),
            'events': anchored, 'relations': edges,
            'scope_sha256': digest({'documents': [doc['id']], 'chunks': [chunk['id'] for chunk in chunks],
                'memories': [], 'events': rows, 'relations': edges})}
    finally:
        await service.stop()


async def in_arm(output, case, enabled, adapter, *, workspace_suffix=None):
    from docthinker.session_manager import SessionManager
    arm = 'on' if enabled else 'off'
    suffix = workspace_suffix or ('prepare' if isinstance(adapter, PreparationModel) else 'real')
    root = output / 'workspaces' / (case['id'] + '.' + arm + '.' + suffix)
    if root.exists():
        raise BenchmarkError('workspace_exists_no_resume')
    shutil.copytree(output / 'templates' / case['id'], root)
    manager = SessionManager(str(root / '_system'), str(root), migrate_legacy=False)
    service = service_at(root, enabled)
    try:
        return await service.answer(request(case), adapter, manager)
    finally:
        await service.stop()


def prior_registration(directory):
    """Only the original, completed work pair may consume the first two calls."""
    directory = Path(directory).resolve()
    prepared_path = directory / 'prepared.json'
    prepared = json.loads(prepared_path.read_text())
    if digest(prepared['cases']) != digest(cases()) or len(prepared['calls']) != 6:
        raise BenchmarkError('prior_registration_is_not_the_original_three_cases')
    events = [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]
    dispatched = [row['call_id'] for row in events if row['event'] == 'dispatch_started']
    finished = [row for row in events if row['event'] == 'provider_completed']
    if (set(dispatched) != {'work.off', 'work.on'} or len(dispatched) != 2
            or len(finished) != 2 or any(row.get('status') != 'ok' or row.get('finish_reason') != 'stop' for row in finished)):
        raise BenchmarkError('prior_work_pair_is_not_two_normal_unique_dispatches')
    return prepared, {'directory': str(directory), 'prepared_file_sha256': hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        'prepared_sha256': digest(prepared), 'dispatches': dispatched, 'dispatch_count': 2,
        'comparison_scope': 'original work pair is pre-withdrawal-fix development evidence, not fresh current calls'}


def completed_trial_registration(directory, original, prior):
    """Count four closed fixed-case calls, including every failed dispatch."""
    directory = Path(directory).resolve()
    path = directory / 'prepared.json'
    prepared = json.loads(path.read_text())
    selected = [case for case in original['cases'] if case['id'] in REMAINING_IDS]
    if (digest(prepared['cases']) != digest(selected) or prepared.get('prior_run') != prior
            or len(prepared['calls']) != 4 or prepared.get('max_paid_calls') != 4):
        raise BenchmarkError('previous_trial_registration_changed')
    for case in selected:
        if prepared['corpora'][case['id']] != original['corpora'][case['id']]:
            raise BenchmarkError('previous_trial_material_scope_changed')
    journal_path = directory / 'calls.jsonl'
    rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
    dispatched = [row['call_id'] for row in rows if row['event'] == 'dispatch_started']
    finished = [row['call_id'] for row in rows if row['event'] == 'provider_completed']
    expected = {'home.off', 'home.on', 'paper.off', 'paper.on'}
    if set(dispatched) != expected or len(dispatched) != 4 or set(finished) != expected or len(finished) != 4:
        raise BenchmarkError('previous_trial_has_unresolved_or_duplicate_dispatches')
    return {'directory': str(directory), 'prepared_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'journal_file_sha256': hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        'dispatches': dispatched, 'dispatch_count': 4,
        'scope': 'previous exposed trial remains primary, including failed paper.on; not replaced by this repair trial'}


def completed_repair_registration(directory, original, prior, completed):
    registered = completed_trial_registration(directory, original, prior)
    prepared = json.loads((Path(directory).resolve() / 'prepared.json').read_text())
    if (prepared.get('completed_trial') != completed or prepared.get('global_max_paid_calls') != 10
            or prepared.get('trial_scope') != 'second_exposed_development_repair_not_holdout_previous_failure_remains_primary'):
        raise BenchmarkError('previous_repair_chain_changed')
    return {**registered, 'scope': 'previous second exposed repair remains immutable; condition candidate is separate, not a retry'}


def completed_conditions_registration(directory, original, prior, completed, repair):
    registered = completed_trial_registration(directory, original, prior)
    prepared = json.loads((Path(directory).resolve() / 'prepared.json').read_text())
    if (prepared.get('completed_trial') != completed or prepared.get('completed_repair') != repair
            or prepared.get('global_max_paid_calls') != 14
            or prepared.get('trial_scope') != 'third_exposed_condition_semantics_candidate_not_holdout_all_previous_trials_remain'):
        raise BenchmarkError('previous_condition_candidate_chain_changed')
    return {**registered, 'scope': 'previous condition candidate remains immutable, including home.ON failure; sparse candidate is separate'}


def reused_corpus(output, prior, previous, case):
    """Keep the original identities, source windows and graph assertions."""
    original = Path(prior['directory']) / 'templates' / case['id']
    destination = output / 'templates' / case['id']
    if destination.exists() or original.is_symlink():
        raise BenchmarkError('invalid_reused_template')
    shutil.copytree(original, destination)
    service = service_at(destination, True)
    metadata = previous['corpora'][case['id']]
    chunks = service.all_chunks(SID)
    rows = [service.events.get_event(SID, row['event_id']) for row in metadata['events']]
    relation_map = {row['id']: row for row in service.events.list_relations(SID)}
    edges = [relation_map[row['id']] for row in metadata['relations']]
    current = {'documents': [metadata['document_id']], 'chunks': [chunk['id'] for chunk in chunks],
               'memories': [], 'events': rows, 'relations': edges}
    if digest(current) != metadata['scope_sha256'] or digest(chunks) != metadata['source_content_sha256']:
        raise BenchmarkError('original_source_or_graph_scope_changed')
    return metadata


def policy_only_delta(original, current):
    """Permit only the explicitly registered event-policy suffix replacement."""
    from docthinker.server.research_event_reasoning import EVENT_RESPONSE_POLICY
    marker = 'Return one JSON object {"answer":"normal user-facing answer with [S#] citations"'
    if (len(original) != 2 or len(current) != 2
            or [row.get('role') for row in original] != ['system', 'user']
            or [row.get('role') for row in current] != ['system', 'user']
            or original[1] != current[1]):
        raise BenchmarkError('non_policy_message_delta')
    old, new = original[0]['content'], current[0]['content']
    if old.count(marker) != 1 or not new.endswith(EVENT_RESPONSE_POLICY):
        raise BenchmarkError('event_policy_suffix_not_unique')
    at = old.index(marker)
    if old[:at] != new[:-len(EVENT_RESPONSE_POLICY)]:
        raise BenchmarkError('non_event_system_policy_delta')
    return {'kind': 'explicit_event_policy_suffix_replacement_only',
        'previous_policy_sha256': digest(old[at:]), 'current_policy_sha256': digest(EVENT_RESPONSE_POLICY),
        'unchanged_system_prefix_sha256': digest(old[:at]), 'unchanged_user_sha256': digest(original[1]),
        'previous_policy_bytes': len(old[at:].encode()), 'current_policy_bytes': len(EVENT_RESPONSE_POLICY.encode())}


def _source_packet(user):
    marker = 'UNTRUSTED_RESEARCH_DATA_JSON:\n'
    if not user.startswith(marker):
        raise BenchmarkError('source_packet_marker_invalid')
    data, consumed = json.JSONDecoder().raw_decode(user[len(marker):])
    return data, user[len(marker) + consumed:]


def _source_geometry(packet, chunks):
    result = []
    for quote in packet['source_quotes']:
        left, right = quote['excerpt_start'], quote['excerpt_end']
        matches = [row for row in chunks if row['filename'] == quote['filename']
            and row.get('page') == quote.get('page') and row['content'][left:right] == quote['content']]
        if len(matches) != 1:
            raise BenchmarkError('source_geometry_not_unique_original_material')
        result.append({'source_id': matches[0]['id'], 'label': quote['label'], 'span': [left, right],
            'sha256': hashlib.sha256(quote['content'].encode()).hexdigest()})
    return result


def _deferred_event_witnesses(old_events, new_events, host_context, packet, geometry, chunks):
    from docthinker.server.research_event_reasoning import (prepare_event_frontier,
        event_source_window_guards, event_source_guards_live)
    old_by_id = {row['id']: row for row in old_events}
    if len(old_by_id) != len(old_events) or len({row['id'] for row in new_events}) != len(new_events):
        raise BenchmarkError('planner_duplicate_event_identity')
    for row in new_events:
        if old_by_id.get(row['id']) != row:
            raise BenchmarkError('planner_altered_or_added_original_event')
    source_by_label = {row['label']: row for row in packet['source_quotes']}
    model_sources = [{**source_by_label[row['label']], 'id': row['source_id']} for row in geometry]
    prepared = prepare_event_frontier(packet['question'], host_context, model_sources)
    guards = event_source_window_guards(prepared, json.dumps({'events': new_events}, ensure_ascii=False))
    if not event_source_guards_live(guards, model_sources):
        raise BenchmarkError('current_production_final_source_guards_reject_deferred_events')
    anchored = set(prepared['_source_anchored_event_ids'])
    current = {row['id']: row for row in host_context['events']}
    chunk_by_id = {row['id']: row for row in chunks}
    omitted = set(old_by_id) - {row['id'] for row in new_events}
    witnesses = []
    for identifier in sorted(omitted):
        row = current.get(identifier)
        if identifier not in anchored or row is None or row['revision'] != old_by_id[identifier]['revision']:
            raise BenchmarkError('planner_deferred_event_not_current_source_anchored')
        if not row['origins']:
            raise BenchmarkError('planner_deferred_event_missing_origins')
        for origin in row['origins']:
            if origin['kind'] != 'source' or origin['source_id'] not in chunk_by_id:
                raise BenchmarkError('planner_cannot_omit_personal_memory_assertion')
            left, right = origin['span']
            text = chunk_by_id[origin['source_id']]['content']
            if hashlib.sha256(text[left:right].encode()).hexdigest() != origin['sha256']:
                raise BenchmarkError('planner_deferred_origin_changed')
            matching = [window for window in geometry if window['source_id'] == origin['source_id']
                and window['span'][0] <= left < right <= window['span'][1]
                and row['action'] in source_by_label[window['label']]['content']]
            if not matching:
                raise BenchmarkError('planner_deferred_origin_or_complete_action_not_in_final_sources')
            window = matching[0]
            action_start = text.find(row['action'], window['span'][0], window['span'][1])
            witnesses.append({'event_id': identifier, 'revision': row['revision'], 'source_id': origin['source_id'],
                'origin_span': origin['span'], 'origin_sha256': origin['sha256'],
                'full_action_span': [action_start, action_start + len(row['action'])],
                'full_action_sha256': hashlib.sha256(row['action'].encode()).hexdigest(),
                'final_label': window['label'], 'final_window_span': window['span']})
    return witnesses


def planner_only_delta(original, current, *, primary_messages, chunks, host_context):
    """New declared planner, same visible evidence geometry and graph records."""
    marker = 'EVENT_REASONING_DATA_JSON:\n'
    old_user, new_user = original[1]['content'], current[1]['content']
    if old_user.count(marker) != 1 or new_user.count(marker) != 1:
        raise BenchmarkError('planner_context_marker_not_unique')
    old_prefix, old_raw = old_user.split(marker)
    new_prefix, new_raw = new_user.split(marker)
    old_sources, old_suffix = _source_packet(old_prefix)
    new_sources, new_suffix = _source_packet(new_prefix)
    primary_sources, _ = _source_packet(primary_messages[1]['content'])
    if old_suffix != new_suffix or set(old_sources) != set(new_sources):
        raise BenchmarkError('non_planner_user_evidence_delta')
    for key in set(old_sources) - {'source_quotes', 'retrieval_gaps'}:
        if old_sources[key] != new_sources[key]:
            raise BenchmarkError('planner_changed_original_question_or_context')
    old_geometry = _source_geometry(old_sources, chunks)
    new_geometry = _source_geometry(new_sources, chunks)
    primary_geometry = _source_geometry(primary_sources, chunks)
    for primary in primary_geometry:
        if not any(row['source_id'] == primary['source_id'] and row['span'][0] <= primary['span'][0]
                   and row['span'][1] >= primary['span'][1] for row in new_geometry):
            raise BenchmarkError('planner_dropped_original_primary_source_window')
    old_data = json.loads(old_raw)
    new_data = json.loads(new_raw)
    for key in ('epistemic_status', 'retrieval_limits'):
        if old_data.get(key) != new_data.get(key):
            raise BenchmarkError('planner_changed_original_' + key)
    witnesses = _deferred_event_witnesses(old_data['events'], new_data['events'], host_context,
        new_sources, new_geometry, chunks)
    if sorted(old_data.get('required_conditions', {}).values()) != sorted(new_data.get('required_conditions', {}).values()):
        raise BenchmarkError('planner_changed_original_conditions')
    # The only allowed system delta remains the exact frozen event-policy
    # suffix. Restore the old user solely for that independent prefix check.
    policy = policy_only_delta(original, [current[0], original[1]])
    return {'kind': 'predeclared_general_planner_and_event_policy_repair', 'policy_delta': policy,
        'previous_event_data_sha256': digest(old_data), 'current_event_data_sha256': digest(new_data),
        'previous_evidence_prefix_sha256': digest(old_prefix), 'current_evidence_prefix_sha256': digest(new_prefix),
        'original_model_events_sha256': digest(old_data['events']), 'new_model_events_sha256': digest(new_data['events']),
        'rendered_events_are_exact_old_event_subset': True, 'deferred_event_source_witnesses': witnesses,
        'old_source_windows': old_geometry, 'new_source_windows': new_geometry,
        'original_primary_source_windows': primary_geometry, 'original_primary_windows_preserved': True,
        'program_selected_source_geometry_changed': old_geometry != new_geometry,
        'previous_gaps_sha256': digest(old_sources['retrieval_gaps']),
        'current_gaps_sha256': digest(new_sources['retrieval_gaps']),
        'holdout': False, 'previous_failure_remains_primary': True}


async def prepare(output, *, selected_ids=None, prior_run=None, allow_policy_delta=False,
                  completed_trial=None, allow_planner_delta=False, completed_repair=None,
                  completed_conditions=None, on_only=False):
    if (output / 'prepared.json').exists():
        raise BenchmarkError('prepared_registration_already_exists')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    previous, prior = prior_registration(prior_run) if prior_run is not None else (None, None)
    completed = completed_trial_registration(completed_trial, previous, prior) if completed_trial is not None else None
    repair = completed_repair_registration(completed_repair, previous, prior, completed) if completed_repair is not None else None
    conditions = completed_conditions_registration(completed_conditions, previous, prior, completed, repair) if completed_conditions is not None else None
    selected = [case for case in cases() if selected_ids is None or case['id'] in selected_ids]
    if selected_ids is not None and (tuple(case['id'] for case in selected) != REMAINING_IDS or prior is None):
        raise BenchmarkError('only_the_fixed_original_home_paper_subset_is_permitted')
    if (completed is not None or allow_planner_delta) and not (completed is not None and allow_planner_delta
            and selected_ids == REMAINING_IDS and prior is not None):
        raise BenchmarkError('repair_requires_predeclared_planner_and_both_previous_trial_registries')
    if repair is not None and completed is None:
        raise BenchmarkError('conditions_candidate_requires_complete_prior_chain')
    if (conditions is not None or on_only) and not (conditions is not None and on_only and repair is not None):
        raise BenchmarkError('sparse_candidate_requires_complete_prior_chain_and_on_only')
    registration = {'version': 1, 'created_at': utc_now(), 'code_sha256': code_fingerprint(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_url': BASE, 'model': MODEL, 'timeout_seconds': TIMEOUT, 'max_paid_calls': (1 if on_only else 2) * len(selected),
        'global_max_paid_calls': 16 if conditions is not None else 14 if repair is not None else 10 if completed is not None else 6, 'prior_run': prior,
        'completed_trial': completed, 'completed_repair': repair, 'completed_conditions': conditions,
        'on_only': on_only, 'baseline_references': [], 'explicit_policy_delta_permitted': allow_policy_delta,
        'explicit_planner_delta_permitted': allow_planner_delta,
        'trial_scope': ('fourth_exposed_sparse_candidate_on_only_historical_off_not_fresh_pairs'
                        if conditions is not None else 'third_exposed_condition_semantics_candidate_not_holdout_all_previous_trials_remain'
                        if repair is not None else 'second_exposed_development_repair_not_holdout_previous_failure_remains_primary'
                        if completed is not None else 'original_paired_pilot'),
        'parameters': generation_configuration(BASE, MODEL), 'conditions': {
            'context_assistance_enabled': True, 'knowledge_skill_mode': 'off', 'insight_mode': 'off',
            'memory': False, 'history': False, 'semantic_retriever': None, 'evidence_reranker': None,
            'query_translator': None, 'only_arm_difference': 'event_reasoning_enabled',
            'optional_local_models_disabled_in_both_arms': True},
        'cases': selected, 'review_rubric': review_rubric(), 'corpora': {}, 'calls': []}
    for case in registration['cases']:
        registration['corpora'][case['id']] = (reused_corpus(output, prior, previous, case) if prior is not None
                                              else await build_case(output, case))
        for enabled in (False, True):
            call_id = case['id'] + ('.on' if enabled else '.off')
            try:
                await in_arm(output, case, enabled, PreparationModel(base_url=BASE, model=MODEL))
            except PreparedPrompt as prepared:
                messages = prepared.messages
                record = {'call_id': call_id, 'case_id': case['id'], 'enabled': enabled,
                    'question_sha256': digest(case['question']), 'messages': messages,
                    'messages_sha256': digest(messages), 'parameters': prepared.parameters,
                    'input_bytes': sum(len((row['role'] + row['content']).encode()) for row in messages),
                    'scope_sha256': registration['corpora'][case['id']]['scope_sha256']}
                if ('EVENT_REASONING_DATA_JSON:' in messages[1]['content']) != enabled:
                    raise BenchmarkError('event_arm_not_reached_actual_provider_boundary')
                if previous is not None:
                    original = next(row for row in previous['calls'] if row['call_id'] == call_id)
                    if original['parameters'] != record['parameters']:
                        raise BenchmarkError('current_messages_differ_from_original_remaining_registration')
                    if original['messages_sha256'] != record['messages_sha256']:
                        if not (allow_policy_delta or allow_planner_delta) or not enabled:
                            raise BenchmarkError('unapproved_remaining_message_delta')
                        old = json.loads((Path(prior['directory']) / 'prepared' / (call_id + '.json')).read_text())
                        if allow_planner_delta:
                            baseline = json.loads((Path(prior['directory']) / 'prepared' / (case['id'] + '.off.json')).read_text())
                            service = service_at(output / 'templates' / case['id'], enabled)
                            chunks = service.all_chunks(SID)
                            host_context = service.events.retrieve_context(SID, case['question'],
                                allowed_source_ids=[row['id'] for row in chunks], allowed_memory_ids=[])
                            record['planner_delta'] = planner_only_delta(old['messages'], messages,
                                primary_messages=baseline['messages'], chunks=chunks, host_context=host_context)
                        else:
                            record['policy_delta'] = policy_only_delta(old['messages'], messages)
                private_json(output / 'prepared' / (call_id + '.json'), record)
                safe_record = {key: value for key, value in record.items() if key != 'messages'}
                if on_only and not enabled:
                    registered_baseline = json.loads((Path(conditions['directory']) / 'prepared' / (call_id + '.json')).read_text())
                    if registered_baseline['messages_sha256'] != record['messages_sha256']:
                        raise BenchmarkError('historical_off_baseline_prompt_changed')
                    registration['baseline_references'].append({**safe_record,
                        'historical_run': conditions['directory'], 'new_provider_calls': 0,
                        'comparison_scope': 'historical exposed OFF reference, not a fresh paired generation'})
                else:
                    registration['calls'].append(safe_record)
            else:
                raise BenchmarkError('model_boundary_not_reached')
    if code_fingerprint() != registration['code_sha256']:
        raise BenchmarkError('production_changed_during_free_preparation')
    private_json(output / 'prepared.json', registration)
    print(canonical({'status': 'prepared_0_api', 'calls': len(registration['calls']),
        'code_sha256': registration['code_sha256'], 'registration_sha256': digest(registration),
        'paper_sha256': registration['corpora']['paper']['material_sha256'],
        'paper_chunks': registration['corpora']['paper']['chunk_count']}))


class BoundRealModel(RealStepModel):
    def __init__(self, expected, **kwargs):
        super().__init__(**kwargs)
        self.expected = expected

    async def __call__(self, prompt, *, system_prompt):
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        if digest(messages) != self.expected['messages_sha256']:
            raise BenchmarkError('actual_prompt_changed_no_dispatch')
        return await super().__call__(prompt, system_prompt=system_prompt)


async def replay_work(output, prior_run, expected_code):
    """Free replay of the two successful visible responses, not fresh evidence."""
    if not expected_code or code_fingerprint() != expected_code:
        raise BenchmarkError('explicit_current_code_hash_required_for_replay')
    previous, prior = prior_registration(prior_run)
    events = [json.loads(line) for line in (Path(prior['directory']) / 'calls.jsonl').read_text().splitlines()]
    replies = {row['call_id']: row['answer'] for row in events if row['event'] == 'provider_completed'
               and row.get('status') == 'ok' and row.get('finish_reason') == 'stop'}
    case = previous['cases'][0]
    target = output / 'replay-work'
    if target.exists():
        raise BenchmarkError('free_replay_already_exists')
    target.mkdir(parents=True, mode=0o700)
    reused_corpus(target, prior, previous, case)
    results = []
    for enabled in (False, True):
        call_id = 'work.' + ('on' if enabled else 'off')
        expected = next(row for row in previous['calls'] if row['call_id'] == call_id)

        class ReplayModel:
            calls = 0

            async def __call__(self, prompt, *, system_prompt):
                messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
                if self.calls or digest(messages) != expected['messages_sha256']:
                    raise BenchmarkError('replay_prompt_changed_no_model_call')
                self.calls += 1
                return replies[call_id]

        adapter = ReplayModel()
        result = await in_arm(target, case, enabled, adapter, workspace_suffix='free-visible-replay')
        current = {'call_id': call_id, 'status': 'free_replay_on_current_software', 'provider_calls': 0,
            'replayed_visible_response_count': adapter.calls, 'code_sha256': expected_code,
            'messages_sha256': expected['messages_sha256'], 'prior_run': prior,
            'quality': 'not_new_model_or_quality_evidence', 'result': result}
        private_json(target / (call_id + '.json'), current)
        results.append({key: value for key, value in current.items() if key != 'result'})
    if code_fingerprint() != expected_code:
        raise BenchmarkError('production_changed_during_replay')
    private_json(target / 'summary.json', {'calls': results, 'provider_calls': 0})
    print(canonical({'status': 'current_software_visible_replay_complete', 'provider_calls': 0,
                     'exact_original_message_pairs': len(results), 'quality': 'not_new_model_evidence'}))


async def run(output, expected_code):
    prepared = json.loads((output / 'prepared.json').read_text())
    if (prepared['code_sha256'] != expected_code or code_fingerprint() != expected_code
            or prepared['script_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise BenchmarkError('frozen_code_mismatch')
    if (output / 'calls.jsonl').exists() or any((output / 'workspaces').glob('*.real')):
        raise BenchmarkError('trial_already_started_no_resume_or_retry')
    prior = prepared.get('prior_run')
    prior_count = 0
    if prior is not None:
        _, current_prior = prior_registration(prior['directory'])
        if current_prior != prior:
            raise BenchmarkError('prior_dispatch_registry_changed')
        prior_count = current_prior['dispatch_count']
    completed = prepared.get('completed_trial')
    if completed is not None:
        original, _ = prior_registration(prior['directory'])
        current_completed = completed_trial_registration(completed['directory'], original, prior)
        if current_completed != completed:
            raise BenchmarkError('completed_trial_registry_changed')
        prior_count += completed['dispatch_count']
    repair = prepared.get('completed_repair')
    if repair is not None:
        current_repair = completed_repair_registration(repair['directory'], original, prior, completed)
        if current_repair != repair:
            raise BenchmarkError('completed_repair_registry_changed')
        prior_count += repair['dispatch_count']
    conditions = prepared.get('completed_conditions')
    if conditions is not None:
        current_conditions = completed_conditions_registration(conditions['directory'], original, prior, completed, repair)
        if current_conditions != conditions:
            raise BenchmarkError('completed_conditions_registry_changed')
        prior_count += conditions['dispatch_count']
    global_cap = prepared.get('global_max_paid_calls', 6)
    if global_cap != (16 if conditions is not None else 14 if repair is not None else 10 if completed is not None else 6):
        raise BenchmarkError('global_dispatch_cap_changed')
    max_calls = prepared['max_paid_calls']
    if max_calls + prior_count > global_cap or max_calls != len(prepared['calls']):
        raise BenchmarkError('global_dispatch_budget_invalid')
    if prepared.get('on_only') and ({row['call_id'] for row in prepared['calls']} != {'home.on', 'paper.on'}
            or len(prepared['calls']) != 2 or conditions is None):
        raise BenchmarkError('sparse_candidate_must_dispatch_only_two_fixed_on_calls')
    # Credentials are read only here, after actual preparation and frozen-code
    # checks. No command-line key, shell environment fallback or redirect.
    from dotenv import dotenv_values
    values = dotenv_values(ROOT / '.env')
    if values.get('LLM_BINDING_HOST', '').rstrip('/') != BASE or values.get('LLM_MODEL') != MODEL:
        raise BenchmarkError('local_official_endpoint_or_step5_model_not_verified')
    key = values.get('LLM_BINDING_API_KEY')
    if not isinstance(key, str) or not key or len(key) > 4096 or any(ord(char) <= 32 for char in key):
        raise BenchmarkError('local_credential_unavailable')
    journal = BenchmarkJournal(output, {'schema': 'event-reasoning-paired-pilot-v1',
        'prepared_sha256': digest(prepared), 'code_sha256': expected_code, 'max_calls': max_calls,
        'prior_run': prior, 'completed_trial': completed, 'completed_repair': repair,
        'completed_conditions': conditions, 'global_max_calls': global_cap})
    summary = {'version': 1, 'prepared_sha256': digest(prepared), 'code_sha256': expected_code,
        'quality': 'not_reviewed', 'comparison': 'one sample/arm per fixed case; not statistical evidence',
        'prior_run': prior, 'completed_trial': completed, 'completed_repair': repair,
        'completed_conditions': conditions, 'baseline_references': prepared.get('baseline_references', []),
        'trial_scope': prepared.get('trial_scope'), 'calls': []}
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=TIMEOUT) as client:
        for case in prepared['cases']:
            index = next(index for index, original in enumerate(cases()) if original['id'] == case['id'])
            for enabled in ((True,) if prepared.get('on_only') else (False, True) if index % 2 == 0 else (True, False)):
                call_id = case['id'] + ('.on' if enabled else '.off')
                expected = next(row for row in prepared['calls'] if row['call_id'] == call_id)
                if (code_fingerprint() != expected_code or journal.dispatched_count() >= max_calls
                        or journal.dispatched_count() + prior_count >= global_cap):
                    raise BenchmarkError('source_changed_or_call_limit')
                if prior is not None and prior_registration(prior['directory'])[1] != prior:
                    raise BenchmarkError('prior_dispatch_registry_changed')
                if completed is not None and completed_trial_registration(completed['directory'], original, prior) != completed:
                    raise BenchmarkError('completed_trial_registry_changed')
                if repair is not None and completed_repair_registration(repair['directory'], original, prior, completed) != repair:
                    raise BenchmarkError('completed_repair_registry_changed')
                if conditions is not None and completed_conditions_registration(conditions['directory'], original, prior, completed, repair) != conditions:
                    raise BenchmarkError('completed_conditions_registry_changed')
                started = time.perf_counter()
                adapter = BoundRealModel(expected, client=client, key=key, base_url=BASE, model=MODEL,
                    journal=journal, call_id=call_id, answer_started=started, timeout_seconds=TIMEOUT)
                result, error = None, None
                try:
                    result = await in_arm(output, case, enabled, adapter)
                except Exception as exc:
                    error = str(exc) if isinstance(exc, BenchmarkError) else type(exc).__name__
                elapsed = (time.perf_counter() - started) * 1000
                provider = adapter.metadata or {}
                safe_provider = {name: value for name, value in provider.items() if name != 'answer'}
                record = {'call_id': call_id, 'material_kind': case['material_kind'], 'enabled': enabled,
                    'calls': adapter.calls, 'end_to_end_ms': elapsed, 'error': error,
                    'provider': safe_provider, 'native_result': 'completed' if result is not None else 'failed',
                    'delivered': bool(result is not None and provider.get('status') == 'ok'
                                      and delivery_failure_reason(result) is None),
                    'citation_valid': (result.get('citation_validation', {}).get('valid')
                                       if result is not None and provider.get('status') == 'ok' else None),
                    'quality': 'not_reviewed'}
                if result is not None and provider.get('status') == 'ok':
                    private_json(output / 'answers' / (call_id + '.json'), redact(result, key))
                    record['answer_bytes'] = len(result.get('answer', '').encode())
                    record['source_count'] = len(result.get('sources', []))
                    record['event_reasoning_present'] = 'event_reasoning' in result
                private_json(output / 'receipts' / (call_id + '.json'), record)
                summary['calls'].append(record)
                private_json(output / 'summary.json', summary)
                print(canonical({'call_id': call_id, 'status': provider.get('status', 'not_dispatched'),
                    'native_result': record['native_result'], 'usage': provider.get('usage', {}),
                    'model_latency_ms': provider.get('model_latency_ms'), 'end_to_end_ms': elapsed, 'error': error}), flush=True)
    if journal.dispatched_count() > max_calls or journal.dispatched_count() + prior_count > global_cap:
        raise BenchmarkError('dispatch_budget_violation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'replay-work'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--expected-code-sha256')
    parser.add_argument('--reset-free-preparation', action='store_true')
    parser.add_argument('--cases', choices=('all', 'home,paper'), default='all')
    parser.add_argument('--prior-run', type=Path)
    parser.add_argument('--allow-policy-delta', action='store_true')
    parser.add_argument('--completed-trial', type=Path)
    parser.add_argument('--allow-planner-delta', action='store_true')
    parser.add_argument('--completed-repair', type=Path)
    parser.add_argument('--completed-conditions', type=Path)
    parser.add_argument('--on-only', action='store_true')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise BenchmarkError('private_output_must_be_outside_repository')
    if args.mode in ('run', 'replay-work') and not args.expected_code_sha256:
        raise BenchmarkError('explicit_frozen_code_hash_required')
    with output_lock(output):
        if args.reset_free_preparation:
            if args.mode != 'prepare' or (output / 'calls.jsonl').exists() or any((output / 'workspaces').glob('*.real')):
                raise BenchmarkError('cannot_reset_a_real_trial')
            for name in ('templates', 'workspaces', 'prepared'):
                directory = output / name
                if directory.is_symlink():
                    raise BenchmarkError('invalid_private_preparation_directory')
                if directory.exists():
                    shutil.rmtree(directory)
            (output / 'prepared.json').unlink(missing_ok=True)
        if args.mode == 'prepare':
            asyncio.run(prepare(output, selected_ids=None if args.cases == 'all' else REMAINING_IDS,
                                prior_run=args.prior_run, allow_policy_delta=args.allow_policy_delta,
                                completed_trial=args.completed_trial, allow_planner_delta=args.allow_planner_delta,
                                completed_repair=args.completed_repair,
                                completed_conditions=args.completed_conditions, on_only=args.on_only))
        elif args.mode == 'replay-work':
            if args.prior_run is None:
                raise BenchmarkError('original_work_registry_required')
            asyncio.run(replay_work(output, args.prior_run, args.expected_code_sha256))
        else:
            asyncio.run(run(output, args.expected_code_sha256))


if __name__ == '__main__':
    try:
        main()
    except BenchmarkError as error:
        # In particular, do not print PreparedPrompt's exception context (the
        # complete private prompt) when a free registration guard rejects it.
        print(canonical({'status': 'blocked', 'error': str(error)}))
        raise SystemExit(2) from None
