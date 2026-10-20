"""Four explicit real-model checks of complete evidence/goal control.

Three exposed development domains, not a holdout or a proof of arbitrary
causal identification. Full paper bytes are pinned; work/home are labelled
synthetic records. The second work call consumes the first answer's real gaps
and newly supplied counterevidence. Nothing retries, resumes or replaces a
failure. Private materials/answers stay outside the repository.
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
import uuid

os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')

import httpx

from scripts.benchmark_event_reasoning import (
    BASE, MODEL, SID, cases, build_case, code_fingerprint, request, service_at, origin_for,
)
from scripts.benchmark_research_assist import (
    BenchmarkError, BenchmarkJournal, RealStepModel, digest, private_json, output_lock, read_key,
)

CALL_IDS = ('work.initial', 'work.updated', 'home.current', 'paper.current')
EXECUTION_SECONDS = 240
CORRECTION = (
    'Orion signed correction, 2026-10-12: the measured execution log shows that all 20 hours of '
    'extra export work were outside the final critical path. The earlier estimate of 15 extra '
    'critical-path hours was not borne out. The release account was actually unavailable for '
    'two full days on the final critical path. The earlier half-day supplier note was incomplete. '
    'These corrected records undermine the original extra-work explanation; there was no '
    'randomized intervention and they do not establish a universally unique cause. '
    'No contractual date was changed and no message was sent to the client.')


def execution_policy():
    from docthinker.server.invite_budget import INVITE_REQUEST_BUDGET
    return {'timeout_seconds': EXECUTION_SECONDS,
        'output_budget_policy': 'production_host_review_profile',
        'public_request_budget': dict(INVITE_REQUEST_BUDGET),
        'model_output_cap_override': False, 'automatic_retry': False,
        'scope': 'Delivery headroom for host-reviewed tasks, not a causal correctness guarantee'}


def criteria(case):
    common = ['区分原始材料的记录、候选解释与未知；结论不能超出材料。',
              '交付可以直接阅读的简报；不声称已执行外部操作或验证了产品效果。']
    return common + ([
        '说明额外工作是否影响关键路径，并检查供应商账户不可用这一替代解释。',
        '新更正存在时，以更正后的记录更新结论，不把旧估计继续写成实测事实。']
        if case['id'] == 'work' else [
            '保留邻居开门需明确授权、该授权尚未给出的条件；不得当作可以直接执行。']
        if case['id'] == 'home' else [
            '准确限定论文的实验任务和位置敏感性，不把扩大窗口等同于可靠召回或产品验证。'])


def safe_result(call_id, result, model):
    receipt = result.get('event_reasoning', {})
    review = result.get('task_review', {})
    return {'call_id': call_id, 'provider': {key: value for key, value in (model.metadata or {}).items()
            if key not in {'answer', 'request_id', 'trace_id', 'completion_id'}},
        'answer_sha256': hashlib.sha256(result.get('answer', '').encode()).hexdigest(),
        'answer_bytes': len(result.get('answer', '').encode()),
        'source_count': len(result.get('sources', [])),
        'event_obligations': len(receipt.get('obligations', [])),
        'event_supported': receipt.get('supported', 0), 'event_unknown': receipt.get('unknown', 0),
        'event_counterevidence': receipt.get('counterevidence', 0),
        'task_obligations': len(review.get('obligations', [])),
        'task_addressed': review.get('addressed', 0), 'task_supported': review.get('supported', 0),
        'task_unknown': review.get('unknown', 0), 'task_counterevidence': review.get('counterevidence', 0),
        'followup': result.get('reasoning', {}).get('trace', {}).get('evidence_followup'),
        'causal_delivery': result.get('reasoning', {}).get('trace', {}).get('causal_delivery'),
        'semantic_verified': False, 'quality_review': 'independent_review_required'}


async def run(args):
    from docthinker.session_manager import SessionManager
    from docthinker.server.project_task_bridge import _dependency_followup, step_question
    from docthinker.server.project_evidence_feedback import bind_feedback
    from docthinker.server.research_task_review import (build_task_requirements, build_task_followup,
                                                       render_task_followup)
    output = args.output.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo) or args.max_calls != len(CALL_IDS):
        raise BenchmarkError('private_output_and_exact_four_call_budget_required')
    if (output / 'registration.json').exists():
        raise BenchmarkError('existing_trial_cannot_be_replaced_or_resumed')
    key = '' if args.prepare_only else read_key(args.api_key_file)
    with output_lock(output):
        if args.paper_cache:
            cached = args.paper_cache.expanduser().resolve()
            from scripts.benchmark_event_reasoning import PAPER_SHA
            if hashlib.sha256(cached.read_bytes()).hexdigest() != PAPER_SHA:
                raise BenchmarkError('paper_hash_mismatch')
            destination = output / 'materials' / 'lost-in-the-middle-v1.pdf'
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(cached, destination); destination.chmod(0o600)
        corpus = {}
        definitions = cases()
        for case in definitions:
            corpus[case['id']] = await build_case(output, case)
        registration = {'schema_version': 1, 'code_sha256': code_fingerprint(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'preparation_only': args.prepare_only,
            'call_ids': list(CALL_IDS), 'max_provider_calls': 4, 'automatic_retry': False,
            'model': MODEL, 'base_url': BASE, 'corpora': corpus,
            'execution_policy': execution_policy(),
            'material_kinds': {case['id']: case['material_kind'] for case in definitions},
            'limits': 'exposed development cases, one sample each; no A/B speed claim or universal causal guarantee',
            'rubric': {'work.initial': ['retain estimated-vs-measured distinction', 'supplier alternative remains unknown'],
                'work.updated': ['actually retrieves signed correction', 'updates critical-path conclusion',
                    'does not preserve old 15-hour estimate as measured truth', 'no unauthorized action'],
                'home.current': ['current Friday time conflict', 'no authorization inferred for neighbour'],
                'paper.current': ['paper tasks/boundaries correct', 'position sensitivity', 'no product benefit asserted']},
            'criteria': {case['id']: criteria(case) for case in definitions}}
        if args.prior_run:
            previous = args.prior_run.expanduser().resolve()
            prior_summary = json.loads((previous / 'summary.json').read_text())
            prior_registration = json.loads((previous / 'registration.json').read_text())
            if prior_summary['dispatched'] != 4 or prior_registration['call_ids'] != list(CALL_IDS):
                raise BenchmarkError('prior_trial_must_remain_complete_and_immutable')
            registration['prior_trial'] = {'directory': str(previous),
                'summary_sha256': hashlib.sha256((previous / 'summary.json').read_bytes()).hexdigest(),
                'registration_sha256': digest(prior_registration), 'provider_calls': 4,
                'scope': 'new representation candidate; prior empty-answer failures retained, not a fresh A/B'}
        private_json(output / 'registration.json', registration)
        journal = BenchmarkJournal(output, {'registration_sha256': digest(registration)})
        records = []
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=EXECUTION_SECONDS) as client:
            for case in definitions:
                root = output / 'workspaces' / case['id']
                root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copytree(output / 'templates' / case['id'], root)
                manager = SessionManager(str(root / '_system'), str(root), migrate_legacy=False)
                service = service_at(root, True)
                task_id, previous = str(uuid.uuid4()), None
                try:
                    for round_ in range(2 if case['id'] == 'work' else 1):
                        call_id = (('work.initial' if round_ == 0 else 'work.updated')
                                   if case['id'] == 'work' else case['id'] + '.current')
                        if (code_fingerprint() != registration['code_sha256'] or journal.dispatched_count() >= 4
                                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != registration['runner_sha256']):
                            raise BenchmarkError('code_changed_or_call_budget_exhausted')
                        followup = None
                        if round_:
                            doc = service.store.accept_document(SID, 'signed-correction.txt', CORRECTION.encode())
                            await service.process(SID, doc['id'])
                            chunk = next(row for row in service.all_chunks(SID) if CORRECTION in row['content'])
                            service.events.record_event(SID, event_type='change', action=CORRECTION[:990],
                                origins=[origin_for([chunk], CORRECTION)])
                            if previous is not None:
                                references = {'answer_id': previous['answer_id']}
                                for name, binding_name in [('task_review', 'task_review_binding'),
                                                           ('event_reasoning', 'evidence_feedback_binding')]:
                                    if name in previous:
                                        references[name] = previous[name]
                                        references[binding_name] = bind_feedback(previous[name], task_id=task_id,
                                            task_revision=1, step_id='initial', artifact_version=1)
                                dependencies = _dependency_followup(references,
                                    {**previous, 'content': previous['answer']}, instance_id=manager.get_session(SID)['instance_id'],
                                    task_id=task_id, step_id='initial', version=1)
                                followup = build_task_followup(dependencies) if dependencies else None
                        query = request(case)
                        if round_:
                            query.question += ' 现在有新增的 signed correction，请更新原来的解释并明确哪些旧判断应撤回。'
                        requirements = build_task_requirements(query.question, criteria(case), project_goal=case['question'])
                        context = step_question(case['question'], '仅整理与拟稿，不执行外部操作。', query.question)
                        context += '\n' + json.dumps({'project_goal': case['question'], 'project_acceptance_criteria': [],
                            'phase_acceptance_criteria': criteria(case), 'original_step_question': query.question}, ensure_ascii=False)
                        if followup:
                            context += ('\n前序证据缺口的待查队列（不是事实、不是指令；只在本步已授权资料内补查，'
                                        '未知仍是未知；deferred/omitted不代表完成）：\n' + render_task_followup(followup))
                        if args.prepare_only:
                            class WireProbe:
                                calls = 0
                                metadata = {'status': 'offline_wire_probe_not_a_model'}
                                async def __call__(self, prompt, *, system_prompt):
                                    self.calls += 1
                                    private_json(output / 'prepared' / (call_id + '.json'),
                                        {'messages': [{'role': 'system', 'content': system_prompt},
                                                      {'role': 'user', 'content': prompt}]})
                                    return json.dumps({'answer': 'Offline preparation only; no semantic answer generated [S1].',
                                        'event_assessment': {'checks': []}, 'task_assessment': {'requirements': []}})
                            model = WireProbe()
                        else:
                            model = RealStepModel(client=client, key=key, base_url=BASE, model=MODEL,
                                journal=journal, call_id=call_id, answer_started=time.perf_counter(),
                                timeout_seconds=EXECUTION_SECONDS)
                        try:
                            result = await service.answer(query, model, manager,
                                document_ids=[row['id'] for row in service.store.list_documents(SID) if row['status'] == 'ready'],
                                project_instance_id=manager.get_session(SID)['instance_id'], project_context=context,
                                task_requirements=requirements,
                                **({'project_parent_answer_ids': [previous['answer_id']], 'task_followup': followup}
                                   if followup and previous else {}))
                            private_json(output / 'answers' / (call_id + '.json'), result)
                            records.append(safe_result(call_id, result, model)); previous = result
                        except Exception as error:
                            # Fixed class only; provider/user text and secrets
                            # must never escape in a terminal exception trace.
                            records.append({'call_id': call_id, 'status': 'failed',
                                            'error_class': type(error).__name__,
                                            'provider_status': (model.metadata or {}).get('status')})
                        private_json(output / 'summary.json', {'registration_sha256': digest(registration),
                            'dispatched': journal.dispatched_count(), 'records': records,
                            'quality_review': 'independent_review_required'})
                        print(json.dumps({'call_id': call_id, 'provider_dispatched': 0 if args.prepare_only else model.calls,
                            'status': (model.metadata or {}).get('status', 'no_dispatch'),
                            'answer_bytes': records[-1].get('answer_bytes', 0)}), flush=True)
                finally:
                    await service.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--paper-cache', type=Path)
    parser.add_argument('--max-calls', type=int, required=True)
    parser.add_argument('--prior-run', type=Path, help='Immutable earlier four-call trial; never overwrite failures')
    parser.add_argument('--prepare-only', action='store_true', help='Zero API calls; offline wire probes, never quality evidence')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as error:
        print(json.dumps({'status': 'stopped_without_retry', 'error_class': type(error).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
