"""Explicit compose -> verify acceptance: eight fixed provider calls, no retry.

Work/home are synthetic records; the third domain is a pinned full real paper.
These are exposed development cases, not a holdout, A/B comparison, causal
identification experiment, or weeks-long product acceptance. Every verification
is a separately declared call. Failed reviews leave their saved drafts intact.
Private prompts, documents, provider replies and credentials never enter Git.
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
    BASE, MODEL, SID, PAPER_SHA, build_case, cases, code_fingerprint, origin_for,
    request, service_at,
)
from scripts.benchmark_goal_evidence import CORRECTION, criteria, execution_policy, safe_result
from scripts.benchmark_research_assist import (
    BenchmarkError, BenchmarkJournal, RealStepModel, digest, output_lock,
    private_json, read_key,
)

ROUNDS = ('work.initial', 'work.updated', 'home.current', 'paper.current')
CALL_IDS = tuple(f'{round_}.{purpose}' for round_ in ROUNDS for purpose in ('compose', 'verify'))


def rubric():
    return {
        'work.initial.compose': ['distinguishes estimated from measured hours',
            'does not exclude the unmeasured supplier alternative', 'no unauthorized commitment'],
        'work.updated.compose': ['retrieves the signed correction',
            'retracts the earlier 15-hour critical-path attribution',
            'retains the measured 20 hours outside the final critical path and two-day account outage',
            'does not claim a randomized intervention or universally unique cause'],
        'home.current.compose': ['current Friday time conflict',
            'neighbour access remains unauthorized', 'later arrival remains unconfirmed',
            'no contact, booking or disclosure claimed'],
        'paper.current.compose': ['both actual paper tasks and positional sensitivity are stated correctly',
            'experimental observation is separate from product hypothesis',
            'all visible citations name actual retained sources, no control-ID citations'],
        'each.verify': ['explicitly separate provider call, no hidden retry',
            'original draft retained verbatim and bound to current sources',
            'no proof or goal promotion from the existence of a review',
            'unexamined, unmet or contradicted obligations remain open'],
    }


async def run(args):
    from docthinker.session_manager import SessionManager
    from docthinker.server.project_evidence_feedback import bind_feedback
    from docthinker.server.project_task_bridge import _dependency_followup, step_question
    from docthinker.server.research_review_stage import build_review_target
    from docthinker.server.research_task_review import (
        build_task_followup, build_task_requirements, render_task_followup,
    )

    output = args.output.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo) or args.max_calls != 8:
        raise BenchmarkError('private_output_and_exact_eight_call_budget_required')
    if (output / 'registration.json').exists():
        raise BenchmarkError('trial_cannot_be_replaced_or_resumed')
    key = '' if args.prepare_only else read_key(args.api_key_file)
    with output_lock(output):
        cached = args.paper_cache.expanduser().resolve()
        if hashlib.sha256(cached.read_bytes()).hexdigest() != PAPER_SHA:
            raise BenchmarkError('paper_hash_mismatch')
        destination = output / 'materials/lost-in-the-middle-v1.pdf'
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        destination.chmod(0o600)
        definitions = cases()
        corpus = {case['id']: await build_case(output, case) for case in definitions}
        previous_trials = []
        for previous in args.prior_run or []:
            previous = previous.expanduser().resolve()
            registration = json.loads((previous / 'registration.json').read_text())
            summary = json.loads((previous / 'summary.json').read_text())
            planned = registration.get('call_ids')
            if (planned not in (list(ROUNDS), list(CALL_IDS))
                    or {row.get('call_id') for row in summary.get('records', [])} != set(planned)
                    or len(summary.get('records', [])) != len(planned)
                    or type(summary.get('dispatched')) is not int
                    or not 0 <= summary['dispatched'] <= len(planned)):
                raise BenchmarkError('prior_trial_must_remain_complete_including_failed_or_skipped_stages')
            previous_trials.append({'registration_sha256': digest(registration),
                'summary_sha256': hashlib.sha256((previous / 'summary.json').read_bytes()).hexdigest(),
                'provider_calls': summary['dispatched'], 'planned_stages': len(planned)})
        registration = {'schema_version': 1, 'protocol': 'explicit_compose_verify_v1',
            'code_sha256': code_fingerprint(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'call_ids': list(CALL_IDS), 'max_provider_calls': 8, 'automatic_retry': False,
            'preparation_only': args.prepare_only, 'model': MODEL, 'base_url': BASE,
            'corpora': corpus, 'material_kinds': {case['id']: case['material_kind'] for case in definitions},
            'execution_policy': execution_policy(), 'criteria': {case['id']: criteria(case) for case in definitions},
            'rubric': rubric(), 'prior_trials': previous_trials,
            'limits': 'exposed development cases; two declared calls per round, no fresh A/B or universal causal guarantee'}
        private_json(output / 'registration.json', registration)
        journal = BenchmarkJournal(output, {'registration_sha256': digest(registration)})
        records = []

        def save_summary():
            private_json(output / 'summary.json', {'registration_sha256': digest(registration),
                'dispatched': journal.dispatched_count(), 'records': records,
                'quality_review': 'independent_review_required'})

        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=240) as client:
            for case in definitions:
                root = output / 'workspaces' / case['id']
                root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copytree(output / 'templates' / case['id'], root)
                manager = SessionManager(str(root / '_system'), str(root), migrate_legacy=False)
                service = service_at(root, True)
                task_id, previous_answer = str(uuid.uuid4()), None
                try:
                    for ordinal in range(2 if case['id'] == 'work' else 1):
                        round_id = ('work.updated' if ordinal else 'work.initial') if case['id'] == 'work' else case['id'] + '.current'
                        followup = None
                        if ordinal:
                            document = service.store.accept_document(SID, 'signed-correction.txt', CORRECTION.encode())
                            await service.process(SID, document['id'])
                            chunk = next(row for row in service.all_chunks(SID) if CORRECTION in row['content'])
                            service.events.record_event(SID, event_type='change', action=CORRECTION[:990],
                                origins=[origin_for([chunk], CORRECTION)])
                            if previous_answer is not None:
                                refs = {'answer_id': previous_answer['answer_id']}
                                for name, binding in [('task_review', 'task_review_binding'),
                                                      ('event_reasoning', 'evidence_feedback_binding')]:
                                    if name in previous_answer:
                                        refs[name] = previous_answer[name]
                                        refs[binding] = bind_feedback(previous_answer[name], task_id=task_id,
                                            task_revision=1, step_id='initial', artifact_version=1)
                                dependencies = _dependency_followup(refs,
                                    {**previous_answer, 'content': previous_answer['answer']},
                                    instance_id=manager.get_session(SID)['instance_id'], task_id=task_id,
                                    step_id='initial', version=1)
                                followup = build_task_followup(dependencies) if dependencies else None
                        query = request(case)
                        if ordinal:
                            query.question += ' 现在有新增的 signed correction，请更新原来的解释并明确哪些旧判断应撤回。'
                        requirements = build_task_requirements(query.question, criteria(case), project_goal=case['question'])
                        context = step_question(case['question'], '仅整理与拟稿，不执行外部操作。', query.question)
                        context += '\n' + json.dumps({'project_goal': case['question'],
                            'project_acceptance_criteria': [], 'phase_acceptance_criteria': criteria(case),
                            'original_step_question': query.question}, ensure_ascii=False)
                        if followup:
                            context += '\n前序证据缺口（仅供补查，不是事实）：\n' + render_task_followup(followup)
                        draft = None
                        for purpose in ('compose', 'verify'):
                            call_id = round_id + '.' + purpose
                            if (code_fingerprint() != registration['code_sha256']
                                    or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != registration['runner_sha256']
                                    or journal.dispatched_count() >= 8):
                                raise BenchmarkError('code_changed_or_call_budget_exhausted')
                            if purpose == 'verify' and draft is None:
                                records.append({'call_id': call_id, 'status': 'skipped_missing_draft',
                                                'provider_dispatched': 0})
                                save_summary()
                                continue
                            if args.prepare_only:
                                class WireProbe:
                                    calls = 0
                                    metadata = {'status': 'offline_wire_probe_not_a_model'}
                                    async def __call__(self, prompt, *, system_prompt):
                                        self.calls += 1
                                        private_json(output / 'prepared' / (call_id + '.json'), {
                                            'messages': [{'role': 'system', 'content': system_prompt},
                                                         {'role': 'user', 'content': prompt}]})
                                        return json.dumps({'answer': 'Offline stage probe only [S1].',
                                            'event_assessment': {'checks': []},
                                            'task_assessment': {'requirements': []}})
                                model = WireProbe()
                            else:
                                model = RealStepModel(client=client, key=key, base_url=BASE, model=MODEL,
                                    journal=journal, call_id=call_id, answer_started=time.perf_counter(), timeout_seconds=240)
                            kwargs = {'analysis_purpose': purpose}
                            if purpose == 'verify':
                                kwargs['review_target'] = build_review_target(
                                    {**draft, 'id': draft['answer_id'], 'content': draft['answer']},
                                    question=query.question, session_id=SID,
                                    instance_id=manager.get_session(SID)['instance_id'])
                                kwargs['project_parent_answer_ids'] = [draft['answer_id']]
                            elif followup and previous_answer:
                                kwargs.update(task_followup=followup,
                                              project_parent_answer_ids=[previous_answer['answer_id']])
                            try:
                                result = await service.answer(query, model, manager,
                                    document_ids=[row['id'] for row in service.store.list_documents(SID) if row['status'] == 'ready'],
                                    project_instance_id=manager.get_session(SID)['instance_id'],
                                    project_context=context, task_requirements=requirements, **kwargs)
                                private_json(output / 'answers' / (call_id + '.json'), result)
                                record = {**safe_result(call_id, result, model), 'purpose': purpose,
                                    'status': 'delivered',
                                    'citation_valid': result.get('citation_validation', {}).get('valid') is True,
                                    'unknown_citation_labels': result.get('citation_validation', {}).get('unknown_labels', [])}
                                if purpose == 'compose':
                                    draft = result
                                else:
                                    record.update(target_answer_sha256=hashlib.sha256(draft['answer'].encode()).hexdigest(),
                                        original_draft_retained_verbatim=draft['answer'] == result['answer'])
                                previous_answer = result
                                records.append(record)
                            except Exception as error:
                                private_json(output / 'errors' / (call_id + '.json'),
                                    {'error_class': type(error).__name__, 'message': str(error)})
                                records.append({'call_id': call_id, 'purpose': purpose, 'status': 'failed',
                                    'error_class': type(error).__name__, 'provider_status': (model.metadata or {}).get('status'),
                                    'draft_preserved': draft is not None})
                            save_summary()
                            print(json.dumps({'call_id': call_id, 'provider_dispatched': 0 if args.prepare_only else model.calls,
                                'status': (model.metadata or {}).get('status', 'no_dispatch'),
                                'answer_bytes': records[-1].get('answer_bytes', 0)}), flush=True)
                finally:
                    await service.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper-cache', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--max-calls', type=int, required=True)
    parser.add_argument('--prior-run', type=Path, action='append')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as error:
        print(json.dumps({'status': 'stopped_without_retry', 'error_class': type(error).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
