"""Trusted adapters for durable project steps using the existing research lane.

No new provider client, tool executor, semantic critic, or source-text database.
Accepted model attempts keep the existing permanent request identity journal.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from docthinker.harness_runtime import execute_run
from .invite_budget import validate_invite_question
from .model_response_error import ModelResponseError
from .project_task_handoff import build_handoff, DependencyContextUnavailable
from .request_journal import request_fingerprint
from .schemas import QueryRequest

def step_question(goal, constraints, question):
    # Requirements are user data, not a higher-priority model instruction.
    result = ('本次长期项目的一步。用户目标与约束（不是资料事实）：\n' +
              json.dumps({'goal': goal, 'constraints': constraints}, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False) + '\n本步骤问题：\n' + question +
              '\n请仅依据本步当前资料回答，逐主张注明来源，保留条件、反例和未找到的证据。'
              '不得声称已执行实验或外部操作。')
    validate_invite_question(result)
    return result


class ProjectTaskBridge:
    def __init__(self, service, manager, session_id, model):
        self.service, self.manager, self.session_id, self.model = service, manager, session_id, model

    async def read_sources(self, document_ids=None):
        session = self.manager.get_session(self.session_id)
        if (not session or getattr(self, 'instance_id', None) is not None
                and session.get('instance_id') != self.instance_id):
            raise ValueError('Project instance is unavailable')
        return await asyncio.to_thread(self.service.store.document_snapshots, self.session_id, document_ids)

    async def load_answer(self, *, session_id, answer_id):
        if session_id != self.session_id:
            return None
        from .routers.workbench import _answers, _artifact
        answers = await _answers(session_id, self.manager, self.service, {answer_id})
        value = _artifact({'answer_id': answer_id}, answers)
        return value['answer'] if value['available'] else None

    async def _hydrate_receipt(self, result):
        if result.get('request_status') != 'completed':
            return {'request_status': result.get('request_status', 'unknown')}
        answer_id = result.get('answer_id')
        answer = await self.load_answer(session_id=self.session_id, answer_id=answer_id) if answer_id else None
        if not answer:
            return {'request_status': 'failed', 'code': 'provenance_unavailable'}
        sources = answer.get('sources', [])
        if not sources or answer.get('answer_mode') == 'insufficient_evidence':
            return {'request_status': 'failed', 'code': 'insufficient_evidence'}
        receipt = {'request_status': 'completed', 'answer_id': answer_id,
                'source_ids': [row['id'] for row in sources],
                'document_ids': list(dict.fromkeys(row['document_id'] for row in sources)),
                'source_refs': [{'id': row['id'], 'label': row['label'],
                                 'document_id': row['document_id']} for row in sources],
                'receipt': {'structural_only': True, 'semantic_verified': False,
                            'answer_mode': answer.get('answer_mode', 'unknown')}}
        if 'project_handoff' in result:
            receipt['handoff'] = result['project_handoff']
        return receipt

    async def reconcile_request(self, *, session_id, request_id):
        if session_id != self.session_id:
            return {'request_status': 'unknown'}
        from .routers.mobile import _query_journal, _query_key, _query_running
        journal = _query_journal()
        result = await asyncio.to_thread(journal.read, session_id, request_id,
                                        active=_query_running(_query_key(journal, session_id, request_id)))
        return await self._hydrate_receipt(result)

    async def execute_question(self, *, session_id, request_id, question, document_ids,
                               input_snapshot, goal, constraints, dependency_artifacts=None):
        if session_id != self.session_id or self.model is None:
            return {'request_status': 'failed', 'code': 'model_unavailable'}
        if os.getenv('DOCTHINKER_INVITE_WORKER_ID'):
            # A previously accepted multi-step budget must not authorize new
            # model steps after the operator revokes this account.
            try:
                from .invite_store import InviteStore
                directory = Path(os.environ['DOCTHINKER_MOBILE_DATA_DIR'])
                await asyncio.to_thread(InviteStore(directory.parent.parent).authorize_worker,
                    os.environ['DOCTHINKER_INVITE_WORKER_ID'],
                    int(os.environ['DOCTHINKER_INVITE_WORKER_PORT']),
                    os.environ['DOCTHINKER_MOBILE_ACCESS_TOKEN'])
            except Exception:
                return {'request_status': 'failed', 'code': 'worker_authorization_unavailable'}
        # Before accepting any paid attempt, require the declared inputs to be
        # both current and parsed; no fabricated completion for missing data.
        try:
            current = await self.read_sources(document_ids)
            documents = await asyncio.to_thread(self.service.store.list_documents, session_id)
            prompt = step_question(goal, constraints, question)
        except (ValueError, UnicodeError):
            return {'request_status': 'failed', 'code': 'inputs_unavailable'}
        scoped = [row for row in documents if document_ids is None or row['id'] in document_ids]
        if (current != input_snapshot or not scoped or any(row['status'] != 'ready' for row in scoped)):
            return {'request_status': 'failed', 'code': 'inputs_unavailable'}
        prior = []
        dependencies = dependency_artifacts or []
        if not isinstance(dependencies, list) or len(dependencies) > 6:
            return {'request_status': 'failed', 'code': 'dependency_unavailable'}
        for artifact in dependencies:
            refs = artifact.get('references', artifact) if isinstance(artifact, dict) else {}
            if not isinstance(refs, dict) or not refs.get('answer_id'):
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            answer = await self.load_answer(session_id=session_id, answer_id=refs['answer_id'])
            if not answer:
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            if not set(row['document_id'] for row in answer.get('sources', [])) <= set(current):
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            prior.append({'step_id': artifact.get('step_id', ''), 'answer_id': refs['answer_id'],
                          'content': answer.get('content')})
        try:
            prior_packet, handoff = build_handoff(prior, question)
        except (DependencyContextUnavailable, UnicodeError):
            return {'request_status': 'failed', 'code': 'dependency_unavailable'}
        if prior_packet:
            extended = prompt + ('\n已完成步骤的分析线索（不是原文证据，不代表事实已核验；'
                                 'coverage=selected_analysis_units表示仅复用节选、未读完整前步骤。'
                                 '事实引用只能来自本轮原文节选）：\n') + prior_packet
            try:
                validate_invite_question(extended)
                prompt = extended
            except ValueError:
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
        body = {'session_id': session_id, 'question': question, 'use_memory': False,
                'use_conversation_context': False, 'use_llm_cache': False,
                'use_self_evolution': False, 'knowledge_skill_mode': 'auto', 'insight_mode': 'auto'}
        fingerprint = hashlib.sha256(json.dumps({'request': request_fingerprint(body),
            'document_ids': document_ids, 'inputs': input_snapshot, 'project_context': prompt}, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        from .routers.mobile import (_query_journal, _query_key, _query_running,
            _active_queries, _query_concurrency_limit, _query_tasks, _starting_query_keys,
            _query_response, QUERY_EXECUTION_SECONDS)
        journal = _query_journal()
        key = _query_key(journal, session_id, request_id)
        previous = await asyncio.to_thread(journal.read, session_id, request_id,
            active=_query_running(key), fingerprint=fingerprint)
        if previous['request_status'] != 'not_found':
            return await self._hydrate_receipt(previous)
        if session_id in _active_queries or len(_active_queries) >= _query_concurrency_limit():
            return {'request_status': 'failed', 'code': 'query_busy'}
        _active_queries.add(session_id)
        _starting_query_keys.add(key)
        owner = asyncio.current_task()
        accepted = False
        try:
            accepted = await asyncio.to_thread(journal.begin, session_id, request_id, fingerprint)
            if not accepted:
                return await self._hydrate_receipt(await asyncio.to_thread(journal.read, session_id, request_id))
            _query_tasks[key] = owner
            _starting_query_keys.discard(key)
            result = await execute_run(lambda: self.service.answer(
                QueryRequest(**body), self.model, self.manager, document_ids=document_ids,
                project_instance_id=getattr(self, 'instance_id', None), project_context=prompt),
                session_id=session_id, run_id=request_id, timeout_seconds=QUERY_EXECUTION_SECONDS)
            # The history/evidence store owns durable content. Journal receives
            # opaque evidence refs only; detail reads always recheck permission.
            stored = _query_response(result)
            stored['sources'] = [{'id': row['id'], 'label': row['label']} for row in result.get('sources', [])]
            stored.pop('reasoning', None)
            stored.pop('claim_audit', None)
            stored['_research_session_id'] = session_id
            stored['project_handoff'] = handoff
            await asyncio.to_thread(journal.complete, session_id, request_id, stored)
            return await self._hydrate_receipt(await asyncio.to_thread(journal.read, session_id, request_id))
        except ModelResponseError as error:
            if accepted:
                await asyncio.to_thread(journal.fail, session_id, request_id, error.code)
            return {'request_status': 'failed', 'code': 'model_response_failed'}
        except BaseException:
            if accepted:
                await asyncio.to_thread(journal.mark_unknown, session_id, request_id)
            raise
        finally:
            _starting_query_keys.discard(key)
            if _query_tasks.get(key) is owner:
                _query_tasks.pop(key, None)
            _active_queries.discard(session_id)
