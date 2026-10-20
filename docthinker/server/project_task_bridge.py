"""Trusted adapters for durable project steps using the existing research lane.

No new provider client, tool executor, semantic critic, or source-text database.
Accepted model attempts keep the existing permanent request identity journal.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from docthinker.harness_runtime import execute_run
from .invite_budget import validate_invite_question
from .model_response_error import ModelResponseError
from .project_task_handoff import build_handoff, DependencyContextUnavailable
from .project_evidence_feedback import normalize_event_reasoning
from .research_task_review import build_task_requirements, normalize_task_review
from .request_journal import request_fingerprint
from .schemas import QueryRequest

def step_question(goal, constraints, question, *, personal=False):
    # Requirements are user data, not a higher-priority model instruction.
    result = ('本次长期项目的一步。用户目标与约束（不是资料事实）：\n' +
              json.dumps({'goal': goal, 'constraints': constraints}, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False) + '\n本步骤问题：\n' + question +
              ('\n这是个人任务。只使用本步用户要求、明确选中的已确认记忆以及已提供的前序分析；'
               '区分用户约束、建议与外部事实，不得伪造资料来源或声称个人记忆已经外部核验。'
               '缺少的信息应明确列出，交付可以直接使用的草稿或计划。' if personal else
               '\n请仅依据本步当前资料回答，逐主张注明来源，保留条件、反例和未找到的证据。') +
              '不得声称已执行实验或外部操作。')
    validate_invite_question(result)
    return result


class ProjectTaskBridge:
    def __init__(self, service, manager, session_id, model):
        self.service, self.manager, self.session_id, self.model = service, manager, session_id, model
        self.phase_guard = None
        self.phase_context_reader = None

    async def read_sources(self, document_ids=None):
        session = self.manager.get_session(self.session_id)
        if (not session or getattr(self, 'instance_id', None) is not None
                and session.get('instance_id') != self.instance_id):
            raise ValueError('Project instance is unavailable')
        return await asyncio.to_thread(self.service.store.document_snapshots, self.session_id, document_ids)

    async def read_context(self, memory_ids=None):
        session = self.manager.get_session(self.session_id)
        if (not session or getattr(self, 'instance_id', None) is not None
                and session.get('instance_id') != self.instance_id):
            raise ValueError('Project instance is unavailable')
        rows = await asyncio.to_thread(self.service.task_memory_context, self.session_id, memory_ids or [])
        return {row['id']: row['revision'] for row in rows}

    def validate_inputs(self, steps):
        """Final synchronous, batch snapshot check; no await after this barrier."""
        try:
            session = self.manager.get_session(self.session_id)
            return bool(session and session.get('instance_id') == getattr(self, 'instance_id', None)
                        and self.service.task_inputs_live(self.session_id, steps))
        except (sqlite3.Error, ValueError):
            # A busy/corrupt snapshot is unavailable, never permission to
            # publish stale content, retry a provider or wait seconds here.
            return False

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
        personal = result.get('project_input_kind') == 'personal'
        contexts = result.get('project_context_refs')
        if personal:
            if (type(contexts) is not dict or len(contexts) > 8 or sources
                    or any(type(revision) is not int or revision < 1 for revision in contexts.values())):
                return {'request_status': 'failed', 'code': 'provenance_unavailable'}
            try:
                if await self.read_context(list(contexts)) != contexts:
                    return {'request_status': 'failed', 'code': 'inputs_unavailable'}
            except (ValueError, KeyError):
                return {'request_status': 'failed', 'code': 'inputs_unavailable'}
        if (not sources and not personal) or answer.get('answer_mode') == 'insufficient_evidence':
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
        if 'project_phase_handoff' in result:
            task_id = result.get('project_phase_task_id')
            if self.phase_guard is None or not self.phase_guard(task_id):
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            receipt['phase_handoff'] = result['project_phase_handoff']
        if personal:
            receipt.update(input_kind='personal', context_refs=dict(contexts))
        if 'event_reasoning' in answer or 'event_reasoning' in result:
            try:
                feedback = normalize_event_reasoning(answer.get('event_reasoning', result.get('event_reasoning')))
                if ('event_reasoning' in answer and 'event_reasoning' in result
                        and feedback != normalize_event_reasoning(result['event_reasoning'])):
                    raise ValueError('Evidence feedback differs from its saved answer')
                receipt['event_reasoning'] = feedback
            except (ValueError, TypeError, KeyError):
                return {'request_status': 'failed', 'code': 'provenance_unavailable'}
        if 'task_review' in answer or 'task_review' in result:
            try:
                review = normalize_task_review(answer.get('task_review', result.get('task_review')))
                if ('task_review' in answer and 'task_review' in result
                        and review != normalize_task_review(result['task_review'])):
                    raise ValueError('Task review differs from its saved answer')
                receipt['task_review'] = review
            except (ValueError, TypeError, KeyError):
                return {'request_status': 'failed', 'code': 'provenance_unavailable'}
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
                               input_snapshot, goal, constraints, dependency_artifacts=None,
                               input_kind='documents', memory_ids=None, context_snapshot=None,
                               phase_dependencies=None, phase_task_id=None, phase_context=None,
                               origin_task_id=None):
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
        personal = input_kind == 'personal'
        try:
            if origin_task_id is not None and phase_task_id is not None and origin_task_id != phase_task_id:
                raise ValueError('Task identity differs from phase context')
            if input_kind not in {'documents', 'personal'}:
                raise ValueError('Invalid task input kind')
            if personal:
                if (document_ids != [] or input_snapshot != {} or type(memory_ids) is not list
                        or len(memory_ids) > 8 or len(set(memory_ids)) != len(memory_ids)
                        or type(context_snapshot) is not dict or set(context_snapshot) != set(memory_ids)
                        or any(type(revision) is not int or revision < 1 for revision in context_snapshot.values())
                        or await self.read_context(memory_ids) != context_snapshot):
                    raise ValueError('Personal task context changed')
                from .research_retrieval import MAX_MEMORY_BYTES
                notes = await asyncio.to_thread(self.service.task_memory_context, session_id,
                    memory_ids, expected=context_snapshot)
                if sum(len(row['text'].encode('utf-8')) for row in notes) > MAX_MEMORY_BYTES:
                    raise ValueError('Selected whole memory packet exceeds task input budget')
                current, documents = {}, []
            else:
                if memory_ids or context_snapshot:
                    raise ValueError('Document tasks cannot borrow personal context')
                current = await self.read_sources(document_ids)
                documents = await asyncio.to_thread(self.service.store.list_documents, session_id)
            prompt = step_question(goal, constraints, question, personal=personal)
            if phase_context is not None:
                if (self.phase_context_reader is None or
                        self.phase_context_reader(task_id=phase_task_id) != phase_context):
                    raise ValueError('Phase acceptance context changed')
                prompt += '\n长期项目与阶段验收标准（用户要求，不是事实或已完成声明）：\n' + json.dumps(
                    phase_context, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                validate_invite_question(prompt)
            task_requirements = build_task_requirements(question,
                phase_context.get('phase_acceptance_criteria', []) if isinstance(phase_context, dict) else None,
                input_kind=input_kind)
        except (ValueError, UnicodeError, KeyError):
            return {'request_status': 'failed', 'code': 'inputs_unavailable'}
        scoped = [row for row in documents if document_ids is None or row['id'] in document_ids]
        if not personal and (current != input_snapshot or not scoped or any(row['status'] != 'ready' for row in scoped)):
            return {'request_status': 'failed', 'code': 'inputs_unavailable'}
        prior = []
        parent_task_refs = []
        dependencies, external = dependency_artifacts or [], phase_dependencies or []
        if (not isinstance(dependencies, list) or not isinstance(external, list)
                or len(dependencies) + len(external) > 6
                or (external and (self.phase_guard is None or not self.phase_guard(phase_task_id)))):
            return {'request_status': 'failed', 'code': 'dependency_unavailable'}
        for artifact in [*dependencies, *external]:
            refs = artifact.get('references', artifact) if isinstance(artifact, dict) else {}
            if not isinstance(refs, dict) or not refs.get('answer_id'):
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            answer = await self.load_answer(session_id=session_id, answer_id=refs['answer_id'])
            if not answer:
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            if not set(row['document_id'] for row in answer.get('sources', [])) <= set(current):
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            if refs.get('input_kind') == 'personal':
                # A dependency cannot silently widen the selected memory scope.
                if (not personal or type(refs.get('context_refs')) is not dict
                        or any(context_snapshot.get(mid) != revision
                               for mid, revision in refs['context_refs'].items())):
                    return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            prior.append({'step_id': artifact.get('step_id', ''), 'answer_id': refs['answer_id'],
                          'content': answer.get('content')})
            origin = artifact.get('phase_ref')
            local_task_id = origin_task_id if origin_task_id is not None else phase_task_id
            if origin is not None or local_task_id is not None:
                task_ref = {'instance_id': self.instance_id,
                    'task_id': origin['task_id'] if origin else local_task_id,
                    'step_id': origin['step_id'] if origin else artifact['step_id'],
                    'version': origin['version'] if origin else artifact['version'], 'answer_id': refs['answer_id']}
                # Mobile production keeps research under the worker root but
                # task state under worker/data/_system. Direct-data adapters
                # keep both under data/_system. Bind the actual trusted manager
                # layout explicitly; readers must not search other owned DBs.
                try:
                    task_directory = Path(self.manager.base_storage_path)
                    research_root = self.service.store.root
                    if task_directory == research_root / 'data' / '_system':
                        task_ref['storage_layout'] = 'worker_data_system'
                    elif task_directory != research_root / '_system':
                        return {'request_status': 'failed', 'code': 'dependency_unavailable'}
                except (AttributeError, TypeError, ValueError):
                    return {'request_status': 'failed', 'code': 'dependency_unavailable'}
                parent_task_refs.append(task_ref)
        try:
            prior_packet, handoff = build_handoff(prior, question)
        except (DependencyContextUnavailable, UnicodeError):
            return {'request_status': 'failed', 'code': 'dependency_unavailable'}
        phase_handoff = None
        if external:
            blocks = json.loads(prior_packet)
            def counts(rows):
                full = sum(row['coverage'] == 'full' for row in rows)
                return dict(required=len(rows), used=len(rows), full=full, excerpted=len(rows)-full, omitted=0)
            handoff = counts(blocks[:len(dependencies)])
            phase_handoff = {**counts(blocks[len(dependencies):]), 'refs': [row['phase_ref'] for row in external]}
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
        binding = {'request': request_fingerprint(body), 'document_ids': document_ids,
                   'inputs': input_snapshot, 'project_context': prompt, 'task_requirements': task_requirements}
        if personal:
            binding.update(input_kind='personal', context_refs=context_snapshot)
        if origin_task_id is not None:
            binding.update(origin_task_id=origin_task_id, parent_task_refs=parent_task_refs)
        if external:
            binding.update(phase_task_id=phase_task_id, phase_refs=phase_handoff['refs'])
            if not self.phase_guard(phase_task_id):
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
        fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True,
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
        model_dispatched = False

        async def tracked_model(*args, **kwargs):
            nonlocal model_dispatched
            # This is the same configured model, not a second provider. The
            # flag records attempted dispatch, never assumes network rollback.
            model_dispatched = True
            return await self.model(*args, **kwargs)

        try:
            accepted = await asyncio.to_thread(journal.begin, session_id, request_id, fingerprint)
            if not accepted:
                return await self._hydrate_receipt(await asyncio.to_thread(journal.read, session_id, request_id))
            _query_tasks[key] = owner
            _starting_query_keys.discard(key)
            result = await execute_run(lambda: self.service.answer(
                QueryRequest(**body), tracked_model, self.manager, document_ids=document_ids,
                project_instance_id=getattr(self, 'instance_id', None), project_context=prompt,
                task_requirements=task_requirements,
                **({'task_context_refs': context_snapshot} if personal else {}),
                **({'project_parent_answer_ids': list(dict.fromkeys(row['answer_id'] for row in prior))}
                   if prior and getattr(self, 'instance_id', None) is not None else {}),
                **({'project_parent_task_refs': parent_task_refs} if parent_task_refs else {})),
                session_id=session_id, run_id=request_id, timeout_seconds=QUERY_EXECUTION_SECONDS)
            # The history/evidence store owns durable content. Journal receives
            # opaque evidence refs only; detail reads always recheck permission.
            stored = _query_response(result)
            stored['sources'] = [{'id': row['id'], 'label': row['label']} for row in result.get('sources', [])]
            stored.pop('reasoning', None)
            stored.pop('claim_audit', None)
            stored['_research_session_id'] = session_id
            stored['project_handoff'] = handoff
            if 'event_reasoning' in result:
                stored['event_reasoning'] = normalize_event_reasoning(result['event_reasoning'])
            if 'task_review' in result:
                stored['task_review'] = normalize_task_review(result['task_review'])
            if personal:
                stored.update(project_input_kind='personal', project_context_refs=dict(context_snapshot))
            if phase_handoff is not None:
                stored.update(project_phase_task_id=phase_task_id, project_phase_handoff=phase_handoff)
            await asyncio.to_thread(journal.complete, session_id, request_id, stored)
            return await self._hydrate_receipt(await asyncio.to_thread(journal.read, session_id, request_id))
        except ModelResponseError as error:
            if accepted:
                await asyncio.to_thread(journal.fail, session_id, request_id, error.code)
            return {'request_status': 'failed', 'code': 'model_response_failed'}
        except BaseException as error:
            if accepted:
                if model_dispatched:
                    await asyncio.to_thread(journal.mark_unknown, session_id, request_id)
                else:
                    await asyncio.to_thread(journal.fail, session_id, request_id, 'inputs_unavailable')
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    return {'request_status': 'failed', 'code': 'inputs_unavailable'}
            raise
        finally:
            _starting_query_keys.discard(key)
            if _query_tasks.get(key) is owner:
                _query_tasks.pop(key, None)
            _active_queries.discard(session_id)
