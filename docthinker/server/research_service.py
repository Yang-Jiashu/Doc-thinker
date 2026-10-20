"""A bounded research workflow: deterministic jobs/retrieval, one answer call.

Only the opted-in private mobile profile owns this service. Bounded hybrid search
does not claim semantic completeness, causal-chain discovery or autonomous RSI.
User-confirmed memory is context, never a substitute for document evidence.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from docthinker.harness_runtime import current_run, execute_run
from .invite_budget import bounded_invite_history, completion_output_budget, validate_invite_question
from .research_retrieval import build_retrieval_plan, citation_validation, refresh_query_coverage
from .research_corpus_index import PreparedCorpusCache, CorpusIndexTooLarge
from .research_memory import select_confirmed_memories
from .research_memory_dependencies import MemoryDependencies
from .research_answer_dependencies import AnswerDependencies
from .personal_events import PersonalEvents
from .research_event_dependencies import EventDependencies
from .research_event_reasoning import (build_event_obligations, supplement_event_gaps,
                                       render_event_reasoning, parse_event_response,
                                       prepare_event_frontier, event_source_window_guards,
                                       event_trial_response, EVENT_RESPONSE_POLICY)
from .research_task_review import (build_task_requirements, render_task_requirements,
                                  task_review_contract, parse_task_response, hydrate_task_review)
from .research_store import ResearchStore, ResearchNotFoundError
from .research_source_contract import MAX_ANSWER_SOURCES
from .research_parser_gate import ParserSlotUnavailable, offline_parser_slot
from .research_evidence_index import ResearchEvidenceIndex, hydrate_reasoning
from .research_reasoning_context import render_reasoning, render_evidence
from .research_provenance import audit_answer
from .research_context_assist import build_context_assist
from .research_prompt_budget import empty_assistance, fit_research_prompt, retrieval_question, requires_document_evidence
from .research_scheduler import BackgroundContextScheduler
from .research_semantic import LocalSemanticRetriever
from .research_fallback_sources import filter_fallback_sources
from .research_answer_contract import SYSTEM_POLICY as TRUSTED_RESEARCH_POLICY
from .research_answer_contract import knowledge_skill_system_policy
from .research_knowledge import select_task_knowledge, knowledge_receipt, unavailable_knowledge
from .research_graph_retrieval import BoundedEvidenceGraph, graph_retrieval_intent
from .research_hypotheses import detect_hypothesis_intent
from .research_discovery import build_discovery_context, fit_graph_receipt
from .insight_runtime import InsightRuntime
from .research_trial_delivery import (requests_trial_delivery, trial_contract_packet,
                                     trial_delivery_receipt, unavailable_trial_receipt)

EXTRACTION_SECONDS = 25
MAX_EXTRACT_OUTPUT = 2 * 1024 * 1024
_AUTO_SEMANTIC = object()
_AUTO_RERANKER = object()
_AUTO_TRANSLATOR = object()
WITHDRAWN_ANALYSIS = "本次回答依赖的部分资料已被撤回或暂时无法核验，原回答及相关分析已收起。请先检查资料与引用状态；仍可查看有效原文，不会自动重新请求模型。"
WITHDRAWN_MEMORY = "本次回答使用的记忆已更正、过期、删除或无法核验，原回答已收起，不会通过历史上下文再次使用，也不会自动重新请求模型。"


def _append_insight_sources(sources, extra):
    """Append whole unique excerpts into spare budget without relabeling inputs."""
    from .research_retrieval import MAX_SOURCE_BYTES

    selected = list(sources)
    identifiers = {row['id'] for row in selected}
    texts = {' '.join(row['content'].split()) for row in selected}
    labels = {row['label'] for row in selected}
    remaining = MAX_SOURCE_BYTES - sum(len(row['content'].encode()) for row in selected)
    for row in extra[:2]:
        normalized = ' '.join(row['content'].split())
        size = len(row['content'].encode())
        if row['id'] in identifiers or normalized in texts or size > remaining or len(selected) >= MAX_ANSWER_SOURCES:
            continue
        number = len(selected) + 1
        while number <= MAX_ANSWER_SOURCES and f'S{number}' in labels:
            number += 1
        if number > MAX_ANSWER_SOURCES:
            # Preserve existing labels (including sparse packets), without
            # inventing an out-of-contract S11 citation.
            number = next((value for value in range(1, MAX_ANSWER_SOURCES + 1)
                           if f'S{value}' not in labels), None)
        if number is None:
            continue
        selected.append({**row, 'label': f'S{number}'})
        labels.add(f'S{number}')
        identifiers.add(row['id'])
        texts.add(normalized)
        remaining -= size
    return selected


def _supplement_insight_evidence(question, sources, chunks, proposal):
    """Run the shared bounded incremental pass without altering primary sources."""
    from .research_insight_supplement import supplement_evidence

    return supplement_evidence(question, sources, chunks, proposal,
                               append_sources=_append_insight_sources)


async def extract_offline(content: bytes, filename: str) -> dict:
    try:
        async with offline_parser_slot():
            return await _extract_offline_unlocked(content, filename)
    except ParserSlotUnavailable:
        return {"error_code": "parser_capacity_unavailable"}


async def _extract_offline_unlocked(content: bytes, filename: str) -> dict:
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        sys.executable, "-m", "docthinker.server.research_extract", Path(filename).suffix.lower(),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={key: value for key, value in os.environ.items() if key in {
            "PATH", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
        }},
    ))
    try:
        # Obtain and reap a child even if shutdown arrives while it is spawning;
        # only then can the deployment-wide parser slot safely be released.
        child = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        child = await spawn
        if child.returncode is None:
            child.kill()
            await child.wait()
        raise
    async def exchange():
        async def write_input():
            child.stdin.write(content)
            await child.stdin.drain()
            child.stdin.close()
        writer = asyncio.create_task(write_input())
        try:
            pieces, size = [], 0
            while piece := await child.stdout.read(65536):
                size += len(piece)
                if size > MAX_EXTRACT_OUTPUT:
                    raise ValueError("extraction_output_limit")
                pieces.append(piece)
            await writer
            await child.wait()
            if child.returncode:
                return {"error_code": "extraction_resource_limit"}
            return json.loads(b"".join(pieces))
        finally:
            if not writer.done():
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
    try:
        return await asyncio.wait_for(exchange(), EXTRACTION_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception:
        return {"error_code": "extraction_resource_limit"}
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


class ResearchService:
    def __init__(self, root: Path, *, extractor=extract_offline, context_assistance_enabled=True,
                 semantic_retriever=_AUTO_SEMANTIC, evidence_reranker=_AUTO_RERANKER,
                 query_translator=_AUTO_TRANSLATOR, event_reasoning_enabled=True):
        if type(context_assistance_enabled) is not bool:
            raise ValueError('Context assistance flag must be boolean')
        # An operator/test ablation switch. The normal client still has one
        # conversational interface and does not select reasoning algorithms.
        self.context_assistance_enabled = context_assistance_enabled
        if type(event_reasoning_enabled) is not bool:
            raise ValueError('Event reasoning flag must be boolean')
        # Operator/benchmark ablation only, not a separate user reasoning mode.
        self.event_reasoning_enabled = event_reasoning_enabled
        quota = os.getenv('DOCTHINKER_RESEARCH_STORAGE_BYTES', str(512 * 1024 * 1024))
        if not re.fullmatch(r'[0-9]{1,16}', quota) or not 65536 <= int(quota) <= 2 ** 50:
            raise ValueError('Invalid research storage quota')
        self.store = ResearchStore(root, max_storage_bytes=int(quota))
        self.memory_dependencies = MemoryDependencies(self.store)
        self.answer_dependencies = AnswerDependencies(self.store)
        self.events = PersonalEvents(self.store)
        self.event_dependencies = EventDependencies(self.store)
        self.evidence = ResearchEvidenceIndex(self.store)
        # A derived, bounded lookup owned by this private service. It caches no
        # answer or generated relation and revalidates live source identities.
        self._graph_retrieval = BoundedEvidenceGraph()
        self._corpus_cache = PreparedCorpusCache()
        self._corpus_warm_task = None
        self._closing = False
        semantic_mode = os.getenv('DOCTHINKER_RESEARCH_SEMANTIC', 'auto')
        if semantic_mode not in {'auto', 'off', 'local'}:
            raise ValueError('Invalid local retrieval setting')
        self._semantic = (None if semantic_mode == 'off' else LocalSemanticRetriever(
            private_cache_root=Path(root).resolve() / '_system' / 'research' / 'passage-vectors'
        )) if semantic_retriever is _AUTO_SEMANTIC else semantic_retriever
        if evidence_reranker is _AUTO_RERANKER:
            from .research_reranker import LocalEvidenceReranker
            self._evidence_reranker = LocalEvidenceReranker() if semantic_mode != 'off' else None
        else:
            self._evidence_reranker = evidence_reranker
        if query_translator is _AUTO_TRANSLATOR:
            if semantic_mode != 'off':
                from .research_query_translation import LocalQueryTranslator
                self._query_translator = LocalQueryTranslator()
            else:
                self._query_translator = None
        else:
            self._query_translator = query_translator
        self._evidence_scheduler = BackgroundContextScheduler(timeout_seconds=8.0)
        # Project display IDs repeat across tenants; neither vectors nor query
        # caches may use the display ID alone. Do not expose the private path.
        self._semantic_namespace = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()
        self.extractor = extractor
        self._wake = asyncio.Event()
        self._runner = None
        self._last_parser_session = None
        self._context_scheduler = BackgroundContextScheduler()
        self._semantic_scheduler = BackgroundContextScheduler(timeout_seconds=1.0)
        self._semantic_prepare_task = None

    def start(self):
        if self._runner is None:
            self._runner = asyncio.create_task(self._run())
            self._wake.set()

    async def stop(self):
        self._closing = True
        await self._evidence_scheduler.close()
        await self._context_scheduler.close()
        await self._semantic_scheduler.close()
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
        await self._stop_semantic_warm()
        # CPU work cannot be killed by cancelling to_thread; a short cache lock
        # must not block the event loop during shutdown either.
        await asyncio.to_thread(self._corpus_cache.close)
        if self._corpus_warm_task is not None:
            # Cancelling the asyncio wrapper would not kill its actual thread.
            # Observe genuine completion instead of falsely releasing the lane.
            try:
                await asyncio.shield(self._corpus_warm_task)
            except Exception:
                pass

    def _warm_corpus(self, sid, chunks):
        if (self._closing or not chunks or
                self._corpus_warm_task is not None and not self._corpus_warm_task.done()):
            return False
        task = asyncio.create_task(asyncio.to_thread(self._corpus_cache.get, chunks, namespace=sid))
        self._corpus_warm_task = task
        def finished(value):
            if not value.cancelled():
                value.exception()  # No private exception text or automatic retry.
            if self._corpus_warm_task is value:
                self._corpus_warm_task = None
        task.add_done_callback(finished)
        return True

    def notify(self):
        self._wake.set()

    async def _run(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            attempted = set()
            while True:
                try:
                    # Inspect the bounded (100-document) queue between every
                    # job so a fresh project need not wait for a whole batch.
                    pending = await asyncio.to_thread(self.store.pending_documents, 100)
                except Exception:
                    # Keep the runner alive after a transient storage outage.
                    # No polling/model retry: the next explicit upload wakes it.
                    break
                pending = [document for document in pending
                           if (document["session_id"], document["id"]) not in attempted]
                if not pending:
                    break
                # Stable cyclic project order, not merely "different from the
                # last project", which could starve a third busy project. The
                # store's original creation order gives FIFO within a project.
                sessions = sorted({document["session_id"] for document in pending})
                session = next((sid for sid in sessions
                                if self._last_parser_session is None or sid > self._last_parser_session),
                               sessions[0])
                document = next(row for row in pending if row["session_id"] == session)
                key = (session, document["id"])
                self._last_parser_session = session
                attempted.add(key)
                # A failed claim/status write is still attempted only once in
                # this wake. Other projects can proceed; unchanged rows cannot
                # cause a spin or an automatic retry of the failed document.
                await self.process(*key)

    async def process(self, sid: str, document_id: str) -> bool:
        """Attempt one offline job; report durable completion/failure/removal."""
        failure_code = "extraction_failed"
        try:
            document = await asyncio.to_thread(self.store.claim_document, sid, document_id)
            if document is None:
                return False
            content = await asyncio.to_thread(self.store.read_document_bytes, sid, document_id)
            result = await self.extractor(content, document["filename"])
            if isinstance(result, dict) and isinstance(result.get("chunks"), list):
                await asyncio.to_thread(self.store.complete_document, sid, document_id, result["chunks"])
                try:
                    await asyncio.to_thread(self.evidence.index_document, sid, document_id)
                except Exception:
                    # Derived indexes can be rebuilt. Never mark intact source
                    # extraction failed because an optional index write failed.
                    pass
                try:
                    await self.prepare_retrieval(sid)
                except Exception:
                    # Optional cache preparation must not downgrade a durable
                    # ready document even when a source read races deletion.
                    pass
                return True
            else:
                failure_code = result.get("error_code", "extraction_failed") if isinstance(result, dict) else "extraction_failed"
        except asyncio.CancelledError:
            # Claim remains durable; next process boot reclaims offline work.
            raise
        except ResearchNotFoundError:
            return True  # User deleted it while it was processing.
        except Exception:
            pass  # Persist only the fixed failure code, never private diagnostics.
        if not isinstance(failure_code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", failure_code):
            failure_code = "extraction_failed"
        try:
            # A failed status write is attempted once, not recursively retried.
            await asyncio.to_thread(self.store.fail_document, sid, document_id, failure_code)
            return True
        except ResearchNotFoundError:
            return True
        except Exception:
            return False

    def all_chunks(self, sid: str, *, document_ids=None) -> list[dict]:
        return self.store.all_source_chunks(sid, document_ids=document_ids)

    async def _semantic_operation(self, sid, chunks, question=None):
        if self._semantic is None:
            return {'scores': {}, 'spans': {}, 'trace': {'status': 'disabled', 'provider_calls': 0, 'llm_tokens': 0}}
        try:
            namespace = self._semantic_namespace + '::' + sid
            if question is None:
                if self._closing:
                    return {'scores': {}, 'spans': {}, 'trace': {'status': 'closed', 'provider_calls': 0, 'llm_tokens': 0}}
                if self._semantic_prepare_task is not None and not self._semantic_prepare_task.done():
                    return {'scores': {}, 'spans': {}, 'trace': {'status': 'busy', 'provider_calls': 0, 'llm_tokens': 0}}
                # Reserve before awaiting: even a worker queued in the shared
                # executor occupies this one lane. Cancelling its waiter must
                # not cancel the wrapper or permit another real preparation.
                task = asyncio.create_task(asyncio.to_thread(self._semantic.prepare_documents, chunks, session_id=namespace))
                self._semantic_prepare_task = task
                def finished(value):
                    if not value.cancelled():
                        value.exception()  # Observe failures, never private text.
                    if self._semantic_prepare_task is value:
                        self._semantic_prepare_task = None
                task.add_done_callback(finished)
                return await asyncio.shield(task)
            work = await self._semantic_scheduler.run(lambda: self._semantic.semantic_scores(
                question, chunks, session_id=namespace))
            if work.status == 'completed':
                return work.value
            return {'scores': {}, 'spans': {}, 'trace': {'status': 'inference_unavailable' if work.status == 'failed' else work.status,
                                           'provider_calls': 0, 'llm_tokens': 0,
                                           'semantic_entailment_verified': False}}
        except Exception:
            # Embedding failure cannot abort a paid answer or leak source text
            # through an exception. Lexical retrieval remains available.
            return {'scores': {}, 'spans': {}, 'trace': {'status': 'inference_unavailable', 'provider_calls': 0,
                                           'llm_tokens': 0, 'semantic_entailment_verified': False}}

    async def _stop_semantic_warm(self):
        """Observe the genuine preparation exit, not a cancelled thread wrapper.

        Stop owns the finite local task, including one still queued in the shared
        pool. There is no model retry or hard-preemption/stop-deadline claim.
        """
        task = self._semantic_prepare_task
        if task is not None:
            try:
                await asyncio.shield(task)
            except Exception:
                pass

    async def prepare_retrieval(self, sid: str) -> dict:
        """Prepare vectors and optional local assets, never query inference.

        Preparation is not free: report its entire wait separately from answer
        latency. The existing evidence lane owns any worker surviving a timeout.
        """
        started = time.perf_counter()
        if self._closing:
            return {'status': 'closed', 'provider_calls': 0, 'llm_tokens': 0,
                    'query_embedded': False, 'vector_preparation_ms': 0.0,
                    'asset_preparation': {'status': 'closed', 'asset_only': True,
                                          'query_inference': False, 'provider_calls': 0},
                    'preparation_wait_ms': round((time.perf_counter() - started) * 1000, 3)}
        chunks = await asyncio.to_thread(self.all_chunks, sid) if self._semantic is not None else []
        result = await self._semantic_operation(sid, chunks)
        vector_ms = (time.perf_counter() - started) * 1000
        assets = await self.prepare_assistance_assets() if chunks else {
            'status': 'no_live_documents', 'asset_only': True, 'query_inference': False,
            'provider_calls': 0, 'wait_ms': 0.0, 'worker_ms': None, 'worker_running': False,
        }
        return {**result['trace'], 'query_embedded': False, 'vector_preparation_ms': round(vector_ms, 3),
                'asset_preparation': assets, 'preparation_wait_ms': round((time.perf_counter() - started) * 1000, 3)}

    async def prepare_assistance_assets(self) -> dict:
        """Best-effort, nonqueued asset preparation on the existing helper lane.

        Model gates also prevent prep/inference overlap across service objects.
        Stages run sequentially, not on new executors or independent model jobs.
        This never changes source readiness and is entirely off in the ablation.
        """
        started = time.perf_counter()
        receipt = {'asset_only': True, 'query_inference': False, 'provider_calls': 0,
                   'worker_ms': None, 'worker_running': False}

        def result(status, **values):
            return {**receipt, **values, 'status': status,
                    'wait_ms': round((time.perf_counter() - started) * 1000, 3)}

        if not self.context_assistance_enabled:
            return result('disabled')
        if self._semantic is None or self._evidence_reranker is None:
            return result('unavailable')

        def prepare():
            worker_started = time.perf_counter()
            components = {}
            for name, model in (('reranker', self._evidence_reranker), ('translator', self._query_translator)):
                method = getattr(model, 'prepare_assets', None)
                if not callable(method):
                    components[name] = {'status': 'unavailable'}
                    continue
                try:
                    trace = method()['trace']
                    # A preparation receipt is operational metadata only. Do
                    # not forward arbitrary model/exception text to the client.
                    component = {'status': trace['status'] if trace['status'] in {'ready', 'busy', 'unavailable'} else 'unavailable'}
                    for key in ('model_reused', 'initialization_ms', 'total_ms'):
                        value = trace.get(key)
                        if type(value) in (bool, int, float):
                            component[key] = value
                    components[name] = component
                except Exception:
                    components[name] = {'status': 'unavailable'}
                if components[name]['status'] == 'busy':
                    # Busy means skip this opportunity, not enqueue another
                    # model stage or retry work behind an active answer.
                    break
            return {'components': components, 'worker_ms': round((time.perf_counter() - worker_started) * 1000, 3)}

        work = await self._evidence_scheduler.run(prepare)
        if work.status == 'completed':
            return result('completed', **work.value, scheduler_timing=work.trace)
        return result(work.status, worker_running=self._evidence_scheduler.running, scheduler_timing=work.trace)

    def memory_candidates(self, sid: str, question: str, as_of=None) -> dict:
        memories = self.store.search_memories(sid, question, limit=100, as_of=as_of)
        result = select_confirmed_memories(question, memories)
        result['strategy'] = 'indexed_temporal_' + result['strategy']
        return result

    def task_memory_context(self, sid, memory_ids, *, expected=None):
        """Explicit selected context, never all user/project memory by default."""
        if (type(memory_ids) is not list or len(memory_ids) > 48
                or len(set(memory_ids)) != len(memory_ids)
                or any(type(mid) is not str or not re.fullmatch(
                    r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', mid)
                    for mid in memory_ids)):
            raise ValueError('Invalid explicit memory scope')
        with self.store._read() as db:
            if db is None:
                if memory_ids:
                    raise ResearchNotFoundError('Selected personal context unavailable')
                return []
            # One SQLite read snapshot, not one connection per note followed
            # by a separate lineage read. No private text is copied elsewhere.
            db.execute('BEGIN')
            now = self.store.clock()
            rows = [self.store._read_memory(db, self.store._required_memory(db, sid, mid),
                                           at=now) for mid in memory_ids]
            snapshot = {row['id']: row['revision'] for row in rows}
            if (any(row['status'] not in {'active', 'archived'} or row.get('redacted') for row in rows)
                    or (expected is not None and snapshot != expected)
                    or not self.memory_dependencies._live(db, sid,
                        [(row['id'], row['revision'], None) for row in rows])):
                raise ResearchNotFoundError('Selected personal context changed or unavailable')
            return rows

    def task_inputs_live(self, sid, steps):
        """One bounded publication barrier across ALL report sections.

        Called synchronously after the last await by the task adapter. It
        linearizes the response's current-input check in one SQLite snapshot;
        a withdrawal AFTER this check cannot retroactively revoke sent bytes.
        """
        if type(steps) is not list or not 1 <= len(steps) <= 8:
            raise ValueError('Invalid task publication scope')
        return self.task_input_groups_live(sid, [steps])

    def task_input_groups_live(self, sid, groups, *, deadline=None):
        """Request-local batched barrier; no cross-request freshness cache.

        Each original group keeps the ordinary 8-step/48-memory contract.
        Distinct ancestor scopes may be checked together in one read snapshot;
        identical scopes and transitive memory checks are only done once.
        """
        if type(groups) is not list:
            raise ValueError('Invalid task publication groups')
        unique, memories, answers = {}, {}, set()
        for steps in groups:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if type(steps) is not list or not 1 <= len(steps) <= 8:
                raise ValueError('Invalid task publication scope')
            local_memories = {}
            for step in steps:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                if type(step) is not dict:
                    raise ValueError('Invalid task publication step')
                if step.get('answer_id') is not None:
                    from .research_store import _identifier
                    answers.add(_identifier(step['answer_id']))
                kind = step.get('input_kind', 'documents')
                packet = {key: step.get(key, default) for key, default in (
                    ('document_ids', None), ('input_snapshot', {}), ('context_snapshot', {}))}
                packet['input_kind'] = kind
                if kind == 'personal':
                    for mid, revision in packet['context_snapshot'].items():
                        if mid in local_memories and local_memories[mid] != revision:
                            return False
                        local_memories[mid] = revision
                unique[json.dumps(packet, sort_keys=True, separators=(',', ':'), allow_nan=False)] = packet
            if len(local_memories) > 48:
                return False
            for mid, revision in local_memories.items():
                if mid in memories and memories[mid] != revision:
                    return False
                memories[mid] = revision
        if not unique:
            return True
        # This last synchronous barrier must never wait seconds on a writer.
        # Busy means no export, not stale-data reuse or another model call.
        with self.store._read(timeout=0.05) as db:
            if db is not None:
                db.execute('BEGIN')
                if deadline is not None:
                    db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 256)
            now = self.store.clock()
            if deadline is not None and time.monotonic() >= deadline:
                return False
            documents = self.store._document_snapshots_in(db, sid) if any(
                step['input_kind'] != 'personal' for step in unique.values()) else {}
            for step in unique.values():
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                scope = [] if step.get('input_kind') == 'personal' else step.get('document_ids')
                current = documents if scope is None else {key: value for key, value in documents.items() if key in scope}
                if current != step.get('input_snapshot', {}):
                    return False
            if memories:
                if db is None:
                    return False
                current_memories = {}
                identifiers = list(memories)
                for start in range(0, len(identifiers), 400):
                    batch = identifiers[start:start + 400]
                    rows = db.execute('SELECT memory_id,revision FROM research_memories WHERE session_id=? '
                        "AND status IN ('active','archived') AND memory_id IN (" + ','.join('?' for _ in batch) + ')',
                        [sid, *batch]).fetchall()
                    current_memories.update({row['memory_id']: row['revision'] for row in rows})
                if current_memories != memories:
                    return False
                from .research_memory_dependencies import _bindings_live
                if not _bindings_live(db, sid, [(mid, rev, None) for mid, rev in memories.items()],
                                      now=now, seen={'active': set(), 'memo': {}}):
                    return False
            if answers:
                if db is None:
                    return False
                from .research_memory_dependencies import source_answer_memory_live
                seen = {'active': set(), 'memo': {}}
                for identifier in answers:
                    if deadline is not None and time.monotonic() >= deadline:
                        return False
                    if not source_answer_memory_live(db, sid, identifier, at=now, seen=seen):
                        return False
            return deadline is None or time.monotonic() < deadline

    async def answer(self, request, model, manager, *, document_ids=None, project_instance_id=None,
                     project_context=None, task_context_refs=None, project_parent_answer_ids=None,
                     project_parent_task_refs=None, task_requirements=None) -> dict:
        # HTTP, mobile and direct service calls share one execution owner. A
        # transport that already entered the Harness is reused, not restarted.
        return await execute_run(lambda: self._answer(request, model, manager)
                                 if document_ids is None and project_instance_id is None and project_context is None and task_context_refs is None and project_parent_answer_ids is None and project_parent_task_refs is None and task_requirements is None else
                                 self._answer(request, model, manager, document_ids=document_ids,
                                              project_instance_id=project_instance_id,
                                              project_context=project_context,
                                              **({'task_context_refs': task_context_refs} if task_context_refs is not None else {}),
                                              **({'project_parent_answer_ids': project_parent_answer_ids}
                                                 if project_parent_answer_ids is not None else {}),
                                              **({'project_parent_task_refs': project_parent_task_refs}
                                                 if project_parent_task_refs is not None else {}),
                                              **({'task_requirements': task_requirements} if task_requirements is not None else {})),
                                 session_id=request.session_id)

    async def _answer(self, request, model, manager, *, document_ids=None, project_instance_id=None,
                      project_context=None, task_context_refs=None, project_parent_answer_ids=None,
                      project_parent_task_refs=None, task_requirements=None) -> dict:
        run = current_run()
        sid, question = request.session_id, request.question
        parent_answer_ids = []
        if project_parent_task_refs and not project_parent_answer_ids:
            raise ValueError('Task origins require parent analysis references')
        if project_parent_answer_ids is not None:
            if project_context is None or project_instance_id is None:
                raise ValueError('Parent analyses require explicit task context')
            parent_answer_ids = self.answer_dependencies.decode(json.dumps(project_parent_answer_ids))
            if not await asyncio.to_thread(self.answer_dependencies.parents_live, sid, parent_answer_ids):
                raise ResearchNotFoundError('Parent analysis provenance is unavailable')
        if project_instance_id is not None:
            session = manager.get_session(sid)
            if not session or session.get('instance_id') != project_instance_id:
                raise ResearchNotFoundError('Task project instance is unavailable')
        requested_mode = getattr(request, 'reasoning_mode', 'auto')
        skill_mode = getattr(request, 'knowledge_skill_mode', 'auto')
        insight_mode = getattr(request, 'insight_mode', 'auto')
        memory_as_of = getattr(request, 'memory_as_of', None)
        personal = task_context_refs is not None
        explicit_memories = None
        if personal:
            if (type(task_context_refs) is not dict or len(task_context_refs) > 8
                    or any(type(revision) is not int or revision < 1 for revision in task_context_refs.values())
                    or project_instance_id is None or project_context is None or document_ids != []
                    or request.use_memory or request.use_conversation_context or memory_as_of is not None):
                raise ValueError('Personal task context must be explicit and current')
            explicit_memories = await asyncio.to_thread(self.task_memory_context, sid,
                list(task_context_refs), expected=task_context_refs)
        validate_invite_question(question)
        if project_context is not None:
            # Trusted project context is mandatory analysis data, not a query
            # expansion. Routing/ranking continue to see the actual question.
            if not isinstance(project_context, str) or not project_context.strip():
                raise ValueError('Invalid project context')
            validate_invite_question(project_context)
            if request.use_memory or request.use_conversation_context:
                raise ValueError('Project task context must be explicit')
        task_review_packet, task_review = None, None
        if task_requirements is not None and (project_context is None or project_instance_id is None):
            raise ValueError('Task requirements require an explicit project context')
        if project_context is not None:
            task_requirements = (build_task_requirements(question, input_kind='personal' if personal else 'documents')
                                 if task_requirements is None else task_requirements)
            if (type(task_requirements) is not dict or task_requirements.get('question') != question
                    or task_requirements.get('input_kind') != ('personal' if personal else 'documents')):
                raise ValueError('Task requirements differ from the current approved question')
            task_review_packet = render_task_requirements(task_requirements)
        trial_contract = (trial_contract_packet() if requests_trial_delivery(
            question, literal=requested_mode == 'evidence') else None)
        # Cheap current-task routing precedes optional resource loading. This
        # method-reference lane is independent of experimental graph discovery.
        knowledge_support, knowledge_route = await asyncio.to_thread(
            select_task_knowledge, question, skill_mode, literal=requested_mode == 'evidence')
        insight = InsightRuntime(question, knowledge_support,
                                 mode='off' if skill_mode == 'off' else insight_mode,
                                 literal=requested_mode == 'evidence')
        run.checkpoint()
        proposal = insight.on_stage('before_retrieval', {
            'sources': [], 'allow_supplemental_retrieval': self.context_assistance_enabled,
            'supplement_rounds': 0, 'task_revision': 1})
        run.emit('insight.proposed', 'before_retrieval',
                 {'activation_count': len(proposal.get('actions', []))})
        capacity = await asyncio.to_thread(self.store.answer_capacity, sid)
        if not capacity["available"]:
            from .research_store import ResearchLimitError
            raise ResearchLimitError("Answer metadata capacity reached")
        documents = [] if personal else await asyncio.to_thread(self.store.list_documents, sid)
        scoped_chunk_ids = None
        if document_ids is not None:
            if (not isinstance(document_ids, (list, tuple)) or not (0 if personal else 1) <= len(document_ids) <= 20
                    or any(not isinstance(value, str) or not re.fullmatch(
                        r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', value)
                           for value in document_ids)
                    or len(set(document_ids)) != len(document_ids)):
                raise ValueError('Invalid task document scope')
            allowed = set(document_ids)
            if not allowed <= {row['id'] for row in documents}:
                raise ResearchNotFoundError('Task document scope is unavailable')
            # Scoped project work must not borrow untracked old chat or memory.
            if request.use_memory or request.use_conversation_context:
                raise ValueError('Scoped task context must be explicit')
            documents = [row for row in documents if row['id'] in allowed]
        chunks = ([] if personal else await asyncio.to_thread(self.all_chunks, sid) if document_ids is None else
                  await asyncio.to_thread(self.all_chunks, sid, document_ids=document_ids))
        if document_ids is not None:
            scoped_chunk_ids = {row['id'] for row in chunks}
        selection = ({'memories': explicit_memories, 'omitted_count': 0, 'strategy': 'explicit_task_memory'} if personal else
                     await asyncio.to_thread(self.memory_candidates, sid, question, memory_as_of) if request.use_memory
                     else {"memories": [], "omitted_count": 0, "strategy": "disabled"})
        memories = selection["memories"]
        # Recent history is bounded plain data, never a recovered full transcript.
        raw_history = await asyncio.to_thread(manager.get_history, sid) if request.use_conversation_context else []
        if raw_history:
            # The visible history and model context must share revocation rules.
            # Never re-send an old hidden proof/quote through the next prompt.
            raw_history = await asyncio.to_thread(self.hydrate_history, sid, raw_history[-6:])
            raw_history = [{**item, 'content': '[历史回答的依据已撤回或无法核对，本次不复用该内容。]'}
                           if item.get('role') == 'assistant' and any(
                               row.get('code') in {'source_revoked', 'provenance_unavailable', 'memory_context_revoked'}
                               for row in item.get('citation_validation', {}).get('warnings', [])) else item
                           for item in raw_history]
            # A forgotten value may have occurred in both sides of an old
            # exchange. Do not resend its paired user turn as an indirect bypass.
            for index, item in enumerate(raw_history):
                if (item.get('role') == 'assistant' and any(
                        row.get('code') == 'memory_context_revoked'
                        for row in item.get('citation_validation', {}).get('warnings', []))
                        and index and raw_history[index - 1].get('role') == 'user'):
                    raw_history[index - 1] = {**raw_history[index - 1],
                        'content': '[这轮历史使用的记忆已失效，本次不复用原始上下文。]'}
        history_dependency_ids = [item['id'] for item in raw_history
            if item.get('role') == 'assistant' and isinstance(item.get('id'), str)
            and item.get('content') not in {WITHDRAWN_MEMORY, WITHDRAWN_ANALYSIS}
            and not any(row.get('code') in {'memory_context_revoked', 'provenance_unavailable', 'source_revoked'}
                        for row in item.get('citation_validation', {}).get('warnings', []))]
        history_parent_ids = [item['id'] for item in raw_history if item.get('id') in history_dependency_ids
                              and (item.get('answer_dependency_required') is True
                                   or item.get('used_event_count', 0) > 0)]
        history = bounded_invite_history(raw_history)
        history_text = json.dumps(history, ensure_ascii=False)
        if len(history_text.encode("utf-8")) > 4000:
            history = history[-2:]
            history = [{**item, "content": item["content"].encode("utf-8")[:1200].decode("utf-8", "ignore")} for item in history]
        mode = "explore" if detect_hypothesis_intent(question) else "evidence"
        # One hybrid ranking. Formatting budget fallback never scans/ranks
        # documents again or schedules the same auxiliary job a second time.
        search_question = retrieval_question(question, history)
        event_context, event_obligations, event_supplement = None, None, None
        event_bindings, event_receipt = None, None
        if (self.context_assistance_enabled and self.event_reasoning_enabled
                and requested_mode != 'evidence' and memory_as_of is None):
            # The event graph is a current, scoped index of assertions. It is
            # never authority to borrow another task's documents or notes.
            event_context = await asyncio.to_thread(self.events.retrieve_context, sid, search_question,
                allowed_source_ids=[row['id'] for row in chunks],
                allowed_memory_ids=([row['id'] for row in explicit_memories] if personal else
                                    None if request.use_memory else []))
            if event_context.get('events'):
                event_bindings = event_context['bindings']
                event_obligations = build_event_obligations(event_context)
            else:
                event_context = None
        run.checkpoint()
        run.emit('stage.started', 'retrieval')
        semantic = await self._semantic_operation(sid, chunks, search_question)
        semantic_trace = semantic['trace']
        scores = semantic['scores'] if semantic_trace.get('status') == 'ready' else None
        spans = semantic.get('spans', {}) if scores is not None else None
        try:
            prepared_corpus = await asyncio.to_thread(self._corpus_cache.lookup, chunks, namespace=sid)
            lexical_features = {'status': 'indexed' if prepared_corpus is not None else 'cold_full_scan',
                                'corpus_feature_bytes': prepared_corpus.feature_bytes if prepared_corpus is not None else 0,
                                'current_source_view_rechecked': True, 'answer_cache': False, 'provider_calls': 0}
        except CorpusIndexTooLarge:
            # Memory pressure must not turn into reduced source coverage.
            prepared_corpus = None
            lexical_features = {'status': 'full_scan_memory_budget', 'current_source_view_rechecked': True,
                                'answer_cache': False, 'provider_calls': 0}
        plan = await asyncio.to_thread(build_retrieval_plan, search_question, chunks, memories,
                                      mode=mode, semantic_scores=scores, semantic_spans=spans,
                                      prepared_corpus=prepared_corpus)
        if chunks and semantic_trace.get('status') not in {'ready', 'disabled', 'empty_corpus'}:
            plan['gaps'].append({'code': 'semantic_retrieval_unavailable',
                                'message': '本地语义检索暂不可用，已降级为词面检索；跨语言和同义表达可能遗漏。'})
        elif (semantic_trace.get('query_truncated') or semantic_trace.get('corpus_scan_limited')
              or semantic_trace.get('limited_chunks')):
            plan['gaps'].append({'code': 'semantic_retrieval_windowed',
                                'message': '本地语义检索达到扫描窗口上限，不代表已完整覆盖问题或全部资料。'})
        assistance = empty_assistance(plan['sources'])
        fallback_filter_trace = None
        lexical_coverage_trace = None
        if not self.context_assistance_enabled:
            assistance = empty_assistance(plan['sources'], reason='operator_disabled')
        elif requested_mode != 'evidence' and self._evidence_reranker is not None and self._semantic is not None:
            from .research_faceted_assist import build_faceted_assistance
            work = await self._evidence_scheduler.run(lambda: build_faceted_assistance(
                search_question, plan['sources'], chunks, self._semantic, self._evidence_reranker,
                session_id=self._semantic_namespace + '::' + sid, primary_semantic=semantic,
                query_translator=self._query_translator))
            assistance = work.value if work.status == 'completed' else empty_assistance(
                plan['sources'], reason='evidence_scheduler_' + work.status)
            assistance['reasoning']['trace']['scheduler_status'] = work.status
            if work.trace is not None:
                assistance['reasoning']['trace']['scheduler_timing'] = dict(work.trace)
            if (work.status in {'timed_out', 'busy', 'failed'} or
                    assistance['reasoning']['trace'].get('skipped_reason') == 'reranker_unavailable'):
                # Only the existing bounded pack is inspected, without touching
                # a timed-out worker's input/result or adding retrieval/model work.
                cleaned = filter_fallback_sources(search_question, plan['sources'])
                fallback_filter_trace = cleaned['trace']
                if fallback_filter_trace['filtered_count']:
                    plan['sources'] = plan['selected_chunks'] = cleaned['sources']
                    assistance['sources'] = cleaned['sources']
                    # Deletion cannot close an existing retrieval gap. Only
                    # recompute the property of the retained excerpt geometry.
                    plan['gaps'] = [row for row in plan['gaps'] if row.get('code') != 'excerpt_truncated']
                    plan['gaps'].append({'code': 'fallback_metadata_filtered',
                        'message': '辅助筛选不可用；仅从原有来源移除明确的纯元信息，未验证其余证据的相关性或完整性。'})
                    if any(row.get('truncated') for row in cleaned['sources']):
                        plan['gaps'].append({'code': 'excerpt_truncated',
                            'message': '部分依据仅为连续节选，限定条件可能在节选之外。'})
                    if not cleaned['sources'] and not any(row.get('code') == 'insufficient_evidence' for row in plan['gaps']):
                        plan['gaps'].append({'code': 'insufficient_evidence',
                            'message': '现有来源仅有明确的元信息，没有可用于本次内容问题的原文片段。'})
                    assistance['reasoning']['gaps'] = list(plan['gaps'])
                    assistance['reasoning']['trace'].update(
                        selected_source_count=len(cleaned['sources']),
                        source_bytes=sum(len(row['content'].encode()) for row in cleaned['sources']))
            if assistance['reasoning'].get('status') == 'evidence_selected':
                # Replacing the source set invalidates old lexical-coverage
                # hints. Do not tell the model a now-supplied term is absent.
                plan['gaps'] = [row for row in plan['gaps'] if row.get('code') not in {
                    'uncovered_query_terms', 'documents_not_selected', 'excerpt_truncated',
                    'complete_statement_exceeds_budget', 'insufficient_evidence'}]
                plan['gaps'].extend(assistance['reasoning'].get('gaps', []))
                # Budget fallback keeps the selected evidence, never silently
                # restoring the old background-heavy source set.
                plan['sources'] = assistance['sources']
                plan['selected_chunks'] = assistance['sources']
        elif requested_mode != 'evidence':
            # Legacy reasoning_mode values are receipt compatibility only.
            # Ordinary questions, not a user-facing mode, route the assistance.
            from .research_lexical_coverage import select_lexical_coverage
            pointer_ids = list(dict.fromkeys(source for memory in memories for source in memory.get('source_chunk_ids', [])))[:8]
            pointers = await asyncio.to_thread(self.store.source_chunks, sid, pointer_ids) if pointer_ids else []

            def organize_literal_windows():
                # Both passes share the existing single nonqueued worker and
                # foreground deadline. Late coverage results cannot leak into
                # the answer, or leave an unbounded extra thread per request.
                coverage = select_lexical_coverage(search_question, plan['sources'], chunks)
                def candidates(operator, seeds):
                    claims, gaps = self.evidence.candidates(sid, seeds, operator)
                    if scoped_chunk_ids is not None:
                        kept = [claim for claim in claims if set(claim.get('source_ids', [])) <= scoped_chunk_ids]
                        if len(kept) != len(claims):
                            gaps = [*gaps, {'code': 'document_scope_omitted',
                                'message': '本步骤仅使用声明的资料范围，其他资料未参与推导。'}][:8]
                        claims = kept
                    return claims, gaps
                result = build_context_assist(
                    search_question, coverage['sources'], chunks,
                    candidates,
                    candidate_sources=pointers, semantic_scores=scores, semantic_spans=spans)
                return {**result, '_lexical_coverage': coverage}

            work = await self._context_scheduler.run(organize_literal_windows)
            if work.status == 'completed':
                assistance = work.value
                coverage = assistance.pop('_lexical_coverage', None)
            else:
                # A timeout does not retry retrieval. Inspect only the existing
                # eight-source packet for unambiguous whole-window metadata.
                clean = filter_fallback_sources(search_question, plan['sources'])
                coverage = {'sources': clean['sources'], 'gaps': [], 'trace': {
                    'algorithm': 'literal_window_coverage_v1', 'status': 'unavailable',
                    'reason': 'scheduler_' + work.status, 'metadata_filter': clean['trace'],
                    'model_calls': 0, 'provider_calls': 0, 'llm_tokens': 0}}
                assistance = empty_assistance(clean['sources'], reason='scheduler_' + work.status)
            if coverage is not None:
                lexical_coverage_trace = coverage['trace']
            if coverage is not None and coverage['sources'] != plan['sources']:
                plan['sources'] = plan['selected_chunks'] = coverage['sources']
                # Source replacement invalidates old pack-specific lexical
                # hints, but cannot eliminate semantic/scan-window caveats.
                plan['gaps'] = [row for row in plan['gaps'] if row.get('code') not in {
                    'uncovered_query_terms', 'documents_not_selected', 'excerpt_truncated',
                    'complete_statement_exceeds_budget', 'insufficient_evidence'}]
                if any(row.get('truncated') for row in coverage['sources']):
                    plan['gaps'].append({'code': 'excerpt_truncated',
                        'message': '部分依据仅为连续节选，限定条件可能在节选之外。'})
                if not coverage['sources']:
                    plan['gaps'].append({'code': 'insufficient_evidence',
                        'message': '当前来源仅有元信息，没有可用于本次内容问题的原文片段。'})
            if coverage is not None:
                existing_codes = {row.get('code') for row in plan['gaps']}
                plan['gaps'].extend(row for row in coverage['gaps'] if row.get('code') not in existing_codes)
            assistance['reasoning']['trace']['scheduler_status'] = work.status
            if work.trace is not None:
                assistance['reasoning']['trace']['scheduler_timing'] = dict(work.trace)
        discovery_trace = None
        if (self.context_assistance_enabled and requested_mode != 'evidence' and
                (graph_retrieval_intent(search_question) or detect_hypothesis_intent(question))):
            # One bounded local pass AFTER source reranking; no provider work,
            # no graph fan-out for ordinary/literal questions, no task backlog.
            work = await self._context_scheduler.run(lambda p=plan, a=assistance: build_discovery_context(
                question, p, a, chunks, self._graph_retrieval, search_question=search_question))
            if work.status == 'completed':
                plan, assistance = work.value['plan'], work.value['assistance']
                discovery_trace = assistance['reasoning']['trace'].get('graph_retrieval')
            assistance['reasoning']['trace']['discovery_scheduler'] = {
                'status': work.status, 'model_calls': 0,
                **({'timing': work.trace} if work.trace is not None else {})}
        run.checkpoint()
        task_queries = ([value.encode('utf-8')[:400].decode('utf-8', errors='ignore')
                         for value in task_requirements['acceptance_criteria'][:4]]
                        if task_requirements is not None else [])
        if event_context is not None or task_queries:
            # Link witnesses and counterevidence get the existing two spare
            # source slots BEFORE generic Insight extras. This does not evict
            # primary evidence or increase the shared 10-source/12KB budget.
            query_needs = {**(event_obligations or {}), 'suggested_queries': list(dict.fromkeys([
                *(event_obligations or {}).get('suggested_queries', [])[:4], *task_queries]))[:8]}
            event_supplement = await asyncio.to_thread(supplement_event_gaps, search_question,
                assistance['sources'], chunks, query_needs, context=event_context)
            assistance = {**assistance, 'sources': event_supplement['sources']}
            plan = refresh_query_coverage(plan, event_supplement['sources'])
        supplement = {'status': 'skipped', 'reason': 'no_actions', 'retrieval_passes': 0,
                      'model_calls': 0, 'added_source_ids': [], 'action_ids': []}
        if proposal.get('actions'):
            work = await self._context_scheduler.run(lambda: _supplement_insight_evidence(
                search_question, assistance['sources'], chunks, proposal))
            if work.status == 'completed':
                supplement = work.value['receipt']
                # Baseline and assisted evidence have distinct fallback
                # semantics. Append new evidence to each without replacing
                # the primary set with an assisted selection on a no-op.
                added = [row for row in work.value['sources'] if row['id'] in supplement['added_source_ids']]
                if added:
                    plan['sources'] = plan['selected_chunks'] = _append_insight_sources(plan['sources'], added)
                assistance['sources'] = work.value['sources']
                if supplement['added_source_ids']:
                    plan['gaps'] = [row for row in plan['gaps'] if row.get('code') not in {
                        'uncovered_query_terms', 'insufficient_evidence', 'documents_not_selected', 'excerpt_truncated'}]
                    unselected = {row['document_id'] for row in chunks} - {row['document_id'] for row in plan['sources']}
                    if unselected:
                        plan['gaps'].append({'code': 'documents_not_selected', 'count': len(unselected),
                            'message': '部分资料未进入本次有限上下文；补查没有穷尽全部资料。'})
                    if any(row.get('truncated') for row in plan['sources']):
                        plan['gaps'].append({'code': 'excerpt_truncated',
                            'message': '部分依据仅为连续节选，限定条件可能在节选之外。'})
            else:
                supplement.update(reason='scheduler_' + work.status)
            supplement['scheduler_status'] = work.status
        # Refresh even when selection/assistance was a no-op: diagnostics from
        # the initial pack must not survive solely because source IDs match.
        # The formatter repeats this cheap check for its final excerpt variant.
        plan = refresh_query_coverage(plan, plan['sources'])
        event_prompt, event_source_guards = None, None
        if event_context is not None:
            # All obligations remain host-side; only a bounded, task-relevant
            # review frontier consumes model attention this turn. Deferred
            # checks remain unknown, never silently counted as complete.
            event_context = prepare_event_frontier(search_question, event_context, assistance['sources'])
            event_obligations = build_event_obligations(event_context)
            event_prompt = render_event_reasoning(event_context, event_obligations)
            event_source_guards = event_source_window_guards(event_context, event_prompt)
        run.emit('stage.completed', 'retrieval', {'source_count': len(assistance['sources'])})
        run.checkpoint()
        run.emit('stage.started', 'analysis')
        fitted = fit_research_prompt(question, plan, history, assistance,
                                     system_policy=knowledge_skill_system_policy(bool(knowledge_support and knowledge_support['prompt'])),
                                     memory_as_of=memory_as_of, knowledge_support=knowledge_support,
                                     insight_runtime=insight, trial_contract=trial_contract,
                                     **({'event_context': event_prompt, 'event_contract': EVENT_RESPONSE_POLICY,
                                         'event_source_guards': event_source_guards}
                                        if event_prompt is not None else {}),
                                     **({'task_review_context': task_review_packet,
                                         'task_review_contract': task_review_contract()}
                                        if task_review_packet is not None else {}),
                                     **({'project_context': project_context} if project_context is not None else {}))
        plan, prompt, reasoning = fitted['plan'], fitted['prompt'], fitted['reasoning']
        if fitted.get('history_included'):
            parent_answer_ids = list(dict.fromkeys([*parent_answer_ids, *history_parent_ids]))
        knowledge_support = fitted['knowledge_support']
        if discovery_trace is not None:
            reasoning['trace']['graph_retrieval'] = fit_graph_receipt(discovery_trace, plan['sources'])
        if fallback_filter_trace is not None:
            # Keep the receipt even if optional assistance was shed for budget.
            reasoning['trace']['fallback_metadata_filter'] = fallback_filter_trace
        if lexical_coverage_trace is not None:
            reasoning['trace']['lexical_evidence_coverage'] = {
                **lexical_coverage_trace, 'scope': 'prepacking_selection_not_final_coverage',
                'final_context_revalidated_by': 'final_need_ledger'}
        reasoning['trace']['retrieval_context'] = 'previous_user_turn' if search_question != question else 'current_question'
        reasoning['trace']['semantic_retrieval'] = semantic_trace
        reasoning['trace']['lexical_features'] = lexical_features
        reasoning['trace']['knowledge_skill_router'] = knowledge_route
        if task_requirements is not None and event_supplement is not None:
            reasoning['trace']['task_requirement_retrieval'] = event_supplement['trace']
            if event_context is None:
                reasoning['trace']['context_budget']['retrieval_passes'] += 1
        if knowledge_support:
            reasoning['trace']['knowledge_skill_retrieval'] = knowledge_support['trace']
        refs = plan['sources']
        # Receipts describe the final packet, never just pre-fit candidates.
        kept_ids = {row['id'] for row in refs}
        supplement['kept_source_ids'] = [identifier for identifier in supplement['added_source_ids'] if identifier in kept_ids]
        insight_receipt = {
            'version': 1, 'status': 'ready' if (fitted.get('insight_included') or supplement['kept_source_ids']) else 'inactive',
            'mode': insight_mode, 'enabled': skill_mode != 'off' and insight_mode != 'off' and requested_mode != 'evidence',
            'schema_versions': insight.schema_refs(), 'source_ids': [row['id'] for row in refs],
            'supplement': supplement, 'analysis_included': bool(fitted.get('insight_included')),
            'analysis_actions': [{'id': item['id'], 'schema_id': item.get('schema_id'),
                                  'schema_version': item.get('schema_version'), 'status': 'input_included'}
                                 for item in fitted.get('insight_proposal', {}).get('actions', [])]
                                if fitted.get('insight_included') else [],
            'model_calls': 0, 'semantic_entailment_verified': False,
            'method_effectiveness_verified': False,
        }
        reasoning['trace']['insight_runtime'] = insight_receipt
        reasoning['trace']['context_budget']['retrieval_passes'] += supplement['retrieval_passes']
        run.emit('stage.completed', 'analysis', {'input_bytes': fitted['budget']['input_bytes']})
        # Retain the old literal-extract API for accepted/legacy clients; it is
        # not advertised or sent by the normal chat/writing interface.
        algorithm_answer = render_evidence(refs) if requested_mode == 'evidence' and refs else None
        memory_bindings = []
        pending = any(doc["status"] in {"queued", "processing"} for doc in documents)
        explicit_document = requires_document_evidence(search_question)
        if algorithm_answer is not None:
            answer, answer_mode = algorithm_answer, 'evidence'
        elif requested_mode == 'evidence' or (not refs and explicit_document):
            answer = ("资料还在解析中，完成后再问这个问题；现在没有可引用的原文。" if pending else
                      "当前项目没有检索到与这个问题足够相关的原文片段。我不能据此确认结论。请补充资料，或用论文中的具体术语重新提问。")
            answer_mode = "insufficient_evidence"
        else:
            if not model:
                raise ValueError("Chat model is not configured")
            retained_memory_ids = {row['id'] for row in plan['confirmed_memories']}
            retained_memories = [row for row in memories if row['id'] in retained_memory_ids]
            if personal and retained_memory_ids != set(task_context_refs):
                raise ResearchNotFoundError('Explicit personal context does not fit whole into this request')
            if len(retained_memories) != len(retained_memory_ids):
                raise ResearchNotFoundError('Selected memory revision unavailable')
            memory_bindings = await asyncio.to_thread(self.memory_dependencies.collect, sid,
                retained_memories, as_of=memory_as_of,
                history_ids=history_dependency_ids if fitted.get('history_included') else [])
            if memory_bindings and not await asyncio.to_thread(self.memory_dependencies.live, sid, memory_bindings):
                raise ResearchNotFoundError('Memory context changed before generation')
            if parent_answer_ids and not await asyncio.to_thread(self.answer_dependencies.parents_live, sid, parent_answer_ids):
                raise ResearchNotFoundError('Parent analysis changed before generation')
            if project_parent_task_refs and not await asyncio.to_thread(
                    self.answer_dependencies.task_origins_live, sid, project_parent_task_refs):
                raise ResearchNotFoundError('Parent task version changed before generation')
            if event_bindings and not await asyncio.to_thread(self.event_dependencies.live, sid, event_bindings):
                raise ResearchNotFoundError('Event context changed before generation')
            # Dispatch exactly the messages whose complete byte budget and
            # evidence receipt were checked, independent of provider/model.
            with run.stage('generation'), completion_output_budget(fitted['budget']['max_output_tokens']):
                answer = str(await model(fitted['messages'][1]['content'],
                                         system_prompt=fitted['messages'][0]['content']) or "").strip()
            if not answer or len(answer.encode("utf-8")) > 100000:
                raise ValueError("Chat model returned an invalid answer")
            answer_mode = mode if refs else "memory" if plan["confirmed_memories"] else "general"
        model_called = algorithm_answer is None and answer_mode != 'insufficient_evidence'
        raw_event_response = answer
        if event_context is not None and model_called:
            answer, event_receipt = parse_event_response(answer, event_context, refs)
            reasoning['trace']['event_reasoning'] = event_receipt
            reasoning['trace']['event_retrieval'] = event_context['trace']
            reasoning['trace']['event_frontier'] = event_context['_frontier_trace']
            reasoning['trace']['event_supplement'] = event_supplement['trace']
            reasoning['trace']['context_budget']['retrieval_passes'] += 1
        elif event_context is not None:
            # An insufficient-evidence/literal response did not use events.
            event_bindings = None
        if task_review_packet is not None and model_called:
            answer, _ = parse_task_response(raw_event_response, task_requirements, refs)
        trial_receipt = None
        if trial_contract and model_called:
            from .research_trial_contract import parse_trial_response
            from .research_trial_renderer import render_trial_response
            trial_response = answer
            if event_receipt is not None or task_review_packet is not None:
                trial_response = event_trial_response(raw_event_response)
            parsed_trial = parse_trial_response(trial_response)
            answer = render_trial_response(parsed_trial)
            trial_receipt = trial_delivery_receipt(parsed_trial, [source['id'] for source in refs],
                                                   trial_contract['schema_sha256'])
            reasoning['trace']['trial_delivery'] = trial_receipt
        if task_review_packet is not None and model_called:
            _, task_review = parse_task_response(raw_event_response, task_requirements, refs, final_answer=answer)
            reasoning['trace']['task_review'] = task_review
        run.checkpoint()
        run.emit('stage.started', 'delivery')
        knowledge = knowledge_receipt(knowledge_support, included=fitted['knowledge_skill_included'],
                                      answer=answer, model_called=model_called)
        knowledge_used = bool(knowledge and knowledge['status'] == 'ready')
        validation = citation_validation(answer, [source['label'] for source in refs],
                                        knowledge_labels=[row['label'] for row in knowledge['sources']] if knowledge_used else [])
        if event_bindings:
            try:
                events_current = await asyncio.to_thread(self.event_dependencies.live, sid, event_bindings)
            except Exception:
                events_current = False
            if not events_current:
                answer = WITHDRAWN_ANALYSIS
                event_receipt = None
                reasoning['trace'].pop('event_reasoning', None)
                validation = {**validation, 'valid': False, 'warnings': [*validation['warnings'],
                    {'code': 'provenance_unavailable', 'message': '生成期间事件或其来源已改变，派生回答已收起。'}]}
        if parent_answer_ids and (not await asyncio.to_thread(self.answer_dependencies.parents_live, sid, parent_answer_ids)
                or (project_parent_task_refs and not await asyncio.to_thread(
                    self.answer_dependencies.task_origins_live, sid, project_parent_task_refs))):
            answer = WITHDRAWN_ANALYSIS
            validation = {**validation, 'valid': False, 'warnings': [*validation['warnings'],
                {'code': 'provenance_unavailable', 'message': '前序分析的依据已改变，派生回答已收起。'}]}
        if memory_bindings:
            try:
                memory_current = await asyncio.to_thread(self.memory_dependencies.live, sid, memory_bindings)
            except Exception:
                memory_current = False
            if not memory_current:
                answer = WITHDRAWN_MEMORY
                validation = {**validation, 'valid': False, 'warnings': [*validation['warnings'],
                    {'code': 'memory_context_revoked', 'message': '生成期间记忆已改变或无法核验，回答已收起。'}]}
        # Deletion can race a paid model call. Do not resurrect revoked quotes,
        # lose the answer or bill another call to repair provenance.
        try:
            current = await asyncio.to_thread(self.store.source_chunks, sid, [source["id"] for source in refs]) if refs else []
        except Exception:
            current = []
            refs = []
            validation = {**validation, "valid": False, "warnings": [*validation["warnings"],
                          {"code": "provenance_unavailable", "message": "来源状态未能重新确认，回答请核对；不会重新请求模型。"}]}
        available = {row["id"] for row in current}
        if any(source["id"] not in available for source in refs):
            refs = [source for source in refs if source["id"] in available]
            validation = {**validation, "valid": False, "warnings": [*validation["warnings"],
                          {"code": "source_revoked", "message": "生成期间部分来源已被删除，相关回答请重新核对。"}]}
        delivery = insight.on_stage('before_delivery', {
            'sources': refs, 'answer': answer, 'task_revision': 1,
            'active_method_labels': [row['label'] for row in knowledge['sources']] if knowledge_used else [],
            'provenance_valid': not any(row['code'] in {'source_revoked', 'provenance_unavailable'}
                                        for row in validation['warnings']),
            'executed_action_ids': supplement['action_ids'],
            'supplement_rounds': supplement['retrieval_passes']})
        # These are deterministic citation/structure checks, not a second
        # model critic, factual verdict or a claim that the writer obeyed.
        insight_receipt['delivery'] = delivery
        insight_receipt['delivery_actions'] = [
            {'id': item['id'], 'schema_id': item.get('schema_id'),
             'schema_version': item.get('schema_version'), 'status': 'checked'}
            for item in delivery.get('actions', []) if item.get('kind') == 'delivery_check']
        if not model_called:
            insight_receipt.update(status='inactive', analysis_included=False, analysis_actions=[])
        reasoning = hydrate_reasoning(reasoning, refs)
        if algorithm_answer is not None and requested_mode == 'evidence':
            answer = render_evidence(refs)
        if ((reasoning.get('status') == 'source_unavailable' and reasoning.get('mode') == 'context_assist')
                or ((knowledge_used or event_bindings) and any(row['code'] in {'source_revoked', 'provenance_unavailable'}
                                          for row in validation['warnings']))):
            # Do not ask a model to repair or try to semantically redact a
            # generated passage after a premise is withdrawn. Hide it whole.
            answer = WITHDRAWN_ANALYSIS
            if knowledge:
                knowledge = unavailable_knowledge(knowledge)
        if answer in {WITHDRAWN_ANALYSIS, WITHDRAWN_MEMORY}:
            event_receipt = None
            task_review = None
            reasoning['trace'].pop('event_reasoning', None)
            reasoning['trace'].pop('task_review', None)
        reasoning['trace']['answer_model_calls'] = int(model_called)
        provenance_required = bool(trial_receipt) or knowledge_used or (reasoning.get('mode') == 'context_assist' and bool(
            reasoning.get('conclusions') or reasoning['trace'].get('hypothesis_support', {}).get('source_ids')
            or reasoning['trace'].get('graph_retrieval', {}).get('kept_source_ids')
            or reasoning.get('status') == 'source_unavailable'
            or insight_receipt.get('status') == 'ready'))
        provenance_required = provenance_required or bool(memory_bindings) or bool(parent_answer_ids) or bool(event_bindings) or bool(task_review)
        audit = audit_answer(answer, refs)
        run.checkpoint()
        run.emit('stage.completed', 'delivery', {'model_calls': int(model_called), 'source_count': len(refs)})
        reasoning['trace']['harness'] = run.snapshot()
        if project_instance_id is not None:
            session = manager.get_session(sid)
            if not session or session.get('instance_id') != project_instance_id:
                # An old paid step must never append its body to a newly
                # created project that happens to reuse the visible number.
                raise ResearchNotFoundError('Task project instance changed during generation')
        manager.add_message(sid, "user", question)
        # The marker is stored with the answer, BEFORE provenance persistence.
        # A later metadata failure must not turn derived analysis into an
        # unmarked historical answer after restart or on the next prompt.
        answer_id = manager.add_message(sid, "assistant", answer,
            **({'research_memory_required': True} if memory_bindings else {}),
            **({'research_provenance_required': True} if provenance_required else {}))
        used_memory_count = 0 if algorithm_answer is not None or answer_mode == "insufficient_evidence" else len(plan["confirmed_memories"])
        context_metadata = {"retrieval_gaps": plan["gaps"],
                            "used_memory_count": used_memory_count,
                            "memory_selection": {"strategy": selection["strategy"],
                                                 "omitted_count": selection["omitted_count"],
                                                 "selected_count": used_memory_count}}
        if memory_bindings:
            # Atomic with answer metadata, before the separate lineage write:
            # even partial persistence must not launder a context-only answer.
            context_metadata['memory_dependency_required'] = True
        if parent_answer_ids:
            context_metadata['answer_dependency_required'] = True
        if event_bindings:
            context_metadata.update(self.event_dependencies.context_metadata(event_bindings))
        try:
            await asyncio.to_thread(self.store.record_answer, sid, answer_id,
                                   [{key: source[key] for key in ("id", "label", "excerpt_start", "excerpt_end", "truncated")} for source in refs],
                                   validation, answer_mode, context_metadata=context_metadata)
            await asyncio.to_thread(self.memory_dependencies.record, sid, answer_id, memory_bindings)
            if event_bindings:
                await asyncio.to_thread(self.event_dependencies.record, sid, answer_id, event_bindings)
            await asyncio.to_thread(self.answer_dependencies.record, sid, answer_id, parent_answer_ids,
                                    **({'task_refs': project_parent_task_refs} if project_parent_task_refs else {}))
            await asyncio.to_thread(self.evidence.record, sid, answer_id, reasoning=reasoning, audit=audit,
                                    memory_as_of=memory_as_of, reasoning_mode=requested_mode,
                                    **({'knowledge_skill': knowledge} if knowledge else {}))
        except Exception:
            # History already contains the real answer. A metadata-only outage
            # must not turn a completed paid call into an automatic resubmission.
            refs = []
            reasoning = hydrate_reasoning(reasoning, [])
            if trial_receipt:
                reasoning['trace']['trial_delivery'] = unavailable_trial_receipt()
            if provenance_required:
                answer = WITHDRAWN_ANALYSIS
                event_receipt = None
                task_review = None
                reasoning.get('trace', {}).pop('event_reasoning', None)
                reasoning.get('trace', {}).pop('task_review', None)
            if knowledge:
                knowledge = unavailable_knowledge(knowledge)
            audit = audit_answer(answer, [])
            validation = {**validation, "valid": False, "warnings": [*validation["warnings"],
                          {"code": "provenance_unavailable", "message": "回答已完成，但依据记录未能保存；不要把它当作已核对的资料结论。"}]}
        if refs and lexical_features['status'] == 'cold_full_scan':
            # Preparing features never waits in this answer or queues behind
            # another preparation. No provider call; busy means normal ranking.
            self._warm_corpus(sid, chunks)
        result = {"answer": answer, "answer_id": answer_id, "sources": refs,
                "answer_mode": answer_mode, "citation_validation": validation,
                **context_metadata, "context_windowed": True,
                "research_core": True, "memory_write_enabled": True,
                "research_provenance_required": provenance_required,
                **({'research_memory_required': True} if memory_bindings else {}),
                'reasoning': reasoning, 'claim_audit': audit, 'memory_as_of': memory_as_of, 'reasoning_mode': requested_mode,
                **({'trial_delivery': reasoning['trace'].get('trial_delivery', unavailable_trial_receipt())} if trial_receipt else {}),
                **({'knowledge_skill': knowledge} if knowledge else {}),
                **({'event_reasoning': event_receipt} if event_receipt is not None else {}),
                **({'task_review': task_review} if task_review is not None else {}),
                }
        if event_bindings:
            try:
                events_current = (await asyncio.to_thread(self.event_dependencies.states, sid, [answer_id])).get(answer_id) is True
            except Exception:
                events_current = False
            if not events_current:
                result.update(answer=WITHDRAWN_ANALYSIS, sources=[],
                    citation_validation={**result['citation_validation'], 'valid': False,
                        'warnings': [*result['citation_validation']['warnings'],
                            {'code': 'provenance_unavailable', 'message': '事件依据已改变或无法核验，回答已收起。'}]})
                for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    result.pop(key, None)
        if memory_bindings:
            # Recheck after persistence too: a correction may race metadata
            # writes, not only the provider. No automatic paid repair/retry.
            try:
                memory_current = await asyncio.to_thread(self.memory_dependencies.live, sid, memory_bindings)
            except Exception:
                memory_current = False
            if not memory_current:
                result.update(answer=WITHDRAWN_MEMORY, sources=[], used_memory_count=0,
                    citation_validation={**result['citation_validation'], 'valid': False,
                        'warnings': [*result['citation_validation']['warnings'],
                            {'code': 'memory_context_revoked', 'message': '记忆已改变或无法核验，回答已收起。'}]})
                for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    result.pop(key, None)
        if parent_answer_ids:
            try:
                parents_current = (await asyncio.to_thread(self.answer_dependencies.states, sid, [answer_id])).get(answer_id) is True
            except Exception:
                parents_current = False
            if not parents_current:
                result.update(answer=WITHDRAWN_ANALYSIS, sources=[],
                    citation_validation={**result['citation_validation'], 'valid': False,
                        'warnings': [*result['citation_validation']['warnings'],
                            {'code': 'provenance_unavailable', 'message': '前序分析依据已改变，派生回答已收起。'}]})
                for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    result.pop(key, None)
        return result

    def hydrate_history(self, sid: str, messages: list[dict]) -> list[dict]:
        ids = [m["id"] for m in messages[-200:] if m.get("role") == "assistant" and isinstance(m.get("id"), str)]
        metadata = self.store.answer_metadata(sid, ids)
        # Request-local batches avoid a connection per citation. Do not reuse
        # these rows across requests: deletion and provenance must be rechecked.
        referenced = list(dict.fromkeys(ref['id'] for item in metadata.values()
                                        for ref in item['source_refs'] if ref['available']))
        current = self.store.answer_source_chunks(sid, referenced)
        source_packets, unavailable_counts = {}, {}
        for answer_id, item in metadata.items():
            sources, unavailable = [], 0
            for ref in item['source_refs']:
                row = current.get(ref['id']) if ref['available'] else None
                if row is None:
                    unavailable += 1
                    continue
                start, end = ref.get('excerpt_start', 0), ref.get('excerpt_end', len(row['content']))
                sources.append({**{key: value for key, value in row.items() if key != 'text'},
                                'content': row['content'][start:end], 'label': ref['label'],
                                'excerpt_start': start, 'excerpt_end': end,
                                'original_content_length': len(row['content']),
                                'truncated': ref.get('truncated', False)})
            source_packets[answer_id], unavailable_counts[answer_id] = sources, unavailable
        proofs = self.evidence.hydrate_many(sid, source_packets)
        needs_memory_check = any(row.get('research_memory_required') is True for row in messages) or any(
            item.get('used_memory_count', 0) > 0 or item.get('memory_dependency_required') is True
            for item in metadata.values())
        try:
            memory_states = self.memory_dependencies.states(sid, ids) if needs_memory_check else {}
        except Exception:
            memory_states = {}
        parent_required_ids = [identifier for identifier, item in metadata.items()
                               if item.get('answer_dependency_required') is True]
        try:
            parent_states = self.answer_dependencies.states(sid, parent_required_ids) if parent_required_ids else {}
        except Exception:
            parent_states = {}
        event_required_ids = [identifier for identifier, item in metadata.items()
                              if item.get('used_event_count', 0) or item.get('used_event_relation_count', 0)]
        try:
            event_states = self.event_dependencies.states(sid, event_required_ids) if event_required_ids else {}
        except Exception:
            event_states = {}
        result = []
        for message in messages:
            item = metadata.get(message.get("id"))
            if item:
                sources = source_packets[message['id']]
                unavailable = unavailable_counts[message['id']]
                validation = item["validation"]
                if unavailable:
                    validation = {**validation, "valid": False, "warnings": [*validation["warnings"],
                                  {"code": "source_revoked", "message": "部分原文依据已被删除，无法继续核对。"}]}
                message = {**message, "sources": sources, "answer_mode": item["answer_mode"],
                           "citation_validation": validation, "unavailable_source_count": unavailable,
                           **{key: item[key] for key in ("retrieval_gaps", "used_memory_count", "memory_selection", "answer_dependency_required",
                                                        "used_event_count", "used_event_relation_count") if key in item},
                           "research_core": True}
                proof_metadata = proofs.get(message['id'], {})
                # A request-status replay may carry an old response packet.
                # Fresh proof metadata replaces that packet; a missing record
                # must never leave cached quotes/methods visible in other cards.
                for field in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    message.pop(field, None)
                message.update(proof_metadata)
                if message.get('reasoning', {}).get('trace', {}).get('event_reasoning'):
                    message['event_reasoning'] = message['reasoning']['trace']['event_reasoning']
                task_receipt = message.get('reasoning', {}).get('trace', {}).get('task_review')
                if task_receipt is not None:
                    task_receipt = hydrate_task_review(task_receipt, sources, message.get('content', message.get('answer', '')))
                    if task_receipt is None:
                        message['reasoning']['trace'].pop('task_review', None)
                        message['reasoning']['status'] = 'source_unavailable'
                    else:
                        message['task_review'] = task_receipt
                trial_receipt = message.get('reasoning', {}).get('trace', {}).get('trial_delivery')
                if trial_receipt:
                    message['trial_delivery'] = trial_receipt
                withdrawn_literal = unavailable and message.get('reasoning_mode') == 'evidence'
                legacy_proof = (message.get('reasoning', {}).get('status') == 'source_unavailable'
                                and message.get('reasoning', {}).get('mode') != 'context_assist')
                withdrawn_analysis = (message.get('reasoning', {}).get('status') == 'source_unavailable'
                                      and message.get('reasoning', {}).get('mode') == 'context_assist')
                if message.get('research_provenance_required') is True and (unavailable or not proof_metadata):
                    withdrawn_analysis = True
                    message['citation_validation'] = {**validation, 'valid': False, 'warnings': [*validation['warnings'],
                        {'code': 'provenance_unavailable', 'message': '推断依据不可用，本次分析已收起，不会重新请求模型。'}]}
                if withdrawn_analysis:
                    message.pop('event_reasoning', None)
                    message.pop('task_review', None)
                    message.get('reasoning', {}).get('trace', {}).pop('event_reasoning', None)
                    message.get('reasoning', {}).get('trace', {}).pop('task_review', None)
                    if trial_receipt:
                        message['trial_delivery'] = unavailable_trial_receipt()
                    message.update(content=WITHDRAWN_ANALYSIS, claim_audit=audit_answer(WITHDRAWN_ANALYSIS, sources))
                    if message.get('knowledge_skill'):
                        message['knowledge_skill'] = unavailable_knowledge(message['knowledge_skill'])
                    if 'answer' in message:
                        message['answer'] = WITHDRAWN_ANALYSIS
                elif withdrawn_literal or legacy_proof:
                    rewritten = (render_evidence(sources) + '\n\n部分原文已被撤回，已移除对应摘录。' if withdrawn_literal else
                                 render_reasoning(message['reasoning'], sources) + '\n\n部分前提已被撤回，相关推导已收起。')
                    message.update(content=rewritten, claim_audit=audit_answer(rewritten, sources))
                    if 'answer' in message:
                        message['answer'] = rewritten
            elif message.get("research_core") is True or message.get('research_provenance_required') is True:
                message = {**message, "sources": [], "citation_validation": {
                    "valid": False, "entailment_verified": False, "warnings": [{"code": "provenance_unavailable", "message": "依据记录不可用，不能继续核对本次引用。"}]}}
                had_trial = bool(message.get('trial_delivery') or message.get('reasoning', {}).get('trace', {}).get('trial_delivery'))
                for field in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    message.pop(field, None)
                if had_trial:
                    message['trial_delivery'] = unavailable_trial_receipt()
                if message.get('research_provenance_required') is True:
                    message.update(content=WITHDRAWN_ANALYSIS, claim_audit=audit_answer(WITHDRAWN_ANALYSIS, []))
                    if 'answer' in message:
                        message['answer'] = WITHDRAWN_ANALYSIS
            # Legacy memory-bearing answers have only counts, not revision
            # lineage. Fail closed rather than pretend their text was forgotten.
            memory_required = (message.get('research_memory_required') is True
                or message.get('id') in memory_states or (item or {}).get('used_memory_count', 0) > 0
                or (item or {}).get('memory_dependency_required') is True)
            if memory_required and memory_states.get(message.get('id')) is not True:
                previous = message.get('citation_validation', {})
                message = {**message, 'content': WITHDRAWN_MEMORY, 'research_memory_required': True,
                    'research_provenance_required': True, 'sources': [], 'used_memory_count': 0,
                    'citation_validation': {**previous, 'valid': False, 'warnings': [*previous.get('warnings', []),
                        {'code': 'memory_context_revoked', 'message': '相关记忆已改变或无法核验，旧回答已收起。'}]}}
                for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    message.pop(key, None)
                if 'answer' in message:
                    message['answer'] = WITHDRAWN_MEMORY
            if (item or {}).get('answer_dependency_required') is True and parent_states.get(message.get('id')) is not True:
                previous = message.get('citation_validation', {})
                message = {**message, 'content': WITHDRAWN_ANALYSIS, 'sources': [],
                    'research_provenance_required': True, 'answer_dependency_required': True,
                    'citation_validation': {**previous, 'valid': False, 'warnings': [*previous.get('warnings', []),
                        {'code': 'provenance_unavailable', 'message': '前序分析的依据已撤回或无法核对，派生回答已收起。'}]}}
                for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    message.pop(key, None)
                if 'answer' in message:
                    message['answer'] = WITHDRAWN_ANALYSIS
            if message.get('id') in event_required_ids and event_states.get(message['id']) is not True:
                previous = message.get('citation_validation', {})
                message = {**message, 'content': WITHDRAWN_ANALYSIS, 'sources': [],
                    'research_provenance_required': True,
                    'citation_validation': {**previous, 'valid': False, 'warnings': [*previous.get('warnings', []),
                        {'code': 'provenance_unavailable', 'message': '事件版本或来源已失效，旧回答已收起。'}]}}
                for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery', 'event_reasoning', 'task_review'):
                    message.pop(key, None)
                if 'answer' in message:
                    message['answer'] = WITHDRAWN_ANALYSIS
            result.append(message)
        return result
