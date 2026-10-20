"""Explicit, bounded execution of user-authored research questions.

There is no planner, critic, retry, background startup dispatch, or model writer.
The durable store owns claims/budgets; the supplied bridge owns RequestJournal
and the real ResearchService call. Unknown accepted calls are only reconciled.
"""
from __future__ import annotations

import asyncio
import html
import inspect
import json
import re
import sqlite3
from collections.abc import Mapping

from .project_task_store import ProjectTaskConflictError

MAX_CALLS = 6
MAX_DOCUMENTS = 100
MAX_REPORT_BYTES = 384 * 1024
_DIGEST = re.compile(r'[0-9a-f]{64}')
_UUID4 = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}')
_IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,128}')


class ProjectTaskRuntimeError(ValueError):
    """Safe fixed failure, never an upstream exception or model fragment."""


class ProjectTaskBusyError(ProjectTaskRuntimeError):
    pass


async def _invoke(callback, **kwargs):
    value = callback(**kwargs)
    return await value if inspect.isawaitable(value) else value


def _scope(step):
    if step.get('input_kind') == 'personal':
        return []
    scope = step.get('document_ids')
    if scope is None:
        return None
    if (not isinstance(scope, list) or not 1 <= len(scope) <= MAX_DOCUMENTS
            or any(not isinstance(item, str) or not _IDENTIFIER.fullmatch(item) for item in scope)
            or len(set(scope)) != len(scope)):
        raise ProjectTaskRuntimeError('Invalid document scope')
    return list(scope)


def _snapshot(value, scope=None):
    if not isinstance(value, Mapping) or len(value) > MAX_DOCUMENTS:
        raise ProjectTaskRuntimeError('Source snapshot unavailable')
    result = {}
    for key, digest in value.items():
        if (not isinstance(key, str) or not _IDENTIFIER.fullmatch(key)
                or not isinstance(digest, str) or not _DIGEST.fullmatch(digest)):
            raise ProjectTaskRuntimeError('Source snapshot unavailable')
        result[key] = digest
    if scope is not None:
        if set(result) != set(scope):
            raise ProjectTaskRuntimeError('Scoped sources unavailable')
    return dict(sorted(result.items()))


def _changed(before, after):
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))


def _handoff_counts(value):
    keys = {'required', 'used', 'full', 'excerpted', 'omitted'}
    if (not isinstance(value, dict) or set(value) != keys
            or any(type(value[key]) is not int or not 0 <= value[key] <= MAX_CALLS for key in keys)
            or value['required'] != value['used'] + value['omitted']
            or value['used'] != value['full'] + value['excerpted']):
        raise ProjectTaskRuntimeError('Dependency handoff counts unavailable')
    return {key: value[key] for key in sorted(keys)}


def _artifact(receipt):
    """Persist references only, never a provider answer or hidden reasoning."""
    if not isinstance(receipt, dict) or receipt.get('request_status') != 'completed':
        raise ProjectTaskRuntimeError('Completed request receipt unavailable')
    answer_id = receipt.get('answer_id')
    if not isinstance(answer_id, str) or not _UUID4.fullmatch(answer_id):
        raise ProjectTaskRuntimeError('Answer reference unavailable')
    result = {'answer_id': answer_id}
    for key, maximum in (('source_ids', 10), ('document_ids', MAX_DOCUMENTS)):
        values = receipt.get(key)
        if (not isinstance(values, list) or len(values) > maximum
                or any(not isinstance(item, str) or not _IDENTIFIER.fullmatch(item) for item in values)
                or len(set(values)) != len(values)):
            raise ProjectTaskRuntimeError('Answer references unavailable')
        result[key] = list(values)
    if 'handoff' in receipt:
        result['handoff'] = _handoff_counts(receipt['handoff'])
    if 'phase_handoff' in receipt:
        from .project_task_store import _phase_handoff
        result['phase_handoff'] = _phase_handoff(receipt['phase_handoff'])
    if 'event_reasoning' in receipt:
        from .project_evidence_feedback import normalize_event_reasoning
        try:
            result['event_reasoning'] = normalize_event_reasoning(receipt['event_reasoning'])
        except (ValueError, TypeError, KeyError) as error:
            raise ProjectTaskRuntimeError('Evidence feedback is unavailable') from error
    if 'task_review' in receipt:
        from .research_task_review import normalize_task_review
        try:
            result['task_review'] = normalize_task_review(receipt['task_review'])
        except (ValueError, TypeError, KeyError) as error:
            raise ProjectTaskRuntimeError('Task review is unavailable') from error
    if receipt.get('input_kind') == 'personal':
        refs = receipt.get('context_refs')
        if (type(refs) is not dict or len(refs) > 8 or result['source_ids'] or result['document_ids']
                or any(not _UUID4.fullmatch(key) or type(value) is not int or value < 1
                       for key, value in refs.items())):
            raise ProjectTaskRuntimeError('Personal context references unavailable')
        result.update(input_kind='personal', context_refs=dict(refs))
    # Source text/quotes and arbitrary provider metadata deliberately do not
    # enter the task store. The answer ID resolves current proof on read.
    return result


def _handoff_note(references):
    if 'handoff' not in references:
        result = ['前序产物使用情况未记录（历史版本）；不能据此推断本步已完整复用前序分析。', '']
    else:
        counts = _handoff_counts(references['handoff'])
        result = [f"前序产物：需要 {counts['required']} 项，实际使用 {counts['used']} 项；"
                  f"完整复用 {counts['full']} 项、节选复用 {counts['excerpted']} 项、未使用 {counts['omitted']} 项。", '']
        if counts['excerpted'] or counts['omitted']:
            result += ['本步只接收了前序产物的部分内容；节选或未使用的部分不代表已被综合。前序分析仍不是原文证据。', '']
        else:
            result += ['这些计数仅记录送入本步的内容范围，不表示模型理解正确或推理已经验证。', '']
    if 'phase_handoff' in references:
        from .project_task_store import _phase_handoff
        phase = _phase_handoff(references['phase_handoff'])
        if phase['required']:
            result += [f"跨阶段分析：需要 {phase['required']} 项，实际使用 {phase['used']} 项；"
                       f"完整复用 {phase['full']} 项、节选复用 {phase['excerpted']} 项。", '']
            result += [f"- 父阶段 `{ref['phase_id']}` · 报告版本 {ref['report_version']} · "
                       f"步骤 `{ref['step_id']}` · 产物版本 {ref['version']}。" for ref in phase['refs']]
            result += ['', '跨阶段内容是分析线索，不是原文证据；节选不表示完整阅读，版本引用不证明结论或因果推理正确。', '']
    return result


def _legend_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ProjectTaskRuntimeError('Source filename unavailable')
    # The source filename is data, not a heading/link/HTML supplied by a paper.
    value = html.escape(' '.join(value.split()), quote=True)
    return re.sub(r'([\\`*{}\[\]()#+.!_|>~-])', r'\\\1', value)


def _source_legend(answer, references, body):
    sources = answer.get('sources') if isinstance(answer, dict) else None
    if not isinstance(sources, list) or len(sources) > 10:
        raise ProjectTaskRuntimeError('Source locators unavailable')
    ids, labels, documents, rows = set(), set(), set(), []
    for source in sources:
        if not isinstance(source, dict):
            raise ProjectTaskRuntimeError('Source locators unavailable')
        source_id, document_id, label = source.get('id'), source.get('document_id'), source.get('label')
        ordinal, page = source.get('ordinal'), source.get('page')
        start, end = source.get('excerpt_start'), source.get('excerpt_end')
        if (not isinstance(source_id, str) or not _IDENTIFIER.fullmatch(source_id)
                or not isinstance(document_id, str) or not _IDENTIFIER.fullmatch(document_id)
                or not isinstance(label, str) or not re.fullmatch(r'S(?:[1-9]|10)', label)
                or source_id in ids or label in labels
                or type(ordinal) is not int or not 0 <= ordinal <= 2_000_000
                or (page is not None and (type(page) is not int or not 1 <= page <= 1_000_000))
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= 2_000_000):
            raise ProjectTaskRuntimeError('Source locators unavailable')
        filename = _legend_text(source.get('filename'))
        location = f'第 {page} 页' if page is not None else '无页码（非分页资料或页码不可用）'
        rows.append(f'- [{label}] {filename}；{location}；块序号 {ordinal}（从 0 计）；'
                    f'块内字符 [{start}, {end})（从 0 计）；文档 `{document_id}`；来源 `{source_id}`。')
        ids.add(source_id)
        documents.add(document_id)
        labels.add(label)
    if ids != set(references['source_ids']) or documents != set(references['document_ids']):
        raise ProjectTaskRuntimeError('Source references changed')
    citations = {label for group in re.findall(r'\[[^\]\n]{0,100}\]', body)
                 for label in re.findall(r'\bS[0-9]+\b', group)}
    if not citations <= labels:
        raise ProjectTaskRuntimeError('Source citation locator unavailable')
    return ['### 本节来源定位', '',
            'S 标签仅对应本节来源，不跨节合并；定位可核查不等于每一主张已被证实。', '',
            *(rows or ['本节没有可核查的原文来源；不要把本节当作已获原文支持的结论。']), '']


class ProjectTaskRuntime:
    def __init__(self, store, *, execute_question, read_sources, reconcile_request=None, load_answer=None,
                 read_context=None, validate_inputs=None, read_phase_dependencies=None, validate_task=None,
                 read_task_context=None, bind_task_origins=False):
        self.store = store
        self.execute_question = execute_question
        self.read_sources = read_sources
        self.reconcile_request = reconcile_request
        self.load_answer = load_answer
        self.read_context = read_context
        self.validate_inputs = validate_inputs
        self.read_phase_dependencies = read_phase_dependencies
        self.validate_task = validate_task
        self.read_task_context = read_task_context
        self.bind_task_origins = bind_task_origins
        self._running = set()

    async def _read_snapshot(self, scope=None):
        if scope == []:
            return {}
        return _snapshot(await _invoke(self.read_sources, document_ids=scope), scope)

    async def _read_context(self, memory_ids):
        if not memory_ids:
            return {}
        if self.read_context is None:
            raise ProjectTaskRuntimeError('Personal context reader unavailable')
        value = await _invoke(self.read_context, memory_ids=list(memory_ids))
        if (type(value) is not dict or set(value) != set(memory_ids) or len(value) > 48
                or any(not isinstance(key, str) or not _UUID4.fullmatch(key)
                       or type(revision) is not int or revision < 1 for key, revision in value.items())):
            raise ProjectTaskRuntimeError('Personal context snapshot unavailable')
        return dict(sorted(value.items()))

    async def context_snapshot(self, task):
        if not any(row.get('input_kind') == 'personal' for row in task['steps']):
            return {}
        identifiers = list(dict.fromkeys(mid for row in task['steps']
            if row.get('input_kind') == 'personal' for mid in row.get('memory_ids', [])))
        return await self._read_context(identifiers)

    async def approval_snapshots(self, task_id):
        task = self.store.get_task(task_id)
        documents = any(row['kind'] == 'analysis' and row.get('input_kind') != 'personal'
                        for row in task['steps'])
        return (await self._read_snapshot() if documents else {}, await self.context_snapshot(task))

    async def _step_context(self, step):
        return await self._read_context(step.get('memory_ids', [])) if step.get('input_kind') == 'personal' else {}

    async def current_context(self, task):
        """Current available revisions; missing IDs mean withdrawal, not zero."""
        result = {}
        for identifier in dict.fromkeys(mid for row in task['steps'] if row.get('input_kind') == 'personal'
                                       for mid in row.get('memory_ids', [])):
            try:
                result.update(await self._read_context([identifier]))
            except (ValueError, KeyError):
                continue
        return result

    async def run(self, task_id, *, expected_revision, run_request_id, max_calls=MAX_CALLS):
        """A user-approved bounded run, never a startup or status-read action."""
        if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
            raise ProjectTaskRuntimeError('Invalid run budget')
        if not isinstance(run_request_id, str) or not _UUID4.fullmatch(run_request_id):
            raise ProjectTaskRuntimeError('Invalid run reference')
        if task_id in self._running:
            raise ProjectTaskBusyError('Task is already running')
        self._running.add(task_id)
        try:
            snapshot, contexts = await self.approval_snapshots(task_id)
            task = self.store.begin_run(task_id, expected_revision=expected_revision,
                request_id=run_request_id, max_calls=max_calls, source_snapshots=snapshot,
                context_snapshots=contexts)
            if task.get('run_replayed'):
                return self.store.get_task(task_id)
            run_id = task.get('run_id')
            if not run_id:
                return self.store.get_task(task_id)
            return await self._loop(task_id, task, run_id)
        finally:
            self._running.discard(task_id)

    async def drive_approved(self, task_id, *, run_request_id):
        """Internal continuation of one freshly persisted HTTP approval.

        The route creates the grant exactly once, then queues this owner. This
        method creates no grant or budget, is not a public resume endpoint, and
        must not be called automatically during startup/recovery.
        """
        if not isinstance(run_request_id, str) or not _UUID4.fullmatch(run_request_id):
            raise ProjectTaskRuntimeError('Invalid run reference')
        if task_id in self._running:
            raise ProjectTaskBusyError('Task is already running')
        self._running.add(task_id)
        try:
            task = self.store.get_task(task_id)
            if task.get('run_id') != run_request_id:
                raise ProjectTaskConflictError('Run approval is no longer current')
            if task.get('status') not in {'ready', 'running'}:
                return task
            maximum, remaining = task.get('run_max_calls'), task.get('run_remaining')
            if (type(maximum) is not int or not 1 <= maximum <= MAX_CALLS
                    or type(remaining) is not int or not 0 <= remaining <= maximum):
                raise ProjectTaskRuntimeError('Run allowance unavailable')
            # claim_next checks that this exact durable grant is still active;
            # a paused/exhausted grant cannot be refilled by driving it again.
            return await self._loop(task_id, task, run_request_id)
        finally:
            self._running.discard(task_id)

    async def _loop(self, task_id, task, run_id):
        for _ in range(MAX_CALLS + 2):  # inventory + approved questions + report
            contexts = await self.current_context(task)
            changed = list(dict.fromkeys(identifier for row in task['steps']
                if row['kind'] == 'analysis' and row.get('input_kind') == 'personal'
                for identifier in _changed(row.get('context_snapshot', {}),
                    {key: value for key, value in contexts.items() if key in row.get('memory_ids', [])})))
            if changed:
                self.store.invalidate_memories(changed)
                break
            step = self.store.claim_next(task_id, run_id=run_id, context_snapshots=contexts)
            if step is None:
                break
            try:
                await self._execute_step(task_id, task, step)
            except asyncio.CancelledError:
                if step['kind'] == 'analysis':
                    # Covers cancellation during post-call snapshot reads,
                    # as well as the provider await itself.
                    self._block(task_id, step, 'provider_outcome_unknown')
                raise
            task = self.store.get_task(task_id)
            if task.get('status') not in {'running', 'ready'}:
                break
        return self.store.get_task(task_id)

    async def _execute_step(self, task_id, task, step):
        token = step['claim_token']
        before = step.get('input_snapshot', {})
        try:
            current = await self._read_snapshot(_scope(step))
            context = await self.current_context({'steps': [step]})
        except Exception:
            self._block(task_id, step, 'source_unavailable')
            return
        changed = _changed(before, current)
        context_changed = _changed(step.get('context_snapshot', {}), context)
        if context_changed:
            self._block(task_id, step, 'source_changed')
            self.store.invalidate_memories(context_changed)
            return
        if changed:
            # No external call has been made for this claim. Do not turn a
            # preflight rejection into an ambiguous paid attempt on invalidate.
            self._block(task_id, step, 'source_changed')
            self.store.invalidate_sources(changed, reason='source_changed')
            return
        if step['kind'] == 'inventory':
            artifact = {'documents': [{'id': key, 'digest': value} for key, value in current.items()]}
        elif step['kind'] == 'report':
            complete = [item for item in task['steps'] if item['kind'] == 'analysis' and item['status'] == 'completed']
            if len(complete) != sum(item['kind'] == 'analysis' for item in task['steps']):
                self._block(task_id, step, 'dependency_blocked')
                return
            artifact = {'step_refs': [{'step_id': item['id'], 'version': item['artifact_version']} for item in complete]}
        elif step['kind'] == 'analysis':
            try:
                dependencies = self._dependencies(task_id, task, step)
                phase_dependencies = (await _invoke(self.read_phase_dependencies, task_id=task_id, step=step)
                                      if self.read_phase_dependencies is not None else [])
                phase_context = (await _invoke(self.read_task_context, task_id=task_id)
                                 if self.read_task_context is not None else None)
            except (ProjectTaskRuntimeError, ProjectTaskConflictError):
                self._block(task_id, step, 'dependency_blocked')
                return
            try:
                personal = ({'input_kind': 'personal', 'memory_ids': step.get('memory_ids', []),
                             'context_snapshot': context} if step.get('input_kind') == 'personal' else {})
                receipt = await _invoke(self.execute_question,
                    session_id=self.store.session_id, request_id=step['request_id'],
                    question=step['question'], document_ids=_scope(step),
                    input_snapshot=dict(before), goal=task['goal'], constraints=task['constraints'],
                    dependency_artifacts=dependencies, **personal,
                    **({'origin_task_id': task_id} if self.bind_task_origins else {}),
                    **({'phase_dependencies': phase_dependencies} if phase_dependencies else {}),
                    **({'phase_task_id': task_id, 'phase_context': phase_context} if phase_context is not None else {}))
            except asyncio.CancelledError:
                # Cancellation cannot prove rollback of an accepted model call.
                self._block(task_id, step, 'provider_outcome_unknown')
                raise
            except Exception:
                self._block(task_id, step, 'provider_outcome_unknown')
                return
            if not isinstance(receipt, dict) or receipt.get('request_status') != 'completed':
                reason = 'provider_failed' if isinstance(receipt, dict) and receipt.get('request_status') == 'failed' else 'provider_outcome_unknown'
                if reason == 'provider_failed' and receipt.get('code') == 'dependency_unavailable':
                    reason = 'dependency_blocked'
                self._block(task_id, step, reason)
                return
            try:
                artifact = _artifact(receipt)
                if 'handoff' in artifact and artifact['handoff']['required'] != len(dependencies):
                    raise ProjectTaskRuntimeError('Dependency handoff does not match approved dependencies')
                expected_phase_refs = [row['phase_ref'] for row in phase_dependencies]
                if expected_phase_refs and artifact.get('phase_handoff', {}).get('refs') != expected_phase_refs:
                    raise ProjectTaskRuntimeError('Phase handoff differs from approved parent versions')
                if not set(artifact['document_ids']) <= set(before):
                    raise ProjectTaskRuntimeError('Answer references exceed approved scope')
                after = await self._read_snapshot(_scope(step))
                after_context = await self.current_context({'steps': [step]})
            except ProjectTaskRuntimeError:
                self._block(task_id, step, 'invalid_result')
                return
            except Exception:
                self._block(task_id, step, 'provider_outcome_unknown')
                return
            changed = _changed(before, after)
            context_changed = _changed(context, after_context)
            if context_changed:
                self._block(task_id, step, 'source_changed')
                self.store.invalidate_memories(context_changed)
                return
            if changed:
                # The bridge supplied a definite completed receipt. Its result
                # is unusable against changed sources, but its outcome is not
                # unknown and a later explicit refresh may safely re-plan it.
                self._block(task_id, step, 'source_changed')
                self.store.invalidate_sources(changed, reason='source_changed')
                return
        else:
            self._block(task_id, step, 'invalid_result')
            return
        try:
            final_context = await self.current_context({'steps': [step]})
            changed = _changed(step.get('context_snapshot', {}), final_context)
            if changed:
                self._block(task_id, step, 'source_changed')
                self.store.invalidate_memories(changed)
                return
            if self.validate_task is not None and self.validate_task(task_id) is not True:
                self._block(task_id, step, 'source_changed')
                return
            if (step['kind'] == 'analysis' and self.validate_inputs is not None
                    and self.validate_inputs([{**step, 'answer_id': artifact['answer_id']}]) is not True):
                self._block(task_id, step, 'source_changed')
                return
            self.store.complete_step(task_id, step['id'], token, artifact, context_snapshots=final_context)
        except ProjectTaskConflictError:
            # Revocation/cancel may win while an already accepted call runs.
            # Its paid result stays in the bridge journal, not in this version.
            self._require_lost_claim(task_id, step)

    def _require_lost_claim(self, task_id, step):
        current = self.store.get_task(task_id)
        matching = next((row for row in current['steps'] if row['id'] == step['id']), None)
        if (matching is not None and matching['status'] == 'running'
                and matching['generation'] == step['generation']):
            raise ProjectTaskConflictError('Step claim could not be settled')

    def _block(self, task_id, step, reason):
        try:
            self.store.block_step(task_id, step['id'], step['claim_token'], reason=reason)
        except ProjectTaskConflictError:
            self._require_lost_claim(task_id, step)

    def _dependencies(self, task_id, task, step):
        """Current direct predecessor refs; no unapproved cross-scope content."""
        by_id = {row['id']: row for row in task['steps']}
        result = []
        for key in step['depends_on']:
            predecessor = by_id.get(key)
            if predecessor is None or predecessor['status'] != 'completed':
                raise ProjectTaskRuntimeError('Dependency references unavailable')
            if predecessor['kind'] != 'analysis':
                continue
            stored = self.store.get_artifact(task_id, key, version=predecessor['artifact_version'])
            if not stored.get('available') or stored.get('status') != 'current':
                raise ProjectTaskRuntimeError('Dependency references unavailable')
            refs = stored.get('references')
            if not isinstance(refs, dict) or 'answer_id' not in refs:
                raise ProjectTaskRuntimeError('Dependency references unavailable')
            if not set(refs['document_ids']) <= set(step['input_snapshot']):
                # A declared dependency cannot silently disappear. The user
                # must explicitly fix the scope; no source permission expands.
                raise ProjectTaskRuntimeError('Dependency exceeds approved source scope')
            if refs.get('input_kind') == 'personal':
                current = step.get('context_snapshot', {})
                if any(current.get(key) != value for key, value in refs['context_refs'].items()):
                    raise ProjectTaskRuntimeError('Dependency exceeds approved personal context')
            result.append({'step_id': key, 'version': stored['version'], 'references': refs})
        if len(result) > MAX_CALLS:
            raise ProjectTaskRuntimeError('Dependency references exceed budget')
        return result

    async def reconcile(self, task_id):
        """Read known request results after loss; never call execute_question."""
        task = self.store.get_task(task_id)
        if self.reconcile_request is None:
            return task
        unresolved = self.store.unresolved_requests(task_id)
        for step in task['steps']:
            invalidated = step['status'] == 'stale' and step.get('request_id') in unresolved
            if step['kind'] != 'analysis' or not (invalidated or
                    (step['status'] == 'blocked' and step.get('reason') == 'provider_outcome_unknown')):
                continue
            try:
                receipt = await _invoke(self.reconcile_request, session_id=self.store.session_id,
                                        request_id=step['request_id'])
                if not isinstance(receipt, dict) or receipt.get('request_status') not in {'completed', 'failed'}:
                    continue
                if invalidated:
                    self.store.settle_invalidated_request(task_id, step['id'], request_id=step['request_id'],
                                                         outcome=receipt['request_status'])
                    continue
                current = await self._read_snapshot(_scope(step))
                context = await self.current_context({'steps': [step]})
                changed = _changed(step.get('input_snapshot', {}), current)
                if changed:
                    self.store.invalidate_sources(changed, reason='source_changed')
                    continue
                context_changed = _changed(step.get('context_snapshot', {}), context)
                if context_changed:
                    self.store.invalidate_memories(context_changed)
                    continue
                if receipt['request_status'] == 'failed':
                    # A durable explicit failure may become resettable by a
                    # later user action. It is not an allowance to retry here.
                    self.store.reconcile_failed_step(task_id, step['id'], request_id=step['request_id'],
                                                     source_snapshots=current, context_snapshots=context)
                else:
                    self.store.reconcile_step(task_id, step['id'], request_id=step['request_id'],
                                              artifact=_artifact(receipt), source_snapshots=current, context_snapshots=context)
            except Exception:
                # Expired/corrupt/unavailable journals do not authorize retry.
                continue
        return self.store.get_task(task_id)

    async def refresh(self, task_id, *, expected_revision):
        snapshot, contexts = await self.approval_snapshots(task_id)
        return self.store.refresh_task(task_id, expected_revision=expected_revision, source_snapshots=snapshot,
                                       context_snapshots=contexts)

    def review_current_feedback(self, task):
        """Final synchronous metadata projection, never a retry or state rewrite.

        Execution completion is durable history. Its cached positive review is
        not current evidence after an answer's event/source lineage is revoked.
        One task snapshot supplies the exact answer IDs to one batch barrier.
        """
        keys = ('evidence_feedback', 'task_review_feedback')
        if self.validate_inputs is None or not any(any(key in row for key in keys) for row in task['steps']):
            return task
        result, live = task, False
        try:
            with self.store._db() as db:
                if db is None:
                    raise ProjectTaskRuntimeError('Task feedback snapshot unavailable')
                db.execute('PRAGMA busy_timeout=50')
                db.execute('BEGIN')
                current = self.store._public(db, self.store._task(db, task['id']))
                result = {**task, **current}
                packets = []
                for row in current['steps']:
                    if row['kind'] != 'analysis' or row['status'] != 'completed':
                        continue
                    saved = db.execute("SELECT payload FROM project_task_artifacts WHERE task_id=? AND step_id=? "
                        "AND version=? AND status='current'", (current['id'], row['id'], row['artifact_version'])).fetchone()
                    data = json.loads(saved['payload']) if saved else {}
                    if type(data.get('answer_id')) is not str or not _UUID4.fullmatch(data['answer_id']):
                        raise ProjectTaskRuntimeError('Task feedback answer unavailable')
                    packets.append({**row, 'answer_id': data['answer_id']})
                live = bool(packets) and self.validate_inputs(packets) is True
        except (ValueError, KeyError, TypeError, sqlite3.Error):
            live = False
        if live:
            return result
        result = {**result, 'steps': [dict(row) for row in result['steps']]}
        for row in result['steps']:
            if any(key in row for key in keys):
                for key in keys:
                    row.pop(key, None)
                row.update(content_review='needs_evidence', feedback_status='provenance_unavailable')
        return result

    async def render_report(self, task_id, artifact):
        """Hydrate immutable answer references under today's revocation checks."""
        task = self.store.get_task(task_id)
        if not isinstance(artifact, dict):
            raise ProjectTaskRuntimeError('Report references unavailable')
        wrapper = 'references' in artifact
        stored_report = self.store.get_artifact(task_id, 'report', version=artifact.get('version') if wrapper else None)
        payload = artifact.get('references') if wrapper else artifact
        if (not stored_report.get('available') or stored_report.get('status') != 'current'
                or stored_report.get('references') != payload):
            raise ProjectTaskRuntimeError('Report references unavailable')
        refs = payload.get('step_refs') if isinstance(payload, dict) else None
        if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_CALLS:
            raise ProjectTaskRuntimeError('Report references unavailable')
        steps = {item['id']: item for item in task['steps']}
        has_personal = any(row.get('input_kind') == 'personal' for row in task['steps'])
        lines = ['# ' + ('个人任务报告' if has_personal else '项目研究报告'), '', str(task['goal']), '', '## 约束', '', str(task['constraints']), '']
        complete = True
        rendered = []
        evidence_feedback = []
        task_review_feedback = []
        for ref in refs:
            step = steps.get(ref.get('step_id')) if isinstance(ref, dict) else None
            if step is None or step['kind'] != 'analysis':
                raise ProjectTaskRuntimeError('Report references unavailable')
            lines += ['## ' + step['title'], '', step['question'], '']
            try:
                stored = self.store.get_artifact(task_id, step['id'], version=ref['version'])
                if not stored.get('available') or stored.get('status') != 'current':
                    raise ProjectTaskRuntimeError('Answer references unavailable')
                current = await self._read_snapshot(_scope(step))
                context = await self._step_context(step)
                if current != step['input_snapshot']:
                    raise ProjectTaskRuntimeError('Answer source snapshot changed')
                if context != step.get('context_snapshot', {}):
                    raise ProjectTaskRuntimeError('Answer personal context changed')
                data = stored['references']
                if self.load_answer is None:
                    raise ProjectTaskRuntimeError('Answer body unavailable')
                answer = await _invoke(self.load_answer, session_id=self.store.session_id, answer_id=data['answer_id'])
                body = answer.get('content', answer.get('answer')) if isinstance(answer, dict) else answer
                if not isinstance(body, str) or not body.strip() or len(body.encode()) > 65536:
                    raise ProjectTaskRuntimeError('Answer body unavailable')
                legend = (_source_legend(answer, data, body) if step.get('input_kind') != 'personal' else
                    ['### 本节依据类型', '', '用户任务指令与明确选择的确认记忆；不是文档证据，不证明外部事实。', '',
                     *[f'- 确认记忆 `{key}` · 版本 {revision}。' for key, revision in context.items()], ''])
                handoff_note = _handoff_note(data)
                feedback_note = []
                feedback = None
                if 'event_reasoning' in data:
                    from .project_evidence_feedback import feedback_view, feedback_lines
                    feedback = feedback_view(data['event_reasoning'], data.get('evidence_feedback_binding'),
                        task_id=task_id, step_id=step['id'], artifact_version=ref['version'],
                        current_task_revision=task['revision'])
                    feedback_note = feedback_lines(feedback)
                task_feedback = None
                if 'task_review' in data:
                    from .project_evidence_feedback import feedback_view, feedback_lines
                    task_feedback = feedback_view(data['task_review'], data.get('task_review_binding'),
                        task_id=task_id, step_id=step['id'], artifact_version=ref['version'],
                        current_task_revision=task['revision'], review_kind='task')
                    feedback_note += feedback_lines(task_feedback)
                latest = self.store.get_artifact(task_id, step['id'], version=ref['version'])
                if (not latest['available'] or await self._read_snapshot(_scope(step)) != current
                        or await self._step_context(step) != context):
                    raise ProjectTaskRuntimeError('Answer source snapshot changed')
                lines += [*handoff_note, body, '', *legend, *feedback_note]
                rendered.append(({**step, 'answer_id': data['answer_id']}, current))
                if feedback is not None:
                    evidence_feedback.append(feedback)
                if task_feedback is not None:
                    task_review_feedback.append(task_feedback)
            except Exception:
                complete = False
                lines += ['本节原回答或当前有效依据不可用；未重新生成，不能视为完整结论。', '']
        # A later hydration may yield while another request revokes an earlier
        # section. Recheck before returning any accumulated content.
        current_report = self.store.get_artifact(task_id, 'report', version=stored_report['version'])
        if not current_report['available']:
            raise ProjectTaskRuntimeError('Report inputs changed while rendering')
        for included, expected in rendered:
            if (await self._read_snapshot(_scope(included)) != expected
                    or await self._step_context(included) != included.get('context_snapshot', {})):
                raise ProjectTaskRuntimeError('Report inputs changed while rendering')
        # Per-section await checks cannot be the final authority: a later
        # section's read can race withdrawal of an earlier section. The real
        # adapter checks the complete document+memory set in ONE transaction
        # after every await, without another event-loop yield before return.
        if rendered and self.validate_inputs is not None and self.validate_inputs([row for row, _ in rendered]) is not True:
            raise ProjectTaskRuntimeError('Report inputs changed at publication')
        if self.validate_task is not None and self.validate_task(task_id) is not True:
            raise ProjectTaskRuntimeError('Phase ancestors changed at publication')
        if not self.store.get_artifact(task_id, 'report', version=stored_report['version'])['available']:
            raise ProjectTaskRuntimeError('Report version changed at publication')
        lines += ['---', '程序按已保存的各问题回答汇总，未追加模型写作；内容仍需核对，不表示事实或整体推理已验证。']
        content = '\n'.join(lines)
        if len(content.encode()) > MAX_REPORT_BYTES:
            raise ProjectTaskRuntimeError('Report exceeds export budget')
        result = {'content': content, 'complete': complete, 'format': 'markdown',
                'acceptance_status': 'review_required', 'goal_verified': False,
                'composition': 'deterministic_answer_references', 'model_calls': 0,
                'semantic_verified': False}
        all_feedback = [*evidence_feedback, *task_review_feedback]
        if all_feedback:
            result.update(content_review=('needs_evidence' if any(row['content_review'] == 'needs_evidence'
                    for row in all_feedback) else 'review_required'),
                next_step_suggestions=[{'binding': row['binding'], **row['next_step_suggestion']}
                    for row in all_feedback])
            if evidence_feedback:
                result['evidence_feedback'] = evidence_feedback
            if task_review_feedback:
                result['task_review_feedback'] = task_review_feedback
        return result
