"""Bounded evidence feedback, not an execution approval or a semantic proof.

Only coordinate/hash receipts enter durable task artifacts. Suggestions are
data for a later, explicitly confirmed CAS replan, never executable commands.
"""
from __future__ import annotations

import hashlib
import html
import json
import re

MAX_FEEDBACK_BYTES = 16384
_STATUS = {'evidence_gaps', 'assessed_with_caveats', 'not_assessed',
           'needs_evidence', 'model_assessed_with_support', 'review_required'}
_ITEM_STATUS = {'unknown', 'model_assessed_support', 'model_assessed_counterevidence'}
_TOP = {'status', 'obligations', 'supported', 'counterevidence', 'unknown',
        'gaps', 'suggested_queries', 'causality_verified', 'model_assessed'}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _text(value, limit, *, empty=False):
    if (type(value) is not str or len(value) > limit or (not empty and not value.strip())
            or any(ord(char) < 32 and char not in '\n\t' for char in value)):
        raise ValueError('Invalid evidence feedback text')
    return value


def _metadata(value, depth=0):
    """Permit bounded schema additions, but never another answer/quote body."""
    if depth > 7:
        raise ValueError('Evidence feedback is too deeply nested')
    if value is None or type(value) is bool:
        return
    if type(value) is int and 0 <= value <= 2_000_000:
        return
    if type(value) is str:
        _text(value, 512, empty=True)
        return
    if type(value) is list and len(value) <= 48:
        for item in value:
            _metadata(item, depth + 1)
        return
    if type(value) is dict and len(value) <= 24:
        for key, item in value.items():
            if (type(key) is not str or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', key)
                    or key in {'quote', 'content', 'answer', 'body', 'hidden_reasoning'}):
                raise ValueError('Evidence feedback may only contain bounded metadata')
            if key in {'causality_verified', 'semantic_verified', 'goal_verified'} and item is not False:
                raise ValueError('Evidence feedback is not a verification certificate')
            _metadata(item, depth + 1)
        return
    raise ValueError('Invalid evidence feedback metadata')


def _checked_event_reasoning(value):
    if type(value) is not dict or not _TOP <= set(value):
        raise ValueError('Incomplete evidence feedback receipt')
    _metadata(value)
    if (type(value['status']) is not str or value['status'] not in _STATUS or value['causality_verified'] is not False
            or type(value['model_assessed']) is not bool):
        raise ValueError('Evidence assessment is not verified causality')
    obligations, gaps, queries = value['obligations'], value['gaps'], value['suggested_queries']
    if (type(obligations) is not list or len(obligations) > 24
            or type(gaps) is not list or len(gaps) > 48
            or type(queries) is not list or len(queries) > 8):
        raise ValueError('Evidence feedback exceeds its item budget')
    ids, counts = set(), dict.fromkeys(_ITEM_STATUS, 0)
    for row in obligations:
        if (type(row) is not dict or not {'id', 'kind', 'status', 'evidence', 'gaps'} <= set(row)
                or type(row['kind']) is not str or row['kind'] not in {'relation', 'alternatives', 'path', 'event'}
                or type(row['status']) is not str or row['status'] not in _ITEM_STATUS):
            raise ValueError('Invalid evidence obligation')
        key = _text(row['id'], 32)
        if key in ids:
            raise ValueError('Repeated evidence obligation')
        ids.add(key)
        counts[row['status']] += 1
        if type(row['gaps']) is not list or len(row['gaps']) > 12:
            raise ValueError('Invalid evidence obligation gaps')
        for code in row['gaps']:
            _text(code, 64)
        if type(row['evidence']) is not list or len(row['evidence']) > 3:
            raise ValueError('Invalid evidence coordinates')
        for ref in row['evidence']:
            if type(ref) is not dict or set(ref) != {'label', 'source_id', 'span', 'quote_sha256'}:
                raise ValueError('Evidence feedback stores coordinates, not source copies')
            _text(ref['label'], 32)
            _text(ref['source_id'], 128)
            span = ref['span']
            if (type(span) is not list or len(span) != 2 or any(type(i) is not int for i in span)
                    or not 0 <= span[0] < span[1] <= 2_000_000
                    or type(ref['quote_sha256']) is not str
                    or not re.fullmatch(r'[0-9a-f]{64}', ref['quote_sha256'])):
                raise ValueError('Invalid evidence coordinates')
    expected = {'supported': counts['model_assessed_support'],
                'counterevidence': counts['model_assessed_counterevidence'], 'unknown': counts['unknown']}
    if any(type(value[key]) is not int or value[key] != count for key, count in expected.items()):
        raise ValueError('Evidence counts differ from obligations')
    if not value['model_assessed'] and (expected['supported'] or expected['counterevidence']):
        raise ValueError('Unassessed receipt cannot declare model support')
    for row in gaps:
        if type(row) is not dict or not {'obligation_id', 'code'} <= set(row):
            raise ValueError('Invalid missing-evidence query')
        if type(row['obligation_id']) is not str or row['obligation_id'] not in ids:
            raise ValueError('Gap refers to an unavailable obligation')
        _text(row['code'], 64)
        if 'query' in row:
            _text(row['query'], 400)
            if len(row['query'].encode('utf-8')) > 400:
                raise ValueError('Missing-evidence query exceeds its byte budget')
    for query in queries:
        _text(query, 400)
        if len(query.encode('utf-8')) > 400:
            raise ValueError('Missing-evidence query exceeds its byte budget')
    raw = _json(value)
    if len(raw.encode('utf-8')) > MAX_FEEDBACK_BYTES:
        raise ValueError('Evidence feedback exceeds its byte budget')
    return json.loads(raw)


def normalize_event_reasoning(value):
    # One pure projection is shared with history hydration. A legacy exact
    # quotation establishes neither which condition axis was assessed nor
    # that a prerequisite was satisfied in this case.
    from .research_event_reasoning import normalize_event_condition_schema
    original = _checked_event_reasoning(value)
    return _checked_event_reasoning(normalize_event_condition_schema(original))


def content_review(receipt):
    return ('needs_evidence' if receipt['gaps'] or receipt['unknown']
            or receipt['status'] in {'evidence_gaps', 'needs_evidence', 'not_assessed'}
            else 'review_required')


def bind_feedback(receipt, *, task_id, task_revision, step_id, artifact_version):
    return {'task_id': task_id, 'task_revision': task_revision, 'step_id': step_id,
            'artifact_version': artifact_version,
            'receipt_sha256': hashlib.sha256(_json(receipt).encode('utf-8')).hexdigest()}


def feedback_view(receipt, binding, *, task_id=None, step_id=None, artifact_version=None, current_task_revision=None,
                  review_kind='event'):
    if review_kind == 'task':
        from .research_task_review import normalize_task_review
        original = normalize_task_review(receipt)
    elif review_kind == 'event':
        original = _checked_event_reasoning(receipt)
    else:
        raise ValueError('Unknown evidence feedback kind')
    if type(binding) is not dict or set(binding) != {'task_id', 'task_revision', 'step_id', 'artifact_version', 'receipt_sha256'}:
        raise ValueError('Evidence feedback binding is unavailable')
    if (type(binding['task_revision']) is not int or binding['task_revision'] < 1
            or type(binding['artifact_version']) is not int or not 1 <= binding['artifact_version'] <= 32
            or (task_id is not None and binding['task_id'] != task_id)
            or (step_id is not None and binding['step_id'] != step_id)
            or (artifact_version is not None and binding['artifact_version'] != artifact_version)
            or (current_task_revision is not None and binding['task_revision'] > current_task_revision)):
        raise ValueError('Evidence feedback belongs to a different task artifact')
    # The immutable hash describes the ORIGINAL stored receipt. Verify it
    # before a conservative legacy projection; never retag/rewrite history or
    # discard a still-readable normal answer merely because its badge changed.
    expected = bind_feedback(original, **{key: binding[key] for key in
        ('task_id', 'task_revision', 'step_id', 'artifact_version')})
    if binding != expected:
        raise ValueError('Evidence feedback binding differs from its artifact')
    receipt = normalize_event_reasoning(original) if review_kind == 'event' else original
    result = {'binding': dict(binding), 'content_review': content_review(receipt),
            'task_review' if review_kind == 'task' else 'event_reasoning': receipt, 'next_step_suggestion': {
                'kind': 'request_missing_evidence' if content_review(receipt) == 'needs_evidence' else 'review_evidence',
                'queries': list(receipt['suggested_queries']), 'gaps': list(receipt['gaps']),
                'manual_replan_only': True, 'automatic_execution': False,
                'semantic_verified': False, 'goal_verified': False}}
    if review_kind == 'task':
        result['review_kind'] = 'task'
    elif receipt != original:
        result['receipt_projection'] = 'legacy_condition_unknown'
    return result


def feedback_lines(value):
    """Render metadata as escaped data, never instructions or clickable URLs."""
    def plain(text):
        return re.sub(r'([\\`*{}\[\]()#+.!_|>~-])', r'\\\1',
                      html.escape(' '.join(str(text).split()), quote=True))
    task = 'task_review' in value
    receipt = value['task_review'] if task else value['event_reasoning']
    detail = (f"模型评估交付覆盖 {receipt['addressed']} 项、具有来源定位 {receipt['supported']} 项、" if task else
              f"模型评估支持 {receipt['supported']} 项、")
    lines = ['### ' + ('原任务与验收项核对' if task else '证据核对与下一步'), '',
        f"执行已完成；内容状态：{value['content_review']}。{detail}"
        f"反证 {receipt['counterevidence']} 项、未知 {receipt['unknown']} 项；未验证因果或整体目标。", '']
    for row in receipt['obligations']:
        lines.append(f"- {plain(row['id'])} · {plain(row['kind'])} · {plain(row['status'])}。")
    for row in receipt['gaps']:
        lines.append(f"- 缺口 {plain(row['obligation_id'])}/{plain(row['code'])}" +
                     (f"：{plain(row['query'])}" if 'query' in row else '。'))
    for query in receipt['suggested_queries']:
        lines.append('- 建议补查：' + plain(query))
    lines += ['', '以上仅是待确认建议；修改计划仍须重新预览并明确确认，不会自动检索、付费重试或执行。', '']
    return lines
