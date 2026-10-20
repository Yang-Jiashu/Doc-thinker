"""Goal-first decision drafts; the host checks bounds, never invents a claim.

No model, tool, clock, storage or execution is used here. Coordinate/JSON and
budget validity do not establish factual support, useful science or permission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import re

from .research_citation_syntax import parse_citation_groups

VERSION = 11
MARKER = '\nHOST_DECISION_BRIEF_V1:\n'
TEXT_LIMITS = {'findings': 600, 'do_now': 240, 'not_now': 180, 'hypothesis': 240,
    'conditions_and_alternative': 240, 'validation_target': 160, 'validation_plan': 360,
    'refuter': 200, 'missing_policy': 160, 'day_one': 240, 'day_two': 240, 'privacy': 240}
MINUTE_FIELDS = ('day_one_minutes', 'day_two_minutes')
MAX_TEXT_CHARS = 2400
_POLICY = (
    '\nFlat JSON only. Complete original goal/question,no paper-review substitute. '
    '600-900 Chinese chars;total text<=2400. '
    'findings=S facts+confirmed context;do_now=next actions;not_now=deferrals;'
    'hypothesis=untested claim;conditions_and_alternative=scope+strongest rival;'
    'validation_target=actual predicted observable;validation_plan=concrete feasible method now;'
    'refuter=incompatible observation;missing_policy=missing/failure handling;'
    'day_one/day_two=work;minutes=nonnegative int|null(unknown);privacy=per-action consent. '
    'Keep hypothesis,target,quantifier,refuter;no forced AB/manual proxy/all-samples improvement. '
    'Review real model outputs,not imagined ones;materials are not measurements. '
    'Unknown resources/methods stay unconfirmed,not0/no fabricated experiment. '
    'Cite current [S#] only;prior analyses/memory are not paper proof. '
    'Honor resources/withdrawals/approvals;not executed.'
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def schema():
    """Provider syntax shape only; local validation owns business bounds."""
    fields = {key: {'type': 'string', 'minLength': 1, 'maxLength': cap} for key, cap in TEXT_LIMITS.items()}
    fields.update({key: {'type': ['integer', 'null']} for key in MINUTE_FIELDS})
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def contract():
    prompt = MARKER + '{' + ','.join(k + ':string' for k in TEXT_LIMITS)
    prompt += ',day_one_minutes:int|null,day_two_minutes:int|null}'
    value = prompt + _POLICY
    if len(value.encode()) > 1600:
        raise ValueError('decision_brief_contract_limit')
    return {'version': VERSION, 'prompt': prompt, 'policy': _POLICY, 'contract_sha256': _sha(value)}


def input_sha256(messages):
    current = contract()
    if (type(messages) is not list or len(messages) != 2
            or any(type(row) is not dict or set(row) != {'role', 'content'}
                   or type(row['content']) is not str for row in messages)
            or [row['role'] for row in messages] != ['system', 'user']
            or messages[0]['content'].count(current['policy']) != 1
            or messages[1]['content'].count(current['prompt']) != 1):
        raise ValueError('decision_brief_input_changed')
    return _sha(_json(messages))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _constant(_):
    raise ValueError('nonfinite_number')


def _text(value, cap, *, empty=False):
    if type(value) is not str or len(value) > cap or (not empty and not value.strip()):
        return False
    try:
        value.encode()
    except UnicodeError:
        return False
    return True


def _source_packet(sources):
    if type(sources) is not list or len(sources) > 64:
        raise ValueError('sources_invalid')
    packet, ids, labels = [], set(), set()
    for row in sources:
        if type(row) is not dict:
            raise ValueError('sources_invalid')
        identifier = row.get('id', row.get('source_id'))
        label, content = row.get('label'), row.get('content')
        if (not _text(identifier, 256) or not _text(content, 4096)
                or type(label) is not str or re.fullmatch(r'S[1-9][0-9]?', label) is None
                or identifier in ids or label in labels
                or row.get('source_id', identifier) != identifier
                or type(row.get('revoked', False)) is not bool):
            raise ValueError('sources_invalid')
        if row.get('revoked', False) or row.get('status') in ('revoked', 'withdrawn', 'deleted'):
            raise ValueError('source_revoked')
        if ('excerpt_start' in row) != ('excerpt_end' in row):
            raise ValueError('source_span_invalid')
        span = row.get('span', [row.get('excerpt_start', 0), row.get('excerpt_end', len(content))])
        if (type(span) is not list or len(span) != 2 or any(type(n) is not int for n in span)
                or not 0 <= span[0] < span[1] <= 2_000_000 or span[1] - span[0] != len(content)
                or 'excerpt_start' in row and span != [row['excerpt_start'], row['excerpt_end']]):
            raise ValueError('source_span_invalid')
        digest = _sha(content)
        if any(key in row and row[key] != digest for key in ('sha256', 'quote_sha256')):
            raise ValueError('source_hash_mismatch')
        packet.append({'source_id': identifier, 'label': label, 'span': span[:], 'sha256': digest})
        ids.add(identifier); labels.add(label)
    return packet


def _citations(text, labels):
    result = parse_citation_groups(text)
    if result['truncated'] or any(label not in labels for label in result['labels']):
        raise ValueError('citation_unavailable')
    covered = {row['start'] for row in result['groups']}
    # Legacy P/J and malformed citation-looking brackets cannot disappear from
    # the validation just because the shared S/K parser does not accept them.
    if any(match.start() not in covered for match in re.finditer(r'[\[［][ \t]*[SKPJ][0-9]', text, re.I)):
        raise ValueError('citation_invalid')


def _binding(parsed):
    return _sha(_json({key: parsed[key] for key in (
        'answer', 'brief', 'task', 'task_sha256', 'source_packet', 'evidence_packet_sha256',
        'allocation', 'validation')}))


def parse(raw, sources, *, trusted_budget_minutes, question, goal=None, constraints=None):
    brief, answer, packet, codes = None, '', [], []
    task = {'question': question, 'goal': goal, 'constraints': constraints}
    task_valid = (_text(question, 24000) and all(value is None or _text(value, 24000, empty=True)
                                               for value in (goal, constraints)))
    if not task_valid:
        codes.append('task_invalid')
        task = {'question': question if _text(question, 24000, empty=True) else None,
                'goal': goal if goal is None or _text(goal, 24000, empty=True) else None,
                'constraints': constraints if constraints is None or _text(constraints, 24000, empty=True) else None}
    try:
        packet = _source_packet(sources)
    except ValueError as error:
        codes.append(str(error))
    labels = {row['label'] for row in packet}
    data = None
    try:
        if type(raw) is not str or len(raw.encode()) > 32768:
            raise ValueError('response_limit')
        text = raw.strip()
        fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.I)
        if fence:
            text = fence[1]
        data = json.loads(text, object_pairs_hook=_unique, parse_constant=_constant)
        if type(data) is not dict:
            raise ValueError('schema_invalid')
        if _text(data.get('findings'), TEXT_LIMITS['findings']) and not any(c.startswith('source') for c in codes):
            try:
                _citations(data['findings'], labels)
                answer = data['findings']
            except ValueError:
                pass
        if set(data) != set(TEXT_LIMITS) | set(MINUTE_FIELDS):
            raise ValueError('schema_invalid')
        if any(not _text(data[key], cap) for key, cap in TEXT_LIMITS.items()):
            raise ValueError('field_limit')
        if sum(len(data[key]) for key in TEXT_LIMITS) > MAX_TEXT_CHARS:
            raise ValueError('total_text_limit')
        for key in TEXT_LIMITS:
            _citations(data[key], labels)
        if any(data[key] is not None and (type(data[key]) is not int or not 0 <= data[key] <= 10080) for key in MINUTE_FIELDS):
            raise ValueError('minutes_invalid')
        brief = deepcopy(data)
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError) as error:
        code = str(error)
        codes.append(code if code in {'response_limit', 'duplicate_key', 'nonfinite_number', 'schema_invalid',
            'field_limit', 'total_text_limit', 'citation_unavailable', 'citation_invalid', 'minutes_invalid'} else 'invalid_decision_brief')
    budget_known = type(trusted_budget_minutes) is int and 0 <= trusted_budget_minutes <= 10080
    if trusted_budget_minutes is not None and not budget_known:
        codes.append('trusted_budget_invalid')
    elif trusted_budget_minutes is None:
        codes.append('budget_unconfirmed')
    minutes = [brief[key] for key in MINUTE_FIELDS] if brief else [None, None]
    complete = all(value is not None for value in minutes)
    known = sum(value for value in minutes if value is not None)
    planned = known if complete else None
    if brief is not None and not complete:
        codes.append('minutes_unconfirmed')
    if budget_known and known > trusted_budget_minutes:
        codes.append('budget_exceeded')
    allocation = {'planned_minutes': planned, 'known_minutes': known,
        'remaining_minutes': trusted_budget_minutes - planned if budget_known and planned is not None else None,
        'trusted_budget_minutes': trusted_budget_minutes if budget_known else None,
        'budget_verified': budget_known and complete and known <= trusted_budget_minutes}
    codes = sorted(set(codes))
    parsed = {'answer': answer, 'brief': brief, 'task': task, 'task_sha256': _sha(_json(task)),
        'source_packet': packet, 'evidence_packet_sha256': _sha(_json(packet)), 'allocation': allocation,
        'validation': {'version': VERSION, 'status': 'draft' if codes else 'review_required', 'violations': codes,
            'structural_validation_passed': not codes, 'projection_complete': not codes,
            'scope': 'goal_bound_decision_draft_not_semantic_or_measurement_verification',
            'semantic_verified': False, 'measurement_verified': False, 'observations_present': False,
            'execution_authorized': False, 'executed': False, 'extra_model_calls': 0}}
    parsed['_binding_sha256'] = _binding(parsed)
    return parsed


def _checked(parsed):
    try:
        if type(parsed) is not dict or parsed.get('_binding_sha256') != _binding(parsed):
            raise ValueError('decision_brief_binding_changed')
    except (KeyError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError('decision_brief_binding_changed') from None
    return parsed


def render(parsed):
    parsed = _checked(parsed)
    if any(code in {'sources_invalid', 'source_revoked', 'source_span_invalid', 'source_hash_mismatch'}
           for code in parsed['validation']['violations']):
        return '来源已撤回或无法核对，决策简报已收起；不得据此复用内容或执行操作。'
    brief = parsed['brief']
    if brief is None:
        return (html.escape(parsed['answer'], quote=False) or '未取得可展示正文。') + '\n\n决策简报草稿未通过字段、来源或预算检查；未执行、未自动修补。'
    shown = {key: html.escape(brief[key], quote=False) for key in TEXT_LIMITS}
    lines = ['当前依据与现在做', shown['findings'], shown['do_now'], '',
        '暂不做', shown['not_now'], '',
        '假设、条件与最强替代解释', shown['hypothesis'], shown['conditions_and_alternative'], '',
        '观察目标、检验与反证', '目标：' + shown['validation_target'], '方案：' + shown['validation_plan'],
        '反证：' + shown['refuter'], '缺失与失败：' + shown['missing_policy'], '',
        '两天安排与预算余量']
    for key, title in (('day_one', '第1天'), ('day_two', '第2天')):
        value = brief[key + '_minutes']
        minutes = str(value) + '分钟' if value is not None else '工时待确认（未按0计算）'
        lines.append(title + '（' + minutes + '）：' + shown[key])
    allocation = parsed['allocation']
    if allocation['planned_minutes'] is None:
        lines.append('完整计划工时及预算余量待确认；已知工时小计' + str(allocation['known_minutes']) + '分钟。')
    else:
        lines.append('拟定总工时：' + str(allocation['planned_minutes']) + '分钟。')
    if allocation['remaining_minutes'] is not None:
        lines.append('相对宿主预算的剩余分钟数：' + str(allocation['remaining_minutes']) + '（负值表示超限，不是可用余量）。')
    elif allocation['trusted_budget_minutes'] is None:
        lines.append('宿主总预算待确认，余量未知。')
    lines.extend(('', '隐私与逐次同意边界', shown['privacy'], ''))
    if parsed['validation']['status'] != 'review_required':
        lines.append('草稿仍有未通过或待确认项；不构成有效检验方案认证。')
    lines.append('仅检查字段、来源定位和工时算术；事实支持、目标与测法仍待复核。方案未执行，不授权写入或导出。')
    return '\n'.join(lines)


def receipt(parsed, source_ids, *, requested_contract=None):
    if requested_contract != contract():
        raise ValueError('invalid_decision_brief_contract')
    parsed = _checked(parsed)
    ids = [row['source_id'] for row in parsed['source_packet']]
    if (type(source_ids) not in (list, tuple) or any(type(value) is not str for value in source_ids)
            or len(set(source_ids)) != len(source_ids) or set(source_ids) != set(ids)):
        raise ValueError('decision_brief_source_ids_changed')
    return {**deepcopy(parsed['validation']), **deepcopy(parsed['allocation']),
        'contract_sha256': contract()['contract_sha256'], 'manifest_sha256': _sha(_json(parsed['brief'])) if parsed['brief'] else None,
        'source_ids': list(source_ids), 'source_packet': deepcopy(parsed['source_packet']),
        'evidence_packet_sha256': parsed['evidence_packet_sha256'], 'task': deepcopy(parsed['task']),
        'task_sha256': parsed['task_sha256'],
        **{key + '_sha256': _sha(value) if value is not None else None for key, value in parsed['task'].items()}}


def unavailable(previous=None, *, requested_contract=None):
    if requested_contract is not None and requested_contract != contract():
        raise ValueError('invalid_decision_brief_contract')
    return {'version': VERSION, 'status': 'source_unavailable', 'violations': ['source_unavailable'],
        'source_ids': [], 'source_packet': [], 'contract_sha256': contract()['contract_sha256'],
        'manifest_sha256': None, 'evidence_packet_sha256': None, 'task_sha256': None,
        'planned_minutes': None, 'remaining_minutes': None, 'structural_validation_passed': False,
        'projection_complete': False, 'semantic_verified': False, 'measurement_verified': False,
        'observations_present': False, 'execution_authorized': False, 'executed': False, 'extra_model_calls': 0,
        'scope': 'goal_bound_decision_draft_not_semantic_or_measurement_verification'}
