"""Deterministic, non-executing presentation of an explicitly requested trial."""
from __future__ import annotations

from .research_trial_contract import MAX_RESPONSE_BYTES, VIOLATIONS, validate_trial_spec

MAX_RENDER_BYTES = MAX_RESPONSE_BYTES + 16384

_NUMERATOR = {'correct_answers': '正确答案数', 'grounded_answers': '有原文依据的答案数',
              'completed_answers': '完成的问题数', 'failed_answers': '失败的问题数'}
_OUTCOME = {'pass': '通过', 'stop': '停止', 'collect_evidence': '继续取证'}


def _text(value):
    return str(value).replace('\n', ' ').replace('\r', ' ').replace('|', '\\|')


def _bounded(value, limit=MAX_RESPONSE_BYTES):
    if not isinstance(value, str):
        return '', False
    raw = value.encode('utf-8', errors='replace')
    return raw[:limit].decode('utf-8', errors='ignore'), len(raw) > limit


def render_trial_response(parsed):
    """A valid shape is neither semantic proof nor permission to run anything."""
    if not isinstance(parsed, dict):
        parsed = {'status': 'invalid', 'violations': ['schema_invalid']}
    answer = parsed.get('answer', '')
    verdict = validate_trial_spec(parsed.get('trial_spec'))
    safe_answer, oversized = _bounded(answer)
    valid_answer = isinstance(answer, str) and bool(answer.strip()) and len(answer) <= 24000 and not oversized
    if parsed.get('status') != 'valid' or verdict['status'] != 'valid' or not valid_answer:
        raw = parsed.get('raw_draft')
        draft, draft_truncated = _bounded(raw if isinstance(raw, str) else safe_answer)
        supplied_codes = parsed.get('violations', ())
        supplied_codes = supplied_codes[:32] if isinstance(supplied_codes, (list, tuple)) else ()
        codes = sorted({code for code in supplied_codes if isinstance(code, str) and code in VIOLATIONS})
        codes = codes or verdict['violations'] or ['schema_invalid']
        return (draft + '\n\n---\n试点草稿：结构校验未通过，未执行任何试验。'
                '\n待处理：' + '、'.join(codes) + '。'
                + ('\n原始输出超过保留上限，以上为截断草稿。'
                   if parsed.get('raw_truncated') or draft_truncated or oversized else ''))
    spec = parsed['trial_spec']
    inputs = spec['inputs']
    arms = {item['id']: item for item in spec['arms']}
    baseline, candidate = arms['baseline'], arms['candidate']
    lines = [answer.rstrip(), '', '### 试点建议（待确认，未执行）', '',
             '假设：' + _text(spec['hypothesis']),
             '唯一待测项：' + _text(spec['test_variable']),
             '基线：' + _text(baseline['tested_value']) + '；候选：' + _text(candidate['tested_value']) + '。',
             '固定模型：' + _text(baseline['model']) + '；固定配置：' +
             ', '.join(_text(key) + '=' + _text(value) for key, value in sorted(baseline['config'].items())) + '。',
             '同一题集：' + _text(inputs['question_set']) + '；同一历史：' + _text(inputs['history_set']) + '。',
             f"每臂计划 {inputs['planned_questions']} 题；最低有效配对 {inputs['minimum_paired_questions']} 题。",
             '低于最低配对数时不判通过或停止：' + _text(inputs['insufficient_data_step']), '',
             '| 指标 | 计数口径 | 方向 |', '|---|---|---|']
    for metric in spec['metrics']:
        lines.append('| ' + metric['id'] + ' | ' + _NUMERATOR[metric['numerator']] + '/全部计划问题数；'
                     + _text(metric['scoring']) + ' | ' + ('越高越好' if metric['direction'] == 'higher' else '越低越好') + ' |')
    lines += ['', '失败、缺失均按失败计入全部计划问题的分母，不从分母剔除。',
              '下列规则仅适用于达到最低配对数时；Δ = 候选率 − 基线率，单位为比例差（不是相对增长率）。', '',
              '| 结论 | 条件 | 下一步 |', '|---|---|---|']
    for rule in spec['rules']:
        conditions = []
        for interval in rule['ranges']:
            left = '≤' if interval['lower_inclusive'] else '<'
            right = '≤' if interval['upper_inclusive'] else '<'
            conditions.append(f"{interval['lower']} {left} Δ{interval['metric']} {right} {interval['upper']}")
        lines.append('| ' + _OUTCOME[rule['outcome']] + ' | ' + ' 且 '.join(conditions) + ' | ' + _text(rule['next_step']) + ' |')
    lines += ['', '阈值来源：' + ('模型提出的待确认建议' if spec['threshold_origin'] == 'proposed' else '声明来自用户，仍待确认') + '。',
              '仅校验内部结构与规则覆盖；未验证阈值合理性、指标有效性、来源事实或实际可执行性。费用和延迟不在此结构校验范围。']
    rendered, truncated = _bounded('\n'.join(lines), MAX_RENDER_BYTES)
    # The finite field bounds keep valid parser output below this limit. Do
    # not label a caller-supplied oversized result a complete rendered plan.
    if truncated:
        return _bounded(rendered, MAX_RENDER_BYTES - 256)[0] + '\n\n试点草稿：输出超过渲染上限，未执行。'
    return rendered
