"""Cheap, bounded exact-quantity warnings; never an entailment verifier.

Only explicit number/unit pairs in cited clauses are inspected. A direct match
means the same quantity, unit and comparator occur together in a current quote,
not that the quote establishes the claim, its scope, or scientific validity.
Offsets are codepoint offsets into unchanged input; no source text is returned.
This module neither edits prose nor authorizes execution or model calls.
"""
from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation

from .research_citation_syntax import parse_citation_groups

MAX_TEXT_CHARS = 24_000
MAX_SOURCES = 64
MAX_SOURCE_CHARS = 4_096
MAX_QUANTITIES = 32
MAX_DIAGNOSTICS = 16
_LABEL = re.compile(r'S[0-9]{1,6}\Z')
_CHINESE = {'零': '0', '一': '1', '二': '2', '两': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
_CN = ''.join(_CHINESE)
_NUMBER = r'(?<![0-9A-Za-z_.+\-−–—零一二两三四五六七八九十百千万])(?:[0-9]{1,12}(?:\.[0-9]{1,6})?|[' + _CN + r'])(?![0-9.零一二两三四五六七八九十百千万])'
_UNIT_ALIASES = (
    ('percentage_points', r'percentage[ \t]+points?|percentagepoints?|百分点|pp\b'),
    ('percent', r'percent(?:age)?\b|百分比|%'),
    ('citation_errors', r'citation[ \t]+errors?\b|错引(?:次数|数)?|引用错误(?:次数|数)?'),
    ('options', r'options?\b|选项'),
    ('lines', r'lines?\b|行'),
    ('rounds', r'rounds?\b|turns?\b|轮次|回合|轮'),
    ('minutes', r'minutes?\b|mins?\b|分钟'),
    ('hours', r'hours?\b|hrs?\b|小时'),
    ('tasks', r'tasks?\b|任务'),
)
_UNIT = '(?:' + '|'.join('(?:' + pattern + ')' for _, pattern in _UNIT_ALIASES) + ')'
_COMPARATORS = {
    '≤': 'le', '<=': 'le', '最多': 'le', '不超过': 'le', '至多': 'le',
    'at most': 'le', 'no more than': 'le',
    '≥': 'ge', '>=': 'ge', '至少': 'ge', '不少于': 'ge', 'at least': 'ge',
    '<': 'lt', '少于': 'lt', 'less than': 'lt',
    '>': 'gt', '多于': 'gt', 'more than': 'gt', '=': 'eq', '等于': 'eq',
}
_OP = '(?:' + '|'.join(re.escape(key) for key in sorted(_COMPARATORS, key=len, reverse=True)) + ')'
_FORWARD = re.compile(r'(?P<op>' + _OP + r')?[ \t]*(?P<number>' + _NUMBER + r')[ \t]*(?:个|项|条|次)?[ \t]*(?P<unit>' + _UNIT + ')', re.I)
_REVERSE = re.compile(r'(?P<unit>' + _UNIT + r')[ \t]*(?:数量|数|count)?[ \t]*(?P<op>' + _OP + r')[ \t]*(?P<number>' + _NUMBER + ')', re.I)
_REVERSE_COUNT = re.compile(r'(?P<unit>' + _UNIT + r')[ \t]*(?P<number>' + _NUMBER + r')[ \t]*(?:个|次|项|条)', re.I)
_UNIT_CHECKS = [(name, re.compile('(?:' + pattern + r')\Z', re.I)) for name, pattern in _UNIT_ALIASES]
_SENTENCE_BREAK = re.compile(r'[。！？!?；;\r\n]|(?<![0-9])\.(?![0-9])|(?<=[0-9])\.(?![0-9])')
_PROPOSAL = re.compile(r'我们(?:拟定|暂定|建议|计划|提出)|本(?:方案|产品|次设计)(?:拟定|暂定|建议|计划)|\bwe (?:propose|choose|plan|suggest)\b|\bour proposed\b', re.I)


def _quantities(text):
    matches = []
    for pattern in (_FORWARD, _REVERSE, _REVERSE_COUNT):
        for match in pattern.finditer(text):
            try:
                value = _CHINESE.get(match['number'], match['number'])
                value = str(Decimal(value).normalize())
            except InvalidOperation:
                continue
            unit = next(name for name, check in _UNIT_CHECKS if check.fullmatch(match['unit']))
            op = _COMPARATORS.get((match.groupdict().get('op') or '').lower(), 'eq')
            row = {'start': match.start(), 'end': match.end(), 'value': value,
                   'unit': unit, 'comparator': op}
            if not any(row['start'] < old['end'] and old['start'] < row['end'] for old in matches):
                matches.append(row)
            if len(matches) > MAX_QUANTITIES:
                return sorted(matches, key=lambda row: row['start']), True
    return sorted(matches, key=lambda row: row['start']), False


class _QuantityIndex:
    """Per-call quote snapshot, never a global cache or an authority token."""
    __slots__ = ('by_unit', 'invalid')

    def __init__(self, by_unit, invalid):
        self.by_unit, self.invalid = by_unit, invalid


def build_quantity_index(sources):
    """Build once per current source packet; reuse for its answer fields.

    This copies only bounded numeric metadata. It does not retain private text,
    and must not be cached across requests/users or used instead of the host's
    current source/withdrawal checks. Invalid/over-limit inputs stay visible.
    """
    if type(sources) is not list or len(sources) > MAX_SOURCES:
        return _QuantityIndex({}, True)
    result, invalid = {}, False
    for row in sources:
        if type(row) is not dict:
            invalid = True
            continue
        label, content = row.get('label'), row.get('content')
        if (type(label) is not str or not _LABEL.fullmatch(label)
                or type(content) is not str or len(content) > MAX_SOURCE_CHARS
                or row.get('revoked') is True or row.get('withdrawn') is True
                or row.get('deleted') is True or row.get('available') is False
                or row.get('status') in ('revoked', 'withdrawn', 'deleted')):
            invalid = True
            continue
        if label in result:
            # Ambiguous aliases are not allowed to select a convenient quote.
            result[label] = None
            invalid = True
            continue
        try:
            digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        except UnicodeError:
            invalid = True
            continue
        if any(key in row and row[key] != digest for key in ('sha256', 'quote_sha256')):
            invalid = True
            continue
        quantities, limited = _quantities(content)
        result[label] = None if limited else quantities
        invalid = invalid or limited
    by_unit = {}
    for label, rows in result.items():
        for quantity in rows or []:
            values = by_unit.setdefault(quantity['unit'], {})
            labels = values.setdefault((quantity['value'], quantity['comparator']), {})
            labels.setdefault(label, [quantity['start'], quantity['end']])
    return _QuantityIndex(by_unit, invalid)


def _sentence_bounds(text, quantity):
    start, end = 0, len(text)
    for boundary in _SENTENCE_BREAK.finditer(text):
        if boundary.end() <= quantity['start']:
            start = boundary.end()
        elif boundary.start() >= quantity['end']:
            end = boundary.start()
            break
    return start, end


def _bound_labels(text, groups, quantity, start, end):
    local = [group for group in groups if start <= group['start'] and group['end'] <= end]
    terminals = '.。！？!?；;'
    if start and text[start - 1] in terminals:
        # Citation bundles immediately after the preceding full stop belong
        # to that preceding sentence, not to an independent claim after them.
        cursor, preceding_suffix = start, []
        for group in local:
            if text[cursor:group['start']].strip():
                break
            preceding_suffix.append(group)
            cursor = group['end']
        local = [group for group in local if group not in preceding_suffix]
    if end < len(text) and text[end] in terminals:
        # Chinese prose commonly places the citation after the full stop.
        # Only punctuation + whitespace + an adjacent bundle is accepted;
        # intervening prose never lends a later sentence's citation backward.
        cursor = end + 1
        for group in groups:
            if group['start'] < cursor:
                continue
            if text[cursor:group['start']].strip():
                break
            local.append(group)
            cursor = group['end']
    after = [group for group in local if group['start'] >= quantity['end']]
    if after:
        selected = [after[0]]
        # Adjacent citation groups form one explicit source bundle. Prose
        # between two groups is a boundary, not permission to use either quote.
        for group in after[1:]:
            if text[selected[-1]['end']:group['start']].strip():
                break
            selected.append(group)
    else:
        before = [group for group in local if group['end'] <= quantity['start']]
        selected = before[-1:]
    return list(dict.fromkeys(label for group in selected for label in group['labels'] if label.startswith('S')))


def check_quantity_grounding(text, sources, *, field_role='fact', max_diagnostics=16):
    """Return bounded metadata, not support/entailment or a permission receipt.

    ``field_role='proposal'`` is a trusted caller annotation, not something the
    writer may use to promote its own settings into source facts. Explicit
    first-person proposal wording receives the same separate diagnostic. No
    unit conversion, ranges, arithmetic, implicit quantities, or causal/scope
    inference is performed; an absent warning is not proof of faithfulness.
    """
    result = {'status': 'no_quantity_claims', 'diagnostics': [], 'truncated': False,
              'checked_quantities': 0, 'semantic_verified': False,
              'text_modified': False, 'execution_authorized': False, 'extra_model_calls': 0}
    if (type(text) is not str or type(field_role) is not str or field_role not in ('fact', 'proposal')
            or type(max_diagnostics) is not int or not 1 <= max_diagnostics <= MAX_DIAGNOSTICS):
        result.update(status='needs_review', diagnostics=[{'code': 'input_invalid', 'span': None, 'source_labels': []}])
        return result
    scan = text[:MAX_TEXT_CHARS]
    index = sources if type(sources) is _QuantityIndex else build_quantity_index(sources)
    invalid = index.invalid
    citations = parse_citation_groups(scan)
    quantities, limited = _quantities(scan)
    result['truncated'] = len(scan) != len(text) or citations['truncated'] or limited
    review = invalid or result['truncated']
    proposed = False
    for quantity in quantities[:MAX_QUANTITIES]:
        start, end = _sentence_bounds(scan, quantity)
        labels = _bound_labels(scan, citations['groups'], quantity, start, end)
        if not labels:
            continue
        result['checked_quantities'] += 1
        # Scope the proposal marker to this comma/colon-separated clause so
        # 'we propose X, while the paper reports Y' does not exempt Y.
        clause_start = max(scan.rfind(mark, start, quantity['start']) for mark in ('，', ',')) + 1
        prefix = scan[max(start, clause_start):quantity['start']]
        if field_role == 'proposal' or _PROPOSAL.search(prefix):
            code, matches = 'proposed_not_source_claim', []
            proposed = True
        else:
            matches = []
            source_matches = index.by_unit.get(quantity['unit'], {}).get((quantity['value'], quantity['comparator']), {})
            for label in labels:
                if label in source_matches:
                    matches.append({'source_label': label, 'span': source_matches[label][:]})
            code = 'direct_numeric_match' if matches else 'quantity_not_directly_grounded'
            review = review or not matches
        if len(result['diagnostics']) < max_diagnostics:
            result['diagnostics'].append({'code': code, 'span': [quantity['start'], quantity['end']],
                'unit': quantity['unit'], 'comparator': quantity['comparator'],
                'source_labels': labels, 'matches': matches})
        else:
            result['truncated'] = True
            review = True
    if invalid or result['truncated']:
        code = 'source_input_unavailable' if invalid else 'scan_limit_reached'
        if len(result['diagnostics']) < max_diagnostics:
            result['diagnostics'].append({'code': code, 'span': None, 'source_labels': []})
    result['status'] = ('needs_review' if review else 'proposed_not_source_claim' if proposed
                        else 'direct_numeric_match' if result['checked_quantities'] else 'no_quantity_claims')
    return result
