"""Host-fixed task requirements assessed inside the existing answer call.

An exact answer witness proves only that a delivery exists. A source witness
proves only its location, not entailment. Neither is a semantic/goal certificate.
"""
from __future__ import annotations

import hashlib
import json
import re

from .research_event_reasoning import _source_map, _quotes
from .research_response_envelope import decode_research_envelope

MAX_REVIEW_BYTES = 4096
MAX_RESPONSE_BYTES = 131072
_REQ_KEYS = {'version', 'question', 'acceptance_criteria', 'input_kind', 'requirements_sha256'}
_STATUS = {'model_assessed_addressed', 'model_assessed_counterevidence', 'unknown'}
_BASIS = {'source_and_answer', 'source_only', 'answer_fulfilment_only', 'none'}
_RECEIPT_KEYS = {'status', 'requirements_sha256', 'answer_sha256', 'obligations', 'addressed',
                 'supported', 'counterevidence', 'unknown', 'gaps', 'suggested_queries',
                 'causality_verified', 'semantic_verified', 'goal_verified', 'model_assessed'}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _text(value, limit, *, empty=False):
    if (type(value) is not str or len(value) > limit or (not empty and not value.strip())
            or any(ord(c) < 32 and c not in '\n\t' for c in value)):
        raise ValueError('Invalid task review text')
    return value


def build_task_requirements(question, acceptance_criteria=None, *, input_kind='documents'):
    question = _text(question, 2000)
    criteria = [] if acceptance_criteria is None else acceptance_criteria
    if (type(criteria) is not list or len(criteria) > 12 or type(input_kind) is not str
            or input_kind not in {'documents', 'personal'}):
        raise ValueError('Invalid task requirements')
    criteria = [_text(value, 1000) for value in criteria]
    if len(set(criteria)) != len(criteria) or len(_json(criteria).encode()) > 8192:
        raise ValueError('Invalid task acceptance criteria')
    result = {'version': 1, 'question': question, 'acceptance_criteria': criteria, 'input_kind': input_kind}
    return {**result, 'requirements_sha256': _digest(_json(result))}


def validate_task_requirements(value):
    if type(value) is not dict or set(value) != _REQ_KEYS or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Invalid task requirements packet')
    expected = build_task_requirements(value['question'], value['acceptance_criteria'], input_kind=value['input_kind'])
    if value != expected:
        raise ValueError('Task requirements changed')
    return expected


def _requirements(value):
    return [{'id': 'Q1', 'kind': 'task', 'text': value['question']},
            *[{'id': f'C{i + 1}', 'kind': 'criterion', 'text': text}
              for i, text in enumerate(value['acceptance_criteria'])]]


def render_task_requirements(value):
    value = validate_task_requirements(value)
    # Full immutable texts already occupy their trusted project_context slots.
    # IDs designate those texts, not a model-selected shorter surrogate goal.
    rows = [{'id': row['id'], 'kind': row['kind'], 'slot':
             'original_step_question' if row['id'] == 'Q1' else
             f"phase_acceptance_criteria[{int(row['id'][1:]) - 1}]"} for row in _requirements(value)]
    return _json({'version': 1, 'requirements_sha256': value['requirements_sha256'], 'requirements': rows})


def task_review_contract():
    return ('Finish the normal requested answer FIRST. Optional sparse annotation in the SAME JSON: '
        '"task_assessment":{"requirements":[{"id":"Q1 or C#",'
        '"verdict":"addressed|counterevidence|unknown","evidence":[{"label":"S1",'
        '"quote":"exact retained source text","start":0}],'
        '"answer_witness":{"quote":"exact contiguous answer text","start":0},'
        '"missing_query":"specific missing evidence"}]}. requirements:[] is valid. Annotate at most TWO '
        'host Q1/C# only when direct source/answer witnesses are already clear. The host keeps EVERY original '
        'requirement; omitted items stay unknown, never satisfied. Do not exhaustively prove all criteria, '
        'search for missing proof, delay the answer, or replace a requirement with an easier one. '
        'Up to three complete joint source witnesses and one answer witness per item; if the full proof is not '
        'short and available, abstain rather than truncate it. start is optional only for a unique quote. Draft existence is '
        'delivery coverage, NOT factual support, causality or goal verification. No source support '
        'means factual support remains unknown. Preserve normal answer, citations and any separate '
        'event_assessment/trial_spec fields; do not claim execution or repair via another call.')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Repeated task response field')
        result[key] = value
    return result


def _decode(raw):
    return decode_research_envelope(raw, max_bytes=MAX_RESPONSE_BYTES)


def task_response_without_assessment(raw):
    """Mechanical composition for the existing event/trial parsers."""
    value = _decode(raw)
    if value is None:
        return raw
    return _json({key: value[key] for key in value if key != 'task_assessment'})


def _answer_witness(value, answer):
    if type(value) is not dict or set(value) - {'quote', 'start'}:
        return None
    quote, start = value.get('quote'), value.get('start')
    if type(quote) is not str or not quote.strip() or len(quote) > 400:
        return None
    if start is None:
        start = answer.find(quote)
        if start < 0 or answer.find(quote, start + 1) >= 0:
            return None
    if type(start) is not int or start < 0 or answer[start:start + len(quote)] != quote:
        return None
    return {'span': [start, start + len(quote)], 'quote_sha256': _digest(quote)}


def _query(value):
    if type(value) is not str:
        return ''
    return value.strip().encode('utf-8')[:400].decode('utf-8', errors='ignore')


def parse_task_response(raw, requirements, sources, *, final_answer=None):
    requirements = validate_task_requirements(requirements)
    envelope = _decode(raw)
    answer = envelope['answer'] if envelope is not None else raw
    if final_answer is not None:
        answer = final_answer
    if type(answer) is not str or len(answer.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError('Final task answer is unavailable')
    assessment = envelope.get('task_assessment') if envelope else None
    proposed = assessment.get('requirements') if type(assessment) is dict else None
    by_id, repeated = {}, set()
    if type(proposed) is list and len(proposed) <= 13:
        for row in proposed:
            identifier = row.get('id') if type(row) is dict else None
            if type(identifier) is not str:
                continue
            if identifier in by_id:
                repeated.add(identifier)
            by_id[identifier] = row
    else:
        proposed = None
    source_map = _source_map(sources)
    obligations, queries = [], []
    for need in _requirements(requirements):
        raw_row = by_id.get(need['id']) if need['id'] not in repeated else None
        row = {'id': need['id'], 'kind': need['kind'], 'status': 'unknown', 'basis': 'none',
               'evidence': [], 'gaps': ['requirement_unassessed']}
        query = ''
        if raw_row is not None:
            evidence, invalid = _quotes(raw_row.get('evidence', []), source_map)
            if type(raw_row.get('evidence', [])) is not list or len(raw_row.get('evidence', [])) > 3:
                evidence, invalid = [], True
            witness = _answer_witness(raw_row.get('answer_witness'), answer)
            verdict = raw_row.get('verdict')
            if invalid:
                row['gaps'] = ['source_witness_invalid']
            elif verdict == 'addressed' and witness:
                row.update(status='model_assessed_addressed', basis='source_and_answer' if evidence else 'answer_fulfilment_only',
                           evidence=evidence, answer_witness=witness,
                           gaps=[] if evidence or requirements['input_kind'] == 'personal' else ['factual_support_unassessed'])
            elif verdict == 'counterevidence' and evidence:
                row.update(status='model_assessed_counterevidence', basis='source_and_answer' if witness else 'source_only',
                           evidence=evidence, gaps=['requirement_counterevidence'])
                if witness:
                    row['answer_witness'] = witness
            elif verdict == 'addressed':
                row['gaps'] = ['answer_witness_missing_or_changed']
            else:
                row['gaps'] = ['requirement_unresolved']
            if row['gaps']:
                query = _query(raw_row.get('missing_query'))
                if query:
                    queries.append(query)
        if row['gaps'] and not query:
            queries.append(_query(f"{need['id']}: {need['text']} | missing evidence or delivery"))
        obligations.append(row)
    addressed = sum(row['status'] == 'model_assessed_addressed' for row in obligations)
    unknown = sum(row['status'] == 'unknown' for row in obligations)
    gaps = [{'obligation_id': row['id'], 'code': row['gaps'][0]} for row in obligations if row['gaps']]
    receipt = {'status': 'evidence_gaps' if gaps else 'assessed_with_caveats',
        'requirements_sha256': requirements['requirements_sha256'], 'answer_sha256': _digest(answer),
        'obligations': obligations, 'addressed': addressed,
        'supported': sum(row['status'] == 'model_assessed_addressed' and bool(row['evidence']) for row in obligations),
        'counterevidence': sum(row['status'] == 'model_assessed_counterevidence' for row in obligations),
        'unknown': unknown, 'gaps': gaps, 'suggested_queries': [],
        'causality_verified': False, 'semantic_verified': False, 'goal_verified': False,
        'model_assessed': proposed is not None}
    if proposed is None:
        receipt['status'] = 'not_assessed'
    while len(_json(receipt).encode()) > MAX_REVIEW_BYTES:
        # A paid answer remains a usable draft. Never retain an addressed
        # verdict after dropping one member of its joint proof. Downgrade the
        # WHOLE largest proof-bearing need, with an explicit capacity gap.
        candidates = [(len(_json(row).encode()), index) for index, row in enumerate(obligations)
                      if row['evidence'] or 'answer_witness' in row]
        if not candidates:
            raise ValueError('Task review fixed requirements exceed their byte budget')
        index = max(candidates)[1]
        obligations[index].update(status='unknown', basis='none', evidence=[], gaps=['receipt_budget_exceeded'])
        obligations[index].pop('answer_witness', None)
        receipt.update(status='evidence_gaps',
            addressed=sum(row['status'] == 'model_assessed_addressed' for row in obligations),
            supported=sum(row['status'] == 'model_assessed_addressed' and bool(row['evidence']) for row in obligations),
            counterevidence=sum(row['status'] == 'model_assessed_counterevidence' for row in obligations),
            unknown=sum(row['status'] == 'unknown' for row in obligations),
            gaps=[{'obligation_id': row['id'], 'code': row['gaps'][0]} for row in obligations if row['gaps']])
    for query in list(dict.fromkeys(queries))[:8]:
        candidate = {**receipt, 'suggested_queries': [*receipt['suggested_queries'], query]}
        if len(_json(candidate).encode()) <= MAX_REVIEW_BYTES:
            receipt = candidate
    return answer, receipt


def _checked_review(receipt, sources=None, answer=None):
    """Recheck source/answer coordinates without another model or source search."""
    try:
        if (type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS or len(_json(receipt).encode()) > MAX_REVIEW_BYTES
                or any(receipt.get(key) is not False for key in ('causality_verified', 'semantic_verified', 'goal_verified'))
                or type(receipt.get('model_assessed')) is not bool
                or not all(type(receipt.get(key)) is str and re.fullmatch('[0-9a-f]{64}', receipt[key])
                           for key in ('requirements_sha256', 'answer_sha256'))):
            return None
        rows = receipt.get('obligations')
        if type(rows) is not list or not 1 <= len(rows) <= 13:
            return None
        ids = [row.get('id') if type(row) is dict else None for row in rows]
        if ids != ['Q1', *[f'C{i}' for i in range(1, len(rows))]]:
            return None
        if answer is not None and (type(answer) is not str or _digest(answer) != receipt['answer_sha256']):
            return None
        source_map = _source_map(sources) if sources is not None else None
        for row in rows:
            if (set(row) - {'id', 'kind', 'status', 'basis', 'evidence', 'gaps', 'answer_witness'}
                    or row.get('kind') != ('task' if row['id'] == 'Q1' else 'criterion')
                    or row.get('status') not in _STATUS or row.get('basis') not in _BASIS
                    or type(row.get('evidence')) is not list or len(row['evidence']) > 3
                    or type(row.get('gaps')) is not list or len(row['gaps']) > 4
                    or any(type(code) is not str or len(code) > 64 for code in row['gaps'])):
                return None
            for ref in row['evidence']:
                if type(ref) is not dict or set(ref) != {'label', 'source_id', 'span', 'quote_sha256'}:
                    return None
                span = ref['span']
                if (type(ref['label']) is not str or not re.fullmatch(r'S(?:[1-9]|10)', ref['label'])
                        or type(ref['source_id']) is not str or not 1 <= len(ref['source_id']) <= 256
                        or type(span) is not list or len(span) != 2 or any(type(i) is not int for i in span)
                        or not 0 <= span[0] < span[1] <= 2_000_000
                        or type(ref['quote_sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', ref['quote_sha256'])):
                    return None
                if source_map is not None:
                    source = source_map.get(ref['label'])
                    if source is None or source['id'] != ref['source_id']:
                        return None
                    start, end = (i - source.get('excerpt_start', 0) for i in span)
                    if not 0 <= start < end <= len(source['content']) or _digest(source['content'][start:end]) != ref['quote_sha256']:
                        return None
            witness = row.get('answer_witness')
            if witness is not None:
                if (type(witness) is not dict or set(witness) != {'span', 'quote_sha256'}
                        or type(witness['span']) is not list or len(witness['span']) != 2
                        or any(type(i) is not int for i in witness['span'])
                        or not 0 <= witness['span'][0] < witness['span'][1] <= MAX_RESPONSE_BYTES
                        or type(witness['quote_sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', witness['quote_sha256'])):
                    return None
                if answer is not None:
                    a, b = witness['span']
                    if b > len(answer) or _digest(answer[a:b]) != witness['quote_sha256']:
                        return None
            if (row['status'] == 'model_assessed_addressed' and witness is None
                    or row['status'] == 'model_assessed_counterevidence' and not row['evidence']):
                return None
            actual_basis = ('source_and_answer' if row['evidence'] and witness else 'source_only'
                            if row['evidence'] else 'answer_fulfilment_only' if witness else 'none')
            if row['basis'] != actual_basis:
                return None
        expected = {'addressed': sum(row['status'] == 'model_assessed_addressed' for row in rows),
                    'supported': sum(row['status'] == 'model_assessed_addressed' and bool(row['evidence']) for row in rows),
                    'counterevidence': sum(row['status'] == 'model_assessed_counterevidence' for row in rows),
                    'unknown': sum(row['status'] == 'unknown' for row in rows)}
        if any(type(receipt.get(key)) is not int or receipt[key] != value for key, value in expected.items()):
            return None
        if receipt['model_assessed'] is False and expected['unknown'] != len(rows):
            return None
        if type(receipt.get('gaps')) is not list or type(receipt.get('suggested_queries')) is not list:
            return None
        if len(receipt['gaps']) > 13 or len(receipt['suggested_queries']) > 8:
            return None
        expected_gaps = [{'obligation_id': row['id'], 'code': row['gaps'][0]} for row in rows if row['gaps']]
        if receipt['gaps'] != expected_gaps or any(type(q) is not str or not q.strip() or len(q.encode()) > 400
                                                for q in receipt['suggested_queries']):
            return None
        if (receipt.get('status') not in {'not_assessed', 'evidence_gaps', 'assessed_with_caveats'}
                or receipt['status'] == 'assessed_with_caveats' and bool(expected_gaps)):
            return None
        return json.loads(_json(receipt))
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def normalize_task_review(receipt):
    value = _checked_review(receipt)
    if value is None:
        raise ValueError('Invalid task review receipt')
    return value


def hydrate_task_review(receipt, sources, answer=None):
    """Sources are live proof inputs; an optional answer checks its final body."""
    return _checked_review(receipt, sources, answer)
