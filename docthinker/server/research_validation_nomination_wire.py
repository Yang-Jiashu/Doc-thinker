"""Two scalar wire fields over the unchanged host-owned nomination compiler."""
from __future__ import annotations

import hashlib
import json
import re

from . import research_validation_nomination as base
from .research_validation_delivery import _unique, _invalid_constant

VERSION = 9
MARKER = '\nHOST_PROTOCOL_NOMINATIONS_V2:\n'
_INNER = [key for key in base._FIELDS if key not in ('evidence_analysis', 'samples_json')]


def schema():
    return {'type': 'object', 'properties': {
        'evidence_analysis': {'type': 'string', 'minLength': 1, 'maxLength': 1800},
        'nominations_json': {'type': 'string', 'minLength': 1, 'maxLength': 12000}},
        'required': ['evidence_analysis', 'nominations_json'], 'additionalProperties': False}


def contract():
    inner = '{' + ','.join(key + ':string' for key in _INNER) + ',samples:[{label:S1,task:string}]}'
    prompt = MARKER + '{evidence_analysis:string,nominations_json:string};nominations_json=JSON string encoding:' + inner
    prompt += ';measure_kind=count|qualitative;direction=lower|higher.'
    policy = base._POLICY.replace('JSON only;all fields strings.',
                                 'JSON only;outer two strings;inner as shown.', 1)
    if len((prompt + policy).encode()) > 1600:
        raise ValueError('validation_contract_limit')
    return {'version': VERSION, 'prompt': prompt, 'policy': policy,
        'contract_sha256': hashlib.sha256((prompt + policy).encode()).hexdigest()}


def input_sha256(messages):
    current = contract()
    if (type(messages) is not list or len(messages) != 2
            or any(type(row) is not dict or set(row) != {'role', 'content'}
                   or type(row['content']) is not str for row in messages)
            or [row['role'] for row in messages] != ['system', 'user']
            or messages[0]['content'].count(current['policy']) != 1
            or messages[1]['content'].count(current['prompt']) != 1):
        raise ValueError('validation_input_changed')
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse(raw, sources, *, trusted_budget_minutes, question=None):
    analysis, parsed, failure, complete = '', None, 'response_bound', False
    try:
        if type(raw) is not str or len(raw.encode()) > 32768:
            raise ValueError('response_limit')
        failure = 'outer_json'
        text = raw.strip()
        fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.I)
        if fence:
            text = fence[1]
        outer = json.loads(text, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        failure = 'outer_schema'
        if type(outer) is not dict:
            raise ValueError('schema_invalid')
        value = outer.get('evidence_analysis')
        if type(value) is str and value.strip() and len(value) <= 1800:
            analysis = value.strip()
        if set(outer) != {'evidence_analysis', 'nominations_json'} or not analysis:
            raise ValueError('schema_invalid')
        encoded = outer['nominations_json']
        failure = 'nomination_bound'
        if type(encoded) is not str or not encoded.strip() or len(encoded.encode()) > 12000:
            raise ValueError('nomination_invalid')
        failure = 'nomination_json'
        nominations = json.loads(encoded, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        failure = 'nomination_schema'
        if type(nominations) is not dict or set(nominations) != set(_INNER) | {'samples'}:
            raise ValueError('nomination_invalid')
        # This declared representation conversion changes no nomination value.
        # Pairing/rules/budget remain exclusively in the same base compiler.
        data = {key: nominations[key] for key in _INNER}
        data.update(evidence_analysis=analysis, samples_json=json.dumps(nominations['samples'],
                    ensure_ascii=False, allow_nan=False))
        parsed = base.parse(json.dumps(data, ensure_ascii=False, allow_nan=False), sources,
            trusted_budget_minutes=trusted_budget_minutes, question=question)
        complete, failure = True, None
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        pass
    if parsed is None:
        parsed = base.parse('{}', sources, trusted_budget_minutes=trusted_budget_minutes, question=question)
    parsed['answer'] = analysis
    parsed['validation'] = {**parsed['validation'], 'version': VERSION,
        'wire_schema_checked': complete,
        'framing_failure': failure if failure is not None else parsed['validation'].get('framing_failure')}
    return parsed


def receipt(parsed, source_ids, *, requested_contract=None):
    if requested_contract != contract():
        raise ValueError('invalid_validation_contract')
    return {**base.receipt(parsed, source_ids, requested_contract=base.contract()),
        'version': VERSION, 'contract_sha256': contract()['contract_sha256']}


def unavailable(previous=None, *, requested_contract=None):
    return {**base.unavailable(), 'version': VERSION, 'contract_sha256': contract()['contract_sha256']}


def render(parsed):
    return base.render(parsed)
