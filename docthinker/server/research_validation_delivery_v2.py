"""Scalar wire envelope; paired semantics stay in the unchanged host compiler.

The outer portable schema controls framing, not reasoning. The inner JSON is
decoded and fully checked locally; no value is guessed, repaired or retried.
Recognizable prose remains available if the proposed design fails validation.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import research_validation_delivery as base

VERSION = 7
MARKER = '\nHOST_VALIDATION_DESIGN_V2:\n'
_PLAN_SHAPE = base._SHAPE.split('validation_design:', 1)[1][:-1]
_FENCE = re.compile(r'\A```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\Z', re.I)


def schema():
    return {'type': 'object', 'properties': {
        'answer': {'type': 'string', 'minLength': 1, 'maxLength': 40000},
        'validation_design': {'type': 'string', 'minLength': 1, 'maxLength': 32768}},
        'required': ['answer', 'validation_design'], 'additionalProperties': False}


def contract():
    prompt = MARKER + '{answer:string,validation_design:string};validation_design=JSON string encoding:' + _PLAN_SHAPE
    policy = base._POLICY
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


def parse(raw, sources, *, trusted_budget_minutes):
    answer, parsed, framing_valid, unwrapped = '', None, False, False
    phase, failure = 'response_bound', None
    try:
        if type(raw) is not str or len(raw.encode()) > 65536:
            raise ValueError('response_limit')
        text = raw.strip()
        match = _FENCE.fullmatch(text)
        if match:
            text, unwrapped = match[1], True
        phase = 'outer_json'
        outer = json.loads(text, object_pairs_hook=base._unique, parse_constant=base._invalid_constant)
        phase = 'outer_schema'
        if type(outer) is not dict:
            raise ValueError('schema_invalid')
        candidate = outer.get('answer')
        if type(candidate) is str and candidate.strip() and len(candidate.encode()) <= 40000:
            answer = candidate.strip()
        if set(outer) != {'answer', 'validation_design'} or not answer:
            raise ValueError('schema_invalid')
        encoded_plan = outer['validation_design']
        phase = 'design_bound'
        if type(encoded_plan) is not str or not encoded_plan.strip() or len(encoded_plan.encode()) > 32768:
            raise ValueError('design_invalid')
        phase = 'design_json'
        design = json.loads(encoded_plan, object_pairs_hook=base._unique, parse_constant=base._invalid_constant)
        # This is the explicit wire representation, not a semantic repair.
        # All original plan values pass unaltered into the host validator.
        parsed = base.parse(json.dumps({'answer': answer, 'validation_design': design},
            ensure_ascii=False, allow_nan=False), sources, trusted_budget_minutes=trusted_budget_minutes)
        framing_valid = True
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        failure = phase
    if parsed is None:
        parsed = base.parse('{}', sources, trusted_budget_minutes=trusted_budget_minutes)
    parsed['answer'] = answer
    parsed['validation'] = {**parsed['validation'], 'version': VERSION,
        'wire_schema_checked': framing_valid, 'response_fence_unwrapped': unwrapped,
        'framing_failure': failure,
        'answer_preserved_on_design_failure': bool(answer and parsed['validation']['status'] == 'draft')}
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
