"""Host-owned trial delivery activation and receipts, never execution permission.

Only the current user's explicit deliverable request can activate this contract.
Cards, source text, history and an Insight switch cannot grant or remove it.
"""
from __future__ import annotations

import hashlib
import json
import re

from .research_query_facets import _QUOTED

TRIAL_MARKER = '\nHOST_TRIAL_RESPONSE_SCHEMA_JSON:\n'
MAX_TRIAL_SCHEMA_BYTES = 8192
_NEGATED = re.compile(r'^\s*(?:(?:please\s+)?(?:do not|don.t|skip|avoid)\b|(?:请)?(?:不要|无需|不必|不需要|别))', re.I)
_ACTION_NEGATED = re.compile(r'\b(?:not|never|don.t|avoid|skip)\s+(?:\w+\s+){0,2}$|'
                             r'(?:不要|无需|不必|不需要|不能|不|别)(?:再|去|继续)?\s*$', re.I)
_META_REQUEST = re.compile(r'\b(?:why|whether|should\s+(?:we|I)|do\s+(?:we|I)\s+need\s+to)\b|为什么|为何|是否应该|要不要', re.I)
_OTHER_ARTIFACT = re.compile(r'\b(?:poem|story|article|essay|explanation|summary)\b|诗|故事|文章|解释|总结', re.I)
_LITERAL = re.compile(r'\b(?:translate|transcribe|quote\s+(?:exactly|verbatim)|count\s+(?:words|characters))\b|翻译|逐字|原样|转录', re.I)
_ATTRIBUTED = re.compile(r'(?:论文|文中|作者|研究者)[^,，。;；]{0,24}(?:如何|怎样|设计|制定|验证|检验|要求|提到)|'
                         r'\b(?:paper|authors?|study|document|source)\b[^,.;]{0,35}'
                         r'\b(?:design(?:s|ed)?|tested|validated|conducted|how|says?|asks?|requests?)\b', re.I)
_DESIGN = re.compile(r'\b(?:design|draft|create|prepare|propose|write|give)\b[^.;!?\n]{0,65}'
                     r'\b(?:pilot|trial|(?:validation|acceptance|test)\s+(?:plan|protocol|criteria))\b|'
                     r'(?:设计|制定|安排|给出|起草|提出|写)[^。；;!?！？\n]{0,35}'
                     r'(?:试点|试用方案|小实验|验证方案|验证计划|验收方案|验收标准|测试方案|试验方案)', re.I)
_HOW_TEST = re.compile(r'\bhow\s+(?:(?:can|should|would|do)\s+)?(?:we|I)\s+(?:test|validate|evaluate)\b|'
                       r'(?:如何|怎样)[^。；;!?！？\n]{0,20}(?:检验|验证|验收)[^。；;!?！？\n]{0,30}'
                       r'(?:路线|方案|产品|计划|是否值得|能否)', re.I)


def requests_trial_delivery(question, *, literal=False):
    if literal or not isinstance(question, str) or len(question) > 8192:
        return False
    text = _QUOTED.sub(' ', re.sub(r'(?m)^\s*>[^\n]*', ' ', question))
    if _LITERAL.search(text):
        return False
    for clause in re.split(r'[。；;!?！？\n]|(?<=\.)\s+', text):
        if _NEGATED.match(clause):
            continue
        if _ATTRIBUTED.search(clause):
            continue
        for expression in (_DESIGN, _HOW_TEST):
            for match in expression.finditer(clause):
                prefix = re.split(r'[,，]', clause[:match.start()])[-1]
                if _ACTION_NEGATED.search(prefix) or _META_REQUEST.search(prefix):
                    continue
                # The action's requested artifact must be the trial, not a
                # story/article that merely mentions one later in its topic.
                if expression is _DESIGN and (_OTHER_ARTIFACT.search(match.group()) or
                    re.search(r'\b(?:about|on)\b', prefix, re.I)):
                    continue
                return True
    return False


def trial_contract_packet():
    from .research_trial_contract import trial_response_schema, TRIAL_RESPONSE_INSTRUCTION
    schema = trial_response_schema()
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(payload.encode()) > MAX_TRIAL_SCHEMA_BYTES:
        raise ValueError('Trial contract exceeds its budget')
    return {'prompt': TRIAL_MARKER + payload, 'policy': '\n' + TRIAL_RESPONSE_INSTRUCTION,
            'schema_sha256': hashlib.sha256(payload.encode()).hexdigest()}


def trial_delivery_receipt(parsed, source_ids, schema_hash):
    return {'version': 1, 'status': 'valid' if parsed.get('status') == 'valid' else 'draft',
            'violations': list(parsed.get('violations', []))[:32],
            'schema_sha256': schema_hash, 'source_ids': list(source_ids),
            'scope': 'internal_structure_only',
            'structural_validation_passed': parsed.get('status') == 'valid',
            'semantic_support_verified': False, 'executed': False, 'execution_authorized': False,
            'automatic_retry': False, 'extra_model_calls': 0}


def unavailable_trial_receipt():
    return {'version': 1, 'status': 'source_unavailable', 'violations': ['source_unavailable'],
            'scope': 'internal_structure_only',
            'structural_validation_passed': False, 'semantic_support_verified': False,
            'executed': False, 'execution_authorized': False, 'automatic_retry': False,
            'extra_model_calls': 0}
