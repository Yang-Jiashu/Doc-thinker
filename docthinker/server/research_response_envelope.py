"""Bounded presentation decoding, not evidence repair or tool execution.

Accept one strict root envelope, including a unique terminal envelope after
plain prose. Never search arbitrary nested examples or concatenate objects.
"""
from __future__ import annotations

import json
import re

_FIELDS = frozenset({'answer', 'event_assessment', 'task_assessment', 'trial_spec'})
_ANNOTATIONS = _FIELDS - {'answer'}
_METADATA = re.compile(r'"(?:answer|event_assessment|task_assessment|trial_spec)"\s*:')
_ROOT = re.compile(r'(?m)^[ \t]*\{')
_FENCE = re.compile(r'(?im)^[ \t]*```(?:json)?[ \t]*\r?\n[ \t]*(?=\{)')
_ERROR = 'Invalid research response envelope'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_ERROR)
        result[key] = value
    return result


def _finite(_):
    raise ValueError(_ERROR)


def _qualified(value):
    if (type(value) is not dict or set(value) - _FIELDS or type(value.get('answer')) is not str
            or not value['answer'].strip() or '\x00' in value['answer']):
        return False
    value['answer'].encode('utf-8')
    return True


def _plain_prefix(value):
    # A candidate inside an open JSON object/array or code block is not a root
    # response. Earlier protocol objects/examples also make selection ambiguous.
    if _METADATA.search(value) or len(re.findall(r'(?m)^[ \t]*```', value)) % 2:
        return False
    stack, quoted, escaped = [], False, False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '{[':
            stack.append(char)
        elif char in '}]':
            if not stack or stack.pop() != ('{' if char == '}' else '['):
                return False
    return not stack and not quoted


def decode_research_envelope(raw: str, *, max_bytes: int = 131072,
                             allowed_metadata_fields=()) -> dict | None:
    """Return a unique strict root, None for prose, fixed error for unsafe metadata.

    Only a line-root bare object or a JSON/unlabelled fenced terminal object
    can follow prose. A host that explicitly requested annotations may name
    those fields: one metadata-only terminal object then uses the intact
    nonempty prose prefix as answer. This is presentation, never semantic
    repair. Without that contract an example is not promoted into a response.
    """
    try:
        if type(raw) is not str or type(max_bytes) is not int or not 0 < max_bytes <= 1_048_576:
            raise ValueError(_ERROR)
        if (not isinstance(allowed_metadata_fields, (list, tuple, set, frozenset))
                or any(type(key) is not str or key not in _ANNOTATIONS for key in allowed_metadata_fields)):
            raise ValueError(_ERROR)
        allowed_metadata_fields = set(allowed_metadata_fields)
        if len(raw.encode('utf-8')) > max_bytes:
            raise ValueError(_ERROR)
        text = raw.strip()
        decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_finite)
        starts = [(match.end() - 1, match.start(), False) for match in _ROOT.finditer(text)]
        starts += [(match.end(), match.start(), True) for match in _FENCE.finditer(text)]
        if len(starts) > 64:
            if _METADATA.search(text):
                raise ValueError(_ERROR)
            return None
        found = []
        for start, prefix_end, fenced in starts:
            try:
                value, end = decoder.raw_decode(text, start)
            except (ValueError, TypeError, RecursionError):
                continue
            prefix = text[:prefix_end]
            metadata_only = (type(value) is dict and bool(value) and set(value) <= allowed_metadata_fields
                             and all(type(item) is dict for item in value.values()))
            if not _qualified(value) and metadata_only and prefix.strip() and _plain_prefix(prefix):
                value = {'answer': prefix.rstrip(), **value}
            if not _qualified(value) or not _plain_prefix(prefix):
                continue
            suffix = text[end:]
            complete = (bool(re.fullmatch(r'\s*```[ \t]*\s*', suffix)) if fenced else not suffix.strip())
            if complete:
                found.append((start, value))
        unique = {start: value for start, value in found}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if _METADATA.search(text):
            raise ValueError(_ERROR)
        return None
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ValueError(_ERROR) from None
