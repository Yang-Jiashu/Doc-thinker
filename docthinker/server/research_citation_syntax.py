"""Bounded citation syntax shared by validation, receipts and literal audits.

This recognizes source/method labels, not support, authority or source access.
Comma groups are explicit lists; ranges and inferred labels are never expanded.
Offsets refer to the unchanged input string. Limit exhaustion is always visible.
"""
from __future__ import annotations

import re

MAX_CITATION_SCAN_CHARS = 120_000
MAX_CITATION_GROUP_CHARS = 2_048
MAX_CITATION_GROUPS = 256
MAX_CITATION_LABELS = 64
MAX_CITATION_LABEL_CHARS = 17
MAX_BRACKET_STARTS = 4_096
_LIST = re.compile(r'[ \t]*[SK][0-9]+(?:[ \t]*[,，][ \t]*[SK][0-9]+)*[ \t]*\Z', re.I)
CITATION_GROUP_PATTERN = re.compile(
    r'\[(?=[^\[\]\r\n]{1,2048}\])[ \t]*[SK][0-9]+'
    r'(?:[ \t]*[,，][ \t]*[SK][0-9]+)*[ \t]*\]', re.I)
_LABEL = re.compile(r'[SK][0-9]+', re.I)
_START = re.compile(r'[ \t]*[SK][0-9]', re.I)


def parse_citation_groups(text):
    """Return exact group spans, first-seen labels and an explicit limit flag.

    Unknown numeric labels remain visible to callers. Very long labels have a
    bounded diagnostic prefix and set ``truncated``; they must never pass a
    validation check just because the visible prefix resembles a known label.
    """
    if not isinstance(text, str):
        return {'groups': [], 'labels': [], 'truncated': False}
    scan = text[:MAX_CITATION_SCAN_CHARS]
    result = {'groups': [], 'labels': [], 'truncated': len(scan) < len(text)}
    labels_seen, cursor, starts = set(), 0, 0

    def retain(label):
        label = label.upper()
        if len(label) > MAX_CITATION_LABEL_CHARS:
            result['truncated'] = True
            label = label[:MAX_CITATION_LABEL_CHARS]
        if label not in labels_seen:
            if len(result['labels']) >= MAX_CITATION_LABELS:
                result['truncated'] = True
            else:
                labels_seen.add(label)
                result['labels'].append(label)
        return label

    while cursor < len(scan):
        start = scan.find('[', cursor)
        if start < 0:
            break
        starts += 1
        if starts > MAX_BRACKET_STARTS or len(result['groups']) >= MAX_CITATION_GROUPS:
            result['truncated'] = True
            break
        end = scan.find(']', start + 1, min(len(scan), start + MAX_CITATION_GROUP_CHARS + 2))
        nested = scan.find('[', start + 1, end if end >= 0 else min(len(scan), start + MAX_CITATION_GROUP_CHARS + 2))
        if nested >= 0:
            cursor = nested
            continue
        if end < 0:
            prefix = scan[start + 1:start + 1 + MAX_CITATION_LABEL_CHARS + 2]
            if _START.match(prefix):
                result['truncated'] = True
                match = _LABEL.search(prefix)
                if match:
                    retain(match.group())
            cursor = start + 1
            continue
        body = scan[start + 1:end]
        if _LIST.fullmatch(body):
            group_labels = list(dict.fromkeys(retain(match.group()) for match in _LABEL.finditer(body)))
            if len(group_labels) > MAX_CITATION_LABELS:
                result['truncated'] = True
                group_labels = group_labels[:MAX_CITATION_LABELS]
            result['groups'].append({'start': start, 'end': end + 1, 'labels': group_labels})
        cursor = end + 1
    return result


def extract_citation_labels(text, prefixes='SK'):
    """Convenience only; validators must also inspect the parser's limit flag."""
    allowed = set(prefixes.upper()) & {'S', 'K'} if isinstance(prefixes, str) else set()
    return [label for label in parse_citation_groups(text)['labels'] if label[0] in allowed]


def strip_citation_groups(text, prefixes='S'):
    """Remove selected citation tokens for a derived audit view, not the answer.

    In mixed groups unselected prefixes remain bracketed. Limit exhaustion
    never truncates or discards the rest of the input string.
    """
    if not isinstance(text, str):
        return ''
    allowed = set(prefixes.upper()) & {'S', 'K'} if isinstance(prefixes, str) else set()
    parts, cursor = [], 0
    for group in parse_citation_groups(text)['groups']:
        remaining = [label for label in group['labels'] if label[0] not in allowed]
        if len(remaining) == len(group['labels']):
            continue
        parts.append(text[cursor:group['start']])
        if remaining:
            parts.append('[' + ', '.join(remaining) + ']')
        cursor = group['end']
    parts.append(text[cursor:])
    return ''.join(parts)
