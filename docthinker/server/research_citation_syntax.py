"""Bounded citation syntax shared by validation, receipts and literal audits.

This recognizes source/method labels and citation-shaped control IDs, not
support, authority or source access. Control IDs are diagnostics, never sources.
Comma groups are explicit lists; ranges and inferred labels are never expanded.
Offsets refer to the unchanged input string. Limit exhaustion is always visible.
"""
from __future__ import annotations

import re
from bisect import bisect_right

MAX_CITATION_SCAN_CHARS = 120_000
MAX_CITATION_GROUP_CHARS = 2_048
MAX_CITATION_GROUPS = 256
MAX_CITATION_LABELS = 64
MAX_CITATION_LABEL_CHARS = 17
MAX_BRACKET_STARTS = 4_096
_PREFIXES = 'SKCELPQAG'
_LIST = re.compile(r'[ \t]*[' + _PREFIXES + r'][0-9]+(?:[ \t]*[,，][ \t]*[' + _PREFIXES + r'][0-9]+)*[ \t]*\Z', re.I)
CITATION_GROUP_PATTERN = re.compile(
    r'\[(?=[^\[\]\r\n]{1,2048}\])[ \t]*[' + _PREFIXES + r'][0-9]+'
    r'(?:[ \t]*[,，][ \t]*[' + _PREFIXES + r'][0-9]+)*[ \t]*\]', re.I)
_LABEL = re.compile(r'[' + _PREFIXES + r'][0-9]+', re.I)
_START = re.compile(r'[ \t]*[' + _PREFIXES + r'][0-9]', re.I)
_ROUND_LIST = r'[ \t]*[SK][0-9]+(?:[ \t]*[,，、][ \t]*[SK][0-9]+)*[ \t]*'
# Paired punctuation only. A mismatched or explanatory parenthesis is not
# mechanically rewritten into a citation.
_ROUND_GROUP = re.compile(r'\((?=[^()\r\n]{1,2048}\))' + _ROUND_LIST + r'\)|'
                          r'（(?=[^（）\r\n]{1,2048}）)' + _ROUND_LIST + r'）', re.I)
CITATION_GROUP_PATTERN = re.compile(CITATION_GROUP_PATTERN.pattern + '|' + _ROUND_GROUP.pattern, re.I)
_ROUND_LABEL = re.compile(r'[SK][0-9]+', re.I)
_URL = re.compile(r'(?i)(?:\b(?:https?://|ftp://|mailto:|www\.)|(?<!\w)/)[^\s<>]+')
_FENCE = re.compile(r' {0,3}(`{3,}|~{3,})([^\r\n]*)')
_BACKTICKS = re.compile(r'`+')
_REFERENCE = re.compile(
    r'''(?m)^ {0,3}\[([^\[\]\r\n]+)\]:[ \t]*(?:<[^<>\r\n]+>|(?:https?://|/|#)[^ \t\r\n]+)'''
    r'''(?:[ \t]+(?:"[^"\r\n]*"|'[^'\r\n]*'|\([^()\r\n]*\)))?[ \t]*$''')
_LINK_BODY = re.compile(
    r'''[ \t]*(?:<[^<>\r\n]*>|[^ \t\r\n<>]*)(?:[ \t]+(?:"[^"\r\n]*"|'[^'\r\n]*'|\([^()\r\n]*\)))?[ \t]*\Z''')


def _escaped(text, position):
    cursor = position
    while cursor and text[cursor - 1] == '\\':
        cursor -= 1
    return (position - cursor) % 2 == 1


def _markdown_code_ranges(text, *, include_unclosed=False):
    """Ignore only closed Markdown code, with linear work and exact offsets.

    An unfinished fence is scanned as ordinary text. It cannot hide a later
    fabricated citation, nor become an accidentally paired inline code span.
    """
    ranges, opened, offset = [], None, 0
    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if opened is None:
            if match and not (match[1][0] == '`' and '`' in match[2]):
                opened = (offset, match[1][0], len(match[1]))
        elif (match and match[1][0] == opened[1] and len(match[1]) >= opened[2]
              and not match[2].strip()):
            ranges.append((opened[0], offset + len(line)))
            opened = None
        offset += len(line)
    inline_limit = opened[0] if opened else len(text)
    cursor, inline = 0, []
    for left, right in [*ranges, (inline_limit, inline_limit)]:
        ticks = [match for match in _BACKTICKS.finditer(text, cursor, min(left, inline_limit))
                 if not _escaped(text, match.start())]
        following, next_equal = {}, {}
        for index in range(len(ticks) - 1, -1, -1):
            size = ticks[index].end() - ticks[index].start()
            if size in following:
                next_equal[index] = following[size]
            following[size] = index
        index = 0
        while index < len(ticks):
            end = next_equal.get(index)
            if end is None:
                if include_unclosed:
                    inline.append((ticks[index].start(), min(left, inline_limit)))
                    break
                index += 1
            else:
                inline.append((ticks[index].start(), ticks[end].end()))
                index = end + 1
        cursor = right
    if include_unclosed and opened:
        ranges.append((opened[0], len(text)))
    return sorted([*ranges, *inline])


def _inline_link(text, after):
    """A real, closed inline link is presentation, not a source citation.

    Unclosed/over-budget destinations stay ordinary text so a dangling '(' is
    not an escape hatch for an unknown label. No URLs are opened or trusted.
    """
    if after >= len(text) or text[after] != '(':
        return False
    depth, cursor = 1, after + 1
    limit = min(len(text), after + MAX_CITATION_GROUP_CHARS + 1)
    while cursor < limit:
        if text[cursor] in '\r\n':
            return False
        if text[cursor] == '\\':
            cursor += 2
            continue
        if text[cursor] == '(':
            depth += 1
        elif text[cursor] == ')':
            depth -= 1
            if depth == 0:
                return cursor + 1 if _LINK_BODY.fullmatch(text[after + 1:cursor]) else False
        cursor += 1
    return False


def _round_citation_groups(text, *, preserve_unclosed=False):
    """Read complete S/K parenthetical groups outside literal presentation.

    This reports unknown S/K labels too, so refusing to normalize an unknown
    label never makes a later validator call it clean. No control ID is
    interpreted as a source or converted to a source namespace.
    """
    code = _markdown_code_ranges(text, include_unclosed=preserve_unclosed)
    code_starts = [left for left, _ in code]
    definitions = []
    for match in _REFERENCE.finditer(text):
        index = bisect_right(code_starts, match.start()) - 1
        if index < 0 or match.start() >= code[index][1]:
            definitions.append(match)
    ignored = [*code, *(match.span() for match in _URL.finditer(text)), *(match.span() for match in definitions)]
    # A closed Markdown link/image/ref is presentation, including its label
    # and URL. A malformed link is not made valid or completed by this pass.
    references = {match[1].casefold() for match in definitions}
    bracket_count, scan_end = 0, len(text)
    for match in re.finditer(r'\[', text):
        bracket_count += 1
        if bracket_count > MAX_BRACKET_STARTS:
            scan_end = match.start(); break
        end = text.find(']', match.end(), min(len(text), match.end() + MAX_CITATION_GROUP_CHARS + 1))
        if end < 0:
            continue
        after = _inline_link(text, end + 1)
        if after:
            ignored.append((match.start(), after))
        elif text[match.end():end].casefold() in references:
            ignored.append((match.start(), end + 1))
        elif end + 1 < len(text) and text[end + 1] == '[':
            reference_end = text.find(']', end + 2, min(len(text), end + MAX_CITATION_GROUP_CHARS + 3))
            if reference_end >= 0 and text[end + 2:reference_end].casefold() in references:
                ignored.append((match.start(), reference_end + 1))
    merged = []
    for left, right in sorted(ignored):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    starts = [left for left, _ in merged]
    groups, truncated = [], scan_end < len(text)
    for match in _ROUND_GROUP.finditer(text, 0, scan_end):
        index = bisect_right(starts, match.start()) - 1
        if ((index >= 0 and match.start() < merged[index][1]) or _escaped(text, match.start())):
            continue
        if len(groups) >= MAX_CITATION_GROUPS:
            truncated = True; break
        groups.append({'start': match.start(), 'end': match.end(),
                       'labels': list(dict.fromkeys(item.group().upper() for item in _ROUND_LABEL.finditer(match.group())))})
    for match in re.finditer(r'[(（][ \t]*[SK][0-9]', text[:scan_end], re.I):
        index = bisect_right(starts, match.start()) - 1
        if (index >= 0 and match.start() < merged[index][1]) or _escaped(text, match.start()):
            continue
        closer = ')' if text[match.start()] == '(' else '）'
        if text.find(closer, match.end(), min(scan_end, match.start() + MAX_CITATION_GROUP_CHARS + 2)) < 0:
            truncated = True
    return groups, truncated


def normalize_approved_citations(text, source_labels, *, knowledge_labels=()):
    """Canonicalize presentation only, before final answer witnesses/hashes.

    Every member must already belong to the explicit current S/K selection.
    Unknowns, controls, ranges, explanation text, escaped literals, code, links
    and URLs survive unchanged. No source content or assessment is rewritten.
    """
    if not isinstance(text, str):
        return text
    source_labels = source_labels if isinstance(source_labels, (list, tuple, set, frozenset)) else ()
    knowledge_labels = knowledge_labels if isinstance(knowledge_labels, (list, tuple, set, frozenset)) else ()
    allowed_sources = {label for label in source_labels if type(label) is str and re.fullmatch(r'S[1-9][0-9]*', label)}
    allowed_methods = {label for label in knowledge_labels if type(label) is str and re.fullmatch(r'K[1-9][0-9]*', label)}
    allowed = allowed_sources | allowed_methods
    groups, _ = _round_citation_groups(text[:MAX_CITATION_SCAN_CHARS], preserve_unclosed=True)
    parts, cursor = [], 0
    for group in groups:
        if not set(group['labels']) <= allowed:
            continue
        parts.append(text[cursor:group['start']])
        parts.append('[' + ', '.join(group['labels']) + ']')
        cursor = group['end']
    parts.append(text[cursor:])
    return ''.join(parts)


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
    ignored = _markdown_code_ranges(scan)
    ignored_index = 0
    references, definitions = set(), []
    code_starts = [left for left, _ in ignored]
    for match in _REFERENCE.finditer(scan):
        index = bisect_right(code_starts, match.start()) - 1
        if index < 0 or match.start() >= ignored[index][1]:
            references.add(match[1].casefold())
            definitions.append(match.span())
    ignored.extend(definitions)
    ignored.sort()

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
        while ignored_index < len(ignored) and ignored[ignored_index][1] <= start:
            ignored_index += 1
        if ignored_index < len(ignored) and ignored[ignored_index][0] <= start:
            cursor = ignored[ignored_index][1]
            continue
        if _escaped(scan, start):
            cursor = start + 1
            continue
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
        link_end = _inline_link(scan, end + 1)
        if link_end or body.strip().casefold() in references:
            cursor = link_end or end + 1
            continue
        if end + 1 < len(scan) and scan[end + 1] == '[':
            reference_end = scan.find(']', end + 2, min(len(scan), end + MAX_CITATION_GROUP_CHARS + 3))
            if reference_end >= 0 and scan[end + 2:reference_end].casefold() in references:
                cursor = reference_end + 1
                continue
        if _LIST.fullmatch(body):
            group_labels = list(dict.fromkeys(retain(match.group()) for match in _LABEL.finditer(body)))
            if len(group_labels) > MAX_CITATION_LABELS:
                result['truncated'] = True
                group_labels = group_labels[:MAX_CITATION_LABELS]
            result['groups'].append({'start': start, 'end': end + 1, 'labels': group_labels})
        cursor = end + 1
    rounds, round_truncated = _round_citation_groups(scan)
    if rounds:
        prior_labels = result['labels'][:]
        groups = sorted([*result['groups'], *rounds], key=lambda group: group['start'])
        result['groups'] = groups[:MAX_CITATION_GROUPS]
        result['labels'] = []
        labels_seen.clear()
        for group in groups:
            group['labels'] = [retain(label) for label in group['labels']]
        # Preserve truncated/unclosed bracket diagnostic prefixes too.
        for label in prior_labels:
            retain(label)
        result['truncated'] |= len(groups) > MAX_CITATION_GROUPS
    result['truncated'] |= round_truncated
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
