"""Bounded reuse of visible prior analyses, never new evidence or a summary call.

Every declared direct dependency must contribute. Selection preserves complete
continuous analysis units (including adjacent qualifications), and discloses
partial reuse. Lexical similarity is a scheduling heuristic, not proof.
"""
from __future__ import annotations

import json

from .research_retrieval import lexical_tokens
from .research_sentence_windows import evidence_units

MAX_PRIOR_BYTES = 2000
MAX_SCAN_UNITS = 256
MAX_ANSWER_BYTES = 131072


class DependencyContextUnavailable(ValueError):
    pass


def _encode(packet):
    return json.dumps(packet, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _fits(packet):
    return len(_encode(packet).encode('utf-8')) <= MAX_PRIOR_BYTES


def build_handoff(dependencies, question):
    """Return transient context + refs-only counters; do not persist the packet.

    The host supplies current hydrated answers. No missing dependency can be
    silently skipped to save tokens. A scope/permission check remains the
    caller's responsibility before supplying any answer here.
    """
    if not isinstance(dependencies, list) or len(dependencies) > 6:
        raise DependencyContextUnavailable('Dependency context unavailable')
    counts = {'required': len(dependencies), 'used': len(dependencies),
              'full': len(dependencies), 'excerpted': 0, 'omitted': 0}
    if not dependencies:
        return '', counts
    blocks, texts = [], []
    for row in dependencies:
        if not isinstance(row, dict) or set(row) != {'step_id', 'answer_id', 'content'}:
            raise DependencyContextUnavailable('Dependency context unavailable')
        text = row['content']
        if (not isinstance(row['step_id'], str) or not row['step_id']
                or len(row['step_id']) > 128 or not isinstance(row['answer_id'], str)
                or len(row['answer_id']) > 128 or not row['answer_id']
                or not isinstance(text, str) or not text.strip()
                or len(text.encode('utf-8')) > MAX_ANSWER_BYTES):
            raise DependencyContextUnavailable('Dependency context unavailable')
        texts.append(text)
        blocks.append({'step_id': row['step_id'], 'answer_id': row['answer_id'],
                       'coverage': 'full', 'analysis_not_original_evidence': text})
    if _fits(blocks):
        return _encode(blocks), counts
    terms = set(lexical_tokens(question))
    choices = []
    scan_share = MAX_SCAN_UNITS // len(blocks)
    for index, text in enumerate(texts):
        # Parse the whole bounded answer before choosing units. Never parse a
        # sliced prefix which could turn half a condition into a complete unit.
        options = []
        for start, end in evidence_units(text)[:scan_share]:
            body = text[start:end]
            if not body.strip():
                continue
            score = len(terms & set(lexical_tokens(body)))
            unit = {'start': start, 'end': end, 'text': body}
            options.append((score, start, len(_encode(unit).encode('utf-8')), unit))
        if not options:
            raise DependencyContextUnavailable('Dependency context unavailable')
        choices.append(sorted(options, key=lambda option: (-option[0], option[1])))
        blocks[index] = {'step_id': blocks[index]['step_id'], 'answer_id': blocks[index]['answer_id'],
                         'coverage': 'selected_analysis_units',
                         'analysis_not_original_evidence': [min(options, key=lambda option: option[2])[3]]}
    # First guarantee one whole unit from every dependency. Only then compete
    # for optional space; the first predecessor must not starve all the others.
    if not _fits(blocks):
        raise DependencyContextUnavailable('Dependency context unavailable')
    for index, options in enumerate(choices):
        for _, _, _, unit in options:
            candidate = [dict(block) for block in blocks]
            candidate[index]['analysis_not_original_evidence'] = [unit]
            if _fits(candidate):
                blocks = candidate
                break
    for index, options in enumerate(choices):
        selected = blocks[index]['analysis_not_original_evidence']
        for _, _, _, unit in options:
            if any(item['start'] == unit['start'] for item in selected):
                continue
            candidate = [dict(block) for block in blocks]
            candidate[index]['analysis_not_original_evidence'] = sorted([*selected, unit], key=lambda item: item['start'])
            if _fits(candidate):
                blocks, selected = candidate, candidate[index]['analysis_not_original_evidence']
    # Even if all scanned units happen to fit, coverage is still explicitly
    # partial: unscanned units are never certified as read or semantically used.
    counts.update(full=0, excerpted=len(blocks))
    return _encode(blocks), counts
