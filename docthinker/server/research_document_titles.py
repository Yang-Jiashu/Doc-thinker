"""Bounded, source-located title hints; never document authority or evidence.

Only an initial scholarly title of the form TechnicalName: subtitle is used.
Ordinary body mentions, reference entries and instructions cannot become
names. The result is a soft material locator, not a permission or truth signal.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import islice

MAX_ROWS = 4096
MAX_DOCUMENTS = 256
MAX_HEADER_CHARS = 1200
_PREAMBLE = re.compile(r'^\s*(?:arXiv:\d{4}\.\d{4,5}v?\d*\b[^\n]*|\d{4}\.\d{4,5}v?\d*[^\n]*)$', re.I)
_TITLE = re.compile(r'(?P<name>[A-Za-z][A-Za-z0-9 -]{1,63})\s*[:：]\s*(?P<subtitle>[^\n]{8,220})$')
_GENERIC = frozenset('abstract introduction references bibliography appendix summary report paper research '
                     'method methods results memory cache map plan plans ignore note notes title warning system'.split())


def initial_title_hints(chunks):
    """Return at most one exact title locator per live document, without edits.

    A compact technical name may have PDF small-caps spacing only when its
    entire prefix is uppercase; normal words are never joined into aliases.
    Multiple inconsistent initial chunks yield no hint for that document.
    """
    candidates = {}
    for row in islice(chunks, MAX_ROWS):
        if not isinstance(row, Mapping) or row.get('status') in ('revoked', 'deleted', 'invalidated'):
            continue
        if (row.get('is_generated') in (True, 'true', '1') or row.get('is_discovered') in (True, 'true', '1')
                or row.get('provenance') in ('llm', 'self_study', 'self_evolution', 'hypothesis', 'inferred', 'memory')):
            continue
        doc = row.get('document_id')
        if not isinstance(doc, str) or not doc or len(doc) > 256:
            continue
        if doc not in candidates and len(candidates) >= MAX_DOCUMENTS:
            continue
        if row.get('ordinal') != 0 or row.get('page') not in (None, 0, 1) or row.get('excerpt_start', 0) != 0:
            continue
        text = row.get('content')
        if not isinstance(text, str):
            continue
        header = text[:MAX_HEADER_CHARS]
        abstract = re.search(r'(?im)^\s*abstract\s*$', header)
        if not abstract:
            continue
        lines = list(re.finditer(r'[^\n]+', header[:abstract.start()]))[:24]
        lines = [line for line in lines if line.group().strip()]
        if lines and _PREAMBLE.fullmatch(lines[0].group()):
            lines = lines[1:]
        if not lines:
            continue
        title = _TITLE.fullmatch(lines[0].group().strip())
        if not title:
            continue
        raw_name = title['name'].strip()
        name = re.sub(r'\s+', '', raw_name) if raw_name.isupper() else raw_name
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9-]{2,63}', name)
                or name.casefold() in _GENERIC
                or not (name.isupper() or re.search(r'[a-z][A-Z]', name) or '-' in name)):
            continue
        start, end = lines[0].span()
        locator = {'name': name, 'document_id': doc, 'source_id': row.get('id'),
                   'title_start': start, 'title_end': end, 'raw_title': text[start:end],
                   'abstract_body_start': abstract.end() + len(header[abstract.end():]) - len(header[abstract.end():].lstrip())}
        bucket = candidates.setdefault(doc, [])
        if not any(value['raw_title'] == locator['raw_title'] for value in bucket):
            bucket.append(locator)
    return [values[0] for _, values in sorted(candidates.items()) if len(values) == 1]


def subplot_axis_labels(text, *, immediately_before_caption=False):
    """Dense multi-panel ticks/legends only; tables and normal prose stay.

    This supplements the existing pure-number detector, using the explicit
    following caption and at least two (a)/(b) panel labels. No data values are
    interpreted; figure/coordinate requests do not invoke this filter.
    """
    if not immediately_before_caption or not isinstance(text, str) or len(text) > 8000:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not 12 <= len(lines) <= 256 or re.search(r'\btable\b|表\s*\d', text, re.I):
        return False
    panels = re.findall(r'(?<!\w)\([a-z]\)\s+[A-Za-z][A-Za-z0-9 -]{0,40}', text)
    if len(panels) < 2 or len(set(panels)) < 2:
        return False
    numeric = 0
    for line in lines:
        numbers = re.findall(r'[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)', line)
        residue = re.sub(r'[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)', ' ', line)
        residue = re.sub(r'\([a-z]\)', ' ', residue).strip()
        if re.search(r'[^A-Za-z \t_-]', residue) or len(residue.split()) > 5:
            return False
        if len(line) > 80:
            return False
        numeric += len(numbers)
    return numeric >= 16
