"""Bounded literal need witnesses for the LAST prompt-packing boundary.

This checks retained source geometry, not entailment, truth, causal support or
complete document conditions. No model, I/O, stored private query text or new
retrieval occurs. A whole existing quotation is an indivisible evidence bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .research_retrieval import lexical_tokens
from .research_sentence_windows import evidence_units
from .research_surface_coverage import surface_term_coverage

MAX_NEEDS = 64
MAX_WINDOWS = 32
MAX_WINDOW_CHARS = 20000
_GUARD = re.compile(r'\b(?:not|no|never|cannot|unless|except|only|if|under|without|however|but|provided)\b|'
                    r'不|无|未|仅|只有|除非|如果|但是|然而|前提|条件', re.I)


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _digest(value):
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))


def _identity(row):
    if not isinstance(row, dict):
        return None
    content = row.get('content')
    if (type(content) is not str or not content.strip() or len(content) > MAX_WINDOW_CHARS
            or any(type(row.get(key)) is not str or not 1 <= len(row[key]) <= 256
                   for key in ('id', 'document_id', 'label'))
            or row.get('is_generated') in (True, 'true', '1')
            or row.get('is_discovered') in (True, 'true', '1')
            or row.get('provenance') in ('llm', 'self_study', 'self_evolution', 'llm_expansion', 'hypothesis', 'inferred', 'memory')):
        return None
    start, end = row.get('excerpt_start', 0), row.get('excerpt_end', len(content))
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= 2000000
            or end - start != len(content)
            or row.get('offset_unit', 'characters_in_source_chunk') != 'characters_in_source_chunk'):
        return None
    try:
        for key in ('id', 'document_id', 'label'):
            row[key].encode('utf-8')
        digest = _hash(content)
    except UnicodeError:
        return None
    return (row['id'], row['document_id'], row['label'], start, end, digest)


@dataclass(frozen=True)
class Need:
    identifier: str
    terms: frozenset[str]


@dataclass(frozen=True)
class Bundle:
    identity: tuple
    need_ids: frozenset[str]
    protected_units: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NeedLedger:
    needs: tuple[Need, ...]
    bundles: tuple[Bundle, ...]
    explicit: bool
    omitted_query_terms: int
    invalid_input_windows: int


def build_need_ledger(question, sources, *, required_needs=None):
    """Optional host needs are explicit {id,text}; otherwise literal query terms.

    Explicit host needs are NOT inferred from uploaded material or model prose.
    Their texts stay request-local; receipts contain only IDs/counts/hashes.
    """
    if not isinstance(question, str) or not question.strip() or len(question.encode()) > 8192:
        raise ValueError('Need ledger question budget exceeded')
    if not isinstance(sources, (list, tuple)) or len(sources) > MAX_WINDOWS:
        raise ValueError('Need ledger source window budget exceeded')
    omitted = 0
    if required_needs is None:
        terms = list(dict.fromkeys(lexical_tokens(question)))
        omitted = max(0, len(terms) - MAX_NEEDS)
        needs = [Need('ln_' + _hash(term)[:20], frozenset((term,))) for term in terms[:MAX_NEEDS]]
    else:
        if not isinstance(required_needs, (list, tuple)) or len(required_needs) > MAX_NEEDS:
            raise ValueError('Invalid explicit need ledger')
        needs, identifiers = [], set()
        for row in required_needs:
            if (type(row) is not dict or set(row) != {'id', 'text'}
                    or type(row['id']) is not str or not 1 <= len(row['id']) <= 96
                    or row['id'] in identifiers or type(row['text']) is not str
                    or not row['text'].strip() or len(row['text'].encode()) > 1024):
                raise ValueError('Invalid explicit need ledger')
            terms = frozenset(lexical_tokens(row['text']))
            if not terms or len(terms) > MAX_NEEDS:
                raise ValueError('Invalid explicit need ledger')
            needs.append(Need(row['id'], terms)); identifiers.add(row['id'])
    bundles, seen, invalid = [], set(), 0
    for row in sources:
        identity = _identity(row)
        if identity is None:
            invalid += 1
            continue
        if identity in seen:
            continue
        seen.add(identity)
        tokens = frozenset(lexical_tokens(row['content']))
        # Share the existing exact/case/regular-plural diagnostic. This is not
        # stemming, synonym expansion or a source-text rewrite.
        covered = frozenset(surface_term_coverage(
            (term for need in needs for term in need.terms), tokens)['covered'])
        matched = frozenset(need.identifier for need in needs if need.terms <= covered)
        # evidence_units already joins adjacent explicit qualifiers. Recording
        # these spans does not certify that unseen/outside qualifiers are absent.
        guarded = tuple((identity[3] + left, identity[3] + right)
            for left, right in evidence_units(row['content']) if _GUARD.search(row['content'][left:right]))
        bundles.append(Bundle(identity, matched, guarded))
    return NeedLedger(tuple(needs), tuple(bundles), required_needs is not None, omitted, invalid)


def fallback_source_variants(ledger, sources):
    """One bounded coverage-greedy chain, then legacy rescue; never cut a quote.

    Scarce literal witnesses get more weight. Ties preserve upstream rank.
    This is a heuristic, not an optimality/quality guarantee or new retrieval.
    """
    if not isinstance(ledger, NeedLedger) or len(sources) > MAX_WINDOWS:
        raise ValueError('Invalid need ledger')
    by_identity = {bundle.identity: bundle.need_ids for bundle in ledger.bundles}
    covers = [by_identity.get(_identity(row), frozenset()) for row in sources]
    frequencies = {need.identifier: sum(need.identifier in found for found in covers) for need in ledger.needs}
    covered, order, pending = set(), [], list(range(len(sources)))
    while pending:
        def score(index):
            gain = covers[index] - covered
            return (sum(1. / frequencies[key] for key in sorted(gain)), len(gain), -index)
        chosen = max(pending, key=score)
        pending.remove(chosen); order.append(chosen); covered.update(covers[chosen])
    emitted = set()
    for sequence in (order, list(range(len(sources)))):
        for count in range(len(sources) - 1, 0, -1):
            indexes = tuple(sorted(sequence[:count]))
            if indexes not in emitted:
                emitted.add(indexes)
                yield [sources[index] for index in indexes]
    yield []


def final_need_ledger(ledger, sources):
    """Recheck EVERY final quoted window against exact pre-fit provenance."""
    if not isinstance(ledger, NeedLedger) or len(sources) > MAX_WINDOWS:
        raise ValueError('Invalid need ledger')
    known = {bundle.identity: bundle for bundle in ledger.bundles}
    identities = [_identity(row) for row in sources]
    retained = {identity for identity in identities if identity in known}
    current_origins = {(row.get('id'), row.get('document_id')) for row in sources if isinstance(row, dict)
                       and type(row.get('id')) is str and type(row.get('document_id')) is str}
    invalid = sum(identity not in known for identity in identities)
    rows = []
    for need in ledger.needs:
        witnesses = [bundle for bundle in ledger.bundles if need.identifier in bundle.need_ids]
        present = [bundle for bundle in witnesses if bundle.identity in retained]
        status = 'literal_witness_retained' if present else 'literal_witness_omitted' if witnesses else 'no_literal_witness'
        rows.append({'id': need.identifier, 'status': status,
            'witness_ids': [_digest(bundle.identity) for bundle in witnesses],
            'retained_witness_ids': [_digest(bundle.identity) for bundle in present]})
    windows = [{'bundle_id': _digest(bundle.identity), 'source_id': bundle.identity[0],
        'document_id': bundle.identity[1], 'label': bundle.identity[2],
        'span': list(bundle.identity[3:5]), 'sha256': bundle.identity[5],
        'protected_unit_spans': [list(span) for span in bundle.protected_units],
        'retained_exactly': bundle.identity in retained,
        'drop_reason': None if bundle.identity in retained else
            'changed_or_partial_window' if bundle.identity[:2] in current_origins else 'omitted_whole_window'}
        for bundle in ledger.bundles]
    counts = {status: sum(row['status'] == status for row in rows) for status in
        ('literal_witness_retained', 'literal_witness_omitted', 'no_literal_witness')}
    full = {'version': 1, 'scope': 'literal_window_retention_not_semantic_support',
        'needs': rows, 'bundles': windows, 'explicit_host_needs': ledger.explicit,
        'unrepresented_query_term_count': ledger.omitted_query_terms,
        'invalid_input_window_count': ledger.invalid_input_windows,
        'invalid_retained_window_count': invalid, 'semantic_verified': False,
        'whole_document_conditions_verified': False}
    receipt = {'version': 1, 'algorithm': 'literal_need_witness_budget_v1', 'scope': full['scope'],
        'need_count': len(rows), 'retained_need_count': counts['literal_witness_retained'],
        'omitted_need_count': counts['literal_witness_omitted'], 'unmatched_need_count': counts['no_literal_witness'],
        'known_window_count': len(windows), 'retained_window_count': len(retained),
        'invalid_input_window_count': ledger.invalid_input_windows, 'invalid_retained_window_count': invalid,
        'protected_unit_count': sum(len(known[identity].protected_units) for identity in retained),
        'unrepresented_query_term_count': ledger.omitted_query_terms, 'explicit_host_needs': ledger.explicit,
        'ledger_sha256': _digest(full), 'semantic_verified': False, 'whole_document_conditions_verified': False}
    gaps = []
    if receipt['omitted_need_count']:
        gaps.append({'code': 'final_need_witness_omitted', 'count': receipt['omitted_need_count'],
            'message': '部分字面需求原有的完整原文见证未进入最终输入，不能将未展示内容当作已覆盖。'})
    if receipt['unmatched_need_count'] or ledger.omitted_query_terms:
        gaps.append({'code': 'final_need_witness_unmatched',
            'count': receipt['unmatched_need_count'] + ledger.omitted_query_terms,
            'message': '部分字面需求在本次有限窗口中没有匹配见证；不等于全文不存在，也不证明语义覆盖。'})
    return {'ledger': full, 'receipt': receipt, 'gaps': gaps}
