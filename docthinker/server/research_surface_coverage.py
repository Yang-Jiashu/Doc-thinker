"""Conservative token-surface diagnostics, not retrieval or entailment.

Input/output terms retain the caller's original query-token spelling. Only
case and a narrow set of regular plural rules are considered; no suffix stemming,
synonyms, translation, substring search, or rewrite of quoted source text occurs.
Even a matched term says nothing about the truth, polarity, or scope of a claim.
"""
from __future__ import annotations

from collections.abc import Iterable


def _regular_plural(word: str) -> str | None:
    # Short words, identifiers, compounds and non-Latin terms are exact-only.
    # This intentionally leaves irregular/ambiguous forms unresolved rather
    # than guessing (e.g. new/news, axis/axes, or a Chinese near-synonym).
    if len(word) < 4 or not word.isascii() or not word.isalpha():
        return None
    if word.endswith('y') and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    return word + 's'


def surface_term_coverage(query_terms: Iterable[str], source_terms: Iterable[str]) -> dict[str, list[str]]:
    """Return original query tokens with/without a bounded surface-form match.

    Both singular-query/plural-source and plural-query/singular-source work.
    Chinese tokens (the caller may use bigrams) retain exact-match semantics.
    The result covers only the supplied excerpt tokens, never an entire corpus.
    """
    original = sorted({term for term in query_terms if isinstance(term, str) and term})
    present = {term.lower() for term in source_terms if isinstance(term, str) and term}
    regular_plurals = {plural for term in present if (plural := _regular_plural(term)) is not None}
    covered, uncovered = [], []
    for term in original:
        word = term.lower()
        plural = _regular_plural(word)
        matched = word in present or word in regular_plurals or (plural is not None and plural in present)
        (covered if matched else uncovered).append(term)
    return {'covered': covered, 'uncovered': uncovered}
