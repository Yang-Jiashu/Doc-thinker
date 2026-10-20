"""Deterministic, bounded research evidence selection; no provider or I/O calls.

This combines lexical ranking with optional prevalidated semantic candidates;
it is not an entailment checker or a causal graph builder.
Only uploaded source text receives an ``S`` citation label. Confirmed user memory
is supplied separately as a preference/context hint and can never create evidence.
Ranking scans the input once, then compares at most 40 candidates with at most
8 selected excerpts. It intentionally does not pad a result with unrelated text.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .research_sentence_windows import evidence_units, section_heading_spans, opening_reference_unverified, is_figure_caption_start
from .research_surface_coverage import surface_term_coverage
from .research_citation_syntax import parse_citation_groups
from .research_document_titles import initial_title_hints
from .research_fallback_sources import _METADATA_QUERY

MAX_CHUNKS = 8
MAX_SOURCE_BYTES = 12_000
MAX_CHUNK_CHARS = 1_000
MAX_MEMORY_BYTES = 3_000
MAX_MEMORIES = 8
MAX_SHORTLIST = 40
MAX_SCAN_CHARS = 20_000
MAX_QUERY_BYTES = 8_192
MAX_QUERY_TERMS = 64
MAX_PROMPT_BYTES = 24_576

# Remove conversational/function words, not domain words such as risk or method.
_LATIN_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it me of on or "
    "that the their these this to was what when which who why will with would "
    "you your please paper papers document documents research summarize summary "
    "overview explain tell about".split()
)
_HAN_STOP = frozenset(
    "的 的是 是什 什么 么是 有什 我们 你们 他们 是否 怎么 如何 为什 为什么 "
    "可以 能否 这个 这些 那些 一下 帮我 给我 请问 对于 以及 因为 所以 "
    "论文 文献 资料 文档 文章 总结 概括 概述 综述 结一 结这 述这 析这 "
    "这篇 这几 几篇 篇论 些论 些文 所有 全部 主要 内容 观点 要点".split()
)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")
_SUMMARY_RE = re.compile(r"总结|概括|概述|综述|summari[sz]e|summary|overview", re.I)
_LABEL_RE = re.compile(r"S[1-9]\d*\Z")
_KNOWLEDGE_LABEL_RE = re.compile(r"K[1-9][0-9]*\Z")
_SUMMARY_GENERIC_RE = re.compile(
    r"总结|概括|概述|综述|一下|一下子|给我|帮我|请|这些|这几篇|这篇|所有|全部|"
    r"上传的|已上传|论文|文献|资料|文档|文章|主要|内容|观点|要点|的|和|与|"
    r"summari[sz]e|summary|overview|papers?|documents?|research|these|all|the|"
    r"uploaded|please|for me|of|an?", re.I,
)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    return value


def _token_spans(text: str):
    for match in _TOKEN_RE.finditer(text):
        word = match.group(0).lower()
        start = match.start()
        if "\u3400" <= word[0] <= "\u9fff":
            pieces = ((word[index:index + 2], start + index, start + index + 2)
                      for index in range(len(word) - 1)) if len(word) > 1 else ((word, start, start + 1),)
            for token, left, right in pieces:
                if token not in _HAN_STOP:
                    yield token, left, right
        elif word not in _LATIN_STOP:
            yield word, start, match.end()


def lexical_tokens(text: str) -> list[str]:
    """Latin words and Han bigrams, preserving occurrences for BM25 counts."""
    return [token for token, _, _ in _token_spans(_text(text))]


def refresh_query_coverage(plan: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind a word-form diagnostic to the actual excerpt packet, without ranking.

    Keep raw query tokens in both the receipt and gap; normalized aliases are
    never presented as new query terms or as semantic/whole-document support.
    Old plans without query tokens cannot be reconstructed from gap text.
    """
    # This function is re-run for every final prompt source variant, including
    # supplemental evidence. A diagnostic from the earlier eight-source pack
    # cannot certify later-added or shortened excerpts.
    gaps = [gap for gap in plan.get('gaps', []) if gap.get('code') != 'source_antecedent_unverified']
    if any(opening_reference_unverified(source.get('content', '')) for source in sources):
        gaps.append({'code': 'source_antecedent_unverified',
                     'message': '部分节选开头的回指对象尚未核对；不能把后续章节标题或其他相似片段自动当作其归属。'})
    terms = plan.get('query_terms')
    if not isinstance(terms, (list, tuple)):
        return {**plan, 'gaps': gaps}
    coverage = surface_term_coverage(terms, (
        token for source in sources for token in lexical_tokens(source.get('content', ''))))
    gaps = [gap for gap in gaps if gap.get('code') != 'uncovered_query_terms']
    if coverage['uncovered'] and plan.get('selection_strategy') != 'document_overview':
        gaps.append({'code': 'uncovered_query_terms', 'terms': coverage['uncovered'],
            'message': '部分原始检索词未在本次片段中匹配到原词或保守规则复数形式；'
                       '这只是词形提示，不表示概念缺失、全文未提及或结论已获支持。'})
    return {**plan, 'covered_query_terms': coverage['covered'], 'gaps': gaps}


def _limit_int(value: Any, default: int, cap: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return min(cap, max(0, value))


def _stable_key(row: Mapping[str, Any]) -> tuple:
    ordinal = row.get("ordinal")
    return (
        _text(row.get("document_id")),
        ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) else 0,
        _text(row.get("id")),
        hashlib.sha256(_text(row.get("content")).encode("utf-8")).hexdigest(),
    )


def _source_row(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    content = _text(raw.get("content"))
    identifier = _text(raw.get("id"))
    document_id = _text(raw.get("document_id"))
    if not content.strip() or not identifier or not document_id or len(identifier) > 256 or len(document_id) > 256:
        return None
    # A caller must not turn generated graph edges or saved hypotheses into quotes.
    if raw.get("is_generated") in (True, "true", "1") or raw.get("is_discovered") in (True, "true", "1"):
        return None
    if isinstance(raw.get("provenance"), str) and raw["provenance"] in {
        "llm", "self_study", "self_evolution", "llm_expansion", "hypothesis", "inferred", "memory",
    }:
        return None
    result = {
        "id": identifier, "document_id": document_id,
        "content": content, "filename": _text(raw.get("filename"))[:120] or "未命名资料",
        "ordinal": raw.get("ordinal") if isinstance(raw.get("ordinal"), int)
        and not isinstance(raw.get("ordinal"), bool) else 0,
    }
    page = raw.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        result["page"] = page
    return result


def _eligible_memories(memories: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    eligible = []
    for row in memories:
        if not isinstance(row, Mapping):
            continue
        # Explicit confirmation only. Removed notes stay removed even if an old
        # confirmation flag is retained in an audit record.
        status = row.get("status")
        if status is not None and not isinstance(status, str):
            continue
        if status in {"deleted", "invalidated", "candidate", "rejected", "unconfirmed"}:
            continue
        accepted = (status == "confirmed" or
                    (status in {"active", "archived"} and row.get("confirmed_by_user") is True) or
                    (status in {None, "", "active"} and row.get("confirmed") is True))
        if not accepted:
            continue
        content = _text(row.get("content")) or _text(row.get("text"))
        if not content.strip():
            continue
        eligible.append({"id": _text(row.get("id"))[:128], "content": content,
                         **{key: row[key] for key in ('valid_from', 'valid_until', 'recorded_at') if key in row}})
    return eligible


def _confirmed_memories(memories: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    remaining = MAX_MEMORY_BYTES
    for row in _eligible_memories(memories):
        if not remaining or len(selected) >= MAX_MEMORIES:
            continue
        content = row["content"]
        # Preserve the entire user-approved meaning, especially final negations
        # and qualifying clauses. Oversized notes are skipped, never truncated.
        size = len(content.encode("utf-8"))
        if size > remaining:
            continue
        selected.append(row)
        remaining -= size
    return selected


def _excerpt(row: dict[str, Any], terms: set[str], idf: Mapping[str, float], chars: int,
             byte_limit: int = MAX_SOURCE_BYTES,
             semantic_span: tuple[int, int] | None = None,
             metadata_header_end: int = 0) -> dict[str, Any] | None:
    """Select one untouched span without cutting a sentence or its qualifier.

    An oversized indivisible statement is omitted, not made to look complete by
    dropping its tail. Source chunk boundaries can still be inherited parser
    cuts; the metadata intentionally does not claim semantic completeness.
    """
    text = row["content"]
    start, end = 0, len(text)
    scan = text[:MAX_SCAN_CHARS]
    units = evidence_units(scan)
    structural_headings = section_heading_spans(scan)
    headings, heading_index = [], 0
    for left, right in units:
        while heading_index < len(structural_headings) and structural_headings[heading_index][1] <= left:
            heading_index += 1
        heading = False
        if heading_index < len(structural_headings):
            hleft, hright = structural_headings[heading_index]
            heading = (max(left, hleft) < min(right, hright)
                       and scan[left:right].strip() == scan[hleft:hright].strip())
        # A reliable initial title block is relevant evidence only for explicit
        # author/contact metadata requests. Keep that block continuous rather
        # than dropping its query anchor at each author/title-like line.
        headings.append(heading and not (metadata_header_end and right <= metadata_header_end))
    interior_heading = any(heading and scan[:units[index][0]].strip()
                           for index, heading in enumerate(headings))
    # Some PDFs put an inline subsection label after a caption rather than a
    # standalone heading. Only this narrow visible label shape separates the
    # caption band; ordinary multi-sentence captions remain intact.
    caption_label_boundaries = set()
    after_caption = False
    for index, (left, right) in enumerate(units):
        value = scan[left:right]
        if headings[index]:
            after_caption = False
        if (after_caption and value.startswith('\n') and re.fullmatch(
                r'\s*(?:(?:related|prior|previous|other)\s+[a-z][a-z-]{1,35}|'
                r'[a-z]{1,20}-[a-z]{2,30})\.\s*', value, re.I)):
            caption_label_boundaries.add(index)
            after_caption = False
        if is_figure_caption_start(value.strip()):
            after_caption = True
    if len(text) > chars or len(text.encode("utf-8")) > byte_limit or interior_heading or caption_label_boundaries:
        if len(scan) < len(text):
            # The last scan span might stop inside an as-yet unseen condition.
            units = units[:-1]
        sizes = [len(scan[left:right].encode("utf-8")) for left, right in units]
        matches = [set(lexical_tokens(scan[left:right])) & terms for left, right in units]
        left_index = 0
        counts: Counter[str] = Counter()
        score = 0.0
        size = 0
        best: tuple[float, float, int, int, int] | None = None
        for right_index, (_, right) in enumerate(units):
            if headings[right_index] or right_index in caption_label_boundaries:
                # An explicit new section starts a fresh candidate band.
                # Keep its exact heading with its own body where it fits, but
                # never let a previous task's continuation inherit this title.
                # No chart values or relation between bands are inferred.
                left_index = right_index
                counts.clear()
                score = 0.0
                size = 0
            size += sizes[right_index]
            for token in matches[right_index]:
                if not counts[token]:
                    score += idf.get(token, 1.0)
                counts[token] += 1
            while left_index <= right_index and (right - units[left_index][0] > chars or size > byte_limit):
                size -= sizes[left_index]
                for token in matches[left_index]:
                    counts[token] -= 1
                    if not counts[token]:
                        score -= idf.get(token, 1.0)
                left_index += 1
            if left_index > right_index:
                continue
            if headings[right_index]:
                continue  # A heading alone is not substantive source evidence.
            left = units[left_index][0]
            semantic_overlap = (max(0, min(right, semantic_span[1]) - max(left, semantic_span[0]))
                                if semantic_span is not None else 0)
            if semantic_span is not None and not semantic_overlap:
                continue
            if semantic_span is None and terms and score <= 1e-9:
                continue
            # Equal query coverage keeps more adjacent source context, then the
            # earliest source position. Never concatenate disconnected sentences.
            # A qualified semantic passage locates evidence within a larger
            # chunk even when a Chinese query has no English lexical overlap.
            candidate = (semantic_overlap, round(score, 12), right - left, -left, right)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            return None
        start, end = -best[3], best[4]
    result = dict(row)
    result.update({
        "content": text[start:end], "truncated": start > 0 or end < len(text),
        "excerpt_start": start, "excerpt_end": end,
        "offset_unit": "characters_in_source_chunk", "original_content_length": len(text),
        "evidence_scope": "excerpt_only" if start > 0 or end < len(text) else "source_chunk",
        "sentence_boundary_preserved": True,
    })
    return result


def build_retrieval_plan(
    question: str,
    chunks: Sequence[Mapping[str, Any]],
    memories: Sequence[Mapping[str, Any]] = (),
    *,
    mode: str = "evidence",
    max_chunks: int = MAX_CHUNKS,
    max_source_bytes: int = MAX_SOURCE_BYTES,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    query_terms_override: Sequence[str] | None = None,
    semantic_scores: Mapping[str, float] | None = None,
    semantic_spans: Mapping[str, Sequence[int]] | None = None,
    prepared_corpus=None,
) -> dict[str, Any]:
    """Select real sentence windows with BM25, optional RRF and bounded MMR.

    ``mode='explore'`` changes answer instructions, never the provenance of quotes.
    Generic document summaries sample introductions explicitly. Focused questions
    need lexical hits or caller-prevalidated semantic candidates; no unrelated
    padding is added. ``semantic_scores`` must contain candidates already gated
    by their backend. Neither a lexical hit nor cosine similarity establishes a
    causal connection, correctness or semantic entailment. ``semantic_spans``
    locates a qualified backend passage in original half-open character offsets;
    malformed spans are ignored and cannot admit an unqualified source. Original source chunk
    boundaries may still have been truncated upstream by a legacy parser.
    """
    question = _text(question)
    if not question.strip() or len(question.encode("utf-8")) > MAX_QUERY_BYTES:
        raise ValueError("Question must be nonempty UTF-8 text within 8192 bytes")
    if mode not in {"evidence", "explore"}:
        raise ValueError("Unsupported research mode")
    max_chunks = _limit_int(max_chunks, MAX_CHUNKS, MAX_CHUNKS)
    byte_budget = _limit_int(max_source_bytes, MAX_SOURCE_BYTES, MAX_SOURCE_BYTES)
    char_budget = _limit_int(max_chunk_chars, MAX_CHUNK_CHARS, MAX_CHUNK_CHARS)
    if query_terms_override is not None:
        if (not isinstance(query_terms_override, (list, tuple)) or
                len(query_terms_override) > MAX_QUERY_TERMS or
                any(not _text(value).strip() or len(value.encode("utf-8")) > 256 for value in query_terms_override) or
                sum(len(value.encode("utf-8")) for value in query_terms_override) > MAX_QUERY_BYTES):
            raise ValueError("Query term override must contain at most 64 bounded UTF-8 terms")
        terms = list(dict.fromkeys(token for value in query_terms_override for token in lexical_tokens(value)))[:MAX_QUERY_TERMS]
    else:
        terms = list(dict.fromkeys(lexical_tokens(question)))[:MAX_QUERY_TERMS]
    if semantic_scores is not None and not isinstance(semantic_scores, Mapping):
        raise ValueError("Semantic scores must map source IDs to finite similarity scores")
    # Generic summary is intentional document sampling, not irrelevant padding.
    summary_residue = _SUMMARY_GENERIC_RE.sub(" ", question)
    broad_summary = bool(_SUMMARY_RE.search(question)) and not lexical_tokens(summary_residue)
    query_terms = set(terms)
    rows: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, str]] = set()
    seen_text: set[str] = set()
    duplicates = 0
    document_ids: set[str] = set()
    df: Counter[str] = Counter()
    total_length = 0
    # Sort only compact source rows for stable deduplication of duplicate uploads.
    if prepared_corpus is not None:
        from .research_corpus_index import PreparedResearchCorpus, canonical_source_view
        entries, signature = canonical_source_view(chunks)
        if (not isinstance(prepared_corpus, PreparedResearchCorpus)
                or prepared_corpus.signature != signature):
            raise ValueError('Lexical features do not match the current source view')
        valid_rows = [row for row, _ in entries]
    else:
        valid_rows = [row for raw in chunks if (row := _source_row(raw)) is not None]
        valid_rows.sort(key=_stable_key)
    metadata_headers = ({(hint['document_id'], hint['source_id']): hint['abstract_body_start']
                         for hint in initial_title_hints(valid_rows)}
                        if _METADATA_QUERY.search(question) else {})
    if prepared_corpus is not None:
        document_ids = set(prepared_corpus.document_ids)
        duplicates = prepared_corpus.duplicate_count
        total_length = prepared_corpus.total_length
        matches = {}
        for term in query_terms:
            buffer = prepared_corpus.postings.get(term, b'')
            df[term] = len(buffer) // 8
            for index, frequency in prepared_corpus.hits(term):
                matches.setdefault(index, {})[term] = frequency
        active = set(matches)
        if broad_summary:
            active.update(range(len(prepared_corpus.source_indices)))
        elif semantic_scores is not None:
            # Preserve the backend's exact gate. The semantic lane may inspect
            # IDs, but must not re-tokenize all source text for every query.
            for index, source_index in enumerate(prepared_corpus.source_indices):
                score = semantic_scores.get(valid_rows[source_index]['id'], 0.0)
                if (not isinstance(score, bool) and isinstance(score, (int, float))
                        and math.isfinite(score) and 0.01 <= score <= 1):
                    active.add(index)
        indexed_rows = [(valid_rows[prepared_corpus.source_indices[index]],
                         prepared_corpus.lengths[index], matches.get(index, {}),
                         prepared_corpus.stable_keys[index]) for index in sorted(active)]
    else:
        indexed_rows = [(row, None, None, None) for row in valid_rows]
    for row, cached_length, cached_matching, cached_key in indexed_rows:
        if prepared_corpus is None:
            identity = (row["document_id"], row["id"])
            content_hash = hashlib.sha256(" ".join(row["content"].split()).encode("utf-8")).hexdigest()
            document_ids.add(row["document_id"])
            if identity in seen_ids or content_hash in seen_text:
                duplicates += 1
                continue
            seen_ids.add(identity)
            seen_text.add(content_hash)
            counter = Counter(lexical_tokens(row["content"][:MAX_SCAN_CHARS]))
            length = sum(counter.values())
            matching = {term: counter[term] for term in query_terms if counter[term]}
            df.update(matching.keys())
            total_length += length
        else:
            length, matching = cached_length, cached_matching
        semantic = (semantic_scores or {}).get(row["id"], 0.0)
        if (isinstance(semantic, bool) or not isinstance(semantic, (int, float)) or
                not math.isfinite(semantic) or not 0.01 <= semantic <= 1):
            semantic = 0.0
        span = semantic_spans.get(row["id"]) if isinstance(semantic_spans, Mapping) and semantic else None
        if (not isinstance(span, (list, tuple)) or len(span) != 2 or
                any(type(position) is not int for position in span) or
                not 0 <= span[0] < span[1] <= len(row["content"])):
            span = None
        else:
            span = tuple(span)
        rows.append({"row": row, "length": max(1, length), "matching": matching,
                     "semantic": semantic, "semantic_span": span,
                     "_sort_key": cached_key or _stable_key(row)})
    count = len(prepared_corpus.source_indices) if prepared_corpus is not None else len(rows)
    average_length = max(1.0, total_length / max(1, count))
    idf = {term: math.log(1.0 + (count - df[term] + 0.5) / (df[term] + 0.5)) for term in query_terms}
    candidates = []
    if broad_summary:
        # First source chunk per document: bounded opening excerpts, not full coverage.
        representatives: dict[str, dict] = {}
        for item in rows:
            representatives.setdefault(item["row"]["document_id"], item)
        shortlist = list(representatives.values())[:MAX_SHORTLIST]
        strategy = "document_overview"
    else:
        for item in rows:
            if not item["matching"] and not item["semantic"]:
                continue
            denominator = 1.2 * (0.25 + 0.75 * item["length"] / average_length)
            item["score"] = sum(idf[term] * frequency * 2.2 / (frequency + denominator)
                                for term, frequency in item["matching"].items())
            candidates.append(item)
        if semantic_scores is not None:
            # The semantic backend gates its candidates. Rank fusion avoids
            # treating E5 cosine values as calibrated BM25 scores/probabilities.
            lexical_order = sorted((item for item in candidates if item["matching"]),
                                   key=lambda item: (-item["score"], item['_sort_key']))
            semantic_order = heapq.nsmallest(MAX_SHORTLIST, (item for item in candidates if item["semantic"]),
                                            key=lambda item: (-item["semantic"], item['_sort_key']))
            lexical_ranks = {id(item): rank for rank, item in enumerate(lexical_order, 1)}
            semantic_ranks = {id(item): rank for rank, item in enumerate(semantic_order, 1)}
            for item in candidates:
                item["semantic"] = item["semantic"] if id(item) in semantic_ranks else 0.0
                item["score"] = ((1 / (60 + lexical_ranks[id(item)]) if id(item) in lexical_ranks else 0.0) +
                                 (1 / (60 + semantic_ranks[id(item)]) if id(item) in semantic_ranks else 0.0))
            candidates = [item for item in candidates if item["score"] > 0]
        shortlist = heapq.nsmallest(MAX_SHORTLIST, candidates,
                                   key=lambda item: (-item["score"], item['_sort_key']))
        strategy = "hybrid_bm25_semantic_mmr" if semantic_scores is not None else "lexical_bm25_mmr"
    prepared = []
    omitted_windows = 0
    for item in shortlist if char_budget else []:
        window_terms = set() if broad_summary or not item["matching"] else query_terms
        semantic_span = item["semantic_span"] if not broad_summary and item["semantic"] else None
        metadata_header_end = metadata_headers.get((item['row']['document_id'], item['row']['id']), 0)
        excerpt = _excerpt(item["row"], window_terms, idf, char_budget, byte_budget, semantic_span,
                           metadata_header_end)
        if excerpt is None:
            omitted_windows += 1
            continue
        token_set = set(lexical_tokens(excerpt["content"]))
        matched = token_set & query_terms
        if not excerpt["content"].strip() or (not broad_summary and not matched and not item["semantic"]):
            continue
        prepared.append({"row": excerpt, "tokens": token_set, "matched": matched,
                         "score": item.get("score", 1.0), "original_row": item["row"],
                         "_sort_key": _stable_key(excerpt),
                         "semantic": item["semantic"], "window_terms": window_terms,
                         "semantic_span": semantic_span, "metadata_header_end": metadata_header_end})
    max_score = max((item["score"] for item in prepared), default=1.0)
    selected: list[dict[str, Any]] = []
    selected_tokens: list[set[str]] = []
    covered: set[str] = set()
    selected_documents: set[str] = set()
    remaining = byte_budget
    while prepared and len(selected) < max_chunks and remaining > 0:
        scored = []
        for item in prepared:
            tokens = item["tokens"]
            redundancy = max((len(tokens & other) / max(1, len(tokens | other))
                              for other in selected_tokens), default=0.0)
            # Penalize near duplicates but do not discard them: a changed negation
            # in otherwise identical research prose may be an important conflict.
            # Exact duplicate quotes were already removed above.
            coverage = len(item["matched"] - covered) / max(1, len(query_terms))
            source_bonus = 0.08 if item["row"]["document_id"] not in selected_documents else 0.0
            score = (0.65 * item["score"] / max_score - 0.25 * redundancy
                     + 0.10 * coverage + source_bonus)
            scored.append((score, item['_sort_key'], item))
        if not scored:
            break
        item = min(scored, key=lambda entry: (-entry[0], entry[1]))[2]
        prepared.remove(item)
        row = dict(item["row"])
        if len(row["content"].encode("utf-8")) > remaining:
            row = _excerpt(item["original_row"], item["window_terms"], idf, char_budget, remaining,
                           item["semantic_span"], item['metadata_header_end'])
            if row is None:
                omitted_windows += 1
                continue
        content = row["content"]
        if not content.strip() or (not broad_summary and not item["semantic"] and
                                   not (set(lexical_tokens(content)) & query_terms)):
            continue
        row.update({"content": content, "label": f"S{len(selected) + 1}",
                    "score": round(item["score"], 6)})
        selected.append(row)
        actual_tokens = set(lexical_tokens(content))
        selected_tokens.append(actual_tokens)
        covered.update(actual_tokens & query_terms)
        selected_documents.add(row["document_id"])
        remaining -= len(content.encode("utf-8"))
    gaps: list[dict[str, Any]] = []
    if not selected:
        gaps.append({"code": "insufficient_evidence", "message": "未找到匹配的原文依据；请补充资料或换用原文术语。"})
    if broad_summary:
        gaps.append({"code": "bounded_overview", "message": "仅抽取各资料的开头片段，不代表全文总结或完整比较。"})
    if len(selected_documents) < len(document_ids):
        gaps.append({"code": "documents_not_selected", "count": len(document_ids - selected_documents),
                     "message": "部分资料未进入本次有限上下文；未引用不等于资料不支持或反对。"})
    if any(row["truncated"] for row in selected):
        gaps.append({"code": "excerpt_truncated", "message": "部分依据仅为连续节选，限定条件可能在节选之外。"})
    if omitted_windows:
        gaps.append({"code": "complete_statement_exceeds_budget", "count": omitted_windows,
                     "message": "部分匹配原句及相邻限定条件无法完整放入预算，已跳过；不能把缺失部分当成已核实。"})
    if duplicates:
        gaps.append({"code": "duplicate_chunks_removed", "count": duplicates,
                     "message": "重复原文已合并，不能当作多份独立证据。"})
    sources = [{key: row[key] for key in (
        "id", "document_id", "label", "filename", "content", "ordinal", "truncated",
        "excerpt_start", "excerpt_end", "offset_unit", "original_content_length", "evidence_scope",
        "sentence_boundary_preserved",
    )} | ({"page": row["page"]} if "page" in row else {}) for row in selected]
    context = "\n\n".join(f"[{row['label']}] {row['filename']}\n{row['content']}" for row in selected)
    confirmed_memories = _confirmed_memories(memories)
    omitted_memories = len(_eligible_memories(memories)) - len(confirmed_memories)
    if omitted_memories:
        gaps.append({"code": "confirmed_memory_budget_omitted", "count": omitted_memories,
                     "message": "部分有效确认记忆因上下文预算未载入本次回答；原记忆未被截断或删除。"})
    digest_payload = {"question": question, "mode": mode, "sources": sources,
                      "memories": confirmed_memories}
    return refresh_query_coverage({
        "question": question, "mode": mode, "selection_strategy": strategy,
        "selected_chunks": selected, "sources": sources, "context": context,
        "confirmed_memories": digest_payload["memories"], "gaps": gaps,
        "query_terms": sorted(query_terms), "covered_query_terms": sorted(covered),
        "source_bytes": byte_budget - remaining, "scope": "current_project",
        "evidence_ids": [row["id"] for row in selected],
        "digest": hashlib.sha256(json.dumps(digest_payload, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest(),
        "algorithm": {"name": strategy, "version": 3,
                      "candidate_limit": MAX_SHORTLIST, "model_calls": 0,
                      "max_scanned_chars_per_chunk": MAX_SCAN_CHARS,
                      "sentence_window_selection": True,
                      "semantic_span_selection": isinstance(semantic_spans, Mapping),
                      "semantic_candidates_prevalidated": semantic_scores is not None,
                      "rank_fusion": "rrf_k60" if semantic_scores is not None else None,
                      "semantic_entailment_verified": False},
    }, sources)


def grounded_prompt(question: str, plan: Mapping[str, Any], *, include_component_locators=True) -> str:
    """One bounded evidence-first prompt. JSON fields are data, not instructions.

    Callers should use this as a user-message payload beneath their trusted system
    policy. Neither this delimiter nor citation syntax prevents all prompt attacks.
    """
    from .research_answer_contract import DATA_MARKER, source_quote_payload
    question = _text(question)
    if not question.strip() or len(question.encode("utf-8")) > MAX_QUERY_BYTES:
        raise ValueError("Question must be nonempty UTF-8 text within 8192 bytes")
    payload = {
        "question": question,
        "mode": "explore" if plan.get("mode") == "explore" else "evidence",
        "source_quotes": source_quote_payload(plan.get("selected_chunks", []), question=question,
                                               include_component_locators=include_component_locators),
        "confirmed_user_context_not_evidence": plan.get("confirmed_memories", []),
        "retrieval_gaps": plan.get("gaps", []),
    }
    # Rules live once in the trusted system message, not in a second competing
    # instruction block beside uploaded data. Keep the existing JSON boundary
    # and exact source geometry for adapters, provenance and evaluation tools.
    prompt = DATA_MARKER + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("Research prompt exceeds 24576-byte budget")
    return prompt


def citation_validation(answer: str, plan_or_labels: Mapping[str, Any] | Sequence[str], *,
                        knowledge_labels: Sequence[str] = ()) -> dict[str, Any]:
    """Check syntax/scope, with K methods separate from S user evidence.

    K labels require an explicitly supplied, current knowledge selection. Even
    a known K citation cannot satisfy the requirement to cite uploaded S text.
    Citation-shaped event/condition/goal IDs remain unknown diagnostics, even
    if a caller puts them in source/method labels; they never become evidence.
    This check NEVER claims either kind of reference entails a factual claim.
    """
    if isinstance(plan_or_labels, Mapping):
        labels = {row.get("label") for row in plan_or_labels.get("selected_chunks", [])
                  if isinstance(row, Mapping)}
    else:
        labels = set(plan_or_labels)
    allowed = {label for label in labels if isinstance(label, str) and _LABEL_RE.fullmatch(label)}
    allowed_knowledge = (
        {label for label in knowledge_labels
         if isinstance(label, str) and _KNOWLEDGE_LABEL_RE.fullmatch(label)}
        if isinstance(knowledge_labels, (Sequence, set, frozenset))
        and not isinstance(knowledge_labels, (str, bytes)) else set()
    )
    answer = _text(answer)
    parsed = parse_citation_groups(answer)
    raw_used = parsed['labels']
    raw_source_used = [label for label in raw_used if label.startswith("S")]
    raw_unknown = [label for label in raw_used if label not in allowed | allowed_knowledge]
    # A malicious/generated giant label must remain a warning, not create an
    # oversized response or a persistence failure. Unknown labels stay unknown.
    metadata_truncated = (parsed['truncated'] or len(raw_used) > 32 or len(raw_unknown) > 32 or
                          any(len(label) > 17 for label in raw_used))
    used = list(dict.fromkeys(label[:17] for label in raw_used[:32]))
    unknown = list(dict.fromkeys(label[:17] for label in raw_unknown[:32]))
    missing = bool(allowed and answer.strip() and not raw_source_used)
    warnings = []
    if unknown:
        warnings.append({"code": "unknown_citation_labels", "labels": unknown,
                         "message": "回答包含未提供的引用标签，不能将其作为资料依据。"})
    if missing:
        warnings.append({"code": "missing_citations", "message": "回答未引用本次资料；资料性结论尚未核对。"})
    if not answer.strip():
        warnings.append({"code": "empty_answer", "message": "未收到可用回答。"})
    return {"valid": bool(answer.strip()) and not raw_unknown and not missing and not metadata_truncated,
            "used_labels": used, "unknown_labels": unknown, "missing_citations": missing,
            "warnings": warnings, "scope": "citation_syntax_only", "entailment_verified": False,
            "metadata_truncated": metadata_truncated}


def refresh_citation_validation(answer: str, source_labels: Sequence[str], previous: dict, *,
                                knowledge_labels: Sequence[str] = ()) -> dict[str, Any]:
    """Recheck a replayed body without reviving an older invalid receipt.

    This is a bounded return-time projection, not a metadata rewrite, provider
    request or semantic verdict. Current namespaces expose legacy false-clean
    citations while old revocation/limit warnings remain fail-closed.
    """
    fresh = citation_validation(answer, source_labels, knowledge_labels=knowledge_labels)
    used = list(dict.fromkeys([*fresh['used_labels'], *previous.get('used_labels', [])]))
    unknown = list(dict.fromkeys([*fresh['unknown_labels'], *previous.get('unknown_labels', [])]))
    missing = fresh['missing_citations'] or previous.get('missing_citations', False)
    warnings = {row['code']: row for row in fresh['warnings']}
    for row in previous.get('warnings', []):
        warnings.setdefault(row['code'], row)
    if unknown and 'unknown_citation_labels' in warnings:
        warnings['unknown_citation_labels'] = {**warnings['unknown_citation_labels'], 'labels': unknown[:32]}
    truncated = (fresh['metadata_truncated'] or previous.get('metadata_truncated', False)
                 or len(used) > 32 or len(unknown) > 32 or len(warnings) > 8)
    return {**fresh, 'valid': bool(fresh['valid'] and previous.get('valid') is True
                                  and not unknown and not missing and not truncated),
            'used_labels': used[:32], 'unknown_labels': unknown[:32],
            'missing_citations': missing, 'warnings': list(warnings.values())[:8],
            'metadata_truncated': truncated}


def export_answer_markdown(question: str, answer: str, plan: Mapping[str, Any]) -> str:
    """Literal local export. No evaluation, regeneration or extra source fetching."""
    lines = ["# 研究回答记录", "", "## 问题", "", _text(question), "", "## 回答", "", _text(answer),
             "", "## 本次原文依据", "", "引用标签仅标识原文片段，不代表已验证结论成立。"]
    for row in plan.get("selected_chunks", []):
        title = _text(row.get("filename"))
        page = f"，第 {row['page']} 页" if row.get("page") else ""
        clipped = "；连续节选已截断" if row.get("truncated") else ""
        lines.extend(["", f"[{row['label']}] {title}{page}{clipped}", "", _text(row.get("content"))])
    lines.extend(["", "## 检索限制", ""])
    lines.extend(f"- {_text(gap.get('message'))}" for gap in plan.get("gaps", []) if isinstance(gap, Mapping))
    if not plan.get("gaps"):
        lines.append("- 仅对本次有限原文节选进行回答；未执行语义蕴含核验。")
    return "\n".join(lines)
