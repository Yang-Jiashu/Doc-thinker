"""Frozen pre-index rank oracle (2026-10-13).

Only this ranking function is frozen; excerpt/bounds/token helpers are explicit
imports and must remain unchanged while testing this exact-equivalence change.
Synthetic/offline tests only. No model or corpus data is included.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from docthinker.server.research_retrieval import (
    MAX_CHUNKS, MAX_SOURCE_BYTES, MAX_CHUNK_CHARS, MAX_QUERY_BYTES,
    MAX_QUERY_TERMS, MAX_SCAN_CHARS, MAX_SHORTLIST,
    _text, _limit_int, _SUMMARY_GENERIC_RE, _SUMMARY_RE, _source_row,
    _stable_key, _excerpt, _confirmed_memories, _eligible_memories,
    lexical_tokens, refresh_query_coverage,
)
from docthinker.server.research_document_titles import initial_title_hints
from docthinker.server.research_fallback_sources import _METADATA_QUERY


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
    valid_rows = [row for raw in chunks if (row := _source_row(raw)) is not None]
    valid_rows.sort(key=_stable_key)
    metadata_headers = ({(hint['document_id'], hint['source_id']): hint['abstract_body_start']
                         for hint in initial_title_hints(valid_rows)}
                        if _METADATA_QUERY.search(question) else {})
    for row in valid_rows:
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
                     "semantic": semantic, "semantic_span": span})
    count = len(rows)
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
                                   key=lambda item: (-item["score"], _stable_key(item["row"])))
            semantic_order = sorted((item for item in candidates if item["semantic"]),
                                    key=lambda item: (-item["semantic"], _stable_key(item["row"])))[:MAX_SHORTLIST]
            lexical_ranks = {id(item): rank for rank, item in enumerate(lexical_order, 1)}
            semantic_ranks = {id(item): rank for rank, item in enumerate(semantic_order, 1)}
            for item in candidates:
                item["semantic"] = item["semantic"] if id(item) in semantic_ranks else 0.0
                item["score"] = ((1 / (60 + lexical_ranks[id(item)]) if id(item) in lexical_ranks else 0.0) +
                                 (1 / (60 + semantic_ranks[id(item)]) if id(item) in semantic_ranks else 0.0))
            candidates = [item for item in candidates if item["score"] > 0]
        candidates.sort(key=lambda item: (-item["score"], _stable_key(item["row"])))
        shortlist = candidates[:MAX_SHORTLIST]
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
            scored.append((score, _stable_key(item["row"]), item))
        if not scored:
            break
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        item = scored[0][2]
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
