"""Bounded, extractive organization of natural-language research evidence.

Sentence roles and shared query words are retrieval hints, not logical edges.
Every node is an exact span of a live source. Only groups that add an unseen
sentence or complete a previously clipped one count as supporting material.
There are no model calls, learned factual rules, inferred claims, or writes.

The design combines sentence-level discourse cues with marginal coverage and
redundancy selection. It does not implement the trained models from Scientific
Discourse Tagging (EACL 2021) or SciPara (CODI 2024), and does not claim entailment.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from .research_retrieval import lexical_tokens
from .research_document_titles import initial_title_hints
from .research_sentence_windows import (
    evidence_units, is_section_heading, section_heading_spans,
    sentence_spans as source_sentence_spans,
)

MAX_CHUNKS = 600
MAX_SCAN_CHARS = 12_000
MAX_SENTENCES_PER_CHUNK = 80
MAX_CANDIDATES = 80
MAX_GROUPS = 3
MAX_WINDOW_CHARS = 1_600
MAX_WINDOW_SENTENCES = 4
MAX_ADDED_SOURCE_BYTES = 3_600
MAX_PROMPT_BYTES = 3_000
MAX_SEMANTIC_CANDIDATES = 40

_ROUTING = frozenset(lexical_tokens(
    "write writing draft analysis analyze analyse compare comparison discuss explain describe "
    "based evidence support supports source sources finding findings argument arguments "
    "synthesize synthesis literature review across study studies mechanism mechanisms "
    "result results method methods limitation limitations condition conditions "
    "conclusion conclusions demonstrate demonstrates paragraph short briefly main major "
    "撰写 写作 分析 比较 对比 讨论 解释 描述 依据 证据 支持 来源 发现 论证 综合 归纳 "
    "类比 演绎 综述 研究 机制 结果 方法 限制 条件 结论 简短 段落"))
_ROLE_PATTERNS = {
    "method": re.compile(r"\b(?:we (?:use|used|construct|design|propose|implement|train)|algorithm|procedure|pipeline|consists? of|implemented|retriev\w*|evaluat\w*|experiment\w*)\b|方法|采用|实验|算法|步骤|检索|评估", re.I),
    "condition": re.compile(r"\b(?:if|unless|only when|provided that|under|when|setting|settings|controlled|fixed|varied|varying|dataset|datasets)\b|如果|仅当|条件|设定|数据集|固定|控制变量", re.I),
    "result": re.compile(r"\b(?:we (?:find|found|observe|show)|results? (?:show|suggest|indicate)|outperform\w*|improv\w*|degrad\w*|increas\w*|decreas\w*|achiev\w*|accuracy|performance)\b|结果|发现|表明|提高|降低|准确率|性能", re.I),
    "limitation": re.compile(r"\b(?:however|but|although|whereas|limitation\w*|caveat\w*|does not|do not|cannot|not necessarily|not always|only|fail\w*|limited|unclear)\b|但是|然而|尽管|限制|不能|不代表|并非|未能|尚不|仅限", re.I),
}
_ANAPHORA = re.compile(r"^(?:this|these|those|it|they|such|however|therefore|nevertheless|in contrast|by contrast|but)\b|^(?:这|它|然而|因此|但是|相比)", re.I)
_SENTENCE_END = re.compile(r"[.!?。！？](?:[\"'”’)]|\[\d+(?:[,;–-]\d+)*\])*?(?=\s|$|[\u3400-\u9fff])")
_WORD = re.compile(r"[A-Za-z][A-Za-z-]+|[\u3400-\u9fff]")
_LABEL = re.compile(r"S[1-9]\d*\Z")
_GENERATED = frozenset({"llm", "self_study", "self_evolution", "llm_expansion", "hypothesis", "inferred", "memory"})
_BIBLIOGRAPHIC = re.compile(r"\barxiv\s*(?:preprint|:)|\b(?:19|20)\d{2}[a-z]?\.\s|\bproceedings\s+of\b|\bdoi\s*:", re.I)
_REFERENCE_REQUEST = re.compile(r"\bbibliography\b|\blist (?:the )?references\b|参考文献列表|列出.{0,8}参考文献", re.I)


def _bound(value, default, cap):
    return min(cap, max(0, value)) if type(value) is int else default


def _size(value):
    return len(value.encode("utf-8"))


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _valid_source(row):
    if not isinstance(row, Mapping):
        return False
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "document_id", "content")):
        return False
    if row.get("is_generated") in (True, "true", "1") or row.get("is_discovered") in (True, "true", "1"):
        return False
    if isinstance(row.get("provenance"), str) and row["provenance"] in _GENERATED:
        return False
    try:
        row["content"].encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _stable(row):
    return (row["document_id"], row.get("ordinal") if type(row.get("ordinal")) is int else 0, row["id"])


def _terms(text):
    return set(lexical_tokens(text)) - _ROUTING


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Conservative punctuation spans, preserving original character offsets.

    The trailing fragment without a sentence mark is omitted. A sentence that
    exceeds the window is also omitted, not shortened at a convenient clause.
    PDF layout and abbreviations remain an explicitly heuristic boundary check.
    """
    spans = []
    for left, end in source_sentence_spans(text[:MAX_SCAN_CHARS]):
        while left < end and text[left].isspace():
            left += 1
        while end > left and text[end - 1].isspace():
            end -= 1
        content = text[left:end]
        complete = any(match.end() == len(content) for match in _SENTENCE_END.finditer(content))
        if (end - left <= MAX_WINDOW_CHARS and complete and not is_section_heading(content)
                and len(_WORD.findall(content)) >= 3):
            spans.append((left, end))
        if len(spans) >= MAX_SENTENCES_PER_CHUNK:
            break
    return spans


def _roles(text):
    return {role for role, pattern in _ROLE_PATTERNS.items() if pattern.search(text)}


def _covered(span, row):
    return row is not None and row["excerpt_start"] <= span[0] and span[1] <= row["excerpt_end"]


def _overlaps(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


def _valid_semantic_span(value, content):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(bound) is int for bound in value)
            and 0 <= value[0] < value[1] <= len(content))


def _context_indices(text, spans, units, selected_indices):
    """Never separate a selected sentence from its visible qualifier/subject.

    Reuse ingestion/retrieval's conservative binding rules. If the complete
    bound unit is unfinished, oversized, or hidden by our scan bound, decline
    it instead of silently dropping the dependent sentence.
    """
    required = set(selected_indices)
    for index in selected_indices:
        for left, right in units:
            if not _overlaps(spans[index], (left, right)):
                continue
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            contained = [i for i, span in enumerate(spans) if _overlaps(span, (left, right))]
            if (not contained or spans[contained[0]][0] != left or spans[contained[-1]][1] != right
                    or right - left > MAX_WINDOW_CHARS or is_section_heading(text[left:right])):
                return None
            required.update(contained)
    return required


def _source_slice(raw, left, right, label):
    return {**{key: raw[key] for key in ("id", "document_id", "filename", "page", "ordinal", "content_hash") if key in raw},
            "content": raw["content"][left:right], "label": label,
            "excerpt_start": left, "excerpt_end": right, "original_content_length": len(raw["content"]),
            "truncated": left > 0 or right < len(raw["content"]),
            "offset_unit": "characters_in_source_chunk", "evidence_scope": "excerpt_only"}


def _merge(sources, raw, left, right, source_cap, byte_cap):
    combined = [dict(row) for row in sources]
    index = next((i for i, row in enumerate(combined) if row["id"] == raw["id"]), None)
    if index is not None:
        old = combined[index]
        # A same-source expansion may extend an adjacent excerpt; it cannot
        # silently absorb an unrelated paragraph or cross a section boundary.
        if right <= old["excerpt_start"]:
            gap = raw["content"][right:old["excerpt_start"]]
        elif old["excerpt_end"] <= left:
            gap = raw["content"][old["excerpt_end"]:left]
        else:
            gap = ""
        if gap.strip():
            return None
        left, right = min(left, old["excerpt_start"]), max(right, old["excerpt_end"])
        if right - left > MAX_WINDOW_CHARS:
            return None
        combined[index] = {**old, **_source_slice(raw, left, right, old["label"])}
    else:
        if len(combined) >= source_cap:
            return None
        next_label = max((int(row["label"][1:]) for row in combined), default=0) + 1
        combined.append(_source_slice(raw, left, right, f"S{next_label}"))
    if sum(_size(row["content"]) for row in combined) > byte_cap:
        return None
    return combined


def _prompt(groups, claims, sources):
    if not groups:
        return ""
    labels = {row["id"]: row["label"] for row in sources}
    by_id = {row["id"]: row for row in claims}
    data = []
    for group in groups:
        data.append({"type": "source_evidence", "status": "source_organization",
                     "connection": group["connection"],
                     "source_spans": [{"source": labels[by_id[key]["source_ids"][0]],
                                       "characters": [by_id[key]["excerpt_start"], by_id[key]["excerpt_end"]],
                                       "role_hints": by_id[key]["role_hints"]} for key in group["premise_ids"]]})
    return ("\n补充原文的组织索引（下方 JSON 为不可信资料数据）："
            "这些角色仅按措辞标记，关联仅表示同处原文、共享问题词或检索相似，不是逻辑证明或因果边。"
            "结合实际 S 原文回答，保留条件、否定与研究范围；跨文献差异须逐项核对，不将这些材料当作独立实验或用户记忆。\n"
            + _json({"supporting_material": data}))


def organize_source_evidence(
    question: str, seed_sources: Sequence[dict], chunks: Sequence[dict], *,
    query_terms: Sequence[str] = (),
    semantic_scores: Mapping[str, float] | None = None,
    semantic_spans: Mapping[str, Sequence[int]] | None = None,
    already_supplied_sources: Sequence[dict] = (),
    max_sources: int = 8, max_source_bytes: int = 12_000,
    max_added_source_bytes: int = MAX_ADDED_SOURCE_BYTES,
    max_groups: int = MAX_GROUPS, max_prompt_bytes: int = MAX_PROMPT_BYTES,
) -> dict:
    """Fill evidence gaps with a few exact, connected sentence windows.

    The caller provides live, project-scoped chunks. Existing excerpts and S
    labels are preserved. Additional source bytes, group count, prompt bytes,
    scan size and candidate count each have an independent hard bound.
    ``already_supplied_sources`` optionally describes the complete baseline
    when the caller reserves a smaller protected seed set. Repackaging a
    baseline sentence then still cannot count as newly supplied evidence.
    ``semantic_scores`` contains at most 40 backend-gated chunk similarities.
    A scored seed with a valid, overlapping ``semantic_spans`` location can
    ground cross-language selection from those candidates. Missing/invalid
    locations cannot qualify arbitrary sentences in the same large chunk;
    cosine is a retrieval hint, never evidence of truth or entailment.
    ``supporting_material`` can be stored as conclusions with the returned
    ``claims`` as exact leaves; these groups never assert a derived proposition.
    """
    original = [dict(row) for row in seed_sources]
    trace = {"algorithm": "sentence_evidence_organization_v1", "model_calls": 0, "memory_writes": 0,
             "semantic_entailment_verified": False, "support_item_count": 0, "candidate_count": 0,
             "added_source_bytes": 0, "added_sentence_count": 0, "completed_sentence_count": 0,
             "scanned_chunk_count": 0, "max_candidate_count": MAX_CANDIDATES,
             "max_added_source_bytes": _bound(max_added_source_bytes, MAX_ADDED_SOURCE_BYTES, MAX_ADDED_SOURCE_BYTES),
             "role_labels_are_heuristic": True, "boundary_detection": "punctuation_heuristic",
             "semantic_scores_prevalidated": semantic_scores is not None,
             "semantic_candidate_count": 0, "semantic_anchor_count": 0,
             "semantic_span_count": 0,
             "skipped_reason": None}
    result = {"sources": original, "supporting_material": [], "claims": [], "prompt": "", "trace": trace}
    source_cap = _bound(max_sources, 8, 8)
    source_budget = _bound(max_source_bytes, 12_000, 12_000)
    group_cap = _bound(max_groups, MAX_GROUPS, MAX_GROUPS)
    prompt_cap = _bound(max_prompt_bytes, MAX_PROMPT_BYTES, MAX_PROMPT_BYTES)
    byte_budget = trace["max_added_source_bytes"]
    if not isinstance(question, str) or not question.strip() or not original:
        trace["skipped_reason"] = "no_grounded_question"
        return result
    anchors = _terms(question[:8192])
    # A local multilingual retriever can supply lexical expansions. These
    # remain search terms and never appear as a new claim or quoted evidence.
    if not isinstance(query_terms, (str, bytes)):
        for term in query_terms[:64]:
            if isinstance(term, str):
                anchors |= _terms(term[:128])
    semantic = {}
    if isinstance(semantic_scores, Mapping):
        eligible = [(identifier, float(score)) for identifier, score in semantic_scores.items()
                    if isinstance(identifier, str) and type(score) in (int, float)
                    and math.isfinite(score) and 0.01 <= score <= 1]
        semantic = dict(sorted(eligible, key=lambda pair: (-pair[1], pair[0]))[:MAX_SEMANTIC_CANDIDATES])
    elif semantic_scores is not None:
        trace["skipped_reason"] = "invalid_semantic_scores"
        return result
    if not anchors and not semantic:
        trace["skipped_reason"] = "no_query_anchors"
        return result
    original_bytes = sum(_size(row.get("content", "")) for row in original)
    if not group_cap or not byte_budget or prompt_cap < 500 or len(original) > source_cap or original_bytes > source_budget:
        trace["skipped_reason"] = "no_evidence_budget"
        return result
    live, collisions = {}, set()
    for row in chunks:
        if not _valid_source(row):
            continue
        if row["id"] in live and (row["content"] != live[row["id"]]["content"] or row["document_id"] != live[row["id"]]["document_id"]):
            collisions.add(row["id"])
        else:
            live[row["id"]] = row
    for key in collisions:
        live.pop(key, None)
    labels = set()
    validated = []
    for source in original:
        raw = live.get(source.get("id"))
        left, right = source.get("excerpt_start", 0), source.get("excerpt_end", len(source.get("content", "")))
        label = source.get("label", "")
        if (raw is None or raw["document_id"] != source.get("document_id")
                or type(left) is not int or type(right) is not int or not 0 <= left < right <= len(raw["content"])
                or raw["content"][left:right] != source.get("content")
                or not isinstance(label, str) or not _LABEL.fullmatch(label) or label in labels):
            trace["skipped_reason"] = "source_binding_invalid"
            return result
        labels.add(label)
        validated.append({**source, "excerpt_start": left, "excerpt_end": right})
    if len({row["id"] for row in validated}) != len(validated):
        trace["skipped_reason"] = "source_binding_invalid"
        return result
    original_by_id = {row["id"]: row for row in validated}
    novelty_sources = dict(original_by_id)
    for source in already_supplied_sources:
        if not isinstance(source, Mapping):
            trace["skipped_reason"] = "source_binding_invalid"
            return result
        raw = live.get(source.get("id"))
        left, right = source.get("excerpt_start", 0), source.get("excerpt_end", len(source.get("content", "")))
        if (raw is None or raw["document_id"] != source.get("document_id")
                or type(left) is not int or type(right) is not int or not 0 <= left < right <= len(raw["content"])
                or raw["content"][left:right] != source.get("content")):
            trace["skipped_reason"] = "source_binding_invalid"
            return result
        prior = novelty_sources.get(source["id"])
        # A baseline uses one contiguous excerpt per chunk. Reject inconsistent
        # overlap instead of pretending the unshown gap was already provided.
        if prior is not None and not (left <= prior["excerpt_start"] <= prior["excerpt_end"] <= right):
            trace["skipped_reason"] = "baseline_binding_inconsistent"
            return result
        novelty_sources[source["id"]] = {**source, "excerpt_start": left, "excerpt_end": right}
    seed_terms = set().union(*(_terms(row["content"]) for row in validated))
    grounded = anchors & seed_terms
    semantic_locations = {key: tuple(semantic_spans[key]) for key in semantic if key in live
                          and isinstance(semantic_spans, Mapping)
                          and _valid_semantic_span(semantic_spans.get(key), live[key]["content"])}
    semantic = {key: score for key, score in semantic.items() if key in semantic_locations}
    semantic_seeds = {key for key in original_by_id.keys() & semantic.keys()
                      if _overlaps(semantic_locations[key],
                                   (original_by_id[key]["excerpt_start"], original_by_id[key]["excerpt_end"]))}
    semantic_anchored = bool(semantic_seeds)
    trace["semantic_candidate_count"] = len(semantic)
    trace["semantic_anchor_count"] = len(semantic_seeds)
    trace["semantic_span_count"] = len(semantic_locations)
    clip_only = not grounded and not semantic_anchored
    if clip_only:
        # Cross-language retrieval can establish the seed without sharing any
        # lexical token. Restore only sentences already partially visible in
        # such seeds; a missing lexical anchor cannot justify a new document.
        clipped = any(not _covered(span, row) and
                      max(span[0], row["excerpt_start"]) < min(span[1], row["excerpt_end"])
                      for row in validated for span in sentence_spans(live[row["id"]]["content"]))
        if not clipped:
            trace["skipped_reason"] = "no_grounded_query_anchor"
            return result
    trace["lexical_anchor_unavailable"] = not bool(grounded)
    seed_documents = {row["document_id"] for row in original}
    # Window selection is deterministic regardless of caller/database ordering.
    ordered = sorted(live.values(), key=lambda row: (row["id"] not in original_by_id, _stable(row)))[:MAX_CHUNKS]
    # Reuse the same source-located title boundary as primary retrieval. A
    # partially overlapping punctuation span must not restore author/email
    # metadata that the primary selector deliberately excluded.
    title_headers = {(hint['document_id'], hint['source_id']): hint['abstract_body_start']
                     for hint in initial_title_hints(ordered)}
    trace["scanned_chunk_count"] = len(ordered)
    trace["scan_truncated"] = len(live) > len(ordered) or any(len(row["content"]) > MAX_SCAN_CHARS for row in ordered)
    doc_freq = Counter()
    chunk_terms = {}
    for raw in ordered:
        terms = _terms(raw["content"][:MAX_SCAN_CHARS])
        chunk_terms[raw["id"]] = terms
        doc_freq.update(terms & anchors)
    weights = {term: math.log1p((len(ordered) + 1) / (1 + doc_freq[term])) for term in anchors}
    original_sentence_tokens = []
    covered_roles = set()
    for source in novelty_sources.values():
        raw = live[source["id"]]
        for left, right in sentence_spans(raw["content"]):
            if _covered((left, right), source):
                text = raw["content"][left:right]
                original_sentence_tokens.append(_terms(text))
                if anchors & _terms(text) or source["id"] in semantic_seeds:
                    covered_roles |= _roles(text)
    desired_roles = {role for role, pattern in _ROLE_PATTERNS.items() if pattern.search(question)}
    if not desired_roles:
        desired_roles = set(_ROLE_PATTERNS)
    candidates = []
    skipped_bibliography = 0
    for raw in ordered:
        semantic_score = semantic.get(raw["id"], 0.0) if semantic_anchored else 0.0
        lexical_candidate = bool(chunk_terms[raw["id"]] & grounded)
        if not clip_only and not lexical_candidate and not semantic_score and raw["id"] not in original_by_id:
            continue
        if clip_only and raw["id"] not in original_by_id:
            continue
        # Repeated publication metadata is a bibliography, not this paper's
        # method/results. A cited title alone cannot establish its findings.
        if (not _REFERENCE_REQUEST.search(question)
                and len(_BIBLIOGRAPHIC.findall(raw["content"][:MAX_SCAN_CHARS])) >= 3):
            skipped_bibliography += 1
            continue
        header_end = title_headers.get((raw['document_id'], raw['id']), 0)
        spans = [(left + header_end, right + header_end)
                 for left, right in sentence_spans(raw['content'][header_end:MAX_SCAN_CHARS])]
        units = [(left + header_end, right + header_end)
                 for left, right in evidence_units(raw['content'][header_end:MAX_SCAN_CHARS])]
        original_source = novelty_sources.get(raw["id"])
        for index, (left, right) in enumerate(spans):
            center = raw["content"][left:right]
            terms = _terms(center)
            hit = terms & anchors
            # Shared generic discourse verbs cannot bring in an unrelated paper.
            # A short one-concept question may have one meaningful query anchor.
            minimum = 1 if len(anchors) <= 2 else 2
            partial = original_source is not None and not _covered((left, right), original_source) and (
                max(left, original_source["excerpt_start"]) < min(right, original_source["excerpt_end"]))
            lexical_sentence = len(hit) >= minimum and bool(hit & grounded)
            # Semantic similarity qualifies only the backend's actual winning
            # window, not arbitrary material elsewhere in a large chunk.
            semantic_sentence = bool(semantic_score
                                     and _overlaps((left, right), semantic_locations[raw["id"]])
                                     and (_roles(center) or len(terms & seed_terms) >= 2))
            if clip_only:
                if not partial:
                    continue
            elif not partial and not lexical_sentence and not semantic_sentence:
                continue
            selected_indices = {index}
            # Preserve the previous sentence for explicit anaphora and the next
            # sentence when it qualifies this one. Adjacency alone is not proof.
            allow_context = lexical_sentence or semantic_sentence
            if allow_context and index and _ANAPHORA.search(center):
                prior = spans[index - 1]
                gap = raw["content"][prior[1]:left]
                if "\n\n" not in gap and not section_heading_spans(gap):
                    selected_indices.add(index - 1)
            if allow_context and index + 1 < len(spans):
                following = spans[index + 1]
                next_text = raw["content"][following[0]:following[1]]
                next_roles = _roles(next_text)
                gap = raw["content"][right:following[0]]
                if ("\n\n" not in gap and not section_heading_spans(gap)
                        and (next_roles & {"condition", "limitation"})
                        and (_ANAPHORA.search(next_text) or bool(_terms(next_text) & terms & anchors))):
                    selected_indices.add(index + 1)
            selected_indices = _context_indices(raw["content"], spans, units, selected_indices)
            if selected_indices is None:
                continue
            selected = [spans[i] for i in sorted(selected_indices)]
            left, right = selected[0][0], selected[-1][1]
            if right - left > MAX_WINDOW_CHARS or len(selected) > MAX_WINDOW_SENTENCES:
                continue
            unseen = [span for span in selected if not _covered(span, original_source)]
            if not unseen:
                continue
            unseen_text = " ".join(raw["content"][start:end] for start, end in unseen)
            unseen_terms = _terms(unseen_text)
            if not unseen_terms:
                continue
            duplicate = max((len(unseen_terms & old) / max(1, len(unseen_terms | old)) for old in original_sentence_tokens), default=0.0)
            if duplicate >= 0.88:
                continue
            content = raw["content"][left:right]
            roles = _roles(unseen_text)
            completed = sum(original_source is not None and
                            max(start, original_source["excerpt_start"]) < min(end, original_source["excerpt_end"])
                            for start, end in unseen)
            # New facts, a missing rhetorical aspect, or a different document
            # have value; near-copies with a new chunk identifier do not.
            query_gain = sum(weights[term] for term in unseen_terms & anchors)
            score = query_gain + 1.4 * len((roles & desired_roles) - covered_roles) + 1.6 * completed
            if semantic_sentence:
                # Ranking gain only. Do not turn model similarity into a
                # confidence score on the quotation's correctness.
                score += 2 * semantic_score + 0.2 * min(5, len(unseen_terms & seed_terms))
            if raw["document_id"] not in seed_documents:
                score += 1.0
            score -= duplicate * 2.0
            candidates.append({"raw": raw, "left": left, "right": right, "spans": selected,
                               "terms": _terms(content), "unseen_terms": unseen_terms, "roles": roles,
                               "score": score, "completed": completed, "unseen": unseen,
                               "semantic_selected": semantic_sentence and not lexical_sentence})
    candidates.sort(key=lambda row: (-row["score"], _stable(row["raw"]), row["left"], row["right"]))
    candidates = candidates[:MAX_CANDIDATES]
    trace["candidate_count"] = len(candidates)
    trace["bibliographic_chunks_omitted"] = skipped_bibliography
    sources, groups, claims, chosen_terms = validated, [], [], []
    used_windows, added_roles, added_docs = set(), set(), set()
    while candidates and len(groups) < group_cap:
        def marginal(row):
            novelty = 1 - max((len(row["terms"] & old) / max(1, len(row["terms"] | old)) for old in chosen_terms), default=0.0)
            diversity = 0.7 if row["raw"]["document_id"] not in added_docs else 0.0
            role_gain = 0.5 * len((row["roles"] & desired_roles) - covered_roles - added_roles)
            return row["score"] * novelty + diversity + role_gain
        candidate = max(candidates, key=marginal)
        candidates.remove(candidate)
        key = (candidate["raw"]["id"], candidate["left"], candidate["right"])
        if key in used_windows:
            continue
        current = next((row for row in sources if row["id"] == key[0]), None)
        baseline = novelty_sources.get(key[0])
        newly_visible = [span for span in candidate["spans"] if not _covered(span, current) and not _covered(span, baseline)]
        if not newly_visible:
            continue
        if max((len(candidate["terms"] & old) / max(1, len(candidate["terms"] | old)) for old in chosen_terms), default=0.0) >= 0.88:
            continue
        proposed = _merge(sources, candidate["raw"], candidate["left"], candidate["right"], source_cap,
                          min(source_budget, original_bytes + byte_budget))
        if proposed is None:
            continue
        new_claims = []
        for left, right in candidate["spans"]:
            identifier = "sentence-" + hashlib.sha256(f"{key[0]}:{left}:{right}".encode()).hexdigest()[:20]
            new_claims.append({"id": identifier, "kind": "source_excerpt", "epistemic_status": "observed_quote",
                               "source_ids": [key[0]], "document_id": candidate["raw"]["document_id"],
                               "quote": candidate["raw"]["content"][left:right], "excerpt_start": left, "excerpt_end": right,
                               "role_hints": sorted(_roles(candidate["raw"]["content"][left:right])) or ["background"]})
        group = {"id": "evidence-" + hashlib.sha256(repr(key).encode()).hexdigest()[:20],
                 "kind": "source_evidence", "epistemic_status": "source_organization",
                 "method": "sentence_evidence_organization", "source_ids": [key[0]],
                 "premise_ids": [row["id"] for row in new_claims],
                 "text": "补充原文中的相关句及相邻条件；角色和关联仅供组织材料。",
                 "connection": "adjacent_source_sentences" if len(new_claims) > 1 else
                     "semantic_retrieval_candidate" if candidate["semantic_selected"] else "shared_question_terms",
                 "role_hints": sorted(candidate["roles"]), "new_sentence_count": len(newly_visible),
                 "completed_sentence_count": sum(baseline is not None and
                     max(left, baseline["excerpt_start"]) < min(right, baseline["excerpt_end"])
                     for left, right in newly_visible), "entailment_verified": False}
        proposed_claims = list({row["id"]: row for row in [*claims, *new_claims]}.values())
        prompt = _prompt([*groups, group], proposed_claims, proposed)
        if _size(prompt) > prompt_cap:
            continue
        sources, claims = proposed, proposed_claims
        groups.append(group)
        used_windows.add(key)
        chosen_terms.append(candidate["terms"])
        added_roles |= candidate["roles"]
        added_docs.add(candidate["raw"]["document_id"])
    trace.update(support_item_count=len(groups), added_source_bytes=sum(_size(row["content"]) for row in sources) - original_bytes,
                 added_sentence_count=sum(group["new_sentence_count"] for group in groups),
                 completed_sentence_count=sum(group["completed_sentence_count"] for group in groups),
                 skipped_reason=None if groups else "no_new_grounded_sentence_within_budget")
    result.update(sources=sources if groups else original, supporting_material=groups, claims=claims,
                  prompt=_prompt(groups, claims, sources))
    return result
