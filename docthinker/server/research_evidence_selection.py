"""Window-level evidence coverage under hard source budgets; no provider calls.

The candidate pool is live-source only. Scores express retrieval utility, never
truth, probability, causal validity, or entailment. Selection may replace any
low-value baseline window instead of preserving irrelevant original context.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

from .research_retrieval import _source_row, lexical_tokens
from .research_sentence_windows import evidence_units, is_section_heading, is_figure_caption_start, bounded_evidence_extension, sentence_spans
from .research_figure_axes import pure_axis_ticks, requests_graph_data
from .research_document_titles import subplot_axis_labels, initial_title_hints
from .research_selection_diagnostics import emit_diagnostic, window_identity
from .research_query_facets import valid_quoted_audit_view, _document_mentions, is_role_only_phrase
from .research_component_coverage import component_nomination_queue

MAX_CANDIDATES = 48
MAX_WINDOW_CHARS = 1_600
MAX_SCAN_ROWS = 4_096
MAX_SCAN_CHARS = 8_000
MAX_UNITS = 96
MAX_SOURCES = 8


def literal_query_surface(text, *, positive_evidence_request=False):
    """Only a matching view: explicit non-coverage is not topic coverage.

    The original quote, negated findings and operation constraints are never
    altered. No paraphrase, implied positive claim or new evidence is created.
    """
    if not positive_evidence_request:
        return text
    noncoverage = re.compile(r'\b(?:does|do|did)\s+not\s+(?:describe|explain|address|cover)\b|'
                             r'(?:本文|本节|本段|该步骤)(?:并)?不(?:描述|解释|涉及|涵盖)', re.I)
    return ' '.join(text[left:right] for left, right in sentence_spans(text)
                    if not noncoverage.search(text[left:right])
                    or any(mark in text[left:right] for mark in ('`', '"', '“', '”')))
MAX_SOURCE_BYTES = 12_000
# Give each of the at most eight original sources one chance to compete under
# the full question. A two-source reserve silently removed this chance for late
# baseline evidence when the remaining pair budget went to focused questions.
# This reserves scoring eligibility only, never a final answer-context slot.
MAX_BASELINE_NOMINATIONS = MAX_SOURCES
DEPTH_DISCOUNTS = (1., .4)
COARSE_DEPTH_DISCOUNTS = (1., .4, .2)
_ROLE = {
    "experiment_setup": r"\b(?:experiment\w*|evaluat\w*|benchmark\w*|dataset\w*|ablation\w*|sampled?|participants?|tasks?)\b|实验|评测|数据集|测试|消融",
    "mechanism": r"\b(?:method\w*|algorithm\w*|implement\w*|procedure\w*|retriev\w*|process\w*|construct\w*|comput\w*|architectur\w*|approach\w*)\b|方法|算法|机制|实现|检索|步骤|架构",
    "result": r"\b(?:results?|findings?|accuracy|outperform\w*|improv\w*|decreas\w*|increas\w*|significant\w*)\b|结果|结论|准确率|提升|降低",
    "limitation": r"\b(?:limit\w*|fail\w*|however|although|unless|only|not|cannot|assum\w*|caveat\w*|counterexample\w*)\b|局限|限制|失败|但是|然而|仅|不|条件|前提",
    "comparison": r"\b(?:compar\w*|contrast\w*|differ\w*|versus|baseline\w*)\b|对比|比较|区别|不同",
}
_BACKGROUND = re.compile(r"\b(?:introduction|background|related works?|prior works?|literature review)\b|引言|背景|相关工作", re.I)
_NUMBERED_OUTLINE = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?=[A-Za-z\u3400-\u9fff])')
_REFERENCES = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|acknowledg\w*|参考文献|致谢)\b", re.I)
_REFERENCE_END = re.compile(
    r"^(?:appendi(?:x|ces)\b|附录|[A-Z](?:\.\d{1,2})+\s+|\d{1,2}(?:\.\d{1,2})*[.)]?\s+\S)", re.I,
)
_REFERENCE_REQUEST = re.compile(
    r"\b(?:bibliograph\w*|reference lists?|list (?:the )?references|list of references|"
    r"acknowledg\w*)\b|参考文献(?:列表|清单|章节|部分)|(?:列出|列举|罗列|展示|查看|显示)参考文献|引用列表|致谢", re.I,
)
_TABLE_REQUEST = re.compile(r"\b(?:table\w*|numerical|numbers?|metrics?|accuracy|scores?)\b|表格|表中|数值|数字|指标|准确率|得分", re.I)
_CONTACT_REQUEST = re.compile(
    r"\b(?:e-?mails?|contact (?:details?|information|address)|correspondence address)\b|"
    r"邮箱|电子邮件|联系方式|联系地址|作者(?:是谁|姓名|名字)|\bwho (?:is|are) (?:the )?authors?\b", re.I,
)
_EMAIL_ADDRESS = re.compile(
    r"(?<![\w@])[\w.+|%{}-]+@(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}\b", re.I,
)
_COMMON_TERMS = frozenset("method methods experiment experiments result results compare comparison explain describe evidence".split())
_DETAIL_CUE = re.compile(
    r"\b(?:comput\w*|select\w*|execut\w*|schedul\w*|retriev\w*|updat\w*|transfer\w*|"
    r"rotat\w*|allocat\w*|append\w*|partition\w*|construct\w*|sampl\w*|measur\w*|"
    r"compar\w*|evaluat\w*|train\w*|inject\w*|replace\w*|call\w*|iterat\w*|"
    r"trigger\w*|return\w*|transform\w*|hash\w*|condition\w*|unless|only|cannot)\b|"
    r"计算|选择|执行|调度|检索|更新|传递|分配|划分|构造|采样|测量|比较|评估|训练|调用|迭代|触发|返回|条件|仅当|不能", re.I,
)


def _finite(value: Any) -> float | None:
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None


def _roles(text: str) -> list[str]:
    return [key for key, pattern in _ROLE.items() if re.search(pattern, text, re.I)]


def _advance_outline(stack, heading):
    """At most four source-bound numeric levels, not inferred section content.

    Called only after the existing positive heading parser accepts a unit.
    A child's explicit role overrides its parent, while a sibling/new root
    clears obsolete descendants. Captions and soft wraps never create levels.
    """
    match = _NUMBERED_OUTLINE.match(heading)
    if match and not is_figure_caption_start(heading):
        path = tuple(match[1].split('.'))
        stack[:] = [(prior, roles) for prior, roles in stack
                    if len(prior) < len(path) and path[:len(prior)] == prior]
        stack.append((path, _roles(heading)))
    elif (not is_figure_caption_start(heading) and (_roles(heading) or _BACKGROUND.search(heading)
          or _REFERENCES.search(heading) or re.match(r'^appendi(?:x|ces)\b|^附录', heading, re.I))):
        stack.clear()
        return _roles(heading)
    return next((roles for _, roles in reversed(stack) if roles), [])


def _duplicate_window(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Conservative repetition proxy, never a test of factual equivalence."""
    left_text = left.get("_normalized_content") or " ".join(left["content"].casefold().split())
    right_text = right.get("_normalized_content") or " ".join(right["content"].casefold().split())
    if left_text == right_text:
        return True
    if (left["document_id"], left["source_id"]) == (right["document_id"], right["source_id"]):
        overlap = max(0, min(left["excerpt_end"], right["excerpt_end"])
                      - max(left["excerpt_start"], right["excerpt_start"]))
        shorter = min(left["excerpt_end"] - left["excerpt_start"], right["excerpt_end"] - right["excerpt_start"])
        if overlap / max(1, shorter) >= .5:
            return True
    union = left["_tokens"] | right["_tokens"]
    return bool(union and len(left["_tokens"] & right["_tokens"]) / len(union) >= .9)


def _depth_value(rows: Sequence[Mapping[str, Any]], dimension: str, discounts: Sequence[float]) -> float:
    # Deduplicate within a requirement, not globally: independently named
    # documents still need their own evidence even when their wording agrees.
    ranked = sorted((row for row in rows if row["_coverage"].get(dimension, 0.) > 0),
                    key=lambda row: (-row["_coverage"][dimension], row["id"]))
    # A high-scoring near-duplicate may overlap two mutually distinct windows.
    # Greedy suppression would erase both and *decrease* the set objective when
    # that candidate is added. Maximize over pairwise-distinct subsets instead;
    # depth is at most three and input is bounded by the 48 scored pairs.
    scores = [row["_coverage"][dimension] for row in ranked]
    pairs: dict[tuple[int, int], bool] = {}

    def duplicate(left, right):
        key = (left, right)
        if key not in pairs:
            pairs[key] = _duplicate_window(ranked[left], ranked[right])
        return pairs[key]

    best = 0.
    for first, score in enumerate(scores):
        upper = sum(discount * value for discount, value in zip(discounts, scores[first:]))
        if upper <= best:
            break
        first_value = discounts[0] * score
        best = max(best, first_value)
        for second in range(first + 1, len(scores)):
            pair_value = first_value + discounts[1] * scores[second]
            upper = pair_value + (discounts[2] * scores[second + 1]
                                  if len(discounts) == 3 and second + 1 < len(scores) else 0.)
            if upper <= best:
                break
            if duplicate(first, second):
                continue
            best = max(best, pair_value)
            if len(discounts) == 3:
                for third in range(second + 1, len(scores)):
                    value = pair_value + discounts[2] * scores[third]
                    if value <= best:
                        break
                    if not duplicate(first, third) and not duplicate(second, third):
                        best = value
                        break  # Later third scores cannot improve this pair.
    return best


def _source_round_robin(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep the ranked best window of each source before its second window."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["document_id"], row["source_id"]), []).append(row)
    result = []
    for position in range(max((len(group) for group in grouped.values()), default=0)):
        result.extend(group[position] for group in grouped.values() if position < len(group))
    return result


def _section_round_robin(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Expose distinct explicit sections before repeated windows of one section.

    This changes candidate opportunity only. Different sections of the same
    source may compete, but the final packet still permits just one continuous
    window per source. Unknown section strings do not create nominations.
    """
    grouped = {}
    for row in rows:
        section = row.get('section', '')
        if section and is_section_heading(section):
            grouped.setdefault((row['document_id'], section), []).append(row)
    return [group[position] for position in range(max((len(g) for g in grouped.values()), default=0))
            for group in grouped.values() if position < len(group)]


def _facets(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("facets", ()) if isinstance(plan, Mapping) else ()
    return [dict(row) for row in rows[:8] if isinstance(row, Mapping)
            and isinstance(row.get("id"), str) and isinstance(row.get("text"), str)]


def _anchors(facets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Independent user-explicit technical needs, excluding document titles."""
    titles = {row["text"].casefold().strip() for row in facets if row.get("kind") == "document"}
    found: dict[str, dict[str, Any]] = {}
    for facet in facets:
        if facet.get("kind") in {"document", "query_view"}:
            continue
        for text in facet.get("anchors", ()):
            if not isinstance(text, str) or not 3 <= len(text) <= 160 or text.casefold().strip() in titles:
                continue
            key = " ".join(text.casefold().split())
            if key not in found and len(found) >= 32:
                continue
            item = found.setdefault(key, {"id": "ea_" + hashlib.sha256(key.encode()).hexdigest()[:12], "text": text, "document_ids": set()})
            if facet.get("kind") != "original":
                item["document_ids"].update(doc for doc in facet.get("document_ids", ()) if isinstance(doc, str))
    return list(found.values())


def _anchor_score(content: str, phrase: str, topic_phrases: Sequence[str] = ()) -> float:
    expression = re.escape(phrase).replace(r"\ ", r"\s+")
    matches = list(re.finditer(expression, content, re.I))
    if not matches:
        return 0.
    # Listing the requested concepts is less useful than a source passage that
    # describes operations/conditions around them. This is still only a proxy.
    # A term such as "scheduling" or "retrieval" inside the anchor is a topic
    # name, not evidence of an operation. Inspect only its surrounding prose.
    scores = []
    for hit in matches:
        surrounding = content[max(0, hit.start() - 180):hit.start()] + " " + content[hit.end():hit.end() + 240]
        for topic in topic_phrases:
            surrounding = re.sub(re.escape(topic).replace(r"\ ", r"\s+"), " ", surrounding, flags=re.I)
        scores.append(.90 if _DETAIL_CUE.search(surrounding) else .30)
    return max(scores)


def _span(value: Any, length: int) -> tuple[int, int] | None:
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(number) is int for number in value)
            and 0 <= value[0] < value[1] <= length):
        return value[0], value[1]
    return None


def _bibliographic(text: str) -> bool:
    value = text.strip()
    return bool(re.match(r"^\[\d{1,4}\]\s+[A-Z]", value)
                or (re.match(r"^[A-Z][A-Za-z'-]+,\s+(?:[A-Z]\.\s*){1,3}", value)
                    and re.search(r"\b(?:19|20)\d\d\b|doi:|arxiv|proceedings|journal", value, re.I)))


def _contact_metadata_only(text: str) -> bool:
    """Recognize complete contact-only units, never remove text around an email.

    A method paragraph that merely contains an address stays intact. Names
    require an explicit contact label or a wrapped address, not just Title Case
    words that might describe a method. Ambiguous lines are left to retrieval.
    """
    if not _EMAIL_ADDRESS.search(text):
        return False
    value = _EMAIL_ADDRESS.sub("EMAIL_ADDRESS", text.strip())
    without_addresses = value.replace("EMAIL_ADDRESS", " ")
    labels = re.compile(r"\b(?:e-?mails?|mailto|contact|correspondence|corresponding author|authors?)\b|"
                        r"电子邮箱|电子邮件|邮箱|联系方式|通讯作者", re.I)
    residue = labels.sub(" ", without_addresses)
    if not re.sub(r"[\s:;,，；：<>()\[\]{}|/—–\-.'\"]", "", residue):
        return True
    if len(text) > 160:
        return False
    has_label = bool(labels.search(value))
    has_wrapper = bool(re.search(r"[<(]\s*EMAIL_ADDRESS\s*[>)]", value))
    if not has_label and not has_wrapper:
        return False
    name = re.sub(r"[\s:;,，；：<>()\[\]{}|/—–\-]+", " ", residue).strip().rstrip(".").strip()
    return bool(re.fullmatch(r"(?:[A-Z][a-z]+|[A-Z]\.)(?:\s+(?:[A-Z][a-z]+|[A-Z]\.)){1,4}", name))


def _table_fragment(text: str) -> bool:
    nonspace = re.sub(r"\s", "", text)
    digits = sum(char.isdigit() for char in nonspace)
    words = re.findall(r"[A-Za-z]{3,}", text)
    return bool(nonspace and digits / len(nonspace) > .24 and len(words) < 20)


def _window_row(row: Mapping[str, Any], start: int, end: int, roles: Sequence[str], section: str) -> dict[str, Any]:
    content = row["content"]
    identity = f"{row['document_id']}\0{row['id']}\0{start}:{end}\0{content[start:end]}"
    result = dict(row)
    result.update({
        "id": "ew_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "source_id": row["id"], "content": content[start:end],
        "excerpt_start": start, "excerpt_end": end,
        "original_content_length": len(content), "offset_unit": "characters_in_source_chunk",
        "truncated": start > 0 or end < len(content),
        "evidence_scope": "excerpt_only" if start > 0 or end < len(content) else "source_chunk",
        "sentence_boundary_preserved": True, "roles": sorted(set(roles)), "section": section,
        "_source_content": content,
    })
    return result


def prepare_evidence_candidates(
    question: str, chunks: Sequence[Mapping[str, Any]], baseline_sources: Sequence[Mapping[str, Any]],
    facet_plan: Mapping[str, Any], semantic_results: Mapping[str, Any], max_candidates: int = MAX_CANDIDATES,
    *, diagnostic_callback=None, matching_tokens=None, nomination_prior=None, component_profile=None,
    literal_requested_roles=None, evaluation_profile=None, operation_profile=None,
) -> dict[str, Any]:
    """Union per-facet and named-document windows before the reranker cap.

    Multiple query facets may nominate different windows of a long source.
    Heading priors are generic and carried only within each live document.
    """
    facets = _facets(facet_plan)
    # The literal fallback may normalize word inflections for nomination. This
    # only changes retrieval utility; source strings and offsets stay untouched.
    tokenize = matching_tokens or lexical_tokens
    anchors = _anchors(facets)
    role_only_facets = {facet['id'] for facet in facets if matching_tokens
                        and facet.get('kind') == 'technical_phrase'
                        and is_role_only_phrase(facet['text'], question)}
    if matching_tokens:
        anchors = [anchor for anchor in anchors if not is_role_only_phrase(anchor['text'], question)]
    cap = min(MAX_CANDIDATES, max(0, max_candidates)) if type(max_candidates) is int else MAX_CANDIDATES
    reference_request = bool(_REFERENCE_REQUEST.search(question))
    table_request = bool(_TABLE_REQUEST.search(question))
    contact_request = bool(_CONTACT_REQUEST.search(question))
    graph_data_request = requests_graph_data(question)
    document_terms = {term for facet in facets if facet.get("kind") == "document" for term in tokenize(facet["text"])}
    if contact_request:
        document_terms.clear()  # Names remain useful in explicit metadata lookups.
    terms = {facet["id"]: set(tokenize(facet["text"])) - document_terms for facet in facets}
    named_docs = set(sorted({doc for facet in facets for doc in facet.get("document_ids", ()) if isinstance(doc, str)})[:MAX_SOURCES])
    named_mentions = [(re.compile(r'(?<![\w])' + re.escape(facet['text']).replace(r'\ ', r'\s+')
                                  + r'(?![\w])', re.I), set(facet.get('document_ids', ())))
                      for facet in facets if facet.get('kind') == 'document']
    baseline = {str(row.get("id")): row for row in baseline_sources[:MAX_SOURCES] if isinstance(row, Mapping)}
    nomination_docs = named_docs | {row.get('document_id') for row in baseline.values()}
    rows = [row for raw in islice(chunks, MAX_SCAN_ROWS)
            if isinstance(raw, Mapping) and raw.get("status") not in ("deleted", "revoked", "invalidated")
            and (row := _source_row(raw)) is not None]
    if matching_tokens and not contact_request:
        # A full clause budget may omit the separate document facet. The
        # remaining metadata binding must still not reward self-name repeats.
        document_terms.update(term for mention in _document_mentions(question, rows)
                              for term in tokenize(mention['text']))
        terms = {facet['id']: set(tokenize(facet['text'])) - document_terms for facet in facets}
    rows.sort(key=lambda row: (row["document_id"], row["ordinal"], row["id"]))
    title_headers = {(hint['document_id'], hint['source_id']): hint['abstract_body_start']
                     for hint in initial_title_hints(rows)}
    seen = set()
    candidates = []
    sections: dict[str, str] = {}
    outlines: dict[str, list] = {}
    filtered = 0
    contact_filtered = 0
    axis_filtered = 0
    oversized = 0
    window_count = 0
    window_ids = set()
    broad_comparison = len(named_docs) > 1 and any("comparison" in facet.get("roles", ()) for facet in facets)
    # Keep existing conservative document scope whenever a condition/negation
    # binds the query; do not reinterpret "only these sources" as open scope.
    open_original_scope = not facet_plan.get('trace', {}).get('protected_clause_count', 0)
    literal_roles = set(literal_requested_roles) if literal_requested_roles is not None else None
    broad_method_comparison = broad_comparison and (literal_roles is None or 'mechanism' in literal_roles)

    def prune_pool(values):
        """Bound internal memory, not merely the final reranker output."""
        order = lambda item: (-item["pool_score"], len(item["content"]), item["id"])
        keep = {item["id"]: item for item in _source_round_robin(sorted(values, key=order))[:64]}
        for facet in facets:
            ranked = sorted(values, key=lambda item: (-item["facet_scores"].get(facet["id"], 0.), *order(item)))
            keep.update((item["id"], item) for item in _source_round_robin(ranked)[:8] if item["facet_scores"].get(facet["id"], 0) > 0)
        for doc in named_docs:
            ranked = sorted((item for item in values if item["document_id"] == doc), key=order)
            keep.update((item["id"], item) for item in _source_round_robin(ranked)[:3])
            for role in ("mechanism", "experiment_setup", "limitation"):
                keep.update((item["id"], item) for item in _source_round_robin(
                    [item for item in ranked if role in item["roles"]])[:2])
        for anchor in anchors:
            ranked = sorted((item for item in values if item.get("anchor_scores", {}).get(anchor["id"], 0) > 0),
                            key=lambda item: (-item["anchor_scores"][anchor["id"]], *order(item)))
            keep.update((item["id"], item) for item in _source_round_robin(ranked)[:2])
        # Preserve one already-eligible window for each bounded baseline ID
        # until the bounded baseline reserve is decided. Never restore raw text.
        for identifier in baseline:
            ranked = sorted((item for item in values if item["source_id"] == identifier), key=order)
            if ranked:
                keep[ranked[0]["id"]] = ranked[0]
        if component_profile:
            for doc in nomination_docs:
                keep.update((item['id'], item) for item in component_nomination_queue(
                    [item for item in values if item['document_id'] == doc]))
        if evaluation_profile:
            for doc in nomination_docs:
                for cell in ('setup', 'measurement', 'outcome'):
                    ranked = sorted((item for item in values if item['document_id'] == doc
                                     and cell in item.get('_evaluation_cells', ())), key=order)
                    keep.update((item['id'], item) for item in _source_round_robin(ranked)[:2])
        if operation_profile:
            for doc in nomination_docs:
                for stage in ('write', 'change', 'read'):
                    ranked = sorted((item for item in values if item['document_id'] == doc
                                     and stage in item.get('_operation_stages', ())),
                                    key=lambda item: (-item.get('_operation_quality', {}).get(stage, 0.), *order(item)))
                    keep.update((item['id'], item) for item in _source_round_robin(ranked)[:2])
        emit_diagnostic(diagnostic_callback, "pool_pruned", before_ids=[row["id"] for row in values],
                        kept_ids=list(keep), reason="internal_window_limit")
        return list(keep.values())
    for row in rows:
        identity = row["document_id"], row["id"]
        if identity in seen:
            continue
        seen.add(identity)
        source = row["content"]
        if len(source) > MAX_SCAN_CHARS:
            # Reject oversized legacy chunks rather than silently cut a qualifier.
            oversized += 1
            emit_diagnostic(diagnostic_callback, "source_filtered", source_id=row["id"],
                            document_id=row["document_id"], reason="oversized_chunk")
            continue
        section = sections.get(row["document_id"], "")
        outline = outlines.setdefault(row['document_id'], [])
        contact_header_end = title_headers.get(identity, 0) if contact_request else 0
        header_end = 0 if contact_request else title_headers.get(identity, 0)
        if header_end:
            section = 'Abstract'
            outline.clear()
        outline_roles = next((roles for _, roles in reversed(outline) if roles), _roles(section))
        units = []
        axis_exclusions = []
        inline_references = False
        source_units = [(start + header_end, end + header_end)
                        for start, end in evidence_units(source[header_end:])[:MAX_UNITS]]
        for unit_index, (start, end) in enumerate(source_units):
            text = source[start:end].strip()
            if is_section_heading(text) and not (contact_request and start < contact_header_end):
                # Author names and publication titles can resemble Title Case
                # headings. Once in a reference section, only an explicit new
                # numbered section or appendix can restore answer evidence.
                if (_REFERENCES.search(section) and not reference_request
                        and not _REFERENCE_END.search(text)):
                    filtered += 1
                    emit_diagnostic(diagnostic_callback, "unit_filtered", source_id=row["id"],
                                    document_id=row["document_id"], start=start, end=end,
                                    section=section, reason="reference_section_heading")
                    continue
                section = text[:120]
                outline_roles = _advance_outline(outline, text)
                emit_diagnostic(diagnostic_callback, "unit_filtered", source_id=row["id"],
                                document_id=row["document_id"], start=start, end=end,
                                section=section, reason="section_heading")
                continue
            if not text:
                continue
            next_unit = source_units[unit_index + 1] if unit_index + 1 < len(source_units) else None
            before_caption = bool(next_unit and not source[end:next_unit[0]].strip()
                                  and is_figure_caption_start(source[next_unit[0]:next_unit[1]].strip()))
            if (not graph_data_request and not re.match(r"^\s*Table\b", section, re.I)
                    and (pure_axis_ticks(text, immediately_before_caption=before_caption)
                         or subplot_axis_labels(text, immediately_before_caption=before_caption))):
                axis_filtered += 1
                axis_exclusions.append((start, end))
                emit_diagnostic(diagnostic_callback, "unit_filtered", source_id=row["id"],
                                document_id=row["document_id"], start=start, end=end,
                                section=section, reason="pure_axis_ticks_before_caption")
                continue
            if not contact_request and _contact_metadata_only(text):
                filtered += 1
                contact_filtered += 1
                emit_diagnostic(diagnostic_callback, "unit_filtered", source_id=row["id"],
                                document_id=row["document_id"], start=start, end=end,
                                section=section, reason="contact_only_metadata")
                continue
            isolated_label = (re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,40}", text)
                              or re.fullmatch(r"[A-Z][A-Za-z0-9_-]{0,40}(?:\s+[A-Z][A-Za-z0-9_-]{0,40}){1,3}", text))
            if not table_request and isolated_label:
                filtered += 1
                emit_diagnostic(diagnostic_callback, "unit_filtered", source_id=row["id"],
                                document_id=row["document_id"], start=start, end=end,
                                reason="isolated_latin_label")
                continue
            if _REFERENCES.search(text) and not reference_request:
                inline_references = True
            excluded = (inline_references or bool(_REFERENCES.search(section)) or _bibliographic(text)) and not reference_request
            if excluded:
                filtered += 1
                emit_diagnostic(diagnostic_callback, "unit_filtered", source_id=row["id"],
                                document_id=row["document_id"], start=start, end=end,
                                section=section, reason="bibliographic")
                continue
            units.append((start, end, section, tuple(outline_roles), outline[-1][0] if outline else ()))
        sections[row["document_id"]] = section
        source_window_spans = set()
        for index, (start, _, section, outline_roles, outline_path) in enumerate(units):
            # Nominate one maximal bounded context window per start. Offering
            # 1/2/3-unit variants lets a cheap definition saturate coverage and
            # permanently crowd out its following mechanism/condition. These
            # are coherent-context heuristics, not verified factual support.
            for size in (3, 2, 1):
                if index + size > len(units):
                    continue
                members = units[index:index + size]
                end = members[-1][1]
                if any(member[2] != section for member in members):
                    continue
                # Excluded references/headings must not be bridged in a quote.
                if any(source[members[i][1]:members[i + 1][0]].strip() for i in range(size - 1)):
                    continue
                if end - start > MAX_WINDOW_CHARS:
                    continue
                # A hit is a locator, not a complete explanation. Extend both
                # directions inside the same contiguous section, without
                # crossing skipped units/headings. This is original context,
                # not an invented stitched quote or a support certificate.
                lo, hi = index, index + size
                while end - start < 1000:
                    options = []
                    if lo > 0 and units[lo-1][2] == section and not source[units[lo-1][1]:start].strip():
                        options.append((lo-1, hi, units[lo-1][0], end))
                    if hi < len(units) and units[hi][2] == section and not source[end:units[hi][0]].strip():
                        options.append((lo, hi+1, start, units[hi][1]))
                    options = [value for value in options if value[3]-value[2] <= MAX_WINDOW_CHARS]
                    if not options:
                        break
                    lo, hi, start, end = min(options, key=lambda value: (value[3]-value[2], value[2]))
                if (start, end) in source_window_spans:
                    break  # Do not repeatedly tokenize/score the same span.
                source_window_spans.add((start, end))
                content = source[start:end]
                match_content = literal_query_surface(content, positive_evidence_request=bool(
                    literal_roles and literal_roles & {'mechanism', 'experiment_setup', 'result'})) if matching_tokens else content
                tokens = set(tokenize(match_content))
                roles = sorted(set(_roles(content)) | set(_roles(section)) | set(outline_roles))
                prior = (_finite(nomination_prior({'content': content, 'section': section,
                                                  'outline_roles': outline_roles})) or 0.) if nomination_prior else 0.
                prior = max(0., min(1., prior))
                components = component_profile({**row, 'content': content, 'section': section}) if component_profile else []
                evaluation = evaluation_profile({'content': content, 'section': section,
                                                  'outline_roles': outline_roles}) if evaluation_profile else []
                operations = operation_profile({**row, 'content': content, 'section': section,
                                                 'outline_roles': outline_roles}) if operation_profile else ()
                fragment = _table_fragment(content)
                penalty = .45 if fragment and not table_request else 1.0
                if _BACKGROUND.search(section):
                    penalty *= .70
                facet_scores = {}
                lexical_scores = {}
                semantic_hits = []
                section_role_nomination = False
                cross_document_mentions = set().union(*(docs for pattern, docs in named_mentions
                                                       if pattern.search(content))) if named_mentions else set()
                for facet in facets:
                    key = facet["id"]
                    if facet.get("kind") == "document":
                        continue  # Names give coverage, never substantive relevance.
                    scoped_docs = facet.get("document_ids", ())
                    if scoped_docs and row["document_id"] not in scoped_docs and not (
                            facet.get('kind') == 'original' and open_original_scope):
                        continue
                    query_terms = terms[key]
                    overlap = tokens & query_terms
                    substantive = overlap - _COMMON_TERMS
                    lexical = len(overlap) / max(1, len(query_terms))
                    if overlap and not substantive:
                        lexical *= .30
                    facet_anchors = facet.get("anchors", ())
                    anchor = any(isinstance(item, str) and len(item) >= 3 and item.casefold() in match_content.casefold() for item in facet_anchors)
                    semantic = semantic_results.get(key, {}) if isinstance(semantic_results, Mapping) else {}
                    scores = semantic.get("scores", {}) if isinstance(semantic, Mapping) else {}
                    spans = semantic.get("spans", {}) if isinstance(semantic, Mapping) else {}
                    cosine = _finite(scores.get(row["id"])) if isinstance(scores, Mapping) else None
                    location = _span(spans.get(row["id"]), len(source)) if isinstance(spans, Mapping) else None
                    similarity = 0.0
                    if cosine is not None and .01 <= cosine <= 1.0 and location is not None:
                        hit = max(0, min(end, location[1]) - max(start, location[0]))
                        if hit:
                            similarity = max(0., (cosine - .65) / .35) * min(1., hit / min(end - start, location[1] - location[0]))
                            semantic_hits.append(key)
                    role_match = bool(set(roles) & set(facet.get("roles", ())))
                    score = (.62 * lexical + .63 * similarity + .32 * anchor)
                    if score > 0:
                        score += .10 * role_match
                    score = min(1., score) * penalty
                    if (score <= .03 and broad_comparison and facet.get('kind') == 'original'
                            and row['document_id'] not in named_docs
                            and named_docs <= cross_document_mentions
                            and role_match and _DETAIL_CUE.search(content)
                            and not _BACKGROUND.search(section)):
                        # A third document may directly discuss the requested
                        # comparison. Whole-name mentions plus local operations
                        # only open a low-score candidate opportunity; they do
                        # not turn names into evidence of factual support.
                        score = .08 * penalty
                    if (score <= .03 and row['document_id'] in named_docs
                            and facet.get('kind') in {'original', 'clause'}
                            and set(facet.get('roles', ())) & (set(_roles(section)) | set(outline_roles))
                                & {'mechanism', 'experiment_setup', 'result', 'limitation'}):
                        # Exact metadata scope + an explicit material need may
                        # expose a matching section in another language. This
                        # is only a candidate opportunity, never support or a
                        # compulsory answer slot; source bytes stay unchanged.
                        score = .08 * penalty
                        section_role_nomination = True
                    if (score <= .03 and broad_method_comparison and row["document_id"] in named_docs
                            and facet.get("kind") in {"original", "clause"}
                            and "mechanism" in roles and not _BACKGROUND.search(section)):
                        # In a named-document synthesis, nominate a few method
                        # passages even without query-word overlap. They still
                        # face the SAME full-question reranker as all others.
                        score = .08 * penalty
                    if (nomination_prior and score <= .03 and row['document_id'] in nomination_docs
                            and set(facet.get('roles', ())) & (literal_roles or {'mechanism', 'comparison'}) and prior >= .5):
                        score = .12 * prior * penalty
                    if components and score <= .03 and set(facet.get('roles', ())) & {'mechanism', 'comparison'}:
                        score = .10 * penalty  # Structural nomination only; never semantic support.
                    if (evaluation and score <= .03 and row['document_id'] in named_docs
                            and set(facet.get('roles', ())) & {'experiment_setup', 'result'}):
                        score = .08 * penalty  # Visible evaluation shape, not a support verdict.
                    if (operations and score <= .03 and row['document_id'] in named_docs
                            and 'mechanism' in facet.get('roles', ())):
                        score = .08 * penalty  # User-requested operation opportunity, not semantic proof.
                    if key in role_only_facets:
                        owners = [owner['id'] for owner in facets
                                  if owner.get('kind') in {'original', 'clause'}
                                  and facet['text'].casefold() in owner['text'].casefold()]
                        # A two-word generic request cannot acquire stronger
                        # relevance than the full request it came from. Keep
                        # its candidate opportunity; do not invent a new term.
                        score = min(score, max((facet_scores.get(owner, 0.) for owner in owners), default=score))
                    if score > .03:
                        facet_scores[key] = round(score, 8)
                        lexical_scores[key] = round(lexical, 8)
                anchor_scores = {anchor["id"]: value * penalty for anchor in anchors
                                 if (not anchor["document_ids"] or row["document_id"] in anchor["document_ids"])
                                 and (value := _anchor_score(match_content, anchor["text"], [item["text"] for item in anchors])) > 0}
                if not facet_scores and not anchor_scores:
                    emit_diagnostic(diagnostic_callback, "window_filtered", source_id=row["id"],
                                    document_id=row["document_id"], start=start, end=end,
                                    section=section, reason="no_facet_or_anchor_score")
                    break
                candidate = _window_row(row, start, end, roles, section)
                if axis_exclusions:
                    # Private exact intervals prevent later same-source quote
                    # extension from restoring this already-filtered text.
                    candidate["_axis_tick_exclusions"] = tuple(axis_exclusions)
                if candidate["id"] in window_ids:
                    break
                window_ids.add(candidate["id"])
                candidate.update({"facet_scores": facet_scores, "anchor_scores": anchor_scores, "lexical_scores": lexical_scores,
                                  "semantic_facets": sorted(set(semantic_hits)), "fragment_penalty": penalty,
                                  "baseline_source": row["id"] in baseline,
                                  "nomination_prior": prior,
                                  "section_role_nomination": section_role_nomination,
                                  "outline_roles": list(outline_roles),
                                  "outline_path": outline_path,
                                  "_component_definitions": components,
                                  "_evaluation_cells": evaluation,
                                  "_operation_stages": tuple(operations),
                                  "_operation_quality": dict(operations) if isinstance(operations, Mapping) else {},
                                  "pool_score": max([*facet_scores.values(), *anchor_scores.values()])})
                candidates.append(candidate)
                emit_diagnostic(diagnostic_callback, "window_eligible", window=window_identity(candidate),
                                facet_scores=facet_scores, anchor_scores=anchor_scores,
                                lexical_scores=lexical_scores, semantic_facets=candidate["semantic_facets"],
                                pool_score=candidate["pool_score"], fragment_penalty=penalty)
                window_count += 1
                if len(candidates) >= 512:
                    candidates = prune_pool(candidates)
                break
    # Fair union: each substantive facet and explicitly named document gets
    # nominations before a dominant introduction can consume the whole pool.
    key = lambda row: (-row["pool_score"], len(row["content"]), row.get('filename', ''), row['ordinal'],
                       row['excerpt_start'], hashlib.sha256(row['content'].encode()).hexdigest())
    candidates.sort(key=key)
    selected, selected_ids, selected_sources = [], set(), set()

    def nominate(row, reason="source_round_robin"):
        if len(selected) < cap and row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
            selected_sources.add((row["document_id"], row["source_id"]))
            emit_diagnostic(diagnostic_callback, "candidate_nominated", window=window_identity(row), reason=reason)

    baseline_nominations = 0
    for identifier in baseline:
        if baseline_nominations >= min(MAX_BASELINE_NOMINATIONS, cap // 4):
            break
        candidate = next((row for row in candidates if row["source_id"] == identifier
                          and max(row["facet_scores"].values(), default=0.) >= .08), None)
        if candidate is not None:
            nominate(candidate, "baseline_reserve")
            baseline_nominations += 1

    queues = []
    if operation_profile:
        for doc in sorted(named_docs):
            for stage in ('write', 'change', 'read'):
                ranked = sorted((row for row in candidates
                    if row['document_id'] == doc and stage in row.get('_operation_stages', ())),
                    key=lambda row: (-row.get('_operation_quality', {}).get(stage, 0.), -row['pool_score'],
                                     row['ordinal'], row['excerpt_start']))
                queues.append(_source_round_robin(ranked)[:2])
    if literal_roles is not None:
        for doc in sorted(named_docs):
            for role in sorted(literal_roles & {'mechanism', 'experiment_setup', 'result', 'limitation'}):
                # Equal low-overlap scores previously nominated only a short
                # opening subsection, then filled all 48 slots with distinct
                # sources. An already-recognized later section now has a
                # bounded opportunity; this never forces it into the answer.
                queues.append(_section_round_robin([row for row in candidates
                    if row['document_id'] == doc and role in row.get('outline_roles', ())
                    and not _BACKGROUND.search(row.get('section', ''))])[:5])
    if evaluation_profile:
        for doc in sorted(nomination_docs):
            for cell in ('setup', 'measurement', 'outcome'):
                queue = sorted((row for row in candidates if row['document_id'] == doc
                                and cell in row.get('_evaluation_cells', ())),
                               key=lambda row: (-len(row['_evaluation_cells']), *key(row)))
                queues.append(_source_round_robin(queue)[:2])
    if component_profile:
        # Explicit functional definitions compete before short captions with
        # the same query words can consume every source's nomination chance.
        # Their final selection remains governed by the shared source budget.
        for doc in sorted(nomination_docs):
            queues.append(component_nomination_queue([row for row in candidates if row['document_id'] == doc]))
    for facet in facets:
        if facet.get("kind") == "document":
            continue
        queue = sorted((row for row in candidates if row["facet_scores"].get(facet["id"], 0) >= .08),
                       key=lambda row: (-row["facet_scores"][facet["id"]], *key(row)))
        queues.append(_source_round_robin(queue)[:5])
    for doc in sorted(named_docs):
        queues.append(_source_round_robin([row for row in candidates if row["document_id"] == doc])[:3])
        if broad_comparison:
            for role in (("mechanism", "experiment_setup", "limitation") if literal_roles is None
                         else sorted(literal_roles - {'comparison'})):
                queues.append(_source_round_robin([row for row in candidates
                              if row["document_id"] == doc and role in row["roles"]])[:2])
    if nomination_prior:
        # Literal-only architecture hints reserve eligibility, not output.
        # A named document's low-overlap core can compete with high-overlap
        # appendices; the neural lane's default nomination stays unchanged.
        for doc in sorted(nomination_docs):
            queue = sorted((row for row in candidates if row['document_id'] == doc and row['nomination_prior'] >= .5),
                           key=lambda row: (-row['nomination_prior'], *key(row)))
            queues.append(_source_round_robin(queue)[:3])
    for anchor in anchors:
        queue = sorted((row for row in candidates if row.get("anchor_scores", {}).get(anchor["id"], 0) > 0),
                       key=lambda row: (-row["anchor_scores"][anchor["id"]], *key(row)))
        queues.append(_source_round_robin(queue)[:2])
    for position in range(5):
        for queue in queues:
            if position < len(queue):
                nominate(queue[position], "facet_document_anchor_round_robin")
    for row in candidates:
        if (row["document_id"], row["source_id"]) not in selected_sources:
            nominate(row, "distinct_source_fill")
    for row in _source_round_robin(candidates):
        nominate(row)
    emit_diagnostic(diagnostic_callback, "candidate_pool_final", eligible_ids=[row["id"] for row in candidates],
                    candidate_ids=[row["id"] for row in selected], max_candidates=cap)
    return {"candidates": selected, "trace": {
        "strategy": "literal_facet_window_union_v1", "live_chunks_scanned": len(seen),
        "located_title_headers": len(title_headers), "title_hints_are_semantic_support": False,
        "window_count": window_count, "candidate_count": len(selected), "max_candidates": cap,
        "internal_window_limit": 512,
        "distinct_candidate_sources": len(selected_sources),
        "baseline_nominations": baseline_nominations,
        "max_baseline_nominations": MAX_BASELINE_NOMINATIONS,
        "baseline_nominations_force_selection": False,
        "bibliographic_units_filtered": filtered, "oversized_chunks_skipped": oversized,
        "contact_metadata_units_filtered": contact_filtered,
        "axis_tick_units_filtered": axis_filtered, "graph_data_request_exempt": graph_data_request,
        "window_policy": "bidirectional_section_context_target_1000_cap_1600",
        "named_comparison_method_nominations": broad_method_comparison,
        "literal_structure_nominations": bool(nomination_prior),
        "explicit_document_role_nominations": sum(bool(row.get('section_role_nomination')) for row in selected),
        "explicit_operation_nominations": sum(bool(row.get('_operation_stages')) for row in selected),
        "window_context_is_verified_support": False,
        "provider_calls": 0, "llm_tokens": 0, "entailment_verified": False,
    }}


def select_evidence(
    candidates: Sequence[Mapping[str, Any]], rerank_scores: Mapping[str, float],
    facet_plan: Mapping[str, Any], baseline_sources: Sequence[Mapping[str, Any]],
    max_sources: int = MAX_SOURCES, max_source_bytes: int = MAX_SOURCE_BYTES,
    *, diagnostic_callback=None, input_receipts=None,
) -> dict[str, Any]:
    """Rank-based, novelty-aware selection under hard source/byte limits.

    Ordinal scores are retrieval utility, not confidence or entailment. Length
    is a hard constraint, not a tax that makes a complete explanation lose to
    its last sentence. No fixed per-facet depth can erase a third useful detail.
    """
    facets = _facets(facet_plan)
    anchors = _anchors(facets)
    # Original/phrase queries are retrieval aliases, not additional independent
    # requirements when actual clauses exist. A full-question fallback may
    # contribute its own score but cannot cover an unscored clause.
    literal_facets = [row for row in facets if not row.get("derived_translation") and not row.get("query_only")]
    clauses = [row for row in literal_facets if row.get("kind") == "clause"]
    primary = clauses or [row for row in literal_facets if row.get("kind") == "original"]
    primary_ids = {row["id"] for row in primary}
    facet_by_id = {row["id"]: row for row in facets}
    scored_facets = {row.get("rerank_facet_id") for row in candidates[:MAX_CANDIDATES]
                     if isinstance(row, Mapping) and isinstance(row.get("query"), str)}
    query_views = [row for row in facets if row["id"] not in primary_ids and row["id"] in scored_facets
                   and (row.get("kind") == "original" or (row.get("derived_translation") and row.get("query_only")))]
    # A scored translated/full-question view remains eligible evidence. Its
    # own retrieval receipt is not a new question the user must have answered.
    selection_facets = primary + query_views
    facet_weights = {row["id"]: max(.05, min(2., _finite(row.get("weight")) or .5)) for row in selection_facets}
    anchor_weights = {anchor["id"]: max(.12, min(.32, 2. / max(1, len(anchors)))) for anchor in anchors}
    base_weights = facet_weights | anchor_weights
    scopes = {row["id"]: sorted(set(doc for doc in row.get("document_ids", ()) if isinstance(doc, str)))
              for row in [*facets, *anchors] if row["id"] in base_weights}
    # In an explicit multi-paper comparison, an explanation in paper A does
    # not cover the corresponding requirement in paper B. Split each scoped
    # requirement without increasing its total weight or inventing new facts.
    dimensions = {
        key: [(doc, "ed_" + hashlib.sha256((key + "\0" + doc).encode()).hexdigest()[:16])
              for doc in scopes.get(key, ())] or [(None, key)]
        for key in base_weights
    }
    weights = {dimension: facet_weights[key] / len(values)
               for key, values in dimensions.items() if key in facet_weights for _, dimension in values}
    requirement_dimensions = {dimension for key in primary_ids for _, dimension in dimensions[key]}
    named_docs = {doc for facet in facets for doc in facet.get("document_ids", ()) if isinstance(doc, str)}
    cap = min(MAX_SOURCES, max(0, max_sources)) if type(max_sources) is int else MAX_SOURCES
    budget = min(MAX_SOURCE_BYTES, max(0, max_source_bytes)) if type(max_source_bytes) is int else MAX_SOURCE_BYTES
    fair_documents = len(named_docs) >= 2
    document_soft_quota = math.ceil(cap / len(named_docs)) if fair_documents else 0
    demand_roles = {role for facet in primary for role in facet.get("roles", ())}
    rescue_branches = {"focused", "quoted_audit"}
    has_literal_focus = any(isinstance(row, Mapping) and row.get("rerank_branch") == "focused"
                            for row in candidates[:MAX_CANDIDATES])

    def complete_pair_receipt(row):
        if not isinstance(row.get("query"), str):
            return False
        digest = hashlib.sha256(row["query"].encode()).hexdigest()
        passage_digest = hashlib.sha256(row["content"].encode()).hexdigest()
        receipt = input_receipts.get(row["id"], {}) if isinstance(input_receipts, Mapping) else {}
        if not isinstance(receipt, Mapping):
            return False
        tokens, maximum = receipt.get("pair_tokens_before_truncation"), receipt.get("max_pair_tokens")
        return (row.get("rerank_query_id") == "rq_" + digest[:16]
                and receipt.get("submitted_query_sha256") == digest
                and receipt.get("actual_query_sha256") == digest
                and receipt.get("submitted_passage_sha256") == passage_digest
                and receipt.get("actual_passage_sha256") == passage_digest
                and receipt.get("actual_query_character_end") == len(row["query"])
                and receipt.get("query_complete") is True and receipt.get("passage_complete") is True
                and receipt.get("input_scope") == "complete_pair"
                and receipt.get("basis") == "tokenizer_budget_check_not_semantic_verification"
                and type(tokens) is int and type(maximum) is int and 0 < tokens <= maximum <= 512)

    def valid_audit_pair(row):
        original = facet_by_id.get(row.get("rerank_view_of"), {})
        view = facet_by_id.get(row.get("rerank_facet_id"), {})
        if (has_literal_focus or row.get("rerank_branch") != "quoted_audit"
                or row.get("rerank_view") != "quoted_claim_audit"
                or not valid_quoted_audit_view(view, original)
                or row.get("query") != view.get("text")
                or row.get("nomination_facet_ids") or row.get("nomination_alignment")
                or (view.get("document_ids") and row["document_id"] not in view["document_ids"])):
            return False
        identity = f"{row['document_id']}\0{row['source_id']}\0{row['excerpt_start']}:{row['excerpt_end']}\0{row['content']}"
        expected_id = "ew_" + hashlib.sha256(identity.encode()).hexdigest()[:20] + "::" + view["id"]
        support = _finite(row.get("facet_scores", {}).get(view["id"]))
        nominated = _finite(row.get("retrieval_support"))
        return (row["id"] == expected_id
                and support is not None and support > 0 and nominated is not None
                and math.isclose(support, nominated, rel_tol=1e-9, abs_tol=1e-12)
                and complete_pair_receipt(row))

    def scoped_query_alias(alias, literal, document_id):
        scope = set(literal.get("document_ids", ()))
        return (alias.get("derived_translation") and alias.get("query_only")
                and not alias.get("translation_full_query") and alias.get("query_alias_of") == literal.get("id")
                and alias.get("alias_alignment") == "unique_query_only_requirement_kind"
                and set(alias.get("document_ids", ())) == scope
                and (not scope or document_id in scope))

    def valid_scoring_view(row, facet_id):
        literal = facet_by_id.get(facet_id, {})
        actual = facet_by_id.get(row.get("rerank_facet_id"), {})
        if (row.get("rerank_branch") != "focused" or facet_id not in primary_ids
                or literal.get("kind") != "clause" or literal.get("derived_translation")
                or literal.get("query_only") or not isinstance(row.get("query"), str)
                or row.get("query") != actual.get("text")
                or row.get("rerank_query_id") != "rq_" + hashlib.sha256(row["query"].encode("utf-8")).hexdigest()[:16]):
            return False
        if row.get("rerank_view") == "literal":
            return actual.get("id") == facet_id
        return (row.get("rerank_view") == "guarded_alias"
                and scoped_query_alias(actual, literal, row["document_id"]))

    def alias_nomination(row, facet_id):
        # Query-only translations may nominate a literal pair the multilingual
        # span retriever missed. They are not literal embedding hits, semantic
        # equivalence, or support for any unscored requirement. Association is
        # produced by the query planner's conservative unique-role guards.
        literal = facet_by_id.get(facet_id, {})
        scoring_alias = row.get("rerank_view") == "guarded_alias"
        if scoring_alias:
            # The original need may nominate a window, while the associated
            # language-matched query view is what actually receives a score.
            # Keep these identities separate; neither proves equivalence.
            native = _finite(row.get("facet_scores", {}).get(facet_id))
            if native is not None and native > 0:
                support = _finite(row.get("retrieval_support"))
                if support is not None and math.isclose(support, native, rel_tol=1e-9, abs_tol=1e-12):
                    return (facet_id,), min(1., support)
                return (), 0.
        identifiers = row.get("nomination_facet_ids")
        if (row.get("rerank_branch") != "focused" or facet_id not in primary_ids
                or literal.get("kind") != "clause" or literal.get("derived_translation")
                or (not scoring_alias and row.get("query") != literal.get("text"))
                or row.get("nomination_alignment") != "unique_query_only_requirement_kind"
                or not isinstance(identifiers, (list, tuple)) or not identifiers
                or not all(isinstance(value, str) for value in identifiers)
                or len(identifiers) > len(facets) or len(set(identifiers)) != len(identifiers)):
            return (), 0.
        scope = set(literal.get("document_ids", ()))
        if scope and row["document_id"] not in scope:
            return (), 0.
        scores = []
        for identifier in identifiers:
            alias = facet_by_id.get(identifier, {})
            score = _finite(row.get("facet_scores", {}).get(identifier))
            if not scoped_query_alias(alias, literal, row["document_id"]) or score is None or score <= 0:
                return (), 0.
            scores.append(score)
        support = _finite(row.get("retrieval_support"))
        if support is None or not math.isclose(support, max(scores), rel_tol=1e-9, abs_tol=1e-12):
            return (), 0.
        return tuple(identifiers), min(1., support)

    def ineligible_reason(row):
        # This single gate is used before ordinal ranking AND before merging.
        # Excluded candidates must neither occupy a rank nor reappear by
        # extending a neighbouring quote. Roles remain a lexical heuristic.
        if (max(row["_coverage"].values(), default=0.) <= 0
                and not row.get("_alias_nomination_support") and not row.get("_audit_nomination_support")):
            return "coverage_below_minimum"
        if (_BACKGROUND.search(str(row.get("section", "")))
                and demand_roles & {"mechanism", "experiment_setup"}
                and not set(row.get("roles", ())) & demand_roles):
            return "generic_background_without_requested_role"
        return None

    available = []
    missing_rerank = 0
    source_snapshots: dict[tuple[str, str], str] = {}
    conflicting_sources: set[tuple[str, str]] = set()
    for raw in candidates[:MAX_CANDIDATES]:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            continue
        row = dict(raw)
        source = row.get("_source_content")
        location = _span((row.get("excerpt_start"), row.get("excerpt_end")), len(source)) if isinstance(source, str) else None
        if (location is None or location[1] - location[0] > MAX_WINDOW_CHARS
                or source[location[0]:location[1]] != row.get("content")
                or not isinstance(row.get("source_id"), str) or not isinstance(row.get("document_id"), str)):
            continue
        source_identity = row["document_id"], row["source_id"]
        if source_identity in source_snapshots and source_snapshots[source_identity] != source:
            conflicting_sources.add(source_identity)
        else:
            source_snapshots[source_identity] = source
        score = _finite(rerank_scores.get(row["id"])) if isinstance(rerank_scores, Mapping) else None
        if score is None:
            # An incomplete/unavailable reranker result must not silently be
            # interpreted as middling relevance. Root orchestration owns fallback.
            missing_rerank += 1
            continue
        row["rerank_logit"] = score
        row["_tokens"] = set(lexical_tokens(row["content"]))
        row["_normalized_content"] = " ".join(row["content"].casefold().split())
        row["_bytes"] = len(row["content"].encode("utf-8"))
        query = row.get("query")
        row["_query_group"] = ("rq_" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
                               if isinstance(query, str) and query else
                               str(row.get("rerank_query_id") or row.get("rerank_facet_id") or "__shared_query__"))
        actual_facet = row.get("rerank_facet_id")
        credited_facet = actual_facet
        audit = (row.get("rerank_branch") == "quoted_audit" or row.get("rerank_view") == "quoted_claim_audit"
                 or facet_by_id.get(actual_facet, {}).get("kind") == "query_view")
        if audit:
            if not valid_audit_pair(row):
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=0,
                                window=window_identity(row), reason="invalid_quoted_audit_pair_receipt", stage="before_rank")
                continue
            # This is a view of the original request, not another requirement.
            # Its rank can reserve one source but gives no literal/anchor credit.
            credited_facet = None
        elif "rerank_view_of" in row or "rerank_view" in row:
            credited_facet = row.get("rerank_view_of")
            if not isinstance(credited_facet, str) or not valid_scoring_view(row, credited_facet):
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=0,
                                window=window_identity(row), reason="invalid_scored_query_view_mapping", stage="before_rank")
                continue
        if credited_facet not in facet_weights and not isinstance(query, str):
            # Direct legacy selectors have no pair metadata. Credit just one
            # independently supported clause, never all clauses at once.
            credited_facet = max(primary, key=lambda item: row.get("facet_scores", {}).get(item["id"], 0.)
                                 * facet_weights[item["id"]])["id"] if primary else None
        row["_credited_facet"] = credited_facet
        row["_actual_scored_facet"] = actual_facet
        row["_translated_scoring_view"] = row.get("rerank_view") == "guarded_alias"
        row["_audit_nomination_support"] = min(1., row["retrieval_support"]) if audit else 0.
        row["_coverage"] = {
            dimension: max(0., min(1., float(value)))
            for key, value in row.get("facet_scores", {}).items()
            if key == credited_facet and key in facet_weights and _finite(value) is not None
            and not row["_translated_scoring_view"]
            for doc, dimension in dimensions[key] if doc is None or doc == row["document_id"]
        }
        row["_alias_nomination_ids"], row["_alias_nomination_support"] = (
            alias_nomination(row, credited_facet) if not audit and max(row["_coverage"].values(), default=0.) <= 0 else ((), 0.))
        reason = ineligible_reason(row)
        if reason is not None:
            emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=0,
                            window=window_identity(row), reason=reason,
                            stage="before_rank", coverage=row["_coverage"])
            continue
        available.append(row)
    if conflicting_sources:
        consistent = []
        for row in available:
            if (row["document_id"], row["source_id"]) in conflicting_sources:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=0,
                                window=window_identity(row), reason="source_snapshot_mismatch",
                                stage="before_rank")
            else:
                consistent.append(row)
        # No authoritative snapshot can be inferred from retrieval rank. Fail
        # closed for the entire source instead of arbitrarily choosing a version.
        available = consistent
    # Define the common population before ranking: exact actually-scored
    # full-original/audit windows, with complete receipts on BOTH sides.
    # One deterministic geometry per independent source prevents duplicate
    # windows/IDs from adding votes. Scores never choose this common set.
    audit_rows = [row for row in available if row.get("rerank_branch") == "quoted_audit"]
    full_by_id = {}
    for row in available:
        if row.get("rerank_branch") == "full":
            full_by_id.setdefault(row["id"], []).append(row)
    common_by_source = {}
    for audit in sorted(audit_rows, key=lambda row: (row["document_id"], row["source_id"],
                        row["excerpt_start"], row["excerpt_end"], hashlib.sha256(row["content"].encode()).hexdigest())):
        original = facet_by_id.get(audit["rerank_view_of"], {})
        full = next((row for row in full_by_id.get(audit["id"].split("::", 1)[0], ())
                     if row.get("rerank_facet_id") == original.get("id")
                     and row.get("query") == original.get("text")
                     and not row.get("rerank_view") and not row.get("rerank_view_of")
                     and complete_pair_receipt(row)
                     and all(row[key] == audit[key] for key in ("document_id", "source_id", "excerpt_start", "excerpt_end", "content", "_source_content"))), None)
        if full is None:
            continue
        key = audit["document_id"], audit["source_id"]
        common_by_source.setdefault(key, (full, audit))
    common = list(common_by_source.values())
    common_count = len(common)
    audit_eligible_ids = set()
    for full, audit in common:
        full_rank = 1 + sum(other[0]["rerank_logit"] > full["rerank_logit"] for other in common)
        audit_rank = 1 + sum(other[1]["rerank_logit"] > audit["rerank_logit"] for other in common)
        gain = full_rank - audit_rank
        eligible = common_count >= 4 and audit_rank <= math.ceil(common_count / 2) and gain > 0
        audit.update(_audit_common_full_rank=full_rank, _audit_common_rank=audit_rank,
                     _audit_rank_gain=gain, _audit_common_count=common_count)
        if eligible:
            audit_eligible_ids.add(audit["id"])
        emit_diagnostic(diagnostic_callback, "audit_paired_rank", window=window_identity(audit),
                        full_window=window_identity(full), common_source_count=common_count,
                        full_common_rank=full_rank, audit_common_rank=audit_rank, rank_gain=gain,
                        audit_rank_ceiling=math.ceil(common_count / 2), eligible=eligible,
                        rank_semantics="competition_rank_on_fixed_complete_same_window_source_set",
                        selection_semantics="positive_rank_shift_not_verified_support")
    eligible_audits = {audit["id"]: audit for _, audit in common if audit["id"] in audit_eligible_ids}
    for row in audit_rows:
        if row["id"] not in audit_eligible_ids:
            emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=0,
                            window=window_identity(row), reason="audit_without_eligible_paired_rank_shift", stage="before_rank")
    # Never let merge, repair, or receipt caches resurrect an ineligible view.
    # Full rows stay exactly as before; audit duplicates appear only once.
    available = [row for row in available if row.get("rerank_branch") != "quoted_audit"] + list(eligible_audits.values())

    def audit_priority(row):
        return (-row.get("_audit_rank_gain", 0), row.get("_audit_common_rank", 0),
                row["document_id"], row["source_id"], row["excerpt_start"], row["excerpt_end"], row["id"])

    group_tops: dict[str, float] = {}
    for row in available:
        group = row["_query_group"]
        group_tops[group] = max(group_tops.get(group, -math.inf), row["rerank_logit"])
    group_values = {group: sorted([row["rerank_logit"] for row in available
                                  if row["_query_group"] == group], reverse=True)
                    for group in group_tops}
    full_branch = [row for row in available if row.get("rerank_branch") == "full"
                   and isinstance(row.get("query"), str) and row["query"].strip()]
    focused_branch = [row for row in available if row.get("rerank_branch") in rescue_branches
                      and isinstance(row.get("query"), str) and row["query"].strip()]
    full_queries = {row["_query_group"] for row in full_branch}
    focused_queries = {row["_query_group"] for row in focused_branch}
    focused_enabled = bool(cap and full_queries and focused_queries
                           and not full_queries & focused_queries
                           and any(row["_bytes"] <= budget
                                   and 9. / (9. + group_values[row["_query_group"]].index(row["rerank_logit"])) >= .20
                                   for row in full_branch))
    audit_enabled = focused_enabled and any(row.get("rerank_branch") == "quoted_audit" for row in available)
    rescue_source_limit = 1 if audit_enabled else 2
    if not focused_enabled and any(row.get("rerank_branch") in rescue_branches for row in available):
        # An orphaned/mislabeled rescue branch must not silently become an
        # unrestricted ordinary query. Keep the valid full-query fallback.
        for row in available:
            if row.get("rerank_branch") in rescue_branches:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=0,
                                window=window_identity(row), reason="focused_branch_without_eligible_full",
                                stage="before_rank")
        available = [row for row in available if row.get("rerank_branch") not in rescue_branches]
        group_tops = {}
        for row in available:
            group = row["_query_group"]
            group_tops[group] = max(group_tops.get(group, -math.inf), row["rerank_logit"])
        group_values = {group: sorted([row["rerank_logit"] for row in available
                                      if row["_query_group"] == group], reverse=True)
                        for group in group_tops}
    for row in available:
        logit = row["rerank_logit"]
        top = group_tops[row["_query_group"]]
        rank = group_values[row["_query_group"]].index(logit) + 1
        # A small positive rescaling of logits must not change source selection.
        # This reciprocal-rank utility is NOT a probability; eligibility below
        # still requires independent live-source retrieval support.
        row["_quality"] = 9. / (8. + rank)
        if row.get("rerank_branch") == "quoted_audit":
            # Use the fixed common population, not the eligible-only subset
            # or unrelated unmatched audit logits, for this view's utility.
            row["_quality"] = 9. / (8. + row["_audit_common_rank"])
        if focused_enabled and row.get("rerank_branch") in rescue_branches:
            row["_quality"] *= .75
        row["_rank"] = rank
        if row["_alias_nomination_support"]:
            # Coverage here is explicitly a rank-conditioned nomination proxy
            # for this actually scored query-view/window pair, not a forged
            # literal similarity or verified fact. The input scores stay intact.
            row["_coverage"] = {
                dimension: row["_quality"] * row["_alias_nomination_support"]
                for doc, dimension in dimensions[row["_credited_facet"]]
                if doc is None or doc == row["document_id"]
            }
        row["_anchor_proxy"] = {} if row["_audit_nomination_support"] else {
            dimension: max(0., min(1., float(value))) * row["_quality"]
            for key, value in row.get("anchor_scores", {}).items()
            if key in anchor_weights and _finite(value) is not None
            for doc, dimension in dimensions[key] if doc is None or doc == row["document_id"]
        }
        emit_diagnostic(diagnostic_callback, "selection_candidate", window=window_identity(row),
                        rerank_logit=logit, query_group=row["_query_group"], group_top=top,
                        rank=rank, quality=row["_quality"], credited_facet=row["_credited_facet"],
                        actual_scored_facet=row["_actual_scored_facet"],
                        rerank_view=row.get("rerank_view"), rerank_view_of=row.get("rerank_view_of"),
                        rerank_branch=row.get("rerank_branch"),
                        nomination_facet_ids=list(row["_alias_nomination_ids"]),
                        nomination_support=row["_alias_nomination_support"],
                        audit_nomination_support=row["_audit_nomination_support"],
                        audit_common_full_rank=row.get("_audit_common_full_rank"),
                        audit_common_rank=row.get("_audit_common_rank"),
                        audit_rank_gain=row.get("_audit_rank_gain"),
                        coverage_semantics=("quoted_audit_rank_only_no_requirement_credit"
                                            if row["_audit_nomination_support"] else
                                            "scored_query_view_rank_times_nomination_not_verified_support"
                                            if row["_translated_scoring_view"] else
                                            "scored_literal_rank_times_alias_nomination_not_verified_support"
                                            if row["_alias_nomination_support"] else "retrieval_relevance_proxy"),
                        coverage=row["_coverage"], anchor_proxy=row["_anchor_proxy"], bytes=row["_bytes"])
    # Immutable submitted-window receipts, before greedy packing mutates quote
    # geometry. Retaining the whole submission does NOT prove the cross-encoder
    # saw its whole text: its tokenizer can truncate a long query/passage pair.
    # The reranker owns per-pair input receipts; this selector owns geometry.
    scored_windows = [dict(row) for row in available]
    selected: list[dict[str, Any]] = []
    coverage = {dimension: 0. for values in dimensions.values() for _, dimension in values}
    credited_rows: list[dict[str, Any]] = []
    credited_ids: set[str] = set()
    depth_values = {key: 0. for key in weights}
    documents: set[str] = set()
    document_counts: dict[str, int] = {}
    fair_coverage_selections = 0
    focused_selected_sources = 0
    focused_merge_extensions = 0
    used = 0
    objective = 0.
    rejected_same_chunk = 0
    rejected_same_source_windows = {}
    partial_context_extensions = 0

    def marginal_gain(row):
        # Search aliases do not create an obligation to take two passages each.
        # Rank remains useful after earlier passages discuss the same facet.
        new_anchors = sum(max(0., value - coverage[key])
                          for key, value in row["_anchor_proxy"].items())
        return row["_quality"] + min(.25, .12 * new_anchors)

    def record_credit(row):
        # Store the scored submission, not a later merged quote whose new text has
        # not been independently reranked. Exact merge provenance stays below.
        if row["id"] in credited_ids:
            return
        credited_ids.add(row["id"])
        credited_rows.append(dict(row))
        for key in row["_coverage"]:
            depth_values[key] += row["_quality"]
        for key, value in (row["_coverage"] | row["_anchor_proxy"]).items():
            coverage[key] = max(coverage[key], value)

    while available and len(selected) < cap:
        options = []
        for row in available:
            focused_row = focused_enabled and row.get("rerank_branch") in rescue_branches
            if focused_row and (focused_selected_sources >= rescue_source_limit or row["_quality"] < .20):
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=len(selected),
                                window=window_identity(row), reason="focused_source_cap_or_rank_floor",
                                focused_selected_sources=focused_selected_sources,
                                quality=row["_quality"], minimum_quality=.20)
                continue
            if row["_bytes"] + used > budget:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=len(selected),
                                window=window_identity(row), reason="byte_budget", used_bytes=used)
                continue
            same = next((prior for prior in selected if (prior["document_id"], prior["source_id"])
                         == (row["document_id"], row["source_id"])), None)
            if same is not None:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=len(selected),
                                window=window_identity(row), reason="same_source_pending_merge")
                continue  # Exact-source merge is performed below after selection.
            duplicate = any(_duplicate_window(row, prior)
                            for prior in selected
                            if not (row["document_id"] != prior["document_id"]
                                    and row["document_id"] in named_docs
                                    and prior["document_id"] in named_docs))
            if duplicate:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=len(selected),
                                window=window_identity(row), reason="duplicate_evidence")
                continue
            gain = marginal_gain(row)
            doc_gain = .12 * row["_quality"] if row["document_id"] in named_docs - documents else 0.
            redundancy = max((len(row["_tokens"] & prior["_tokens"]) / max(1, len(row["_tokens"] | prior["_tokens"])) for prior in selected), default=0.)
            penalty = .30 * redundancy
            utility = gain + doc_gain - penalty
            reason = ineligible_reason(row)
            if reason is not None:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=len(selected),
                                window=window_identity(row), reason=reason,
                                gain=gain, utility=utility, coverage=row["_coverage"])
                continue
            if fair_documents and row["_quality"] < .20:
                emit_diagnostic(diagnostic_callback, "selection_rejected", iteration=len(selected),
                                window=window_identity(row), reason="rank_below_document_fairness_floor",
                                quality=row["_quality"], minimum_quality=.20)
                continue
            # Explicit comparisons need evidence from each requested document,
            # not eight individually high-ranked windows of the same paper.
            # Eligibility, the global rank floor and hard budgets still apply;
            # document names never manufacture missing retrieval support.
            required_document = fair_documents and row["document_id"] in named_docs - documents
            balanced_document = (fair_documents and row["document_id"] in named_docs
                                 and document_counts.get(row["document_id"], 0) < document_soft_quota
                                 and utility >= .20)
            focused_reservation = focused_row and focused_selected_sources == 0
            document_priority = ((3 if focused_enabled else 2) if required_document else
                                 2 if focused_reservation else 1 if balanced_document else 0)
            options.append((document_priority, utility, gain + doc_gain, row))
            emit_diagnostic(diagnostic_callback, "selection_option", iteration=len(selected),
                            window=window_identity(row), gain=gain, document_gain=doc_gain,
                            redundancy=redundancy, penalty=penalty, utility=utility,
                            required_document=required_document, document_slot_priority=document_priority,
                            document_soft_quota=document_soft_quota, focused_reservation=focused_reservation)
        if not options:
            emit_diagnostic(diagnostic_callback, "selection_stop", reason="no_eligible_options",
                            selected_ids=[row["id"] for row in selected], used_bytes=used)
            break
        options.sort(key=lambda item: (-item[0],
                                      audit_priority(item[3]) if item[3].get("rerank_branch") == "quoted_audit" else (0, 0, "", "", 0, 0, ""),
                                      -item[1], -item[2], item[3]["_bytes"], item[3]["id"]))
        document_priority, utility, marginal, chosen = options[0]
        required_document = fair_documents and chosen["document_id"] in named_docs - documents
        focused_reservation = (focused_enabled and chosen.get("rerank_branch") in rescue_branches
                               and focused_selected_sources == 0)
        if utility < .20 and not required_document and not focused_reservation:
            emit_diagnostic(diagnostic_callback, "selection_stop", reason="utility_below_minimum",
                            best_window=window_identity(chosen), utility=utility,
                            selected_ids=[row["id"] for row in selected], used_bytes=used)
            break
        available.remove(chosen)
        selected.append(chosen)
        used += chosen["_bytes"]
        objective += marginal
        documents.add(chosen["document_id"])
        document_counts[chosen["document_id"]] = document_counts.get(chosen["document_id"], 0) + 1
        fair_coverage_selections += int(required_document)
        focused_selected_sources += int(focused_enabled and chosen.get("rerank_branch") in rescue_branches)
        record_credit(chosen)
        emit_diagnostic(diagnostic_callback, "selection_chosen", window=window_identity(chosen),
                        utility=utility, marginal=marginal, used_bytes=used,
                        document_slot_priority=document_priority)
    if not available or len(selected) >= cap:
        emit_diagnostic(diagnostic_callback, "selection_stop",
                        reason="source_cap" if len(selected) >= cap else "candidates_exhausted",
                        selected_ids=[row["id"] for row in selected], used_bytes=used)
    # Grow neighbouring evidence to a fixed point within each selected source.
    # A bridge accepted later can make an earlier disconnected window eligible.
    # At most 48 candidates are consumed per source, so retries remain bounded.
    for prior in selected:
        pending = sorted((row for row in available
                          if (row["document_id"], row["source_id"]) == (prior["document_id"], prior["source_id"])),
                         key=lambda item: (-item["_quality"], -max(item["_coverage"].values(), default=0.), item["id"]))
        rejected_geometry = set()
        for _ in range(MAX_CANDIDATES):
            accepted = None
            for row in pending:
                if row.get("rerank_branch") == "quoted_audit":
                    # A view has one reserved source, not a second route to
                    # extend every full-query source after the cap is applied.
                    continue
                reason = ineligible_reason(row)
                if row["_source_content"] != prior["_source_content"]:
                    reason = "source_snapshot_mismatch"
                if reason is not None:
                    emit_diagnostic(diagnostic_callback, "merge_rejected", prior=window_identity(prior),
                                    window=window_identity(row), reason=reason)
                    continue
                left, right = min(prior["excerpt_start"], row["excerpt_start"]), max(prior["excerpt_end"], row["excerpt_end"])
                exclusions = (*prior.get("_axis_tick_exclusions", ()), *row.get("_axis_tick_exclusions", ()))
                if any(left < end and right > start for start, end in exclusions):
                    # Apply before every bridge/partial-extension route. Only
                    # original intervals proven filtered by candidate prep are
                    # blocked; unknown mixtures and graph-data asks add none.
                    rejected_geometry.add(row["id"])
                    emit_diagnostic(diagnostic_callback, "merge_rejected", prior=window_identity(prior),
                                    window=window_identity(row), reason="excluded_axis_interval")
                    continue
                gap = max(prior["excerpt_start"], row["excerpt_start"]) - min(prior["excerpt_end"], row["excerpt_end"])
                gap_start = min(prior["excerpt_end"], row["excerpt_end"])
                gap_end = max(prior["excerpt_start"], row["excerpt_start"])
                bridge = prior["_source_content"][gap_start:gap_end] if gap > 0 else ""
                bridge_safe = (not bridge.strip() or (gap <= 240 and prior.get("section") == row.get("section")
                               and not any(is_section_heading(line) for line in bridge.splitlines()) and not _bibliographic(bridge)))
                if gap > 0 and not bridge_safe:
                    rejected_geometry.add(row["id"])
                    emit_diagnostic(diagnostic_callback, "merge_rejected", prior=window_identity(prior),
                                    window=window_identity(row), reason="unsafe_bridge_or_window_limit",
                                    gap=gap, bridge_safe=bridge_safe, merged_chars=right-left)
                    continue
                partial = False
                if right - left > MAX_WINDOW_CHARS:
                    extension = bounded_evidence_extension(
                        prior["_source_content"], (prior["excerpt_start"], prior["excerpt_end"]),
                        (row["excerpt_start"], row["excerpt_end"]), MAX_WINDOW_CHARS)
                    if extension is None or prior.get("section") != row.get("section"):
                        rejected_geometry.add(row["id"])
                        emit_diagnostic(diagnostic_callback, "merge_rejected", prior=window_identity(prior),
                                        window=window_identity(row), reason="no_complete_unit_extension",
                                        merged_chars=right-left)
                        continue
                    left, right = extension
                    partial = not (left <= row["excerpt_start"] and right >= row["excerpt_end"])
                content = prior["_source_content"][left:right]
                extra = len(content.encode("utf-8")) - prior["_bytes"]
                gain = marginal_gain(row)
                if (focused_enabled and row.get("rerank_branch") == "focused"
                        and extra > 0 and focused_merge_extensions >= 2):
                    emit_diagnostic(diagnostic_callback, "merge_rejected", prior=window_identity(prior),
                                    window=window_identity(row), reason="focused_extension_cap")
                    continue
                if extra + used > budget or row["_quality"] < .20:
                    emit_diagnostic(diagnostic_callback, "merge_rejected", prior=window_identity(prior),
                                    window=window_identity(row), reason="byte_budget_or_low_gain",
                                    extra_bytes=extra, gain=gain, utility=row["_quality"])
                    continue
                emit_diagnostic(diagnostic_callback, "merge_accepted", prior=window_identity(prior),
                                window=window_identity(row), start=left, end=right,
                                gain=0. if partial else gain, extra_bytes=extra,
                                supplemental_window_fully_retained=not partial)
                prior.update({"content": content, "excerpt_start": left, "excerpt_end": right, "_bytes": len(content.encode("utf-8")),
                              "_tokens": set(lexical_tokens(content)), "_normalized_content": " ".join(content.casefold().split())})
                used += extra
                # A contained reranked window may improve the coverage receipt,
                # but is not a newly added quote and earns no extra objective.
                if extra > 0:
                    objective += 0. if partial else gain
                    focused_merge_extensions += int(focused_enabled and row.get("rerank_branch") == "focused")
                    partial_context_extensions += int(partial)
                if not partial:
                    record_credit(row)
                accepted = row
                if partial:
                    rejected_geometry.add(row["id"])
                else:
                    rejected_geometry.discard(row["id"])
                break
            if accepted is None:
                break
            pending.remove(accepted)
        rejected_same_source_windows.update({row["id"]: row for row in scored_windows if row["id"] in rejected_geometry})

    # Bounded source-set repair: first arrival is not permanent ownership of a
    # chunk or the last citation slot. Compare complete candidate windows with
    # the packed set; accept only strict objective improvements. This is a
    # retrieval utility objective, never evidence entailment or a gold oracle.
    def retained_receipts(quote):
        return [row for row in scored_windows
                if (row.get("rerank_branch") != "quoted_audit" or quote.get("rerank_branch") == "quoted_audit")
                and (row["document_id"], row["source_id"]) == (quote["document_id"], quote["source_id"])
                and row["_source_content"] == quote["_source_content"]
                and quote["excerpt_start"] <= row["excerpt_start"]
                and row["excerpt_end"] <= quote["excerpt_end"]]

    quote_receipts = {}
    def geometry(row):
        return row["document_id"], row["source_id"], row["excerpt_start"], row["excerpt_end"]

    def receipts(quote):
        # Identical source geometry is not identical scoring permission: only
        # the one reserved audit source may retain audit-view rank receipts.
        # A cache entry warmed during a hypothetical audit swap must never be
        # reused later to boost the same window on the unrestricted full path.
        key = (*geometry(quote), quote.get("rerank_branch") == "quoted_audit")
        if key not in quote_receipts:
            quote_receipts[key] = retained_receipts(quote)
        return quote_receipts[key]

    def set_proxy(quotes):
        facet_values, anchor_values = {}, {}
        rank_value = 0.
        for quote in quotes:
            members = receipts(quote)
            # Partial extensions cannot inherit the score of the full window
            # they came from. A source gets one best fully retained rank value.
            rank_value += max((row["_quality"] for row in members), default=0.)
            for row in members:
                for key, value in row["_coverage"].items():
                    facet_values[key] = max(facet_values.get(key, 0.), value)
                for key, value in row["_anchor_proxy"].items():
                    anchor_values[key] = max(anchor_values.get(key, 0.), value)
        redundancy = sum(max((len(row["_tokens"] & other["_tokens"]) /
                              max(1, len(row["_tokens"] | other["_tokens"]))
                             for other in quotes if other is not row), default=0.) for row in quotes)
        value = (rank_value + .20 * sum(value for key, value in facet_values.items() if key in requirement_dimensions)
                 + .12 * min(2., sum(anchor_values.values())) - .15 * redundancy)
        return value, facet_values

    initial_named = named_docs & {row["document_id"] for row in selected}
    protected_document_counts = {doc: min(document_soft_quota, sum(row["document_id"] == doc for row in selected))
                                 for doc in initial_named} if fair_documents else {}
    initial_focused = sum(row.get("rerank_branch") in rescue_branches for row in selected) if focused_enabled else 0
    swaps, swap_evaluations = 0, 0
    for _ in range(2):
        before, before_facets = set_proxy(selected)
        protected = {key for key, value in before_facets.items() if key in requirement_dimensions and value >= .12}
        best = None
        for slot, previous in enumerate(selected):
            remaining = selected[:slot] + selected[slot + 1:]
            other_sources = {(row["document_id"], row["source_id"]) for row in remaining}
            for candidate in scored_windows:
                if (previous.get("rerank_branch") == "quoted_audit"
                        and candidate.get("rerank_branch") == "quoted_audit"
                        and audit_priority(candidate) > audit_priority(previous)):
                    continue  # Repair cannot override the reserved rank-shift competition.
                if (candidate["_quality"] < .20 or geometry(candidate) == geometry(previous)
                        or (candidate["document_id"], candidate["source_id"]) in other_sources
                        or candidate["_bytes"] + sum(row["_bytes"] for row in remaining) > budget):
                    continue
                if ((candidate["document_id"], candidate["source_id"]) == (previous["document_id"], previous["source_id"])
                        and previous["excerpt_start"] <= candidate["excerpt_start"]
                        and candidate["excerpt_end"] <= previous["excerpt_end"]):
                    # Merely deleting already admitted complete context cannot
                    # be an improvement just because Jaccard becomes smaller.
                    # No new scored requirement is gained by a contained crop.
                    continue
                proposed = [*remaining[:slot], candidate, *remaining[slot:]]
                if not initial_named <= {row["document_id"] for row in proposed}:
                    continue
                if any(sum(row["document_id"] == doc for row in proposed) < count
                       for doc, count in protected_document_counts.items()):
                    continue
                focused_count = sum(row.get("rerank_branch") in rescue_branches for row in proposed)
                if focused_enabled and (focused_count > rescue_source_limit or (initial_focused and focused_count == 0)):
                    continue
                if any(_duplicate_window(candidate, row) for row in remaining
                       if not (candidate["document_id"] != row["document_id"]
                               and candidate["document_id"] in named_docs and row["document_id"] in named_docs)):
                    continue
                score, next_facets = set_proxy(proposed)
                swap_evaluations += 1
                if not all(next_facets.get(key, 0.) >= .12 for key in protected):
                    continue
                if score <= before + 1e-9:
                    continue
                choice = (score, slot, candidate)
                if best is None or score > best[0] + 1e-9 or (
                        abs(score - best[0]) <= 1e-9 and (slot, candidate["id"]) < (best[1], best[2]["id"])):
                    best = choice
        if best is None:
            break
        score, slot, candidate = best
        emit_diagnostic(diagnostic_callback, "source_set_replacement", removed=window_identity(selected[slot]),
                        added=window_identity(candidate), objective_before=before, objective_after=score)
        selected[slot] = dict(candidate)
        swaps += 1

    # Recompute every public/diagnostic credit from final geometry. In
    # particular a replaced or partially included scored window cannot leave
    # stale facet/depth coverage behind.
    coverage = dict.fromkeys(coverage, 0.)
    credited_rows, credited_ids = [], set()
    depth_values = dict.fromkeys(depth_values, 0.)
    for quote in selected:
        for row in receipts(quote):
            record_credit(row)
    used = sum(row["_bytes"] for row in selected)
    documents = {row["document_id"] for row in selected}
    focused_selected_sources = sum(row.get("rerank_branch") in rescue_branches for row in selected) if focused_enabled else 0
    final_source_keys = {(row["document_id"], row["source_id"]) for row in selected}
    rejected_same_chunk = sum(identifier not in credited_ids and (row["document_id"], row["source_id"]) in final_source_keys
                              for identifier, row in rejected_same_source_windows.items())
    objective, _ = set_proxy(selected)
    emit_diagnostic(diagnostic_callback, "final_window_credits", windows=[window_identity(row) for row in credited_rows],
                    final_quotes=[window_identity(row) for row in selected],
                    semantics="fully_retained_submitted_windows_only",
                    retained_submission_implies_complete_model_input=False)
    sources = []
    for index, row in enumerate(selected, 1):
        keep = {key: row[key] for key in ("document_id", "filename", "content", "ordinal", "excerpt_start", "excerpt_end", "offset_unit", "original_content_length", "sentence_boundary_preserved") if key in row}
        keep.update({"id": row["source_id"], "label": f"S{index}",
                     "truncated": row["excerpt_start"] > 0 or row["excerpt_end"] < row["original_content_length"],
                     "evidence_scope": "excerpt_only" if row["excerpt_start"] > 0 or row["excerpt_end"] < row["original_content_length"] else "source_chunk"})
        if "page" in row:
            keep["page"] = row["page"]
        sources.append(keep)
    previous_ids = {str(row.get("id")) for row in baseline_sources if isinstance(row, Mapping)}
    current_ids = {row["id"] for row in sources}
    requirement_coverage = {key: min(coverage[dimension] for _, dimension in values)
                            for key, values in dimensions.items()}
    missing = [key for key in facet_weights if key in primary_ids and requirement_coverage[key] < .12]
    missing_anchors = [key for key in anchor_weights if requirement_coverage[key] < .45]
    gaps = [{"code": "evidence_coverage_proxy", "message": "当前覆盖仅是检索相关性的代理指标，不代表关键事实已完整覆盖、引用足以支持结论或因果关系已经验证。"}]
    if not sources:
        gaps.insert(0, {"code": "insufficient_evidence", "message": "没有找到达到本次相关性与完整性要求的依据，不能用无关背景补齐。"})
    if missing:
        gaps.append({"code": "evidence_facets_uncovered", "count": len(missing), "message": "部分子问题缺少足够相关的原文片段；请勿将缺失证据补成结论。"})
    if missing_anchors:
        gaps.append({"code": "evidence_anchors_uncovered", "count": len(missing_anchors), "message": "部分显式技术要点缺少描述操作、实验或条件的原文细节，不能用只列出名称的概述代替。"})
    if named_docs - documents:
        gaps.append({"code": "documents_not_selected", "count": len(named_docs - documents), "message": "部分明确指定的资料未进入本次证据；不能据此给出完整的多文档比较。"})
    if rejected_same_chunk:
        gaps.append({"code": "complete_statement_exceeds_budget", "count": rejected_same_chunk, "message": "同一来源的部分补充窗口因间隔或预算限制未合并，不能将这些未展示细节视作已覆盖。"})
    return {"sources": sources, "gaps": gaps, "trace": {
        "strategy": "bounded_source_set_v5", "candidate_count": len(candidates),
        "selected_count": len(sources), "source_bytes": used, "max_source_bytes": budget,
        "coverage_objective": round(objective, 6), "facet_coverage": {key: round(requirement_coverage[key], 6) for key in facet_weights},
        "independent_requirement_ids": [row["id"] for row in primary],
        "scored_query_view_ids": [row["id"] for row in query_views],
        "query_view_coverage": {view["id"]: round(max((max(row["_coverage"].values(), default=0.) for row in credited_rows
                                                       if row["_actual_scored_facet"] == view["id"]), default=0.), 6)
                                for view in query_views},
        "anchor_coverage": {key: round(requirement_coverage[key], 6) for key in anchor_weights}, "uncovered_anchors": missing_anchors,
        "independent_document_requirements": sum(len(dimensions[key]) for key in primary_ids if dimensions[key][0][0] is not None),
        "uncovered_facets": missing, "named_documents_covered": len(named_docs & documents),
        "named_documents_missing": len(named_docs - documents), "same_chunk_disconnected_skipped": rejected_same_chunk,
        "added_source_ids": sorted(current_ids - previous_ids), "removed_source_ids": sorted(previous_ids - current_ids),
        "baseline_source_bytes": sum(len(str(row.get("content", "")).encode("utf-8")) for row in baseline_sources if isinstance(row, Mapping)),
        "rerank_logits_available": sum(row["rerank_logit"] is not None for row in selected),
        "rerank_missing_or_invalid": missing_rerank,
        "rerank_query_group_count": len(group_tops),
        "pair_credit": "one_scored_query_facet_not_broadcast",
        "depth_discounts": [],
        "depth_objective": {key: round(value, 6) for key, value in depth_values.items()},
        "depth_objective_semantics": "sum_rank_utility_not_selection_cap",
        "length_policy": "hard_budget_only", "rank_constant": 8,
        "redundancy_weight": .30, "minimum_rank_utility": .20,
        "document_slot_policy": "eligible_coverage_then_soft_quota" if fair_documents else "ordinary_rank_mmr",
        "document_soft_quota": document_soft_quota, "fair_coverage_selections": fair_coverage_selections,
        "document_fairness_rank_floor": .20 if fair_documents else None,
        "focused_branch_enabled": focused_enabled,
        "focused_selected_sources": focused_selected_sources,
        "focused_source_limit": rescue_source_limit if focused_enabled else 0,
        "quoted_audit_branch_enabled": audit_enabled,
        "quoted_audit_selected_sources": sum(row.get("rerank_branch") == "quoted_audit" for row in selected),
        "quoted_audit_requirement_credit": False,
        "quoted_audit_common_sources": common_count,
        "quoted_audit_eligible_sources": len(audit_eligible_ids),
        "quoted_audit_pairing_policy": "complete_same_window_competition_rank_positive_gain_upper_half",
        "focused_reservation_slots": 1 if focused_enabled else 0,
        "focused_rank_weight": .75 if focused_enabled else None,
        "focused_merge_extensions": focused_merge_extensions,
        "focused_merge_extension_limit": 2 if focused_enabled else 0,
        "complete_unit_partial_extensions": partial_context_extensions,
        "source_set_replacements": swaps, "source_set_swap_evaluations": swap_evaluations,
        "source_set_max_passes": 2,
        "credit_geometry": "fully_retained_submitted_windows_only",
        "retained_submission_implies_complete_model_input": False,
        "reranker_input_completeness": "requires_per_pair_reranker_receipt_not_inferred_from_window_geometry",
        "query_alias_nominated_credits": sum(bool(row.get("_alias_nomination_support")) and not row["_translated_scoring_view"] for row in credited_rows),
        "query_alias_credit_semantics": "scored_literal_rank_times_alias_nomination_not_verified_support",
        "translated_query_view_credits": sum(row["_translated_scoring_view"] for row in credited_rows),
        "translated_query_view_semantics": "actual_alias_pair_rank_not_literal_pair_or_verified_support",
        "anchor_coverage_semantics": "lexical_detail_proxy_not_scored_coverage",
        "selection_proxy_is_verified_support": False,
        "score_semantics": "relative_relevance_not_probability_or_entailment", "provider_calls": 0, "llm_tokens": 0,
    }}
