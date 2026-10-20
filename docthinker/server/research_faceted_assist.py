"""Bounded local evidence retrieval, reranking and coverage packing.

No facts, proofs or answer drafts are generated here. Facets come from the
question, all returned quotes are continuous live source spans. A model score
only ranks relevance; it does not verify truth or completeness.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import math
import re
import time

from .research_prompt_budget import empty_assistance
from .research_query_facets import MAX_QUERY_BYTES, build_query_facets, literal_requirement_kind, valid_quoted_audit_view
from .research_query_translation import _lexical_guard, _NUMBER, _NEG_EN, _NEG_ZH
from .research_evidence_selection import prepare_evidence_candidates, select_evidence, _source_round_robin
from .research_selection_diagnostics import emit_diagnostic, window_identity


def _spread(values, limit):
    if len(values) <= limit:
        return values
    if limit <= 0:
        return []
    if limit == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (limit - 1))] for index in range(limit)]


def _query_nomination_aliases(literal, translated):
    """Conservatively associate unique query roles, not assert equivalence.

    A translated verification/limitation clause can locate a source span the
    literal multilingual retriever missed. It never creates a new demand or
    receives an extra scoring quota. The literal query must still be scored
    against the complete nominated window. Ambiguous roles and surface guard
    failures keep the views independent. Passing guards is not a semantic test.
    """
    kinds = {"verification", "limitation", "experiment", "comparison"}
    originals = [row for row in literal if row.get("kind") == "clause"]
    aliases = [row for row in translated if not row.get("translation_full_query")]
    result = {row["id"]: row for row in translated}
    for kind in sorted(kinds):
        left = [row for row in originals if literal_requirement_kind(row["text"]) == kind]
        right = [row for row in aliases if literal_requirement_kind(row["text"]) == kind]
        if len(left) != 1 or len(right) != 1:
            continue
        source, alias = left[0], right[0]
        a, b = source["text"], alias["text"]
        if (_lexical_guard(a, b)
                or Counter(_NUMBER.findall(a)) != Counter(_NUMBER.findall(b))
                or len(_NEG_ZH.findall(a)) + len(_NEG_EN.findall(a)) != len(_NEG_EN.findall(b))
                or set(source.get("document_ids", ())) != set(alias.get("document_ids", ()))):
            continue
        result[alias["id"]] = {**alias, "query_alias_of": source["id"],
                               "alias_nomination": True,
                               "alias_alignment": "unique_query_only_requirement_kind",
                               "semantic_equivalence_verified": False}
    return [result[row["id"]] for row in translated]


def _with_query_translation(question, chunks, plan, translator, session_id):
    """Add bounded local query variants, never source facts or answer prose.

    The original facet remains unchanged and first, including every retained
    literal anchor and explicitly named document. Translation is fallible and
    can influence retrieval only; it is not evidence or an answer expansion.
    """
    receipt = {"status": "disabled", "applied": False, "derived_translation": False,
               "max_english_facets": 4, "english_facet_count": 0,
               "max_total_facets": 8, "total_facet_count": len(plan["facets"]),
               "provider_calls": 0, "llm_tokens": 0, "query_only": True}
    if translator is None:
        return plan, receipt
    try:
        translated = translator.translate(question, session_id=session_id)
    except Exception:
        receipt["status"] = "inference_unavailable"
        return plan, receipt
    if not isinstance(translated, Mapping):
        receipt["status"] = "invalid_output"
        return plan, receipt
    translation_trace = translated.get("trace")
    translation_trace = translation_trace if isinstance(translation_trace, Mapping) else {}
    status = translation_trace.get("status")
    allowed_statuses = {"ready", "busy", "disabled", "unavailable", "model_unavailable", "assets_missing",
                        "unsupported_language", "inference_unavailable", "invalid_input", "not_needed",
                        "no_chinese", "input_limit", "source_limit", "output_limit", "lexical_guard",
                        "translation_error", "invalid_translation", "placeholder_collision",
                        "placeholder_limit", "placeholder_guard"}
    receipt["status"] = status if isinstance(status, str) and status in allowed_statuses else "unavailable"
    # Only bounded scalar diagnostics are copied. Exception messages, file
    # paths, model inputs and arbitrary translator metadata stay private.
    for name in ("inference_ms", "total_ms"):
        value = translation_trace.get(name)
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 3_600_000:
            receipt[name] = round(value, 3)
    if type(translation_trace.get("cache_hit")) is bool:
        receipt["cache_hit"] = translation_trace["cache_hit"]
    for name in ("manifest_fingerprint", "model_sha256"):
        value = translation_trace.get(name)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            receipt[name] = value
    if isinstance(translation_trace.get("guard_reasons"), list):
        receipt["guard_reasons"] = [name for name in translation_trace["guard_reasons"]
                                   if name in {"missing_numbers", "missing_terms", "missing_negation", "missing_invariance"}][:4]
    value = translated.get("text")
    if status != "ready":
        return plan, receipt
    if not isinstance(value, str) or not value.strip():
        receipt["status"] = "invalid_output"
        return plan, receipt
    value = value.strip()
    try:
        if len(value.encode("utf-8")) > MAX_QUERY_BYTES:
            receipt["status"] = "invalid_output"
            return plan, receipt
    except UnicodeEncodeError:
        receipt["status"] = "invalid_output"
        return plan, receipt
    if value == question.strip():
        receipt["status"] = "unchanged"
        return plan, receipt
    translated_plan = build_query_facets(value, chunks)
    originals = plan["facets"]
    named_docs = set(originals[0].get("document_ids", ()))
    seen = {row["text"].casefold().strip() for row in originals}
    english = []
    for row in translated_plan["facets"]:
        if row["kind"] not in {"original", "clause"} or row["text"].casefold().strip() in seen:
            continue
        seen.add(row["text"].casefold().strip())
        scope = named_docs & set(row.get("document_ids", ()))
        facet = {"id": "qt_" + hashlib.sha256(row["text"].encode("utf-8")).hexdigest()[:12],
                 "text": row["text"], "kind": "clause", "weight": 1.0,
                 "roles": list(row.get("roles", ())), "derived_translation": True,
                 "query_only": True, "translation_full_query": row["kind"] == "original"}
        # A translated name cannot introduce a document that the literal
        # question did not explicitly identify. Missing names retain that
        # original scope instead of searching an unrelated new collection.
        if named_docs:
            facet["document_ids"] = sorted(scope or named_docs)
        english.append(facet)
    all_english = english
    english = _spread(english, 4)
    if not english:
        receipt["status"] = "unchanged"
        return plan, receipt
    # Independent literal demands precede translated aliases. Four translated
    # queries must not displace a late condition/verification in a compound ask.
    clauses = [row for row in originals[1:] if row["kind"] == "clause"]
    other_literal = [row for row in originals[1:] if row["kind"] == "technical_phrase"]
    literal = _spread(clauses, 4)
    literal += _spread(other_literal, min(4 - len(literal), len(other_literal)))
    english = _spread(english, min(4, 7 - len(literal)))
    remaining = 8 - 1 - len(english)
    if len(literal) < remaining:
        literal += [row for row in originals[1:] if row["kind"] == "document"][:remaining - len(literal)]
    if not receipt.get("guard_reasons") and not _lexical_guard(question, value):
        # Check uniqueness before the eight-view quota drops any aliases.
        mapped = {row["id"]: row for row in _query_nomination_aliases(originals[1:], all_english)}
        literal_ids = {row["id"] for row in literal}
        english = [mapped[row["id"]] if mapped[row["id"]].get("query_alias_of") in literal_ids
                   else row for row in english]
    combined = {**plan, "facets": [originals[0], *literal, *english]}
    combined["trace"] = {**plan["trace"], "facet_count": len(combined["facets"]),
                         "literal_facet_count": 1 + len(literal),
                         "derived_translation_facet_count": len(english),
                         "query_translation_only": True}
    receipt.update(status="ready", applied=True, derived_translation=True,
                   english_facet_count=len(english), total_facet_count=len(combined["facets"]),
                   nomination_alias_count=sum(bool(row.get("query_alias_of")) for row in english),
                   alias_semantic_equivalence_verified=False)
    return combined, receipt


_QUESTION_NEED = re.compile(r"\b(?:how|why|what|which|when|where|whether)\b|如何|怎么|怎样|为什么|为何|什么|哪些|是否|哪里|何处|何时|何种", re.I)
_ANSWER_CONSTRAINT = re.compile(
    r"^(?:do not|don't|avoid|write|format|answer in)\b|"
    r"^(?:不要|不得|别|避免|请用|请按)|"
    r"^(?:这个|这种|该|上述|这些)(?:类比|比较|结论|说法|推理)|"
    r"^(?:this|that|these|those)\s+(?:analogy|analogies|comparison|conclusion|claim)\b|"
    r"\b(?:format|formatting|citations?|word count|writing style|output language)\b|"
    r"排版|引用格式|字数|文风|用什么语言", re.I,
)
_NEED_COMMAND = re.compile(r"^(?:说明|解释|讨论|请说明|请解释|请讨论)\s*|^(?:please\s+)?(?:describe|explain|discuss)\s+", re.I)


def _requirement_terms(text):
    # Only for alias deduplication and budget routing, never relevance credit.
    value = _NEED_COMMAND.sub("", text.casefold(), count=1)
    value = _QUESTION_NEED.sub(" ", value)
    values = re.findall(r"[a-z0-9_-]+|[\u3400-\u9fff]+", value)
    stop = {"the", "a", "an", "does", "did", "is", "are", "it", "its", "of", "for", "and", "or", "to", "in"}
    result = set()
    for item in values:
        if item in stop:
            continue
        if re.search(r"[\u3400-\u9fff]", item):
            result.update(item[index:index + 2] for index in range(max(1, len(item) - 1)))
        else:
            result.add(item)
    return result


def _content_requirements(facets):
    original = next((row for row in facets if row.get("kind") == "original"), {})
    if len(set(original.get("document_ids", ()))) > 1:
        return []  # Explicit document comparisons already own scoped branches.
    needs, fingerprints = [], []
    for row in facets:
        if (row.get("kind") != "clause" or row.get("derived_translation")
                or _ANSWER_CONSTRAINT.search(_NEED_COMMAND.sub("", row["text"], count=1))):
            continue
        kind = literal_requirement_kind(row["text"])
        if kind is None:
            continue
        tokens = _requirement_terms(row["text"])
        if not tokens or any(len(tokens & prior) / max(1, len(tokens | prior)) >= .85
                             for prior in fingerprints):
            continue
        needs.append({**row, "requirement_kind": kind})
        fingerprints.append(tokens)
    return needs if len(needs) >= 2 else []


def _requirement_priority(row):
    # Whether something was actually demonstrated and under what restrictions
    # are easy to lose behind broad method descriptions. This is a transparent
    # query-only budget policy, not evidence of correctness or a source oracle.
    return ({"verification": 3, "limitation": 2, "experiment": 2, "comparison": 1}.get(row.get("requirement_kind"), 0),
            len(_QUESTION_NEED.findall(row["text"])), len(row["text"]))


def _focused_requirement(facets):
    """Compatibility view of the highest-priority independent literal demand."""
    needs = _content_requirements(facets)
    return max(needs, key=_requirement_priority) if needs else None


def _focused_scoring_view(need, facets, nominees):
    """One query view for the entire focus group, never an extra alias quota.

    Script matching is deliberately conservative: every nominated window must
    carry Latin text with English grammatical markers and no Han. Mixed/unknown material
    retains the literal query. This picks a scoring language, not the best
    answer or a claim that translation preserved the meaning.
    """
    aliases = [row for row in facets if row.get("derived_translation") and row.get("query_only")
               and not row.get("translation_full_query") and row.get("query_alias_of") == need["id"]]
    if len(aliases) != 1 or not re.search(r"[\u3400-\u9fff]", need["text"]):
        return need
    alias = aliases[0]
    if (re.search(r"[\u3400-\u9fff]", alias["text"])
            or len(re.findall(r"[A-Za-z]", alias["text"])) < 16
            or not nominees
            or any(not _english_scoring_material(row.get("content")) for row in nominees)):
        return need
    return alias


def _english_scoring_material(value):
    # Bounded surface heuristic, not a language model. Latin script alone would
    # wrongly treat French/Spanish and citation-only tables as English evidence.
    if (not isinstance(value, str) or re.search(r"[\u3400-\u9fff]", value)
            or len(re.findall(r"[A-Za-z]", value)) < 32):
        return False
    grammar = {"the", "a", "an", "and", "of", "to", "in", "for", "with", "on", "by", "from",
               "this", "that", "these", "those", "is", "are", "was", "were", "it", "we", "our",
               "which", "while", "as", "has", "have", "can", "be", "been"}
    words = re.findall(r"[a-z]+", value.casefold())
    matches = [word for word in words if word in grammar]
    return len(set(matches)) >= 3 and len(matches) / max(1, len(words)) >= .08


def _focused_nominations(facets, candidates):
    """Choose at most two complementary needs inside one 24-pair rescue budget.

    Positive per-facet retrieval support nominates a pair, never earns a slot.
    Source overlap with full-query nominees is only a tie-break for deciding
    which literal requirement gets scored. No unscored alias obtains credit.
    """
    needs = _content_requirements(facets)
    if not needs:
        return []
    full_sources = {row.get("source_id") for row in candidates[:24]}
    profiles = []
    for need in needs:
        scope = set(need.get("document_ids", ()))
        aliases = {row["id"] for row in facets if row.get("derived_translation")
                   and row.get("query_only") and not row.get("translation_full_query")
                   and row.get("query_alias_of") == need["id"]}
        eligible = []
        for row in candidates:
            if ((scope and row.get("document_id") not in scope)
                    or "source_id" not in row or "document_id" not in row):
                continue
            scores = row.get("facet_scores", {})
            supported = {key: value for key, value in scores.items()
                         if key in {need["id"], *aliases} and type(value) in (int, float)
                         and math.isfinite(value) and value > 0}
            if not supported:
                continue
            if need["id"] in supported:
                eligible.append({**row, "retrieval_support": supported[need["id"]]})
            else:
                # Preserve literal facet_scores (including absence). The
                # selector checks the actual scored query and alias receipts;
                # no translated similarity is forged as a literal hit.
                eligible.append({**row, "retrieval_support": max(supported.values()),
                                 "nomination_facet_ids": sorted(supported),
                                 "nomination_alignment": "unique_query_only_requirement_kind"})
        eligible.sort(key=lambda row: (-row["retrieval_support"], row["id"]))
        nominees = _source_round_robin(eligible)[:24]
        if len(nominees) < 4:
            continue
        distinct = {row["source_id"] for row in nominees[:12]}
        novelty = len(distinct - full_sources) / max(1, len(distinct))
        profiles.append((need, nominees, novelty))
    profiles.sort(key=lambda item: (-_requirement_priority(item[0])[0], -item[2],
                                    -_requirement_priority(item[0])[1], -_requirement_priority(item[0])[2], item[0]["id"]))
    # Keep at least one content demand represented only by the full question;
    # two-clause asks retain their prior single-rescue shape.
    profiles = profiles[:min(2, len(needs) - 1)]
    selected = []
    for index in range(24):
        progressed = False
        for need, nominees, _ in profiles:
            if index < len(nominees):
                selected.append((need, nominees[index]))
                progressed = True
                if len(selected) == 24:
                    break
        if len(selected) == 24:
            break
        if not progressed:
            break
    # Decide one language view only after the joint pair budget is allocated.
    # Discarded proposals (e.g. an unscored figure label) must not determine the
    # language of a different group of windows that will actually be scored.
    group_rows = {need["id"]: [row for focus, row in selected if focus["id"] == need["id"]]
                  for need, _, _ in profiles}
    views = {need["id"]: _focused_scoring_view(need, facets, group_rows[need["id"]]) for need, _, _ in profiles}
    return [({**need, "_scoring_view": views[need["id"]]}, row) for need, row in selected]


def _rerank_pairs(question, candidates, facets):
    """48 total pairs: common-query ranking plus bounded literal-need rescue.

    On multi-need questions, reserve up to 24 pairs for up to two requirements
    and at least 24 for the full query. The selector permits only one reserved
    rescue source and at most two focused sources, not a champion per alias.
    Scoring the same window twice consumes two pairs; no hidden extra inference.
    """
    candidates = candidates[:48]
    original = next((facet for facet in facets if facet.get("kind") == "original"), None)
    facet = next((facet for facet in facets if facet.get("translation_full_query")), original)
    query = facet["text"] if facet else question
    audits = [value for value in facets if original and valid_quoted_audit_view(value, original)]
    # Auditing never edits the assertion (even its genre words) in the full
    # scoring view. The optional outside request is a separately paid pair.
    if not audits:
        query = re.sub(r"\b(?:related work|literature review)\b", " ", query, flags=re.I).strip() or question
    result = []
    retrieval_views = {value["id"] for value in facets if value.get("kind") not in {"document", "query_view"}}
    named = {doc for value in facets if value.get("kind") == "original" for doc in value.get("document_ids", ())}
    document_names = {value["document_ids"][0]: value["text"] for value in facets
                      if value.get("kind") == "document" and len(value.get("document_ids", ())) == 1}
    # An explicit comparison has independent document intents. Peer names can
    # dominate a small cross-encoder even for another paper's own mechanism.
    # Remove ONLY the other literal names on that document's scoring branch;
    # retain all substantive demands/conditions, and the full answer question.
    scoped_comparison = len(named) > 1 and named <= document_names.keys()
    for row in candidates:
        scores = dict(row.get("facet_scores", {}))
        support = max((value for key, value in scores.items() if key in retrieval_views
                       and type(value) in (float, int) and math.isfinite(value)), default=0.)
        if facet:
            # A match through one real retrieval view supports nomination to
            # the full query, not completeness of all its component demands.
            scores[facet["id"]] = support
        scoped_query = query
        if scoped_comparison and row.get("document_id") in named:
            for doc in sorted(named - {row["document_id"]}):
                scoped_query = re.sub(r"(?<!\w)" + re.escape(document_names[doc]) + r"(?!\w)",
                                      " ", scoped_query, flags=re.I)
        result.append({**row, "query": scoped_query, "document_scoped_query": scoped_query != query,
                       "facet_scores": scores, "retrieval_support": support,
                       "rerank_facet_id": facet["id"] if facet else "original",
                       "rerank_query_id": "rq_" + hashlib.sha256(scoped_query.encode("utf-8")).hexdigest()[:16]})
    rescue = _focused_nominations(facets, candidates)
    if not rescue and len(audits) == 1 and facet is original:
        view = audits[0]
        scope = set(view.get("document_ids", ()))
        nominees = []
        for row in candidates:
            support = row.get("facet_scores", {}).get(view["id"])
            if (isinstance(row.get("source_id"), str) and isinstance(row.get("document_id"), str)
                    and (not scope or row["document_id"] in scope)
                    and type(support) in (int, float) and math.isfinite(support) and support > 0):
                nominees.append({**row, "retrieval_support": support})
        nominees.sort(key=lambda row: (-row["retrieval_support"], row["id"]))
        nominees = _source_round_robin(nominees)[:12]
        if len(nominees) >= 4:
            # A fixed 36/12 ceiling; unused view capacity is not extra full
            # inference. No independent requirement or translation is forged.
            full = [{**row, "rerank_branch": "full"} for row in result[:36]]
            audit_pairs = [{**row, "id": row["id"] + "::" + view["id"],
                            "query": view["text"], "rerank_branch": "quoted_audit",
                            "rerank_facet_id": view["id"], "rerank_view_of": original["id"],
                            "rerank_view": "quoted_claim_audit",
                            "rerank_query_id": "rq_" + hashlib.sha256(view["text"].encode()).hexdigest()[:16]}
                           for row in nominees]
            return full + audit_pairs
    if not rescue:
        return result
    full = [{**row, "rerank_branch": "full"} for row in result[:48 - len(rescue)]]
    focused = []
    for focus, row in rescue:
        view = focus.get("_scoring_view", focus)
        focused.append({**row, "id": row["id"] + "::" + view["id"],
                        "query": view["text"], "rerank_branch": "focused",
                        "rerank_facet_id": view["id"], "rerank_view_of": focus["id"],
                        "rerank_view": "guarded_alias" if view.get("derived_translation") else "literal",
                        "rerank_query_id": "rq_" + hashlib.sha256(view["text"].encode("utf-8")).hexdigest()[:16],
                        "retrieval_support": row["retrieval_support"]})
    return full + focused


def build_faceted_assistance(question, baseline_sources, chunks, semantic_retriever,
                             reranker, *, session_id, primary_semantic=None, query_translator=None,
                             diagnostic_callback=None):
    started = time.perf_counter()
    result = empty_assistance(baseline_sources, reason="evidence_selection_unavailable")
    trace = result["trace"]
    trace.update(algorithm="faceted_evidence_selection_v5", model_calls=0, provider_calls=0,
                 llm_tokens=0, memory_writes=0, assist_bytes=0,
                 retrieval_reselection=False, semantic_entailment_verified=False)
    plan = build_query_facets(question, chunks)
    trace["query_plan"] = plan["trace"]
    trace["facet_count"] = len(plan["facets"])
    if not chunks or not plan["facets"]:
        trace["skipped_reason"] = "no_evidence_query"
        return result
    plan, translation_receipt = _with_query_translation(question, chunks, plan, query_translator, session_id)
    emit_diagnostic(diagnostic_callback, "query_plan", facets=plan["facets"], translation=translation_receipt)
    trace["local_query_translation"] = translation_receipt
    trace["query_plan"] = plan["trace"]
    trace["facet_count"] = len(plan["facets"])
    semantic_results = {}
    statuses = Counter()
    semantic_started = time.perf_counter()
    for facet in plan["facets"]:
        if facet["kind"] == "original" and primary_semantic is not None:
            found = primary_semantic
        else:
            # Explicitly named documents scope their own branch; one popular
            # source cannot consume the candidate quota for every document.
            wanted = set(facet.get("document_ids", []))
            scoped = [row for row in chunks if row.get("document_id") in wanted] if wanted else chunks
            try:
                found = semantic_retriever.semantic_scores(facet["text"], scoped, session_id=session_id)
            except Exception:
                found = {"scores": {}, "spans": {}, "trace": {"status": "inference_unavailable"}}
        status = found.get("trace", {}).get("status", "unavailable")
        statuses[status] += 1
        if status == "ready":
            semantic_results[facet["id"]] = found
        emit_diagnostic(diagnostic_callback, "facet_semantic", facet_id=facet["id"],
                        scores=found.get("scores", {}), spans=found.get("spans", {}), status=status)
    trace["facet_semantic_statuses"] = dict(statuses)
    trace["facet_retrieval_ms"] = round((time.perf_counter() - semantic_started) * 1000, 3)
    pool = prepare_evidence_candidates(question, chunks, baseline_sources, plan, semantic_results,
                                       diagnostic_callback=diagnostic_callback)
    candidates = pool["candidates"]
    trace["candidate_pool"] = pool["trace"]
    trace["candidate_count"] = len(candidates)
    if not candidates:
        # A completed scan rejecting every passage is not a model outage.
        # Reinstating the baseline here would reintroduce filtered noise.
        result["sources"] = []
        result["reasoning"].update(status="evidence_selected", gaps=[{
            "code": "insufficient_evidence",
            "message": "没有找到符合本次问题的可用原文，不用无关背景填充。"}])
        trace.update(skipped_reason="no_eligible_evidence_candidates",
                     retrieval_reselection=bool(baseline_sources),
                     baseline_source_count=len(baseline_sources), selected_source_count=0,
                     source_bytes=0, total_ms=round((time.perf_counter() - started) * 1000, 3))
        return result
    # A requested writing genre is not a paper section to retrieve. Preserve
    # the actual content demand, including all negations and conditions.
    candidates = _rerank_pairs(question, candidates, plan["facets"])
    trace["pair_allocation"] = {
        "max_pairs": 48, "pair_count": len(candidates),
        "full_pairs": sum(row.get("rerank_branch") not in {"focused", "quoted_audit"} for row in candidates),
        "focused_pairs": sum(row.get("rerank_branch") == "focused" for row in candidates),
        "quoted_audit_pairs": sum(row.get("rerank_branch") == "quoted_audit" for row in candidates),
        "quoted_audit_source_cap": 1, "quoted_audit_creates_requirement": False,
        "focused_query_count": len({row["rerank_query_id"] for row in candidates if row.get("rerank_branch") == "focused"}),
        "alias_nominated_pairs": sum(bool(row.get("nomination_facet_ids")) for row in candidates),
        "alias_scored_pairs": sum(row.get("rerank_view") == "guarded_alias" for row in candidates),
        "max_focused_queries": 2,
        "distinct_windows": len({(row["source_id"], row["excerpt_start"], row["excerpt_end"]) for row in candidates}),
        "focused_source_cap": 2, "focused_reserved_slots": 1,
        "document_scoped_pairs": sum(bool(row.get("document_scoped_query")) for row in candidates),
        "source_derived_queries": False, "additional_provider_calls": 0,
    }
    original = next((facet for facet in plan["facets"] if facet.get("kind") == "original"), {})
    audit_query = any(valid_quoted_audit_view(facet, original) for facet in plan["facets"])
    focus = question if audit_query else re.sub(r"\b(?:related work|literature review)\b", " ", question, flags=re.I).strip() or question
    try:
        if callable(getattr(reranker, "rank_pairs", None)):
            ranked = reranker.rank_pairs(candidates, session_id=session_id)
        else:
            # Compatibility for injected old scorers; their logits share one
            # query and must not be normalized as independently scored facets.
            original = next((facet["id"] for facet in plan["facets"] if facet.get("kind") == "original"), "original")
            query_id = "rq_" + hashlib.sha256(focus.encode("utf-8")).hexdigest()[:16]
            # Old injected scorers cannot score distinct requirements. Keep
            # the original pool instead of silently crediting an unscored branch.
            candidates = [{**row, "query": focus, "rerank_facet_id": original,
                           "rerank_query_id": query_id,
                           "facet_scores": {**row.get("facet_scores", {}), original:
                               max((value for key, value in row.get("facet_scores", {}).items()
                                    if key in {facet["id"] for facet in plan["facets"] if facet["kind"] not in {"document", "query_view"}}
                                    and type(value) in (int, float) and math.isfinite(value)), default=0.)}}
                          for row in pool["candidates"]]
            trace["pair_allocation"].update(pair_count=len(candidates), full_pairs=len(candidates),
                                             focused_pairs=0, compatibility_shared_query=True,
                                             focused_query_count=0,
                                             alias_nominated_pairs=0,
                                             alias_scored_pairs=0,
                                             quoted_audit_pairs=0,
                                             document_scoped_pairs=0,
                                             distinct_windows=len({(row["source_id"], row["excerpt_start"], row["excerpt_end"])
                                                                   for row in candidates}))
            ranked = reranker.rank(focus, candidates, session_id=session_id)
    except Exception:
        ranked = {"scores": {}, "trace": {"status": "inference_unavailable"}}
    trace["reranker"] = ranked.get("trace", {})
    input_receipts = ranked.get("input_receipts", {})
    if trace["reranker"].get("status") != "ready" or not isinstance(input_receipts, Mapping):
        input_receipts = {}
    for candidate in candidates:
        emit_diagnostic(diagnostic_callback, "rerank_result", window=window_identity(candidate),
                        query=candidate.get("query"), facet_id=candidate.get("rerank_facet_id"),
                        query_id=candidate.get("rerank_query_id"),
                        rerank_view=candidate.get("rerank_view"), rerank_view_of=candidate.get("rerank_view_of"),
                        nomination_facet_ids=candidate.get("nomination_facet_ids", []),
                        nomination_alignment=candidate.get("nomination_alignment"),
                        input_receipt=input_receipts.get(candidate["id"], {}),
                        logit=ranked.get("scores", {}).get(candidate["id"]))
    if trace["reranker"].get("status") != "ready" or not ranked.get("scores"):
        trace["skipped_reason"] = "reranker_unavailable"
        trace["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result
    selection = select_evidence(candidates, ranked["scores"], plan, baseline_sources,
                                diagnostic_callback=diagnostic_callback, input_receipts=input_receipts)
    trace["selection"] = selection["trace"]
    sources = selection["sources"]
    emit_diagnostic(diagnostic_callback, "sources_final", sources=[{key: row.get(key) for key in
                    ("id", "document_id", "page", "ordinal", "excerpt_start", "excerpt_end")} for row in sources])
    before = [(row["id"], row["content"]) for row in baseline_sources]
    after = [(row["id"], row["content"]) for row in sources]
    trace.update(retrieval_reselection=before != after, skipped_reason=None,
                 source_bytes=sum(len(row["content"].encode()) for row in sources),
                 baseline_source_count=len(baseline_sources), selected_source_count=len(sources),
                 algorithm_modes=["evidence_reselection"], support_item_count=0,
                 total_ms=round((time.perf_counter() - started) * 1000, 3))
    result["sources"] = sources
    result["reasoning"].update(status="evidence_selected", gaps=selection.get("gaps", []))
    # Selection is expressed by the actual quotes, not by a second token-heavy
    # organization index or unsupported synthesized statements.
    return result
