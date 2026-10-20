"""Offline paired accounting and blinded review packets for real-paper A/B runs.

No credentials, provider clients, model calls or synthetic usage are involved.
Only fully completed, same-model pairs enter primary comparisons. Run again as
answers arrive; existing blind candidate IDs and their presentation order stay
stable. Reviewers receive blind/batch_* only, never private_mapping.json.
An optional --scores-dir performs a second pass over independent blind reviews.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import statistics
import sys

if __package__:
    from .research_benchmark_contract import DELIVERY_GATE_VERSION, delivery_failure_reason
else:
    from research_benchmark_contract import DELIVERY_GATE_VERSION, delivery_failure_reason

MODEL = "step-5-preview"
SCHEMA = "research-real-ab-analysis-v1"
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")
LATENCY_FIELDS = ("pre_model_latency_ms", "model_latency_ms", "post_model_latency_ms", "total_answer_latency_ms")
SETUP_LATENCY_FIELDS = ("retrieval_setup_latency_ms", "per_arm_setup_latency_ms", "upfront_setup_latency_ms")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value, *, private=False):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".analysis-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def number(value):
    return value if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None


def usage(record, name):
    provider = record.get("provider", {})
    raw = provider.get("usage", {})
    if type(raw.get(name)) is int and raw[name] >= 0:
        return raw[name]
    for key in ("prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details"):
        nested = raw.get(key, {})
        if isinstance(nested, dict) and type(nested.get(name)) is int and nested[name] >= 0:
            return nested[name]
    # These flattened values are also supplied by the runner, never inferred.
    if name in {"cached_tokens", "reasoning_tokens"} and type(provider.get(name)) is int and provider[name] >= 0:
        return provider[name]
    return None


def latency(record, name):
    return number(record.get(name) if name in {"post_model_latency_ms", "total_answer_latency_ms", *SETUP_LATENCY_FIELDS}
                  else record.get("provider", {}).get(name))


def support_kind(record):
    """Distinguish extractive organization from formal, conditional operators.

    Source organization adds/arranges actual quotes; it is not a proof. Formal
    auxiliary outputs also do not establish that their premises are true.
    Legacy untyped support stays unclassified rather than becoming a proof.
    """
    conclusions = record.get("result", {}).get("reasoning", {}).get("conclusions", [])
    counts = Counter()
    for item in conclusions if isinstance(conclusions, list) else []:
        if not isinstance(item, dict):
            counts["unclassified"] += 1
        elif item.get("kind", item.get("type")) == "source_evidence" or item.get("epistemic_status") == "source_organization":
            counts["source_organization"] += 1
        elif item.get("kind", item.get("type")) in {"deduced", "analogy_hypothesis", "inductive_candidate"}:
            counts["formal_auxiliary"] += 1
        else:
            counts["unclassified"] += 1
    if not counts:
        category = "unclassified_support" if record.get("assistance", {}).get("has_actual_support") else "none"
    elif counts["unclassified"]:
        category = "mixed_typed_and_unclassified_support" if len(counts) > 1 else "unclassified_support"
    elif counts["source_organization"] and counts["formal_auxiliary"]:
        category = "mixed_source_and_formal"
    else:
        category = "source_organization" if counts["source_organization"] else "formal_auxiliary"
    return {"category": category, "item_counts": dict(counts)}


def semantic_retrieval_status(record):
    value = record.get("result", {}).get("reasoning", {}).get("trace", {}).get("semantic_retrieval")
    if not isinstance(value, dict) or not isinstance(value.get("status"), str) or not value["status"]:
        return "missing"
    status = value["status"]
    return status if status in {"ready", "disabled", "empty_corpus", "busy", "timed_out", "closed",
                                "model_not_installed", "model_unavailable", "inference_unavailable", "failed"} else "other"


def semantic_status_counts(records):
    counts = Counter(semantic_retrieval_status(record) for record in records)
    return {"record_count": len(records), "status_counts": dict(counts),
            "available_count": counts["ready"], "busy_count": counts["busy"],
            "timed_out_count": counts["timed_out"], "missing_count": counts["missing"],
            "availability_definition": "ready means the recorded retrieval operation ran; it does not establish relevant evidence or answer quality"}


def eligibility(record, *, expected_model=MODEL):
    if record.get("status") != "completed":
        return record.get("status", "unknown_answer_status")
    provider = record.get("provider", {})
    if provider.get("status") != "ok":
        return "provider_not_ok"
    actual_model = provider.get("model_actual")
    if not isinstance(actual_model, str) or not actual_model or actual_model != expected_model:
        return "unexpected_actual_model"
    if provider.get("finish_reason") != "stop":
        return "incomplete_finish_reason"
    delivery_failure = delivery_failure_reason(record.get('result'))
    if delivery_failure:
        return delivery_failure
    return None


def load_records(run_directory):
    records, errors = {}, []
    for path in sorted((Path(run_directory) / "answers").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or not isinstance(record.get("question_id"), str)
                    or type(record.get("repetition")) is not int or record["repetition"] < 1
                    or type(record.get("assistance_enabled")) is not bool):
                raise ValueError("invalid_record_identity")
            identity = (record["question_id"], record["repetition"], record["assistance_enabled"])
            if identity in records:
                raise ValueError("duplicate_record_identity")
            records[identity] = record
        except (ValueError, TypeError, OSError) as error:
            errors.append({"file": path.name, "error": str(error) if str(error) in {
                "invalid_record_identity", "duplicate_record_identity"} else "invalid_answer_artifact"})
    return records, errors


def exposure(off, on):
    before = off.get("provider", {}).get("prompt_digest")
    after = on.get("provider", {}).get("prompt_digest")
    if isinstance(before, str) and before and before == after:
        return "identical_prompt"
    assistance = on.get("assistance", {})
    reasoning = on.get("result", {}).get("reasoning", {})
    trace = reasoning.get("trace", {})
    if on.get('contrast') == 'insight':
        insight = trace.get('insight_runtime', {})
        context_added = insight.get('analysis_included') is True
        evidence_added = bool(insight.get('supplement', {}).get('kept_source_ids')) and (
            off.get('source_content_digest') and on.get('source_content_digest')
            and off['source_content_digest'] != on['source_content_digest'])
        if context_added or evidence_added:
            return ('insight_context_and_evidence' if context_added and evidence_added else
                    'insight_context' if context_added else 'insight_evidence')
    # A reselection changes actual quotes, not proof/support objects. Inspect
    # the saved source packs rather than trusting an activation flag or digest.
    if trace.get("retrieval_reselection") is True:
        def source_pack(record):
            rows = record.get("result", {}).get("sources")
            if not isinstance(rows, list) or any(not isinstance(row, dict) or
                    not isinstance(row.get("content"), str) for row in rows):
                return None
            return [{key: row.get(key) for key in ("label", "paper_id", "page", "content",
                                                   "excerpt_start", "excerpt_end")} for row in rows]
        old_sources, new_sources = source_pack(off), source_pack(on)
        if old_sources is not None and new_sources is not None and old_sources != new_sources:
            return "evidence_reselection"
    if assistance.get("has_actual_support") or reasoning.get("conclusions"):
        kind = support_kind(on)["category"]
        # Preserve the legacy untyped label without asserting what algorithm
        # produced it. Typed source organization and formal work are separate.
        return "actual_support" if kind == "unclassified_support" else kind
    if number(trace.get("assist_bytes")) not in {None, 0} or number(trace.get("context_budget", {}).get("assist_bytes")) not in {None, 0}:
        return "hint_only"
    return "changed_without_visible_assist" if before and after else "unclassified"


def paired_stat(pairs, getter, field):
    values = [(getter(row["off"], field), getter(row["on"], field)) for row in pairs]
    available = [(off, on) for off, on in values if off is not None and on is not None]
    off = [row[0] for row in available]
    on = [row[1] for row in available]
    differences = [right - left for left, right in available]
    return {"pair_count": len(pairs), "measured_pair_count": len(available),
            "unavailable_pair_count": len(pairs) - len(available),
            "off_mean": statistics.mean(off) if off else None,
            "on_mean": statistics.mean(on) if on else None,
            "paired_delta_mean": statistics.mean(differences) if differences else None,
            "off_median": statistics.median(off) if off else None,
            "on_median": statistics.median(on) if on else None,
            "paired_delta_median": statistics.median(differences) if differences else None,
            "delta_direction": "on_minus_off"}


def summarize_pairs(pairs):
    return {"pair_count": len(pairs),
            "unique_question_count": len({row["question_id"] for row in pairs}),
            "repetition_count": len({row["repetition"] for row in pairs}),
            "usage": {name: paired_stat(pairs, usage, name) for name in USAGE_FIELDS},
            "latency_ms": {name: paired_stat(pairs, latency, name) for name in LATENCY_FIELDS},
            "setup_latency_ms": {name: paired_stat(pairs, latency, name) for name in SETUP_LATENCY_FIELDS},
            "semantic_retrieval": {arm: semantic_status_counts([row[arm] for row in pairs])
                                   for arm in ("off", "on")}}


def load_dispatch_records(run_directory):
    """Use the real durable dispatch events, never planned AB/BA alternation."""
    result = {}
    if run_directory is None:
        return result
    path = Path(run_directory) / "calls.jsonl"
    if not path.exists():
        return result
    for position, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("event") != "dispatch_started" or not isinstance(event.get("call_id"), str):
            continue
        identity = event["call_id"]
        if identity in result:
            result[identity]["ambiguous_duplicate"] = True
            continue
        timestamp = event.get("recorded_at")
        try:
            parsed = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
            timestamp = timestamp if parsed is not None and parsed.tzinfo is not None else None
        except ValueError:
            timestamp = None
        result[identity] = {"journal_line": position, "recorded_at": timestamp, "ambiguous_duplicate": False}
    return result


def actual_call_order(pair, dispatches):
    actual = {arm: dispatches.get(pair[arm].get("call_id")) for arm in ("off", "on")}
    reason = None
    if any(event is None for event in actual.values()):
        first = None
        reason = "missing_actual_dispatch_event"
    elif any(event["ambiguous_duplicate"] for event in actual.values()):
        first = None
        reason = "duplicate_dispatch_event"
    else:
        first = min(actual, key=lambda arm: actual[arm]["journal_line"])
    return {"first_called_arm": first, "unknown_reason": reason, "dispatch_events": actual,
            "order_basis": "calls.jsonl dispatch_started line order; durable pre-HTTP-dispatch events of the sequential runner",
            "timestamp_field": "dispatch_started.recorded_at; timezone-aware UTC recording time, not server receipt or first-token time"}


def cache_match(pair):
    off, on = usage(pair["off"], "cached_tokens"), usage(pair["on"], "cached_tokens")
    if off is None or on is None:
        return "unknown"
    return "equal_known" if off == on else "different_known"


def collect_pairs(fixture, records, *, expected_model=MODEL):
    pairs, diagnostics, excluded = [], [], []
    for question in fixture["questions"]:
        repetitions = sorted({key[1] for key in records if key[0] == question["id"]})
        if not repetitions:
            excluded.append({"question_id": question["id"], "reason": "no_answers_yet"})
        for repetition in repetitions:
            off, on = records.get((question["id"], repetition, False)), records.get((question["id"], repetition, True))
            pair = {"question_id": question["id"], "repetition": repetition, "category": question.get("category"),
                    "eval_split": question.get("eval_split", "unspecified"),
                    "off": off, "on": on, "diagnostic_only": bool(question.get("diagnostic_only"))}
            if question.get("diagnostic_only"):
                diagnostics.append(pair)
                continue
            if off is None or on is None:
                excluded.append({"question_id": question["id"], "repetition": repetition, "reason": "incomplete_pair"})
                continue
            contrast = off.get('contrast', 'context-assistance')
            if contrast not in {'context-assistance', 'insight'} or contrast != on.get('contrast', 'context-assistance'):
                excluded.append({'question_id': question['id'], 'repetition': repetition,
                                 'reason': 'contrast_mismatch'})
                continue
            if contrast == 'insight' and any(
                    record.get('arm_configuration', {}).get('context_assistance_enabled') is not True
                    or record.get('arm_configuration', {}).get('knowledge_skill_mode') != 'auto'
                    or record.get('arm_configuration', {}).get('insight_mode') != mode
                    for record, mode in ((off, 'off'), (on, 'auto'))):
                excluded.append({'question_id': question['id'], 'repetition': repetition,
                                 'reason': 'insight_contrast_configuration_mismatch'})
                continue
            pair['contrast'] = contrast
            reasons = {arm: eligibility(record, expected_model=expected_model)
                       for arm, record in (("off", off), ("on", on))}
            if any(reasons.values()):
                excluded.append({"question_id": question["id"], "repetition": repetition,
                                 "reason": "answer_not_eligible", "arm_reasons": reasons})
                continue
            if (off.get("corpus_digest") != on.get("corpus_digest") or not off.get("corpus_digest")
                    or off.get("provider", {}).get("parameters") != on.get("provider", {}).get("parameters")):
                excluded.append({"question_id": question["id"], "repetition": repetition,
                                 "reason": "corpus_or_generation_settings_mismatch"})
                continue
            if not question.get("include_in_primary_quality_average", True):
                excluded.append({"question_id": question["id"], "repetition": repetition, "reason": "fixture_exclusion"})
                continue
            pair["exposure"] = exposure(off, on)
            pairs.append(pair)
    return pairs, diagnostics, excluded


def diagnostic_record(record):
    if record is None:
        return None
    result = record.get("result", {})
    return {"status": record.get("status"), "answer_mode": result.get("answer_mode"),
            "source_count": len(result.get("sources", [])), "answer": result.get("answer"),
            "actual_model_calls": result.get("reasoning", {}).get("trace", {}).get("answer_model_calls"),
            "assistance": record.get("assistance", {}),
            "support_kind": support_kind(record), "semantic_retrieval_status": semantic_retrieval_status(record),
            "usage": {name: usage(record, name) for name in USAGE_FIELDS},
            "setup_latency_ms": {name: latency(record, name) for name in SETUP_LATENCY_FIELDS},
            "total_answer_latency_ms": latency(record, "total_answer_latency_ms")}


def provider_accounting(run_directory):
    """Count receipts as well as successful answer artifacts, including failures."""
    path = Path(run_directory) / "calls.jsonl"
    dispatches, receipts, parse_errors = set(), {}, []
    if path.exists():
        for position, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("call_id"), str):
                    raise ValueError
                if event.get("event") == "dispatch_started":
                    dispatches.add(event["call_id"])
                elif event.get("event") == "provider_completed":
                    if event["call_id"] in receipts:
                        raise ValueError
                    receipts[event["call_id"]] = event
            except (ValueError, TypeError):
                parse_errors.append({"line": position, "error": "invalid_or_duplicate_provider_journal_event"})
    for identity in receipts:
        dispatches.add(identity)
    counters = {}
    for name in USAGE_FIELDS:
        values = [usage({"provider": record}, name) for record in receipts.values()]
        known = [value for value in values if value is not None]
        unknown = len(dispatches) - len(known)
        counters[name] = {"reported_total": sum(known) if known and unknown == 0 else None,
                          "known_reported_subtotal": sum(known), "known_call_count": len(known),
                          "unknown_call_count": unknown}
    return {"dispatched_call_count": len(dispatches), "provider_receipt_count": len(receipts),
            "indeterminate_call_count": len(dispatches - set(receipts)),
            "usage_including_failed_and_incomplete_calls": counters,
            "provider_status_counts": dict(Counter(row.get("status", "unknown") for row in receipts.values())),
            "non_ok_calls": [{"call_id": identity, "status": record.get("status"),
                              "model_actual": record.get("model_actual"), "finish_reason": record.get("finish_reason"),
                              "usage": {name: usage({"provider": record}, name) for name in USAGE_FIELDS}}
                             for identity, record in receipts.items() if record.get("status") != "ok"],
            "journal_errors": parse_errors,
            "definition": "All provider-reported consumption, including failed or truncated attempts. Not currency charges; unknown usage is not zero. Separate from successful matched-pair averages."}


def metrics(fixture, records, errors, *, run_directory=None, expected_model=MODEL):
    pairs, diagnostics, excluded = collect_pairs(fixture, records, expected_model=expected_model)
    dispatches = load_dispatch_records(run_directory)
    for row in pairs:
        row["call_order"] = actual_call_order(row, dispatches)
        row["cached_token_match"] = cache_match(row)
    exposures = sorted({row["exposure"] for row in pairs})
    cache = []
    details = []
    for row in pairs:
        off, on = row["off"], row["on"]
        counters = {}
        for name in USAGE_FIELDS:
            before, after = usage(off, name), usage(on, name)
            counters[name] = {"off": before, "on": after, "delta": after - before if before is not None and after is not None else None}
        timing = {}
        for name in LATENCY_FIELDS:
            before, after = latency(off, name), latency(on, name)
            timing[name] = {"off": before, "on": after, "delta": after - before if before is not None and after is not None else None}
        setup = {}
        for name in SETUP_LATENCY_FIELDS:
            before, after = latency(off, name), latency(on, name)
            setup[name] = {"off": before, "on": after, "delta": after - before if before is not None and after is not None else None}
        cache.append({"question_id": row["question_id"], "repetition": row["repetition"],
                      "exposure": row["exposure"], "off_cached_tokens": usage(off, "cached_tokens"),
                      "on_cached_tokens": usage(on, "cached_tokens"),
                      "cache_hit_changed": (usage(off, "cached_tokens") != usage(on, "cached_tokens")
                                            if usage(off, "cached_tokens") is not None and usage(on, "cached_tokens") is not None else None)})
        details.append({"question_id": row["question_id"], "repetition": row["repetition"], "exposure": row["exposure"],
                        'contrast': row['contrast'],
                        "eval_split": row["eval_split"],
                        "call_order": row["call_order"], "cached_token_match": row["cached_token_match"],
                        "source_content_identical": off.get("source_content_digest") == on.get("source_content_digest"),
                        "off_assistance": off.get("assistance", {}), "on_assistance": on.get("assistance", {}),
                        "support_kind": {"off": support_kind(off), "on": support_kind(on)},
                        "semantic_retrieval_status": {"off": semantic_retrieval_status(off), "on": semantic_retrieval_status(on)},
                        "usage": counters, "latency_ms": timing, "setup_latency_ms": setup})
    corpus = {}
    for record in records.values():
        if record.get("corpus_digest"):
            corpus.setdefault(record["corpus_digest"], {"ingestion_latency_ms": number(record.get("ingestion_latency_ms"))})
    report = {"schema": SCHEMA, "actual_model_required": expected_model,
            'delivery_gate_version': DELIVERY_GATE_VERSION,
            'delivery_gate_scope': 'Applied to actual result fields, including legacy records labelled completed; provider billing remains independently counted.',
            'contrast_counts': dict(Counter(row['contrast'] for row in pairs)),
            'contrast_definitions': {
                'context-assistance': 'Existing background-assistance constructor switch OFF/ON; Insight uses its current default in both arms.',
                'insight': 'Insight runtime OFF/AUTO through the real service pipeline; static method cards and other background assistance remain enabled in both arms. Final card count may differ under the shared budget.',
            },
            "primary": summarize_pairs(pairs), "exposure_strata": {
                name: summarize_pairs([row for row in pairs if row["exposure"] == name]) for name in exposures},
            "by_repetition": {str(repetition): summarize_pairs([row for row in pairs if row["repetition"] == repetition])
                              for repetition in sorted({row["repetition"] for row in pairs})},
            "by_eval_split": {split: summarize_pairs([row for row in pairs if row["eval_split"] == split])
                              for split in sorted({row["eval_split"] for row in pairs})},
            "by_first_called_arm": {arm: summarize_pairs([row for row in pairs if (row["call_order"]["first_called_arm"] or "unknown") == arm])
                                    for arm in ("off", "on", "unknown")},
            "cached_token_match_strata": {name: summarize_pairs([row for row in pairs if row["cached_token_match"] == name])
                                          for name in ("equal_known", "different_known", "unknown")},
            "stratification_scope": "Descriptive only. Repetitions are repeated generations of the same questions, not new independent topics. Call order uses actual recorded dispatch events; unknown records remain unknown. Equal cached-token counts do not establish identical server cache state.",
            "pairs": details, "excluded": excluded, "artifact_errors": errors,
            "diagnostics": [{"question_id": row["question_id"], "repetition": row["repetition"],
                             "off": diagnostic_record(row["off"]), "on": diagnostic_record(row["on"])} for row in diagnostics],
            "cache_comparison": cache,
            "cache_warning": "Vendor prompt-cache differences are reported separately and cannot be attributed to background-assistance token savings.",
            "semantic_retrieval_all_answer_records": {
                "scope": "all saved answer artifacts, including diagnostic, incomplete or unmatched answers; no missing provider result is inferred",
                **{arm: semantic_status_counts([record for record in records.values() if record.get("assistance_enabled") is enabled])
                   for arm, enabled in (("off", False), ("on", True))}},
            "corpus_ingestion": corpus, "corpus_clone_latency_ms": paired_stat(pairs, lambda r, _: number(r.get("corpus_clone_latency_ms")), "clone"),
            "latency_definition": "Wall-clock pre-model, full non-streaming HTTP model call, post-model and total answer; not first-token latency. New runs exclude PDF ingestion, corpus cloning and document-vector prewarm from the answer timer and record them separately. Query encoding and retrieval remain inside the answer timer. Legacy missing setup measurements stay null.",
            "setup_cost_definition": "retrieval_setup_latency_ms measures document-vector prewarm in each fresh storage namespace; per_arm_setup_latency_ms adds that arm's corpus clone. upfront_setup_latency_ms additionally includes the shared, amortizable frozen-corpus ingestion cost. Its paired mean describes cold-start-equivalent cost, not total run expenditure: never sum upfront_setup_latency_ms across answers. corpus_ingestion retains each corpus once; setup is local work with real elapsed cost, not free provider tokens.",
            "support_kind_definition": "evidence_reselection exposure requires changed actual source quotes plus a successful reselection trace; it does not create support items. source_organization is extractive source-evidence selection/organization, not a deduction or causal proof. formal_auxiliary contains conditional deduction, analogy or induction outputs, not verified empirical truth. Mixed support is separate; legacy untyped actual_support is not silently assigned an algorithm.",
            "usage_definition": "Actual provider counts only; prompt/completion/total/cache/reasoning stay separate. Missing values are null and incomplete measured pairs do not receive an invented zero.",
            "inference_scope": "Small paired pilot; repeated generations of the same questions do not add independent topics. All strata and averages are descriptive, not evidence of statistical significance or general algorithm quality."}
    if run_directory is not None:
        report["all_provider_consumption"] = provider_accounting(run_directory)
    return report


class PaperPages:
    def __init__(self, fixture, directory):
        self.directory = Path(directory).resolve()
        self.metadata = {row["id"]: row for row in fixture["papers"]}
        self.readers = {}

    def page(self, identity, number):
        if identity not in self.readers:
            from pypdf import PdfReader
            metadata = self.metadata[identity]
            path = (self.directory / metadata["file"]).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("paper_path_outside_directory")
            content = path.read_bytes()
            if metadata.get("sha256") and hashlib.sha256(content).hexdigest() != metadata["sha256"]:
                raise ValueError("paper_hash_mismatch")
            self.readers[identity] = PdfReader(path, strict=True)
        reader = self.readers[identity]
        if type(number) is not int or not 1 <= number <= len(reader.pages):
            raise ValueError("reference_page_out_of_range")
        return {"paper_id": identity, "title": self.metadata[identity]["title"], "pdf_page": number,
                "text": reader.pages[number - 1].extract_text() or ""}


def candidate_packet(record, identity, paper_metadata):
    sources = []
    for source in record["result"].get("sources", []):
        paper_id = source.get("paper_id")
        if paper_id not in paper_metadata:
            raise ValueError("answer_source_missing_paper_mapping")
        sources.append({"label": source["label"], "paper_id": paper_id, "paper_title": paper_metadata[paper_id]["title"],
                        "pdf_page": source.get("page"), "excerpt": source["content"],
                        "excerpt_start": source.get("excerpt_start"), "excerpt_end": source.get("excerpt_end"),
                        "offset_unit": "characters_in_source_chunk", "truncated": source.get("truncated")})
    return {"candidate_id": identity, "answer": record["result"]["answer"], "actually_supplied_sources": sources,
            "supplied_retrieval_metadata": {"retrieval_gaps": record["result"].get("retrieval_gaps", [])}}


def build_blind_packets(fixture, pairs, output, papers_directory, *, batches=2, page_loader=None):
    output = Path(output)
    mapping_path = output / "private_mapping.json"
    fixture_digest = fingerprint(fixture)
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if mapping.get("fixture_digest") != fixture_digest:
            raise ValueError("blind_fixture_mismatch_use_new_output")
    else:
        mapping = {"schema": SCHEMA, "fixture_digest": fixture_digest, "packets": {},
                   "note": "PRIVATE: do not provide this mapping or metrics to blinded reviewers."}
    questions = {row["id"]: row for row in fixture["questions"]}
    question_positions = {row["id"]: index for index, row in enumerate(fixture["questions"])}
    papers = {row["id"]: row for row in fixture["papers"]}
    pages = page_loader or PaperPages(fixture, papers_directory).page
    prepared = []
    for pair in pairs:
        key = pair["question_id"] + ".r" + str(pair["repetition"])
        existing = next((entry for entry in mapping["packets"].values() if entry["pair_key"] == key), None)
        answer_hashes = {arm: fingerprint(pair[arm]["result"]["answer"]) for arm in ("off", "on")}
        if existing:
            if existing["answer_hashes"] != answer_hashes:
                raise ValueError("blind_answer_changed_use_new_output")
            entry = existing
        else:
            packet_id = "packet-" + secrets.token_hex(8)
            arms = ["off", "on"]
            secrets.SystemRandom().shuffle(arms)
            entry = {"packet_id": packet_id, "pair_key": key, "question_id": pair["question_id"],
                     "repetition": pair["repetition"], "answer_hashes": answer_hashes,
                     "candidate_order": ["candidate-" + secrets.token_hex(8) for _ in arms],
                     "candidates": {}, "batch": question_positions[pair["question_id"]] % batches + 1}
            entry["candidates"] = {identifier: {"arm": arm, "call_id": pair[arm]["call_id"]}
                                   for identifier, arm in zip(entry["candidate_order"], arms)}
            mapping["packets"][packet_id] = entry
            # Persist unblinding before emitting its packet; reruns never reshuffle.
            write_json(mapping_path, mapping, private=True)
        question = questions[pair["question_id"]]
        required_pages = {(fact["paper_id"], page) for fact in question.get("key_facts", []) for page in fact.get("pages", [])}
        for arm in ("off", "on"):
            cited = set(re.findall(r"\bS[1-9][0-9]*\b", pair[arm]["result"]["answer"]))
            required_pages.update((source["paper_id"], source["page"]) for source in pair[arm]["result"].get("sources", [])
                                  if source.get("label") in cited and source.get("paper_id") in papers and type(source.get("page")) is int)
        packet = {"packet_id": entry["packet_id"], "question": question["question"],
                  "question_language": question.get("language", "unspecified"),
                  "language_scoring_rule": "Judge task usefulness in the language requested by the question. A legacy rubric mentioning Chinese does not require Chinese for an English request; do not penalize an otherwise appropriate English answer. Do not infer experimental arm from language.",
                  "rubric": fixture["rubric"], "required_facts": question.get("key_facts", []),
                  "forbidden_claims": question.get("forbidden_claims", []),
                  "candidates": [candidate_packet(pair[entry["candidates"][identifier]["arm"]], identifier, papers)
                                 for identifier in entry["candidate_order"]],
                  "original_pdf_reference_pages": [pages(identity, page) for identity, page in sorted(required_pages)],
                  "review_instructions": "Score each candidate independently before comparing. First rate each required fact's evidence in the actually supplied source pack as full, partial or absent, independently of whether the answer mentions it. Full requires correct attribution and every material component/condition; a topic match or correct fact known only from the original PDF is not retrieved evidence. List entirely irrelevant source labels separately; partly relevant quotes are not entirely irrelevant. Then score the generated answer and its citations. Actual S-number excerpts, not full-paper knowledge, must support its citations. PDF pages verify facts and context. supplied_retrieval_metadata reports actual retrieval gaps supplied to that answer; it is not paper evidence and cannot support a paper conclusion. A reference to retrieval_gaps is not invented solely because it is not an S-number, but must not replace original paper evidence. Distinguish unknown or absent excerpt evidence from proving absence in the full paper. Do not infer experimental arm from answer length or style.",
                  "score_output_contract": {"packet_id": entry["packet_id"], "reviewer_id": "your reviewer identifier",
                                            "reviewer_kind": "agent_blind_review", "candidates": [
                                                {"candidate_id": identifier, "dimensions": {dimension["id"]: "integer 0..4" for dimension in fixture["rubric"]["dimensions"]},
                                                 "key_fact_checks": [], "citation_checks": [], "critical_errors": [],
                                                 "retrieved_evidence_checks": [
                                                     {"fact_id": fact["id"], "status": "full | partial | absent",
                                                      "source_labels": [], "reason": "Short source-grounded reason; full/partial need actual S labels, absent needs an empty list."}
                                                     for fact in question.get("key_facts", [])],
                                                 "irrelevant_source_labels": [],
                                                 "irrelevant_source_reason": "Explain entirely irrelevant labels or why none are flagged; distinguish mixed quotes.",
                                                 "rationale": "Concrete sentences and source excerpt support, not arm guesses."}
                                                for identifier in entry["candidate_order"]]}}
        path = output / "blind" / f"batch_{entry['batch']:02d}" / (entry["packet_id"] + ".json")
        write_json(path, packet)
        prepared.append({"packet_id": entry["packet_id"], "batch": entry["batch"], "path": str(path)})
    if not mapping_path.exists():
        write_json(mapping_path, mapping, private=True)
    return mapping, prepared


def quality_group(rows):
    deltas = [row["paired_delta"] for row in rows]
    return {"scored_reviewer_pair_count": len(rows),
            "unique_generation_pair_count": len({row["pair_key"] for row in rows}),
            "unique_question_count": len({row["question_id"] for row in rows}),
            "off_mean": statistics.mean(row["off_score"] for row in rows) if rows else None,
            "on_mean": statistics.mean(row["on_score"] for row in rows) if rows else None,
            "paired_delta_mean": statistics.mean(deltas) if deltas else None,
            "paired_delta_median": statistics.median(deltas) if deltas else None,
            "wins_on": sum(value > 0 for value in deltas), "wins_off": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas)}


def retrieved_evidence_audit(item, question, record):
    """Validate reviewer annotations, not factual entailment.

    Historic reviews may omit both fields. Incomplete fact annotations remain
    visible but never turn unreviewed facts into zero-recall observations.
    """
    expected = {fact["id"] for fact in question.get("key_facts", [])}
    labels = {row["label"] for row in record.get("result", {}).get("sources", [])}
    checks = item.get("retrieved_evidence_checks")
    checked, counts, supporting_labels = set(), Counter(), set()
    if checks is not None:
        if not isinstance(checks, list):
            raise ValueError("invalid_retrieved_evidence_checks")
        for check in checks:
            if not isinstance(check, dict):
                raise ValueError("invalid_retrieved_evidence_checks")
            fact_id, status, refs = check.get("fact_id"), check.get("status"), check.get("source_labels")
            if (not isinstance(fact_id, str) or fact_id not in expected or fact_id in checked or
                    status not in {"full", "partial", "absent"} or not isinstance(refs, list) or
                    any(not isinstance(label, str) or label not in labels for label in refs) or
                    len(set(refs)) != len(refs) or not isinstance(check.get("reason"), str) or
                    not check["reason"].strip() or (status == "absent" and refs) or
                    (status != "absent" and not refs)):
                raise ValueError("invalid_retrieved_evidence_checks")
            checked.add(fact_id)
            counts[status] += 1
            supporting_labels.update(refs)
    irrelevant = item.get("irrelevant_source_labels")
    irrelevant_reason = item.get("irrelevant_source_reason")
    if irrelevant is not None and (not isinstance(irrelevant, list) or
            any(not isinstance(label, str) or label not in labels for label in irrelevant) or
            len(set(irrelevant)) != len(irrelevant) or supporting_labels.intersection(irrelevant) or
            not isinstance(irrelevant_reason, str) or not irrelevant_reason.strip()):
        raise ValueError("invalid_irrelevant_source_labels")
    complete = checks is not None and checked == expected
    denominator = len(expected)
    return {"status": "not_applicable" if not denominator else
            "complete" if complete else "missing" if checks is None else "incomplete",
            "expected_fact_count": denominator, "checked_fact_count": len(checked),
            "unreviewed_fact_count": denominator - len(checked),
            "full_fact_count": counts["full"] if checks is not None else None,
            "partial_fact_count": counts["partial"] if checks is not None else None,
            "absent_fact_count": counts["absent"] if checks is not None else None,
            "evidence_recall": counts["full"] / denominator if complete and denominator else None,
            "partial_evidence_rate": counts["partial"] / denominator if complete and denominator else None,
            "all_facts_full": counts["full"] == denominator if complete and denominator else None,
            "source_label_count": len(labels), "irrelevant_source_labels": irrelevant,
            "irrelevant_source_reason": irrelevant_reason,
            "irrelevant_source_label_rate": len(irrelevant) / len(labels)
            if irrelevant is not None and labels else None,
            "retrieved_evidence_checks": checks}


def retrieval_quality_group(rows):
    """Separate supplied-evidence review from answer scores and model usage."""
    translated = [{arm: row[arm + "_review"].get("retrieval_audit", {}) for arm in ("off", "on")}
                  for row in rows]
    def value(audit, field):
        result = audit.get(field)
        if field == "all_facts_full":
            return int(result) if type(result) is bool else None
        return number(result)
    return {"scored_reviewer_pair_count": len(rows),
            "candidate_audit_coverage": {arm: {
                "scored_candidate_count": len(rows),
                "complete_fact_audit_count": sum(row[arm].get("status") == "complete" for row in translated),
                "missing_fact_audit_count": sum(row[arm].get("status") == "missing" for row in translated),
                "incomplete_fact_audit_count": sum(row[arm].get("status") == "incomplete" for row in translated),
                "not_applicable_fact_audit_count": sum(row[arm].get("status") == "not_applicable" for row in translated),
                "source_relevance_audit_count": sum(row[arm].get("irrelevant_source_labels") is not None for row in translated),
            } for arm in ("off", "on")},
            **{field: paired_stat(translated, value, field) for field in (
                "evidence_recall", "partial_evidence_rate", "all_facts_full", "irrelevant_source_label_rate")},
            "definition": "Each question uses its frozen required facts: evidence_recall is full facts / required facts; partial is not full. All-facts-full is a strict question-level indicator. Means use matched complete annotations only. Missing/incomplete reviews stay null, not zero. Irrelevant-source rates concern wholly irrelevant labels, not sentence/byte precision. Reviewer judgments are not verified entailment or human evaluation."}


def aggregate_scores(fixture, mapping, pairs, directory):
    weights = {row["id"]: row["weight"] for row in fixture["rubric"]["dimensions"]}
    if sum(weights.values()) != 100:
        raise ValueError("rubric_weights_must_sum_100")
    questions = {row["id"]: row for row in fixture["questions"]}
    pairs_by_key = {row["question_id"] + ".r" + str(row["repetition"]): row for row in pairs}
    expected = {identifier: {**candidate, "packet_id": packet_id,
                             "pair_key": entry["pair_key"], "question_id": entry["question_id"]}
                for packet_id, entry in mapping["packets"].items() for identifier, candidate in entry["candidates"].items()}
    scores, errors = {}, []
    for path in sorted(Path(directory).rglob("*.json")):
        try:
            review = json.loads(path.read_text(encoding="utf-8"))
            reviewer = review.get("reviewer_id")
            if not isinstance(reviewer, str) or not reviewer or not isinstance(review.get("candidates"), list):
                raise ValueError("invalid_review_identity")
            for item in review["candidates"]:
                identifier = item.get("candidate_id")
                if identifier not in expected or expected[identifier]["packet_id"] != review.get("packet_id"):
                    raise ValueError("unknown_blind_candidate")
                dimensions = item.get("dimensions")
                if (not isinstance(dimensions, dict) or set(dimensions) != set(weights)
                        or any(type(value) is not int or not 0 <= value <= 4 for value in dimensions.values())):
                    raise ValueError("invalid_dimension_scores")
                identity = (identifier, reviewer)
                if identity in scores:
                    raise ValueError("duplicate_reviewer_candidate_score")
                critical = item.get("critical_errors", [])
                if critical and (dimensions.get("factual_accuracy") != 0 or dimensions.get("scope_and_uncertainty") != 0):
                    raise ValueError("critical_error_requires_zero_fact_and_scope")
                candidate = expected[identifier]
                pair = pairs_by_key.get(candidate["pair_key"])
                audit = (retrieved_evidence_audit(item, questions[candidate["question_id"]], pair[candidate["arm"]])
                         if pair is not None else None)
                scores[identity] = {**expected[identifier], "reviewer_id": reviewer,
                                    "reviewer_kind": review.get("reviewer_kind", "unspecified"),
                                    "dimensions": dimensions,
                                    "weighted_score": sum(weights[name] * value / 4 for name, value in dimensions.items()),
                                    "critical_errors": critical, "key_fact_checks": item.get("key_fact_checks", []),
                                    "citation_checks": item.get("citation_checks", []), "rationale": item.get("rationale"),
                                    "retrieved_evidence_checks": item.get("retrieved_evidence_checks"),
                                    "irrelevant_source_labels": item.get("irrelevant_source_labels"),
                                    "irrelevant_source_reason": item.get("irrelevant_source_reason"),
                                    "retrieval_audit": audit}
        except (ValueError, TypeError, OSError) as error:
            errors.append({"file": path.name, "error": str(error) if str(error) in {
                "invalid_review_identity", "unknown_blind_candidate", "invalid_dimension_scores",
                "invalid_retrieved_evidence_checks", "invalid_irrelevant_source_labels",
                "duplicate_reviewer_candidate_score", "critical_error_requires_zero_fact_and_scope"} else "invalid_score_artifact"})
    grouped = {}
    for score in scores.values():
        grouped.setdefault((score["pair_key"], score["reviewer_id"]), {})[score["arm"]] = score
    complete, partial = [], []
    for (pair_key, reviewer), arms in grouped.items():
        if set(arms) != {"off", "on"} or pair_key not in pairs_by_key:
            partial.append({"pair_key": pair_key, "reviewer_id": reviewer, "reason": "not_a_complete_eligible_scored_pair"})
            continue
        complete.append({"pair_key": pair_key, "question_id": pairs_by_key[pair_key]["question_id"],
                         "eval_split": questions[pairs_by_key[pair_key]["question_id"]].get("eval_split", "unspecified"),
                         "repetition": pairs_by_key[pair_key]["repetition"],
                         "reviewer_id": reviewer, "exposure": pairs_by_key[pair_key]["exposure"],
                         "off_score": arms["off"]["weighted_score"], "on_score": arms["on"]["weighted_score"],
                         "paired_delta": arms["on"]["weighted_score"] - arms["off"]["weighted_score"],
                         "off_review": arms["off"], "on_review": arms["on"]})
    deltas = [row["paired_delta"] for row in complete]
    strata = {}
    for exposure_name in sorted({row["exposure"] for row in complete}):
        selected = [row for row in complete if row["exposure"] == exposure_name]
        differences = [row["paired_delta"] for row in selected]
        strata[exposure_name] = {
            "scored_reviewer_pair_count": len(selected),
            "off_mean": statistics.mean(row["off_score"] for row in selected),
            "on_mean": statistics.mean(row["on_score"] for row in selected),
            "paired_delta_mean": statistics.mean(differences), "paired_delta_median": statistics.median(differences),
            "wins_on": sum(value > 0 for value in differences), "wins_off": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences)}
    by_question = {identity: quality_group([row for row in complete if row["question_id"] == identity])
                   for identity in sorted({row["question_id"] for row in complete})}
    question_means = list(by_question.values())
    question_balanced = {
        "question_count": len(question_means),
        "off_mean": statistics.mean(row["off_mean"] for row in question_means) if question_means else None,
        "on_mean": statistics.mean(row["on_mean"] for row in question_means) if question_means else None,
        "paired_delta_mean": statistics.mean(row["paired_delta_mean"] for row in question_means) if question_means else None,
        "paired_delta_median": statistics.median(row["paired_delta_mean"] for row in question_means) if question_means else None,
        "definition": "Average available paired review/generation scores within each question, then give each question equal weight. Descriptive, not independent-question evidence from every repetition."}
    return {"scored_reviewer_pair_count": len(complete), "unique_question_pair_count": len({row["pair_key"] for row in complete}),
            "score_coverage": {"eligible_generation_pair_count": len(pairs),
                               "scored_generation_pair_count": len({row["pair_key"] for row in complete}),
                               "unscored_generation_pair_count": len(pairs) - len({row["pair_key"] for row in complete})},
            "unique_question_count": len(by_question), "question_balanced": question_balanced,
            "by_question": by_question,
            "retrieval_audit": retrieval_quality_group(complete),
            "by_eval_split": {split: {
                **quality_group([row for row in complete if row["eval_split"] == split]),
                "eligible_generation_pair_count": sum(row.get("eval_split", "unspecified") == split for row in pairs),
                "retrieval_audit": retrieval_quality_group([row for row in complete if row["eval_split"] == split]),
            } for split in sorted({row.get("eval_split", "unspecified") for row in pairs})},
            "by_repetition": {str(repetition): quality_group([row for row in complete if row["repetition"] == repetition])
                              for repetition in sorted({row["repetition"] for row in complete})},
            "off_mean": statistics.mean(row["off_score"] for row in complete) if complete else None,
            "on_mean": statistics.mean(row["on_score"] for row in complete) if complete else None,
            "paired_delta_mean": statistics.mean(deltas) if deltas else None,
            "paired_delta_median": statistics.median(deltas) if deltas else None,
            "wins_on": sum(value > 0 for value in deltas), "wins_off": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas), "exposure_strata": strata,
            "paired_reviews": complete, "partial_reviews": partial,
            "score_errors": errors, "reviewer_kinds": dict(Counter(row["reviewer_kind"] for row in scores.values())),
            "review_scope": "Independent blind agent reviews are not human evaluation. Multiple reviewer scores of one answer pair are not independent generation trials."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--papers-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores-dir")
    parser.add_argument("--model", default=MODEL, help="Required exact provider-reported model name")
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.batches <= 12:
            raise ValueError("invalid_batch_count")
        fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
        records, errors = load_records(args.run_dir)
        report = metrics(fixture, records, errors, run_directory=args.run_dir, expected_model=args.model)
        pairs, _, _ = collect_pairs(fixture, records, expected_model=args.model)
        mapping, packets = build_blind_packets(fixture, pairs, args.output, args.papers_dir, batches=args.batches)
        if args.scores_dir:
            report["quality"] = aggregate_scores(fixture, mapping, pairs, args.scores_dir)
        write_json(Path(args.output) / "metrics.json", report, private=True)
        write_json(Path(args.output) / "packet_inventory.json", {"packets": packets, "reviewers_receive": "blind/batch_* only"}, private=True)
        print(canonical({"eligible_pairs": len(pairs), "blind_packets": len(packets),
                         "output": str(Path(args.output).resolve()), "artifact_errors": len(errors),
                         "quality_scored_pairs": report.get("quality", {}).get("scored_reviewer_pair_count")}))
        return 0 if not errors else 1
    except Exception as error:
        print(canonical({"status": "analysis_failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
