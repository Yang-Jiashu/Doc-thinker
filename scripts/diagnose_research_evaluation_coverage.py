"""Zero-provider, actual-service coverage-stage diagnostic for development papers.

This is NOT a selector, a reference answer, or a quality score. Fixed public-text
locators are only read after the unchanged production prepare flow has finished.
They never enter a question, retrieval function, prompt, or model input. Missing
one locator does not establish that no equivalent evidence exists elsewhere.

Example (a fresh output directory is mandatory):
  python scripts/diagnose_research_evaluation_coverage.py --suite newpapers \
      --output dist/evaluation-stage-diagnostic-new

The old-three suite uses --suite old3. Both routes force semantic off and
prepare-only, have no credential argument, reject network connections, and
replace the real provider adapter with a fail-fast guard. No existing experiment
or fixture is modified. Instrumentation can affect preparation latency; do not
use its timings as an uninstrumented performance benchmark.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CALL = ContextVar("coverage_diagnostic_call", default=None)
SUITES = {
    "newpapers": (
        "benchmarks/research_real/insight_newpapers_smoke_20261013.json",
        "dist/insight-newpapers-smoke-20261013/papers", 2),
    "old3": (
        "benchmarks/research_real/insight_development_smoke_v2.json",
        "dist/real-paper-ab-20261010/papers", 6),
}


# These locators pin previously reviewed raw spans, not keywords to reward. A
# pair of anchors must occur in order in one original chunk; no cross-chunk
# stitching, text normalization, stemming, or answer-derived rubric is used.
# Each reported window is an independent diagnostic, not an all-must-pass gate.
LOCATORS = {
    "mb_retrieval": ("memorybank_2305.10250v1.pdf", 4,
        "we adopt a dual-tower dense retrieval model", "interchanged to any suitable model."),
    "mb_simulated_sample": ("memorybank_2305.10250v1.pdf", 6,
        "simulated long-term dialog history and 194 memory probing questions.",
        "each embodying the users’ personality."),
    "mb_measurement": ("memorybank_2305.10250v1.pdf", 8,
        "Evaluation Metrics The performance", "1 : coherent)."),
    "mb_results_and_base_model_confound": ("memorybank_2305.10250v1.pdf", 8,
        "Result Analysis.", "(3) Models’ performance varies on different languages."),
    "mb_update_exploratory_limit": ("memorybank_2305.10250v1.pdf", 4,
        "We increase S by 1 and reset t to 0,", "exploratory and highly simpliﬁed memory updating model."),
    "lm_bounded_queue": ("longmem_2306.07174v1.pdf", 4,
        "Cached Memory Bank is a cached head-wise vector queue", "previous context for the current inputs."),
    "lm_token_chunk_retrieval": ("longmem_2306.07174v1.pdf", 5,
        "Instead of performing token-to-token retrieval,", "K key-value pairs at token-level"),
    "lm_training_and_backbone_access": ("longmem_2306.07174v1.pdf", 4,
        "We perform a memory-augmented adaptation training", "the left context:"),
    "lm_chapterbreak_measurement": ("longmem_2306.07174v1.pdf", 7,
        "The splits of 4k, 6k, and 8k prefix are selected for", "the evaluation metric."),
    "lm_chapterbreak_results": ("longmem_2306.07174v1.pdf", 8,
        "Table 3: Zero-shot Suffix Identification Accuracy", "with 313x larger parameters."),
    "lm_nlu_measurement": ("longmem_2306.07174v1.pdf", 8,
        "Evaluation Setting. Here, we evaluate the in-context learning capability",
        "the best chunk-size 2 is used to report the results for MemTRM and our model."),
    "lm_nlu_results": ("longmem_2306.07174v1.pdf", 8,
        "achieves remarkable improvements on all NLU tasks", "over pretrained GPT-2* and MemTRM."),
    "rag_core_and_variants": ("rag.pdf", 1,
        "We introduce RAG models where the parametric", "per token."),
    "rag_sequence_token": ("rag.pdf", 3,
        "In one approach, RAG-Sequence,", "on a different document."),
    "memgpt_recall_archival": ("memgpt.pdf", 4,
        "We make a distinction between two types of external context:", "expansive document database."),
    "memgpt_external_access": ("memgpt.pdf", 4,
        "External context refers to out-of-context storage", "query external context: timestamp-based search, text-based search, and embedding-based search."),
    "memgpt_dmr_scoring": ("memgpt.pdf", 7,
        "We introduce a new ‘deep memory retrieval’", "calclating the ROUGE-L (F1) score."),
    "memgpt_dmr_task_and_primary_scoring": ("memgpt.pdf", 7,
        "We introduce a new ‘deep memory retrieval’", "sponse is consistent with the gold response"),
    "graph_entity_community_entry": ("graphrag.pdf", 1,
        "Our approach uses an LLM to build a graph-based text index in two stages:",
        "all partial responses are again summarized in a final response to the user."),
    "graph_map_reduce": ("graphrag.pdf", 5,
        "• Prepare community summaries.", "the global answer returned to the user."),
    "lost_controlled_position_task": ("lost_middle.pdf", 4,
        "we use accuracy as our primary evaluation", "(Figure 4)."),
    "lost_architecture_condition": ("lost_middle.pdf", 8,
        "When\nFlan-UL2 is evaluated", "mation in the middle of the input context."),
    "lost_query_position_manipulation": ("lost_middle.pdf", 9,
        "Our experiments in §3 and §4 place the query", "query-aware contextualization of documents (or\nkey-value pairs)?"),
    "lost_tuning_control": ("lost_middle.pdf", 9,
        "To better understand the potential effects of", "same experimental setup as §3."),
}

QUESTION_LOCATORS = {
    "newpapers_personal_assistant_roadmap": tuple(key for key in LOCATORS if key.startswith(("mb_", "lm_"))),
    "smoke_assistant_report": ("rag_core_and_variants", "rag_sequence_token", "memgpt_recall_archival",
                               "memgpt_external_access", "memgpt_dmr_scoring", "memgpt_dmr_task_and_primary_scoring"),
    "smoke_literature_comparison": ("rag_core_and_variants", "graph_entity_community_entry", "graph_map_reduce",
                                   "lost_controlled_position_task", "lost_architecture_condition"),
    "smoke_testable_hypothesis": ("rag_core_and_variants", "rag_sequence_token", "lost_controlled_position_task",
                                 "lost_architecture_condition", "lost_query_position_manipulation", "lost_tuning_control"),
}


def source_rows(rows):
    """Snapshot only original evidence geometry, never arbitrary model fields."""
    return [{"id": row.get("source_id", row.get("id")), "document_id": row.get("document_id"),
             "filename": row.get("filename"), "page": row.get("page"), "ordinal": row.get("ordinal"),
             "excerpt_start": row.get("excerpt_start", 0),
             "excerpt_end": row.get("excerpt_end", len(row.get("content", ""))),
             "content": row.get("content", "")} for row in rows]


def locate_raw_span(rows, locator):
    filename, page, start_anchor, end_anchor = locator
    matches = []
    for row in rows:
        if row["filename"] != filename or row["page"] != page:
            continue
        text = row["content"]
        start = text.find(start_anchor)
        if start < 0 or text.find(start_anchor, start + 1) >= 0:
            continue
        end = text.find(end_anchor, start + len(start_anchor))
        if end < 0:
            continue
        end += len(end_anchor)
        matches.append({"id": row["id"], "document_id": row["document_id"], "filename": filename,
                        "page": page, "ordinal": row["ordinal"], "start": start, "end": end,
                        "content": text[start:end], "sha256": hashlib.sha256(text[start:end].encode()).hexdigest()})
    return {"status": "unique" if len(matches) == 1 else "ambiguous" if matches else "not_located",
            "matches": matches}


def span_present(span, rows):
    # Complete contiguous literal coverage, not merely both endpoint words.
    return [index for index, row in enumerate(rows) if row["id"] == span["id"]
            and row["document_id"] == span["document_id"] and row["page"] == span["page"]
            and row["excerpt_start"] <= span["start"] and row["excerpt_end"] >= span["end"]
            and row["content"][span["start"] - row["excerpt_start"]:
                               span["end"] - row["excerpt_start"]] == span["content"]]


def source_geometry(rows, original):
    parents = {row["id"]: row for row in original}
    failures = []
    for index, row in enumerate(rows):
        parent = parents.get(row["id"])
        if not parent or any(row[key] != parent[key] for key in ("document_id", "filename", "page")):
            failures.append(index)
        elif parent["content"][row["excerpt_start"]:row["excerpt_end"]] != row["content"]:
            failures.append(index)
    return {"count": len(rows), "bytes": sum(len(row["content"].encode()) for row in rows),
            "exact_coordinates": not failures, "invalid_indexes": failures}


def coverage_report(question_id, capture):
    original = capture.get("corpus", [])
    stage_names = ("candidate_pool", "lexical_selected", "before_fit", "after_fit")
    output = {}
    for key in QUESTION_LOCATORS.get(question_id, ()):
        located = locate_raw_span(original, LOCATORS[key])
        item = {"locator": key, **located}
        if located["status"] == "unique":
            span = located["matches"][0]
            item["stages"] = {
                stage: {"observed": stage in capture,
                        "covering_window_indexes": span_present(span, capture.get(stage, []))}
                for stage in stage_names}
            absent = [stage for stage in stage_names if stage in capture
                      and not item["stages"][stage]["covering_window_indexes"]]
            item["first_absent_stage"] = absent[0] if absent else None
            item["finally_present"] = bool(item["stages"]["after_fit"]["covering_window_indexes"])
        output[key] = item
    return {"locators": output, "stage_geometry": {name: source_geometry(capture[name], original)
            for name in stage_names if name in capture}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("diagnostics require a fresh output directory; historical results are immutable")
    diagnostic_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["DOCTHINKER_RESEARCH_SEMANTIC"] = "off"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("diagnostic_benchmark", ROOT / "scripts/benchmark_research_assist.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    from docthinker.server import research_lexical_coverage as lexical
    from docthinker.server import research_service as service

    captures = {}
    fitted_objects = {}
    connections = []
    original_run = bench.run_one
    original_candidates = lexical.prepare_evidence_candidates
    original_select = lexical._select_windows
    original_fit = service.fit_research_prompt

    async def run_one(job, *a, **kw):
        assert kw.get("prepare_only") is True
        token = CALL.set(job["id"])
        capture = captures.setdefault(job["id"], {"question_id": job["question"]["id"]})
        try:
            result = await original_run(job, *a, **kw)
            capture["prepared_record"] = {key: result.get(key) for key in
                ("status", "provider_calls", "prompt_digest", "input_bytes", "preparation_latency_ms", "parameters")}
            quotes = result.get("diagnostics", {}).get("source_quotes", [])
            compared_fields = ("filename", "page", "excerpt_start", "excerpt_end", "content")
            capture["prepared_quotes_match_after_fit"] = bool(quotes) and [
                tuple(row.get(key) for key in compared_fields) for row in quotes] == [
                tuple(row.get(key) for key in compared_fields) for row in capture.get("after_fit", [])]
            trace = fitted_objects.get(job["id"], {}).get("reasoning", {}).get("trace", {})
            capture["adoption_trace"] = {key: trace[key] for key in
                ("lexical_evidence_coverage", "graph_retrieval", "fallback_metadata_filter", "semantic_retrieval")
                if key in trace}
            capture["audit"] = coverage_report(job["question"]["id"], capture)
            bench.private_json(output / "coverage_stages" / (job["id"] + ".json"), capture)
            return result
        finally:
            CALL.reset(token)

    def candidates(*a, **kw):
        value = original_candidates(*a, **kw)
        capture = captures[CALL.get()]
        capture["corpus"] = source_rows(a[1])
        capture["candidate_pool"] = source_rows(value["candidates"])
        capture["candidate_trace"] = value["trace"]
        return value

    def select(*a, **kw):
        value = original_select(*a, **kw)
        capture = captures[CALL.get()]
        capture["lexical_selected"] = source_rows(value[0])
        capture["selection_trace"] = value[1]
        return value

    def fit(*a, **kw):
        capture = captures[CALL.get()]
        capture["primary_before_fit"] = source_rows(a[1]["sources"])
        capture["before_fit"] = source_rows(a[3]["sources"])
        value = original_fit(*a, **kw)
        capture["after_fit"] = source_rows(value["plan"]["sources"])
        capture["fit_budget"] = value["budget"]
        fitted_objects[CALL.get()] = value
        return value

    def reject_connection(_socket, address):
        connections.append(type(address).__name__)
        raise RuntimeError("network_disabled_in_offline_coverage_diagnostic")

    async def reject_provider(*a, **kw):
        raise RuntimeError("real_provider_disabled_in_offline_coverage_diagnostic")

    fixture, papers, maximum = SUITES[args.suite]
    before = bench.code_fingerprint()
    with patch.object(bench, "run_one", run_one), patch.object(lexical, "prepare_evidence_candidates", candidates), \
         patch.object(lexical, "_select_windows", select), patch.object(service, "fit_research_prompt", fit), \
         patch.object(socket.socket, "connect", reject_connection), \
         patch.object(socket.socket, "connect_ex", reject_connection), \
         patch.object(bench.RealStepModel, "__call__", reject_provider):
        status = bench.main(["--fixture", str(ROOT / fixture), "--papers-dir", str(ROOT / papers),
                             "--output", str(output), "--contrast", "insight", "--prepare-only",
                             "--max-requests", str(maximum), "--temperature", "0.1"])
    after = bench.code_fingerprint()
    diagnostic_after = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {"schema": "literal-coverage-stage-diagnostic-v1", "suite": args.suite,
               "provider_calls": 0, "network_attempts": len(connections), "exit_status": status,
               "code_fingerprint_before": before, "code_fingerprint_after": after,
               "fingerprints_match": before == after,
               "diagnostic_sha256_before": diagnostic_before,
               "diagnostic_sha256_after": diagnostic_after,
               "diagnostic_file_unchanged_during_run": diagnostic_before == diagnostic_after,
               "scope": "Exact reviewed raw-span coverage only, not semantic support or answer quality. "
                        "No locator or review text was supplied to production retrieval or prompts. "
                        "A missing locator is not proof no equivalent evidence exists elsewhere. "
                        "Instrumentation timings are not uninstrumented performance results.",
               "calls": {key: {"prepared": value.get("prepared_record"), "audit": value.get("audit"),
                               "prepared_quotes_match_after_fit": value.get("prepared_quotes_match_after_fit")}
                         for key, value in captures.items()}}
    if output.exists():
        bench.private_json(output / "coverage_stage_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("suite", "provider_calls", "network_attempts",
                                                   "exit_status", "fingerprints_match")}, ensure_ascii=False))
    return int(bool(status or connections or before != after or diagnostic_before != diagnostic_after))


if __name__ == "__main__":
    raise SystemExit(main())
