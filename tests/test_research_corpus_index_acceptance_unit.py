"""Independent full-plan equivalence checks against the frozen rank oracle.

All documents are invented here. No model, database, private corpus or network.
The oracle shares unchanged excerpt/token helpers, not the new corpus index.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random

import pytest

from docthinker.server.research_retrieval import build_retrieval_plan


_REFERENCE = Path(__file__).parent / "fixtures" / "reference_research_rank_20261013.py"
_SPEC = importlib.util.spec_from_file_location("frozen_research_rank_20261013", _REFERENCE)
_ORACLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ORACLE)


def row(text, identifier="c1", document="d1", ordinal=0, **metadata):
    return {"id": identifier, "document_id": document, "filename": document + ".pdf",
            "ordinal": ordinal, "content": text, **metadata}


def cases():
    mixed = [row("The archive retains the old revision only after validation fails.", "c1", "d1", page=2),
             row("索引记录更新版本。旧版本无法恢复时必须报告缺口。", "c2", "d2", page=4),
             row("Solar cells convert sunlight into electrical energy.", "c3", "d3")]
    for question in ("archive old revision", "索引如何更新版本？", "archive 旧版本 validation",
                     "unmatched volcanology", "总结所有资料", "Summarize all papers"):
        yield question, mixed, (), {}
    yield "archive validation", [], (), {}
    yield "archive validation", [None, 1, {}, {"id": "orphan", "content": "archive"},
        row("archive", is_generated=True), row("archive", is_discovered="1"),
        row("archive", provenance="hypothesis"), row("archive", identifier="x" * 257),
        row("archive\ud800"), *mixed], (), {}
    yield "archive revision", [row("Archive revision stays valid.", "z", "z"),
        row("Archive  revision\nstays valid. ", "a", "a"),
        row("Archive revision does not stay valid.", "n", "n"),
        row("Archive revision stays valid.", "a", "a")], (), {}
    yield "archive", [row("Archive retains records.", "same", "same"),
        row("Archive deletes records.", "same", "same"),
        row("The and of.", "empty-tokens", "empty-tokens"), row("!!!", "punct", "punct")], (), {}
    for options in ({"max_chunks": 0}, {"max_chunk_chars": 0}, {"max_source_bytes": 0},
                    {"max_source_bytes": 13}, {"max_chunks": 1},
                    {"max_chunks": 999, "max_chunk_chars": 99999, "max_source_bytes": 99999},
                    {"max_chunks": True, "max_source_bytes": False, "max_chunk_chars": "300"}):
        yield "archive 更新版本", mixed, (), options
    yield "irrelevant question", mixed, (), {"query_terms_override": ["archive", "旧版本", "validation archive"]}
    yield "archive", mixed, (), {"query_terms_override": []}
    yield "archive old revision", mixed, (), {"mode": "explore"}
    for scores in ({}, {"c1": .01, "c2": .8, "c3": .8},
                   {"c1": True, "c2": float("nan"), "c3": float("inf")},
                   {"c1": .009, "c2": 1.01, "outside": .9}):
        yield "archive", mixed, (), {"semantic_scores": scores}
    for span in ([0, 20], [0, 20000], [20, 3], [True, 20], [0.0, 20], "0:20", None):
        yield "unmatched", mixed, (), {"semantic_scores": {"c1": .7}, "semantic_spans": {"c1": span}}
    yield "archive", [row("背景。" * 550 + "The archive updates a revision only after validation succeeds. "
        + "后续说明。" * 200, page=7)], (), {"max_chunk_chars": 120, "max_source_bytes": 240}
    yield "archive", [row("Archive💡💡💡 valid only if approved.")], (), {"max_source_bytes": 12}
    yield "archive", [row("Archive entries remain valid.", page=True),
        row("Archive updates records.", "c2", "d2", page="3")], (), {}
    yield "archive validation", mixed, [
        {"id": "m1", "content": "Use a concise explanation, not an invented causal chain.", "status": "confirmed"},
        {"id": "m2", "content": "Old choice", "status": "deleted", "confirmed": True},
        {"id": "m3", "content": "过长偏好。" * 1000, "status": "confirmed"},
    ], {}
    yield "archive validation", [row("Archive validation " + "x " * 100, "hit", "a"),
        row("Archive archive validation.", "short", "b"),
        *[row(("Unrelated photosynthesis. " * (i + 1)), f"other-{i}", f"doc-{i}") for i in range(12)],
        row("and the of", "zero", "zero")], (), {}
    prefix = "Archive records are retained. " * 720
    yield "archive", [row(prefix + "Different tail A.", "a", "a"),
        row(prefix + "Different tail B.", "b", "b")], (), {}
    yield "archive", [row(f"Archive revision {i} remains valid under condition {i}.",
        f"c{i:03}", f"d{i % 5}", i // 5) for i in range(73)], (), {}
    yield "AsterLatch author contact", [row(
        "AsterLatch: An Archive Study\nExample Author\nauthor@example.invalid\nAbstract\n"
        "The archive retains revisions only after validation.\n1 Introduction\nFurther context.")], (), {}


_CASES = list(cases())


@pytest.mark.parametrize("question,chunks,memories,options", _CASES)
def test_fresh_and_prepared_full_plan_equal_frozen_rank(question, chunks, memories, options):
    from docthinker.server.research_corpus_index import prepare_research_corpus

    original = deepcopy(chunks)
    expected = _ORACLE.build_retrieval_plan(question, chunks, memories, **options)
    prepared = prepare_research_corpus(chunks)
    assert build_retrieval_plan(question, chunks, memories, **options) == expected
    assert build_retrieval_plan(question, chunks, memories, prepared_corpus=prepared, **options) == expected
    # The same index answers a second query independently: no answer/query cache.
    second = "validation 旧版本"
    assert build_retrieval_plan(second, chunks, prepared_corpus=prepared) == _ORACLE.build_retrieval_plan(second, chunks)
    assert chunks == original


def test_permutations_preserve_complete_plan_and_source_geometry():
    from docthinker.server.research_corpus_index import prepare_research_corpus

    chunks = [row(f"Archive revision {i} cannot replace older records before validation {i % 3}.",
                  f"c{i}", f"d{i % 4}", i // 4, page=i + 1) for i in range(27)]
    chunks += [row("Archive revision 9 cannot replace older records before validation 0.", "dup", "zz")]
    query = "archive revision validation"
    expected = _ORACLE.build_retrieval_plan(query, chunks)
    randomizer = random.Random(4609)
    for _ in range(7):
        randomizer.shuffle(chunks)
        prepared = prepare_research_corpus(chunks)
        actual = build_retrieval_plan(query, chunks, prepared_corpus=prepared)
        assert actual == expected
        originals = {(chunk["document_id"], chunk["id"]): chunk["content"] for chunk in chunks}
        for source in actual["sources"]:
            assert source["content"] == originals[source["document_id"], source["id"]][source["excerpt_start"]:source["excerpt_end"]]


def test_mutating_returned_plan_cannot_poison_next_prepared_answer():
    from docthinker.server.research_corpus_index import prepare_research_corpus

    chunks = [row("Archive validation preserves source records.")]
    prepared = prepare_research_corpus(chunks)
    result = build_retrieval_plan("archive", chunks, prepared_corpus=prepared)
    result["sources"][0]["content"] = "Injected replacement"
    result["selected_chunks"][0]["filename"] = "mutated.txt"
    result["gaps"].append({"code": "injected"})
    assert build_retrieval_plan("archive", chunks, prepared_corpus=prepared) == _ORACLE.build_retrieval_plan("archive", chunks)
