"""Independent synthetic source-window competition, not paper-specific gold.

Fixed before the goal-family selector change. The selector receives an already
admitted pool, so failures identify selection rather than query nomination.
No prompt, model, provider, expected answer or real-paper term is involved.
"""
from __future__ import annotations

from copy import deepcopy
import random

import pytest

from docthinker.server.research_component_coverage import functional_definitions
from docthinker.server.research_lexical_coverage import (
    _select_windows, evaluation_cells, matching_tokens,
)


def window(identifier, doc, text, *, section="2 Methods", ordinal=0, score=.6,
           whole=None, start=0, outline_roles=()):
    whole = text if whole is None else whole
    row = {
        "source_id": identifier, "document_id": doc, "filename": doc + ".txt",
        "content": text, "section": section, "ordinal": ordinal,
        "pool_score": score, "excerpt_start": start, "excerpt_end": start + len(text),
        "offset_unit": "characters_in_source_chunk", "original_content_length": len(whole),
        "sentence_boundary_preserved": True, "truncated": text != whole,
        "outline_path": (section,), "outline_roles": list(outline_roles),
    }
    assert whole[start:start + len(text)] == text
    row["_evaluation_cells"] = evaluation_cells(row)
    row["_component_definitions"] = functional_definitions(row)
    return row


def facets(*docs):
    return {"facets": [{"document_ids": list(docs),
                         "roles": ["mechanism", "experiment_setup", "result"]}]}


def assert_geometry(selected, candidates):
    assert len(selected) <= 8
    assert sum(len(source["content"].encode()) for source in selected) <= 12000
    assert len({(source["document_id"], source["id"]) for source in selected}) == len(selected)
    for source in selected:
        assert any(source["id"] == row["source_id"] and
                   all(source[key] == row[key] for key in (
                       "document_id", "content", "excerpt_start", "excerpt_end", "offset_unit"))
                   for row in candidates)


def symmetric_pool(noise_role):
    rows = []
    for doc, material, operation in (
        ("OpalGate", "permission ledger", "record identifiers"),
        ("BirchRelay", "message archive", "retrieval keys"),
    ):
        rows.extend([
            window(doc + "-prepare", doc,
                f"We first construct an index from the {material}, then encode stable {operation}. "
                "The index is rebuilt only after an authorized source change.",
                section="2.1 Preparation", ordinal=1, outline_roles=("mechanism",)),
            window(doc + "-answer", doc,
                f"We first retrieve candidates from the {material}, then generate a response using selected {operation}. "
                "The answer keeps original links and does not alter the stored material.",
                section="2.2 Answer Operation", ordinal=2, outline_roles=("mechanism",)),
            window(doc + "-protocol", doc,
                "We evaluated 43 independently held-out requests against a fixed baseline over 9 days. "
                "Both conditions used the same model and output budget. "
                "Annotators score correctness using recorded targets; accuracy is defined as the fraction of correct answers.",
                section="3.1 Experiment Protocol", ordinal=3, outline_roles=("experiment_setup",)),
            window(doc + "-outcome", doc,
                f"Results show no improvement for the {material} under concurrent changes. "
                "The evaluation does not establish equivalence outside the held-out requests.",
                section="3.2 Results", ordinal=4, outline_roles=("result",)),
        ])
        for i in range(6):
            if noise_role == "mechanism":
                body = "The reader retrieves stored records, generates an example response, and selects a sample."
                section, roles = "2.3 Worked Examples", ("mechanism",)
            else:
                body = "We evaluated a baseline response. The example appears helpful in the experiment."
                section, roles = "3.3 Worked Examples", ("experiment_setup",)
            rows.append(window(f"{doc}-noise-{i}", doc, body, section=section,
                               ordinal=10 + i, outline_roles=roles))
    return rows


@pytest.mark.parametrize("noise_role", ["mechanism", "evaluation"])
@pytest.mark.parametrize("seed", [7, 29])
def test_two_requested_goal_families_keep_eight_complementary_windows_under_either_noise(noise_role, seed):
    candidates = symmetric_pool(noise_role)
    snapshot = deepcopy(candidates)
    random.Random(seed).shuffle(candidates)
    question = "Compare the mechanisms and actual evaluation of OpalGate and BirchRelay."
    sources, trace = _select_windows(question, candidates, facets("OpalGate", "BirchRelay"), [])
    expected = {doc + suffix for doc in ("OpalGate", "BirchRelay")
                for suffix in ("-prepare", "-answer", "-protocol", "-outcome")}
    assert {source["id"] for source in sources} == expected
    assert sorted(candidates, key=lambda row: row["source_id"]) == sorted(snapshot, key=lambda row: row["source_id"])
    assert trace["selection_proxy_is_verified_support"] is False
    assert trace["evaluation_coverage_is_verified_support"] is False
    assert_geometry(sources, candidates)


def test_near_identical_words_cannot_hide_a_new_measurement_cell():
    # Opaque record identifiers create genuinely high lexical overlap, while
    # the two short suffixes provide different source-bound study information.
    common = ("All observations are logged under these opaque record identifiers: " +
              " ".join("v" + str(n).zfill(3) for n in range(180)) + ". ")
    setup = window("protocol-outcome", "FennelSwitch", common +
        "The experiment evaluated a baseline dataset using 31 samples and a fixed model. "
        "Results show no improvement.", section="3 Experiments", outline_roles=("experiment_setup",))
    measure = window("measurement", "FennelSwitch", common +
        "Annotators score correctness on a three-level scale.",
        section="3.1 Measurement", ordinal=1, outline_roles=("experiment_setup",))
    assert {"setup", "outcome"} <= set(setup["_evaluation_cells"])
    assert measure["_evaluation_cells"] == ["measurement"]
    left, right = set(matching_tokens(setup["content"])), set(matching_tokens(measure["content"]))
    assert len(left & right) / len(left | right) >= .9
    candidates = [setup, measure]
    sources, trace = _select_windows("Explain the actual evaluation of FennelSwitch.",
                                    candidates, facets("FennelSwitch"), [])
    assert {source["id"] for source in sources} == {"protocol-outcome", "measurement"}
    assert trace["evaluation_coverage_is_verified_support"] is False
    assert_geometry(sources, candidates)


def test_same_source_alternatives_choose_strong_window_without_stitching_or_double_counting():
    early = "We evaluated one example against a baseline. The response appears useful."
    detail = ("We evaluated 62 held-out requests against a fixed baseline using the same model. "
              "Annotators score correctness using recorded targets. Results show no improvement; "
              "concurrent writes were not tested.")
    whole = early + "\n" + detail
    candidates = [
        window("one-chunk", "AsterSpool", early, whole=whole,
               section="3.1 Example", outline_roles=("experiment_setup",)),
        window("one-chunk", "AsterSpool", detail, whole=whole, start=len(early) + 1,
               section="3.2 Evaluation", outline_roles=("experiment_setup",)),
    ]
    sources, trace = _select_windows("Explain the actual evaluation of AsterSpool.",
                                    candidates, facets("AsterSpool"), [])
    assert len(sources) == 1
    assert sources[0]["content"] == detail
    assert whole[sources[0]["excerpt_start"]:sources[0]["excerpt_end"]] == detail
    assert trace["evaluation_coverage_is_verified_support"] is False
    assert_geometry(sources, candidates)


def test_renamed_ingest_ids_and_extra_duplicates_do_not_change_selected_content():
    rows = symmetric_pool("evaluation")
    question = "Compare the mechanisms and actual evaluation of OpalGate and BirchRelay."
    first, _ = _select_windows(question, rows, facets("OpalGate", "BirchRelay"), [])
    copied = deepcopy(rows)
    for row in copied:
        row["source_id"] = "new-ingest-" + row["source_id"]
    # Repetition changes source identity, not information or document identity.
    copied.extend({**row, "source_id": "duplicate-" + row["source_id"]}
                  for row in copied[:4])
    random.Random(101).shuffle(copied)
    second, _ = _select_windows(question, copied, facets("OpalGate", "BirchRelay"), [])
    projection = lambda sources: [(s["filename"], s["content"], s["excerpt_start"], s["excerpt_end"])
                                  for s in sources]
    assert projection(first) == projection(second)
    assert_geometry(second, copied)
