"""Evidence-only evaluation packets must not expose treatment or answer text."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "evidence_adjudication", Path(__file__).parents[1] / "benchmarks/research_real/prepare_evidence_adjudication.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    packet = {
        "packet_id": "packet-test", "question": "Which condition applies?",
        "required_facts": [{"id": "fact-1", "description": "Only under condition Z."}],
        "original_pdf_reference_pages": [{"text": "Full page is not retrieved evidence."}],
        "candidates": [{"candidate_id": name, "answer": "SECRET_ANSWER_" + name,
                        "actually_supplied_sources": [{"label": "S1", "excerpt": "Condition Z."}],
                        "supplied_retrieval_metadata": {"arm": "SECRET_ARM"}}
                       for name in ("candidate-a", "candidate-b")],
        "rubric": {"secret_score": 99},
    }
    review = {"candidates": [{"candidate_id": name,
                              "retrieved_evidence_checks": [{"fact_id": "fact-1", "status": "full"}],
                              "irrelevant_source_labels": [], "dimensions": {"secret_score": 99}}
                             for name in ("candidate-a", "candidate-b")]}
    return packet, [copy.deepcopy(review), copy.deepcopy(review)]


def test_agreed_reviews_do_not_need_adjudication():
    packet, reviews = inputs()
    assert module.build(packet, reviews) is None


def test_evidence_conflict_hides_answers_scores_and_metadata():
    packet, reviews = inputs()
    reviews[1]["candidates"][0]["retrieved_evidence_checks"][0]["status"] = "partial"
    result = module.build(packet, reviews)
    assert result["evidence_packs"][0]["fact_ids_to_check"] == ["fact-1"]
    text = module.encoded(result).decode()
    for forbidden in ("SECRET_ANSWER", "SECRET_ARM", "secret_score", '"dimensions"'):
        assert forbidden not in text


def test_identical_sources_get_one_consistent_adjudication():
    packet, reviews = inputs()
    # Same reviewer disagrees with itself across identical source packs.
    for review in reviews:
        review["candidates"][1]["retrieved_evidence_checks"][0]["status"] = "absent"
    result = module.build(packet, reviews)
    assert len(result["evidence_packs"]) == 1
    assert result["evidence_packs"][0]["candidate_ids"] == ["candidate-a", "candidate-b"]


def test_noise_disagreement_kept_separate_from_fact_agreement():
    packet, reviews = inputs()
    reviews[1]["candidates"][0]["irrelevant_source_labels"] = ["S1"]
    result = module.build(packet, reviews)
    assert result["evidence_packs"][0]["fact_ids_to_check"] == []
    assert result["evidence_packs"][0]["check_irrelevance"] is True


def test_missing_review_fact_is_not_treated_as_absent():
    packet, reviews = inputs()
    reviews[1]["candidates"][0]["retrieved_evidence_checks"] = []
    with pytest.raises(ValueError, match="incomplete"):
        module.build(packet, reviews)


def test_different_excerpts_are_never_merged():
    packet, reviews = inputs()
    packet["candidates"][1]["actually_supplied_sources"][0]["excerpt"] = "Different evidence."
    for candidate in reviews[1]["candidates"]:
        candidate["retrieved_evidence_checks"][0]["status"] = "partial"
    result = module.build(packet, reviews)
    assert len(result["evidence_packs"]) == 2
    assert all(len(pack["candidate_ids"]) == 1 for pack in result["evidence_packs"])
