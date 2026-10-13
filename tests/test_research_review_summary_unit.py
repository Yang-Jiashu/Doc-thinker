"""Review reporting must not pool away a language regression or missing data."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "review_summary", Path(__file__).parents[1] / "benchmarks/research_real/summarize_evidence_trace_reviews.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    fixture = {
        "questions": [
            {"id": "old", "language": "zh", "eval_split": "development_regression", "key_facts": []},
            {"id": "new-en", "language": "en", "eval_split": "new_document_holdout", "key_facts": []},
            {"id": "new-zh", "language": "zh", "eval_split": "new_document_holdout", "key_facts": []}],
        "rubric": {"dimensions": [{"id": "quality", "weight": 100}]},
    }
    protocol = {"quality_and_retrieval_review": {"independent_reviewers_per_main_pair": 2},
                "frozen_descriptive_acceptance": {"quality": "unchanged gate"}}
    return fixture, protocol, {}


def test_language_strata_keep_missing_reviews_unavailable_and_preserve_inputs():
    fixture, protocol, metrics = inputs()
    original = copy.deepcopy((fixture, protocol, metrics))
    result = module.build(fixture, protocol, metrics)
    assert set(result["by_language"]) == {"zh", "en"}
    assert set(result["by_eval_split_language"]["development_regression"]) == {"zh"}
    assert set(result["by_eval_split_language"]["new_document_holdout"]) == {"zh", "en"}
    assert result["eligible_generation_pair_count"] == 0
    assert result["release_gate"]["status"] == "not_evaluated"
    assert len([issue for issue in result["validation_issues"] if issue["code"] == "quality_unavailable"]) == 3
    assert result["limitations"][0].startswith("3 questions, not 6")
    assert result["frozen_acceptance"] == protocol["frozen_descriptive_acceptance"]
    assert (fixture, protocol, metrics) == original


def test_legacy_language_is_unspecified_not_inferred_from_question_or_answer():
    fixture, protocol, metrics = inputs()
    fixture["questions"][0].pop("language")
    result = module.build(fixture, protocol, metrics)
    assert "unspecified" in result["by_language"]
    assert result["per_question"][0]["language"] == "unspecified"


@pytest.mark.parametrize("language", [None, 42, "", []])
def test_invalid_language_metadata_fails_closed(language):
    fixture, protocol, metrics = inputs()
    fixture["questions"][0]["language"] = language
    with pytest.raises(ValueError, match="invalid_question_language"):
        module.build(fixture, protocol, metrics)
