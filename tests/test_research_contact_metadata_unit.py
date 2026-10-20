"""Contact-only units must not occupy a content question's evidence budget."""
import pytest

from docthinker.server.research_evidence_selection import _contact_metadata_only, prepare_evidence_candidates


@pytest.mark.parametrize("value", [
    "lee@university.example", "first|second@lab.org\n", "Email: jane.doe+lab@research.edu.",
    "mailto:lead@group.ac.uk", "{alice|bob}@institute.org", "邮箱：name@example.com",
    "Jane Doe <jane.doe@lab.net>", "Author: Jane Doe, jane@research.org.",
    "A. Doe (a.doe@research.edu)", "alice@example.org; bob@other.edu",
])
def test_complete_contact_unit_is_metadata(value):
    assert _contact_metadata_only(value)


@pytest.mark.parametrize("value", [
    "The procedure sends alerts to operator@lab.org after computing a gradient update.",
    "Contact Jane Doe <jane@lab.org> for details of the control procedure.",
    "We compare retrieval using exact email matches such as user@example.com and report accuracy.",
    "Jane Doe devised the calibration procedure, jane@lab.org.",
    "The system computes a likelihood ratio before selecting an action.",
    "A Method <team@lab.org> computes calibrated scores.",
])
def test_substantive_text_with_an_email_is_not_removed_or_cropped(value):
    assert not _contact_metadata_only(value)


def prepare(question, text):
    row = {"id": "source", "document_id": "doc", "filename": "instrument.pdf", "ordinal": 0, "content": text}
    plan = {"facets": [{"id": "original", "kind": "original", "text": question, "roles": ["mechanism"]}]}
    semantic = {"original": {"scores": {"source": .99}, "spans": {"source": [0, len(text)]}}}
    events = []
    result = prepare_evidence_candidates(question, [row], [row], plan, semantic, diagnostic_callback=events.append)
    return result, events


def test_high_semantic_score_and_baseline_reserve_cannot_reintroduce_contact_only_unit():
    result, events = prepare("Explain the author's experimental controls and computational cost", "first|second@laboratory.org\n")
    assert result["candidates"] == []
    assert result["trace"]["contact_metadata_units_filtered"] == 1
    assert any(row.get("reason") == "contact_only_metadata" for row in events)


@pytest.mark.parametrize("question", ["What is the author's email address?", "作者的邮箱是什么？", "作者是谁？"])
def test_explicit_contact_request_retains_the_original_unit(question):
    text = "Author: Jane Doe <jane@lab.org>"
    result, _ = prepare(question, text)
    assert any(row["content"] == text for row in result["candidates"])
    assert result["trace"]["contact_metadata_units_filtered"] == 0


def test_mixed_method_and_contact_in_one_complete_unit_remains_exact_source_text():
    text = "The method computes a calibrated threshold and reports to jane@lab.org when drift exceeds the limit."
    result, _ = prepare("Explain the method and calibration", text)
    assert result["candidates"] and result["candidates"][0]["content"] == text
    assert result["trace"]["contact_metadata_units_filtered"] == 0
