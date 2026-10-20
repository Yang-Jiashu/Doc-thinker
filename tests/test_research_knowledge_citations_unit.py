"""Knowledge-method K citations never substitute for uploaded-source S evidence."""

import uuid

import pytest

from docthinker.server.research_retrieval import citation_validation
from docthinker.server.research_store import ResearchStore, ResearchStoreError


def test_separate_current_s_and_k_references_can_both_be_syntax_valid():
    result = citation_validation("事实 [S1]，可借鉴的方法 [K1]。", ["S1"], knowledge_labels=["K1"])
    assert result["valid"]
    assert result["used_labels"] == ["S1", "K1"]
    assert result["unknown_labels"] == [] and not result["missing_citations"]
    assert result["scope"] == "citation_syntax_only" and not result["entailment_verified"]


def test_known_knowledge_labels_accept_an_already_deduplicated_set():
    assert citation_validation("[K1]", [], knowledge_labels={"K1"})["valid"]


def test_known_k_does_not_satisfy_uploaded_evidence_requirement():
    result = citation_validation("所谓项目事实 [K1]。", ["S1"], knowledge_labels=["K1"])
    assert not result["valid"] and result["missing_citations"]
    assert result["unknown_labels"] == []
    assert [warning["code"] for warning in result["warnings"]] == ["missing_citations"]


def test_no_uploaded_sources_can_use_known_k_without_inventing_s_requirement():
    result = citation_validation("方法参考而非用户事实 [K1]。", [], knowledge_labels=["K1"])
    assert result["valid"] and not result["missing_citations"]
    assert not result["entailment_verified"]


def test_k_with_no_selected_skill_is_unknown_even_on_legacy_two_argument_call():
    result = citation_validation("参考 [K1]。", [])
    assert result["unknown_labels"] == ["K1"] and not result["valid"]
    assert result["warnings"][0]["code"] == "unknown_citation_labels"


def test_k_in_uploaded_plan_cannot_self_authorize_method_reference():
    result = citation_validation("[S1] [K1]", {"selected_chunks": [{"label": "S1"}, {"label": "K1"}]})
    assert result["unknown_labels"] == ["K1"]
    assert not result["valid"]


def test_s_in_knowledge_labels_cannot_authorize_uploaded_source():
    result = citation_validation("[S1] [K1]", [], knowledge_labels=["S1", "K1"])
    assert result["unknown_labels"] == ["S1"]
    assert not result["valid"]


def test_interleaved_labels_keep_order_without_prefix_collision():
    result = citation_validation("[k1] [S1] [K2] [K1] [s01] [K01] [K0]", ["S1"], knowledge_labels=["K1"])
    assert result["used_labels"] == ["K1", "S1", "K2", "S01", "K01", "K0"]
    assert result["unknown_labels"] == ["K2", "S01", "K01", "K0"]
    assert not result["valid"]


@pytest.mark.parametrize("knowledge_labels", [None, "K1", {"K1": True}, ["K01", "K0", "k1", 3]])
def test_invalid_selection_labels_never_authorize_k(knowledge_labels):
    result = citation_validation("[K1]", [], knowledge_labels=knowledge_labels)
    assert result["unknown_labels"] == ["K1"]


def test_k_metadata_bounded_like_s_metadata():
    answer = "[K" + "9" * 5_000 + "] " + " ".join(f"[K{i}]" for i in range(100, 200))
    result = citation_validation(answer, [], knowledge_labels=["K1"])
    assert result["metadata_truncated"] and not result["valid"]
    assert len(result["used_labels"]) <= 32 and len(result["unknown_labels"]) <= 32
    assert all(len(label) <= 17 for label in result["used_labels"])


def test_existing_s_only_validation_unchanged_by_unused_knowledge_labels():
    for answer, sources in [("事实 [S1]", ["S1"]), ("没有引用", ["S1"]),
                            ("[S0] [S01] [S9]", ["S1"]), ("", []), ("你好", [])]:
        assert citation_validation(answer, sources) == citation_validation(answer, sources, knowledge_labels=["K1"])


def test_store_preserves_unknown_k_warning_not_as_source_reference(tmp_path):
    store = ResearchStore(tmp_path)
    answer_id = str(uuid.uuid4())
    validation = citation_validation("[K1] [K0] [K0001]", [])
    store.record_answer("#00001", answer_id, [], validation, "general")
    metadata = store.answer_metadata("#00001", [answer_id])[answer_id]
    assert metadata["source_refs"] == []
    assert metadata["validation"]["unknown_labels"] == ["K1", "K0", "K0001"]
    assert not metadata["validation"]["valid"]
    assert metadata["validation"]["warnings"][0]["labels"] == ["K1", "K0", "K0001"]


def test_store_preserves_valid_s_and_k_syntax_metadata(tmp_path):
    store = ResearchStore(tmp_path)
    document = store.accept_document("#00001", "notes.txt", b"actual project facts")
    store.claim_document("#00001", document["id"])
    store.complete_document("#00001", document["id"], [{"text": "actual project facts", "page": 1}])
    source = store.source_chunks("#00001", document_id=document["id"])[0]
    answer_id = str(uuid.uuid4())
    validation = citation_validation("[S1] [K1]", ["S1"], knowledge_labels=["K1"])
    store.record_answer("#00001", answer_id, [{"id": source["id"], "label": "S1"}], validation, "evidence")
    saved = store.answer_metadata("#00001", [answer_id])[answer_id]
    assert saved["validation"]["valid"]
    assert saved["validation"]["used_labels"] == ["S1", "K1"]
    assert [reference["label"] for reference in saved["source_refs"]] == ["S1"]


def test_store_still_refuses_k_as_an_uploaded_source_label(tmp_path):
    store = ResearchStore(tmp_path)
    with pytest.raises(ResearchStoreError, match="source label"):
        store.record_answer("#00001", str(uuid.uuid4()),
                            [{"id": str(uuid.uuid4()), "label": "K1"}], {"valid": True}, "evidence")


@pytest.mark.parametrize("label", ["R1", "../K1", "k1", "K-1", "K1;execute", "K"])
def test_store_unknown_label_allowlist_is_still_strict(tmp_path, label):
    store = ResearchStore(tmp_path)
    with pytest.raises(ResearchStoreError, match="validation label"):
        store.record_answer("#00001", str(uuid.uuid4()), [],
                            {"valid": False, "unknown_labels": [label]}, "general")
