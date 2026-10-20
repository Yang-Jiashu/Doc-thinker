"""Offline contracts for private project research and explicit confirmed memory."""
from pathlib import Path
import os
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from docthinker.server.research_store import (
    ResearchStore, ResearchStoreError, ResearchLimitError,
    ResearchNotFoundError, ResearchRevisionConflictError,
)


@pytest.fixture
def store(tmp_path):
    return ResearchStore(tmp_path, boot_id="first")


def ready(store, session="#00001", content=b"Climate policy evidence", filename="paper.txt", chunks=None):
    document = store.accept_document(session, filename, content)
    assert store.claim_document(session, document["id"])["status"] == "processing"
    store.complete_document(session, document["id"], chunks or [{"text": content.decode(), "page": 1}])
    return store.get_document(session, document["id"]), store.source_chunks(session, document_id=document["id"])


def test_read_only_empty_store_does_not_create_data(store):
    assert store.list_documents("#00001") == []
    assert store.pending_documents() == []
    assert store.source_chunks("#00001") == []
    assert store.list_memories("#00001") == []
    assert store.active_memory_context("#00001") == []
    assert store.export_memories("#00001")["memories"] == []
    assert list(store.root.iterdir()) == []


def test_same_document_different_projects_are_not_deduplicated(store):
    first = store.accept_document("#00001", "paper.txt", b"same document")
    second = store.accept_document("#00002", "paper.txt", b"same document")
    same = store.accept_document("#00001", "renamed.md", b"same document")
    assert first["id"] != second["id"]
    assert same["id"] == first["id"] and same["deduplicated"]
    assert store.list_documents("#00002") == [second]
    assert "content_hash" not in first and "path" not in first
    with pytest.raises(ResearchNotFoundError):
        store.get_document("#00003", first["id"])


def test_document_job_restart_is_recoverable_without_read_mutation(store):
    document = store.accept_document("#00001", "paper.md", b"a finding")
    store.claim_document("#00001", document["id"])
    before = store.path.read_bytes()
    restarted = ResearchStore(store.root, boot_id="second")
    assert restarted.get_document("#00001", document["id"])["status"] == "queued"
    assert restarted.pending_documents()[0]["id"] == document["id"]
    assert store.path.read_bytes() == before
    assert restarted.claim_document("#00001", document["id"])["attempt"] == 2
    with pytest.raises(ResearchRevisionConflictError):
        store.complete_document("#00001", document["id"], [{"text": "old process result"}])
    restarted.complete_document("#00001", document["id"], [{"text": "a finding"}])
    assert restarted.claim_document("#00001", document["id"]) is None


def test_failed_extraction_requires_explicit_local_retry(store):
    document = store.accept_document("#00001", "paper.pdf", b"%PDF-1.7 invalid remainder")
    store.claim_document("#00001", document["id"])
    failed = store.fail_document("#00001", document["id"], "invalid_pdf")
    assert failed["error_code"] == "invalid_pdf" and failed["status"] == "failed"
    assert store.pending_documents() == []
    assert store.claim_document("#00001", document["id"]) is None
    queued = store.retry_document("#00001", document["id"])
    assert queued["status"] == "queued" and queued["error_code"] is None
    assert store.claim_document("#00001", document["id"])["attempt"] == 2


@pytest.mark.parametrize("session", ["#1", "../#00001", "#００００１", "#00001/", "_system", None])
def test_project_references_are_canonical(store, session):
    with pytest.raises(ResearchStoreError):
        store.list_documents(session)
    assert list(store.root.iterdir()) == []


@pytest.mark.parametrize("filename", ["../paper.txt", "/paper.txt", "folder\\paper.md", "paper.exe", "paper.txt\x00", "paper\n.txt", " paper.txt", "paper.txt ", "a" * 121 + ".txt"])
def test_filename_is_never_a_storage_path(store, filename):
    with pytest.raises(ResearchStoreError):
        store.accept_document("#00001", filename, b"example")
    assert list(store.root.iterdir()) == []


@pytest.mark.parametrize("filename,content", [("paper.pdf", b"not pdf"), ("paper.txt", b"\xff"), ("paper.md", b""), ("paper.txt", b"x\x00y"), ("paper.txt", b" " * 20), ("paper.txt", b"x" * 300001), ("paper.pdf", b"%PDF-" + b"x" * (10 * 1024 * 1024))])
def test_document_content_limits_before_persistence(store, filename, content):
    with pytest.raises(ResearchStoreError):
        store.accept_document("#00001", filename, content)
    assert list(store.root.iterdir()) == []


def test_document_and_database_permissions(store):
    doc = store.accept_document("#00001", "Paper.TXT", b"valid utf8")
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.directory.stat().st_mode & 0o777 == 0o700
    assert store.files_directory.stat().st_mode & 0o777 == 0o700
    source = store.files_directory / (doc["id"] + ".txt")
    assert source.stat().st_mode & 0o777 == 0o600
    assert store.read_document_bytes("#00001", doc["id"]) == b"valid utf8"
    source.write_bytes(b"changed utf8")
    with pytest.raises(ResearchStoreError):
        store.read_document_bytes("#00001", doc["id"])


def test_source_chunks_project_scoped_and_pageable(store):
    first, chunks = ready(store, chunks=[{"text": "one", "page": 1}, {"content": "two", "page": 2}, {"text": "three"}])
    assert store.source_chunks("#00001", limit=1, offset=1)[0]["text"] == "two"
    assert store.source_chunks("#00002", [chunks[0]["id"]]) == []
    assert store.source_chunks("#00001", [str(uuid.uuid4())]) == []
    assert store.source_chunks("#00001", []) == []
    assert store.source_chunks("#00001", document_id=str(uuid.uuid4())) == []
    assert first["chunk_count"] == 3 and first["text_chars"] == 11
    assert chunks[2]["page"] is None
    for chunk in chunks:
        assert uuid.UUID(chunk["id"]).version == 4


@pytest.mark.parametrize("chunks", [[], [{"text": "x"}] * 601, [{"text": ""}], [{"text": "x" * 8001}], [{"text": "bad\ud800"}], [{"text": "yes", "page": 0}], [{"text": "yes", "page": True}], ["notdict"], [{"text": "x" * 8000}] * 38])
def test_extraction_result_bounded_and_atomic(store, chunks):
    doc = store.accept_document("#00001", "paper.txt", b"content")
    store.claim_document("#00001", doc["id"])
    with pytest.raises(ResearchStoreError):
        store.complete_document("#00001", doc["id"], chunks)
    assert store.get_document("#00001", doc["id"])["status"] == "processing"
    assert store.source_chunks("#00001") == []


def test_memory_requires_explicit_current_project_sources(store):
    _, chunks = ready(store)
    answer_id = str(uuid.uuid4())
    store.record_answer("#00001", answer_id, [{"id": chunks[0]["id"], "label": "S1"}], {}, "general")
    memory = store.create_memory("#00001", "finding", "Use this source as evidence", source_chunk_ids=[chunks[0]["id"]], source_answer_id=answer_id)
    assert memory["status"] == "active" and memory["confirmed_by_user"]
    assert memory["revision"] == 1
    with pytest.raises(ResearchStoreError):
        store.create_memory("#00002", "finding", "Cannot steal source", source_chunk_ids=[chunks[0]["id"]])
    with pytest.raises(ResearchNotFoundError):
        store.get_memory("#00002", memory["id"])
    assert store.list_memories("#00002") == []
    # A user may explicitly confirm a preference without pretending it is a fact.
    preference = store.create_memory("#00001", "preference", "Answer concisely")
    assert preference["source_chunk_ids"] == []


def test_memory_correction_replaces_context_not_accumulates_confidence(store):
    memory = store.create_memory("#00001", "decision", "Use route A")
    corrected = store.update_memory("#00001", memory["id"], expected_revision=1, text="Use route B")
    assert corrected["revision"] == 2
    assert store.active_memory_context("#00001")[0]["text"] == "Use route B"
    versions = store.memory_versions("#00001", memory["id"])
    assert [item["status"] for item in versions] == ["active", "superseded"]
    assert versions[1]["text"] == "Use route A"
    assert "confidence" not in corrected
    with pytest.raises(ResearchRevisionConflictError):
        store.update_memory("#00001", memory["id"], expected_revision=1, text="Lost concurrent change")
    with pytest.raises(ResearchRevisionConflictError):
        store.delete_memory("#00001", memory["id"], expected_revision=1)


def test_memory_delete_scrubs_all_versions_and_does_not_restore(store):
    memory = store.create_memory("#00001", "preference", "Private previous note")
    store.update_memory("#00001", memory["id"], expected_revision=1, text="Private next note")
    deleted = store.delete_memory("#00001", memory["id"], expected_revision=2)
    assert deleted["status"] == "deleted" and deleted["revision"] == 3
    assert deleted["text"] == "" and deleted["source_chunk_ids"] == []
    assert store.list_memories("#00001") == []
    assert store.active_memory_context("#00001") == []
    assert all(version["text"] == "" and version["status"] == "deleted" for version in store.memory_versions("#00001", memory["id"]))
    assert store.delete_memory("#00001", memory["id"], expected_revision=3) == deleted
    with pytest.raises(ResearchRevisionConflictError):
        store.update_memory("#00001", memory["id"], expected_revision=3, text="Do not revive")


def test_document_deletion_revokes_sources_and_memory_context(store):
    doc, chunks = ready(store)
    memory = store.create_memory("#00001", "finding", "Derived finding", source_chunk_ids=[chunks[0]["id"]])
    independent = store.create_memory("#00001", "preference", "Use Chinese")
    result = store.delete_document("#00001", doc["id"])
    assert result["invalidated_memory_ids"] == [memory["id"]]
    assert store.source_chunks("#00001", [chunks[0]["id"]]) == []
    assert store.list_documents("#00001") == []
    assert store.get_memory("#00001", memory["id"])["status"] == "invalidated"
    assert store.memory_versions("#00001", memory["id"])[0]["revision"] == 2
    assert store.active_memory_context("#00001") == [independent]
    assert not (store.files_directory / (doc["id"] + ".txt")).exists()
    with pytest.raises(ResearchStoreError):
        store.create_memory("#00001", "finding", "Cannot use revoked source", source_chunk_ids=[chunks[0]["id"]])
    with pytest.raises(ResearchNotFoundError):
        store.read_document_bytes("#00001", doc["id"])
    replacement = store.accept_document("#00001", "paper.txt", b"Climate policy evidence")
    assert replacement["id"] != doc["id"]


def test_context_budget_contains_only_complete_newest_notes(tmp_path):
    clock = iter(range(100))
    store = ResearchStore(tmp_path, clock=lambda: next(clock))
    store.create_memory("#00001", "preference", "a" * 100)
    newest = store.create_memory("#00001", "decision", "b" * 200)
    assert store.active_memory_context("#00001", max_chars=250) == [newest]
    assert store.active_memory_context("#00001", max_chars=50) == []
    assert store.active_memory_context("#00002") == []


@pytest.mark.parametrize("kind,text,sources", [("guess", "ok", []), ("finding", "", []), ("decision", "x" * 2001, []), ("preference", "bad\ud800", []), ("finding", "ok", "notlist"), ("finding", "ok", [str(uuid.uuid4())] * 21), ("finding", "ok", ["../chunk"])])
def test_memory_input_is_bounded(store, kind, text, sources):
    with pytest.raises(ResearchStoreError):
        store.create_memory("#00001", kind, text, source_chunk_ids=sources)
    assert store.list_memories("#00001") == []


def test_document_limits_do_not_break_session_dedup(store, monkeypatch):
    monkeypatch.setattr("docthinker.server.research_store.MAX_DOCUMENTS_PER_SESSION", 2)
    store.accept_document("#00001", "one.txt", b"one")
    store.accept_document("#00001", "two.txt", b"two")
    assert store.accept_document("#00001", "one.txt", b"one")["deduplicated"]
    with pytest.raises(ResearchLimitError):
        store.accept_document("#00001", "three.txt", b"three")
    assert store.accept_document("#00002", "one.txt", b"one")["status"] == "queued"


def test_storage_quota_replaces_revision_lifetime_limit_and_allows_privacy_deletion(store, monkeypatch):
    monkeypatch.setattr("docthinker.server.research_store.MAX_MEMORY_VERSIONS", 2)
    memory = store.create_memory("#00001", "finding", "one")
    store.update_memory("#00001", memory["id"], expected_revision=1, text="two")
    assert store.update_memory("#00001", memory["id"], expected_revision=2, text="three")["revision"] == 3
    monkeypatch.setattr(store, "max_storage_bytes", store.path.stat().st_size)
    with pytest.raises(ResearchLimitError):
        store.update_memory("#00001", memory["id"], expected_revision=3, text="four")
    with pytest.raises(ResearchLimitError):
        store.create_memory("#00001", "decision", "four")
    assert store.delete_memory("#00001", memory["id"], expected_revision=3)["status"] == "deleted"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "world_readable"])
def test_unsafe_database_rejected(store, tmp_path, kind):
    store.create_memory("#00001", "preference", "example")
    if kind == "symlink":
        real = tmp_path / "real.db"
        store.path.rename(real)
        store.path.symlink_to(real)
    elif kind == "hardlink":
        os.link(store.path, tmp_path / "other.db")
    else:
        store.path.chmod(0o644)
    with pytest.raises(ResearchStoreError):
        store.list_memories("#00001")
    with pytest.raises(ResearchStoreError):
        store.create_memory("#00001", "preference", "unsafe")


def test_symlink_parent_and_source_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "worker"
    root.mkdir()
    (root / "_system").symlink_to(outside, target_is_directory=True)
    store = ResearchStore(root)
    with pytest.raises(ResearchStoreError):
        store.accept_document("#00001", "paper.txt", b"do not write outside")
    assert list(outside.iterdir()) == []


def test_unsafe_root_rejected(tmp_path):
    with pytest.raises(ResearchStoreError):
        ResearchStore(Path("relative"))
    with pytest.raises(ResearchStoreError):
        ResearchStore(tmp_path / "missing")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ResearchStoreError):
        ResearchStore(alias)


def test_answer_provenance_project_scoped_survives_restart_and_retracts_sources(store):
    doc, chunks = ready(store)
    answer_id = str(uuid.uuid4())
    validation = {"valid": True, "used_labels": ["S1"], "unknown_labels": [], "missing_citations": False,
                  "warnings": [], "entailment_verified": True, "scope": "semantic"}
    source_refs = [{"id": chunks[0]["id"], "label": "S1", "content": "never copy this"}]
    assert store.record_answer("#00001", answer_id, source_refs, validation, "evidence")["recorded"]
    assert store.record_answer("#00001", answer_id, source_refs, validation, "evidence")["recorded"]
    assert store.answer_metadata("#00002", [answer_id]) == {}
    restarted = ResearchStore(store.root)
    metadata = restarted.answer_metadata("#00001", [answer_id])[answer_id]
    assert metadata["source_refs"] == [{"id": chunks[0]["id"], "label": "S1", "available": True}]
    assert metadata["validation"]["scope"] == "citation_syntax_only"
    assert metadata["validation"]["entailment_verified"] is False
    assert "answer" not in metadata
    store.delete_document("#00001", doc["id"])
    assert store.answer_metadata("#00001", [answer_id])[answer_id]["source_refs"][0]["available"] is False
    assert store.record_answer("#00001", answer_id, source_refs, validation, "evidence")["recorded"]


def test_answer_provenance_rejects_foreign_sources_and_collisions(store):
    _, chunks = ready(store)
    answer_id = str(uuid.uuid4())
    validation = {"valid": True}
    with pytest.raises(ResearchStoreError):
        store.record_answer("#00002", answer_id, [{"id": chunks[0]["id"], "label": "S1"}], validation, "evidence")
    store.record_answer("#00001", answer_id, [], validation, "explore")
    with pytest.raises(ResearchRevisionConflictError):
        store.record_answer("#00002", answer_id, [], validation, "explore")
    with pytest.raises(ResearchRevisionConflictError):
        store.record_answer("#00001", answer_id, [], validation, "evidence")


def test_answer_metadata_is_read_only_and_bounded(store, monkeypatch):
    assert store.answer_metadata("#00001", [str(uuid.uuid4())]) == {}
    assert list(store.root.iterdir()) == []
    with pytest.raises(ResearchStoreError):
        store.answer_metadata("#00001", ["../answer"])
    with pytest.raises(ResearchLimitError):
        store.answer_metadata("#00001", [str(uuid.uuid4())] * 201)
    monkeypatch.setattr("docthinker.server.research_store.MAX_ANSWER_RECORDS", 1)
    store.record_answer("#00001", str(uuid.uuid4()), [], {"valid": True}, "explore")
    assert store.record_answer("#00001", str(uuid.uuid4()), [], {"valid": True}, "explore")["recorded"]
    assert store.create_memory("#00001", "preference", "Capacity does not prevent notes")["status"] == "active"


@pytest.mark.parametrize("refs,validation,mode", [
    ([{"id": str(uuid.uuid4()), "label": "S11"}], {}, "evidence"),
    ([{"id": "notuuid", "label": "S1"}], {}, "evidence"),
    ([], {"used_labels": "wrong"}, "evidence"),
    ([], {"valid": "true"}, "evidence"),
    ([], {"warnings": [{"code": "model_claims_correct"}]}, "evidence"),
    ([], {}, "unbounded"),
])
def test_answer_validation_metadata_contract(store, refs, validation, mode):
    with pytest.raises(ResearchStoreError):
        store.record_answer("#00001", str(uuid.uuid4()), refs, validation, mode)


def test_final_packet_accepts_ten_live_sources_but_rejects_eleventh_or_bad_label(store):
    _, chunks = ready(store, chunks=[{'text': f'Exact excerpt {i}.'} for i in range(11)])
    refs = [{'id': row['id'], 'label': f'S{i}'} for i, row in enumerate(chunks, 1)]
    answer_id = str(uuid.uuid4())
    store.record_answer('#00001', answer_id, refs[:10], {'valid': True}, 'evidence')
    saved = ResearchStore(store.root).answer_metadata('#00001', [answer_id])[answer_id]
    assert saved['source_refs'] == [{**row, 'available': True} for row in refs[:10]]
    with pytest.raises(ResearchLimitError):
        store.record_answer('#00001', str(uuid.uuid4()), refs, {'valid': True}, 'evidence')
    with pytest.raises(ResearchStoreError, match='source label'):
        store.record_answer('#00001', str(uuid.uuid4()), [refs[10]], {'valid': True}, 'evidence')


def test_excessive_citation_metadata_is_clipped_and_not_certified_valid(store):
    answer_id = str(uuid.uuid4())
    validation = {"valid": True, "used_labels": ["S" + str(value) for value in range(1, 50)]}
    store.record_answer("#00001", answer_id, [], validation, "explore")
    saved = store.answer_metadata("#00001", [answer_id])[answer_id]["validation"]
    assert saved["metadata_truncated"] and not saved["valid"]
    assert len(saved["used_labels"]) == 32


def test_fabricated_citation_zero_is_warning_not_storage_failure(store):
    answer_id = str(uuid.uuid4())
    validation = {"valid": False, "used_labels": ["S0", "S0001"], "unknown_labels": ["S0", "S0001"],
                  "warnings": [{"code": "unknown_citation_labels", "labels": ["S0", "S0001"]}]}
    store.record_answer("#00001", answer_id, [], validation, "insufficient_evidence")
    metadata = store.answer_metadata("#00001", [answer_id])[answer_id]
    assert metadata["validation"]["unknown_labels"] == ["S0", "S0001"]
    assert not metadata["validation"]["valid"]


def test_concurrent_upload_and_claim_are_single_acceptance(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        docs = list(pool.map(lambda _: store.accept_document("#00001", "same.txt", b"same local source"), range(4)))
    assert len({doc["id"] for doc in docs}) == 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: store.claim_document("#00001", docs[0]["id"]), range(4)))
    assert sum(claim is not None for claim in claims) == 1
    assert len(list(store.files_directory.iterdir())) == 1


def test_concurrent_memory_corrections_do_not_lose_update(store):
    memory = store.create_memory("#00001", "decision", "original")

    def correct(value):
        try:
            return store.update_memory("#00001", memory["id"], expected_revision=1, text=value)
        except ResearchRevisionConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        changes = list(pool.map(correct, ["first choice", "second choice"]))
    assert sum(change is not None for change in changes) == 1
    assert store.get_memory("#00001", memory["id"])["revision"] == 2


def test_world_writable_parent_rejected_without_following_layout(store):
    system = store.root / "_system"
    system.mkdir(mode=0o700)
    system.chmod(0o777)
    with pytest.raises(ResearchStoreError):
        store.create_memory("#00001", "preference", "safe parent required")
    assert list(system.iterdir()) == []


def test_answer_provenance_preserves_exact_excerpt_without_text_snapshot(store):
    _, chunks = ready(store, chunks=[{"text": "before evidence after"}])
    answer_id = str(uuid.uuid4())
    reference = {"id": chunks[0]["id"], "label": "S1", "excerpt_start": 7, "excerpt_end": 15, "truncated": True}
    store.record_answer("#00001", answer_id, [reference], {"valid": True, "used_labels": ["S1"]}, "evidence")
    saved = store.answer_metadata("#00001", [answer_id])[answer_id]["source_refs"][0]
    assert saved == {**reference, "available": True}
    assert "content" not in saved


@pytest.mark.parametrize("start,end,truncated", [(0, 999, True), (3, 2, True), (-1, 4, True), (0, 21, True), (0, 5, False), (True, 5, True)])
def test_answer_excerpt_bounds_must_match_source(store, start, end, truncated):
    _, chunks = ready(store, chunks=[{"text": "before evidence after"}])
    reference = {"id": chunks[0]["id"], "label": "S1", "excerpt_start": start, "excerpt_end": end, "truncated": truncated}
    with pytest.raises(ResearchStoreError):
        store.record_answer("#00001", str(uuid.uuid4()), [reference], {"valid": True}, "evidence")


def test_visible_memories_exclude_deleted_not_revoked_and_are_bounded(store):
    doc, chunks = ready(store)
    revoked = store.create_memory("#00001", "finding", "will lose evidence", source_chunk_ids=[chunks[0]["id"]])
    deleted = store.create_memory("#00001", "preference", "delete this")
    active = store.create_memory("#00001", "decision", "continue this")
    store.delete_document("#00001", doc["id"])
    store.delete_memory("#00001", deleted["id"], expected_revision=1)
    visible = store.list_visible_memories("#00001")
    assert {memory["id"] for memory in visible} == {revoked["id"], active["id"]}
    assert store.list_visible_memories("#00002") == []
    with pytest.raises(ResearchStoreError):
        store.list_visible_memories("#00001", limit=101)


def test_answer_capacity_preflight_is_read_only_and_rejects_at_storage_bound(store, monkeypatch):
    capacity = store.answer_capacity("#00001")
    assert capacity["available"] and capacity["remaining"] == store.max_storage_bytes and capacity["unit"] == "bytes"
    assert list(store.root.iterdir()) == []
    store.record_answer("#00001", str(uuid.uuid4()), [], {"valid": True}, "general")
    monkeypatch.setattr(store, "max_storage_bytes", store.path.stat().st_size)
    assert not store.answer_capacity("#00001")["available"]
    assert not store.answer_capacity("#00002")["available"]
    with pytest.raises(ResearchLimitError):
        store.record_answer("#00001", str(uuid.uuid4()), [], {"valid": True}, "general")


def test_source_revocation_metadata_is_persistable_without_reissuing_answer(store):
    answer_id = str(uuid.uuid4())
    store.record_answer("#00001", answer_id, [], {"valid": False, "used_labels": ["S1"],
                        "warnings": [{"code": "source_revoked", "message": "private custom text ignored"}]}, "evidence")
    metadata = store.answer_metadata("#00001", [answer_id])[answer_id]
    assert metadata["validation"]["warnings"][0]["code"] == "source_revoked"
    assert not metadata["validation"]["valid"]


def test_answer_limitation_metadata_survives_restart_without_text_snapshot(store):
    answer_id = str(uuid.uuid4())
    context = {
        "retrieval_gaps": [{"code": "bounded_overview", "message": "private caller prose"},
                           {"code": "uncovered_query_terms", "terms": ["private query term"]},
                           {"code": "documents_not_selected", "count": 12},
                           {"code": "confirmed_memory_budget_omitted", "count": 3},
                           {"code": "evidence_coverage_proxy"},
                           {"code": "evidence_facets_uncovered", "count": 2},
                           {"code": "evidence_anchors_uncovered", "count": 1}],
        "used_memory_count": 2,
        "memory_selection": {"strategy": "lexical_bm25_with_preferences", "omitted_count": 80, "selected_count": 2},
    }
    store.record_answer("#00001", answer_id, [], {"valid": True}, "evidence", context_metadata=context)
    restarted = ResearchStore(store.root)
    saved = restarted.answer_metadata("#00001", [answer_id])[answer_id]
    assert saved["used_memory_count"] == 2
    assert saved["memory_selection"] == context["memory_selection"]
    assert [gap["code"] for gap in saved["retrieval_gaps"]] == [gap["code"] for gap in context["retrieval_gaps"]]
    assert "完整比较" in saved["retrieval_gaps"][0]["message"]
    assert "terms" not in saved["retrieval_gaps"][1]
    assert saved["retrieval_gaps"][2]["count"] == 12
    assert restarted.answer_metadata("#00002", [answer_id]) == {}
    with sqlite3.connect(store.path) as db:
        raw = db.execute("SELECT context_metadata_json FROM research_answers WHERE answer_id=?", (answer_id,)).fetchone()[0]
    assert "private" not in raw and '"message":' not in raw and '"terms":' not in raw
    assert json.loads(raw)["retrieval_gaps"][0] == {"code": "bounded_overview"}
    assert "answer" not in saved and "text" not in saved


def test_literal_coverage_scan_limit_is_persistable_and_survives_restart(store):
    document, chunks = ready(store)
    answer_id = str(uuid.uuid4())
    reference = {"id": chunks[0]["id"], "document_id": document["id"], "label": "S1"}
    context = {"retrieval_gaps": [{"code": "literal_coverage_scan_limited",
                                  "message": "private query-dependent explanation"}]}
    recorded = store.record_answer("#00001", answer_id, [reference],
                                   {"valid": True, "used_labels": ["S1"]}, "evidence",
                                   context_metadata=context)
    assert recorded["recorded"]
    restarted = ResearchStore(store.root)
    saved = restarted.answer_metadata("#00001", [answer_id])[answer_id]
    assert saved["validation"]["valid"] is True
    assert saved["source_refs"][0]["id"] == chunks[0]["id"]
    assert saved["retrieval_gaps"] == [{
        "code": "literal_coverage_scan_limited",
        "message": "本地原文筛选达到扫描上限，未覆盖全部资料；没有找到不代表不存在。",
    }]
    assert restarted.answer_metadata("#00002", [answer_id]) == {}
    with sqlite3.connect(store.path) as db:
        raw = db.execute("SELECT context_metadata_json FROM research_answers WHERE answer_id=?", (answer_id,)).fetchone()[0]
    assert json.loads(raw) == {"retrieval_gaps": [{"code": "literal_coverage_scan_limited"}]}
    assert "private" not in raw


@pytest.mark.parametrize("context", [
    "wrong",
    {"answer": "never store answer text here"},
    {"retrieval_gaps": "wrong"},
    {"retrieval_gaps": [{"code": "bounded_overview"}] * 9},
    {"retrieval_gaps": [{"code": "unknown_model_claim"}]},
    {"retrieval_gaps": [{"code": []}]},
    {"retrieval_gaps": [{"code": "bounded_overview"}, {"code": "bounded_overview"}]},
    {"retrieval_gaps": [{"code": "bounded_overview", "content": "source text"}]},
    {"retrieval_gaps": [{"code": "documents_not_selected", "count": 12001}]},
    {"retrieval_gaps": [{"code": "documents_not_selected", "count": True}]},
    {"used_memory_count": 9},
    {"used_memory_count": -1},
    {"used_memory_count": True},
    {"memory_selection": {"strategy": "unknown", "omitted_count": 0, "selected_count": 0}},
    {"memory_selection": {"strategy": [], "omitted_count": 0, "selected_count": 0}},
    {"memory_selection": {"strategy": "disabled", "omitted_count": 101, "selected_count": 0}},
    {"memory_selection": {"strategy": "disabled", "omitted_count": 0, "selected_count": 9}},
    {"memory_selection": {"strategy": "disabled", "omitted_count": 0, "selected_count": False}},
    {"memory_selection": {"strategy": "disabled", "omitted_count": 0, "selected_count": 0, "text": "private memory"}},
])
def test_answer_limitation_metadata_rejects_unknown_or_unbounded_values(store, context):
    with pytest.raises(ResearchStoreError):
        store.record_answer("#00001", str(uuid.uuid4()), [], {"valid": True}, "general", context_metadata=context)
    assert list(store.root.iterdir()) == []


def test_answer_context_serialized_byte_budget_is_checked_before_storage(store, monkeypatch):
    monkeypatch.setattr("docthinker.server.research_store.MAX_ANSWER_CONTEXT_BYTES", 10)
    with pytest.raises(ResearchLimitError):
        store.record_answer("#00001", str(uuid.uuid4()), [], {"valid": True}, "general",
                            context_metadata={"used_memory_count": 0})
    assert list(store.root.iterdir()) == []


def test_answer_limitation_metadata_idempotent_and_content_bound(store):
    answer_id = str(uuid.uuid4())
    context = {"retrieval_gaps": [{"code": "context_budget_omitted"}], "used_memory_count": 0,
               "memory_selection": {"strategy": "disabled", "omitted_count": 0, "selected_count": 0}}
    store.record_answer("#00001", answer_id, [], {"valid": True}, "general", context_metadata=context)
    same = {**context, "retrieval_gaps": [{"code": "context_budget_omitted", "message": "discard this"}]}
    assert store.record_answer("#00001", answer_id, [], {"valid": True}, "general", context_metadata=same)["recorded"]
    with pytest.raises(ResearchRevisionConflictError):
        store.record_answer("#00001", answer_id, [], {"valid": True}, "general", context_metadata={"used_memory_count": 1})


def test_old_answer_context_schema_reads_without_write_then_migrates_additively(store):
    legacy_id = str(uuid.uuid4())
    store.record_answer("#00001", legacy_id, [], {"valid": True}, "general")
    # Recreate precisely the original six-column table, not a new empty store.
    with sqlite3.connect(store.path) as db:
        db.executescript("""
            ALTER TABLE research_answers RENAME TO previous_answers;
            CREATE TABLE research_answers (
                answer_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                source_refs TEXT NOT NULL, validation TEXT NOT NULL,
                answer_mode TEXT NOT NULL, created_at REAL NOT NULL);
            INSERT INTO research_answers SELECT answer_id,session_id,source_refs,validation,answer_mode,created_at FROM previous_answers;
            DROP TABLE previous_answers;
        """)
    before = store.path.read_bytes()
    legacy = store.answer_metadata("#00001", [legacy_id])[legacy_id]
    assert legacy["answer_mode"] == "general" and "retrieval_gaps" not in legacy
    assert before == store.path.read_bytes()
    assert store.record_answer("#00001", legacy_id, [], {"valid": True}, "general")["recorded"]
    new_id = str(uuid.uuid4())
    store.record_answer("#00001", new_id, [], {"valid": True}, "general",
                        context_metadata={"retrieval_gaps": [{"code": "uncovered_query_terms"}], "used_memory_count": 0})
    with sqlite3.connect(store.path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(research_answers)")}
        old_context = db.execute("SELECT context_metadata_json FROM research_answers WHERE answer_id=?", (legacy_id,)).fetchone()[0]
    assert "context_metadata_json" in columns and old_context == "{}"
    assert "retrieval_gaps" not in store.answer_metadata("#00001", [legacy_id])[legacy_id]
    assert store.answer_metadata("#00001", [new_id])[new_id]["retrieval_gaps"][0]["code"] == "uncovered_query_terms"
