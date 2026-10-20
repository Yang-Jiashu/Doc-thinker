"""Real temporary research stores and HTTP workbench facade, no paid providers."""
import json
import sqlite3
import uuid

import pytest

from tests.test_research_mobile_unit import mobile as research_mobile, mobile_client, upload_ready, SID, OTHER_SID, PREFIX
from docthinker.server.workbench_store import WorkbenchStore

BRIEF = {"session_id": SID, "expected_revision": 0, "goal": "写综述", "output_format": "比较方法与限制", "constraints": "引用原文"}


@pytest.fixture
def mobile(research_mobile):
    research_mobile.manager.get_session.side_effect = lambda sid: {"id": sid, "instance_id": "instance-" + sid[1:]} if sid in research_mobile.manager.histories else None
    return research_mobile


async def prepared_answer(mobile, browser):
    document_id = await upload_ready(mobile, browser)
    # Legacy literal evidence path exercises actual source/answer persistence
    # and revocation without requesting an answer model or local embeddings.
    response = await browser.post(PREFIX + "/query", json={"session_id": SID, "question": "renewable supply imports", "reasoning_mode": "evidence"})
    assert response.status_code == 200, response.text
    return document_id, response.json()


async def test_brief_works_with_model_and_research_disabled(mobile, monkeypatch):
    monkeypatch.setattr(mobile.state, "research_service", None)
    monkeypatch.setattr(mobile.state, "mobile_chat_func", None)
    async with mobile_client(mobile) as browser:
        caps = (await browser.get(PREFIX + "/capabilities")).json()["workbench"]
        assert caps == {"brief": True, "artifacts": False, "max_artifacts_per_session": 100}
        empty = await browser.get(PREFIX + "/workbench/brief", params={"session_id": SID})
        assert empty.json()["brief"]["revision"] == 0
        response = await browser.post(PREFIX + "/workbench/brief", json=BRIEF)
        assert response.status_code == 200
        saved = response.json()["brief"]
        assert saved["session_id"] == SID and saved["revision"] == 1
        assert (await browser.post(PREFIX + "/workbench/brief", json=BRIEF)).status_code == 409
        assert (await browser.get(PREFIX + "/workbench/brief", params={"session_id": OTHER_SID})).json()["brief"]["revision"] == 0
        assert (await browser.get(PREFIX + "/workbench/artifacts", params={"session_id": SID})).status_code == 503
    mobile.model.assert_not_called()
    assert not mobile.manager.histories[SID]


async def test_saved_artifact_hydrates_then_honors_source_deletion(mobile):
    async with mobile_client(mobile) as browser:
        document_id, answer = await prepared_answer(mobile, browser)
        payload = {"session_id": SID, "answer_id": answer["answer_id"], "title": "证据摘录"}
        saved = await browser.post(PREFIX + "/workbench/artifacts", json=payload)
        assert saved.status_code == 200, saved.text
        artifact = saved.json()["artifact"]
        assert artifact["available"] and artifact["answer"]["role"] == "assistant"
        assert artifact["answer"]["sources"][0]["content"] == answer["sources"][0]["content"]
        assert artifact["answer"]["content"] == answer["answer"]
        repeated = await browser.post(PREFIX + "/workbench/artifacts", json=payload)
        assert repeated.json()["artifact"]["updated_at"] == artifact["updated_at"]
        listing = await browser.get(PREFIX + "/workbench/artifacts", params={"session_id": SID})
        assert "answer" not in listing.json()["items"][0]
        assert "PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in listing.text
        other = await browser.get(PREFIX + "/workbench/artifacts", params={"session_id": OTHER_SID})
        assert other.json() == {"items": [], "limit": 100}
        assert (await browser.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": document_id})).status_code == 200
        read = (await browser.get(PREFIX + "/workbench/artifact", params={"session_id": SID, "answer_id": answer["answer_id"]})).json()["artifact"]
        assert read["answer"] is None and read["available"] is False
        assert "PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in json.dumps(read)
        assert read["unavailable_reason"] == "source_revoked"
    path = WorkbenchStore(mobile.manager.base_storage_path).path
    assert b"PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in path.read_bytes()
    mobile.model.assert_not_called()


async def test_normal_generated_answer_with_raw_quote_is_hidden_after_revocation(mobile):
    async with mobile_client(mobile) as browser:
        document_id = await upload_ready(mobile, browser)
        mobile.model.return_value = "原文明确写道 PRIVATE_SOURCE_QUOTE_TOKEN_ONLY [S1]."
        response = await browser.post(PREFIX + "/query", json={"session_id": SID, "question": "renewable supply imports"})
        assert response.status_code == 200, response.text
        aid = response.json()["answer_id"]
        payload = {"session_id": SID, "answer_id": aid, "title": "generated answer"}
        assert (await browser.post(PREFIX + "/workbench/artifacts", json=payload)).status_code == 200
        assert (await browser.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": document_id})).status_code == 200
        result = await browser.get(PREFIX + "/workbench/artifact", params={"session_id": SID, "answer_id": aid})
        artifact = result.json()["artifact"]
        assert artifact["available"] is False and artifact["answer"] is None
        assert artifact["unavailable_reason"] == "source_revoked"
        assert "PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in result.text
        assert (await browser.post(PREFIX + "/workbench/artifacts", json=payload)).status_code == 503
    mobile.model.assert_awaited_once()


async def test_artifact_gallery_does_not_hydrate_answers(mobile, monkeypatch):
    async with mobile_client(mobile) as browser:
        _, answer = await prepared_answer(mobile, browser)
        await browser.post(PREFIX + "/workbench/artifacts", json={"session_id": SID, "answer_id": answer["answer_id"], "title": "pointer"})
        def forbidden(*_args):
            raise AssertionError("Gallery must not hydrate full answers")
        monkeypatch.setattr(mobile.service, "hydrate_history", forbidden)
        listing = await browser.get(PREFIX + "/workbench/artifacts", params={"session_id": SID})
        assert listing.status_code == 200
        assert listing.json()["items"][0]["available"] is True
        assert "answer" not in listing.json()["items"][0]


async def test_artifact_detail_validates_id_and_requires_saved_pointer(mobile):
    async with mobile_client(mobile) as browser:
        for answer_id, status in (("bad", 422), (str(uuid.uuid4()), 404)):
            response = await browser.get(PREFIX + "/workbench/artifact", params={"session_id": SID, "answer_id": answer_id})
            assert response.status_code == status


async def test_old_bookmark_outside_last_200_turns_still_hydrates(mobile):
    async with mobile_client(mobile) as browser:
        _, answer = await prepared_answer(mobile, browser)
        for index in range(220):
            mobile.manager.add_message(SID, "assistant", "Unrelated later answer " + str(index))
        saved = await browser.post(PREFIX + "/workbench/artifacts", json={"session_id": SID, "answer_id": answer["answer_id"], "title": "old result"})
        assert saved.status_code == 200
        assert saved.json()["artifact"]["answer"]["sources"]
    mobile.model.assert_not_called()


@pytest.mark.parametrize("tables", [("research_answers",), ("research_answer_evidence",),
                                    ("research_answers", "research_answer_evidence")])
async def test_provenance_metadata_loss_never_returns_raw_answer(mobile, tables):
    async with mobile_client(mobile) as browser:
        _, answer = await prepared_answer(mobile, browser)
        payload = {"session_id": SID, "answer_id": answer["answer_id"], "title": "result"}
        await browser.post(PREFIX + "/workbench/artifacts", json=payload)
        with sqlite3.connect(mobile.service.store.path) as db:
            for table in tables:
                db.execute("DELETE FROM " + table + " WHERE answer_id=?", (answer["answer_id"],))
        response = await browser.get(PREFIX + "/workbench/artifact", params={"session_id": SID, "answer_id": answer["answer_id"]})
        artifact = response.json()["artifact"]
        assert artifact["available"] is False and artifact["answer"] is None
        assert artifact["unavailable_reason"] == "provenance_unavailable"
        assert "PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in response.text
        assert (await browser.post(PREFIX + "/workbench/artifacts", json=payload)).status_code == 503


async def test_missing_history_disables_bookmark_without_deleting_it(mobile):
    async with mobile_client(mobile) as browser:
        _, answer = await prepared_answer(mobile, browser)
        await browser.post(PREFIX + "/workbench/artifacts", json={"session_id": SID, "answer_id": answer["answer_id"], "title": "result"})
        mobile.manager.histories[SID].clear()
        response = await browser.get(PREFIX + "/workbench/artifacts", params={"session_id": SID})
        assert response.json()["items"][0]["available"] is False
        response = await browser.get(PREFIX + "/workbench/artifact", params={"session_id": SID, "answer_id": answer["answer_id"]})
        assert response.json()["artifact"]["answer"] is None


async def test_cross_session_user_turn_and_fabricated_answer_ids_rejected(mobile):
    user_id = mobile.manager.add_message(SID, "user", "user message")
    other_answer = mobile.manager.add_message(OTHER_SID, "assistant", "other session private message")
    async with mobile_client(mobile) as browser:
        for answer_id in (user_id, other_answer, str(uuid.uuid4())):
            response = await browser.post(PREFIX + "/workbench/artifacts", json={"session_id": SID, "answer_id": answer_id, "title": "bad"})
            assert response.status_code == 404
        response = await browser.get(PREFIX + "/workbench/brief", params={"session_id": "#09999"})
        assert response.status_code == 404
    assert not WorkbenchStore(mobile.manager.base_storage_path).path.exists()


@pytest.mark.parametrize("changes", [{"user_id": "other"}, {"goal": "x" * 1201}, {"output_format": "x" * 401},
                                    {"constraints": "x" * 1201}, {"expected_revision": True}, {"goal": "\x00"},
                                    {"goal": 1}, {"session_id": "../x"}])
async def test_strict_brief_validation(mobile, changes):
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/workbench/brief", json={**BRIEF, **changes})
        assert response.status_code == 422
    assert not WorkbenchStore(mobile.manager.base_storage_path).path.exists()


@pytest.mark.parametrize("changes", [{"content": "forged answer"}, {"sources": []}, {"title": " "},
                                    {"title": "x" * 121}, {"answer_id": "not-uuid"}, {"title": None}])
async def test_artifact_input_cannot_include_source_or_answer_text(mobile, changes):
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/workbench/artifacts", json={"session_id": SID, "answer_id": str(uuid.uuid4()), "title": "result", **changes})
        assert response.status_code == 422


async def test_delete_is_idempotent_and_allowed_without_research(mobile, monkeypatch):
    aid = str(uuid.uuid4())
    store = WorkbenchStore(mobile.manager.base_storage_path, session_instance_id="instance-" + SID[1:])
    store.save_artifact(SID, aid, "pointer only")
    monkeypatch.setattr(mobile.state, "research_service", None)
    async with mobile_client(mobile) as browser:
        for _ in range(2):
            response = await browser.post(PREFIX + "/workbench/artifacts/delete", json={"session_id": SID, "answer_id": aid})
            assert response.json() == {"removed": True}
    assert store.list_artifacts(SID) == []
