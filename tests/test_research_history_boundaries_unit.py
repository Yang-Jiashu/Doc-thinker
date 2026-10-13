"""No raw derived answer escapes via legacy/mobile/history/replay boundaries."""
from __future__ import annotations

from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from docthinker.server.research_history import (
    UNVERIFIABLE_RESEARCH_ANALYSIS, hide_unverified_research_reply, hydrate_research_history,
)

SID = "#00001"
ANSWER = "11111111-1111-4111-8111-111111111111"
PRIVATE = "SYNTHETIC-DERIVED-PRIVATE-CONTENT"


def marked():
    return {"id": ANSWER, "role": "assistant", "content": PRIVATE,
            "research_provenance_required": True, "research_core": True,
            "sources": [{"id": "source-1", "content": PRIVATE}],
            "reasoning": {"trace": {"path": PRIVATE}},
            "claim_audit": {"claims": [{"text": PRIVATE}]}}


def ordinary():
    return {"id": "ordinary-answer", "role": "assistant", "content": "Ordinary chat remains available."}


@pytest.fixture
def routes(monkeypatch):
    with patch("dotenv.load_dotenv", return_value=False):
        sessions = importlib.import_module("docthinker.server.routers.sessions")
        mobile = importlib.import_module("docthinker.server.routers.mobile")
        query = importlib.import_module("docthinker.server.routers.query")
    rows = [ordinary(), marked()]
    manager = SimpleNamespace(get_session=lambda sid: {"id": sid},
                              get_history=lambda sid: deepcopy(rows))
    # These routers share the same state singleton; never start app lifespans.
    assert sessions.state is mobile.state is query.state
    monkeypatch.setattr(sessions.state, "session_manager", manager)
    monkeypatch.setattr(sessions.state, "research_service", None)
    monkeypatch.setattr(mobile, "_feedback_store", lambda: SimpleNamespace(list_for_session=lambda sid: []))
    monkeypatch.setattr(mobile, "_query_journal", lambda: SimpleNamespace(
        path="synthetic-journal", recent_unresolved=lambda *a, **k: []))
    return SimpleNamespace(sessions=sessions, mobile=mobile, query=query, rows=rows)


def service_for(mode):
    calls = []
    def hydrate(sid, rows):
        calls.append((sid, deepcopy(rows)))
        if mode == "failed":
            raise OSError("synthetic private internal error")
        return [{**row, "content": "Live provenance revalidated."} if row.get("research_provenance_required") else row
                for row in rows]
    return SimpleNamespace(hydrate_history=hydrate), calls


def assert_hidden(row):
    assert PRIVATE not in str(row)
    assert row["sources"] == []
    assert row["citation_validation"]["valid"] is False
    assert row["citation_validation"]["warnings"][0]["code"] == "provenance_unavailable"
    assert row["research_provenance_required"] is True


def test_pure_redaction_is_idempotent_and_keeps_unmarked_history():
    row = marked()
    row["answer"] = PRIVATE
    before = deepcopy(row)
    hidden = hide_unverified_research_reply(row)
    assert_hidden(hidden)
    assert hidden["answer"] == hidden["content"] == UNVERIFIABLE_RESEARCH_ANALYSIS
    assert hide_unverified_research_reply(hidden) == hidden
    assert row == before
    assert hide_unverified_research_reply(ordinary()) == ordinary()


@pytest.mark.parametrize("service", [None, SimpleNamespace(hydrate_history=lambda *a: None),
                                     SimpleNamespace(hydrate_history=lambda *a: [])])
def test_disabled_or_malformed_hydration_only_hides_marked(service):
    rows = [ordinary(), marked()]
    result = hydrate_research_history(rows, SID, service)
    assert result[0] == rows[0]
    assert_hidden(result[1])
    assert result[1]["content"] == UNVERIFIABLE_RESEARCH_ANALYSIS


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "failed", "live"])
async def test_classic_history_endpoint_uses_same_provenance_boundary(routes, monkeypatch, mode):
    service, calls = service_for(mode)
    monkeypatch.setattr(routes.sessions.state, "research_service", service if mode != "disabled" else None)
    result = await routes.sessions.get_session_history(SID)
    assert result["history"][0] == ordinary()
    if mode == "live":
        assert result["history"][1]["content"] == "Live provenance revalidated."
    else:
        assert_hidden(result["history"][1])
    assert len(calls) == (mode != "disabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "failed", "live"])
async def test_mobile_history_cannot_resurface_marked_answer_after_feature_shutdown(routes, monkeypatch, mode):
    service, calls = service_for(mode)
    monkeypatch.setattr(routes.mobile.state, "research_service", service if mode != "disabled" else None)
    result = await routes.mobile.history(SID)
    assert result["history"][0] == ordinary()
    if mode == "live":
        assert result["history"][1]["content"] == "Live provenance revalidated."
    else:
        assert_hidden(result["history"][1])
    assert len(calls) == (mode != "disabled")


@pytest.mark.parametrize("mode", ["disabled", "failed", "live"])
def test_legacy_query_context_does_not_feed_raw_derived_answer_back_to_a_model(routes, monkeypatch, mode):
    service, calls = service_for(mode)
    monkeypatch.setattr(routes.query.state, "research_service", service if mode != "disabled" else None)
    result = routes.query._get_recent_conversation_history(SID)
    assert result[0]["content"] == ordinary()["content"]
    assert PRIVATE not in str(result)
    assert result[1]["content"] == ("Live provenance revalidated." if mode == "live" else UNVERIFIABLE_RESEARCH_ANALYSIS)
    assert len(calls) == (mode != "disabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "failed", "live"])
async def test_completed_journal_reply_does_not_merge_discarded_proof_fields_back(routes, monkeypatch, mode):
    service, calls = service_for(mode)
    if mode == "live":
        def hydrate(sid, rows):
            calls.append((sid, deepcopy(rows)))
            return [hide_unverified_research_reply(row) for row in rows]
        service.hydrate_history = hydrate
    monkeypatch.setattr(routes.mobile.state, "research_service", service if mode != "disabled" else None)
    raw = {**marked(), "answer": PRIVATE, "answer_id": ANSWER, "_research_session_id": SID,
           "request_id": "22222222-2222-4222-8222-222222222222", "request_status": "completed"}
    before = deepcopy(raw)
    result = await routes.mobile._hydrate_research_reply(raw)
    assert_hidden(result)
    assert result["answer"] == UNVERIFIABLE_RESEARCH_ANALYSIS
    assert result["request_status"] == "completed"
    assert result["request_id"] == raw["request_id"]
    assert "_research_session_id" not in result
    assert "reasoning" not in result and "claim_audit" not in result
    assert raw == before
    assert len(calls) == (mode != "disabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["answer_id", "_research_session_id"])
async def test_marked_receipt_without_rehydration_identity_fails_closed(routes, monkeypatch, missing):
    service, calls = service_for("live")
    monkeypatch.setattr(routes.mobile.state, "research_service", service)
    raw = {**marked(), "answer": PRIVATE, "answer_id": ANSWER, "_research_session_id": SID}
    raw.pop(missing)
    result = await routes.mobile._hydrate_research_reply(raw)
    assert_hidden(result)
    assert result["answer"] == UNVERIFIABLE_RESEARCH_ANALYSIS
    assert calls == []


@pytest.mark.asyncio
async def test_generic_completed_reply_survives_research_shutdown(routes):
    raw = {"answer": "Ordinary answer", "sources": [], "request_status": "completed", "request_id": "id"}
    assert await routes.mobile._hydrate_research_reply(raw) == raw


def test_query_response_preserves_marker_for_later_receipt_replay(routes):
    result = routes.mobile._query_response({**marked(), "answer": PRIVATE, "answer_id": ANSWER})
    assert result["research_provenance_required"] is True
    assert result["answer_id"] == ANSWER


@pytest.mark.asyncio
async def test_real_query_status_endpoint_redacts_completed_receipt_without_provider_call(routes, monkeypatch):
    receipt = {**marked(), "answer": PRIVATE, "answer_id": ANSWER, "_research_session_id": SID,
               "request_id": "22222222-2222-4222-8222-222222222222", "request_status": "completed"}
    journal = SimpleNamespace(path="synthetic-journal", read=Mock(return_value=receipt))
    monkeypatch.setattr(routes.mobile, "_query_journal", lambda: journal)
    result = await routes.mobile.query_status(SID, receipt["request_id"])
    assert_hidden(result)
    assert result["answer"] == UNVERIFIABLE_RESEARCH_ANALYSIS
    journal.read.assert_called_once()


@pytest.mark.asyncio
async def test_workbench_answer_projection_preserves_required_provenance_marker(routes):
    with patch("dotenv.load_dotenv", return_value=False):
        workbench = importlib.import_module("docthinker.server.routers.workbench")
    received = []
    def hydrate(sid, rows):
        received.extend(rows)
        return [hide_unverified_research_reply(row) for row in rows]
    answers = await workbench._answers(SID, routes.mobile.state.session_manager,
                                       SimpleNamespace(hydrate_history=hydrate), [ANSWER])
    assert received[0]["research_provenance_required"] is True
    assert answers[ANSWER]["content"] == UNVERIFIABLE_RESEARCH_ANALYSIS
    assert PRIVATE not in str(answers)
