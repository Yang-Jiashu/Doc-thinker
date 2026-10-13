"""Fail-closed boundaries for source-dependent history and receipt replay.

No SessionManager/service imports, database access, or model calls. The pure
redactor only hides explicitly marked derived answers; ordinary chat survives
research feature shutdown. A caller-provided live service may rehydrate them.
"""
from __future__ import annotations

UNVERIFIABLE_RESEARCH_ANALYSIS = (
    "本次分析需要核对原文依据，但当前核对服务不可用或依据无法确认，原回答及相关分析已收起。"
    "普通聊天仍可继续；恢复服务后可重新核对这条回答。"
)


def hide_unverified_research_reply(message):
    """Pure, idempotent redaction; retain request IDs/status and ordinary chat."""
    if not isinstance(message, dict) or message.get("research_provenance_required") is not True:
        return message
    result = {**message, "research_core": True, "sources": [],
              "citation_validation": {
                  "valid": False, "entailment_verified": False,
                  "warnings": [{"code": "provenance_unavailable",
                                "message": "依据暂时无法核对，已收起依赖来源的回答。"}]}}
    if "content" in result:
        result["content"] = UNVERIFIABLE_RESEARCH_ANALYSIS
    if "answer" in result:
        result["answer"] = UNVERIFIABLE_RESEARCH_ANALYSIS
    # Raw proof/claim snippets must not escape through another UI card after
    # the main answer was hidden. No placeholder claims semantic verification.
    result.pop("reasoning", None)
    result.pop("claim_audit", None)
    result.pop("knowledge_skill", None)
    result.pop("trial_delivery", None)
    return result


def hydrate_research_history(messages, session_id, service=None):
    """Revalidate via the current service, or hide marked answers on outage.

    This function does not reload a model or create a research service simply
    to replay history. Missing/disabled service, failed metadata reads, or an
    unusable session must never reactivate the raw persisted analysis.
    """
    if service is not None and session_id:
        try:
            hydrated = service.hydrate_history(session_id, messages)
            if not isinstance(hydrated, list) or len(hydrated) != len(messages):
                raise ValueError("Invalid hydrated history")
            return hydrated
        except Exception:
            # Do not expose diagnostic text or sacrifice generic chat because
            # a source-provenance read failed.
            pass
    return [hide_unverified_research_reply(message) for message in messages]
