from types import SimpleNamespace

from docthinker.server.routers.query import (
    _build_conversation_fallback_prompt,
    _is_identity_query,
    _needs_conversation_fallback,
)


def test_no_context_response_uses_conversation_fallback():
    assert _needs_conversation_fallback(
        "Sorry, I'm not able to provide an answer to that question.[no-context]"
    )
    assert not _needs_conversation_fallback("这是正常回答。")


def test_docthinker_name_is_not_mistaken_for_hi():
    assert _is_identity_query("你好")
    assert _is_identity_query("who are you")
    assert not _is_identity_query("按照我的表达偏好，给 DocThinker 写一句宣传语。")


def test_fallback_prompt_keeps_memory_and_deduplicates_current_turn():
    context = SimpleNamespace(
        conversation_history=[
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": "当前问题"},
        ],
        retrieval_instruction="表达克制且可验证",
    )
    prompt = _build_conversation_fallback_prompt("当前问题", context)
    assert "表达克制且可验证" in prompt
    assert prompt.count("当前问题") == 1
