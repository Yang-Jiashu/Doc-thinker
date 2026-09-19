"""One payload contract for streaming and non-streaming UI proxies."""


def build_query_payload(data: dict) -> dict:
    mode = str(data.get("ui_mode") or "standard").lower()
    payload = {
        "question": data.get("question", data.get("text", "")),
        "mode": {"quick": "naive", "standard": "local", "deep": "mix"}.get(
            mode, "local"
        ),
        "enable_thinking": mode == "deep",
        "enable_rerank": mode != "quick",
        "enable_expanded_matching": mode == "deep",
        "enable_image_asset_activation": mode == "deep",
    }
    # Preserve values for Pydantic validation. bool("false") would incorrectly
    # re-enable a control; silently discarding budgets also breaks A/B tests.
    fields = (
        "session_id",
        "mode",
        "memory_mode",
        "use_memory",
        "use_conversation_context",
        "use_llm_cache",
        "use_self_evolution",
        "evolution_mode",
        "remember_turn",
        "memory_excluded_layers",
        "memory_write_scope",
        "retrieval_instruction",
        "enable_thinking",
        "enable_rerank",
        "enable_expanded_matching",
        "expanded_top_k",
        "expanded_min_score",
        "include_discovered_edges",
        "top_k",
        "chunk_top_k",
        "max_relation_tokens",
        "max_total_tokens",
        "max_relations",
        "max_discovered_relations",
        "min_discovered_edge_confidence",
        "require_discovered_evidence",
        "enable_image_asset_activation",
        "image_activation_threshold",
        "image_activation_top_k",
        "adaptive_context",
        "enable_path_completion",
        "max_history_tokens",
        "max_auxiliary_tokens",
    )
    payload.update({key: data[key] for key in fields if key in data})
    return payload
