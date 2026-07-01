import unittest
from types import SimpleNamespace

from docthinker.harness import QueryHarness
from docthinker.memory_core.core import MemoryTrace, RecallBundle


class _MemoryCore:
    def __init__(self):
        self.calls = []

    async def recall(self, **kwargs):
        self.calls.append(kwargs)
        trace = MemoryTrace(memory_mode="session", memory_hits=1)
        return RecallBundle(
            retrieval_instruction="memory instruction",
            memory_summaries=[{"source": "test"}],
            episodic_matches=[{"id": "episode"}],
            expanded_matches=[{"name": "expanded"}],
            long_horizon_matches=[],
            memory_reasoning={},
            trace=trace,
        )


def _request(**overrides):
    values = {
        "question": "why",
        "session_id": "#00001",
        "mode": "local",
        "retrieval_instruction": "base",
        "enable_thinking": True,
        "enable_expanded_matching": True,
        "expanded_top_k": 2,
        "expanded_min_score": 0.2,
        "use_memory": True,
        "use_conversation_context": True,
        "use_llm_cache": True,
        "use_self_evolution": True,
        "remember_turn": True,
        "enable_rerank": True,
        "top_k": 20,
        "chunk_top_k": 12,
        "max_relation_tokens": 5000,
        "max_total_tokens": 24000,
        "include_discovered_edges": True,
        "max_relations": 32,
        "max_discovered_relations": 8,
        "min_discovered_edge_confidence": 0.8,
        "require_discovered_evidence": True,
        "enable_image_asset_activation": True,
        "image_activation_threshold": 0.62,
        "image_activation_top_k": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class QueryHarnessUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_optional_context_can_be_disabled(self):
        memory = _MemoryCore()
        history_calls = []
        harness = QueryHarness(
            memory_core_factory=lambda: memory,
            history_loader=lambda sid: history_calls.append(sid) or [{"role": "user", "content": "old"}],
        )
        request = _request(
            use_memory=False,
            use_conversation_context=False,
            use_llm_cache=False,
            use_self_evolution=False,
        )

        context = await harness.prepare(request=request)
        options = context.graph_query_options(request)

        self.assertEqual([], memory.calls)
        self.assertEqual([], history_calls)
        self.assertEqual([], context.conversation_history)
        self.assertFalse(options["use_llm_cache"])
        self.assertFalse(options["include_discovered_edges"])
        self.assertFalse(harness.should_enrich(context))
        self.assertEqual(
            False,
            context.controls.to_schema()["use_conversation_context"],
        )

    async def test_enabled_run_uses_memory_history_and_evolution(self):
        memory = _MemoryCore()
        harness = QueryHarness(
            memory_core_factory=lambda: memory,
            history_loader=lambda _sid: [{"role": "assistant", "content": "old"}],
        )
        request = _request()

        context = await harness.prepare(request=request)
        options = context.graph_query_options(request)

        self.assertEqual(1, len(memory.calls))
        self.assertTrue(memory.calls[0]["enable_expanded_matching"])
        self.assertEqual(1, len(context.conversation_history))
        self.assertTrue(options["include_discovered_edges"])
        self.assertTrue(options["use_llm_cache"])
        self.assertTrue(harness.should_enrich(context))

    async def test_self_evolution_master_switch_disables_expanded_matching(self):
        memory = _MemoryCore()
        harness = QueryHarness(
            memory_core_factory=lambda: memory,
            history_loader=lambda _sid: [],
        )

        await harness.prepare(request=_request(use_self_evolution=False))

        self.assertFalse(memory.calls[0]["enable_expanded_matching"])


if __name__ == "__main__":
    unittest.main()
