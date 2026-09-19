import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


class _Graph:
    async def get_nodes_batch(self, names):
        return {name: {"id": name} for name in names}

    async def get_nodes_edges_batch(self, names):
        return {name: [("A", "B")] for name in names}

    async def get_edges_batch(self, pairs):
        return {("A", "B"): (await self.get_all_edges())[0]}

    async def get_all_nodes(self):
        return [{"id": "A"}, {"id": "B"}]

    async def get_all_edges(self):
        return [
            {
                "source": "A",
                "target": "B",
                "keywords": "导致",
                "description": "A导致B",
                "source_id": "chunk-a",
            }
        ]


class _GraphCore:
    chunk_entity_relation_graph = _Graph()
    text_chunks = None
    entities_vdb = SimpleNamespace(
        query=AsyncMock(return_value=[{"entity_name": "A"}, {"entity_name": "B"}])
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
            history_loader=lambda sid: (
                history_calls.append(sid) or [{"role": "user", "content": "old"}]
            ),
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
        self.assertFalse(memory.calls[0]["enable_expanded_matching"])
        self.assertFalse(memory.calls[0]["enable_cognition"])
        self.assertEqual(1, len(context.conversation_history))
        self.assertFalse(options["include_discovered_edges"])
        self.assertTrue(options["use_llm_cache"])
        self.assertTrue(harness.should_enrich(context))

    async def test_only_explore_mode_opens_broad_discovered_edge_pool(self):
        memory = _MemoryCore()
        harness = QueryHarness(
            memory_core_factory=lambda: memory,
            history_loader=lambda _sid: [],
        )
        request = _request(
            question="还可能有哪些潜在影响？",
            evolution_mode="explore",
        )

        context = await harness.prepare(request=request)

        self.assertTrue(
            context.graph_query_options(request)["include_discovered_edges"]
        )

    async def test_self_evolution_master_switch_disables_expanded_matching(self):
        memory = _MemoryCore()
        harness = QueryHarness(
            memory_core_factory=lambda: memory,
            history_loader=lambda _sid: [],
        )

        await harness.prepare(request=_request(use_self_evolution=False))

        self.assertFalse(memory.calls[0]["enable_expanded_matching"])

    async def test_path_policy_adds_contiguous_graph_instruction(self):
        memory = _MemoryCore()
        harness = QueryHarness(
            memory_core_factory=lambda: memory,
            history_loader=lambda _sid: [],
        )
        request = _request(
            question="A为什么导致B？",
            evolution_mode="path",
        )
        context = await harness.prepare(request=request)

        await harness.enrich_graph_reasoning(
            context=context,
            graphcore=_GraphCore(),
            question=request.question,
        )

        self.assertTrue(context.graph_reasoning["applied"])
        self.assertEqual("path", context.question_policy.mode)
        self.assertIn("A → B", context.retrieval_instruction)

    async def test_explicit_opt_out_survives_exploration(self):
        harness = QueryHarness(
            memory_core_factory=_MemoryCore, history_loader=lambda _: []
        )
        request = _request(evolution_mode="explore", include_discovered_edges=False)
        context = await harness.prepare(request=request)
        self.assertFalse(
            context.graph_query_options(request)["include_discovered_edges"]
        )
        self.assertFalse(context.allow_graph_candidates)

    async def test_off_mode_does_not_recall_derived_knowledge(self):
        memory = _MemoryCore()
        harness = QueryHarness(
            memory_core_factory=lambda: memory, history_loader=lambda _: []
        )
        await harness.prepare(request=_request(evolution_mode="off"))
        self.assertFalse(memory.calls[0]["enable_expanded_matching"])
        self.assertFalse(memory.calls[0]["enable_cognition"])

    async def test_context_budget_keeps_recent_history_and_user_limits(self):
        history = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "recent"},
        ]
        harness = QueryHarness(
            memory_core_factory=_MemoryCore, history_loader=lambda _: history
        )
        request = _request(
            use_memory=False,
            retrieval_instruction="a" * 10000,
            max_history_tokens=100,
            max_auxiliary_tokens=256,
            top_k=3,
        )
        context = await harness.prepare(request=request)
        self.assertEqual("recent", context.conversation_history[-1]["content"])
        self.assertLessEqual(context.budget_trace["history_tokens"], 100)
        self.assertLessEqual(context.budget_trace["instruction_tokens"], 256)
        self.assertEqual(3, context.graph_query_options(request)["top_k"])
        self.assertEqual(2000, len(history[0]["content"]))

    async def test_graph_failure_degrades_without_llm_call(self):
        harness = QueryHarness(
            memory_core_factory=_MemoryCore, history_loader=lambda _: []
        )
        context = await harness.prepare(request=_request(evolution_mode="path"))
        graphcore = SimpleNamespace(
            chunk_entity_relation_graph=_Graph(),
            entities_vdb=SimpleNamespace(
                query=AsyncMock(side_effect=RuntimeError("offline"))
            ),
        )
        llm = AsyncMock()
        await harness.enrich_graph_reasoning(
            context=context, graphcore=graphcore, question="A为什么导致B", llm_func=llm
        )
        self.assertFalse(context.graph_reasoning["applied"])
        self.assertEqual(
            "graph_unavailable", context.graph_reasoning["diagnostic"]["reason"]
        )
        llm.assert_not_awaited()

    async def test_missing_path_does_not_call_llm_by_default(self):
        harness = QueryHarness(
            memory_core_factory=_MemoryCore, history_loader=lambda _: []
        )
        context = await harness.prepare(
            request=_request(evolution_mode="path", use_memory=False)
        )
        graph = _Graph()
        graph.get_nodes_edges_batch = AsyncMock(return_value={})
        graph.get_edges_batch = AsyncMock(return_value={})
        graphcore = SimpleNamespace(
            chunk_entity_relation_graph=graph,
            entities_vdb=_GraphCore.entities_vdb,
            text_chunks=None,
        )
        llm = AsyncMock()
        await harness.enrich_graph_reasoning(
            context=context, graphcore=graphcore, question="A为什么导致B", llm_func=llm
        )
        llm.assert_not_awaited()
        self.assertEqual(
            "disabled", context.graph_reasoning["diagnostic"]["completion"]["reason"]
        )

    async def test_local_graph_fetch_is_bounded(self):
        graph = SimpleNamespace(
            get_nodes_edges_batch=AsyncMock(
                return_value={"A": [("A", f"N{i}") for i in range(1000)]}
            ),
            get_nodes_batch=AsyncMock(
                side_effect=lambda names: {name: {"id": name} for name in names}
            ),
            get_edges_batch=AsyncMock(
                side_effect=lambda pairs: {
                    (p["src"], p["tgt"]): {"keywords": "related"} for p in pairs
                }
            ),
            get_all_nodes=AsyncMock(side_effect=AssertionError("full scan")),
        )
        core = SimpleNamespace(
            chunk_entity_relation_graph=graph,
            entities_vdb=SimpleNamespace(
                query=AsyncMock(return_value=[{"entity_name": "A"}])
            ),
        )
        nodes, edges = await QueryHarness._load_local_graph(core, "A")
        self.assertLessEqual(len(nodes), 96)
        self.assertLessEqual(len(edges), 256)
        graph.get_all_nodes.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
