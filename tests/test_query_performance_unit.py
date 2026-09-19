"""Deterministic operation-count and IO-overlap checks, without model calls."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from docthinker import reasoning_policy
from docthinker.harness import QueryControls, QueryHarness, QueryRunContext


def _core(graph, seeds):
    return SimpleNamespace(
        chunk_entity_relation_graph=graph,
        entities_vdb=SimpleNamespace(
            query=AsyncMock(return_value=[{"entity_name": name} for name in seeds])
        ),
    )


def _graph(neighborhoods):
    return SimpleNamespace(
        get_nodes_edges_batch=AsyncMock(return_value=neighborhoods),
        get_nodes_batch=AsyncMock(
            side_effect=lambda names: {name: {"description": name} for name in names}
        ),
        get_edges_batch=AsyncMock(
            side_effect=lambda pairs: {
                (pair["src"], pair["tgt"]): {"keywords": "related"} for pair in pairs
            }
        ),
    )


class LocalGraphIOPerformanceTest(unittest.IsolatedAsyncioTestCase):
    async def test_node_and_edge_reads_overlap_without_timing_assumptions(self):
        nodes_started, edges_started = asyncio.Event(), asyncio.Event()
        graph = _graph({"A": [("A", "B")]})

        async def nodes(names):
            nodes_started.set()
            await edges_started.wait()
            return {name: {"description": name} for name in names}

        async def edges(pairs):
            edges_started.set()
            await nodes_started.wait()
            return {
                (pair["src"], pair["tgt"]): {"keywords": "related"} for pair in pairs
            }

        graph.get_nodes_batch.side_effect = nodes
        graph.get_edges_batch.side_effect = edges
        result_nodes, result_edges = await asyncio.wait_for(
            QueryHarness._load_local_graph(_core(graph, ["A"]), "A"), timeout=1
        )
        self.assertEqual({"A", "B"}, {node["id"] for node in result_nodes})
        self.assertEqual(
            [("A", "B")], [(edge["source"], edge["target"]) for edge in result_edges]
        )
        graph.get_nodes_batch.assert_awaited_once()
        graph.get_edges_batch.assert_awaited_once()

    async def test_failed_batch_cancels_sibling_before_returning(self):
        nodes_started, nodes_cancelled = asyncio.Event(), asyncio.Event()
        graph = _graph({"A": [("A", "B")]})

        async def nodes(_names):
            nodes_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                nodes_cancelled.set()

        async def edges(_pairs):
            await nodes_started.wait()
            raise RuntimeError("storage unavailable")

        graph.get_nodes_batch.side_effect = nodes
        graph.get_edges_batch.side_effect = edges
        with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
            await QueryHarness._load_local_graph(_core(graph, ["A"]), "A")
        self.assertTrue(nodes_cancelled.is_set())

    async def test_empty_retrieval_does_not_contact_graph_storage(self):
        graph = _graph({})
        self.assertEqual(
            ([], []), await QueryHarness._load_local_graph(_core(graph, []), "unknown")
        )
        graph.get_nodes_edges_batch.assert_not_awaited()
        graph.get_nodes_batch.assert_not_awaited()
        graph.get_edges_batch.assert_not_awaited()

    async def test_no_edges_skips_empty_edge_attribute_read(self):
        graph = _graph({})
        nodes, edges = await QueryHarness._load_local_graph(_core(graph, ["A"]), "A")
        self.assertEqual(["A"], [node["id"] for node in nodes])
        self.assertEqual([], edges)
        graph.get_edges_batch.assert_not_awaited()

    async def test_edge_budget_stops_extra_neighborhood_rounds(self):
        seeds = [f"S{index}" for index in range(8)]
        graph = _graph(
            {seed: [(seed, f"N{index:03d}") for index in range(88)] for seed in seeds}
        )
        nodes, edges = await QueryHarness._load_local_graph(_core(graph, seeds), "S0")
        self.assertEqual(256, len(edges))
        self.assertLessEqual(len(nodes), 96)
        graph.get_nodes_edges_batch.assert_awaited_once_with(seeds)

    async def test_faithful_mode_performs_no_optional_graph_io(self):
        graph = _graph({})
        core = _core(graph, ["A"])
        harness = QueryHarness(
            memory_core_factory=lambda: None, history_loader=lambda _: []
        )
        context = QueryRunContext(controls=QueryControls(use_self_evolution=False))
        await harness.enrich_graph_reasoning(
            context=context, graphcore=core, question="A"
        )
        core.entities_vdb.query.assert_not_awaited()
        self.assertFalse(context.graph_reasoning["applied"])


class PathScoringPerformanceTest(unittest.TestCase):
    @staticmethod
    def graph():
        names = [f"Node{index}" for index in range(5)]
        edges = [
            {
                "source": left,
                "target": right,
                "keywords": "causes",
                "source_id": f"chunk-{index}",
            }
            for index, (left, right) in enumerate(zip(names, names[1:]))
        ]
        return [{"id": name} for name in names], edges

    def test_each_edge_is_scored_once_without_changing_path_results(self):
        nodes, edges = self.graph()
        question = "Node0如何导致Node4？"
        original_score = reasoning_policy._path_score
        original_quality = reasoning_policy._edge_quality

        def recompute_score(path, query, **_kwargs):
            return original_score(path, query)

        with patch.object(
            reasoning_policy, "_edge_quality", wraps=original_quality
        ) as score_calls:
            with patch.object(
                reasoning_policy, "_path_score", side_effect=recompute_score
            ):
                baseline, baseline_diagnostic = (
                    reasoning_policy.evidence_constrained_paths(question, nodes, edges)
                )
            recomputed_count = score_calls.call_count

        with patch.object(
            reasoning_policy, "_edge_quality", wraps=original_quality
        ) as score_calls:
            optimized, diagnostic = reasoning_policy.evidence_constrained_paths(
                question, nodes, edges
            )
            self.assertEqual(len(edges), score_calls.call_count)
            self.assertGreater(recomputed_count, 4 * score_calls.call_count)

        self.assertTrue(optimized)
        self.assertEqual(
            [path.to_schema() for path in baseline],
            [path.to_schema() for path in optimized],
        )
        self.assertEqual(baseline_diagnostic, diagnostic)

    def test_scores_are_not_reused_between_queries_or_graph_changes(self):
        nodes, edges = self.graph()
        original_quality = reasoning_policy._edge_quality
        with patch.object(
            reasoning_policy, "_edge_quality", wraps=original_quality
        ) as score_calls:
            reasoning_policy.evidence_constrained_paths(
                "Node0如何导致Node4？", nodes, edges
            )
            edges[0]["description"] = "changed session-local evidence"
            reasoning_policy.evidence_constrained_paths(
                "Node0和Node4的原因链？", nodes, edges
            )
            self.assertEqual(2 * len(edges), score_calls.call_count)


if __name__ == "__main__":
    unittest.main()
