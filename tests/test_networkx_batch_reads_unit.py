"""Batch-read compatibility and lock/reload-check counts for local graphs."""

import unittest
from unittest.mock import AsyncMock

import networkx as nx

from graphcore.coregraph.base import BaseGraphStorage
from graphcore.coregraph.kg.networkx_impl import NetworkXStorage


class NetworkXBatchReadsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_node("A", description="source")
        self.graph.add_node("empty")
        self.graph.add_edge("A", "B", relation="related")
        self.graph.add_edge("B", "empty")
        self.storage = object.__new__(NetworkXStorage)
        self.storage._get_graph = AsyncMock(return_value=self.graph)

    async def test_nodes_match_single_reads_including_empty_attributes_and_duplicates(
        self,
    ):
        names = ["A", "empty", "missing", "A"]
        expected = await BaseGraphStorage.get_nodes_batch(self.storage, names)
        self.assertEqual(len(names), self.storage._get_graph.await_count)
        self.storage._get_graph.reset_mock()

        actual = await self.storage.get_nodes_batch(names)
        self.assertEqual(expected, actual)
        self.assertEqual({}, actual["empty"])
        self.assertNotIn("missing", actual)
        self.assertIs(expected["A"], actual["A"])
        self.storage._get_graph.assert_awaited_once()
        self.graph.nodes["A"]["description"] = "updated"
        self.assertEqual("updated", actual["A"]["description"])

    async def test_edges_match_single_reads_and_keep_requested_orientation(self):
        pairs = [
            {"src": "A", "tgt": "B"},
            {"src": "B", "tgt": "A"},
            {"src": "B", "tgt": "empty"},
            {"src": "A", "tgt": "missing"},
            {"src": "A", "tgt": "B"},
        ]
        expected = await BaseGraphStorage.get_edges_batch(self.storage, pairs)
        self.assertEqual(len(pairs), self.storage._get_graph.await_count)
        self.storage._get_graph.reset_mock()

        actual = await self.storage.get_edges_batch(pairs)
        self.assertEqual(expected, actual)
        self.assertEqual({}, actual[("B", "empty")])
        self.assertNotIn(("A", "missing"), actual)
        self.assertIs(expected[("A", "B")], actual[("A", "B")])
        self.storage._get_graph.assert_awaited_once()
        self.graph.edges["A", "B"]["relation"] = "changed"
        self.assertEqual("changed", actual[("B", "A")]["relation"])

    async def test_adjacency_matches_single_reads_including_missing_nodes(self):
        names = ["A", "B", "empty", "missing", "A"]
        expected = await BaseGraphStorage.get_nodes_edges_batch(self.storage, names)
        self.assertEqual(len(names), self.storage._get_graph.await_count)
        self.storage._get_graph.reset_mock()

        actual = await self.storage.get_nodes_edges_batch(names)
        self.assertEqual(expected, actual)
        self.assertEqual([], actual["missing"])
        self.assertIsNot(expected["A"], actual["A"])
        self.storage._get_graph.assert_awaited_once()

    async def test_empty_batches_skip_graph_access(self):
        self.assertEqual({}, await self.storage.get_nodes_batch([]))
        self.assertEqual({}, await self.storage.get_edges_batch([]))
        self.assertEqual({}, await self.storage.get_nodes_edges_batch([]))
        self.storage._get_graph.assert_not_awaited()

    async def test_full_local_graph_budget_reduces_reload_checks_from_352_to_two(self):
        graph = nx.relabel_nodes(nx.gnm_random_graph(96, 256, seed=0), str)
        self.storage._get_graph.return_value = graph
        names = list(graph.nodes)
        pairs = [{"src": source, "tgt": target} for source, target in graph.edges]
        expected_nodes = await BaseGraphStorage.get_nodes_batch(self.storage, names)
        expected_edges = await BaseGraphStorage.get_edges_batch(self.storage, pairs)
        self.assertEqual(352, self.storage._get_graph.await_count)
        self.storage._get_graph.reset_mock()

        self.assertEqual(expected_nodes, await self.storage.get_nodes_batch(names))
        self.assertEqual(expected_edges, await self.storage.get_edges_batch(pairs))
        self.assertEqual(2, self.storage._get_graph.await_count)

    async def test_batches_recheck_graph_on_each_call_and_do_not_share_instances(self):
        other_graph = nx.Graph()
        other_graph.add_node("A", description="another session")
        other = object.__new__(NetworkXStorage)
        other._get_graph = AsyncMock(return_value=other_graph)
        self.assertEqual(
            "source", (await self.storage.get_nodes_batch(["A"]))["A"]["description"]
        )
        self.assertEqual(
            "another session", (await other.get_nodes_batch(["A"]))["A"]["description"]
        )

        replacement = nx.Graph()
        replacement.add_node("A", description="reloaded")
        self.storage._get_graph.return_value = replacement
        self.assertEqual(
            "reloaded", (await self.storage.get_nodes_batch(["A"]))["A"]["description"]
        )
        self.assertEqual(2, self.storage._get_graph.await_count)
        other._get_graph.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
