import json
import unittest
from types import SimpleNamespace

from docthinker.kg_expansion.eclrr_v4 import ECLRRConfig, run_eclrr_v4
from docthinker.kg_expansion.eclrr_v4.llm_pipeline import (
    parse_judge_decision,
    parse_proposals,
)
from docthinker.kg_expansion.eclrr_v4.models import GateResult
from docthinker.kg_expansion.eclrr_v4.writeback import commit_promotion
from docthinker.retrieval_policy import select_relations_for_query
from graphcore.coregraph.operate import _find_related_text_unit_from_relations


class FakeGraph:
    workspace = "test"
    namespace = "graph"

    def __init__(self, edge=None, fail_commit=False):
        self.edge = dict(edge) if edge else None
        self.fail_commit = fail_commit

    async def get_edge(self, source, target):
        return dict(self.edge) if self.edge is not None else None

    async def upsert_edge(self, source, target, data):
        self.edge = dict(data)

    async def remove_edges(self, pairs):
        self.edge = None

    async def index_done_callback(self, force_save=False):
        if self.fail_commit:
            self.fail_commit = False
            return False
        return True


class FakeVDB:
    def __init__(self, fail_upsert=False, fail_commit=False):
        self.data = {}
        self.fail_upsert = fail_upsert
        self.fail_commit = fail_commit

    async def get_by_id(self, value):
        return self.data.get(value)

    async def upsert(self, values):
        if self.fail_upsert:
            self.fail_upsert = False
            raise RuntimeError("vdb upsert failed")
        self.data.update(values)

    async def delete(self, ids):
        for value in ids:
            self.data.pop(value, None)

    async def index_done_callback(self):
        if self.fail_commit:
            self.fail_commit = False
            return False
        return True


class FullGraph:
    workspace = "test"
    namespace = "graph"

    def __init__(self):
        self.nodes = [{"id": name, "entity_type": "person"} for name in "ABCD"]
        self.edges = {}
        for left, right in zip("ABC", "BCD"):
            self.edges[tuple(sorted((left, right)))] = {
                "source": left,
                "target": right,
                "relation": "causes",
                "relation_family": "causation",
                "direction": "source_to_target",
                "description": f"{left} directly causes {right}",
                "source_id": f"chunk-{left}{right}",
            }

    async def get_all_nodes(self):
        return list(self.nodes)

    async def get_all_edges(self):
        return [dict(edge) for edge in self.edges.values()]

    async def get_edge(self, source, target):
        edge = self.edges.get(tuple(sorted((source, target))))
        return dict(edge) if edge else None

    async def upsert_edge(self, source, target, data):
        self.edges[tuple(sorted((source, target)))] = {
            "source": source,
            "target": target,
            **dict(data),
        }

    async def remove_edges(self, pairs):
        for source, target in pairs:
            self.edges.pop(tuple(sorted((source, target))), None)

    async def index_done_callback(self, force_save=False):
        return True


class FullChunks:
    global_config = {"kg_chunk_pick_method": "WEIGHT", "related_chunk_number": 10}
    embedding_func = None

    def __init__(self):
        self.values = {
            f"chunk-{left}{right}": {
                "content": f"{left} directly causes {right} in the archived event record.",
                "file_path": "novel.txt",
            }
            for left, right in zip("ABC", "BCD")
        }

    async def get_by_ids(self, ids):
        return [
            (
                {"chunk_id": chunk_id, **self.values[chunk_id]}
                if chunk_id in self.values
                else None
            )
            for chunk_id in ids
        ]


def promotion(action="create", relation="indirectly_causes"):
    return GateResult(
        action=action,
        reason="gate_passed",
        review_id="review-1",
        source="A",
        target="D",
        relation=relation,
        relation_family="causation",
        direction="source_to_target",
        description="A indirectly causes D.",
        canonical_key=f"A|causation|{relation}|source_to_target|D",
        relation_id=f"rel-{relation}",
        evidence_chain=[
            {
                "edge_id": "e1",
                "chunk_id": "c1",
                "quote": "A causes B",
                "start": 0,
                "end": 10,
            }
        ],
        evidence_chunk_ids=["c1"],
        path_used=["A", "B", "C", "D"],
        proposal={"review_id": "review-1"},
        judge_decision={
            "decision": "accept",
            "evidence_coverage": 4,
            "semantic_composability": 3,
            "relation_direction": 2,
            "uncertainty_calibration": 1,
            "total": 10,
        },
    )


class StrictJsonTest(unittest.TestCase):
    def test_generator_and_judge_require_complete_strict_json(self):
        valid = json.dumps(
            {
                "proposals": [
                    {
                        "review_id": "r1",
                        "source": "A",
                        "target": "D",
                        "relation": "indirectly_causes",
                        "relation_family": "causation",
                        "direction": "source_to_target",
                        "description": "supported",
                        "evidence_refs": ["ev1", "ev2", "ev3"],
                    }
                ]
            }
        )
        self.assertEqual(1, len(parse_proposals(valid, {"r1"})))
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            parse_proposals(valid[:-3], {"r1"})

        judge = json.dumps(
            {
                "decisions": [
                    {
                        "review_id": "r1",
                        "decision": "accept",
                        "evidence_coverage": 4,
                        "semantic_composability": 3,
                        "relation_direction": 2,
                        "uncertainty_calibration": 1,
                        "total": 10,
                        "reason_codes": [],
                        "revised_relation": None,
                        "revised_relation_family": None,
                        "verified_evidence_refs": ["ev1", "ev2", "ev3"],
                    }
                ]
            }
        )
        self.assertEqual("accept", parse_judge_decision(judge, "r1").decision)
        mismatched_total = json.loads(judge)
        mismatched_total["decisions"][0]["total"] = 9
        with self.assertRaises(ValueError):
            parse_judge_decision(json.dumps(mismatched_total), "r1")


class AtomicWritebackTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_and_parallel_relations_do_not_overwrite_original(self):
        graph = FakeGraph(
            {"relation": "original_relation", "description": "source fact"}
        )
        vdb = FakeVDB()
        first = promotion(relation="indirectly_causes")
        second = promotion(relation="influences")
        await commit_promotion(first, graph, vdb)
        await commit_promotion(second, graph, vdb)
        self.assertEqual("original_relation", graph.edge["relation"])
        stored = json.loads(graph.edge["eclrr_relations"])
        self.assertEqual(2, len(stored))
        self.assertEqual({first.relation_id, second.relation_id}, set(vdb.data))

    async def test_refine_discards_fuzzy_edge_fields(self):
        graph = FakeGraph(
            {"relation": "关系不明", "description": "某种联系", "source_id": "old"}
        )
        result = promotion(action="refine")
        await commit_promotion(result, graph, FakeVDB())
        self.assertNotIn("关系不明", graph.edge.values())
        self.assertNotIn("某种联系", graph.edge.values())
        self.assertFalse(any(key.startswith("base_") for key in graph.edge))
        self.assertEqual("promoted", graph.edge["review_status"])

    async def test_vdb_failure_restores_graph_preimage(self):
        original = {"relation": "source_fact", "description": "unchanged"}
        graph = FakeGraph(original)
        with self.assertRaises(RuntimeError):
            await commit_promotion(promotion(), graph, FakeVDB(fail_upsert=True))
        self.assertEqual(original, graph.edge)

    async def test_vdb_commit_failure_removes_half_commit(self):
        original = {"relation": "source_fact", "description": "unchanged"}
        graph = FakeGraph(original)
        vdb = FakeVDB(fail_commit=True)
        with self.assertRaises(RuntimeError):
            await commit_promotion(promotion(), graph, vdb)
        self.assertEqual(original, graph.edge)
        self.assertEqual({}, vdb.data)

    async def test_graph_commit_failure_never_writes_vdb(self):
        graph = FakeGraph(fail_commit=True)
        vdb = FakeVDB()
        with self.assertRaises(RuntimeError):
            await commit_promotion(promotion(), graph, vdb)
        self.assertIsNone(graph.edge)
        self.assertEqual({}, vdb.data)


class ECLRRServiceIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_promoted_relation_is_retrieved_with_all_source_chunks(self):
        graph = FullGraph()
        chunks = FullChunks()
        vdb = FakeVDB()

        async def generator(prompt, **_kwargs):
            payload = json.loads(prompt.split("INPUT=", 1)[1])
            proposals = []
            for item in payload["items"]:
                proposals.append(
                    {
                        "review_id": item["review_id"],
                        "source": item["source"],
                        "target": item["target"],
                        "relation": "indirectly_causes",
                        "relation_family": "causation",
                        "direction": "source_to_target",
                        "description": "The complete chain supports indirect causation.",
                        "evidence_refs": [
                            evidence["evidence_id"]
                            for evidence in item["primary_evidence"]
                        ],
                    }
                )
            return json.dumps({"proposals": proposals})

        async def judge(prompt, **_kwargs):
            payload = json.loads(prompt.split("INPUT=", 1)[1])
            package = payload["evidence_package"]
            return json.dumps(
                {
                    "decisions": [
                        {
                            "review_id": package["review_id"],
                            "decision": "accept",
                            "evidence_coverage": 4,
                            "semantic_composability": 3,
                            "relation_direction": 2,
                            "uncertainty_calibration": 1,
                            "total": 10,
                            "reason_codes": [],
                            "revised_relation": None,
                            "revised_relation_family": None,
                            "verified_evidence_refs": [
                                evidence["evidence_id"]
                                for evidence in package["primary_evidence"]
                            ],
                        }
                    ]
                }
            )

        result = await run_eclrr_v4(
            graph=graph,
            text_chunks=chunks,
            generator_func=generator,
            judge_func=judge,
            relationships_vdb=vdb,
            config=ECLRRConfig(),
        )
        self.assertEqual(1, len(result.committed))
        stored = next(iter(vdb.data.values()))
        selected = select_relations_for_query(
            [stored],
            SimpleNamespace(
                include_discovered_edges=True,
                max_relations=8,
                max_discovered_relations=4,
            ),
        )
        self.assertEqual(1, len(selected))
        relation_chunks = await _find_related_text_unit_from_relations(
            selected,
            SimpleNamespace(),
            chunks,
        )
        self.assertEqual(
            {"chunk-AB", "chunk-BC", "chunk-CD"},
            {item["chunk_id"] for item in relation_chunks},
        )

    async def test_generator_json_failure_is_zero_write(self):
        graph = FullGraph()
        chunks = FullChunks()
        vdb = FakeVDB()

        async def invalid_generator(_prompt, **_kwargs):
            return '{"proposals":['

        async def unused_judge(_prompt, **_kwargs):
            raise AssertionError("Judge must not run without a valid proposal")

        result = await run_eclrr_v4(
            graph=graph,
            text_chunks=chunks,
            generator_func=invalid_generator,
            judge_func=unused_judge,
            relationships_vdb=vdb,
            config=ECLRRConfig(),
        )
        self.assertEqual([], result.committed)
        self.assertEqual({}, vdb.data)
        self.assertIsNone(await graph.get_edge("A", "D"))


if __name__ == "__main__":
    unittest.main()
