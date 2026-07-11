import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from docthinker.evaluation import compare_answers, score_answer
from docthinker.kg_expansion.eclrr_v4 import (
    ECLRRConfig,
    FactGraphView,
    JudgeDecision,
    Proposal,
    build_evidence_package,
    deterministic_gate,
    discover_review_items,
)
from docthinker.retrieval_policy import select_relations_for_query


class FakeChunks:
    def __init__(self, values):
        self.values = dict(values)

    async def get_by_ids(self, ids):
        return [
            {"chunk_id": chunk_id, "content": self.values[chunk_id]}
            for chunk_id in ids
            if chunk_id in self.values
        ]


def chain_graph(names="ABCD", *, fuzzy=False):
    nodes = [
        {
            "id": name,
            "entity_type": "person" if name in {names[0], names[-1]} else "event",
        }
        for name in names
    ]
    edges = []
    chunks = {}
    for left, right in zip(names, names[1:]):
        chunk_id = f"chunk-{left}{right}"
        chunks[chunk_id] = (
            f"{left} directly causes {right} in the archived event record."
        )
        edges.append(
            {
                "source": left,
                "target": right,
                "relation": "causes",
                "relation_family": "causation",
                "direction": "source_to_target",
                "description": f"{left} directly causes {right}",
                "source_id": chunk_id,
            }
        )
    if fuzzy:
        edges.append(
            {
                "source": names[0],
                "target": names[-1],
                "relation": "关系不明",
                "description": "二者存在未明说的某种联系",
                "source_id": "chunk-fuzzy",
            }
        )
    return FactGraphView.build(nodes, edges), FakeChunks(chunks)


class ECLRRSearchTest(unittest.TestCase):
    def test_long_chain_discovers_minimal_endpoint_slices(self):
        view, _ = chain_graph("ABCDEFGHI")
        items = discover_review_items(view, ECLRRConfig(max_review_items=200))
        pairs = {(item.source, item.target) for item in items}
        self.assertIn(("A", "I"), pairs)
        self.assertIn(("B", "H"), pairs)
        self.assertIn(("C", "H"), pairs)
        self.assertTrue(all(3 <= item.primary_path.hops <= 8 for item in items))

    def test_non_person_nodes_are_valid_bridges(self):
        view, _ = chain_graph("ABCD")
        self.assertEqual("event", view.node_type("B"))
        pairs = {
            (item.source, item.target)
            for item in discover_review_items(view, ECLRRConfig())
        }
        self.assertIn(("A", "D"), pairs)

    def test_search_is_stable_and_marks_inverse_traversal(self):
        nodes = [{"id": name} for name in "ABCD"]
        edges = [
            {
                "source": "B",
                "target": "A",
                "relation": "causes",
                "direction": "source_to_target",
                "source_id": "c1",
            },
            {"source": "B", "target": "C", "relation": "causes", "source_id": "c2"},
            {"source": "C", "target": "D", "relation": "causes", "source_id": "c3"},
        ]
        view = FactGraphView.build(nodes, edges)
        first = discover_review_items(view, ECLRRConfig())
        second = discover_review_items(view, ECLRRConfig())
        self.assertEqual(
            [item.primary_path.signature for item in first],
            [item.primary_path.signature for item in second],
        )
        path = next(
            item.primary_path
            for item in first
            if (item.source, item.target) == ("A", "D")
        )
        self.assertEqual("inverse", path.steps[0].traversal_direction)

    def test_fuzzy_promoted_and_legacy_edges_never_enter_fact_paths(self):
        nodes = [{"id": name} for name in "ABCDE"]
        edges = [
            {"source": "A", "target": "B", "relation": "causes", "source_id": "c1"},
            {"source": "B", "target": "C", "relation": "关系不明", "source_id": "c2"},
            {
                "source": "C",
                "target": "D",
                "relation": "causes",
                "source_id": "c3",
                "review_status": "promoted",
                "provenance": "eclrr_v4",
            },
            {
                "source": "D",
                "target": "E",
                "relation": "causes",
                "source_id": "c4",
                "review_status": "candidate",
            },
        ]
        view = FactGraphView.build(nodes, edges)
        self.assertEqual(1, len(view.fact_edges))
        self.assertEqual([], discover_review_items(view, ECLRRConfig()))


class ECLRRDeterministicGateTest(unittest.IsolatedAsyncioTestCase):
    async def _package(self, *, fuzzy=False):
        view, chunks = chain_graph("ABCD", fuzzy=fuzzy)
        item = next(
            item
            for item in discover_review_items(view, ECLRRConfig())
            if (item.source, item.target) == ("A", "D")
        )
        package, reason = await build_evidence_package(
            item, view, chunks, ECLRRConfig()
        )
        self.assertEqual("ok", reason)
        self.assertIsNotNone(package)
        return view, chunks, package

    @staticmethod
    def _proposal(package, relation="indirectly_causes"):
        return Proposal(
            review_id=package.review_item.review_id,
            source="A",
            target="D",
            relation=relation,
            relation_family="causation",
            direction="source_to_target",
            description="A indirectly causes D through B and C.",
            evidence_refs=tuple(item.evidence_id for item in package.primary_evidence),
        )

    @staticmethod
    def _decision(package, decision="accept"):
        return JudgeDecision(
            review_id=package.review_item.review_id,
            decision=decision,
            evidence_coverage=4 if decision != "reject" else 0,
            semantic_composability=3 if decision != "reject" else 0,
            relation_direction=2 if decision != "reject" else 0,
            uncertainty_calibration=1,
            total=10 if decision != "reject" else 1,
            reason_codes=(),
            revised_relation=None,
            revised_relation_family=None,
            verified_evidence_refs=tuple(
                item.evidence_id for item in package.primary_evidence
            ),
        )

    async def test_cross_chunk_chain_passes_as_create(self):
        view, chunks, package = await self._package()
        result = await deterministic_gate(
            package,
            self._proposal(package),
            self._decision(package),
            view,
            chunks,
            ECLRRConfig(),
        )
        self.assertEqual("create", result.action)
        self.assertEqual(3, len(result.evidence_chain))
        for evidence in package.primary_evidence:
            text = chunks.values[evidence.chunk_id]
            self.assertEqual(evidence.quote, text[evidence.start : evidence.end])

    async def test_fuzzy_edge_is_refined_instead_of_edge_exists(self):
        view, chunks, package = await self._package(fuzzy=True)
        result = await deterministic_gate(
            package,
            self._proposal(package),
            self._decision(package),
            view,
            chunks,
            ECLRRConfig(),
        )
        self.assertEqual("refine", result.action)

    async def test_judge_revision_is_applied_before_canonicalization(self):
        view, chunks, package = await self._package()
        decision = replace(
            self._decision(package),
            decision="revise",
            revised_relation="indirectly_influences",
            revised_relation_family="influence",
        )
        result = await deterministic_gate(
            package,
            self._proposal(package),
            decision,
            view,
            chunks,
            ECLRRConfig(),
        )
        self.assertEqual("create", result.action)
        self.assertEqual("indirectly_influences", result.relation)
        self.assertEqual("influence", result.relation_family)

    async def test_forged_quote_and_judge_reject_are_no_op(self):
        view, chunks, package = await self._package()
        package.primary_evidence[1] = replace(
            package.primary_evidence[1], quote="invented quote"
        )
        forged = await deterministic_gate(
            package,
            self._proposal(package),
            self._decision(package),
            view,
            chunks,
            ECLRRConfig(),
        )
        rejected = await deterministic_gate(
            package,
            self._proposal(package),
            self._decision(package, "reject"),
            view,
            chunks,
            ECLRRConfig(),
        )
        self.assertEqual("quote_not_exact_substring", forged.reason)
        self.assertEqual("judge_reject", rejected.reason)

    async def test_missing_middle_chunk_stops_before_llm(self):
        view, chunks = chain_graph("ABCD")
        del chunks.values["chunk-BC"]
        item = next(
            item
            for item in discover_review_items(view, ECLRRConfig())
            if (item.source, item.target) == ("A", "D")
        )
        package, reason = await build_evidence_package(
            item, view, chunks, ECLRRConfig()
        )
        self.assertIsNone(package)
        self.assertIn("missing_primary_evidence", reason)


class RelationBudgetTest(unittest.TestCase):
    def _params(self, include=False):
        return SimpleNamespace(
            include_discovered_edges=include,
            max_relations=3,
            max_discovered_relations=1,
            min_discovered_edge_confidence=0.8,
            require_discovered_evidence=True,
        )

    @staticmethod
    def _promoted():
        chain = [
            {
                "edge_id": "e1",
                "chunk_id": "c1",
                "quote": "A causes B",
                "start": 0,
                "end": 10,
            },
            {
                "edge_id": "e2",
                "chunk_id": "c2",
                "quote": "B causes C",
                "start": 0,
                "end": 10,
            },
            {
                "edge_id": "e3",
                "chunk_id": "c3",
                "quote": "C causes D",
                "start": 0,
                "end": 10,
            },
        ]
        return {
            "src_id": "A",
            "tgt_id": "D",
            "is_discovered": "1",
            "query_eligible": "1",
            "review_status": "promoted",
            "provenance": "eclrr_v4",
            "algorithm_version": "eclrr_v4",
            "relation_id": "rel-eclrr-1",
            "source_id": "c1<SEP>c2<SEP>c3",
            "evidence_chain": json.dumps(chain),
            "evidence_chunk_ids": json.dumps(["c1", "c2", "c3"]),
            "judge_scores": json.dumps(
                {
                    "evidence_coverage": 4,
                    "semantic_composability": 3,
                    "relation_direction": 2,
                    "uncertainty_calibration": 1,
                    "total": 10,
                }
            ),
        }

    def test_only_promoted_v4_edges_are_opted_in(self):
        original = {"src_id": "X", "tgt_id": "Y", "provenance": "source_document"}
        legacy = {
            **self._promoted(),
            "review_status": "candidate",
            "provenance": "self_study",
        }
        self.assertEqual(
            [original],
            select_relations_for_query(
                [original, self._promoted()], self._params(False)
            ),
        )
        selected = select_relations_for_query(
            [original, legacy, self._promoted()], self._params(True)
        )
        self.assertEqual(2, len(selected))
        self.assertEqual("rel-eclrr-1", selected[-1]["relation_id"])

    def test_promoted_relation_is_deduplicated_across_graph_and_vdb(self):
        promoted = self._promoted()
        physical = {
            "src_id": "X",
            "tgt_id": "Y",
            "relation": "source_fact",
            "eclrr_relations": json.dumps([promoted]),
        }
        selected = select_relations_for_query([physical, promoted], self._params(True))
        self.assertEqual(
            1,
            sum(item.get("relation_id") == "rel-eclrr-1" for item in selected),
        )


class MultiMetricEvaluationTest(unittest.TestCase):
    def test_grounded_concise_answer_beats_noisy_answer(self):
        reference = "1. High power increases thermal load.\n2. Insufficient cooling triggers throttling."
        evidence = [
            "A discrete GPU with high power draw increases system thermal load.",
            "When cooling capacity is insufficient, the GPU can trigger thermal throttling.",
        ]
        concise = "High power raises thermal load; insufficient cooling can trigger GPU throttling."
        noisy = "High power raises thermal load. It may also cause battery swelling and supply-chain interruption."
        result = compare_answers(
            answer_a=noisy,
            answer_b=concise,
            reference_answer=reference,
            evidence_a=evidence,
            evidence_b=evidence,
            context_chunk_count_a=120,
            context_chunk_count_b=12,
        )
        self.assertEqual("B", result["preferred"])

    def test_score_exposes_grounding_metrics(self):
        score = score_answer(
            answer="Insufficient cooling causes throttling.",
            reference_answer="Insufficient cooling capacity can trigger GPU throttling.",
            evidence_chunks=[
                "When cooling is insufficient, the GPU can trigger throttling."
            ],
        )
        self.assertIn("grounded_claim_rate", score)
        self.assertIn("unsupported_claim_rate", score)


if __name__ == "__main__":
    unittest.main()
