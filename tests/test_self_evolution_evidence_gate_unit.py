import json
import asyncio
import unittest
from types import SimpleNamespace

from docthinker.evaluation import compare_answers, score_answer
from docthinker.kg_expansion.path_edge_discovery import (
    load_chunk_evidence,
    parse_path_discovered_edges,
)
from docthinker.retrieval_policy import select_relations_for_query


class PathEdgeDiscoveryEvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.valid_names = {"GPU", "Thermal Load", "Cooling System"}
        self.edges_by_pair = {
            tuple(sorted(("GPU", "Thermal Load"))): {
                "source": "GPU",
                "target": "Thermal Load",
            },
            tuple(sorted(("Thermal Load", "Cooling System"))): {
                "source": "Thermal Load",
                "target": "Cooling System",
            },
        }
        self.chunk_text_by_id = {
            "chunk-gpu": "GPU sustained full load increases power draw and creates high thermal load.",
            "chunk-cooling": "Cooling System capacity limits can trigger GPU thermal throttling.",
        }

    def _raw(
        self,
        confidence=0.9,
        target_quote="Cooling System capacity limits can trigger GPU thermal throttling",
    ):
        return json.dumps(
            [
                {
                    "source": "GPU",
                    "target": "Cooling System",
                    "relation": "requires_thermal_control",
                    "keywords": "thermal path inference",
                    "description": (
                        "GPU creates thermal load, and the cooling system constrains "
                        "thermal load, so GPU performance depends on cooling capacity."
                    ),
                    "inference_type": "path_composition",
                    "path_used": ["GPU", "Thermal Load", "Cooling System"],
                    "evidence_chain": [
                        {
                            "edge": ["GPU", "Thermal Load"],
                            "chunk_id": "chunk-gpu",
                            "quote": "GPU sustained full load increases power draw",
                        },
                        {
                            "edge": ["Thermal Load", "Cooling System"],
                            "chunk_id": "chunk-cooling",
                            "quote": target_quote,
                        },
                    ],
                    "evidence_chunk_ids": ["chunk-gpu", "chunk-cooling"],
                    "confidence": confidence,
                }
            ]
        )

    def _parse(self, raw):
        return parse_path_discovered_edges(
            raw,
            valid_names=self.valid_names,
            existing_pairs=set(self.edges_by_pair),
            edges_by_pair=self.edges_by_pair,
            chunk_text_by_id=self.chunk_text_by_id,
        )

    def test_accepts_only_high_confidence_grounded_path_edge(self):
        edges, rejected = self._parse(self._raw())
        self.assertEqual({}, rejected)
        self.assertEqual(1, len(edges))
        self.assertEqual(["chunk-gpu", "chunk-cooling"], edges[0].evidence_chunk_ids)
        self.assertEqual(0.9, edges[0].confidence)
        self.assertEqual(["GPU", "Thermal Load", "Cooling System"], edges[0].path_used)

    def test_rejects_low_confidence_or_unquoted_evidence(self):
        low, low_rejected = self._parse(self._raw(confidence=0.6))
        invented, invented_rejected = self._parse(
            self._raw(target_quote="an invented quote that is not in the source chunk")
        )
        self.assertEqual([], low)
        self.assertEqual({"low_confidence": 1}, low_rejected)
        self.assertEqual([], invented)
        self.assertEqual({"quote_not_grounded": 1}, invented_rejected)

    def test_recovers_complete_objects_from_truncated_array(self):
        first = json.loads(self._raw())[0]
        raw = "[\n" + json.dumps(first) + ',\n{"source": "GPU", "target":'
        edges, rejected = self._parse(raw)
        self.assertEqual({}, rejected)
        self.assertEqual(1, len(edges))
        self.assertEqual(("GPU", "Cooling System"), (edges[0].source, edges[0].target))

    def test_rejects_when_path_segment_lacks_evidence(self):
        item = json.loads(self._raw())[0]
        item["evidence_chain"] = item["evidence_chain"][:1]
        edges, rejected = self._parse(json.dumps([item]))
        self.assertEqual([], edges)
        self.assertEqual({"insufficient_evidence_chain": 1}, rejected)

    def test_rejects_grounded_quote_that_does_not_describe_evidence_edge(self):
        item = json.loads(self._raw())[0]
        item["evidence_chain"][0]["quote"] = "and creates high thermal load."
        edges, rejected = self._parse(json.dumps([item]))
        self.assertEqual([], edges)
        self.assertEqual({"quote_not_about_evidence_edge": 1}, rejected)

    def test_rejects_when_evidence_chain_does_not_cover_full_path(self):
        self.valid_names.add("Fan Controller")
        self.edges_by_pair[tuple(sorted(("Cooling System", "Fan Controller")))] = {
            "source": "Cooling System",
            "target": "Fan Controller",
        }
        item = json.loads(self._raw())[0]
        item["target"] = "Fan Controller"
        item["path_used"] = ["GPU", "Thermal Load", "Cooling System", "Fan Controller"]
        edges, rejected = self._parse(json.dumps([item]))
        self.assertEqual([], edges)
        self.assertEqual({"missing_path_evidence": 1}, rejected)

    def test_chunk_evidence_uses_focus_windows_not_prefix_only(self):
        class FakeChunks:
            async def get_by_ids(self, ids):
                return [
                    {
                        "file_path": "doc.txt",
                        "content": (
                            "intro " * 300
                            + "GPU raises thermal load. Cooling System responds later."
                        ),
                    }
                ]

        chunks, full = asyncio.run(
            load_chunk_evidence(
                FakeChunks(),
                ["chunk-late"],
                max_chunk_chars=220,
                focus_terms=["GPU", "Cooling System"],
                evidence_window_chars=120,
            )
        )
        self.assertIn("GPU raises thermal load", chunks[0]["content"])
        self.assertIn("Cooling System responds later", chunks[0]["content"])
        self.assertTrue(chunks[0]["quote_candidates"])
        self.assertIn("GPU", " ".join(chunks[0]["quote_candidates"]))
        self.assertGreater(len(full["chunk-late"]), len(chunks[0]["content"]))


class RelationBudgetTest(unittest.TestCase):
    def _params(self, include=False):
        return SimpleNamespace(
            include_discovered_edges=include,
            max_relations=3,
            max_discovered_relations=1,
            min_discovered_edge_confidence=0.8,
            require_discovered_evidence=True,
        )

    def test_discovered_edges_are_opt_in(self):
        relations = [
            {"src_id": "A", "tgt_id": "B", "weight": 1.0},
            {
                "src_id": "B",
                "tgt_id": "C",
                "is_discovered": "1",
                "query_eligible": "1",
                "confidence": "0.95",
                "evidence_chain": "[\"e1\", \"e2\"]",
            },
        ]
        selected = select_relations_for_query(relations, self._params(False))
        self.assertEqual(1, len(selected))
        self.assertEqual("A", selected[0]["src_id"])

    def test_inferred_relations_require_evidence_and_have_a_separate_cap(self):
        relations = [
            {"src_id": "A", "tgt_id": "B"},
            {"src_id": "C", "tgt_id": "D"},
            {"src_id": "E", "tgt_id": "F"},
            {
                "src_id": "B",
                "tgt_id": "C",
                "is_discovered": "1",
                "query_eligible": "1",
                "confidence": 0.95,
                "evidence_chain": "[\"e1\", \"e2\"]",
            },
            {
                "src_id": "D",
                "tgt_id": "E",
                "is_discovered": "1",
                "query_eligible": "1",
                "confidence": 0.99,
            },
        ]
        selected = select_relations_for_query(relations, self._params(True))
        self.assertEqual(3, len(selected))
        self.assertEqual(1, sum(str(item.get("is_discovered")) == "1" for item in selected))
        self.assertEqual(("B", "C"), (selected[-1]["src_id"], selected[-1]["tgt_id"]))


class MultiMetricEvaluationTest(unittest.TestCase):
    def test_grounded_concise_answer_beats_noisy_answer(self):
        reference = "1. High power increases thermal load.\n2. Insufficient cooling triggers throttling."
        evidence = [
            "A discrete GPU with high power draw increases system thermal load.",
            "When cooling capacity is insufficient, the GPU can trigger thermal throttling.",
        ]
        concise = "High power raises thermal load; insufficient cooling can trigger GPU throttling."
        noisy = (
            "High power raises thermal load. It may also cause battery swelling, "
            "PCB insulation failure, and supply-chain interruption."
        )
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
        self.assertGreater(
            result["answer_b"]["balanced_score"],
            result["answer_a"]["balanced_score"],
        )
        self.assertGreater(
            result["answer_b"]["reference_point_coverage"],
            result["answer_a"]["reference_point_coverage"],
        )
        self.assertEqual(-108.0, result["delta_b_minus_a"]["context_chunk_count"])

    def test_score_exposes_more_than_keyword_coverage(self):
        score = score_answer(
            answer="Insufficient cooling causes throttling.",
            reference_answer="Insufficient cooling capacity can trigger GPU throttling.",
            evidence_chunks=["When cooling capacity is insufficient, the GPU can trigger thermal throttling."],
        )
        self.assertIn("reference_point_coverage", score)
        self.assertIn("grounded_claim_rate", score)
        self.assertIn("unsupported_claim_rate", score)
        self.assertIn("focus_rate", score)


if __name__ == "__main__":
    unittest.main()
