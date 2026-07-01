import json
import unittest
from types import SimpleNamespace

from docthinker.evaluation import compare_answers, score_answer
from docthinker.kg_expansion.edge_discovery import _parse_edges
from docthinker.retrieval_policy import select_relations_for_query


class EdgeDiscoveryEvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.nodes = {
            "GPU": {
                "description": "GPU 长时间满载会提高功耗并产生大量热量。",
                "source_id": "chunk-gpu",
            },
            "散热系统": {
                "description": "散热系统能力不足会触发 GPU 热降频。",
                "source_id": "chunk-cooling",
            },
        }

    def _raw(self, confidence=0.9, target_quote="散热系统能力不足会触发 GPU 热降频"):
        return json.dumps(
            [
                {
                    "source": "GPU",
                    "target": "散热系统",
                    "keywords": "热负载影响",
                    "description": "GPU 热负载需要匹配散热能力。",
                    "confidence": confidence,
                    "evidence": [
                        {"entity": "GPU", "quote": "GPU 长时间满载会提高功耗"},
                        {"entity": "散热系统", "quote": target_quote},
                    ],
                }
            ],
            ensure_ascii=False,
        )

    def test_accepts_only_high_confidence_grounded_edge(self):
        edges = _parse_edges(
            self._raw(),
            set(self.nodes),
            set(),
            nodes_by_name=self.nodes,
        )
        self.assertEqual(1, len(edges))
        self.assertEqual(["chunk-gpu", "chunk-cooling"], edges[0].evidence_chunk_ids)
        self.assertEqual(0.9, edges[0].confidence)

    def test_rejects_low_confidence_or_unquoted_evidence(self):
        low = _parse_edges(
            self._raw(confidence=0.6),
            set(self.nodes),
            set(),
            nodes_by_name=self.nodes,
        )
        invented = _parse_edges(
            self._raw(target_quote="不存在于原文的结论"),
            set(self.nodes),
            set(),
            nodes_by_name=self.nodes,
        )
        self.assertEqual([], low)
        self.assertEqual([], invented)


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
                "evidence": "[\"chunk-1\"]",
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
        reference = "1. 高功耗提高热负载。\n2. 散热不足会触发降频。"
        evidence = [
            "独立 GPU 的高功耗会提高整机热负载。",
            "散热能力不足时 GPU 会触发热降频。",
        ]
        concise = "高功耗提高热负载；散热能力不足会触发 GPU 降频。"
        noisy = "高功耗提高热负载。还可能导致电池鼓包、PCB 绝缘失效和供应链中断。"
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
            result["answer_b"]["grounded_claim_rate"],
            result["answer_a"]["grounded_claim_rate"],
        )
        self.assertEqual(-108.0, result["delta_b_minus_a"]["context_chunk_count"])

    def test_score_exposes_more_than_keyword_coverage(self):
        score = score_answer(
            answer="散热不足导致降频。",
            reference_answer="散热能力不足会触发 GPU 降频。",
            evidence_chunks=["散热能力不足时 GPU 会触发热降频。"],
        )
        self.assertIn("reference_point_coverage", score)
        self.assertIn("grounded_claim_rate", score)
        self.assertIn("unsupported_claim_rate", score)
        self.assertIn("focus_rate", score)


if __name__ == "__main__":
    unittest.main()
