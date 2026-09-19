import asyncio
import json
import unittest
from types import SimpleNamespace

from docthinker.evaluation import score_reasoning_paths
from docthinker.reasoning_policy import (
    classify_question,
    complete_path_from_evidence,
    evidence_constrained_paths,
    exploratory_pagerank,
    format_path_instruction,
)
from docthinker.retrieval_policy import select_relations_for_query

CHAIN = [
    "气候目标",
    "政策工具",
    "可再生能源投资",
    "国内供给增加",
    "进口依赖下降",
    "能源多样化",
    "供应风险下降",
    "能源安全提高",
]


def _nodes():
    return [
        {"id": name, "description": f"{name}是能源政策因果链中的概念。"}
        for name in CHAIN
    ] + [
        {"id": "绿色就业", "description": "绿色产业可能创造就业。"},
        {"id": "国际合作", "description": "气候政策可能促进国际合作。"},
    ]


def _edge(source, target, *, candidate=False, grounded=True, confidence=0.9):
    data = {
        "source": source,
        "target": target,
        "keywords": "导致",
        "description": f"{source}导致{target}",
        "source_id": f"chunk-{source}",
    }
    if candidate:
        data.update(
            {
                "is_discovered": "1",
                "query_eligible": "1",
                "confidence": confidence,
            }
        )
        if grounded:
            data.update(
                {
                    "provenance": "eclrr_v4",
                    "algorithm_version": "eclrr_v4",
                    "review_status": "promoted",
                    "evidence_chunk_ids": [f"chunk-{source}"],
                    "evidence_chain": [
                        {
                            "chunk_id": f"chunk-{source}",
                            "quote": data["description"],
                            "start": 0,
                            "end": len(data["description"]),
                            "edge_id": f"{source}-{target}",
                        }
                    ],
                    "judge_scores": {
                        "total": 9,
                        "evidence_coverage": 4,
                        "semantic_composability": 3,
                        "relation_direction": 1,
                    },
                }
            )
    return data


class QuestionPolicyTest(unittest.TestCase):
    def test_routes_three_question_types(self):
        faithful = classify_question("文档中供应商具体叫什么？")
        path = classify_question("气候目标为什么会提高能源安全？")
        explore = classify_question("气候政策还可能有哪些潜在影响？")

        self.assertEqual("faithful", faithful.mode)
        self.assertEqual("path", path.mode)
        self.assertEqual("explore", explore.mode)

    def test_disabled_or_uncertain_defaults_to_faithful(self):
        disabled = classify_question("为什么？", self_evolution_enabled=False)
        uncertain = classify_question("请回答这个问题")
        self.assertEqual("faithful", disabled.mode)
        self.assertEqual("faithful", uncertain.mode)


class EvidenceConstrainedPathTest(unittest.TestCase):
    def test_does_not_substitute_a_reverse_path(self):
        edges = [
            _edge(CHAIN[index + 1], CHAIN[index]) for index in range(len(CHAIN) - 1)
        ]
        paths, _ = evidence_constrained_paths(
            "气候目标如何导致能源安全提高？", _nodes(), edges
        )
        self.assertEqual([], paths)

    def test_honours_explicit_reverse_direction(self):
        edges = [
            {**_edge(CHAIN[index + 1], CHAIN[index]), "direction": "target_to_source"}
            for index in range(len(CHAIN) - 1)
        ]
        paths, _ = evidence_constrained_paths(
            "气候目标如何导致能源安全提高？", _nodes(), edges
        )
        self.assertEqual(CHAIN, paths[0].nodes)
        self.assertEqual(CHAIN[0], paths[0].to_schema()["hops"][0]["source"])

    def test_undirected_edge_does_not_establish_a_directed_path(self):
        edge = {**_edge("A", "B"), "direction": "undirected"}
        paths, _ = evidence_constrained_paths(
            "A如何导致B？", [{"id": "A"}, {"id": "B"}], [edge]
        )
        self.assertEqual([], paths)

    def test_finds_complete_chain_instead_of_distractor_edges(self):
        edges = [
            _edge(CHAIN[index], CHAIN[index + 1]) for index in range(len(CHAIN) - 1)
        ]
        edges.extend(
            [
                _edge("气候目标", "绿色就业", candidate=True),
                _edge("政策工具", "国际合作", candidate=True),
            ]
        )

        paths, diagnostic = evidence_constrained_paths(
            "气候目标如何导致能源安全提高？",
            _nodes(),
            edges,
        )

        self.assertEqual("ok", diagnostic["reason"])
        self.assertGreaterEqual(len(paths), 1)
        self.assertEqual(CHAIN, paths[0].nodes)
        self.assertEqual(7, len(paths[0].edges))
        instruction = format_path_instruction(paths, diagnostic)
        self.assertIn("来源：chunk-气候目标", instruction)
        self.assertIn("排序分数不是正确率", instruction)
        self.assertNotIn("经过证据约束和连续性检查", instruction)

    def test_rejects_unsupported_candidate_bridge(self):
        edges = [
            _edge("气候目标", "政策工具"),
            _edge("政策工具", "可再生能源投资"),
            _edge(
                "可再生能源投资",
                "能源多样化",
                candidate=True,
                grounded=False,
                confidence=0.99,
            ),
            _edge("能源多样化", "供应风险下降"),
            _edge("供应风险下降", "能源安全提高"),
        ]

        paths, diagnostic = evidence_constrained_paths(
            "气候目标如何导致能源安全提高？",
            _nodes(),
            edges,
        )

        self.assertEqual([], paths)
        self.assertEqual("no_continuous_evidence_path", diagnostic["reason"])


class ExploratoryPageRankTest(unittest.TestCase):
    def test_dangling_rank_returns_to_personalization_seeds(self):
        result = exploratory_pagerank(
            "A和Z的关系",
            [{"id": name} for name in ("A", "B", "Z")],
            [_edge("A", "B")],
            iterations=1,
        )
        self.assertAlmostEqual(1.0, result["rank_mass"])
        self.assertEqual("B", result["items"][0]["entity"])
        self.assertAlmostEqual(0.425, result["items"][0]["score"])

    def test_duplicate_nodes_and_absent_edge_endpoints_do_not_change_rank_mass(self):
        result = exploratory_pagerank(
            "A和Z的关系",
            [{"id": name} for name in ("A", "B", "Z", "A")],
            [_edge("A", "B"), _edge("A", "missing")],
        )
        self.assertAlmostEqual(1.0, result["rank_mass"])
        self.assertEqual(["A", "Z"], result["anchors"])
        self.assertEqual(["B"], [item["entity"] for item in result["items"]])

    def test_legacy_inference_markers_cannot_bypass_either_query_gate(self):
        markers = [
            {"is_discovered": "1"},
            {"is_expanded": True},
            {"review_status": "candidate"},
            {"provenance": "llm_expansion"},
            {"source_id": "llm_expansion"},
            {"provenance": "edge_discovery"},
        ]
        for marker in markers:
            with self.subTest(marker=marker):
                edge = {**_edge("A", "B"), **marker}
                result = exploratory_pagerank("A", [{"id": "A"}, {"id": "B"}], [edge])
                self.assertEqual([], result["items"])
                self.assertEqual(
                    [],
                    select_relations_for_query(
                        [edge], SimpleNamespace(include_discovered_edges=True)
                    ),
                )

    def test_nonfinite_confidence_cannot_pass_promoted_gate(self):
        for confidence in (float("nan"), float("inf"), "NaN"):
            with self.subTest(confidence=confidence):
                edge = _edge("A", "B", candidate=True, confidence=confidence)
                result = exploratory_pagerank("A", [{"id": "A"}, {"id": "B"}], [edge])
                self.assertEqual([], result["items"])
                self.assertEqual(
                    [],
                    select_relations_for_query(
                        [edge], SimpleNamespace(include_discovered_edges=True)
                    ),
                )

    def test_promoted_record_without_legacy_confidence_is_accepted(self):
        promoted = _edge("A", "B", candidate=True)
        del promoted["confidence"]
        physical = {
            **_edge("A", "B"),
            "is_discovered": "1",
            "eclrr_relations": [promoted],
        }
        result = exploratory_pagerank("A", [{"id": "A"}, {"id": "B"}], [physical])
        self.assertEqual("B", result["items"][0]["entity"])
        self.assertTrue(result["items"][0]["candidate"])

    def test_exploration_returns_diverse_neighbours(self):
        edges = [
            _edge("气候目标", "政策工具"),
            _edge("气候目标", "绿色就业", candidate=True),
            _edge("气候目标", "国际合作", candidate=True),
            _edge("政策工具", "可再生能源投资"),
        ]

        result = exploratory_pagerank(
            "气候目标还可能有哪些潜在影响？",
            _nodes(),
            edges,
            top_k=4,
        )

        entities = {item["entity"] for item in result["items"]}
        self.assertTrue({"绿色就业", "国际合作"} & entities)
        self.assertEqual(len(entities), len(result["items"]))


class QueryLocalPathCompletionTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def proposal(quote="A导致B", confidence=0.9):
        return json.dumps(
            {
                "hops": [
                    {
                        "source": "A",
                        "target": "B",
                        "relation": "导致",
                        "confidence": confidence,
                        "chunk_id": "chunk-1",
                        "evidence_quote": quote,
                    }
                ]
            },
            ensure_ascii=False,
        )

    async def test_total_input_and_provider_output_budget(self):
        calls = []

        async def llm(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return self.proposal()

        path, diagnostic = await complete_path_from_evidence(
            question="A为什么导致B？",
            start="A",
            goal="B",
            chunks=[
                {"id": f"chunk-{index}", "content": "A导致B。" * 1000}
                for index in range(1, 13)
            ],
            llm_func=llm,
            max_input_chars=1000,
            max_output_tokens=200,
        )
        self.assertIsNotNone(path)
        self.assertLessEqual(len(calls[0][0]), 1000)
        self.assertEqual({"max_tokens": 200}, calls[0][1])
        self.assertTrue(diagnostic["provider_output_limit"])
        self.assertEqual("0", path.edges[0]["query_eligible"])

    async def test_no_call_when_endpoints_are_outside_visible_budget(self):
        calls = []

        async def llm(prompt):
            calls.append(prompt)
            return self.proposal()

        path, diagnostic = await complete_path_from_evidence(
            question="A为什么导致B？",
            start="A",
            goal="B",
            chunks=[{"id": "chunk-1", "content": "x" * 1500 + "A导致B"}],
            llm_func=llm,
        )
        self.assertIsNone(path)
        self.assertEqual("endpoints_not_in_visible_evidence", diagnostic["reason"])
        self.assertEqual([], calls)

    async def test_quote_in_hidden_chunk_tail_is_not_accepted(self):
        async def llm(_prompt):
            return self.proposal("A明显导致B")

        path, diagnostic = await complete_path_from_evidence(
            question="A为什么导致B？",
            start="A",
            goal="B",
            chunks=[
                {"id": "chunk-1", "content": "A和B被提及。" + "x" * 1500 + "A明显导致B"}
            ],
            llm_func=llm,
        )
        self.assertIsNone(path)
        self.assertEqual("unverified_quote", diagnostic["reason"])

    async def test_nonfinite_generated_confidence_is_rejected(self):
        async def llm(_prompt):
            return self.proposal(confidence=float("nan"))

        path, diagnostic = await complete_path_from_evidence(
            question="A为什么导致B？",
            start="A",
            goal="B",
            chunks=[{"id": "chunk-1", "content": "A导致B"}],
            llm_func=llm,
        )
        self.assertIsNone(path)
        self.assertEqual("low_confidence_hop", diagnostic["reason"])

    async def test_slow_call_is_cancelled_and_oversized_output_rejected(self):
        cancelled = asyncio.Event()

        async def slow(_prompt):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.set()

        arguments = dict(
            question="A导致B？",
            start="A",
            goal="B",
            chunks=[{"id": "chunk-1", "content": "A导致B"}],
        )
        path, diagnostic = await complete_path_from_evidence(
            **arguments, llm_func=slow, timeout_seconds=0.01
        )
        self.assertIsNone(path)
        self.assertEqual("llm_bridge_timeout", diagnostic["reason"])
        self.assertTrue(cancelled.is_set())

        async def oversized(_prompt):
            return "x" * 1000

        path, diagnostic = await complete_path_from_evidence(
            **arguments, llm_func=oversized, max_output_tokens=100
        )
        self.assertIsNone(path)
        self.assertEqual("invalid_or_oversized_completion_output", diagnostic["reason"])

    async def test_accepts_only_continuous_exactly_quoted_hops(self):
        chunks = [
            {
                "id": "chunk-1",
                "content": (
                    "可再生能源投资推动国内供给增加。"
                    "国内供给增加导致进口依赖下降。"
                    "进口依赖下降促进能源多样化。"
                ),
            }
        ]

        async def llm(_prompt):
            return """{
              "hops": [
                {"source":"可再生能源投资","target":"国内供给增加","relation":"推动","confidence":0.92,"chunk_id":"chunk-1","evidence_quote":"可再生能源投资推动国内供给增加"},
                {"source":"国内供给增加","target":"进口依赖下降","relation":"导致","confidence":0.91,"chunk_id":"chunk-1","evidence_quote":"国内供给增加导致进口依赖下降"},
                {"source":"进口依赖下降","target":"能源多样化","relation":"促进","confidence":0.90,"chunk_id":"chunk-1","evidence_quote":"进口依赖下降促进能源多样化"}
              ]
            }"""

        path, diagnostic = await complete_path_from_evidence(
            question="可再生能源投资如何促进能源多样化？",
            start="可再生能源投资",
            goal="能源多样化",
            chunks=chunks,
            llm_func=llm,
        )

        self.assertIsNotNone(path)
        self.assertEqual("query_local_bridge", diagnostic["reason"])
        self.assertEqual(3, len(path.edges))

    async def test_rejects_invented_quote(self):
        async def llm(_prompt):
            return """{"hops":[{"source":"A","target":"B","relation":"导致","confidence":0.99,"chunk_id":"chunk-1","evidence_quote":"原文不存在的结论"}]}"""

        path, diagnostic = await complete_path_from_evidence(
            question="A为什么导致B？",
            start="A",
            goal="B",
            chunks=[{"id": "chunk-1", "content": "A和B被分别提及。"}],
            llm_func=llm,
        )

        self.assertIsNone(path)
        self.assertEqual("unverified_quote", diagnostic["reason"])


class PathEvaluationTest(unittest.TestCase):
    def test_quote_coverage_checks_supplied_source_text(self):
        path = {
            "nodes": ["A", "B"],
            "score": 0.8,
            "hops": [
                {
                    "source": "A",
                    "target": "B",
                    "source_id": "chunk-1",
                    "evidence": [{"chunk_id": "chunk-1", "quote": "A导致B"}],
                }
            ],
        }
        verified = score_reasoning_paths(
            expected_chain=["A", "B"],
            paths=[path],
            source_chunks={"chunk-1": "A导致B。"},
        )
        unsupported = score_reasoning_paths(
            expected_chain=["A", "B"],
            paths=[path],
            source_chunks={"chunk-1": "A和B被分别提及。"},
        )
        self.assertEqual(1.0, verified["verified_quote_hop_rate"])
        self.assertEqual(0.0, unsupported["verified_quote_hop_rate"])

    def test_citations_are_not_scored_as_verified_evidence(self):
        path = {
            "nodes": ["A", "B"],
            "score": float("nan"),
            "hops": [
                {
                    "source": "A",
                    "target": "B",
                    "source_id": "invented-id",
                    "quote_verified": True,
                }
            ],
        }
        score = score_reasoning_paths(expected_chain=["A", "B"], paths=[path])
        self.assertEqual(1.0, score["citation_hop_rate"])
        self.assertEqual(0.0, score["verified_quote_hop_rate"])
        self.assertNotIn("grounded_hop_rate", score)
        self.assertEqual(0.0, score["path_score"])

    def test_shuffled_hops_do_not_receive_complete_chain_credit(self):
        score = score_reasoning_paths(
            expected_chain=["A", "B", "C"],
            paths=[
                {
                    "nodes": ["A", "B", "C"],
                    "score": 0.9,
                    "hops": [
                        {"source": "B", "target": "C"},
                        {"source": "A", "target": "B"},
                    ],
                }
            ],
        )
        self.assertEqual(0.0, score["expected_hop_coverage"])
        self.assertEqual(0.0, score["continuous_hop_rate"])

    def test_scores_complete_ordered_chain(self):
        edges = [
            _edge(CHAIN[index], CHAIN[index + 1]) for index in range(len(CHAIN) - 1)
        ]
        paths, _ = evidence_constrained_paths(
            "气候目标如何导致能源安全提高？",
            _nodes(),
            edges,
        )

        score = score_reasoning_paths(
            expected_chain=CHAIN,
            paths=[path.to_schema() for path in paths],
        )

        self.assertEqual(1.0, score["ordered_node_coverage"])
        self.assertEqual(1.0, score["expected_hop_coverage"])
        self.assertEqual(1.0, score["endpoint_coverage"])


if __name__ == "__main__":
    unittest.main()
