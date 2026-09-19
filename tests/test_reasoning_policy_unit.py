import unittest

from docthinker.evaluation import score_reasoning_paths
from docthinker.reasoning_policy import (
    classify_question,
    complete_path_from_evidence,
    evidence_constrained_paths,
    exploratory_pagerank,
)


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
        data.update({
            "is_discovered": "1",
            "query_eligible": "1",
            "confidence": confidence,
        })
        if grounded:
            data["evidence_chunk_ids"] = f'["chunk-{source}"]'
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
    def test_finds_complete_chain_instead_of_distractor_edges(self):
        edges = [
            _edge(CHAIN[index], CHAIN[index + 1])
            for index in range(len(CHAIN) - 1)
        ]
        edges.extend([
            _edge("气候目标", "绿色就业", candidate=True),
            _edge("政策工具", "国际合作", candidate=True),
        ])

        paths, diagnostic = evidence_constrained_paths(
            "气候目标如何导致能源安全提高？",
            _nodes(),
            edges,
        )

        self.assertEqual("ok", diagnostic["reason"])
        self.assertGreaterEqual(len(paths), 1)
        self.assertEqual(CHAIN, paths[0].nodes)
        self.assertEqual(7, len(paths[0].edges))

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
    async def test_accepts_only_continuous_exactly_quoted_hops(self):
        chunks = [{
            "id": "chunk-1",
            "content": (
                "可再生能源投资推动国内供给增加。"
                "国内供给增加导致进口依赖下降。"
                "进口依赖下降促进能源多样化。"
            ),
        }]

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
        self.assertEqual("ungrounded_hop", diagnostic["reason"])


class PathEvaluationTest(unittest.TestCase):
    def test_scores_complete_ordered_chain(self):
        edges = [
            _edge(CHAIN[index], CHAIN[index + 1])
            for index in range(len(CHAIN) - 1)
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
