import unittest

from raganything.cognitive.processor import CognitiveProcessor, CognitiveInsight, PotentialLink


class _Entity:
    def __init__(self, entity_id: str, name: str, entity_type: str = "concept", confidence: float = 0.9):
        self.id = entity_id
        self.name = name
        self.type = entity_type
        self.confidence = confidence


class _KG:
    def __init__(self):
        self._entities = [
            _Entity("e1", "Project Chronos"),
            _Entity("e2", "Temporal Database"),
            _Entity("e3", "Emotion Tagging"),
        ]

    def search_entities(self, query: str, limit: int = 20):
        q = query.lower()
        hits = [e for e in self._entities if q in e.name.lower()]
        return hits[:limit]


async def _fake_llm(_prompt: str, **_):
    return """{\"summary\":\"s\",\"concepts\":[\"Project Chronos\",\"Temporal\"],\"reasoning\":\"r\",\"action_items\":[]}"""


class CognitiveAssociateUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_associate_produces_links(self):
        p = CognitiveProcessor(llm_func=_fake_llm, embedding_func=None, knowledge_graph=_KG())
        insight = await p.process("x", source_type="test")
        self.assertTrue(isinstance(insight, CognitiveInsight))
        self.assertGreaterEqual(len(insight.potential_links), 1)
        self.assertTrue(all(isinstance(l, PotentialLink) for l in insight.potential_links))


if __name__ == "__main__":
    unittest.main()

