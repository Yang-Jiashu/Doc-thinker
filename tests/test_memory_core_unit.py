import unittest
from docthinker.memory_core import AgentMemoryCore


class _FakeClawManager:
    def __init__(self):
        self.updated = False

    async def build_memory_context(self, query, *, enable_archive=True):
        return f"memory for {query}"

    async def post_query_update(self, question, answer, session_id=None, timestamp=None):
        self.updated = True


class _FakeExpandedManager:
    def __init__(self):
        self.hit_entities = []
        self.usage_recorded = False

    def match_nodes(self, *, query, top_k=2, min_score=0.2, memory_terms=None):
        return [
            {
                "entity": "Working Memory",
                "description": "A short-lived agent memory layer",
                "score": 0.9,
                "root_ids": ["Agent Memory"],
            }
        ][:top_k]

    def mark_hits(self, entities):
        self.hit_entities.extend(entities)

    def build_forced_instruction(self, matches, *, limit=2):
        names = ", ".join(m["entity"] for m in matches[:limit])
        return f"use expanded nodes: {names}"

    def record_response_usage(self, *, answer, matches, attached_entities=None):
        self.usage_recorded = True
        return {"used": ["Working Memory"], "promoted": []}


class _FakeEpisode:
    episode_id = "ep-1"
    summary = "A previous agent solved a planning problem by recalling recent goals."
    source_type = "chat"
    concepts = ["planning", "goals"]
    entity_ids = ["Agent Memory"]


class _FakeMemoryEngine:
    def __init__(self):
        self.added = False

    async def retrieve_analogies(self, query, *, top_k=10, then_spread=True, spread_top_k=5):
        return [(_FakeEpisode(), 0.82, "similar goal-driven recall")]

    async def add_observation(self, **kwargs):
        self.added = True
        return _FakeEpisode()


class AgentMemoryCoreUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_recall_merges_claw_context_and_expanded_nodes(self):
        claw = _FakeClawManager()
        expanded = _FakeExpandedManager()
        core = AgentMemoryCore(
            get_claw_manager=lambda _sid: claw,
            get_expanded_node_manager=lambda _sid: expanded,
            get_session_rag=lambda _sid: None,
        )

        bundle = await core.recall(
            session_id="#00001",
            query="what should the agent remember?",
            base_instruction="base",
            mode="hybrid",
            enable_thinking=True,
            enable_expanded_matching=True,
        )

        self.assertIn("base", bundle.retrieval_instruction)
        self.assertIn("memory for what should the agent remember?", bundle.retrieval_instruction)
        self.assertIn("use expanded nodes: Working Memory", bundle.retrieval_instruction)
        self.assertEqual(1, len(bundle.memory_summaries))
        self.assertEqual(1, len(bundle.expanded_matches))
        self.assertEqual(["Working Memory"], expanded.hit_entities)
        self.assertTrue(bundle.trace.memory_context_injected)
        self.assertEqual(1, bundle.trace.expanded_hits)

    async def test_recall_includes_episodic_analogies(self):
        memory = _FakeMemoryEngine()
        core = AgentMemoryCore(
            get_claw_manager=lambda _sid: None,
            get_expanded_node_manager=lambda _sid: None,
            get_session_rag=lambda _sid: None,
            get_memory_engine=lambda _sid: memory,
        )

        bundle = await core.recall(
            session_id="#00001",
            query="how should the agent use memory?",
            enable_thinking=True,
            enable_expanded_matching=False,
        )

        self.assertEqual(1, len(bundle.episodic_matches))
        self.assertIn("情节记忆与类比参考", bundle.retrieval_instruction)
        self.assertEqual(1, bundle.trace.episodic_hits)
        self.assertEqual("episodic_recall", bundle.trace.events[0]["type"])

    async def test_after_response_updates_memory_layers(self):
        claw = _FakeClawManager()
        expanded = _FakeExpandedManager()
        memory = _FakeMemoryEngine()
        core = AgentMemoryCore(
            get_claw_manager=lambda _sid: claw,
            get_expanded_node_manager=lambda _sid: expanded,
            get_session_rag=lambda _sid: None,
            get_memory_engine=lambda _sid: memory,
            ingest_chat_turn=None,
            chat_turn_ingest_enabled=lambda: False,
        )

        result = await core.after_response(
            session_id="#00001",
            question="q",
            answer="Working Memory is useful for recent context.",
            matched_expanded=[{"entity": "Working Memory", "score": 0.9}],
        )

        self.assertTrue(result["updated"])
        self.assertTrue(result["claw_updated"])
        self.assertTrue(claw.updated)
        self.assertTrue(result["episode_added"])
        self.assertTrue(memory.added)
        self.assertTrue(expanded.usage_recorded)


if __name__ == "__main__":
    unittest.main()
