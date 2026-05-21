import unittest
from docthinker.memory_core import AgentMemoryBackends, AgentMemoryCore, MemoryPolicy


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


class _ProtocolConversationBackend:
    def __init__(self):
        self.consolidated = False

    async def build_context(self, session_id, query):
        return f"protocol memory {session_id}: {query}"

    async def consolidate(self, session_id, question, answer):
        self.consolidated = True
        return True


class _ProtocolEpisodicBackend:
    def __init__(self):
        self.written = False
        self.last_top_k = None

    async def retrieve(self, session_id, query, *, top_k):
        self.last_top_k = top_k
        return [{
            "episode_id": "proto-ep",
            "summary": "Protocol backend recalled a reusable agent habit.",
            "score": 0.7,
            "reason": "same memory contract",
        }]

    async def write(self, session_id, question, answer, *, concepts, timestamp):
        self.written = True
        return "proto-ep-new"


class _ProtocolExpandedBackend:
    def __init__(self):
        self.recorded = False
        self.last_top_k = None
        self.last_min_score = None

    def match(self, session_id, query, *, top_k, min_score):
        self.last_top_k = top_k
        self.last_min_score = min_score
        return [{"entity": "Protocol Memory", "score": 0.8, "root_ids": ["Agent Memory"]}]

    def build_instruction(self, session_id, matches, *, limit):
        return "use protocol expanded memory"

    def record_usage(self, session_id, answer, matches, *, attached_entities):
        self.recorded = True
        return ["Protocol Memory"]

    def get_record(self, session_id, name):
        return {"description": "A backend supplied by a plugin.", "root_ids": []}


class _ProtocolGraphBackend:
    def __init__(self):
        self.promoted = []

    async def promote(self, session_id, promoted_names, *, answer_entities, expanded_backend):
        self.promoted = list(promoted_names)
        return self.promoted


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

    async def test_accepts_protocol_backends_for_plugin_usage(self):
        conversation = _ProtocolConversationBackend()
        episodic = _ProtocolEpisodicBackend()
        expanded = _ProtocolExpandedBackend()
        graph = _ProtocolGraphBackend()
        core = AgentMemoryCore(
            backends=AgentMemoryBackends(
                conversation=conversation,
                episodic=episodic,
                expanded=expanded,
                graph=graph,
            )
        )

        bundle = await core.recall(
            session_id="plugin-session",
            query="how can plugins provide memory?",
            enable_thinking=True,
            enable_expanded_matching=True,
        )
        self.assertIn("protocol memory plugin-session", bundle.retrieval_instruction)
        self.assertIn("Protocol backend recalled", bundle.retrieval_instruction)
        self.assertIn("use protocol expanded memory", bundle.retrieval_instruction)

        result = await core.after_response(
            session_id="plugin-session",
            question="q",
            answer="Protocol Memory should be promoted.",
            matched_expanded=bundle.expanded_matches,
        )
        self.assertTrue(result["claw_updated"])
        self.assertTrue(result["episode_added"])
        self.assertEqual(["Protocol Memory"], result["expanded_promoted"])
        self.assertTrue(conversation.consolidated)
        self.assertTrue(episodic.written)
        self.assertTrue(expanded.recorded)
        self.assertEqual(["Protocol Memory"], graph.promoted)

    async def test_memory_policy_controls_layers_and_recall_breadth(self):
        conversation = _ProtocolConversationBackend()
        episodic = _ProtocolEpisodicBackend()
        expanded = _ProtocolExpandedBackend()
        graph = _ProtocolGraphBackend()
        core = AgentMemoryCore(
            backends=AgentMemoryBackends(
                conversation=conversation,
                episodic=episodic,
                expanded=expanded,
                graph=graph,
            ),
            policy=MemoryPolicy(
                episodic_top_k=1,
                expanded_top_k=4,
                expanded_min_score=0.55,
                enabled_layers=("episodic", "expanded"),
            ),
        )

        bundle = await core.recall(
            session_id="policy-session",
            query="policy driven memory",
            enable_thinking=True,
            enable_expanded_matching=True,
        )

        self.assertNotIn("protocol memory policy-session", bundle.retrieval_instruction)
        self.assertIn("Protocol backend recalled", bundle.retrieval_instruction)
        self.assertEqual(1, episodic.last_top_k)
        self.assertEqual(4, expanded.last_top_k)
        self.assertEqual(0.55, expanded.last_min_score)

        result = await core.after_response(
            session_id="policy-session",
            question="q",
            answer="Protocol Memory should be remembered.",
            matched_expanded=bundle.expanded_matches,
        )
        self.assertFalse(result["claw_updated"])
        self.assertEqual([], graph.promoted)
        self.assertEqual(["episodic", "expanded"], result["memory_trace"]["consolidation"]["enabled_layers"])


if __name__ == "__main__":
    unittest.main()
