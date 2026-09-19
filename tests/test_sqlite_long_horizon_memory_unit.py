import tempfile
import unittest
from pathlib import Path

from docthinker.memory_core import SQLiteLongHorizonBackend
from docthinker.memory_core import AgentMemoryBackends, AgentMemoryCore, MemoryPolicy


class SQLiteLongHorizonMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "long_horizon.db"

    def tearDown(self):
        self.tempdir.cleanup()

    def _backend(self) -> SQLiteLongHorizonBackend:
        return SQLiteLongHorizonBackend(self.db_path)

    def test_memory_survives_backend_recreation(self):
        first = self._backend()
        stored = first.consolidate(
            "persistent-session",
            "请记住这个长期产品规则",
            "DocThinker should preserve durable product rules across backend restarts and future conversations.",
            concepts=["DocThinker", "durable memory"],
            scope="session",
            timestamp=100.0,
        )
        self.assertIsNotNone(stored)
        self.assertEqual("sqlite", first.stats("persistent-session")["storage"])

        second = self._backend()
        items = second.list_insights("persistent-session")
        self.assertEqual(1, len(items))
        self.assertEqual(stored["id"], items[0]["id"])
        self.assertEqual(1, items[0]["version"])
        recalled = second.retrieve(
            "persistent-session",
            "durable product rules across restart",
            scopes=("session",),
            top_k=3,
            min_confidence=0.35,
        )
        self.assertEqual(1, len(recalled))
        self.assertEqual("store", second.last_write_decision()["action"])

    def test_update_delete_and_restore_keep_immutable_revisions(self):
        backend = self._backend()
        stored = backend.consolidate(
            "revision-session",
            "记住宣传策略规则",
            "Product messaging should target developers and explain implementation details with verifiable examples.",
            concepts=["messaging", "developers"],
            scope="session",
            timestamp=200.0,
        )
        self.assertIsNotNone(stored)
        initial_revision = backend.list_revisions(stored["id"], "revision-session")[0]
        self.assertEqual("create", initial_revision["action"])

        updated = backend.update_insight(
            stored["id"],
            {"summary": "Product messaging should target ordinary users and avoid unexplained technical terms."},
            "revision-session",
        )
        self.assertEqual(2, updated["version"])
        self.assertIn("ordinary users", updated["summary"])
        revisions = backend.list_revisions(stored["id"], "revision-session")
        self.assertEqual([2, 1], [item["version"] for item in revisions])

        self.assertTrue(backend.delete_insight(stored["id"], "revision-session"))
        self.assertEqual([], backend.list_insights("revision-session"))
        deleted_revisions = backend.list_revisions(stored["id"], "revision-session")
        self.assertEqual("delete", deleted_revisions[0]["action"])

        restored = backend.restore_revision(
            stored["id"],
            initial_revision["revision_id"],
            "revision-session",
        )
        self.assertIsNotNone(restored)
        self.assertEqual("active", restored["status"])
        self.assertEqual(4, restored["version"])
        self.assertIn("target developers", restored["summary"])

        reopened = self._backend().list_insights("revision-session")
        self.assertEqual(1, len(reopened))
        self.assertEqual(4, reopened[0]["version"])

    def test_memory_edges_are_persistent(self):
        backend = self._backend()
        first = backend.consolidate(
            "edge-session",
            "记住首次使用体验",
            "Users abandoned onboarding when configuration was required before any visible result appeared.",
            concepts=["onboarding", "configuration"],
            scope="session",
            timestamp=300.0,
        )
        second = backend.consolidate(
            "edge-session",
            "记住导出体验规则",
            "People completed export more often when a preview appeared before format settings were requested.",
            concepts=["export", "preview"],
            scope="session",
            timestamp=301.0,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first["id"], second["id"])

        edge = backend.upsert_edge(
            first["id"],
            second["id"],
            "analogous_to",
            weight=0.82,
            evidence={"reason": "show value before configuration"},
        )
        self.assertEqual(0.82, edge["weight"])

        reopened = self._backend()
        edges = reopened.list_edges(session_id="edge-session")
        self.assertEqual(1, len(edges))
        self.assertEqual("analogous_to", edges[0]["relation_type"])
        self.assertEqual(
            "show value before configuration",
            edges[0]["evidence"]["reason"],
        )
        self.assertEqual(1, reopened.stats("edge-session")["edge_count"])

    def test_retrieve_expands_from_direct_hit_along_memory_edges(self):
        backend = self._backend()
        seed = backend.consolidate(
            "graph-recall-session",
            "记住结账流失案例",
            "Checkout abandonment increased when customers saw mandatory registration before payment.",
            concepts=["checkout", "registration"],
            scope="session",
            timestamp=400.0,
        )
        related = backend.consolidate(
            "graph-recall-session",
            "记住试用转化案例",
            "Trial conversion improved after a visible result appeared before configuration settings.",
            concepts=["trial", "configuration"],
            scope="session",
            timestamp=401.0,
        )
        backend.upsert_edge(
            seed["id"],
            related["id"],
            "analogous_to",
            weight=0.9,
            evidence={"structure": "show value before friction"},
        )

        recalled = backend.retrieve(
            "graph-recall-session",
            "checkout abandonment registration",
            scopes=("session",),
            top_k=3,
            min_confidence=0.35,
        )
        by_id = {item["id"]: item for item in recalled}
        self.assertIn(seed["id"], by_id)
        self.assertIn(related["id"], by_id)
        self.assertEqual("graph", by_id[related["id"]]["recall_origin"])
        self.assertEqual("analogous_to", by_id[related["id"]]["graph_path"][0]["relation_type"])

    def test_cognition_is_separate_and_keeps_memory_evidence(self):
        backend = self._backend()
        first = backend.consolidate(
            "cognition-session",
            "记录项目A",
            "Project A accepted five custom features on a small budget and finished nineteen days late.",
            concepts=["custom scope", "delay"],
            scope="session",
            timestamp=500.0,
        )
        second = backend.consolidate(
            "cognition-session",
            "记录项目B",
            "Project B limited delivery to two core features and shipped on schedule.",
            concepts=["core scope", "on schedule"],
            scope="session",
            timestamp=501.0,
        )
        cognition = backend.create_cognition(
            "cognition-session",
            "Small-budget projects with more than three custom features should be split into phases.",
            evidence_memory_ids=[first["id"], second["id"]],
            cognition_type="induction",
            conditions=["budget is small", "custom features > 3"],
            confidence=0.78,
        )

        self.assertEqual(2, len(backend.list_insights("cognition-session")))
        self.assertEqual(1, len(backend.list_cognitions("cognition-session")))
        self.assertEqual({first["id"], second["id"]}, set(cognition["evidence_memory_ids"]))
        recalled_memories = backend.retrieve(
            "cognition-session",
            "Project A custom features delay",
            scopes=("session",),
            top_k=3,
            min_confidence=0.35,
        )
        recalled_cognitions = backend.retrieve_cognitions(
            "cognition-session",
            "What should we quote?",
            scopes=("session",),
            top_k=2,
            min_confidence=0.4,
            evidence_memory_ids=[item["id"] for item in recalled_memories],
        )
        self.assertEqual(cognition["id"], recalled_cognitions[0]["id"])
        self.assertEqual("evidence_graph", recalled_cognitions[0]["recall_origin"])

        backend.update_insight(first["id"], {"confidence": 0.4}, "cognition-session")
        self.assertEqual([], backend.list_cognitions("cognition-session", status="active"))
        needs_review = backend.list_cognitions("cognition-session", status="needs_review")
        self.assertEqual(cognition["id"], needs_review[0]["id"])
        self.assertEqual(2, len(backend.list_insights("cognition-session")))

    def test_similar_experiences_are_linked_not_overwritten(self):
        backend = self._backend()
        first = backend.consolidate(
            "preserve-experience-session",
            "记录第一次延期",
            "A small fixed-price project accepted five custom features and finished nineteen days late.",
            concepts=["fixed price", "custom features", "delay"],
            scope="session",
            timestamp=550.0,
        )
        second = backend.consolidate(
            "preserve-experience-session",
            "记录第二次延期",
            "Another small fixed-price project accepted four custom features and finished twelve days late.",
            concepts=["fixed price", "custom features", "delay"],
            scope="session",
            timestamp=551.0,
        )
        items = backend.list_insights("preserve-experience-session")
        self.assertEqual(2, len(items))
        self.assertNotEqual(first["id"], second["id"])
        summaries = {item["summary"] for item in items}
        self.assertIn(first["summary"], summaries)
        self.assertIn(second["summary"], summaries)
        edges = backend.list_edges(session_id="preserve-experience-session")
        self.assertTrue(any(edge["relation_type"] in {"similar_to", "analogous_to"} for edge in edges))


class CognitionRecallIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_core_injects_graph_memory_and_cognition(self):
        with tempfile.TemporaryDirectory() as td:
            backend = SQLiteLongHorizonBackend(Path(td) / "memory.db")
            first = backend.consolidate(
                "core-cognition-session",
                "记录延期案例",
                "A fixed-price project with many custom features was delayed and required repeated rework.",
                concepts=["fixed price", "custom features"],
                scope="session",
                timestamp=600.0,
            )
            second = backend.consolidate(
                "core-cognition-session",
                "记录分期案例",
                "A phased project delivered core features first and reached acceptance on schedule.",
                concepts=["phased delivery", "core features"],
                scope="session",
                timestamp=601.0,
            )
            backend.upsert_edge(first["id"], second["id"], "analogous_to", weight=0.9)
            backend.create_cognition(
                "core-cognition-session",
                "High custom scope under a fixed price should trigger phased delivery.",
                evidence_memory_ids=[first["id"], second["id"]],
                confidence=0.8,
            )
            core = AgentMemoryCore(
                backends=AgentMemoryBackends(long_horizon=backend),
                policy=MemoryPolicy(
                    long_horizon_top_k=3,
                    cognition_top_k=2,
                    enabled_layers=("long_horizon", "cognition"),
                ),
            )
            bundle = await core.recall(
                session_id="core-cognition-session",
                query="fixed-price custom features risk",
            )
            self.assertTrue(any(item.get("recall_origin") == "graph" for item in bundle.long_horizon_matches))
            self.assertEqual(1, len(bundle.cognition_matches))
            self.assertIn("Derived cognition", bundle.retrieval_instruction)
            event_types = {event["type"] for event in bundle.trace.events}
            self.assertIn("memory_graph_recall", event_types)
            self.assertIn("cognition_recall", event_types)


if __name__ == "__main__":
    unittest.main()
