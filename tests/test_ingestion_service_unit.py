import unittest

from docthinker.services.ingestion_service import IngestionService


class _GraphCore:
    def __init__(self):
        self.items = []
        self.insert_calls = []

    async def ainsert(self, text: str, **kwargs):
        self.items.append(text)
        self.insert_calls.append({"text": text, **kwargs})


class _RAG:
    def __init__(self):
        self.graphcore = None
        self._lr = _GraphCore()
        self.folders = []

    async def _ensure_graphcore_initialized(self):
        self.graphcore = self._lr

    async def process_folder_complete(self, folder_path: str):
        self.folders.append(folder_path)


class _SessionManager:
    def __init__(self):
        self.rags = {}

    def get_session_rag(self, session_id, _config):
        if session_id not in self.rags:
            self.rags[session_id] = _RAG()
        return self.rags[session_id]


async def _fake_llm():
    async def f(_prompt: str, **_):
        return "ok"
    return f


async def _fake_embed():
    return object()


def _create_config():
    return object()


class IngestionServiceUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_text_is_session_scoped(self):
        global_rag = _RAG()
        sm = _SessionManager()
        svc = IngestionService(
            rag_global=global_rag,
            session_manager=sm,
            create_rag_config=_create_config,
            get_llm_model_func=_fake_llm,
            get_embedding_func=_fake_embed,
        )
        await svc.ingest_text("hello", session_id="s1")
        self.assertIn("hello", sm.rags["s1"].graphcore.items)
        self.assertIsNone(global_rag.graphcore)

    async def test_same_text_can_be_ingested_into_two_sessions(self):
        global_rag = _RAG()
        sm = _SessionManager()
        svc = IngestionService(
            rag_global=global_rag,
            session_manager=sm,
            create_rag_config=_create_config,
            get_llm_model_func=_fake_llm,
            get_embedding_func=_fake_embed,
        )

        await svc.ingest_text("same document", session_id="s2")
        await svc.ingest_text("same document", session_id="s3")

        self.assertEqual(["same document"], sm.rags["s2"].graphcore.items)
        self.assertEqual(["same document"], sm.rags["s3"].graphcore.items)
        self.assertIsNot(sm.rags["s2"].graphcore, sm.rags["s3"].graphcore)
        s2_id = sm.rags["s2"].graphcore.insert_calls[0]["ids"]
        s3_id = sm.rags["s3"].graphcore.insert_calls[0]["ids"]
        self.assertNotEqual(s2_id, s3_id)
        self.assertTrue(s2_id.startswith("doc-session-"))

    async def test_same_text_is_deduplicated_deterministically_inside_one_session(self):
        global_rag = _RAG()
        sm = _SessionManager()
        svc = IngestionService(
            rag_global=global_rag,
            session_manager=sm,
            create_rag_config=_create_config,
            get_llm_model_func=_fake_llm,
            get_embedding_func=_fake_embed,
        )

        await svc.ingest_text("same document", session_id="s4", file_path="same.txt")
        await svc.ingest_text("same document", session_id="s4", file_path="renamed.txt")

        calls = sm.rags["s4"].graphcore.insert_calls
        self.assertEqual(calls[0]["ids"], calls[1]["ids"])
        self.assertEqual("same.txt", calls[0]["file_paths"])


if __name__ == "__main__":
    unittest.main()
