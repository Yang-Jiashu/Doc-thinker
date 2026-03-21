import unittest

from docthinker.services.ingestion_service import IngestionService


class _GraphCore:
    def __init__(self):
        self.items = []

    async def ainsert(self, text: str):
        self.items.append(text)


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
    async def test_dual_ingest_text(self):
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
        self.assertIn("hello", global_rag.graphcore.items)
        self.assertIn("hello", sm.rags["s1"].graphcore.items)


if __name__ == "__main__":
    unittest.main()
