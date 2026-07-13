import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docthinker.config import DocThinkerConfig
from docthinker.session_manager import SessionManager


class _FakeDocThinker:
    def __init__(self, *, config, graphcore_kwargs):
        self.config = config
        self.graphcore_kwargs = graphcore_kwargs


class SessionFileStatusUnitTest(unittest.TestCase):
    def test_get_session_rag_passes_distinct_workspace_and_working_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = SessionManager(
                base_storage_path=str(Path(tmp) / "_system"),
                data_root_path=str(Path(tmp) / "data"),
            )
            first = sm.create_session("first")
            second = sm.create_session("second")
            config = DocThinkerConfig(working_dir=str(Path(tmp) / "global"))

            with patch("docthinker.session_manager.DocThinker", _FakeDocThinker):
                rag_a = sm.get_session_rag(
                    first["id"], config, {"working_dir": "unsafe-global"}
                )
                rag_b = sm.get_session_rag(second["id"], config)

            self.assertNotEqual(
                rag_a.graphcore_kwargs["workspace"],
                rag_b.graphcore_kwargs["workspace"],
            )
            self.assertNotIn("working_dir", rag_a.graphcore_kwargs)
            self.assertEqual(first["path"], rag_a.config.working_dir)
            self.assertEqual(second["path"], rag_b.config.working_dir)

    def test_new_sessions_receive_distinct_graphcore_workspaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "#00001" / "knowledge"
            second = root / "#00002" / "knowledge"
            first.mkdir(parents=True)
            second.mkdir(parents=True)

            workspace_a = SessionManager._graphcore_workspace_for_session(
                "#00001", first
            )
            workspace_b = SessionManager._graphcore_workspace_for_session(
                "#00002", second
            )

            self.assertNotEqual(workspace_a, workspace_b)
            self.assertEqual("session_00001", workspace_a)
            self.assertEqual("session_00002", workspace_b)

    def test_existing_graphcore_files_keep_legacy_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp)
            (working_dir / "graph_chunk_entity_relation.graphml").write_text(
                "<graphml />", encoding="utf-8"
            )
            self.assertEqual(
                "",
                SessionManager._graphcore_workspace_for_session(
                    "#00014", working_dir
                ),
            )

    def test_status_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_storage = Path(tmp) / "_system"
            data_root = Path(tmp) / "data"
            sm = SessionManager(
                base_storage_path=str(base_storage),
                data_root_path=str(data_root),
            )
            sid = sm.create_session("test")["id"]
            sm.add_document_record(
                sid,
                "sample.pdf",
                file_path=str((data_root / sid / "content" / "sample.pdf")),
                file_ext=".pdf",
            )

            files = sm.get_files(sid)
            self.assertEqual(1, len(files))
            self.assertEqual("pending", files[0]["status"])

            self.assertTrue(sm.set_document_status(sid, "sample.pdf", "processing"))
            files = sm.get_files(sid)
            self.assertEqual("processing", files[0]["status"])

            self.assertTrue(sm.set_document_status(sid, "sample.pdf", "processed"))
            files = sm.get_files(sid)
            self.assertEqual("processed", files[0]["status"])


if __name__ == "__main__":
    unittest.main()
