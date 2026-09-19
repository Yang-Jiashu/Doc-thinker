import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docthinker.session_manager import SessionManager


class SessionIdNormalizationUnitTest(unittest.TestCase):
    @patch.object(SessionManager, "_find_legacy_rag_root", return_value=None)
    def test_short_hash_id_is_normalized(self, _legacy_root):
        with tempfile.TemporaryDirectory() as tmp:
            base_storage = Path(tmp) / "_system"
            data_root = Path(tmp) / "data"
            sm = SessionManager(
                base_storage_path=str(base_storage),
                data_root_path=str(data_root),
            )
            created = sm.create_session("test")
            sid = created["id"]
            self.assertEqual("#00001", sid)

            by_short_hash = sm.get_session("#1")
            by_short_num = sm.get_session("1")
            by_full = sm.get_session("#00001")

            self.assertIsNotNone(by_short_hash)
            self.assertIsNotNone(by_short_num)
            self.assertIsNotNone(by_full)
            self.assertEqual(sid, by_short_hash["id"])
            self.assertEqual(sid, by_short_num["id"])
            self.assertEqual(sid, by_full["id"])

            sm.add_message("#1", "user", "hello")
            history = sm.get_history("#00001")
            self.assertEqual(1, len(history))
            self.assertEqual("user", history[0]["role"])

    @patch.object(SessionManager, "_find_legacy_rag_root", return_value=None)
    def test_restart_preserves_numbered_session_ids_with_gaps(self, _legacy_root):
        with tempfile.TemporaryDirectory() as tmp:
            base_storage = Path(tmp) / "_system"
            data_root = Path(tmp) / "data"
            sm = SessionManager(
                base_storage_path=str(base_storage),
                data_root_path=str(data_root),
            )
            first = sm.create_session("first")["id"]
            second = sm.create_session("second")["id"]
            third = sm.create_session("third")["id"]
            self.assertEqual(("#00001", "#00002", "#00003"), (first, second, third))
            self.assertTrue(sm.delete_session(second))

            reopened = SessionManager(
                base_storage_path=str(base_storage),
                data_root_path=str(data_root),
            )
            self.assertIsNotNone(reopened.get_session(first))
            self.assertIsNotNone(reopened.get_session(third))
            self.assertIsNone(reopened.get_session(second))


if __name__ == "__main__":
    unittest.main()
