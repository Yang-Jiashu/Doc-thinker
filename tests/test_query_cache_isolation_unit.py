import unittest

from graphcore.coregraph.operate import _query_cache_scope


class _CacheStorage:
    def __init__(self, workspace: str):
        self.workspace = workspace


class QueryCacheIsolationUnitTest(unittest.TestCase):
    def test_scope_changes_between_sessions(self):
        config = {"working_dir": "C:/data/shared"}
        first = _query_cache_scope(_CacheStorage("session_00001"), config, "ctx")
        second = _query_cache_scope(_CacheStorage("session_00002"), config, "ctx")
        self.assertNotEqual(first, second)

    def test_scope_changes_when_retrieved_context_changes(self):
        storage = _CacheStorage("session_00001")
        config = {"working_dir": "C:/data/session-1"}
        before = _query_cache_scope(storage, config, "old graph context")
        after = _query_cache_scope(storage, config, "new graph context")
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
