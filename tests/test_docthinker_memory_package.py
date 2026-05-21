import importlib
import sys
import unittest
from pathlib import Path


class DocThinkerMemoryPackageTest(unittest.TestCase):
    def test_package_exports_memory_core_api(self):
        package_root = Path(__file__).resolve().parents[1] / "packages" / "docthinker-memory"
        sys.path.insert(0, str(package_root))
        try:
            module = importlib.import_module("docthinker_memory")
            self.assertTrue(hasattr(module, "AgentMemoryCore"))
            self.assertTrue(hasattr(module, "AgentMemoryBackends"))
            self.assertTrue(hasattr(module, "MemoryPolicy"))
        finally:
            try:
                sys.path.remove(str(package_root))
            except ValueError:
                pass
