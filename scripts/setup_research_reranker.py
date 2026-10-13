#!/usr/bin/env python3
"""Explicitly install the pinned public local reranker (about 136 MB).

No API credential or user document is read. Runtime inference is offline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docthinker.server.research_reranker import (
    DEFAULT_MODEL_DIR, MODEL_FILES, MODEL_HASHES, MODEL_NAME, MODEL_REVISION, _sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--verify-only", action="store_true", help="Verify already downloaded pinned files, without network")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    started = time.perf_counter()
    destination = args.output.expanduser().resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_NAME, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES),
                          local_dir=destination, token=False, max_workers=2)
    hashes = {name: _sha256(destination / name) for name in MODEL_FILES}
    if hashes != MODEL_HASHES:
        raise SystemExit("Downloaded reranker asset hash mismatch; runtime will not use it")
    manifest = {"model": MODEL_NAME, "revision": MODEL_REVISION, "sha256": hashes,
                "license": "apache-2.0", "download_seconds": round(time.perf_counter() - started, 3)}
    temporary = destination / "docthinker-model.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "docthinker-model.json")
    print(json.dumps({"status": "installed", "model": MODEL_NAME, "revision": MODEL_REVISION,
                      "directory": str(destination), "download_seconds": manifest["download_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
