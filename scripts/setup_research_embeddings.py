#!/usr/bin/env python3
"""Explicit public model download. No user documents, credentials or API calls.

Install docthinker[research-semantic], then run this command before starting the
service. Inference only opens this completed local directory and cannot download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docthinker.server.research_semantic import (DEFAULT_MODEL_DIR, MODEL_FILES, MODEL_NAME,
                                               MODEL_REVISION, MODEL_FILE, MODEL_SHA256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    destination = args.output.expanduser().resolve()
    snapshot_download(repo_id=MODEL_NAME, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES),
                      local_dir=destination, token=False, max_workers=2)
    hashes = {}
    for name in MODEL_FILES:
        digest = hashlib.sha256()
        with (destination / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    if hashes[MODEL_FILE] != MODEL_SHA256:
        raise SystemExit("Downloaded model hash mismatch; runtime will not use it")
    manifest = {"model": MODEL_NAME, "revision": MODEL_REVISION, "sha256": hashes,
                "download_seconds": round(time.perf_counter() - started, 3)}
    temporary = destination / "docthinker-model.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "docthinker-model.json")
    print(json.dumps({"status": "installed", "model": MODEL_NAME, "revision": MODEL_REVISION,
                      "directory": str(destination), "download_seconds": manifest["download_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
