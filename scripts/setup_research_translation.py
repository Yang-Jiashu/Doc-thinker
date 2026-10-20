#!/usr/bin/env python3
"""Explicitly download pinned public OPUS assets and convert to local CT2 int8.

Build dependencies (runtime only needs the first two):
  pip install ctranslate2==4.8.2 sentencepiece==0.2.2 transformers==4.57.6 torch==2.14.1 sacremoses==0.2.0
This command never reads project documents or sends them to a provider. Public
Hub downloads use token=False. A fresh private staging directory prevents model
downloads from following destination symlinks into unrelated local files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docthinker.server.research_query_translation import (
    ASSET_FORMAT, DEFAULT_MODEL_DIR, MODEL_FILES, MODEL_NAME, MODEL_REVISION,
    _model_manifest, _safe_path, _sha256,
)

SOURCE_FILES = ("config.json", "generation_config.json", "pytorch_model.bin", "source.spm", "target.spm", "tokenizer_config.json", "vocab.json")
SOURCE_SHA256 = {
    "config.json": "d84104da12fa90da59acf5f0ed30428e43cc92574ad4b9dfa88f29eeeb2b0eb0",
    "generation_config.json": "81f06422302b9f4666f20c6f20f43b904e38ccce9326a208f82ff97c2c9699e5",
    "pytorch_model.bin": "9d8ceb91d103ef89400c9d9d62328b4858743cf8924878aee3b8afc594242ce0",
    "source.spm": "e27a3a1b539f4959ec72ea60e453f49156289f95d4e6000b29332efc45616203",
    "target.spm": "6a881f4717cd7265f53fea54fd3dc689c767c05338fac7a4590f3088cb2d7855",
    "tokenizer_config.json": "53cc7a6f05224f287476af4798d96b1b85da2edae60257e58633f092c7b5b823",
    "vocab.json": "c0f79ee4c413ccb24cd808a0c727eec85b2130f23c82802cc1caeb07b5f0d458",
}
ATTRIBUTION = """OPUS-MT Chinese-to-English translation model
Original authors: Language Technology Research Group, University of Helsinki.
Source: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
Revision: cf109095479db38d6df799875e34039d4938aaa6
License: Creative Commons Attribution 4.0 International (CC-BY-4.0).
License text: https://creativecommons.org/licenses/by/4.0/legalcode
Modifications: converted from Hugging Face Marian weights to CTranslate2 format
and quantized to int8 using CTranslate2 4.8.2. Not endorsed by original authors.
Citation: Jorg Tiedemann and Santhosh Thottingal, OPUS-MT — Building open
translation services for the World, EAMT 2020.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    destination = args.output.expanduser().absolute()
    if not _safe_path(destination):
        raise SystemExit("Refusing a symlink destination")
    if destination.exists():
        _model_manifest(destination)
        print(json.dumps({"status": "already_installed", "model": MODEL_NAME, "revision": MODEL_REVISION}))
        return
    # Do not leave downloaded model weights in a source checkout or traverse a
    # symlink when constructing the private staging directory.
    repository = Path(__file__).resolve().parents[1]
    if destination.is_relative_to(repository):
        raise SystemExit("Choose a model cache outside the source checkout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _safe_path(destination.parent):
        raise SystemExit("Refusing a symlink parent")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from huggingface_hub import hf_hub_download
    import ctranslate2
    import torch

    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".translation-build-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        source = stage / "source"
        source.mkdir()

        def download(name):
            hf_hub_download(repo_id=MODEL_NAME, revision=MODEL_REVISION, filename=name,
                            local_dir=source, token=False)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, SOURCE_FILES))
        source_hashes = {}
        for name in SOURCE_FILES:
            asset = source / name
            if asset.is_symlink() or not asset.is_file():
                raise ValueError("Invalid downloaded asset")
            source_hashes[name] = _sha256(asset)
            if source_hashes[name] != SOURCE_SHA256[name]:
                raise ValueError("Pinned source asset checksum mismatch")
        output = stage / "converted"
        converter = ctranslate2.converters.TransformersConverter(str(source), load_as_float16=False)
        converter.convert(str(output), quantization="int8", force=False)
        for name in ("source.spm", "target.spm"):
            shutil.copyfile(source / name, output / name)
        (output / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
        actual = {p.name for p in output.iterdir()}
        if actual != set(MODEL_FILES):
            raise ValueError("Unexpected conversion asset layout")
        hashes = {name: _sha256(output / name) for name in MODEL_FILES}
        manifest = {"model": MODEL_NAME, "revision": MODEL_REVISION, "format": ASSET_FORMAT,
                    "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "attribution": "Language Technology Research Group, University of Helsinki",
                    "source_url": "https://huggingface.co/" + MODEL_NAME + "/tree/" + MODEL_REVISION,
                    "modifications": "Hugging Face Marian converted to CTranslate2 and quantized to int8",
                    "quantization": "int8", "source_sha256": source_hashes, "sha256": hashes,
                    "build_versions": {name: importlib.metadata.version(name) for name in
                                       ("ctranslate2", "sentencepiece", "transformers", "torch")},
                    "setup_seconds": round(time.perf_counter() - started, 3)}
        (output / "docthinker-model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _model_manifest(output)
        if not _safe_path(destination) or destination.exists():
            raise ValueError("Destination changed during setup")
        output.rename(destination)
    print(json.dumps({"status": "installed", "model": MODEL_NAME, "revision": MODEL_REVISION,
                      "bytes": sum((destination / name).stat().st_size for name in MODEL_FILES),
                      "setup_seconds": manifest["setup_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
