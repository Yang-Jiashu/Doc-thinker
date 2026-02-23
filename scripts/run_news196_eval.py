from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force local endpoints (aligned with run_batch_eval defaults)
os.environ.setdefault("LIGHYRAG_FORCE_LOCAL", "1")

from scripts.run_qa_eval import run_evaluation, parse_questions_file  # noqa: E402


# Single-folder eval config
RAW_DIR = Path("data/raw/News/196")
MINERU_ROOT = Path("data/mineru_output/news")
OUTPUT_DIR = Path("outputs/news_196_eval")

# Local service defaults
LOCAL_CHAT_HOST = "http://127.0.0.1:22004/v1"
LOCAL_CHAT_MODEL = "/home/yjs/robot/VLM/32B"
LOCAL_EMBED_HOST = "http://127.0.0.1:8808/v1/embeddings"
LOCAL_EMBED_MODEL = "/home/yjs/robot/VLM/8Bembedding"
LOCAL_AUTOTHINK_HOST = "http://127.0.0.1:8810/v1"
LOCAL_AUTOTHINK_MODEL = "/home/yjs/robot/VLM/3B"
LOCAL_API_KEY = "EMPTY"  # vLLM typically ignores API key


def iter_question_files(root: Path) -> Iterable[Path]:
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in {".json", ".jsonl"}:
            yield f


def build_api_config() -> dict:
    # Auto-think to local
    os.environ["AUTO_THINK_API_BASE"] = LOCAL_AUTOTHINK_HOST
    os.environ["AUTO_THINK_API_KEY"] = LOCAL_API_KEY
    os.environ["AUTO_THINK_MODEL"] = LOCAL_AUTOTHINK_MODEL

    # Embedding config
    os.environ["EMBEDDING_MODEL"] = LOCAL_EMBED_MODEL

    # Evaluator model (same as generator by default)
    os.environ["EVAL_MODEL"] = LOCAL_CHAT_MODEL

    return {
        "api_key": LOCAL_API_KEY,
        "chat_host": LOCAL_CHAT_HOST,
        "embed_host": LOCAL_EMBED_HOST,
        "embed_model": LOCAL_EMBED_MODEL,
        "model": LOCAL_CHAT_MODEL,
    }


async def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_log = OUTPUT_DIR / "detail_log.json"
    manifest: list[Dict[str, str]] = []
    api_config = build_api_config()

    for q_file in iter_question_files(RAW_DIR):
        try:
            _ = parse_questions_file(q_file)
        except Exception as exc:  # noqa: PERF203
            print(f"[SKIP] {q_file}: failed to parse questions ({exc})")
            continue

        folder_name = q_file.parent.name
        out_path = OUTPUT_DIR / f"{q_file.stem}_results.json"
        print(f"\n===== Evaluating {q_file} =====")
        await run_evaluation(
            q_file,
            MINERU_ROOT,
            output_path=out_path,
            detail_log_path=detail_log,
            folder_name=folder_name,
            api_config=api_config,
            enable_multi_step=True,
            max_sub_questions=4,
            max_parallel_subqueries=3,
            show_plan=False,
        )
        manifest.append({"question_file": str(q_file), "result_file": str(out_path)})

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nManifest written to {manifest_path}")


def main() -> None:
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
