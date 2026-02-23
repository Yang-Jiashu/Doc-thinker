"""
Batch evaluation for MMLongBench categories using local Qwen endpoints.

This mirrors scripts/run_batch_eval.py, but datasets are the MMLongBench
subfolders under data/mineru_results/mmlongbench (Acad/Admin/Broch/...).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

# ---- Project setup (same as run_batch_eval.py) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("LIGHYRAG_FORCE_LOCAL", "1")

from scripts.run_qa_eval import run_evaluation, parse_questions_file


# ==== Dataset hyperparameters for MMLongBench ====
# 可选 key：
#   acad, admin, broch, fin, guid, res, tut
DEFAULT_DATASET = "broch"

DATASETS: Dict[str, Dict[str, Path]] = {
    "acad": {
        "raw_root": Path("data/mineru_results/mmlongbench/Acad"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Acad"),
        "output_dir": Path("outputs/mmlongbench_acad_eval"),
    },
    "admin": {
        "raw_root": Path("data/mineru_results/mmlongbench/Admin"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Admin"),
        "output_dir": Path("outputs/mmlongbench_admin_eval"),
    },
    "broch": {
        "raw_root": Path("data/mineru_results/mmlongbench/Broch"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Broch"),
        "output_dir": Path("outputs/mmlongbench_broch_eval"),
    },
    "fin": {
        "raw_root": Path("data/mineru_results/mmlongbench/Fin"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Fin"),
        "output_dir": Path("outputs/mmlongbench_fin_eval"),
    },
    "guid": {
        "raw_root": Path("data/mineru_results/mmlongbench/Guid"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Guid"),
        "output_dir": Path("outputs/mmlongbench_guid_eval"),
    },
    "res": {
        "raw_root": Path("data/mineru_results/mmlongbench/Res"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Res"),
        "output_dir": Path("outputs/mmlongbench_res_eval"),
    },
    "tut": {
        "raw_root": Path("data/mineru_results/mmlongbench/Tut"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Tut"),
        "output_dir": Path("outputs/mmlongbench_tut_eval"),
    },
}


# ==== Local service defaults (完全照抄 run_batch_eval.py) ====
LOCAL_CHAT_HOST = "http://127.0.0.1:22003/v1"
LOCAL_CHAT_MODEL = "/home/yjs/robot/VLM/32B"

LOCAL_EMBED_HOST = "http://127.0.0.1:8808/v1/embeddings"
LOCAL_EMBED_MODEL = "/home/yjs/robot/VLM/8Bembedding"

LOCAL_AUTOTHINK_HOST = "http://127.0.0.1:8810/v1"
LOCAL_AUTOTHINK_MODEL = "/home/yjs/robot/VLM/3B"

# vLLM typically ignores API key; placeholder is enough
LOCAL_API_KEY = "EMPTY"


def iter_question_files(raw_root: Path) -> Iterable[Path]:
    """
    在 raw_root 下递归找所有 .json / .jsonl 问题文件。

    对 mmlongbench 来说，会自动跳过 content_list.json 等非 QA 文件，
    因为 parse_questions_file 解析失败会被上层 try/except 忽略。
    """
    for file in sorted(raw_root.rglob("*")):
        if file.suffix.lower() in {".json", ".jsonl"}:
            yield file


def build_api_config() -> dict:
    # Auto-think to local
    os.environ["AUTO_THINK_API_BASE"] = LOCAL_AUTOTHINK_HOST
    os.environ["AUTO_THINK_API_KEY"] = LOCAL_API_KEY
    os.environ["AUTO_THINK_MODEL"] = LOCAL_AUTOTHINK_MODEL

    # Embedding config
    os.environ["EMBEDDING_MODEL"] = LOCAL_EMBED_MODEL
    embed_model = LOCAL_EMBED_MODEL

    # Evaluator model (same as generator by default)
    os.environ["EVAL_MODEL"] = LOCAL_CHAT_MODEL

    return {
        "api_key": LOCAL_API_KEY,
        "chat_host": LOCAL_CHAT_HOST,
        "embed_host": LOCAL_EMBED_HOST,
        "embed_model": embed_model,
        "model": LOCAL_CHAT_MODEL,
    }


def validate_dataset(name: str) -> Dict[str, Path]:
    cfg = DATASETS.get(name)
    if not cfg:
        raise SystemExit(f"Unknown dataset '{name}'. Choices: {', '.join(DATASETS)}")
    if not cfg["raw_root"].exists():
        raise SystemExit(f"Raw root not found: {cfg['raw_root']}")
    if not cfg["mineru_root"].exists():
        raise SystemExit(f"MinerU root not found: {cfg['mineru_root']}")
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def collect_manifest_entry(question_file: Path, output_path: Path) -> Dict[str, str]:
    return {
        "question_file": str(question_file),
        "result_file": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch eval for MMLongBench categories using local Qwen endpoints.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset key to run "
             f"(default={DEFAULT_DATASET}; choices: {', '.join(DATASETS)})",
    )
    parser.add_argument(
        "--multi-step",
        action="store_true",
        help="Enable query decomposition + multi-step retrieval.",
    )
    parser.add_argument(
        "--no-multi-step",
        dest="multi_step",
        action="store_false",
        help="Disable query decomposition + multi-step retrieval.",
    )
    parser.add_argument(
        "--max-sub-questions",
        type=int,
        default=4,
        help="Maximum sub-questions when multi-step is enabled.",
    )
    parser.add_argument(
        "--max-sub-parallel",
        type=int,
        default=3,
        help="Maximum parallel sub-questions.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print sub-question plans and intermediate answers.",
    )
    parser.set_defaults(multi_step=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset or DEFAULT_DATASET
    cfg = validate_dataset(dataset_name)
    api_config = build_api_config()

    detail_log = cfg["output_dir"] / "detail_log.json"
    manifest: List[Dict[str, str]] = []

    async def run_all() -> None:
        for q_file in iter_question_files(cfg["raw_root"]):
            try:
                _ = parse_questions_file(q_file)
            except Exception as exc:  # noqa: PERF203
                print(f"[SKIP] {q_file}: failed to parse questions ({exc})")
                continue

            folder_name = q_file.parent.name
            folder_output_dir = cfg["output_dir"] / folder_name
            folder_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = folder_output_dir / f"{q_file.stem}_results.json"
            print(f"\n===== Evaluating {q_file} =====")
            await run_evaluation(
                q_file,
                cfg["mineru_root"],
                output_path=output_path,
                detail_log_path=detail_log,
                folder_name=folder_name,
                api_config=api_config,
                enable_multi_step=args.multi_step,
                max_sub_questions=args.max_sub_questions,
                max_parallel_subqueries=args.max_sub_parallel,
                show_plan=args.show_plan,
            )
            manifest.append(collect_manifest_entry(q_file, output_path))

    asyncio.run(run_all())

    manifest_path = cfg["output_dir"] / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
