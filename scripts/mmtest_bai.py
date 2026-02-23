"""
Batch evaluation for MMLongBench categories using BLTCY cloud APIs
with SiliconFlow reranker.

This mirrors scripts/mmtest.py, but instead of local Qwen endpoints it
talks directly to remote APIs:

- Auto-think (VLM): qwen2.5-vl-3b-instruct   -> https://api.bltcy.ai/v1/chat/completions
- Embedding:       qwen3-embedding-8b        -> https://api.bltcy.ai/v1/embeddings
- Gen / eval:      qwen3-32b                 -> https://api.bltcy.ai/v1/chat/completions
- Rerank:          BAAI/bge-reranker-v2-m3   -> https://api.siliconflow.cn/v1/rerank

Edit the hyperparameters at the top of this file, then run:

    python scripts/mmtest_bltcy.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

# ================== API hyperparameters (edit here) ==================

# ================== API hyperparameters (edit here) ==================

# Alibaba Bailian (DashScope) API key：请设置环境变量 DASHSCOPE_API_KEY 或在 .env 中配置，勿提交真实密钥
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")

# BLTCY main API key (chat / eval / embedding / auto-think)
# Reuse DASHSCOPE_API_KEY for Alibaba Bailian compatible-mode endpoints.
BLTCY_API_KEY: str = DASHSCOPE_API_KEY or os.getenv("BLTCY_API_KEY", "")

# BLTCY endpoints
# Alibaba Bailian (DashScope) compatible-mode endpoints
# VLMClient will append /chat/completions if not present.
BLTCY_CHAT_HOST: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BLTCY_CHAT_COMPLETIONS_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# Embedding endpoint follows OpenAI schema under the same base.
BLTCY_EMBED_HOST: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

# BLTCY model names
BLTCY_AUTOTHINK_MODEL: str = "qwen2.5-vl-3b-instruct"
BLTCY_EMBED_MODEL: str = "text-embedding-v4"
BLTCY_CHAT_MODEL: str = "qwen3-vl-32b-instruct"
BLTCY_IMAGE_DESC_MODEL: str = "qwen2.5-vl-7b-instruct"
BLTCY_SYNTHESIS_MODEL: str = "qwen3-32b"

# SiliconFlow rerank API (BAAI/bge-reranker-v2-m3)：请设置环境变量 SILICONFLOW_API_KEY，勿提交真实密钥
SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_RERANK_BASE: str = "https://api.siliconflow.cn/v1"
SILICONFLOW_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"


# ---- Project setup (same as mmtest.py) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("LIGHYRAG_FORCE_LOCAL", "1")

from scripts.run_qa_eval import run_evaluation, parse_questions_file


# ==== Dataset hyperparameters for MMLongBench (same as mmtest.py) ====
# valid keys:
#   acad1, acad2, admin, broch, fin, guid1, guid2, res1, res2, res3, tut1, tut2
DEFAULT_DATASET = "tut1"

DATASETS: Dict[str, Dict[str, Path]] = {
    "acad1": {
        "raw_root": Path("data/mineru_results/mmlongbench/Acad1"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Acad1"),
        "output_dir": Path("outputs/mmlongbench_acad1_eval"),
    },
    "acad2": {
        "raw_root": Path("data/mineru_results/mmlongbench/Acad2"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Acad2"),
        "output_dir": Path("outputs/mmlongbench_acad2_eval"),
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
    "guid1": {
        "raw_root": Path("data/mineru_results/mmlongbench/Guid1"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Guid1"),
        "output_dir": Path("outputs/mmlongbench_guid1_eval"),
    },
    "guid2": {
        "raw_root": Path("data/mineru_results/mmlongbench/Guid2"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Guid2"),
        "output_dir": Path("outputs/mmlongbench_guid2_eval"),
    },
    "res1": {
        "raw_root": Path("data/mineru_results/mmlongbench/Res1"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Res1"),
        "output_dir": Path("outputs/mmlongbench_res1_eval"),
    },
    "res2": {
        "raw_root": Path("data/mineru_results/mmlongbench/Res2"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Res2"),
        "output_dir": Path("outputs/mmlongbench_res2_eval"),
    },
    "res3": {
        "raw_root": Path("data/mineru_results/mmlongbench/Res3"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Res3"),
        "output_dir": Path("outputs/mmlongbench_res3_eval"),
    },
    "tut1": {
        "raw_root": Path("data/mineru_results/mmlongbench/Tut1"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Tut1"),
        "output_dir": Path("outputs/mmlongbench_tut1_eval"),
    },
    "tut2": {
        "raw_root": Path("data/mineru_results/mmlongbench/Tut2"),
        "mineru_root": Path("data/mineru_results/mmlongbench/Tut2"),
        "output_dir": Path("outputs/mmlongbench_tut2_eval"),
    },
}


def iter_question_files(raw_root: Path) -> Iterable[Path]:
    """
    Recursively yield all .json / .jsonl question files under raw_root.
    Non-QA files will be skipped by parse_questions_file.
    """
    for file in sorted(raw_root.rglob("*")):
        if file.suffix.lower() in {".json", ".jsonl"}:
            yield file


def build_api_config() -> dict:
    """
    Configure BLTCY and SiliconFlow endpoints and return api_config for run_qa_eval.
    All keys are defined as hyperparameters at the top of this file (no env needed
    from the outside).
    """
    if not BLTCY_API_KEY or BLTCY_API_KEY.startswith("YOUR_"):
        raise SystemExit(
            "Please set DASHSCOPE_API_KEY or BLTCY_API_KEY (env or .env). See scripts/mmtest_bai.py."
        )
    if not SILICONFLOW_API_KEY or SILICONFLOW_API_KEY.startswith("YOUR_"):
        raise SystemExit(
            "Please set SILICONFLOW_API_KEY (env or .env). See scripts/mmtest_bai.py."
        )

    # Generic LLM binding (used by build_rag_instance / openai_complete_if_cache)
    os.environ["LLM_BINDING_API_KEY"] = BLTCY_API_KEY
    os.environ["LLM_BINDING_HOST"] = BLTCY_CHAT_HOST

    # BLTCY main config (HyperGraphRAG + bltcy_adapter)
    os.environ["BLTCY_API_KEY"] = BLTCY_API_KEY
    os.environ["BLTCY_API_BASE"] = BLTCY_CHAT_COMPLETIONS_BASE
    os.environ["BLTCY_MODEL"] = BLTCY_CHAT_MODEL

    # Auto-think (complexity classification / question decomposition)
    os.environ["AUTO_THINK_API_BASE"] = BLTCY_CHAT_COMPLETIONS_BASE
    os.environ["AUTO_THINK_API_KEY"] = BLTCY_API_KEY
    os.environ["AUTO_THINK_MODEL"] = BLTCY_AUTOTHINK_MODEL

    # Embedding (Qwen3-Embedding-8B)
    os.environ["EMBEDDING_MODEL"] = BLTCY_EMBED_MODEL
    os.environ["QWEN_EMBEDDING_API_KEY"] = BLTCY_API_KEY
    os.environ["QWEN_EMBEDDING_BASE"] = BLTCY_EMBED_HOST

    # Evaluation model (same as generator: qwen3-32b)
    os.environ["EVAL_MODEL"] = BLTCY_CHAT_MODEL

    # LightRAG embedding concurrency: small batch size and limited async calls
    # to reduce cases where remote API returns fewer vectors than inputs.
    os.environ["EMBEDDING_BATCH_NUM"] = "1"
    os.environ["EMBEDDING_FUNC_MAX_ASYNC"] = "1"
    os.environ["MAX_ASYNC"] = str(8)

    # Rerank is configured explicitly via api_config (no env variables)
    return {
        "api_key": BLTCY_API_KEY,
        "chat_host": BLTCY_CHAT_HOST,
        "embed_host": BLTCY_EMBED_HOST,
        "embed_model": BLTCY_EMBED_MODEL,
        "model": BLTCY_CHAT_MODEL,
        "image_desc_model": BLTCY_IMAGE_DESC_MODEL,
        "synthesis_model": BLTCY_SYNTHESIS_MODEL,
        "modal_max_parallel": 8,
        "llm_max_async": 16,
        "chat_api_base": BLTCY_CHAT_COMPLETIONS_BASE,
        # SiliconFlow rerank configuration (matches your requests example)
        "rerank_host": SILICONFLOW_RERANK_BASE,
        "rerank_api_key": SILICONFLOW_API_KEY,
        "rerank_model": SILICONFLOW_RERANK_MODEL,
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
        description="Batch eval for MMLongBench categories using BLTCY cloud endpoints.",
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

    synth_models = ["qwen3-32b"]

    async def run_all() -> None:
        for synth in synth_models:
            api_cfg = dict(api_config)
            api_cfg["synthesis_model"] = synth
            base_out = cfg["output_dir"]
            base_out.mkdir(parents=True, exist_ok=True)
            detail_log = base_out / "detail_log.json"
            manifest: List[Dict[str, str]] = []

            for q_file in iter_question_files(cfg["raw_root"]):
                try:
                    _ = parse_questions_file(q_file)
                except Exception as exc:  # noqa: PERF203
                    print(f"[SKIP] {q_file}: failed to parse questions ({exc})")
                    continue

                folder_name = q_file.parent.name
                folder_output_dir = base_out / folder_name
                folder_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = folder_output_dir / f"{q_file.stem}_results.json"
                print(f"\n===== Evaluating {q_file} (synthesis={synth}) =====")
                try:
                    await run_evaluation(
                        q_file,
                        cfg["mineru_root"],
                        output_path=output_path,
                        detail_log_path=detail_log,
                        folder_name=folder_name,
                        api_config=api_cfg,
                        enable_multi_step=args.multi_step,
                        max_sub_questions=args.max_sub_questions,
                        max_parallel_subqueries=args.max_sub_parallel,
                        show_plan=args.show_plan,
                    )
                except Exception as exc:  # noqa: PERF203
                    # Fatal error for this file; log and continue with next file
                    print(f"[ERROR] Evaluation crashed for {q_file}: {exc}")
                    continue

                manifest.append(collect_manifest_entry(q_file, output_path))

            manifest_path = base_out / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nManifest written to {manifest_path}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
