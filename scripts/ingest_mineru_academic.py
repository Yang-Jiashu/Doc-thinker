
"""
Ingest MinerU content_list outputs into RAG-Anything.

This script walks through a parsed MinerU directory (e.g. data/mineru_output/aca_prepared),
loads each `<doc_id>_content_list.json`, fixes image paths, and inserts the content into the
current LightRAG storage using the project’s BLTCY configuration.
"""
#读取环境变量apikey等，并从mineru的输出文件content_list.json文件中把内容写入raganything存储
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple

# Ensure we import the local project package even if a pip version is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag.llm.openai import openai_complete_if_cache  # type: ignore  # noqa: E402
from lightrag.utils import EmbeddingFunc  # type: ignore  # noqa: E402

from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402
from raganything.auto_thinking.vlm_client import VLMClient  # noqa: E402
from raganything.utils import create_bltcy_rerank_func  # noqa: E402
from raganything.hypergraph.llm import qwen3_embedding  # noqa: E402


def build_rag_instance(
    working_dir: str | None = None,
    api_config: dict | None = None,
) -> RAGAnything:
    """Create a RAGAnything instance wired to BLTCY endpoints."""
    api_cfg = api_config or {}

    api_key = api_cfg.get("api_key") or os.environ.get("LLM_BINDING_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable LLM_BINDING_API_KEY is required.")

    chat_host = api_cfg.get("chat_host") or os.getenv("LLM_BINDING_HOST", "https://api.bltcy.ai/v1")
    embed_host = api_cfg.get("embed_host") or os.getenv("EMBEDDING_BINDING_HOST", chat_host)
    rerank_host = api_cfg.get("rerank_host") or os.getenv("RERANK_HOST", embed_host)
    model_name = api_cfg.get("model") or os.getenv("BLTCY_MODEL", "gpt-4o-mini")
    chat_api_base = (
        api_cfg.get("chat_api_base")
        or os.getenv("BLTCY_API_BASE")
        or f"{chat_host.rstrip('/')}/chat/completions"
    )

    # embedding_func = EmbeddingFunc(
    #     embedding_dim=3072,
    #     max_token_size=8192,
    #     func=lambda texts: openai_embed(
    #         texts,
    #         model="text-embedding-3-large",
    #         api_key=api_key,
    #         base_url=embed_host,
    #     ),
    # )
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: qwen3_embedding(
            texts,
            api_key=os.getenv("QWEN_EMBEDDING_API_KEY") or api_key,
            base_url=os.getenv("QWEN_EMBEDDING_BASE") or embed_host,
        ),
    )

    # Rerank configuration: prefer values from api_config, then fall back to env, then main api_key
    rerank_api_key = api_cfg.get("rerank_api_key") or os.getenv("RERANK_API_KEY") or api_key
    rerank_model = api_cfg.get("rerank_model") or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_func = create_bltcy_rerank_func(
        api_key=rerank_api_key,
        base_url=rerank_host,
        model_name=rerank_model,
    )

    vlm_client = VLMClient(
        api_key=api_key,
        api_base=chat_host,
        model=model_name,
    )

    async def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        max_tokens = kwargs.get("max_tokens", 350)
        temperature = kwargs.get("temperature", 0.2)

        if messages:
            return await vlm_client.generate(
                "",
                extra_messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        images_list = []
        if image_data:
            if isinstance(image_data, (list, tuple)):
                images_list = list(image_data)
            else:
                images_list = [image_data]

        return await vlm_client.generate(
            prompt or "",
            images=images_list or None,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    config_kwargs = {}
    if working_dir is not None:
        config_kwargs["working_dir"] = working_dir
    config_kwargs.setdefault("bltcy_api_key", api_key)
    config_kwargs.setdefault("bltcy_api_base", chat_api_base)
    config_kwargs.setdefault("bltcy_model", model_name)

    return RAGAnything(
        config=RAGAnythingConfig(**config_kwargs),
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=chat_host,
            extra_body={"enable_thinking": False},
            timeout=360,
            **kwargs,
        ),
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            **({"rerank_model_func": rerank_func} if rerank_func else {}),
            "max_parallel_insert": int(api_cfg.get("modal_max_parallel", 4)),
            "llm_model_max_async": int(api_cfg.get("llm_max_async", 8)),
        },
    )


def iter_content_jsons(base_dir: Path) -> Iterable[Path]:
    """Yield all MinerU content_list.json files under base_dir."""
    yield from sorted(base_dir.glob("**/*_content_list.json"))


def load_content_list(json_path: Path) -> list[dict[str, Any]]:
    """Load content_list from the MinerU JSON output."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Some dumps wrap the list inside {"content_list": [...]}
    content_list = data.get("content_list") if isinstance(data, dict) else None
    if content_list is None:
        if isinstance(data, list):
            content_list = data
        else:
            raise ValueError(f"Unrecognised JSON format in {json_path}")

    # Normalise image paths to absolute form.
    base_dir = json_path.parent
    for block in content_list:
        if isinstance(block, dict) and "img_path" in block:
            img_path = Path(block["img_path"])
            if not img_path.is_absolute():
                img_path = (base_dir / img_path).resolve()
            block["img_path"] = img_path.as_posix()

    return content_list


def find_reference_pdf(json_path: Path, doc_id: str) -> str:
    """Try to locate a reference PDF near the content_list file."""
    candidates = [
        json_path.parent / f"{doc_id}.pdf",
        json_path.parent / f"{doc_id}_origin.pdf",
        json_path.parent.parent / f"{doc_id}.pdf",
        json_path.parent.parent / f"{doc_id}_origin.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return doc_id  # Fallback to doc_id string


async def ingest_document(
    rag: RAGAnything,
    json_path: Path,
    dry_run: bool = False,
) -> None:
    """Insert a single document's content list into RAGAnything."""
    content_list = load_content_list(json_path)
    doc_id = json_path.stem.replace("_content_list", "")
    reference_pdf = find_reference_pdf(json_path, doc_id)

    if dry_run:
        print(f"[DRY-RUN] Would ingest {doc_id} ({len(content_list)} blocks) -> {reference_pdf}")
        return

    await rag.insert_content_list(
        content_list=content_list,
        file_path=reference_pdf,
        doc_id=doc_id,
        display_stats=False,
    )
    print(f"Ingested {doc_id} ({len(content_list)} blocks)")


def _doc_id_variants(doc_id: str) -> List[str]:
    """
    Generate possible MinerU doc_id variants for lookup.

    MinerU often names files like:
      <uuid>_content_list.json
      <uuid>_origin.pdf

    While our QA pipeline may use either "<uuid>" or "<uuid>_origin" as doc_id.
    This helper returns a de-duplicated list of candidates to try when locating
    the content_list.json files.
    """
    variants: List[str] = [doc_id]
    # Common MinerU pattern: strip trailing "_origin" if present.
    if doc_id.endswith("_origin"):
        variants.append(doc_id[: -len("_origin")])
    # Clean up whitespace and remove empties.
    variants = [v.strip() for v in variants if v.strip()]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def find_content_list_for_doc(base_dir: Path, doc_id: str) -> Path:
    """Locate the MinerU content_list.json file for a specific document ID."""
    last_error: FileNotFoundError | None = None
    for candidate_id in _doc_id_variants(doc_id):
        pattern = f"**/{candidate_id}_content_list.json"
        matches = list(base_dir.glob(pattern))
        if matches:
            if len(matches) > 1:
                print(
                    f"Warning: multiple content_list.json files found for {candidate_id}. "
                    f"Using the first match: {matches[0]}"
                )
            return matches[0]
        last_error = FileNotFoundError(
            f"No MinerU content_list.json found for document ID '{candidate_id}' under {base_dir}"
        )

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(
        f"No MinerU content_list.json found for document ID '{doc_id}' under {base_dir}"
    )


def gather_content_for_doc(
    base_dir: Path,
    doc_id: str,
) -> Tuple[List[dict], str]:
    """
    Return combined content list and reference PDF for a document.

    If the exact doc_id content_list exists, use it directly. Otherwise look for
    chunked variants like `<doc_id>_partX_content_list.json` and merge them.
    """
    try:
        main_path = find_content_list_for_doc(base_dir, doc_id)
    except FileNotFoundError:
        # If a single content_list is not found, look for chunked variants.
        part_paths: List[Path] = []
        for candidate_id in _doc_id_variants(doc_id):
            part_pattern = f"**/{candidate_id}_part*_content_list.json"
            part_paths.extend(base_dir.glob(part_pattern))

        # De-duplicate and sort for stable merge order.
        unique_parts = sorted({p for p in part_paths})
        if not unique_parts:
            raise FileNotFoundError(
                f"No MinerU content_list.json or part files found for document ID '{doc_id}' under {base_dir}"
            )

        combined: List[dict] = []
        reference_pdf: str | None = None
        for idx, part_path in enumerate(unique_parts):
            part_content = load_content_list(part_path)
            combined.extend(part_content)
            if reference_pdf is None:
                part_doc_id = part_path.stem.replace("_content_list", "")
                reference_pdf = find_reference_pdf(part_path, part_doc_id)
            print(
                f"[Merge] Included {part_path.name} ({len(part_content)} blocks) for {doc_id}"
            )

        return combined, reference_pdf or doc_id

    content_list = load_content_list(main_path)
    reference_pdf = find_reference_pdf(main_path, doc_id)
    return content_list, reference_pdf


async def ingest_document_by_id(
    rag: RAGAnything,
    doc_id: str,
    base_dir: Path,
    dry_run: bool = False,
) -> None:
    """Ingest a single document identified by doc_id."""
    json_path = find_content_list_for_doc(base_dir, doc_id)
    await ingest_document(rag, json_path, dry_run=dry_run)


async def run_ingestion(base_dir: Path, dry_run: bool = False) -> None:
    """Main ingestion routine."""
    rag = build_rag_instance()
    await rag._ensure_lightrag_initialized()

    json_files = list(iter_content_jsons(base_dir))
    if not json_files:
        print(f"No MinerU output folders found under {base_dir}")
        return

    for json_path in json_files:
        try:
            await ingest_document(rag, json_path, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to ingest {json_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest MinerU content_list outputs into RAG-Anything.",
    )
    parser.add_argument(
        "--root",
        default="data/mineru_output/aca_prepared",
        help="Root directory containing MinerU outputs (default: data/mineru_output/aca_prepared)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List documents without inserting into RAGAnything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.root).resolve()
    if not base_dir.exists():
        raise SystemExit(f"MinerU output directory not found: {base_dir}")

    asyncio.run(run_ingestion(base_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
