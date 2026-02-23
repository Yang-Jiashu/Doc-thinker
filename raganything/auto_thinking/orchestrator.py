"""Hybrid orchestrator that routes between LightRAG and HyperGraphRAG."""
#auto-think的调节中枢，负责调用所有子文件，判断问题的复杂度，决定走raganything，或者hypergraphrag流程
#决定何时同步chunk,是否拆分子任务，并选择最终答案等内容。
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from ..hypergraph.hypergraphrag import HyperGraphRAG, QueryParam
from ..hypergraph.operate import chunking_by_token_size
from ..hypergraph.schemas import StructuredChunk
from ..hypergraph.utils import compute_mdhash_id
from .answer_aggregator import AnswerAggregator, RetrievalAnswer
from .chunk_buffer import ChunkBuffer
from .classifier import ComplexityClassifier, ComplexityVote
from .decomposer import QuestionDecomposer, QuestionPlan, SubQuestion, SubQuestionAnswer
from .vlm_client import VLMClient
from .prompts import RETRIEVAL_STRATEGY_PROMPT, LOOKUP_ROUTER_PROMPT
from .page_lookup import load_content_list as load_pages, parse_page_range, find_text_by_page
from .image_lookup import (
    load_content_list as load_images,
    find_image_by_position,
    parse_page as parse_image_page,
    parse_position,
)

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from ..raganything import RAGAnything
else:
    RAGAnything = Any


class HybridRAGOrchestrator:
    """Coordinate ingestion and querying between RAGAnything and HyperGraphRAG."""

    def __init__(
        self,
        rag_system: RAGAnything,
        hyper_system: HyperGraphRAG,
        *,
        classifier: ComplexityClassifier,
        vlm_client: VLMClient,
        sync_mode: str = "lazy",
        chunk_token_size: int = 1200,
        chunk_overlap: int = 100,
        decomposer: Optional[QuestionDecomposer] = None,
        enable_multi_step: bool = False,
        max_parallel_subqueries: int = 3,
    ) -> None:
        self.rag = rag_system
        self.hyper = hyper_system
        self.classifier = classifier
        self.vlm = vlm_client
        self.sync_mode = sync_mode
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap = chunk_overlap
        self.decomposer = decomposer
        self.enable_multi_step = enable_multi_step and decomposer is not None
        self.max_parallel_subqueries = max(1, max_parallel_subqueries)
        self._retrieval_prompt = RETRIEVAL_STRATEGY_PROMPT

        rag_config = getattr(self.rag, "config", None)
        self.hyper_enabled = bool(
            getattr(rag_config, "enable_hyper_entity_extraction", True)
        )

        self.buffer = ChunkBuffer()
        self.aggregator = AnswerAggregator()

        # Register callback so processor can push parsed content.
        if self.hyper_enabled:
            self.rag.hyper_chunk_sink = self._collect_hyper_chunks  # type: ignore[attr-defined]
        else:
            self.rag.hyper_chunk_sink = None  # type: ignore[attr-defined]

        rag_twi = getattr(self.rag, "twi_runner", None)
        hyper_twi = getattr(self.hyper, "twi_runner", None) if self.hyper else None
        if hyper_twi is None and rag_twi is not None and self.hyper:
            self.hyper.twi_runner = rag_twi
        elif rag_twi is None and hyper_twi is not None and self.rag:
            self.rag.twi_runner = hyper_twi

    async def ingest(self, file_path: str, **kwargs: Any) -> None:
        """Process a document and optionally sync chunks to HyperGraphRAG."""
        await self.rag.process_document_complete(file_path=file_path, **kwargs)

        if self.sync_mode == "eager":
            pending = self.buffer.get_pending()
            await asyncio.gather(*(self._sync_doc(doc.doc_id) for doc in pending))

    async def ensure_synced(self) -> None:
        """Synchronise all pending documents into HyperGraphRAG."""
        if not self.hyper_enabled:
            return
        pending = self.buffer.get_pending()
        if not pending:
            return
        await asyncio.gather(*(self._sync_doc(doc.doc_id) for doc in pending))

    async def query(
        self,
        query: str,
        *,
        rag_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        content_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route query execution based on complexity."""
        lookup = await self._maybe_lookup_page_or_image(
            query,
            doc_id=doc_id or (rag_kwargs or {}).get("doc_id"),
            content_root=content_root,
        )
        work_query = query
        lookup_details = None
        lookup_blocks: List[Dict[str, Any]] = []
        if lookup:
            supplement = lookup.get("text") or ""
            if supplement.strip():
                work_query = f"[Supplement from content_list]\n{supplement.strip()}\n\n[Original question]\n{query}"
            lookup_details = lookup.get("details")
            lookup_blocks = lookup.get("blocks") or []
            if lookup_blocks and (doc_id or (rag_kwargs or {}).get("doc_id")):
                await self._ingest_lookup_blocks_to_lightrag(
                    doc_id or (rag_kwargs or {}).get("doc_id"),
                    lookup_blocks,
                )
        if self.enable_multi_step and self.decomposer:
            multi_result = await self._run_multi_step(
                work_query,
                rag_kwargs=rag_kwargs,
                hyper_kwargs=hyper_kwargs,
            )
            if multi_result is not None:
                if lookup_details:
                    details = multi_result.get("details") or {}
                    details["lookup"] = lookup_details
                    multi_result["details"] = details
                return multi_result
        vote = await self.classifier.assess(work_query)
        overrides = await self._decide_retrieval_overrides(work_query)
        return await self._answer_single(
            work_query,
            rag_kwargs=rag_kwargs,
            hyper_kwargs=hyper_kwargs,
            vote=vote,
            retrieval_overrides=overrides,
            extra_details=lookup_details,
        )

    async def _run_multi_step(
        self,
        query: str,
        *,
        rag_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.decomposer:
            return None
        plan = await self.decomposer.build_plan(query)
        sub_answers = await self._run_sub_questions(
            plan,
            rag_kwargs=rag_kwargs,
            hyper_kwargs=hyper_kwargs,
        )
        synthesis = await self.decomposer.synthesize_final_answer(query, plan, sub_answers)
        final_answer = str(synthesis.get("final_answer") or "").strip()
        if not final_answer:
            final_answer = "\n".join(
                f"{ans.sub_id}: {ans.answer or '[no answer obtained]'}" for ans in sub_answers
            )
        details: Dict[str, Any] = {
            "sub_plan": plan.to_dict(),
            "sub_answers": [
                {
                    "id": ans.sub_id,
                    "question": ans.question,
                    "answer": ans.answer,
                    "source": ans.source,
                    "routing": ans.routing,
                    "confidence": ans.confidence,
                    "error": ans.error,
                    "augmented_question": ans.augmented_question,
                    "context": ans.context,
                    "reasoning": ans.reasoning,
                    "retrieval_strategy": ans.retrieval_strategy,
                }
                for ans in sub_answers
            ],
            "final_synthesis": synthesis,
        }
        return {
            "answer": final_answer,
            "source": "multi-step",
            "routing": "multi-step",
            "confidence": float(synthesis.get("confidence", 0.6)),
            "details": details,
        }

    async def _answer_single(
        self,
        query: str,
        *,
        rag_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
        vote: Optional[ComplexityVote] = None,
        retrieval_overrides: Optional[Dict[str, str]] = None,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        vote = vote or await self.classifier.assess(query)
        overrides = retrieval_overrides or {}

        async def _run_rag() -> RetrievalAnswer:
            effective_kwargs = dict(overrides)
            if rag_kwargs:
                effective_kwargs.update(rag_kwargs)
            response = await self.rag.aquery(query, mode="mix", **effective_kwargs)
            return RetrievalAnswer(
                content=response,
                source="raganything",
                meta={"confidence": max(vote.confidence, 0.5)},
            )

        async def _run_hyper() -> RetrievalAnswer:
            hyper_params = dict(hyper_kwargs or {})
            for key, value in overrides.items():
                hyper_params.setdefault(key, value)
            if "mode" not in hyper_params:
                hyper_params["mode"] = "mix"
            response = await self.hyper.aquery(query, param=QueryParam(**hyper_params))
            return RetrievalAnswer(
                content=response,
                source="hypergraph",
                meta={"confidence": max(vote.confidence, 0.6)},
            )

        rag_task = asyncio.create_task(_run_rag())
        hyper_task: Optional[asyncio.Task[RetrievalAnswer]] = None

        if vote.use_hyper and self.hyper_enabled:
            await self.ensure_synced()
            hyper_task = asyncio.create_task(_run_hyper())

        rag_answer: Optional[RetrievalAnswer] = None
        hyper_answer: Optional[RetrievalAnswer] = None

        if hyper_task:
            rag_answer, hyper_answer = await asyncio.gather(rag_task, hyper_task)
        else:
            rag_answer = await rag_task

        best: Optional[RetrievalAnswer]
        if rag_answer and hyper_answer:
            best = self.aggregator.select(rag_answer, hyper_answer)
        else:
            best = rag_answer or hyper_answer

        evidence: Optional[Dict[str, Any]] = None
        if best:
            if (
                best.source == "raganything"
                and hasattr(self.rag, "get_last_query_evidence")
                and callable(self.rag.get_last_query_evidence)
            ):
                evidence = self.rag.get_last_query_evidence()
            elif (
                best.source == "hypergraph"
                and hasattr(self.hyper, "get_last_query_evidence")
                and callable(self.hyper.get_last_query_evidence)
            ):
                evidence = self.hyper.get_last_query_evidence()

        return {
            "answer": best.content if best else "",
            "source": best.source if best else "undefined",
            "routing": vote.complexity,
            "confidence": best.meta.get("confidence", 0.5) if best else 0.0,
            "details": {
                **vote.features,
                **({"lookup": extra_details} if extra_details else {}),
            },
            "evidence": evidence,
            "retrieval_strategy": overrides,
        }

    async def _maybe_lookup_page_or_image(
        self,
        query: str,
        *,
        doc_id: Optional[str],
        content_root: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Optional supplement from MinerU content_list when page/image is explicit."""
        base_root = content_root or os.getenv("MINERU_ROOT")
        if not base_root or not doc_id or not self.vlm:
            return None
        try:
            raw = await self.vlm.generate(
                LOOKUP_ROUTER_PROMPT.format(query=query),
                system_prompt="You decide whether to call page/image lookup functions.",
                max_tokens=200,
            )
            data = self._safe_json(raw)
            if not isinstance(data, dict):
                return None
        except Exception:
            return None

        action = str(data.get("action") or "").lower()
        if action not in {"page_lookup", "image_position_lookup"}:
            return None

        page_field = str(data.get("page") or "").strip()
        position_field = str(data.get("position") or "").strip() or None

        def _parse_page_str(val: str) -> Optional[Tuple[int, int]]:
            m = re.search(r"(\d+)(?:\s*[-~到至]\s*(\d+))?", val)
            if not m:
                return None
            start = int(m.group(1))
            end = int(m.group(2) or start)
            return (start, end) if start <= end else (end, start)

        if action == "page_lookup":
            range_from_router = _parse_page_str(page_field) if page_field else None
            range_from_query = parse_page_range(query)
            start_end = range_from_router or range_from_query
            if not start_end:
                return None
            contents = load_pages(base_root, doc_id)
            if not contents:
                return None
            snippet = find_text_by_page(contents, start_end[0], start_end[1])
            if not snippet:
                return None
            text_body = f"[DOC:{doc_id}] [PAGE {start_end[0]}-{start_end[1]}] {snippet}"
            chunk_id = compute_mdhash_id(text_body, prefix="page-lookup-")
            structured = StructuredChunk(
                text=text_body,
                metadata={
                    "chunk_id": chunk_id,
                    "type": "text",
                    "doc_id": doc_id,
                    "page_range": start_end,
                    "source": "page-lookup",
                },
            )
            self.buffer.append_structured(doc_id, chunk=structured)
            await self.ensure_synced()
            return {
                "text": f"[Page lookup {start_end[0]}-{start_end[1]}]\n{snippet}",
                "details": {
                    "lookup_source": "page-lookup",
                    "doc_id": doc_id,
                    "content_root": base_root,
                    "page_range": start_end,
                    "action": action,
                },
                "blocks": [
                    {
                        "type": "text",
                        "text": snippet,
                        "page_idx": start_end[0] - 1,
                        "doc_id": doc_id,
                    }
                ],
            }

        if action == "image_position_lookup":
            page_num: Optional[int] = None
            if page_field:
                m = re.search(r"(\d+)", page_field)
                if m:
                    page_num = int(m.group(1))
            if page_num is None:
                page_num = parse_image_page(query)
            pos = position_field or parse_position(query) or None
            contents = load_images(base_root, doc_id)
            if not contents:
                return None
            image = find_image_by_position(contents, page_num, pos)
            if not image:
                return None
            if isinstance(image, list):
                items = []
                blocks = []
                for img in image:
                    pg_idx = img.get("page_idx")
                    bbox = img.get("bbox")
                    caption = img.get("caption") or img.get("description") or ""
                    img_path = img.get("img_path") or img.get("image_path") or img.get("path") or ""
                    text_body = f"[DOC:{doc_id}] [IMAGE] page={pg_idx + 1 if pg_idx is not None else ''} path={img_path} bbox={bbox} caption={caption}"
                    chunk_id = compute_mdhash_id(text_body, prefix="img-lookup-")
                    structured = StructuredChunk(
                        text=text_body,
                        metadata={
                            "chunk_id": chunk_id,
                            "type": "image",
                            "doc_id": doc_id,
                            "source_path": img_path,
                            "image_path": img_path,
                            "caption": caption or None,
                            "bbox": bbox,
                            "page_idx": pg_idx,
                            "lookup_source": "image-lookup",
                        },
                    )
                    self.buffer.append_structured(doc_id, chunk=structured)
                    items.append(
                        {
                            "page": pg_idx + 1 if pg_idx is not None else None,
                            "bbox": bbox,
                            "caption": caption,
                            "image_path": img_path,
                        }
                    )
                    blocks.append(
                        {
                            "type": "image",
                            "caption": caption,
                            "img_path": img_path,
                            "bbox": bbox,
                            "page_idx": pg_idx,
                            "doc_id": doc_id,
                        }
                    )
                await self.ensure_synced()
                answer_lines = []
                for it in items:
                    line_parts = []
                    if it["page"] is not None:
                        line_parts.append(f"Page {it['page']}")
                    if it["bbox"]:
                        line_parts.append(f"bbox:{it['bbox']}")
                    if it["image_path"]:
                        line_parts.append(f"path:{it['image_path']}")
                    if it["caption"]:
                        line_parts.append(f"caption:{it['caption']}")
                    answer_lines.append(" | ".join(line_parts))
                answer_text = "[Image lookup]" + ("\n" + "\n".join(answer_lines) if answer_lines else "")
                return {
                    "details": {
                        "lookup_source": "image-lookup",
                        "doc_id": doc_id,
                        "content_root": base_root,
                        "page_idx": [img.get("page_idx") for img in image],
                        "position": pos,
                        "bbox": [img.get("bbox") for img in image],
                        "image_path": [img.get("img_path") or img.get("image_path") or img.get("path") for img in image],
                        "action": action,
                        "count": len(image),
                    },
                    "text": answer_text,
                    "blocks": blocks,
                }

            pg_idx = image.get("page_idx")
            bbox = image.get("bbox")
            caption = image.get("caption") or image.get("description") or ""
            img_path = image.get("img_path") or image.get("image_path") or image.get("path") or ""
            text_body = f"[DOC:{doc_id}] [IMAGE] page={pg_idx + 1 if pg_idx is not None else ''} path={img_path} bbox={bbox} caption={caption}"
            chunk_id = compute_mdhash_id(text_body, prefix="img-lookup-")
            structured = StructuredChunk(
                text=text_body,
                metadata={
                    "chunk_id": chunk_id,
                    "type": "image",
                    "doc_id": doc_id,
                    "source_path": img_path,
                    "image_path": img_path,
                    "caption": caption or None,
                    "bbox": bbox,
                    "page_idx": pg_idx,
                    "lookup_source": "image-lookup",
                },
            )
            self.buffer.append_structured(doc_id, chunk=structured)
            await self.ensure_synced()
            answer_parts = []
            if pg_idx is not None:
                answer_parts.append(f"Page {pg_idx + 1}")
            if pos:
                answer_parts.append(f"{pos} image")
            if caption:
                answer_parts.append(f"caption: {caption}")
            if bbox:
                answer_parts.append(f"bbox: {bbox}")
            if img_path:
                answer_parts.append(f"path: {img_path}")
            answer_text = "; ".join(answer_parts)
            return {
                "text": "[Image lookup] " + answer_text,
                "details": {
                    "lookup_source": "image-lookup",
                    "doc_id": doc_id,
                    "content_root": base_root,
                    "page_idx": pg_idx,
                    "position": pos,
                    "bbox": bbox,
                    "image_path": img_path,
                    "action": action,
                },
                "blocks": [
                    {
                        "type": "image",
                        "caption": caption,
                        "img_path": img_path,
                        "bbox": bbox,
                        "page_idx": pg_idx,
                        "doc_id": doc_id,
                    }
                ],
            }
        return None

    async def _ingest_lookup_blocks_to_lightrag(
        self,
        doc_id: str,
        blocks: Sequence[Dict[str, Any]],
    ) -> None:
        """Insert lookup-derived blocks into LightRAG as a temporary doc."""
        if not blocks or not hasattr(self.rag, "insert_content_list"):
            return
        temp_doc_id = f"{doc_id}__lookup"
        try:
            await self.rag.insert_content_list(
                content_list=list(blocks),
                file_path=temp_doc_id,
                doc_id=temp_doc_id,
                display_stats=False,
            )
        except Exception:
            return

    async def _decide_retrieval_overrides(self, query: str) -> Dict[str, str]:
        if not self.vlm or not self._retrieval_prompt:
            return {}
        try:
            raw = await self.vlm.generate(
                self._retrieval_prompt.format(query=query),
                system_prompt="You are a retrieval router for a RAG system.",
                max_tokens=200,
            )
            data = self._safe_json(raw)
            if not isinstance(data, dict):
                return {}
        except Exception:
            return {}
        overrides: Dict[str, str] = {}
        for key in ("entity_retrieval", "relation_retrieval", "chunk_retrieval"):
            value = str(data.get(key, "")).lower()
            if value in {"bm25", "embedding", "hybrid"}:
                overrides[key] = value
        return overrides

    def _safe_json(self, text: str) -> Any:
        cleaned = str(text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        match_list = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match_list:
            try:
                return json.loads(match_list.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    async def _run_sub_questions(
        self,
        plan: QuestionPlan,
        *,
        rag_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[SubQuestionAnswer]:
        if plan.strategy.lower() == "dependent":
            return await self._run_serial_sub_questions(
                plan,
                rag_kwargs=rag_kwargs,
                hyper_kwargs=hyper_kwargs,
            )
        return await self._run_parallel_sub_questions(
            plan,
            rag_kwargs=rag_kwargs,
            hyper_kwargs=hyper_kwargs,
        )

    async def _run_parallel_sub_questions(
        self,
        plan: QuestionPlan,
        *,
        rag_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[SubQuestionAnswer]:
        answers: List[SubQuestionAnswer] = []
        semaphore = asyncio.Semaphore(self.max_parallel_subqueries)

        async def _execute(sub) -> None:
            async with semaphore:
                try:
                    overrides = await self._decide_retrieval_overrides(sub.question)
                    result = await self._answer_single(
                        sub.question,
                        rag_kwargs=rag_kwargs,
                        hyper_kwargs=hyper_kwargs,
                        retrieval_overrides=overrides,
                    )
                    raw_answer = str(result.get("answer", ""))
                    clean_answer, reasoning = await self._synthesize_single_step_answer(
                        sub=sub,
                        raw_answer=raw_answer,
                        source=result.get("source", "unknown"),
                        routing=result.get("routing", "unknown"),
                        confidence=float(result.get("confidence", 0.0)),
                        error=None,
                        strategy=plan.strategy,
                    )
                    evidence = result.get("evidence")
                    evidence_context = self._format_evidence_context(evidence)
                    image_paths = self._extract_image_paths(evidence)
                    answers.append(
                        SubQuestionAnswer(
                            sub_id=sub.id,
                            question=sub.question,
                            answer=clean_answer,
                            source=result.get("source", "unknown"),
                            routing=result.get("routing", "unknown"),
                            confidence=float(result.get("confidence", 0.0)),
                            error=None,
                            context=evidence_context or None,
                            reasoning=reasoning,
                            image_paths=image_paths or None,
                            retrieval_strategy=overrides or None,
                        )
                    )
                except Exception as exc:  # pragma: no cover - best effort logging
                    answers.append(
                        SubQuestionAnswer(
                            sub_id=sub.id,
                            question=sub.question,
                            answer="",
                            source="error",
                            routing="error",
                            confidence=0.0,
                            error=str(exc),
                            reasoning=None,
                        )
                    )

        await asyncio.gather(*(_execute(sub) for sub in plan.sub_questions))

        # Preserve ordering as defined in the plan.
        ordering = {sub.id: idx for idx, sub in enumerate(plan.sub_questions)}
        answers.sort(key=lambda ans: ordering.get(ans.sub_id, len(ordering)))
        return answers

    async def _run_serial_sub_questions(
        self,
        plan: QuestionPlan,
        *,
        rag_kwargs: Optional[Dict[str, Any]] = None,
        hyper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[SubQuestionAnswer]:
        answers: List[SubQuestionAnswer] = []
        for sub in plan.sub_questions:
            prior_text = ""
            augmented_question = sub.question
            if answers:
                prior_text = self._summarize_previous_answers(answers)
                augmented_question = (
                    f"{sub.question}\n\nPrior findings:\n{prior_text}\n"
                    "Use these findings to improve retrieval and reasoning."
                )
            try:
                overrides = await self._decide_retrieval_overrides(augmented_question)
                result = await self._answer_single(
                    augmented_question,
                    rag_kwargs=rag_kwargs,
                    hyper_kwargs=hyper_kwargs,
                    retrieval_overrides=overrides,
                )
                clean_answer, reasoning = await self._synthesize_single_step_answer(
                    sub=sub,
                    raw_answer=str(result.get("answer", "")),
                    source=result.get("source", "unknown"),
                    routing=result.get("routing", "unknown"),
                    confidence=float(result.get("confidence", 0.0)),
                    error=None,
                    strategy=plan.strategy,
                )
                evidence = result.get("evidence")
                evidence_context = self._format_evidence_context(evidence)
                image_paths = self._extract_image_paths(evidence)
                answers.append(
                    SubQuestionAnswer(
                        sub_id=sub.id,
                        question=sub.question,
                        answer=clean_answer,
                        source=result.get("source", "unknown"),
                        routing=result.get("routing", "unknown"),
                        confidence=float(result.get("confidence", 0.0)),
                        error=None,
                        augmented_question=augmented_question if augmented_question != sub.question else None,
                        context=evidence_context or None,
                        reasoning=reasoning,
                        image_paths=image_paths or None,
                        retrieval_strategy=overrides or None,
                    )
                )
            except Exception as exc:  # pragma: no cover - best effort logging
                answers.append(
                    SubQuestionAnswer(
                        sub_id=sub.id,
                        question=sub.question,
                        answer="",
                        source="error",
                        routing="error",
                        confidence=0.0,
                        error=str(exc),
                        augmented_question=augmented_question if augmented_question != sub.question else None,
                        context=None,
                        reasoning=None,
                        retrieval_strategy=None,
                    )
                )
        return answers

    def _format_evidence_context(self, evidence: Any) -> str:
        if not isinstance(evidence, dict):
            return ""
        parts: List[str] = []
        paths = evidence.get("image_paths") or []
        if paths:
            listed = "\n".join(str(path) for path in paths)
            parts.append(f"Image Paths:\n{listed}")
        raw_prompt = evidence.get("raw_prompt")
        if raw_prompt:
            snippet = raw_prompt if len(raw_prompt) <= 600 else f"{raw_prompt[:600]}..."
            parts.append(f"Context Snippet:\n{snippet}")
        return "\n\n".join(parts)

    def _extract_image_paths(self, evidence: Any) -> List[str]:
        if not isinstance(evidence, dict):
            return []
        raw_paths = evidence.get("image_paths") or []
        if not isinstance(raw_paths, (list, tuple)):
            return []
        cleaned = []
        for path in raw_paths:
            if isinstance(path, str):
                stripped = path.strip()
                if stripped:
                    cleaned.append(stripped)
        return cleaned

    def _summarize_previous_answers(self, answers: Sequence[SubQuestionAnswer]) -> str:
        snippets = []
        for ans in answers:
            summary = ans.answer or "[no answer]"
            if ans.context:
                context_note = ans.context if len(ans.context) < 400 else f"{ans.context[:400]}..."
                summary = f"{summary}\n  Evidence: {context_note}"
            snippets.append(f"{ans.sub_id}: {summary}")
        return "\n".join(snippets)

    async def _synthesize_single_step_answer(
        self,
        *,
        sub: SubQuestion,
        raw_answer: str,
        source: str,
        routing: str,
        confidence: float,
        error: Optional[str],
        strategy: str,
    ) -> Tuple[str, Optional[str]]:
        if not self.decomposer:
            return raw_answer, None
        temp_plan = QuestionPlan(
            original_question=sub.question,
            sub_questions=[
                SubQuestion(
                    id=sub.id,
                    question=sub.question,
                    rationale=sub.rationale,
                    depends_on=sub.depends_on,
                )
            ],
            strategy=strategy,
            notes=None,
        )
        temp_answer = SubQuestionAnswer(
            sub_id=sub.id,
            question=sub.question,
            answer=raw_answer,
            source=source,
            routing=routing,
            confidence=confidence,
            error=error,
            augmented_question=None,
            context=None,
        )
        synthesis = await self.decomposer.synthesize_step_answer(
            sub.question,
            temp_plan,
            [temp_answer],
        )
        final_text = str(synthesis.get("final_answer") or "").strip()
        reasoning = synthesis.get("reasoning")
        if isinstance(reasoning, str):
            reasoning = reasoning.strip() or None
        elif reasoning is not None:
            reasoning = str(reasoning).strip() or None
        return (final_text or raw_answer, reasoning)

    async def _collect_hyper_chunks(
        self,
        doc_id: str,
        *,
        text_content: str,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
    ) -> None:
        """Callback invoked by RAGAnything after parsing."""
        self.buffer.set_file_path(doc_id, file_path)

        for chunk in chunking_by_token_size(
            text_content,
            overlap_token_size=self.chunk_overlap,
            max_token_size=self.chunk_token_size,
            tiktoken_model=self.hyper.tiktoken_model_name,
        ):
            content = chunk["content"].strip()
            if not content:
                continue
            chunk_id = compute_mdhash_id(content, prefix="chunk-")
            structured = StructuredChunk(
                text=f"[DOC:{doc_id}] {content}",
                metadata={
                    "chunk_id": chunk_id,
                    "source_path": file_path,
                    "type": "text",
                    "doc_id": doc_id,
                    "tokens": chunk.get("tokens"),
                },
            )
            self.buffer.append_structured(doc_id, chunk=structured)

        for item in multimodal_items or []:
            structured = self._convert_multimodal_item(
                doc_id=doc_id, file_path=file_path, item=item
            )
            if structured is None:
                continue
            self.buffer.append_structured(doc_id, chunk=structured)

        if self.sync_mode == "eager":
            await self._sync_doc(doc_id)

    def _convert_multimodal_item(
        self, *, doc_id: str, file_path: str, item: Dict[str, Any]
    ) -> Optional[StructuredChunk]:
        """Convert parsed multimodal metadata into structured chunk form."""
        caption = item.get("description") or item.get("caption") or ""
        fallback_desc = json.dumps(item, ensure_ascii=False, indent=2)
        text_body = caption.strip() or fallback_desc

        image_path: Optional[str] = None
        for key in ("img_path", "image_path", "path"):
            maybe = item.get(key)
            if isinstance(maybe, str):
                image_path = maybe
                break

        chunk_source = item.get("source_path", file_path)
        chunk_id = compute_mdhash_id(
            f"{doc_id}:{text_body}:{image_path or ''}", prefix="chunk-"
        )
        metadata: Dict[str, Union[str, int, float, bool, None]] = {
            "chunk_id": chunk_id,
            "source_path": chunk_source,
            "type": item.get("type", "modal"),
            "doc_id": doc_id,
            "caption": caption or None,
            "image_path": image_path,
        }
        if "page_idx" in item:
            metadata["page_idx"] = item["page_idx"]
        if "entity_info" in item:
            metadata["entity_info"] = item["entity_info"]
        metadata["raw_item"] = item

        return StructuredChunk(
            text=f"[DOC:{doc_id}] {text_body}",
            metadata=metadata,
        )

    async def _sync_doc(self, doc_id: str) -> None:
        doc = self.buffer.get(doc_id)
        if not doc or doc.synced:
            return
        payload = [record.content for record in doc.chunks]
        if not payload:
            return
        await self.hyper.ainsert(payload)
        self.buffer.mark_synced(doc_id)
