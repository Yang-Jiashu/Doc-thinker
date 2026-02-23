"""Question decomposition and synthesis helpers."""
#负责将复杂问题拆分成子问题，以及汇总子答案作为最终回答。
from __future__ import annotations

import json
import re
import mimetypes
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .prompts import FINAL_SYNTHESIS_PROMPT, QUESTION_DECOMPOSITION_PROMPT
from .vlm_client import VLMClient
from ..utils import encode_image_to_base64, validate_image_file


@dataclass
class SubQuestion:
    """Single sub-question unit."""

    id: str
    question: str
    rationale: Optional[str] = None
    depends_on: Optional[List[str]] = None


@dataclass
class QuestionPlan:
    """Decomposition result for a user query."""

    original_question: str
    sub_questions: List[SubQuestion]
    strategy: str = "dependent"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_question": self.original_question,
            "strategy": self.strategy,
            "notes": self.notes,
            "sub_questions": [
                {
                    "id": sq.id,
                    "question": sq.question,
                    "rationale": sq.rationale,
                    "depends_on": sq.depends_on,
                }
                for sq in self.sub_questions
            ],
        }


@dataclass
class SubQuestionAnswer:
    """Holds retrieval results per sub-question."""

    sub_id: str
    question: str
    answer: str
    source: str
    routing: str
    confidence: float
    error: Optional[str] = None
    augmented_question: Optional[str] = None
    context: Optional[str] = None
    reasoning: Optional[str] = None
    image_paths: Optional[List[str]] = None
    image_descriptions: Optional[List[str]] = None
    retrieval_strategy: Optional[Dict[str, Any]] = None


class QuestionDecomposer:
    """LLM-backed sub-question planner and final answer synthesizer."""

    def __init__(
        self,
        vlm_client: VLMClient,
        *,
        max_sub_questions: int = 4,
        decomposition_temperature: float = 0.0,
        synthesis_temperature: float = 0.2,
        synthesis_client: Optional[VLMClient] = None,
        final_synthesis_client: Optional[VLMClient] = None,
        image_describer_client: Optional[VLMClient] = None,
        image_description_prompt: str = "",
        image_describe_max_parallel: int = 4,
    ) -> None:
        self.vlm_client = vlm_client
        self.synthesis_client: VLMClient = synthesis_client or vlm_client
        self.final_synthesis_client: VLMClient = final_synthesis_client or self.synthesis_client
        self.image_describer_client: Optional[VLMClient] = image_describer_client
        self.max_sub_questions = max(1, max_sub_questions)
        self.decomposition_temperature = decomposition_temperature
        self.synthesis_temperature = synthesis_temperature
        self.image_description_prompt = image_description_prompt
        self.image_describe_max_parallel = max(1, int(image_describe_max_parallel))

    async def build_plan(self, query: str) -> QuestionPlan:
        """Ask the model to decompose the query into sub-questions."""
        prompt = QUESTION_DECOMPOSITION_PROMPT.format(
            query=query.strip(),
            max_steps=self.max_sub_questions,
        )
        response = await self.vlm_client.generate(
            prompt,
            max_tokens=600,
            temperature=self.decomposition_temperature,
        )
        payload = self._safe_json(response)
        sub_questions = self._parse_sub_questions(payload)
        if not sub_questions:
            sub_questions = [
                SubQuestion(
                    id="step-1",
                    question=query.strip(),
                    rationale="Fallback: failed to decompose, using original query.",
                )
            ]
        strategy = self._parse_strategy(payload)
        notes = None
        if isinstance(payload, dict):
            notes = str(payload.get("notes") or "").strip() or None
        return QuestionPlan(
            query.strip(),
            sub_questions[: self.max_sub_questions],
            strategy=strategy,
            notes=notes,
        )

    async def synthesize_final_answer(
        self,
        query: str,
        plan: QuestionPlan,
        answers: Sequence[SubQuestionAnswer],
    ) -> Dict[str, Any]:
        """Combine answers into a final response."""
        if self.image_describer_client:
            tasks = []
            semaphore = asyncio.Semaphore(self.image_describe_max_parallel)
            async def _describe(ans: SubQuestionAnswer, p: str) -> tuple[SubQuestionAnswer, Optional[str]]:
                async with semaphore:
                    try:
                        desc_prompt = (
                            (self.image_description_prompt or "[IMAGE_DESCRIPTION_PROMPT]")
                            + "\n\n[Original query]\n"
                            + query
                        )
                        t = await self.image_describer_client.generate(
                            desc_prompt,
                            images=[p],
                            max_tokens=300,
                            temperature=0.0,
                        )
                        return ans, (t.strip() if isinstance(t, str) else None)
                    except Exception:
                        return ans, None
            for ans in answers:
                for p in (ans.image_paths or []):
                    if isinstance(p, str) and p.strip():
                        tasks.append(_describe(ans, p.strip()))
            if tasks:
                results = await asyncio.gather(*tasks)
                for a, d in results:
                    if d:
                        if a.image_descriptions is None:
                            a.image_descriptions = []
                        a.image_descriptions.append(d)
        plan_json = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
        answers_payload = [
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
                "image_paths": ans.image_paths,
                "image_descriptions": ans.image_descriptions,
            }
            for ans in answers
        ]
        answers_json = json.dumps(answers_payload, ensure_ascii=False, indent=2)
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            query=query.strip(),
            plan=plan_json,
            answers=answers_json,
        )
        response = await self.final_synthesis_client.generate(
            prompt,
            max_tokens=700,
            temperature=self.synthesis_temperature,
            extra_body={"enable_thinking": False}
            )
        result = self._safe_json(response)
        return result if isinstance(result, dict) else {}

    async def synthesize_step_answer(
        self,
        query: str,
        plan: QuestionPlan,
        answers: Sequence[SubQuestionAnswer],
    ) -> Dict[str, Any]:
        """Synthesize a single-step answer using step synthesis client."""
        plan_json = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
        answers_payload = [
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
            }
            for ans in answers
        ]
        answers_json = json.dumps(answers_payload, ensure_ascii=False, indent=2)
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            query=query.strip(),
            plan=plan_json,
            answers=answers_json,
        )
        response = await self.synthesis_client.generate(
            prompt,
            max_tokens=400,
            temperature=self.synthesis_temperature,
        )
        result = self._safe_json(response)
        return result if isinstance(result, dict) else {}

    def _parse_sub_questions(self, payload: Any) -> List[SubQuestion]:
        sub_questions: List[SubQuestion] = []
        if isinstance(payload, dict):
            items = payload.get("sub_questions")
        else:
            items = None

        if isinstance(items, list):
            for idx, entry in enumerate(items, start=1):
                question = ""
                rationale = None
                sq_id = f"step-{idx}"
                depends_on = None
                if isinstance(entry, dict):
                    question = str(entry.get("question") or "").strip()
                    rationale = str(entry.get("rationale") or "").strip() or None
                    candidate_id = str(entry.get("id") or "").strip()
                    raw_depends = entry.get("depends_on")
                    if isinstance(raw_depends, list):
                        depends = [
                            str(dep).strip()
                            for dep in raw_depends
                            if isinstance(dep, str) and dep.strip()
                        ]
                        depends_on = depends or None
                    if candidate_id:
                        sq_id = candidate_id
                elif isinstance(entry, str):
                    question = entry.strip()
                if not question:
                    continue
                if not sq_id.startswith("step-"):
                    sq_id = f"step-{idx}"
                sub_questions.append(
                    SubQuestion(
                        id=sq_id,
                        question=question,
                        rationale=rationale,
                        depends_on=depends_on,
                    )
                )
        return sub_questions

    def _parse_strategy(self, payload: Any) -> str:
        if isinstance(payload, dict):
            strategy = str(payload.get("strategy") or "").strip().lower()
            if strategy in {"dependent", "independent"}:
                return strategy
        return "dependent"

    def _safe_json(self, text: str) -> Any:
        cleaned = text.strip()
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

    def _build_final_messages(self, prompt: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, str]] = [{"type": "text", "text": prompt}]
        for idx, img_path in enumerate(image_paths, start=1):
            if not isinstance(img_path, str):
                continue
            clean_path = img_path.strip().strip('"').strip("'").strip("`").replace("\\", "/")
            if not clean_path:
                continue
            if not validate_image_file(clean_path, max_size_mb=50):
                candidate = Path(clean_path)
                fname = candidate.name
                relocated = None
                search_roots: List[Path] = []
                if candidate.parent:
                    search_roots.append(candidate.parent)
                base_root = os.getenv("MINERU_ROOT")
                if base_root and Path(base_root).exists():
                    search_roots.append(Path(base_root))
                for root in search_roots:
                    try:
                        for r, _dirs, files in os.walk(root):
                            if fname in files:
                                relocated = Path(r) / fname
                                break
                        if relocated:
                            break
                    except Exception:
                        pass
                if relocated and validate_image_file(str(relocated), max_size_mb=50):
                    clean_path = relocated.as_posix()
                else:
                    continue
            encoded = encode_image_to_base64(clean_path)
            if not encoded:
                continue
            mime_type, _ = mimetypes.guess_type(clean_path)
            if not mime_type:
                mime_type = "image/png"
            data_uri = f"data:{mime_type};base64,{encoded}"
            user_content.append(
                {
                    "type": "text",
                    "text": f"[Image {idx}: {Path(clean_path).name}]",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
        return [
            {
                "role": "user",
                "content": user_content,
            }
        ]
