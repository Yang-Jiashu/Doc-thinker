"""LLM-backed causal extraction with a deterministic persistence contract."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from .dag import CausalDAG


class CausalExtractor:
    def __init__(
        self,
        *,
        llm_func: Callable[..., Awaitable[str]],
        dag: CausalDAG,
        min_chars: int = 200,
    ) -> None:
        self.llm_func = llm_func
        self.dag = dag
        self.min_chars = min_chars

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        source = str(text or "").strip()
        if source.startswith("```"):
            source = re.sub(r"^```(?:json)?", "", source).strip()
            source = re.sub(r"```$", "", source).strip()
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", source, re.DOTALL)
            return json.loads(match.group(0)) if match else {}

    async def extract_from_text(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if len(str(text or "")) < self.min_chars:
            return {"skipped": True, "reason": "text_too_short", "extracted": 0, "added": 0, "rejected": 0}

        prompt = (
            "Extract causal cause-effect relations as JSON with key causal_relations. "
            "Each relation should include cause, effect, mechanism, strength, and evidence.\n\n"
            f"Text:\n{text}"
        )
        raw = await self.llm_func(prompt)
        payload = self._parse_json(raw)
        relations = payload.get("causal_relations") or payload.get("relations") or []
        added = 0
        rejected = 0
        for rel in relations:
            edge = self.dag.add_edge(
                rel.get("cause") or rel.get("source") or "",
                rel.get("effect") or rel.get("target") or "",
                mechanism=rel.get("mechanism") or rel.get("relation") or "causes",
                strength=rel.get("strength") or rel.get("confidence") or 0.7,
                evidence=rel.get("evidence") or "",
                source_id=source_id or "",
            )
            if edge is None:
                rejected += 1
            else:
                added += 1
        return {"extracted": len(relations), "added": added, "rejected": rejected}
