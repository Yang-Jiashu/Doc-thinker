"""Integration helpers for Think-With-Image (TWI) pipeline."""
#think with image模块的封装器
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from twi.pipeline import TWIPipeline, GroundingDinoDetector  # type: ignore
except Exception:  # pragma: no cover
    TWIPipeline = None
    GroundingDinoDetector = None

logger = logging.getLogger("raganything.twi")


def _unique_existing_paths(paths: Iterable[str]) -> List[Path]:
    seen: set[Path] = set()
    ordered: List[Path] = []
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            logger.debug("TWI skipped missing image: %s", raw)
            continue
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


@dataclass
class TWIResult:
    image_path: str
    search_target: Optional[str]
    detections: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_prompt_block(self) -> str:
        """Return a human-readable string suitable for prompt injection."""
        detections_repr = ""
        if self.detections:
            detections_repr = "\n".join(
                f"- {det.get('label', 'unknown')} (confidence: {det.get('confidence', 0):.2f}, "
                f"box: {det.get('box_xyxy')})"
                for det in self.detections
            )
        summary_text = self.summary.get("answer") or self.summary.get("summary") or ""
        rationale = self.summary.get("rationale")
        if isinstance(rationale, list):
            rationale_text = "\n".join(f"- {item}" for item in rationale)
        else:
            rationale_text = str(rationale) if rationale else ""

        parts = [
            f"Image: {self.image_path}",
        ]
        if self.search_target:
            parts.append(f"Search target: {self.search_target}")
        if detections_repr:
            parts.append("Detections:\n" + detections_repr)
        if summary_text:
            parts.append(f"TWI summary: {summary_text}")
        if rationale_text:
            parts.append(f"TWI rationale:\n{rationale_text}")
        return "\n".join(parts)


class ThinkWithImageRunner:
    """Wrapper around TWIPipeline providing async-friendly helpers."""

    def __init__(
        self,
        *,
        pipeline: Optional[Any] = None,
        auto_initialize: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        self._pipeline = pipeline
        self._cache_enabled = cache_enabled
        self._cache: Dict[Tuple[str, str], TWIResult] = {}
        self._auto_initialize = auto_initialize
        self._pipelines: Dict[str, Any] = {}
        self._pipeline_cycle = None
        self._shared_detector = None

    @property
    def available(self) -> bool:
        return bool(self._pipelines) or self._pipeline is not None or (self._auto_initialize and TWIPipeline is not None)

    def configure_from_env(self) -> None:
        if self._pipeline is not None:
            return
        if TWIPipeline is None:
            logger.info("TWIPipeline is not available; skipping Think-With-Image integration.")
            return
        keys_env = os.getenv("TWI_BLTYC_API_KEYS")
        keys: List[str] = []
        if keys_env:
            keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        else:
            llm_keys = os.getenv("LLM_BINDING_API_KEYS")
            if llm_keys:
                keys = [k.strip() for k in llm_keys.split(",") if k.strip()]
        if not keys:
            single_key = (
                os.getenv("TWI_BLTYC_API_KEY")
                or os.getenv("LLM_BINDING_API_KEY")
            )
            keys = [single_key] if single_key else []
        if not keys:
            logger.info("TWI disabled because no BLTCY API key found (TWI_BLTYC_API_KEYS/TWI_BLTYC_API_KEY).")
            return
        api_base = os.getenv("TWI_BLTYC_API_BASE")
        model = os.getenv("TWI_BLTYC_MODEL")
        timeout_env = os.getenv("TWI_BLTYC_TIMEOUT")
        request_timeout = float(timeout_env) if timeout_env else None
        detector = None
        if GroundingDinoDetector is not None:
            detector = GroundingDinoDetector()
            self._shared_detector = detector
        for key in keys:
            try:
                kwargs: Dict[str, Any] = {"api_key": key}
                if api_base:
                    kwargs["api_base"] = api_base
                if model:
                    kwargs["model"] = model
                if request_timeout is not None:
                    kwargs["request_timeout"] = request_timeout
                if detector is not None:
                    kwargs["detector"] = detector
                pipeline = TWIPipeline(**kwargs)
                self._pipelines[key] = pipeline
            except Exception as exc:  # pragma: no cover - runtime dependency
                logger.warning("Failed to initialize TWI pipeline for key %s: %s", key[-6:], exc)
        if self._pipelines:
            keys_order = list(self._pipelines.keys())
            self._pipeline_cycle = cycle(keys_order)
            # retain first pipeline for backward compatibility with callers expecting _pipeline
            self._pipeline = self._pipelines[keys_order[0]]
            logger.info("Think-With-Image pipeline initialized with %d key(s).", len(self._pipelines))
        else:
            logger.warning("No TWI pipelines could be initialized; disabling TWI.")

    async def arun(self, question: str, image_paths: Sequence[str]) -> List[TWIResult]:
        """Run the full TWI workflow for the provided images."""
        if not self.available:
            return []
        if self._pipeline is None and self._auto_initialize:
            self.configure_from_env()
        if self._pipeline is None and not self._pipelines:
            return []

        paths = _unique_existing_paths(image_paths)
        if not paths:
            return []

        def _pick_pipeline():
            if self._pipelines:
                key = next(self._pipeline_cycle)
                return self._pipelines[key]
            return self._pipeline

        async def _run_single(path: Path) -> Optional[TWIResult]:
            cache_key = (question, str(path))
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]

            pipeline = _pick_pipeline()
            if pipeline is None:
                return None

            def _execute() -> Optional[TWIResult]:
                try:
                    memory = pipeline.show(question, path)
                    try:
                        pipeline.research(memory)
                    except Exception as research_exc:  # pragma: no cover
                        logger.debug("TWI research step failed: %s", research_exc)
                    summary = pipeline.tell(memory)
                    result = TWIResult(
                        image_path=str(path),
                        search_target=memory.search_target,
                        detections=[det.to_dict() for det in memory.target_locations],
                        summary=summary or {},
                    )
                    if self._cache_enabled:
                        self._cache[cache_key] = result
                    return result
                except Exception as exc:  # pragma: no cover
                    logger.warning("TWI execution failed for %s: %s", path, exc)
                    return None

            return await asyncio.to_thread(_execute)

        tasks = [_run_single(path) for path in paths]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res is not None]


__all__ = ["ThinkWithImageRunner", "TWIResult"]
