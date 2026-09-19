"""Append-only audit artifacts for ECLRR-v4 runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


class AuditTrail:
    def __init__(self, artifact_dir: str | None, config: Any):
        self.root: Path | None = None
        if artifact_dir:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.root = Path(artifact_dir) / f"run-{stamp}"
            self.root.mkdir(parents=True, exist_ok=False)
        self.summary: dict[str, Any] = {
            "algorithm_version": "eclrr_v4",
            "config": _jsonable(config),
            "stages": {},
        }

    def record(self, name: str, value: Any) -> None:
        payload = _jsonable(value)
        self.summary["stages"][name] = payload
        self._write(f"{name}.json", payload)

    def record_llm(
        self, role: str, index: int, prompt: str, raw: Any, parsed: Any
    ) -> None:
        self._write(
            f"{role}-{index:04d}.json",
            {"prompt": prompt, "raw": raw, "parsed": _jsonable(parsed)},
        )

    def finalize(self, metrics: dict[str, Any]) -> None:
        self.summary["metrics"] = _jsonable(metrics)
        self._write("summary.json", self.summary)

    def _write(self, name: str, payload: Any) -> None:
        if self.root is None:
            return
        target = self.root / name
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
