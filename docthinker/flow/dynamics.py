"""Activation dynamics for long-horizon knowledge flow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


class KnowledgeFlowDynamics:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.activations: Dict[str, float] = {}
        self.last_access: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.activations = {str(k): float(v) for k, v in (data.get("activations") or {}).items()}
        self.last_access = {str(k): float(v) for k, v in (data.get("last_access") or {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"activations": self.activations, "last_access": self.last_access}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_access(self, entity: str, *, boost: float = 0.6) -> float:
        key = str(entity).strip()
        if not key:
            return 0.0
        value = min(1.0, self.activations.get(key, 0.0) + float(boost))
        self.activations[key] = value
        self.last_access[key] = time.time()
        return value

    def record_batch_access(self, entities: Iterable[str], *, boost: float = 0.35) -> None:
        for entity in entities:
            self.record_access(entity, boost=boost)

    def get_activation(self, entity: str) -> float:
        return float(self.activations.get(str(entity), 0.0))

    def get_top_activated(self, limit: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.activations.items(), key=lambda item: item[1], reverse=True)[:limit]

    def decay_all(self, *, rate: float = 0.95, prune_below: float = 0.01) -> Dict[str, int]:
        decayed = 0
        pruned = 0
        for key in list(self.activations):
            self.activations[key] *= rate
            decayed += 1
            if self.activations[key] < prune_below:
                del self.activations[key]
                self.last_access.pop(key, None)
                pruned += 1
        return {"decayed": decayed, "pruned": pruned}

    def propagate_causal(
        self,
        forward_graph: Dict[str, List[str]],
        seeds: Iterable[str],
        *,
        depth: int = 3,
        decay: float = 0.55,
    ) -> int:
        activated = 0
        frontier = [(str(seed), 1.0, 0) for seed in seeds]
        seen = set()
        while frontier:
            current, strength, current_depth = frontier.pop(0)
            if (current, current_depth) in seen or current_depth >= depth:
                continue
            seen.add((current, current_depth))
            for nxt in forward_graph.get(current, []):
                self.record_access(nxt, boost=max(0.05, strength * decay))
                activated += 1
                frontier.append((nxt, strength * decay, current_depth + 1))
        return activated
