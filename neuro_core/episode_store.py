# neuro_memory/episode_store.py
"""Episode 的持久化与向量检索（内存版 + 可插拔向量后端）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Episode


class InMemoryEpisodeStore:
    """内存存储所有 Episode；可选持久化到 JSON。"""

    def __init__(self, persist_path: Optional[str] = None):
        self._episodes: Dict[str, Episode] = {}
        self._persist_path = Path(persist_path) if persist_path else None

    def get(self, episode_id: str) -> Optional[Episode]:
        return self._episodes.get(episode_id)

    def put(self, episode: Episode) -> None:
        self._episodes[episode.episode_id] = episode

    def list_ids(self) -> List[str]:
        return list(self._episodes.keys())

    def all_episodes(self) -> Dict[str, Episode]:
        return dict(self._episodes)

    def load(self) -> int:
        if not self._persist_path or not self._persist_path.exists():
            return 0
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for item in data.get("episodes", []):
                ep = Episode.from_dict(item)
                self._episodes[ep.episode_id] = ep
            return len(self._episodes)
        except Exception:
            return 0

    def save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "episodes": [ep.to_dict() for ep in self._episodes.values()],
        }
        self._persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class EpisodeVectorStore:
    """
    用 content_embedding 做向量检索。内存版：列表存 (id, embedding)，查询时线性扫描 + 余弦相似度。
    可替换为外部向量库（如 NanoVectorDBStorage 的 namespace）。
    """

    def __init__(self):
        self._ids: List[str] = []
        self._embeddings: List[List[float]] = []

    def upsert(self, episode_id: str, embedding: List[float]) -> None:
        if episode_id in self._ids:
            idx = self._ids.index(episode_id)
            self._embeddings[idx] = embedding
        else:
            self._ids.append(episode_id)
            self._embeddings.append(embedding)

    def query(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        if not self._embeddings:
            return []
        try:
            import math
            def cos_sim(a: List[float], b: List[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a)) or 1e-8
                nb = math.sqrt(sum(x * x for x in b)) or 1e-8
                return dot / (na * nb)
            scores = [(self._ids[i], cos_sim(query_embedding, self._embeddings[i])) for i in range(len(self._ids))]
            scores.sort(key=lambda x: -x[1])
            return scores[:top_k]
        except Exception:
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {"ids": self._ids, "embeddings": self._embeddings}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeVectorStore":
        s = cls()
        s._ids = data.get("ids", [])
        s._embeddings = data.get("embeddings", [])
        return s
