# neuro_memory/engine.py
"""记忆引擎：新记忆写入与即时联想、巩固、类比检索的统一入口。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Episode, EdgeType
from .graph_store import MemoryGraphStore
from .episode_store import InMemoryEpisodeStore, EpisodeVectorStore
from .spreading_activation import spreading_activation, top_k_activated
from .consolidation import consolidate, build_structure_description
from .analogical_retrieval import retrieve_analogies, structure_description_from_triples


def _make_episode_id(summary: str, timestamp: float, source: str) -> str:
    raw = f"{summary[:200]}|{timestamp}|{source}"
    return "ep-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class MemoryEngine:
    """
    类人脑记忆引擎：负责 Episode 的创建、联想图更新、巩固与类比检索。
    所有 I/O（embedding、LLM、外部图）通过注入的函数完成，便于与现有 RAG 对接。
    """

    def __init__(
        self,
        *,
        embedding_func: Optional[Callable[..., Any]] = None,
        llm_func: Optional[Callable[..., Any]] = None,
        working_dir: Optional[str] = None,
    ):
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.working_dir = working_dir or "neuro_memory_data"
        self.graph = MemoryGraphStore()
        self.episode_store = InMemoryEpisodeStore(
            persist_path=str(Path(self.working_dir) / "episodes.json")
        )
        self.episode_vectors = EpisodeVectorStore()
        self._consolidation_count = 0

    def _ensure_working_dir(self) -> None:
        import os
        os.makedirs(self.working_dir, exist_ok=True)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_func or not texts:
            return []
        fn = self.embedding_func
        if isinstance(texts, str):
            texts = [texts]
        out = fn(texts) if not asyncio.iscoroutinefunction(fn) else await fn(texts)
        if isinstance(out, list) and out and not isinstance(out[0], list):
            return [out]
        return list(out) if out else []

    async def add_observation(
        self,
        *,
        summary: str,
        key_points: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
        relation_triples: Optional[List[Tuple[str, str, str]]] = None,
        raw_text_refs: Optional[List[str]] = None,
        source_type: str = "generic",
        session_id: Optional[str] = None,
        existing_insight: Optional[Any] = None,
    ) -> Optional[Episode]:
        """
        写入一次新经历并做即时联想：建 Episode、算 content/graph embedding、
        与已有 chunk/entity/episode 建边。
        existing_insight 可为 CognitiveInsight，用于补全 summary/concepts/entities。
        """
        key_points = key_points or []
        concepts = concepts or []
        entity_ids = entity_ids or []
        relation_triples = relation_triples or []
        raw_text_refs = raw_text_refs or []
        if existing_insight:
            if getattr(existing_insight, "summary", None):
                summary = existing_insight.summary or summary
            if getattr(existing_insight, "key_points", None):
                key_points = key_points or existing_insight.key_points
            if getattr(existing_insight, "concepts", None):
                concepts = concepts or existing_insight.concepts
            if getattr(existing_insight, "entities", None):
                entity_ids = entity_ids or [getattr(e, "name", str(e)) for e in existing_insight.entities[:30]]
            if getattr(existing_insight, "relations", None):
                for r in existing_insight.relations[:20]:
                    s, t = getattr(r, "source", ""), getattr(r, "target", "")
                    rel = getattr(r, "relation", "related_to")
                    if s and t:
                        relation_triples.append((s, rel, t))

        ts = time.time()
        episode_id = _make_episode_id(summary, ts, source_type)
        ep = Episode(
            episode_id=episode_id,
            timestamp=ts,
            source_type=source_type,
            session_id=session_id,
            summary=summary,
            key_points=key_points,
            concepts=concepts,
            entity_ids=entity_ids,
            relation_triples=relation_triples,
            raw_text_refs=raw_text_refs,
            structure_description="",
        )
        ep.structure_description = build_structure_description(ep)

        content_text = ep.content_for_embedding()
        content_emb: List[float] = []
        if self.embedding_func and content_text:
            embs = await self._embed([content_text])
            content_emb = embs[0] if embs else []
        ep.content_embedding = content_emb

        structure_text = ep.structure_description
        graph_emb: List[float] = []
        if self.embedding_func and structure_text:
            embs = await self._embed([structure_text])
            graph_emb = embs[0] if embs else []
        ep.graph_embedding = graph_emb

        self.episode_store.put(ep)
        self.graph.add_node(episode_id, "episode", {"episode_id": episode_id})
        if content_emb:
            self.episode_vectors.upsert(episode_id, content_emb)

        # 即时联想：与已有 episode 建边
        all_episodes = self.episode_store.all_episodes()
        if content_emb and all_episodes:
            similar = self.episode_vectors.query(content_emb, top_k=11)
            for eid, sim in similar:
                if eid == episode_id or sim <= 0.3:
                    continue
                self.graph.add_edge(episode_id, eid, EdgeType.EPISODE_SIMILARITY, weight=sim)
                self.graph.add_edge(eid, episode_id, EdgeType.EPISODE_SIMILARITY, weight=sim)

        # 与 entity/chunk 建边（若调用方传入了 entity_ids / raw_text_refs）
        for eid in entity_ids[:20]:
            if eid:
                self.graph.add_node(eid, "entity", {})
                self.graph.add_edge(episode_id, eid, EdgeType.CONCEPT_LINK, weight=0.7)
        for ref in raw_text_refs[:20]:
            if ref:
                self.graph.add_node(ref, "chunk", {})
                self.graph.add_edge(episode_id, ref, EdgeType.SAME_DOCUMENT, weight=0.6)

        self._consolidation_count += 1
        self._ensure_working_dir()
        self.episode_store.save()
        return ep

    async def consolidate(
        self,
        *,
        recent_n: int = 50,
        content_sim_threshold: float = 0.45,
        run_llm: bool = True,
    ) -> Dict[str, Any]:
        """执行一次记忆巩固。"""
        episodes = self.episode_store.all_episodes()
        if not episodes:
            return {"edges_added": 0, "pairs_processed": 0}

        async def content_sim(a: str, b: str) -> float:
            emb_a = await self._embed([a])
            emb_b = await self._embed([b])
            if not emb_a or not emb_b:
                return 0.0
            try:
                dot = sum(x * y for x, y in zip(emb_a[0], emb_b[0]))
                na = (sum(x * x for x in emb_a[0]) ** 0.5) or 1e-8
                nb = (sum(x * x for x in emb_b[0]) ** 0.5) or 1e-8
                return max(0.0, min(1.0, dot / (na * nb)))
            except Exception:
                return 0.0

        async def structure_sim(a: str, b: str) -> float:
            if not a.strip() or not b.strip():
                return 0.0
            ea = await self._embed([a])
            eb = await self._embed([b])
            if not ea or not eb:
                return 0.0
            try:
                dot = sum(x * y for x, y in zip(ea[0], eb[0]))
                na = (sum(x * x for x in ea[0]) ** 0.5) or 1e-8
                nb = (sum(x * x for x in eb[0]) ** 0.5) or 1e-8
                return max(0.0, min(1.0, dot / (na * nb)))
            except Exception:
                return 0.0

        llm_fn = self.llm_func if run_llm else None
        return await consolidate(
            self.graph,
            episodes,
            recent_n=recent_n,
            content_sim_threshold=content_sim_threshold,
            structure_sim_threshold=0.3,
            llm_func=llm_fn,
            content_sim_fn=content_sim,
            structure_sim_fn=structure_sim,
        )

    async def retrieve_analogies(
        self,
        query_text: str,
        *,
        query_structure: Optional[str] = None,
        top_k: int = 10,
        then_spread: bool = True,
        spread_top_k: int = 5,
    ) -> List[Tuple[Episode, float, Optional[str]]]:
        """
        类比检索：先按内容+结构+显著性取 top_k episode；
        若 then_spread 为 True，再以这些 episode 为种子做扩散激活，扩展相关节点（含 entity/chunk）。
        """
        episodes = self.episode_store.all_episodes()
        if not episodes:
            return []

        query_emb = []
        if self.embedding_func and query_text:
            query_emb = (await self._embed([query_text]))[0] if (await self._embed([query_text])) else []
        if not query_emb:
            return []

        def content_search(emb: List[float], k: int) -> List[Tuple[str, float]]:
            return self.episode_vectors.query(emb, top_k=k)

        query_struct = query_structure or structure_description_from_triples([], [])
        results = await retrieve_analogies(
            query_text,
            query_struct,
            episodes,
            content_embed_fn=lambda t: self._embed([t]),
            content_search_fn=content_search,
            structure_sim_fn=None,
            top_k=top_k,
        )
        for ep, score, _ in results:
            ep.record_retrieval()

        if then_spread and results and self.graph:
            seed_ids = [r[0].episode_id for r in results[:spread_top_k]]
            activated = top_k_activated(self.graph, seed_ids, k=spread_top_k * 2, exclude_seeds=True)
            for nid, _ in activated:
                node = self.graph.get_node(nid)
                if node and node.get("type") == "episode":
                    extra = self.episode_store.get(nid)
                    if extra and not any(r[0].episode_id == nid for r in results):
                        results.append((extra, 0.35, "扩散激活关联"))
            results.sort(key=lambda x: -x[1])
            results = results[:top_k]

        return results

    def load(self) -> None:
        """从 working_dir 加载已持久化的 episode 列表与图。"""
        self._ensure_working_dir()
        self.episode_store.load()
        graph_path = Path(self.working_dir) / "memory_graph.json"
        if graph_path.exists():
            try:
                data = json.loads(graph_path.read_text(encoding="utf-8"))
                self.graph = MemoryGraphStore.from_dict(data)
            except Exception:
                pass
        vec_path = Path(self.working_dir) / "episode_vectors.json"
        if vec_path.exists():
            try:
                data = json.loads(vec_path.read_text(encoding="utf-8"))
                self.episode_vectors = EpisodeVectorStore.from_dict(data)
            except Exception:
                pass

    def save(self) -> None:
        self.episode_store.save()
        graph_path = Path(self.working_dir) / "memory_graph.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(
            json.dumps(self.graph.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        vec_path = Path(self.working_dir) / "episode_vectors.json"
        vec_path.write_text(
            json.dumps(self.episode_vectors.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
