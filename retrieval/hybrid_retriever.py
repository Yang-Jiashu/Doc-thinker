"""
混合检索器

结合多种检索策略：
1. 向量语义检索（相似度）
2. 图扩散激活（关联性）
3. 类比检索（结构性）
"""

from typing import Any, Dict, List, Optional, Tuple
from neuro_core import MemoryEngine, Episode, top_k_activated, retrieve_analogies


class HybridRetriever:
    """混合检索器"""
    
    def __init__(
        self,
        memory_engine: MemoryEngine,
        vector_weight: float = 0.4,
        graph_weight: float = 0.35,
        analogy_weight: float = 0.25,
    ):
        self.memory = memory_engine
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.analogy_weight = analogy_weight
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict] = None,
    ) -> List[Tuple[Episode, float, str]]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            context: 上下文信息
            
        Returns:
            [(Episode, score, source), ...]
            source: "vector" | "graph" | "analogy"
        """
        results = []
        
        # 1. 向量语义检索（通过 MemoryEngine 的 episode_vectors）
        vector_results = await self._vector_search(query, top_k * 2)
        for ep, score in vector_results:
            results.append((ep, score * self.vector_weight, "vector"))
        
        # 2. 类比检索（内容 + 结构 + 显著性）
        analogy_results = await self._analogy_search(query, top_k * 2)
        for ep, score, _ in analogy_results:
            # 检查是否已存在
            existing = next((r for r in results if r[0].episode_id == ep.episode_id), None)
            if existing:
                # 合并分数
                idx = results.index(existing)
                combined_score = existing[1] + score * self.analogy_weight
                results[idx] = (ep, combined_score, existing[2] + "/analogy")
            else:
                results.append((ep, score * self.analogy_weight, "analogy"))
        
        # 3. 图扩散激活（以高分为种子）
        if self.memory.graph:
            seed_ids = [r[0].episode_id for r in results[:5]]
            activated = top_k_activated(
                self.memory.graph,
                seed_ids,
                k=top_k,
                exclude_seeds=True
            )
            for node_id, activation in activated:
                # 查找对应的 episode
                ep = self.memory.episode_store.get(node_id)
                if ep:
                    existing = next((r for r in results if r[0].episode_id == ep.episode_id), None)
                    if existing:
                        idx = results.index(existing)
                        combined_score = existing[1] + activation * self.graph_weight
                        results[idx] = (ep, combined_score, existing[2] + "/graph")
                    else:
                        results.append((ep, activation * self.graph_weight, "graph"))
        
        # 排序并返回 top_k
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    async def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Episode, float]]:
        """向量语义检索"""
        if not self.memory.embedding_func:
            return []
        
        # 生成查询向量
        query_emb = await self.memory._embed([query])
        if not query_emb or not query_emb[0]:
            return []
        
        # 在 episode_vectors 中检索
        similar = self.memory.episode_vectors.query(query_emb[0], top_k=top_k)
        
        results = []
        for episode_id, similarity in similar:
            ep = self.memory.episode_store.get(episode_id)
            if ep:
                results.append((ep, similarity))
        
        return results
    
    async def _analogy_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Episode, float, Optional[str]]]:
        """类比检索"""
        return await self.memory.retrieve_analogies(
            query,
            top_k=top_k,
            then_spread=False,  # 我们已经单独做了扩散
        )
