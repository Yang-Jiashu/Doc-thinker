"""
NeuroAgent - 类人脑智能体

核心循环：
1. Perceive: 感知输入（文档/对话/API）→ Episode
2. Memorize: 存入记忆系统（即时联想）
3. Recall: 检索相关记忆（混合检索）
4. Reason: 推理生成回答
5. Reflect: 反思并巩固记忆
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from neuro_core import MemoryEngine, Episode
from cognition import CognitiveProcessor
from retrieval import HybridRetriever


class NeuroAgent:
    """类人脑智能体"""
    
    def __init__(
        self,
        memory_engine: Optional[MemoryEngine] = None,
        cognitive_processor: Optional[CognitiveProcessor] = None,
        retriever: Optional[HybridRetriever] = None,
        llm_func=None,
        embedding_func=None,
        working_dir: str = "./neuro_agent_data",
    ):
        # 初始化记忆引擎
        self.memory = memory_engine or MemoryEngine(
            embedding_func=embedding_func,
            llm_func=llm_func,
            working_dir=working_dir,
        )
        
        # 初始化认知处理器
        self.cognition = cognitive_processor or CognitiveProcessor(
            llm_func=llm_func,
            embedding_func=embedding_func,
        )
        
        # 初始化检索器
        self.retriever = retriever or HybridRetriever(self.memory)
        
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        
        # 感知器注册表
        self.perceivers: Dict[str, Any] = {}
        
        # 加载已有记忆
        self.memory.load()
    
    def register_perceiver(self, source_type: str, perceiver):
        """注册感知器"""
        self.perceivers[source_type] = perceiver
    
    async def perceive(
        self,
        input_data: Any,
        source_type: str = "chat",
        **kwargs
    ) -> Episode:
        """
        感知输入并生成 Episode
        
        Args:
            input_data: 输入数据
            source_type: 输入类型 (document/chat/api)
            **kwargs: 额外参数
            
        Returns:
            Episode: 生成的情节记忆
        """
        perceiver = self.perceivers.get(source_type)
        if perceiver is None:
            raise ValueError(f"No perceiver registered for source_type: {source_type}")
        
        # 感知输入
        result = await perceiver.perceive(input_data, **kwargs)
        
        # 存入记忆
        episode = result.episode
        await self.memory.add_observation(
            summary=episode.summary,
            key_points=episode.key_points,
            concepts=episode.concepts,
            entity_ids=episode.entity_ids,
            relation_triples=episode.relation_triples,
            raw_text_refs=episode.raw_text_refs,
            source_type=episode.source_type,
            session_id=episode.session_id,
        )
        
        return episode
    
    async def recall(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        回忆相关记忆
        
        Args:
            query: 查询
            top_k: 返回数量
            context: 上下文
            
        Returns:
            检索结果列表
        """
        results = await self.retriever.retrieve(query, top_k, context)
        
        # 格式化输出
        formatted = []
        for episode, score, source in results:
            formatted.append({
                "episode": episode,
                "score": score,
                "source": source,
                "summary": episode.summary,
                "concepts": episode.concepts,
                "timestamp": episode.timestamp,
            })
        
        return formatted
    
    async def respond(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        生成回答
        
        流程：
        1. 检索相关记忆
        2. 构建上下文
        3. LLM 生成回答
        4. 可选：反思并存储这次交互
        
        Args:
            query: 用户查询
            session_id: 会话ID
            context: 上下文
            
        Returns:
            包含回答和检索结果的字典
        """
        # 1. 检索记忆
        memories = await self.recall(query, top_k=10, context=context)
        
        # 2. 构建上下文
        context_text = self._build_context(query, memories)
        
        # 3. 生成回答
        answer = await self._generate_answer(query, context_text)
        
        # 4. 存储这次交互（异步，不阻塞返回）
        asyncio.create_task(self._store_interaction(
            query, answer, session_id, memories
        ))
        
        return {
            "answer": answer,
            "query": query,
            "memories": memories,
            "session_id": session_id,
        }
    
    def _build_context(
        self,
        query: str,
        memories: List[Dict],
    ) -> str:
        """构建上下文"""
        parts = []
        
        if memories:
            parts.append("相关记忆：")
            for i, mem in enumerate(memories[:5], 1):
                ep = mem["episode"]
                parts.append(f"{i}. {ep.summary}")
                if ep.key_points:
                    parts.append(f"   要点: {', '.join(ep.key_points[:3])}")
        
        return "\n".join(parts)
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
    ) -> str:
        """生成回答"""
        if not self.llm_func:
            return f"[No LLM configured] Query: {query}"
        
        prompt = f"""基于以下相关信息回答用户问题。

{context}

用户问题：{query}

请给出详细、准确的回答。如果相关信息不足，请说明。"""
        
        try:
            response = await self.llm_func(prompt)
            return response.strip()
        except Exception as e:
            return f"[生成回答时出错: {e}]"
    
    async def _store_interaction(
        self,
        query: str,
        answer: str,
        session_id: Optional[str],
        memories: List[Dict],
    ):
        """存储交互到记忆"""
        try:
            # 使用认知处理器生成总结
            content = f"用户: {query}\n助手: {answer}"
            insight = await self.cognition.process(content, source_type="chat")
            
            # 创建 Episode
            await self.memory.add_observation(
                summary=insight.summary or f"Q: {query[:100]}...",
                key_points=[f"Q: {query}", f"A: {answer[:200]}"] + insight.key_points,
                concepts=insight.concepts,
                entity_ids=[e.name for e in insight.entities],
                relation_triples=[(r.source, r.relation, r.target) for r in insight.relations],
                source_type="chat",
                session_id=session_id,
            )
        except Exception as e:
            print(f"Store interaction failed: {e}")
    
    async def consolidate(self):
        """执行记忆巩固"""
        return await self.memory.consolidate()
    
    def save(self):
        """保存记忆状态"""
        self.memory.save()
