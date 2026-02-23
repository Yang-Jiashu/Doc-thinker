"""
对话感知器

将对话消息转换为 Episode，支持多轮对话上下文
"""

import time
from typing import Any, Dict, List, Optional

from neuro_core import Episode
from perception.base import BasePerceiver, PerceptionResult


class ChatPerceiver(BasePerceiver):
    """对话感知器"""
    
    def __init__(
        self,
        cognitive_processor=None,
        context_window: int = 3,
    ):
        super().__init__(source_type="chat")
        self.cognitive_processor = cognitive_processor
        self.context_window = context_window
    
    async def perceive(
        self,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        **kwargs
    ) -> PerceptionResult:
        """
        感知对话并生成 Episode
        
        Args:
            messages: 对话消息列表 [{"role": "user/assistant", "content": "..."}]
            session_id: 会话ID
            **kwargs: 额外参数
            
        Returns:
            PerceptionResult: 包含 Episode
        """
        # 拼接最近几轮对话
        recent_messages = messages[-self.context_window:]
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in recent_messages
        ])
        
        # 生成总结（如果有认知处理器）
        summary = recent_messages[-1].get("content", "")[:200] if recent_messages else ""
        key_points = []
        concepts = []
        entities = []
        relations = []
        
        if self.cognitive_processor:
            try:
                insight = await self.cognitive_processor.process(
                    conversation_text,
                    source_type="chat"
                )
                summary = insight.summary or summary
                key_points = insight.key_points
                concepts = insight.concepts
                entities = [e.name for e in insight.entities]
                relations = [(r.source, r.relation, r.target) for r in insight.relations]
            except Exception as e:
                print(f"Cognitive processing failed: {e}")
        
        # 创建 Episode
        ts = time.time()
        episode_id = self._create_episode_id(conversation_text, ts)
        
        episode = Episode(
            episode_id=episode_id,
            timestamp=ts,
            source_type="chat",
            session_id=session_id,
            summary=summary,
            key_points=key_points,
            concepts=concepts,
            entity_ids=entities,
            relation_triples=relations,
            raw_text_refs=[f"session:{session_id}"] if session_id else [],
        )
        
        return PerceptionResult(
            episode=episode,
            raw_chunks=recent_messages,
            metadata={
                "session_id": session_id,
                "message_count": len(recent_messages),
            }
        )
