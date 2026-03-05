# neuro_memory/models.py
"""记忆与联想图的数据模型：Episode、边类型、图节点。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EdgeType(str, Enum):
    """边类型：对应人脑中的不同联想方式。"""
    SEMANTIC_SIMILARITY = "semantic_similarity"   # 语义相似，想到易联想到
    SAME_DOCUMENT = "same_document"               # 同文档邻近
    CONCEPT_LINK = "concept_link"                  # 概念/关键词共现
    INFERRED_RELATION = "inferred_relation"       # 推断出的关系
    EPISODE_SIMILARITY = "episode_similarity"      # 事件级相似
    ANALOGOUS_TO = "analogous_to"                 # 结构类比
    SAME_THEME = "same_theme"                     # 同一主题
    CO_ACTIVATED = "co_activated"                 # 巩固时共同激活
    MENTIONS = "mentions"                         # 桥接边：记忆图 episode 引用主 KG 实体


# 边权上限：达到后不再增强
DEFAULT_MAX_EDGE_WEIGHT: float = 1.0
# 扩散激活时每次激活的权重增量
DEFAULT_ACTIVATION_WEIGHT_DELTA: float = 0.05
# 巩固时对近期激活边的权重增量
DEFAULT_CONSOLIDATION_WEIGHT_DELTA: float = 0.08

# 每跳衰减系数（越大衰减越慢）
EDGE_TYPE_DECAY: Dict[EdgeType, float] = {
    EdgeType.SEMANTIC_SIMILARITY: 0.85,
    EdgeType.EPISODE_SIMILARITY: 0.85,
    EdgeType.ANALOGOUS_TO: 0.82,
    EdgeType.SAME_THEME: 0.80,
    EdgeType.CONCEPT_LINK: 0.75,
    EdgeType.INFERRED_RELATION: 0.75,
    EdgeType.CO_ACTIVATED: 0.70,
    EdgeType.MENTIONS: 0.75,
    EdgeType.SAME_DOCUMENT: 0.60,
}


@dataclass
class Episode:
    """单次经历/记忆胶囊：对应人脑中的一次事件记忆。"""
    episode_id: str
    timestamp: float = field(default_factory=time.time)
    source_type: str = "generic"  # doc | chat | event
    session_id: Optional[str] = None

    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)

    entity_ids: List[str] = field(default_factory=list)
    relation_triples: List[Tuple[str, str, str]] = field(default_factory=list)  # (s, r, t)

    content_embedding: Optional[List[float]] = None
    graph_embedding: Optional[List[float]] = None

    raw_text_refs: List[str] = field(default_factory=list)  # chunk_id / doc_id
    structure_description: str = ""  # 用于结构类比的结构描述文本

    retrieval_count: int = 0
    last_retrieved_at: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "session_id": self.session_id,
            "summary": self.summary,
            "key_points": self.key_points,
            "concepts": self.concepts,
            "entity_ids": self.entity_ids,
            "relation_triples": [list(t) for t in self.relation_triples],
            "raw_text_refs": self.raw_text_refs,
            "structure_description": self.structure_description,
            "retrieval_count": self.retrieval_count,
            "last_retrieved_at": self.last_retrieved_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        triples = d.get("relation_triples", [])
        return cls(
            episode_id=d["episode_id"],
            timestamp=d.get("timestamp", time.time()),
            source_type=d.get("source_type", "generic"),
            session_id=d.get("session_id"),
            summary=d.get("summary", ""),
            key_points=d.get("key_points", []),
            concepts=d.get("concepts", []),
            entity_ids=d.get("entity_ids", []),
            relation_triples=[(t[0], t[1], t[2]) for t in triples if len(t) >= 3],
            raw_text_refs=d.get("raw_text_refs", []),
            structure_description=d.get("structure_description", ""),
            retrieval_count=d.get("retrieval_count", 0),
            last_retrieved_at=d.get("last_retrieved_at"),
            metadata=d.get("metadata", {}),
        )

    def content_for_embedding(self) -> str:
        """用于生成 content_embedding 的拼接文本。"""
        parts = [self.summary]
        if self.key_points:
            parts.append(" ".join(self.key_points[:15]))
        if self.concepts:
            parts.append(" ".join(self.concepts[:20]))
        return "\n".join(parts).strip() or "(no content)"

    def record_retrieval(self) -> None:
        self.retrieval_count += 1
        self.last_retrieved_at = time.time()


@dataclass
class MemoryEdge:
    """记忆图中的一条边。"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_activated_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def edge_key(self) -> str:
        return f"{self.source_id}\t{self.edge_type.value}\t{self.target_id}"


def get_decay_for_edge_type(edge_type: EdgeType) -> float:
    return EDGE_TYPE_DECAY.get(edge_type, 0.7)
