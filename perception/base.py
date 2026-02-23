"""
感知器基类

所有感知器都继承此类，将外部输入转换为统一的 Episode 格式
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from neuro_core import Episode


@dataclass
class PerceptionResult:
    """感知结果"""
    episode: Episode                    # 生成的 Episode
    raw_chunks: List[Dict] = field(default_factory=list)  # 原始分块
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


class BasePerceiver(ABC):
    """感知器基类"""
    
    def __init__(self, source_type: str):
        self.source_type = source_type
    
    @abstractmethod
    async def perceive(self, input_data: Any, **kwargs) -> PerceptionResult:
        """
        将输入数据感知为 Episode
        
        Args:
            input_data: 原始输入数据
            **kwargs: 额外参数
            
        Returns:
            PerceptionResult: 包含 Episode 和元数据
        """
        pass
    
    def _create_episode_id(self, content: str, timestamp: float) -> str:
        """生成 Episode ID"""
        import hashlib
        raw = f"{content[:200]}|{timestamp}|{self.source_type}"
        return "ep-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
