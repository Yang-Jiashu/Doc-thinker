"""
Perception - 感知层

将外部输入转换为 Episode（情节记忆）：
- DocumentPerceiver: 文档感知（PDF、图片等）
- ChatPerceiver: 对话感知
- APIPerceiver: API数据感知
"""

from .base import BasePerceiver, PerceptionResult

__all__ = ["BasePerceiver", "PerceptionResult"]
