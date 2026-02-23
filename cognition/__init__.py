"""
Cognition - 认知层

提供认知处理能力：
- CognitiveProcessor: 认知处理器（理解、关联、推理）
- IntentClassifier: 意图分类
- Reasoner: 推理规划
"""

from .processor import CognitiveProcessor, CognitiveInsight

__all__ = ["CognitiveProcessor", "CognitiveInsight"]
