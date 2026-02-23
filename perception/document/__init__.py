"""
Document Perception - 文档感知

支持多种文档格式的解析：
- PDF（通过 MinerU/Docling）
- 图片（OCR + VLM）
- Office 文档
"""

from .parser import Parser as DocumentParser
from .perceiver import DocumentPerceiver

__all__ = ["DocumentParser", "DocumentPerceiver"]
