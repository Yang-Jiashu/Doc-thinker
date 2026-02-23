"""
文档感知器 (ToC 场景优化版)

核心功能:
1. 多格式文档解析 (PDF/Word/Markdown/图片)
2. 层级化信息提取 (高阶概念 → 低阶具体内容)
3. 文档结构保留 (章节层级 → 记忆层级)
4. 自动概念提取和层级映射
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuro_core import Episode
from perception.base import BasePerceiver, PerceptionResult


class DocumentPerceiver(BasePerceiver):
    """
    文档感知器 - ToC 个人知识助手场景
    
    设计目标:
    - 用户上传的文档是个人知识的输入
    - 从文档提取层级化知识结构
    - 建立"文档 → 章节 → 概念 → 实例"的记忆映射
    """
    
    def __init__(
        self,
        parser=None,
        cognitive_processor=None,
        enable_multimodal: bool = True,
        extract_hierarchy: bool = True,  # 提取层级结构
    ):
        super().__init__(source_type="document")
        self.parser = parser
        self.cognitive_processor = cognitive_processor
        self.enable_multimodal = enable_multimodal
        self.extract_hierarchy = extract_hierarchy
    
    async def perceive(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        user_context: Optional[Dict] = None,
        **kwargs
    ) -> PerceptionResult:
        """
        感知文档并生成层级化 Episode
        
        流程:
        1. 解析文档结构
        2. 提取层级化信息 (标题层级 → 概念层级)
        3. 生成主 Episode + 子 Episode (对应章节)
        4. 建立层级关联
        """
        user_context = user_context or {}
        
        # 1. 解析文档
        print(f"[文档感知] 解析: {file_path}")
        doc_structure = await self._parse_document_structure(file_path)
        
        # 2. 提取文档级信息 (高阶抽象)
        doc_summary = await self._extract_document_summary(doc_structure)
        
        # 3. 识别文档领域和核心概念
        domain_concepts = await self._extract_domain_concepts(doc_structure)
        
        # 4. 创建主 Episode (代表整个文档)
        ts = time.time()
        main_episode_id = doc_id or self._create_episode_id(file_path, ts)
        
        main_episode = Episode(
            episode_id=main_episode_id,
            timestamp=ts,
            source_type="document",
            summary=doc_summary.get("title", Path(file_path).stem),
            key_points=doc_summary.get("key_points", []),
            concepts=domain_concepts.get("concepts", []),  # 高阶概念
            entity_ids=domain_concepts.get("entities", []),
            relation_triples=domain_concepts.get("relations", []),
            raw_text_refs=[file_path],
            metadata={
                "doc_type": doc_structure.get("type", "unknown"),
                "page_count": doc_structure.get("page_count", 0),
                "hierarchy_depth": doc_structure.get("max_depth", 1),
                **user_context,
            }
        )
        
        # 5. 提取章节级 Episode (中阶概念)
        chapter_episodes = []
        if self.extract_hierarchy and "chapters" in doc_structure:
            for chapter in doc_structure["chapters"]:
                chapter_ep = await self._create_chapter_episode(
                    chapter, main_episode_id, ts
                )
                if chapter_ep:
                    chapter_episodes.append(chapter_ep)
        
        # 6. 组装结果
        return PerceptionResult(
            episode=main_episode,
            raw_chunks=doc_structure.get("chunks", []),
            metadata={
                "file_path": file_path,
                "doc_id": doc_id,
                "domain": domain_concepts.get("domain"),  # 高阶领域
                "chapter_count": len(chapter_episodes),
                "chapter_episodes": [ep.episode_id for ep in chapter_episodes],
                "structure": doc_structure.get("outline", []),
            }
        )
    
    async def _parse_document_structure(self, file_path: str) -> Dict[str, Any]:
        """
        解析文档结构
        
        返回包含层级结构的信息:
        {
            "type": "pdf",
            "title": "...",
            "outline": [章节层级],
            "chapters": [章节内容],
            "chunks": [文本块],
        }
        """
        # 这里应该调用实际解析器 (MinerU/Docling)
        # 简化版本: 模拟层级结构
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".pdf":
            # 调用 PDF 解析器
            return await self._parse_pdf(file_path)
        elif file_ext in [".md", ".markdown"]:
            # Markdown 天然有层级结构
            return await self._parse_markdown(file_path)
        elif file_ext in [".docx", ".doc"]:
            return await self._parse_word(file_path)
        else:
            # 通用文本解析
            return await self._parse_text(file_path)
    
    async def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """解析 PDF 结构"""
        # 实际应该调用 MinerU
        # 这里返回模拟结构
        return {
            "type": "pdf",
            "title": Path(file_path).stem,
            "page_count": 10,
            "max_depth": 3,
            "outline": [
                {"level": 1, "title": "引言", "page": 1},
                {"level": 1, "title": "相关工作", "page": 2},
                {"level": 2, "title": "深度学习", "page": 2},
                {"level": 2, "title": "CNN 架构", "page": 3},
                {"level": 1, "title": "方法", "page": 4},
                {"level": 1, "title": "实验", "page": 7},
            ],
            "chapters": [
                {"level": 1, "title": "引言", "content": "..."},
                {"level": 1, "title": "相关工作", "content": "...", 
                 "subsections": [
                     {"level": 2, "title": "深度学习", "content": "..."},
                     {"level": 2, "title": "CNN 架构", "content": "..."},
                 ]},
                {"level": 1, "title": "方法", "content": "..."},
            ],
            "chunks": [],
        }
    
    async def _parse_markdown(self, file_path: str) -> Dict[str, Any]:
        """解析 Markdown 结构"""
        content = Path(file_path).read_text(encoding="utf-8")
        
        # 提取标题层级
        import re
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        
        outline = []
        for level, title in headers:
            outline.append({
                "level": len(level),
                "title": title.strip(),
            })
        
        return {
            "type": "markdown",
            "title": outline[0]["title"] if outline else Path(file_path).stem,
            "outline": outline,
            "content": content,
            "max_depth": max([len(h[0]) for h in headers]) if headers else 1,
        }
    
    async def _parse_word(self, file_path: str) -> Dict[str, Any]:
        """解析 Word 结构"""
        # 简化版
        return {
            "type": "word",
            "title": Path(file_path).stem,
            "outline": [],
        }
    
    async def _parse_text(self, file_path: str) -> Dict[str, Any]:
        """通用文本解析"""
        content = Path(file_path).read_text(encoding="utf-8")
        return {
            "type": "text",
            "title": Path(file_path).stem,
            "content": content,
            "outline": [],
        }
    
    async def _extract_document_summary(self, doc_structure: Dict) -> Dict[str, Any]:
        """提取文档级摘要 (高阶抽象)"""
        # 提取标题
        title = doc_structure.get("title", "")
        outline = doc_structure.get("outline", [])
        
        # 从大纲提取关键点
        key_points = [item["title"] for item in outline if item.get("level") == 1]
        
        # 如果有认知处理器，生成智能摘要
        if self.cognitive_processor and "content" in doc_structure:
            try:
                content = doc_structure["content"][:3000]  # 取前3000字符
                insight = await self.cognitive_processor.process(content, "document")
                return {
                    "title": insight.summary or title,
                    "key_points": insight.key_points or key_points,
                }
            except Exception as e:
                print(f"[认知处理失败] {e}")
        
        return {
            "title": title,
            "key_points": key_points[:5],  # 最多5个关键点
        }
    
    async def _extract_domain_concepts(self, doc_structure: Dict) -> Dict[str, Any]:
        """
        提取领域和概念 (层级映射)
        
        从高阶到低阶:
        - Domain (领域): "人工智能"
        - Concepts (概念): ["深度学习", "CNN"]
        - Entities (实体): ["ResNet", "ImageNet"]
        """
        title = doc_structure.get("title", "")
        outline = doc_structure.get("outline", [])
        
        # 简化版：基于关键词推断
        # 实际应该用 LLM 做更智能的提取
        
        all_text = title + " " + " ".join([item.get("title", "") for item in outline])
        all_text = all_text.lower()
        
        # 推断领域
        domain = None
        if any(kw in all_text for kw in ["深度学习", "神经网络", "cnn", "rnn"]):
            domain = "人工智能"
        elif any(kw in all_text for kw in ["医学", "诊断", "治疗"]):
            domain = "医学"
        elif any(kw in all_text for kw in ["法律", "合同", "法规"]):
            domain = "法律"
        
        # 提取概念 (从大纲)
        concepts = []
        for item in outline:
            title = item.get("title", "")
            # 去除常见虚词，提取核心概念
            concept = title.strip()
            if concept and len(concept) < 50:  # 不要太长
                concepts.append(concept)
        
        # 去重并限制数量
        concepts = list(dict.fromkeys(concepts))[:10]  # 最多10个概念
        
        return {
            "domain": domain,  # 高阶
            "concepts": concepts,  # 中阶
            "entities": [],  # 低阶 (需要更细粒度提取)
            "relations": [],
        }
    
    async def _create_chapter_episode(
        self,
        chapter: Dict,
        parent_doc_id: str,
        parent_timestamp: float
    ) -> Optional[Episode]:
        """为章节创建 Episode (子记忆)"""
        chapter_title = chapter.get("title", "")
        chapter_level = chapter.get("level", 1)
        
        if not chapter_title:
            return None
        
        # 生成章节级 Episode ID
        chapter_id = f"{parent_doc_id}_ch{hash(chapter_title) % 10000:04d}"
        
        episode = Episode(
            episode_id=chapter_id,
            timestamp=parent_timestamp,
            source_type="document_chapter",
            summary=chapter_title,
            key_points=[],
            concepts=[chapter_title],  # 章节标题作为概念
            entity_ids=[],
            relation_triples=[],
            raw_text_refs=[parent_doc_id],
            metadata={
                "parent_doc": parent_doc_id,
                "chapter_level": chapter_level,
                "chapter_type": "subsection" if chapter_level > 1 else "chapter",
            }
        )
        
        return episode
    
    def _extract_text(self, content_list: List[Dict]) -> str:
        """提取纯文本"""
        texts = []
        for item in content_list:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    
    def _extract_images(self, content_list: List[Dict]) -> List[Dict]:
        """提取图片"""
        return [item for item in content_list if item.get("type") == "image"]
    
    def _extract_tables(self, content_list: List[Dict]) -> List[Dict]:
        """提取表格"""
        return [item for item in content_list if item.get("type") == "table"]
