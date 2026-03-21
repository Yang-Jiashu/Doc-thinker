"""Page-level lookup utilities for MinerU content_list."""
#funcation call,查询中出现对应页码，则去content_list中查找对应页码的内容，合并到检索内容中。
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


PAGE_PATTERN = re.compile(r"page\s*(\d+)(?:\s*[-~到至]\s*(\d+))?", re.IGNORECASE)


def parse_page_range(query: str) -> Optional[Tuple[int, int]]:
    """Extract 1-based page range from query text."""
    match = PAGE_PATTERN.search(query)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or start)
    return (start, end) if start <= end else (end, start)


def _find_content_list_path(content_root: str, doc_id: str) -> Optional[Path]:
    """Locate <doc_id>_content_list.json under content_root."""
    root = Path(content_root)
    if root.is_file() and root.name.endswith("_content_list.json"):
        return root
    pattern = f"{doc_id}_content_list.json"
    candidates = sorted(root.rglob(pattern))
    return candidates[0] if candidates else None


def load_content_list(content_root: str, doc_id: str) -> List[Dict[str, Any]]:
    path = _find_content_list_path(content_root, doc_id)
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "content_list" in data:
        return data["content_list"] or []
    if isinstance(data, list):
        return data
    return []


def find_text_by_page(
    contents: List[Dict[str, Any]], start: int, end: int
) -> str:
    """Return concatenated text from page range [start, end], using page_idx (0-based)."""
    if start <= 0 or end <= 0:
        return ""
    lines: List[str] = []
    for block in contents:
        pg = block.get("page_idx")
        if pg is None:
            continue
        if not (start - 1 <= pg <= end - 1):
            continue
        text = block.get("text") or block.get("content") or ""
        if isinstance(text, str) and text.strip():
            lines.append(f"[p{pg + 1}] {text.strip()}")
    return "\n".join(lines)
