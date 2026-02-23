"""Image position lookup utilities for MinerU content_list."""
#实现了一个function call的功能，用于解析对应页码特殊位置的图像内容，并加入到检索内容中
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict,       List, Optional, Tuple, Union


POSITION_KEYWORDS = {
    "top": ["top", "upper", "上", "顶部", "最上"],
    "bottom": ["bottom", "下", "底部", "最下"],
    "left": ["left", "左", "最左"],
    "right": ["right", "右", "最右"],
}


def parse_page(query: str) -> Optional[int]:
    match = re.search(r"page\s*(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_position(query: str) -> Optional[str]:
    q = query.lower()
    for pos, keys in POSITION_KEYWORDS.items():
        if any(k in q for k in keys):
            return pos
    return None


def _find_content_list_path(content_root: str, doc_id: str) -> Optional[Path]:
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


def find_image_by_position(
    contents: List[Dict[str, Any]],
    page_hint: Optional[int],
    position: Optional[str],
) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Pick image block(s) by page and rough position using bbox."""
    candidates: List[Tuple[int, float, float, float, float, Dict[str, Any]]] = []
    for block in contents:
        if str(block.get("type", "")).lower() not in {"image", "figure", "picture"}:
            continue
        pg = block.get("page_idx")
        if page_hint is not None and pg is not None and pg != page_hint - 1:
            continue
        bbox = block.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            x0, y0, x1, y1 = map(float, bbox)
        except Exception:
            continue
        candidates.append((pg if pg is not None else -1, x0, y0, x1, y1, block))

    if not candidates:
        return None

    pos = position
    if pos is None:
        # return all candidates on the hinted page (or all if no hint), sorted by page then y0
        candidates.sort(key=lambda t: (t[0], t[2]))
        return [c[5] for c in candidates]

    if pos == "top":
        candidates.sort(key=lambda t: (t[2], t[0]))  # y0 asc, then page
    elif pos == "bottom":
        candidates.sort(key=lambda t: (-t[4], t[0]))  # y1 desc
    elif pos == "left":
        candidates.sort(key=lambda t: (t[1], t[0]))  # x0 asc
    elif pos == "right":
        candidates.sort(key=lambda t: (-t[3], t[0]))  # x1 desc
    else:
        candidates.sort(key=lambda t: (t[2], t[0]))

    return candidates[0][5]
