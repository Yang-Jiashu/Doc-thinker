#!/usr/bin/env python3
"""启动 Doc Thinker / AutoThink 后端 API（默认端口 8000）。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 模块级 app，供 python scripts/start_api.py 与 hypercorn scripts.start_api:app 使用
from raganything.server.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
