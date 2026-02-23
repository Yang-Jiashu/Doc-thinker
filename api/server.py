"""
API Server - FastAPI 服务

提供 HTTP API 接口：
- /chat: 对话接口
- /memory: 记忆操作
- /document: 文档处理
- /session: 会话管理
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: fastapi not installed, API server unavailable")


class ChatRequest(BaseModel):
    query: str
    session_id: str = None


class ChatResponse(BaseModel):
    answer: str
    memories: list = []
    session_id: str = None


def create_app(agent=None) -> "FastAPI":
    """创建 FastAPI 应用"""
    if not HAS_FASTAPI:
        raise ImportError("fastapi is required")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """生命周期管理"""
        # 启动时加载
        if agent:
            agent.memory.load()
        yield
        # 关闭时保存
        if agent:
            agent.memory.save()
    
    app = FastAPI(
        title="NeuroAgent API",
        description="类人脑智能体 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    @app.get("/health")
    async def health():
        """健康检查"""
        return {"status": "ok", "agent_loaded": agent is not None}
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """对话接口"""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        try:
            result = await agent.respond(
                query=request.query,
                session_id=request.session_id,
            )
            
            return ChatResponse(
                answer=result["answer"],
                memories=[
                    {
                        "summary": m["summary"],
                        "score": m["score"],
                        "source": m["source"],
                    }
                    for m in result["memories"]
                ],
                session_id=result["session_id"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/memory/consolidate")
    async def consolidate():
        """触发记忆巩固"""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        try:
            result = await agent.consolidate()
            return {"status": "ok", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/memory/stats")
    async def memory_stats():
        """记忆统计"""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        episodes = agent.memory.episode_store.all_episodes()
        return {
            "episode_count": len(episodes),
            "graph_nodes": len(agent.memory.graph.nodes) if agent.memory.graph else 0,
            "graph_edges": len(agent.memory.graph.edges) if agent.memory.graph else 0,
        }
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    # 简单测试
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
