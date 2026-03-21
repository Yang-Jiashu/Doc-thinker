from fastapi import APIRouter, HTTPException

from ..state import state
from ..schemas import CreateSessionRequest
from ..memory import remove_session_memory_engine


router = APIRouter()


@router.post("/sessions")
async def create_session(request: CreateSessionRequest):
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    try:
        session = state.session_manager.create_session(title=request.title)
        return {"status": "success", "session": session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions():
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    try:
        sessions = state.session_manager.list_sessions()
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    session = state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session": session}


@router.put("/sessions/{session_id}")
async def update_session(session_id: str, request: CreateSessionRequest):
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    if state.session_manager.update_session(session_id, request.title):
        return {"status": "success", "message": "Session updated"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    if state.session_manager.delete_session(session_id):
        remove_session_memory_engine(session_id, save_before_remove=False)
        return {"status": "success", "message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    try:
        history = state.session_manager.get_history(session_id)
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/files")
async def get_session_files(session_id: str):
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    try:
        files = state.session_manager.get_files(session_id)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

