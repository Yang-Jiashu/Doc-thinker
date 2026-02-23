from fastapi import APIRouter


router = APIRouter()


@router.post("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Multi-Document Enhanced RAG API",
        "version": "1.0.0",
    }

