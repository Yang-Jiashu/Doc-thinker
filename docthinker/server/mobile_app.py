"""Private mobile API entrypoint. Does not mount legacy/admin endpoints."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException


@asynccontextmanager
async def _safe_mobile_lifespan(app):
    # Fail before the production lifecycle can create stores or model clients.
    if os.getenv("DOCTHINKER_MOBILE_ENABLED", "").lower() not in {"1", "true"}:
        raise ValueError("Mobile access must be explicitly enabled before startup")
    if len(os.getenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", "")) < 32:
        raise ValueError("Mobile startup requires a dedicated access token of at least 32 characters")
    directory = os.getenv("DOCTHINKER_MOBILE_DATA_DIR", "").strip()
    if not directory or not Path(directory).expanduser().is_absolute():
        raise ValueError("DOCTHINKER_MOBILE_DATA_DIR must be an explicit absolute directory")
    from .app import lifespan

    async with lifespan(app, mobile_data_dir=Path(directory).expanduser()):
        yield


def create_mobile_app(*, lifespan=None):
    # Resolve the production lifecycle lazily: offline tests never initialize RAG.
    if lifespan is None:
        lifespan = _safe_mobile_lifespan
    from .routers.mobile import router as mobile_router
    from .routers.voice import router as voice_router
    app = FastAPI(title="DocThinker Private Mobile API", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    @app.exception_handler(HTTPException)
    async def safe_http_error(request, exc):
        # Legacy handlers sometimes wrap internal filesystem/provider errors in
        # 404/500 details. Do not relay those details to the mobile client.
        messages = {400: "Invalid request", 401: "Invalid access token", 404: "Resource unavailable",
                    408: "Request timed out", 409: "Request ID belongs to different content", 413: "Request too large", 415: "Unsupported format",
                    422: "Invalid input", 429: "Service is busy", 503: "Service is not configured",
                    504: "Request timed out; check history before retrying"}
        return JSONResponse({"detail": messages.get(exc.status_code, "Service unavailable")},
                            status_code=exc.status_code, headers={**(exc.headers or {}), "Cache-Control": "no-store"})

    @app.exception_handler(Exception)
    async def safe_error(request, exc):
        return JSONResponse({"detail": "Service unavailable"}, status_code=500, headers={"Cache-Control": "no-store"})
    origins = [s.strip() for s in os.getenv("DOCTHINKER_MOBILE_CORS_ORIGINS", "").split(",") if s.strip()]
    if origins:
        if "*" in origins:
            raise ValueError("Mobile CORS origins must be explicit, never wildcard")
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
    app.include_router(mobile_router, prefix="/api/v1")
    app.include_router(voice_router, prefix="/api/v1")
    return app


# Run with uvicorn docthinker.server.mobile_app:create_mobile_app --factory.
