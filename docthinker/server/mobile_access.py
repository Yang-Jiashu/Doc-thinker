"""Opt-in access for a single-owner development instance, NOT tenant auth."""
import asyncio
import hmac
import os
import threading

from fastapi import Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute


async def require_mobile_access(authorization: str | None = Header(default=None)) -> None:
    if os.getenv("DOCTHINKER_MOBILE_ENABLED", "").lower() not in {"1", "true"}:
        raise HTTPException(404, "Mobile development access is disabled")
    expected = os.getenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", "")
    if len(expected) < 32:
        raise HTTPException(503, "Configure a mobile access token of at least 32 characters")
    provided = (authorization or "").removeprefix("Bearer ")
    if not authorization or not authorization.startswith("Bearer ") or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(401, "Invalid mobile access token", headers={"WWW-Authenticate": "Bearer"})


_inbound_slots = threading.BoundedSemaphore(4)


class MobileRoute(APIRoute):
    """Authenticate and bound bytes BEFORE FastAPI parses multipart/JSON."""
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request: Request):
            await require_mobile_access(request.headers.get("authorization"))
            acquired = False
            try:
                if request.method == "POST":
                    acquired = _inbound_slots.acquire(blocking=False)
                    if not acquired:
                        raise HTTPException(429, "Mobile service is busy")
                    limit = 20 * 1024 * 1024 + 65536 if self.path.endswith("/upload") else 65536
                    length = request.headers.get("content-length")
                    if length is not None:
                        try:
                            length = int(length)
                        except ValueError:
                            raise HTTPException(400, "Invalid body size") from None
                        if length < 0 or length > limit:
                            raise HTTPException(413, "Request too large")

                    async def bounded_body():
                        chunks, total = [], 0
                        async for chunk in request.stream():
                            total += len(chunk)
                            if total > limit:
                                raise HTTPException(413, "Request too large")
                            chunks.append(chunk)
                        request._body = b"".join(chunks)

                    try:
                        await asyncio.wait_for(bounded_body(), 15)
                    except asyncio.TimeoutError:
                        raise HTTPException(408, "Upload timed out") from None
                response = await handler(request)
                response.headers["Cache-Control"] = "no-store"
                return response
            except RequestValidationError:
                raise HTTPException(422, "Invalid mobile request") from None
            finally:
                if acquired:
                    _inbound_slots.release()
        return guarded
