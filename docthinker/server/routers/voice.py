"""Small, authenticated voice requests for the separate mobile application."""

import asyncio
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from docthinker.providers.voice import (
    MAX_AUDIO_BYTES,
    VoiceError,
    VoiceService,
    capabilities,
)

from ..mobile_access import require_mobile_access

MAX_MULTIPART_BYTES = MAX_AUDIO_BYTES + 64 * 1024
MAX_SPEAK_BODY_BYTES = 16 * 1024
UPLOAD_TIMEOUT_SECONDS = 15.0
_inbound_slots = threading.BoundedSemaphore(4)


class _VoiceRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request: Request):
            # FastAPI normally parses multipart before dependencies. Authenticate
            # first, before buffering/parsing any recording or text. The router's
            # normal dependency remains for its explicit access-control contract.
            await require_mobile_access(request.headers.get("authorization"))
            acquired = False
            form = None
            try:
                if request.method == "POST":
                    acquired = _inbound_slots.acquire(blocking=False)
                    if not acquired:
                        raise HTTPException(429, "Voice upload service is busy")
                    limit = MAX_MULTIPART_BYTES if self.path.endswith("/transcribe") else MAX_SPEAK_BODY_BYTES
                    length = request.headers.get("content-length")
                    if length is not None:
                        try:
                            size = int(length)
                        except ValueError:
                            raise HTTPException(400, "Invalid request size") from None
                        if size < 0:
                            raise HTTPException(400, "Invalid request size")
                        if size > limit:
                            raise HTTPException(413, "Voice request exceeds the size limit")

                    async def read_bounded():
                        chunks = []
                        total = 0
                        async for chunk in request.stream():
                            total += len(chunk)
                            if total > limit:
                                raise HTTPException(413, "Voice request exceeds the size limit")
                            chunks.append(chunk)
                        # Replay the bounded body through Starlette's ordinary
                        # JSON/multipart parser, including UploadFile handling.
                        request._body = b"".join(chunks)

                    try:
                        await asyncio.wait_for(read_bounded(), timeout=UPLOAD_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        raise HTTPException(408, "Voice upload timed out") from None
                    if self.path.endswith("/transcribe") and request.headers.get("content-type", "").startswith("multipart/form-data"):
                        try:
                            form = await request.form(max_files=1, max_fields=0)
                        except StarletteHTTPException:
                            raise HTTPException(400, "Invalid audio upload") from None
                response = await handler(request)
                response.headers["Cache-Control"] = "no-store"
                response.headers["X-Content-Type-Options"] = "nosniff"
                return response
            except RequestValidationError:
                raise HTTPException(422, "Invalid voice request payload", headers={"Cache-Control": "no-store"}) from None
            finally:
                try:
                    if form is not None:
                        await form.close()
                finally:
                    if acquired:
                        _inbound_slots.release()

        return guarded


router = APIRouter(
    prefix="/mobile/voice",
    dependencies=[Depends(require_mobile_access)],
    route_class=_VoiceRoute,
)


class SpeakRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1500, strict=True)


def _http_error(error: VoiceError) -> HTTPException:
    return HTTPException(error.status_code, str(error), headers={"Cache-Control": "no-store"})


@router.get("/capabilities")
async def voice_capabilities():
    return JSONResponse(capabilities())


@router.post("/transcribe")
async def transcribe(file: Annotated[UploadFile, File()]):
    try:
        service = VoiceService()
        service.require("asr")
        audio = await file.read(MAX_AUDIO_BYTES + 1)
        text = await service.transcribe(audio, file.content_type)
        return {"text": text}
    except VoiceError as error:
        raise _http_error(error) from None
    finally:
        await file.close()


@router.post("/speak")
async def speak(payload: SpeakRequest):
    try:
        audio = await VoiceService().speak(payload.text)
    except VoiceError as error:
        raise _http_error(error) from None
    return Response(audio, media_type="audio/mpeg", headers={
        "Cache-Control": "no-store",
        "Content-Disposition": 'inline; filename="speech.mp3"',
    })
