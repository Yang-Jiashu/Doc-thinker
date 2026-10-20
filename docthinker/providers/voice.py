"""Opt-in, separately configured OpenAI-compatible ASR and speech adapters.

No settings/.env loading, implicit vendor defaults, retries, logging, audio
storage, or reuse of the RAG model's credentials. Containers are sniffed, not
decoded: a matching signature does not guarantee a valid playable recording.
"""

import asyncio
import ipaddress
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 1500
MAX_TRANSCRIPT_CHARS = 20_000
MAX_ASR_RESPONSE_BYTES = 256 * 1024
PROVIDER_TIMEOUT_SECONDS = 30.0
MAX_CONCURRENT_REQUESTS = 2
_provider_slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

_MIMES = {
    "mp3": {"audio/mpeg", "audio/mp3", "audio/x-mp3"},
    "wav": {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"},
    "m4a": {"audio/mp4", "audio/m4a", "audio/x-m4a"},
    "aac": {"audio/aac", "audio/aacp", "audio/x-aac"},
    "webm": {"audio/webm", "video/webm"},
}
_CANONICAL_MIME = {
    "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
    "aac": "audio/aac", "webm": "audio/webm",
}


class VoiceError(Exception):
    """Only fixed, safe messages cross the HTTP boundary."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class VoiceEndpoint:
    base_url: str
    model: str
    key: str = field(default="", repr=False)
    voice: str = ""


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = False
    asr: VoiceEndpoint | None = None
    tts: VoiceEndpoint | None = None


def _endpoint(env: Mapping[str, str], role: str) -> VoiceEndpoint | None:
    prefix = f"VOICE_{role}_"
    base = env.get(prefix + "BASE_URL", "").strip().rstrip("/")
    model = env.get(prefix + "MODEL", "").strip()
    key = env.get(prefix + "KEY", "").strip()
    voice = env.get(prefix + "VOICE", "").strip() if role == "TTS" else ""
    if not base or not model or (role == "TTS" and not voice):
        return None
    try:
        url = urlsplit(base)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or url.port == 0 or any(c in base for c in "{}\\\r\n\t ")
                or url.path.endswith(("/audio/speech", "/audio/transcriptions"))):
            return None
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            local = url.hostname == "localhost"
        # Audio, spoken text and the dedicated credential must not travel in
        # plaintext to remote gateways. Loopback keeps local demos practical.
        if not local and (url.scheme != "https" or not key):
            return None
    except ValueError:
        return None
    if any(ord(c) < 32 or ord(c) > 126 for c in key) or len(key) > 4096:
        return None
    if len(model) > 200 or len(voice) > 200 or any(c in model + voice for c in "\r\n\0"):
        return None
    return VoiceEndpoint(base, model, key, voice)


def load_voice_config(env: Mapping[str, str] | None = None) -> VoiceConfig:
    env = os.environ if env is None else env
    enabled = env.get("DOCTHINKER_VOICE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    return VoiceConfig(enabled, _endpoint(env, "ASR"), _endpoint(env, "TTS"))


def capabilities(config: VoiceConfig | None = None) -> dict:
    config = config or load_voice_config()
    return {
        "enabled": config.enabled,
        "asr": {"available": config.enabled and config.asr is not None},
        "tts": {"available": config.enabled and config.tts is not None},
        "max_audio_bytes": MAX_AUDIO_BYTES,
        "max_text_chars": MAX_TEXT_CHARS,
        "accepted_formats": list(_MIMES),
    }


def detect_audio_format(data: bytes) -> str | None:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 16 and data[4:8] == b"ftyp":
        box_size = int.from_bytes(data[:4], "big")
        if 16 <= box_size <= len(data):
            brands = [data[i:i + 4] for i in range(8, min(box_size, 128), 4)]
            if any(brand in {b"M4A ", b"M4B ", b"isom", b"iso2", b"mp41", b"mp42"} for brand in brands):
                return "m4a"
    if len(data) >= 8 and data[:4] == b"\x1aE\xdf\xa3" and b"webm" in data[:4096]:
        return "webm"
    if len(data) >= 7 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0 and (data[2] >> 2) & 0xF != 0xF:
        return "aac"
    if len(data) >= 10 and data[:3] == b"ID3" and data[3] in {2, 3, 4} and all(value < 128 for value in data[6:10]):
        return "mp3"
    if (len(data) >= 4 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
            and data[1] & 0x06 and data[1] & 0x18 != 0x08
            and data[2] >> 4 not in {0, 15} and (data[2] >> 2) & 3 != 3):
        return "mp3"
    return None


def validate_audio(data: bytes, content_type: str | None) -> str:
    if not data:
        raise VoiceError(400, "Audio upload is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise VoiceError(413, "Audio upload exceeds 10 MiB")
    detected = detect_audio_format(data)
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    # Native mobile uploaders commonly use octet-stream; bytes still decide type.
    if not detected or (mime not in {"", "application/octet-stream"} and mime not in _MIMES[detected]):
        raise VoiceError(415, "Unsupported audio format or MIME/signature mismatch")
    return detected


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(PROVIDER_TIMEOUT_SECONDS, connect=10.0, pool=5.0),
        follow_redirects=False,
        trust_env=False,
    )


async def _request(endpoint: VoiceEndpoint, operation: str, max_bytes: int, **kwargs) -> tuple[bytes, str]:
    if not _provider_slots.acquire(blocking=False):
        raise VoiceError(429, "Voice service is busy; try again later")

    async def send() -> tuple[bytes, str]:
        headers = {"Authorization": f"Bearer {endpoint.key}"} if endpoint.key else {}
        async with (
            _make_client() as client,
            client.stream("POST", endpoint.base_url + "/audio/" + operation, headers=headers, **kwargs) as response,
        ):
            if not 200 <= response.status_code < 300:
                # Never read/return the error body or follow redirects with credentials.
                raise VoiceError(502, "Voice provider rejected the request")
            body = bytearray()
            async for chunk in response.aiter_bytes(64 * 1024):
                if len(body) + len(chunk) > max_bytes:
                    raise VoiceError(502, "Voice provider response exceeds the size limit")
                body.extend(chunk)
            return bytes(body), response.headers.get("content-type", "")

    try:
        started = time.monotonic()
        result = await asyncio.wait_for(send(), timeout=PROVIDER_TIMEOUT_SECONDS)
        if time.monotonic() - started >= PROVIDER_TIMEOUT_SECONDS:
            raise VoiceError(504, "Voice provider timed out")
        return result
    except (httpx.TimeoutException, asyncio.TimeoutError):
        raise VoiceError(504, "Voice provider timed out") from None
    except httpx.HTTPError:
        raise VoiceError(502, "Voice provider connection failed") from None
    finally:
        _provider_slots.release()


class VoiceService:
    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or load_voice_config()

    def require(self, role: str) -> VoiceEndpoint:
        if not self.config.enabled:
            raise VoiceError(503, "Voice service is disabled")
        endpoint = self.config.asr if role == "asr" else self.config.tts
        if endpoint is None:
            raise VoiceError(503, "Voice provider is not configured")
        return endpoint

    async def transcribe(self, data: bytes, content_type: str | None) -> str:
        endpoint = self.require("asr")
        format_name = validate_audio(data, content_type)
        body, _ = await _request(
            endpoint, "transcriptions", MAX_ASR_RESPONSE_BYTES,
            data={"model": endpoint.model, "response_format": "json"},
            files={"file": (f"recording.{format_name}", data, _CANONICAL_MIME[format_name])},
        )
        try:
            decoded = json.loads(body)
            text = decoded.get("text") if isinstance(decoded, dict) else None
            if not isinstance(text, str) or len(text) > MAX_TRANSCRIPT_CHARS:
                raise ValueError
        except (ValueError, UnicodeError, RecursionError):
            raise VoiceError(502, "Voice provider returned an invalid transcription") from None
        return text.strip()

    async def speak(self, text: str) -> bytes:
        endpoint = self.require("tts")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise VoiceError(422, "Speech text must contain 1 to 1500 characters")
        body, mime = await _request(
            endpoint, "speech", MAX_AUDIO_BYTES,
            json={"model": endpoint.model, "voice": endpoint.voice, "input": text.strip(), "response_format": "mp3"},
        )
        if (detect_audio_format(body) != "mp3"
                or mime.split(";", 1)[0].lower().strip() not in {"audio/mpeg", "audio/mp3", "application/octet-stream", ""}):
            raise VoiceError(502, "Voice provider returned invalid MP3 audio")
        return body
