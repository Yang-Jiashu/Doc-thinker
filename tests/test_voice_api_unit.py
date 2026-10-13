"""Voice transport tests use only in-memory audio, MockTransport and isolated env."""

import asyncio
import importlib
import io
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

with patch("dotenv.load_dotenv"):
    voice = importlib.import_module("docthinker.providers.voice")
    voice_api = importlib.import_module("docthinker.server.routers.voice")


TOKEN = "mobile-test-token-that-is-at-least-32-characters"
AUTH = {"Authorization": "Bearer " + TOKEN}
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x64" + bytes(32)
WAV = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + bytes(32)
M4A = (24).to_bytes(4, "big") + b"ftypM4A " + bytes(4) + b"isommp42" + bytes(16)
AAC = b"\xff\xf1\x50\x80\x01\x3f\xfc" + bytes(10)
WEBM = b"\x1aE\xdf\xa3\x87\x42\x82\x84webm" + bytes(20)


@pytest.fixture
def setup(monkeypatch):
    for key in list(voice.os.environ):
        if key.startswith(("VOICE_", "DOCTHINKER_VOICE_", "DOCTHINKER_MOBILE_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DOCTHINKER_MOBILE_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", TOKEN)
    # Deliberately unrelated LLM keys must never make a voice provider available.
    monkeypatch.setenv("LLM_API_KEY", "private-llm-key")
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-key")
    app = FastAPI()
    app.include_router(voice_api.router, prefix="/api/v1")
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/transcriptions"):
            return httpx.Response(200, json={"text": " 转录结果 "})
        return httpx.Response(200, content=MP3, headers={"content-type": "audio/mpeg"})

    def configure(provider_handler=None):
        monkeypatch.setenv("DOCTHINKER_VOICE_ENABLED", "true")
        for role in ("ASR", "TTS"):
            monkeypatch.setenv(f"VOICE_{role}_BASE_URL", "https://voice.example/v1")
            monkeypatch.setenv(f"VOICE_{role}_KEY", f"private-{role}-key")
            monkeypatch.setenv(f"VOICE_{role}_MODEL", f"test-{role}-model")
        monkeypatch.setenv("VOICE_TTS_VOICE", "test-voice")
        transport = httpx.MockTransport(provider_handler or handler)
        monkeypatch.setattr(voice, "_make_client", lambda: httpx.AsyncClient(transport=transport, follow_redirects=False))

    def no_network():
        raise AssertionError("An unconfigured/invalid request tried to contact a provider")

    monkeypatch.setattr(voice, "_make_client", no_network)
    with TestClient(app) as client:
        yield app, client, configure, calls, monkeypatch


def test_disabled_capabilities_and_no_implicit_llm_credentials(setup):
    _, client, _, _, monkeypatch = setup
    result = client.get("/api/v1/mobile/voice/capabilities", headers=AUTH)
    assert result.status_code == 200
    assert result.json()["enabled"] is False
    assert result.json()["asr"] == {"available": False}
    assert "private" not in result.text
    assert client.post("/api/v1/mobile/voice/speak", headers=AUTH, json={"text": "hello"}).status_code == 503
    monkeypatch.setenv("DOCTHINKER_VOICE_ENABLED", "true")
    assert client.get("/api/v1/mobile/voice/capabilities", headers=AUTH).json()["tts"]["available"] is False
    assert client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("x.mp3", MP3, "audio/mpeg")}).status_code == 503


@pytest.mark.parametrize("role", ["ASR", "TTS"])
def test_roles_configure_independently_and_capabilities_reveal_no_secrets(setup, role):
    _, client, configure, _, monkeypatch = setup
    configure()
    monkeypatch.delenv(f"VOICE_{role}_MODEL")
    response = client.get("/api/v1/mobile/voice/capabilities", headers=AUTH)
    assert response.json()[role.lower()]["available"] is False
    assert response.json()["tts" if role == "ASR" else "asr"]["available"] is True
    assert "private" not in response.text and "voice.example" not in response.text
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("data,mime,extension", [
    (MP3, "audio/mpeg", "mp3"), (WAV, "audio/wav", "wav"),
    (M4A, "audio/mp4", "m4a"), (AAC, "audio/aac", "aac"),
    (WEBM, "audio/webm;codecs=opus", "webm"), (MP3, "application/octet-stream", "mp3"),
])
def test_transcribe_uses_bytes_not_user_filename_and_sends_only_asr_key(setup, data, mime, extension):
    _, client, configure, calls, _ = setup
    configure()
    response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("private-user-name.exe", data, mime)})
    assert response.status_code == 200 and response.json() == {"text": "转录结果"}
    request = calls[0]
    assert str(request.url) == "https://voice.example/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer private-ASR-key"
    assert f'recording.{extension}'.encode() in request.content
    assert b"private-user-name" not in request.content
    assert b"test-ASR-model" in request.content and b"response_format" in request.content


@pytest.mark.parametrize("data,mime,status", [
    (b"", "audio/mpeg", 400), (b"not audio", "audio/mpeg", 415),
    (MP3, "audio/wav", 415), (MP3, "text/html", 415),
    (b"ID3", "audio/mpeg", 415), (b"\x1aE\xdf\xa3garbage", "audio/webm", 415),
])
def test_invalid_audio_never_calls_provider(setup, data, mime, status):
    _, client, configure, calls, _ = setup
    configure()
    response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("x.mp3", data, mime)})
    assert response.status_code == status
    assert not calls


def test_file_limit_and_multipart_limit_are_distinct_and_enforced(setup):
    _, client, configure, calls, _ = setup
    configure()
    oversized = MP3 + bytes(voice.MAX_AUDIO_BYTES + 1 - len(MP3))
    response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("x.mp3", oversized, "audio/mpeg")})
    assert response.status_code == 413 and not calls
    response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files=[
        ("file", ("a.mp3", MP3, "audio/mpeg")), ("other", ("b.mp3", MP3, "audio/mpeg")),
    ])
    assert response.status_code == 400 and not calls


def test_speech_success_is_mp3_nonpersistent_and_separate_tts_credentials(setup):
    _, client, configure, calls, _ = setup
    configure()
    response = client.post("/api/v1/mobile/voice/speak", headers=AUTH, json={"text": "  请朗读  "})
    assert response.status_code == 200 and response.content == MP3
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["cache-control"] == "no-store"
    assert calls[0].headers["authorization"] == "Bearer private-TTS-key"
    assert json.loads(calls[0].content) == {
        "model": "test-TTS-model", "voice": "test-voice", "input": "请朗读", "response_format": "mp3",
    }


@pytest.mark.parametrize("payload", [{"text": "x" * 1501}, {"text": "   "}, {"text": 123}, {"text": "ok", "api_key": "must-not-echo"}, {}])
def test_invalid_speech_input_is_bounded_and_redacted(setup, payload):
    _, client, configure, calls, _ = setup
    configure()
    response = client.post("/api/v1/mobile/voice/speak", headers=AUTH, json=payload)
    assert response.status_code == 422 and not calls
    assert "must-not-echo" not in response.text and "xxx" not in response.text


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_provider_errors_and_redirects_do_not_echo_secrets_or_retry(setup, status):
    _, client, configure, _, _ = setup
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private-TTS-key original provider body", headers={"location": "https://leak.example"})
    configure(handler)
    response = client.post("/api/v1/mobile/voice/speak", headers=AUTH, json={"text": "private-user-speech"})
    assert response.status_code == 502 and len(calls) == 1
    assert all(secret not in response.text for secret in ["private-TTS-key", "original provider body", "private-user-speech", "leak.example"])


@pytest.mark.parametrize("error,expected", [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)])
def test_provider_transport_errors_are_generic(setup, error, expected):
    _, client, configure, _, _ = setup
    def handler(request):
        raise error("private-ASR-key", request=request)
    configure(handler)
    response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("x", MP3, "audio/mpeg")})
    assert response.status_code == expected and "private" not in response.text


@pytest.mark.parametrize("body", [b"not json private", b"[]", b'{"text":42}', b'{"text":"' + b"x" * 20001 + b'"}', b"[" * 2000 + b"]" * 2000])
def test_bad_transcription_shapes_are_not_returned(setup, body):
    _, client, configure, _, _ = setup
    configure(lambda _: httpx.Response(200, content=body))
    response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("x", MP3, "audio/mpeg")})
    assert response.status_code == 502 and "private" not in response.text


@pytest.mark.parametrize("body,mime", [(b"private response", "audio/mpeg"), (MP3, "text/html"), (b"", "audio/mpeg")])
def test_speech_must_actually_have_an_mp3_signature(setup, body, mime):
    _, client, configure, _, _ = setup
    configure(lambda _: httpx.Response(200, content=body, headers={"content-type": mime}))
    response = client.post("/api/v1/mobile/voice/speak", headers=AUTH, json={"text": "hello"})
    assert response.status_code == 502 and "private response" not in response.text


@pytest.mark.asyncio
async def test_auth_runs_before_reading_any_body(setup):
    app, _, _, _, _ = setup
    read = False
    async def receive():
        nonlocal read
        read = True
        raise AssertionError("Unauthenticated recording body was read")
    events = []
    async def send(event):
        events.append(event)
    scope = {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/api/v1/mobile/voice/transcribe", "raw_path": b"/api/v1/mobile/voice/transcribe",
             "query_string": b"", "headers": [], "server": ("test", 80), "client": ("test", 1), "root_path": ""}
    await app(scope, receive, send)
    assert not read
    assert next(event["status"] for event in events if event["type"] == "http.response.start") == 401


@pytest.mark.asyncio
async def test_chunked_upload_limit_does_not_trust_content_length(setup):
    app, _, _, _, monkeypatch = setup
    monkeypatch.setattr(voice_api, "MAX_MULTIPART_BYTES", 100)
    async def stream():
        yield b"x" * 60
        yield b"x" * 60
        raise AssertionError("Read beyond bounded request")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, content=stream())
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_file_is_closed_even_when_disabled(setup):
    file = UploadFile(io.BytesIO(MP3), filename="x.mp3", headers=Headers({"content-type": "audio/mpeg"}))
    with pytest.raises(voice_api.HTTPException) as error:
        await voice_api.transcribe(file)
    assert error.value.status_code == 503 and file.file.closed


@pytest.mark.asyncio
async def test_shared_concurrency_limit_and_cancellation_release_slots(setup):
    _, _, configure, _, _ = setup
    entered = asyncio.Event()
    async def handler(_):
        entered.set()
        await asyncio.Event().wait()
    configure(handler)
    service = voice.VoiceService()
    first = asyncio.create_task(service.speak("one"))
    second = asyncio.create_task(service.speak("two"))
    await entered.wait()
    await asyncio.sleep(0)
    with pytest.raises(voice.VoiceError) as error:
        await service.speak("third")
    assert error.value.status_code == 429
    first.cancel(); second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    configure()
    assert await voice.VoiceService().speak("slot released") == MP3


@pytest.mark.parametrize("base", ["", "file:///tmp", "https://user:secret@voice.example/v1", "https://voice.example/v1?key=secret", "https://voice.example:99999/v1", "https://{workspace}/v1"])
def test_invalid_config_fails_closed_without_inheriting_credentials(base):
    config = voice.load_voice_config({"DOCTHINKER_VOICE_ENABLED": "true", "VOICE_ASR_BASE_URL": base, "VOICE_ASR_MODEL": "asr", "VOICE_ASR_KEY": "voice-key", "OPENAI_API_KEY": "other-key"})
    assert config.asr is None


@pytest.mark.parametrize("base", ["http://voice.example/v1", "http://192.168.1.5:8101/v1", "http://localhost.evil.example/v1"])
@pytest.mark.parametrize("role", ["ASR", "TTS"])
def test_remote_plaintext_endpoints_are_unavailable_even_with_explicit_keys(base, role):
    config = voice.load_voice_config({
        "DOCTHINKER_VOICE_ENABLED": "true",
        f"VOICE_{role}_BASE_URL": base,
        f"VOICE_{role}_MODEL": "model",
        f"VOICE_{role}_KEY": "dedicated-key",
        f"VOICE_{role}_VOICE": "alloy",
    })
    assert getattr(config, role.lower()) is None
    assert voice.capabilities(config)[role.lower()]["available"] is False


@pytest.mark.parametrize("base", ["http://127.0.0.1:8101/v1", "http://localhost:8101/v1", "http://[::1]:8101/v1"])
def test_explicit_loopback_compatible_service_can_run_without_a_provider_key(base):
    config = voice.load_voice_config({"DOCTHINKER_VOICE_ENABLED": "true", "VOICE_ASR_BASE_URL": base, "VOICE_ASR_MODEL": "local-asr"})
    assert config.asr is not None and config.asr.key == ""
    assert config.tts is None


@pytest.mark.parametrize("operation", ["speak", "transcribe"])
def test_large_provider_responses_are_bounded_before_decoding(setup, operation):
    _, client, configure, _, monkeypatch = setup
    monkeypatch.setattr(voice, "MAX_AUDIO_BYTES", 64)
    monkeypatch.setattr(voice, "MAX_ASR_RESPONSE_BYTES", 64)
    configure(lambda _: httpx.Response(200, content=MP3 + bytes(100), headers={"content-type": "audio/mpeg"}))
    if operation == "speak":
        response = client.post("/api/v1/mobile/voice/speak", headers=AUTH, json={"text": "hello"})
    else:
        response = client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, files={"file": ("x", MP3, "audio/mpeg")})
    assert response.status_code == 502
    assert "size limit" in response.text


@pytest.mark.asyncio
async def test_total_deadline_cancels_waiting_provider_and_releases_capacity(setup):
    _, _, configure, _, monkeypatch = setup
    monkeypatch.setattr(voice, "PROVIDER_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()
    async def handler(_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    configure(handler)
    with pytest.raises(voice.VoiceError) as error:
        await voice.VoiceService().speak("hello")
    assert error.value.status_code == 504 and cancelled.is_set()
    configure()
    assert await voice.VoiceService().speak("next call") == MP3


@pytest.mark.asyncio
async def test_slow_inbound_body_is_timed_out_without_calling_provider(setup):
    app, _, _, _, monkeypatch = setup
    monkeypatch.setattr(voice_api, "UPLOAD_TIMEOUT_SECONDS", 0.01)
    async def stream():
        await asyncio.sleep(1)
        yield b"late"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/mobile/voice/transcribe", headers=AUTH, content=stream())
    assert response.status_code == 408


def test_provider_endpoint_repr_hides_credentials():
    assert "private-key" not in repr(voice.VoiceEndpoint("https://voice.example/v1", "model", "private-key"))


@pytest.mark.asyncio
async def test_provider_cannot_turn_a_swallowed_cancellation_into_late_success(setup):
    _, _, configure, _, monkeypatch = setup
    monkeypatch.setattr(voice, "PROVIDER_TIMEOUT_SECONDS", 0.01)
    async def handler(_):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return httpx.Response(200, content=MP3, headers={"content-type": "audio/mpeg"})
    configure(handler)
    with pytest.raises(voice.VoiceError) as error:
        await voice.VoiceService().speak("must not be accepted late")
    assert error.value.status_code == 504
