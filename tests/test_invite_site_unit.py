"""Static invite shell never serves secrets or turns failed APIs into HTML."""
import pytest
import httpx
from fastapi import FastAPI

from docthinker.server.invite_site import InviteShell, create_invite_site


@pytest.fixture
def shell(tmp_path):
    for name, body in {"index.html": "<h1>test shell</h1>", "manifest.webmanifest": "{}",
                       "service-worker.js": "// test", "pwa-register.js": "// test"}.items():
        (tmp_path / name).write_text(body)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/app.js").write_text("// static")
    (tmp_path / "assets/app.js.map").write_text("private-source-map")
    return tmp_path


async def test_invite_entry_redirects_to_login_and_serves_original_shell(shell):
    app = FastAPI()
    @app.get("/api/v1/invite/status")
    def status():
        return {"service": "docthinker-invite"}
    create_invite_site(api_app=app, web_dir=shell)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        response = await client.get("/")
        assert response.status_code == 302
        assert response.headers["location"] == "/index.html#/pages/invite/index"
        response = await client.get("/index.html")
        assert response.text == (shell / "index.html").read_text()
        assert response.headers["cache-control"] == "no-store"
        assert "connect-src 'self'" in response.headers["content-security-policy"]
        assert (await client.get("/api/v1/invite/status")).json()["service"] == "docthinker-invite"
        for path in ("/api/v1/settings", "/api/v1/mobile/admin", "/missing", "/.env", "/assets/app.js.map", "/README.md"):
            result = await client.get(path)
            assert result.status_code == 404
            assert "test shell" not in result.text
        assert (await client.get("/assets/app.js")).status_code == 200


async def test_shell_refuses_symlink_assets(shell):
    (shell / "assets/link.js").symlink_to(shell / "assets/app.js")
    app = create_invite_site(api_app=FastAPI(), web_dir=shell)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        assert (await client.get("/assets/link.js")).status_code == 404


async def test_active_original_brand_images_are_served_without_opening_source_directories(shell):
    original = shell / "static/original-brand"
    original.mkdir(parents=True)
    (original / "logo-64.png").write_bytes(b"synthetic-png")
    (original / "private.txt").write_text("not-a-build-image")
    (original / "linked.png").symlink_to(original / "logo-64.png")
    app = create_invite_site(api_app=FastAPI(), web_dir=shell)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        response = await client.get("/static/original-brand/logo-64.png")
        assert response.status_code == 200 and response.content == b"synthetic-png"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        for path in ("/static/original-brand/private.txt", "/static/original-brand/linked.png",
                     "/original-brand/private.txt", "/static/other/logo-64.png"):
            assert (await client.get(path)).status_code == 404


def test_source_and_missing_build_directories_are_rejected(shell, tmp_path):
    with pytest.raises(ValueError):
        InviteShell(tmp_path / "absent")
    with pytest.raises(ValueError):
        InviteShell(type(shell)("relative"))
    (shell / "package.json").write_text("{}")
    with pytest.raises(ValueError, match="source"):
        InviteShell(shell)


def test_web_directory_must_be_explicit(monkeypatch):
    monkeypatch.delenv("DOCTHINKER_INVITE_WEB_DIR", raising=False)
    with pytest.raises(ValueError, match="WEB_DIR"):
        create_invite_site(api_app=FastAPI())
