"""Same-origin H5 shell for the invite gateway; no desktop app imports.

Run behind a trusted HTTPS proxy which preserves Host.  This module neither
opens a public port nor provisions accounts, tunnels or model workers.
"""
import os
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles


class InviteShell(StaticFiles):
    """Serve only build assets, never an arbitrary directory or SPA fallback."""

    def __init__(self, directory: Path):
        root = directory.expanduser()
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError("Invite web directory must be an absolute real H5 build directory")
        root = root.resolve()
        for name in ("index.html", "manifest.webmanifest", "service-worker.js"):
            item = root / name
            if not item.is_file() or item.is_symlink():
                raise ValueError("Build the H5 shell before starting the invite site")
        # Explicitly reject a source/repository directory, even if someone has
        # accidentally copied an index.html into it.
        if any((root / name).exists() for name in (".git", ".env", "pyproject.toml", "package.json")):
            raise ValueError("Do not serve a source directory as the invite site")
        super().__init__(directory=root, html=False, follow_symlink=False)

    async def get_response(self, path, scope):
        if path in ("", "."):
            return RedirectResponse("/index.html#/pages/invite/index", status_code=302,
                                    headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
        pieces = Path(path).parts
        if any(part.startswith(".") or part == ".." for part in pieces):
            raise HTTPException(404)
        allowed = path in {"index.html", "manifest.webmanifest", "pwa-register.js", "service-worker.js"}
        if path.startswith("assets/"):
            allowed = Path(path).suffix.lower() in {".js", ".css", ".woff", ".woff2", ".ttf", ".png", ".svg"}
        if path.startswith(("static/brand/", "static/original-brand/")):
            allowed = Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
        if not allowed:
            raise HTTPException(404)
        # Refuse symlinks even when their targets happen to remain in the build.
        current = Path(self.directory)
        for part in pieces:
            current /= part
            if current.is_symlink():
                raise HTTPException(404)
        response = await super().get_response(path, scope)
        response.headers.update({
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Permissions-Policy": "camera=(), geolocation=(), microphone=(self)",
            # uni-app inserts inline CSS; scripts remain local to this origin.
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "font-src 'self' data:; connect-src 'self'; media-src 'self' blob:; "
                "worker-src 'self'; object-src 'none'; base-uri 'self'; "
                "frame-ancestors 'none'; form-action 'self'",
        })
        return response


def create_invite_site(*, api_app=None, web_dir=None):
    """Mount an explicit built H5 directory after the allowlisted API routes."""
    directory = str(web_dir or os.getenv("DOCTHINKER_INVITE_WEB_DIR", "")).strip()
    if not directory:
        raise ValueError("DOCTHINKER_INVITE_WEB_DIR must point to the built H5 directory")
    shell = InviteShell(Path(directory))
    if api_app is None:
        from .invite_app import create_invite_app

        api_app = create_invite_app()
    api_app.mount("/", shell, name="invite-shell")
    return api_app
