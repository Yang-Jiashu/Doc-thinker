#!/usr/bin/env python3
"""Read-only restricted-beta launch report; no network, Docker or model calls.

The report separates checks of local bytes from operator declarations. It never
certifies a public launch, enterprise security, legal compliance or readiness
of services which have not actually been exercised on the target host.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from urllib.parse import urlsplit

if "manage" not in globals():
    if __package__:
        from . import manage
    else:
        import manage

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS_FILE = "launch-readiness.json"
MAX_ASSETS = 512
MAX_BUILD_BYTES = 10 * 1024 * 1024
_SETTINGS = {"origin", "api_base", "model", "sharing_authorized"}
_ATTESTATIONS = {
    "provider_sharing_permission", "docker_runtime", "public_https_mobile",
    "two_user_isolation", "backup_restore", "provider_hard_spend_limit",
}
_OPERATIONS = {
    "format", "profile", "worker_ports", "privacy_notice_url", "terms_url",
    "support_email", "deletion_instructions_url", "retention_days",
    "backup_retention_days", "attestations",
}


def _public_https(value) -> bool:
    if not isinstance(value, str) or len(value) > 2048:
        return False
    try:
        url = urlsplit(value)
        host = url.hostname or ""
        if (url.scheme != "https" or url.username is not None or url.password is not None
                or url.query or url.fragment or any(ord(c) <= 32 or c in "\\{}" for c in value)
                or not re.fullmatch(r"[a-z0-9.-]+", host)
                or "." not in host or url.port not in {None, 443}
                or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split("."))
                or host.endswith((".test", ".invalid", ".example", ".local", ".localhost"))
                or host.endswith((".example.com", ".example.org", ".example.net"))
                or host in {"localhost", "example.com", "example.org", "example.net"}):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return True
        return False  # Require an owned DNS hostname, not an IP-address claim.
    except ValueError:
        return False


def _secret_metadata(directory: Path) -> bool:
    """Use stat only: checking permission must not read a provider credential."""
    try:
        with manage._directory_fd(directory) as fd:
            info = os.stat(manage.KEY_FILE, dir_fd=fd, follow_symlinks=False)
        return (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1
                and 0 < info.st_size <= 4096)
    except (OSError, manage.ConfigurationError):
        return False


def _json_file(directory: Path, name: str) -> dict:
    with manage._directory_fd(directory) as fd:
        body = manage._read_file(fd, name, limit=16384)
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError
    return value


def _asset_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or len(value) > 300:
        raise ValueError
    relative = PurePosixPath(value)
    if (relative.is_absolute() or str(relative) != value or not relative.parts
            or any(p in {".", ".."} or p.startswith(".") for p in relative.parts)
            or "\\" in value or any(ord(c) <= 32 for c in value)):
        raise ValueError
    allowed = value in {"index.html", "manifest.webmanifest", "pwa-register.js"}
    allowed |= value.startswith("assets/") and relative.suffix in {".js", ".css", ".woff", ".woff2", ".ttf", ".png", ".svg"}
    allowed |= value.startswith("static/brand/") and relative.suffix in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    path = root / value
    if not allowed or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError
    return path


class _EntryResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attribute = "href" if tag == "link" else "src" if tag in {"script", "img"} else None
        if attribute:
            for name, value in attrs:
                if name == attribute:
                    self.urls.append(value)


def _resource_in_manifest(value, seen):
    if not isinstance(value, str):
        raise ValueError
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError
    path = parsed.path
    path = path[2:] if path.startswith("./") else path[1:] if path.startswith("/") else path
    if path not in seen:
        raise ValueError


def build_integrity(directory: Path) -> dict:
    """Verify the existing PWA SHA256 allowlist without executing JavaScript."""
    directory = directory.expanduser()
    if (not directory.is_absolute() or not directory.is_dir()
            or directory.resolve(strict=True) != directory):
        raise ValueError
    # No stale hidden files, links, source maps or source/repository directories.
    count = total = all_count = all_bytes = unserved = 0
    for base, directories, files in os.walk(directory, followlinks=False):
        for name in (*directories, *files):
            path = Path(base) / name
            if name.startswith(".") or path.is_symlink():
                raise ValueError
        for name in files:
            path = Path(base) / name
            if not path.is_file():
                raise ValueError
            all_count += 1
            all_bytes += path.stat().st_size
            if all_count > MAX_ASSETS or all_bytes > MAX_BUILD_BYTES:
                raise ValueError
            relative = path.relative_to(directory).as_posix()
            # uni's raw build also contains a design showcase and a catalogue.
            # InviteShell deliberately does not serve these; Docker omits the
            # showcase. Keep its effective public surface, not a dev directory,
            # as the integrity contract. Still reject hidden files/links above.
            if relative.startswith("brand/") or relative == "static/brand/catalog.json":
                unserved += 1
                continue
            count += 1
            total += path.stat().st_size
            if relative != "service-worker.js":
                _asset_path(directory, relative)
    worker = directory / "service-worker.js"
    if worker.stat().st_size > 2 * 1024 * 1024:
        raise ValueError
    text = worker.read_text(encoding="utf-8")
    match = re.search(r"^const FILES = (\[.*\]);$", text, re.MULTILINE)
    files = json.loads(match.group(1)) if match else None
    if not isinstance(files, list) or not 3 <= len(files) <= MAX_ASSETS:
        raise ValueError
    seen = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "integrity"}:
            raise ValueError
        name, integrity = item["path"], item["integrity"]
        path = _asset_path(directory, name)
        if name in seen or not isinstance(integrity, str) or not integrity.startswith("sha256-"):
            raise ValueError
        expected = base64.b64decode(integrity[7:], validate=True)
        if len(expected) != 32 or hashlib.sha256(path.read_bytes()).digest() != expected:
            raise ValueError
        seen.add(name)
    if (not {"index.html", "manifest.webmanifest", "pwa-register.js"}.issubset(seen)
            or count != len(seen) + 1):
        raise ValueError
    entry = _EntryResources()
    entry.feed((directory / "index.html").read_text(encoding="utf-8"))
    for value in entry.urls:
        _resource_in_manifest(value, seen)
    manifest = json.loads((directory / "manifest.webmanifest").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("icons", []), list):
        raise ValueError
    for item in manifest.get("icons", []):
        if not isinstance(item, dict):
            raise ValueError
        _resource_in_manifest(item.get("src"), seen)
    return {"asset_count": count, "bytes": total, "unserved_build_files": unserved,
            "build_sha256": hashlib.sha256(worker.read_bytes()).hexdigest()}


def _capacity_contract(root: Path) -> dict:
    """Inspect constants as syntax; importing server modules is not necessary."""
    wanted = {"docthinker/server/invite_store.py": {"MAX_USERS": 5, "QUERY_DAILY_LIMIT": 50,
                                                "GLOBAL_QUERY_DAILY_LIMIT": 250},
              "docthinker/server/invite_workers.py": {"MAX_WORKERS": 5}}
    for filename, expected in wanted.items():
        source = root / filename
        if source.is_symlink() or source.stat().st_size > 128 * 1024:
            raise ValueError
        tree = ast.parse(source.read_text(encoding="utf-8"))
        values = {}
        for item in tree.body:
            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = item.value.value
        if any(type(values.get(k)) is not int or values[k] != v for k, v in expected.items()):
            raise ValueError
    return {"max_users": 5, "queries_per_user_per_utc_day": 50,
            "max_daily_requests": 250, "hard_cost_cap_verified": False}


def launch_report(config_dir: Path, web_dir: Path, *, root=ROOT) -> dict:
    checks = []

    def add(name, status, reason, **detail):
        checks.append({"id": name, "status": status, "reason": reason, **detail})

    configuration = operations = None
    try:
        directory = manage.config_directory(config_dir)
        configuration = _json_file(directory, manage.SETTINGS_FILE)
        if (set(configuration) != _SETTINGS or type(configuration["sharing_authorized"]) is not bool):
            raise ValueError
        manage._url(configuration["origin"], origin=True)
        manage._url(configuration["api_base"])
        manage._model(configuration["model"])
        add("private_configuration", "passed", "Owner-only settings are valid; no model credential was read.")
        safe = _secret_metadata(directory)
        add("secret_metadata", "passed" if safe else "blocked",
            "Source secret must be an owner-only 0600 regular single-link file in a 0700 directory; contents are not checked.")
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        configuration = None
        add("private_configuration", "blocked", "Run manage.py setup with a new private directory; existing files are never overwritten.")
    secure = configuration is not None and _public_https(configuration.get("origin"))
    add("public_https_origin", "passed" if secure else "blocked",
        "Configure the final HTTPS DNS origin; this check does not resolve DNS or verify a certificate.")
    sharing = configuration is not None and configuration.get("sharing_authorized") is True
    add("sharing_acknowledgement", "declared" if sharing else "pending",
        "An operator acknowledgement is not provider permission; subscription sharing needs explicit approval.")
    try:
        add("static_build_integrity", "passed", "All gateway-serveable static bytes match the build-time SHA256 allowlist; unserved design files are counted separately.", **build_integrity(web_dir))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        add("static_build_integrity", "blocked", "Build H5, then check the complete build directory; no links, hidden files or extra source files are allowed.")
    try:
        add("bounded_beta_capacity", "passed", "The source contract remains a maximum five-user chat/history beta, not an enterprise service.", **_capacity_contract(root))
    except (OSError, ValueError, SyntaxError, UnicodeError):
        add("bounded_beta_capacity", "blocked", "The fixed beta capacity changed; update and independently verify the deployment contract.")
    try:
        operations = _json_file(directory, OPERATIONS_FILE)
        if (set(operations) != _OPERATIONS or type(operations["format"]) is not int or operations["format"] != 1
                or operations["profile"] != "invite-chat-beta"
                or not isinstance(operations["attestations"], dict)
                or set(operations["attestations"]) != _ATTESTATIONS
                or any(type(v) is not bool for v in operations["attestations"].values())):
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError, UnboundLocalError, UnicodeError):
        operations = None
        add("operations_record", "pending", "Create owner-only launch-readiness.json from the documented template; do not include credentials or conversations.")
    ports = operations.get("worker_ports") if operations else None
    fixed = (isinstance(ports, list) and 1 <= len(ports) <= 5
             and all(type(p) is int and 8101 <= p <= 8105 for p in ports)
             and len(set(ports)) == len(ports))
    add("fixed_worker_plan", "declared" if fixed else "pending",
        "Declare one to five distinct worker ports 8101–8105; actual account bindings still require target-host check and two-user isolation tests.")
    policy = operations is not None and all(_public_https(operations.get(k)) for k in (
        "privacy_notice_url", "terms_url", "deletion_instructions_url"))
    email = operations.get("support_email") if operations else None
    contact = (isinstance(email, str) and len(email) <= 254
               and re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email)
               and not email.endswith(("@example.com", "@example.org", ".test", ".invalid")))
    retention = operations is not None and all(
        type(operations.get(k)) is int and 1 <= operations[k] <= 365
        for k in ("retention_days", "backup_retention_days"))
    add("published_user_policies", "declared" if policy and contact and retention else "pending",
        "Declare reachable privacy/terms/deletion pages, a support email, and bounded chat/backup retention. This does not verify their contents or implement deletion.")
    for name in sorted(_ATTESTATIONS):
        value = operations and operations["attestations"][name]
        add(name, "declared" if value else "pending",
            "Operator declaration only; verify this acceptance item on the actual host and retain redacted evidence. No provider or public endpoint was tested by this command.")
    blocked = any(c["status"] in {"blocked", "pending"} for c in checks)
    return {"format": 1, "profile": "invite-chat-beta", "status": "blocked" if blocked else "review_required",
            "launch_ready": False, "network_requests": 0, "model_requests": 0,
            "secret_contents_read": False, "checks": checks,
            "next_step": "Resolve blocked/pending items, then perform independent target-host acceptance; declarations are not verified readiness."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=manage.default_config_dir())
    parser.add_argument("--web-dir", type=Path, default=ROOT / "clients/mobile/dist/build/h5")
    args = parser.parse_args(argv)
    report = launch_report(args.config_dir, args.web_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
