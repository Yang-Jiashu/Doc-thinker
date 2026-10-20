"""Read-only launch gates: synthetic files only, no provider or Docker access."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from deploy.invite import preflight

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "SYNTHETIC-SECRET-NOT-A-REAL-KEY"


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
    path.chmod(0o600)


def build(tmp_path):
    root = tmp_path / "h5"
    assets = {"index.html": b"<html></html>", "manifest.webmanifest": b"{}",
              "pwa-register.js": b"/* synthetic */", "assets/index.js": b"export {}"}
    entries = []
    for name, body in assets.items():
        write(root / name, body)
        entries.append({"path": name, "integrity": "sha256-" + base64.b64encode(hashlib.sha256(body).digest()).decode()})
    write(root / "service-worker.js", "const FILES = " + json.dumps(entries) + ";\n")
    return root


def configuration(tmp_path, **changes):
    root = tmp_path / "configuration"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    data = {"origin": "https://assistant.customer-domain.com", "api_base": "https://api.vendor-domain.com/v1",
            "model": "synthetic-model", "sharing_authorized": True, **changes}
    write(root / "settings.json", json.dumps(data))
    write(root / "llm-api-key", PRIVATE)
    return root


def operations(path, **changes):
    data = {"format": 1, "profile": "invite-chat-beta", "worker_ports": [8101, 8102],
            "privacy_notice_url": "https://assistant.customer-domain.com/privacy",
            "terms_url": "https://assistant.customer-domain.com/terms",
            "support_email": "support@customer-domain.com",
            "deletion_instructions_url": "https://assistant.customer-domain.com/delete",
            "retention_days": 30, "backup_retention_days": 14,
            "attestations": {name: True for name in preflight._ATTESTATIONS}, **changes}
    write(path / preflight.OPERATIONS_FILE, json.dumps(data))


def check(report, name):
    return next(item for item in report["checks"] if item["id"] == name)


@pytest.fixture(autouse=True)
def forbid_effects(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Preflight must not call a network, provider or container")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def test_report_of_missing_conditions_is_actionable_and_never_ready(tmp_path):
    report = preflight.launch_report(tmp_path / "missing", tmp_path / "unbuilt")
    assert report["status"] == "blocked" and report["launch_ready"] is False
    assert check(report, "private_configuration")["status"] == "blocked"
    assert check(report, "operations_record")["status"] == "pending"
    assert report["network_requests"] == report["model_requests"] == 0
    assert report["secret_contents_read"] is False


def test_complete_local_inputs_do_not_self_certify_launch(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    operations(config)
    web = build(tmp_path)
    original = preflight.manage._read_file
    def guarded(fd, name, **kwargs):
        assert name not in {"llm-api-key", "container-llm-key"}
        return original(fd, name, **kwargs)
    monkeypatch.setattr(preflight.manage, "_read_file", guarded)
    report = preflight.launch_report(config, web)
    assert report["status"] == "review_required"
    assert report["launch_ready"] is False
    assert check(report, "static_build_integrity")["asset_count"] == 5
    assert check(report, "secret_metadata")["status"] == "passed"
    assert check(report, "provider_sharing_permission")["status"] == "declared"
    body = json.dumps(report)
    assert PRIVATE not in body and "customer-domain.com" not in body and str(config) not in body
    assert check(report, "bounded_beta_capacity")["hard_cost_cap_verified"] is False


@pytest.mark.parametrize("value", ["http://127.0.0.1:8088", "https://127.0.0.1", "https://192.168.1.5",
    "https://host.local", "https://demo.example.com", "https://example.com", "https://a..com",
    "https://customer-domain.com:8443", "https://x.test", "https://user:SECRET@domain.com", "https://domain.com/?key=SECRET"])
def test_origin_cannot_pass_with_local_example_or_credential_url(value):
    # Subdomains of example.com are also reserved placeholders.
    assert not preflight._public_https(value)


@pytest.mark.parametrize("mode", [0o644, 0o777, 0o400])
def test_secret_metadata_rejects_unsafe_permissions_without_reading(tmp_path, mode):
    config = configuration(tmp_path)
    (config / "llm-api-key").chmod(mode)
    assert check(preflight.launch_report(config, build(tmp_path)), "secret_metadata")["status"] == "blocked"


@pytest.mark.parametrize("name", ["llm-api-key", "settings.json"])
def test_private_files_reject_symlinks(tmp_path, name):
    config = configuration(tmp_path)
    target = tmp_path / "outside"
    (config / name).rename(target)
    (config / name).symlink_to(target)
    report = preflight.launch_report(config, build(tmp_path))
    assert report["status"] == "blocked"
    assert PRIVATE not in json.dumps(report)


@pytest.mark.parametrize("kind", ["tamper", "missing", "extra", "hidden", "link", "duplicate", "traversal"])
def test_static_integrity_rejects_partial_tampered_or_unsafe_build(tmp_path, kind):
    web = build(tmp_path)
    if kind == "tamper":
        write(web / "assets/index.js", "tampered")
    elif kind == "missing":
        (web / "assets/index.js").unlink()
    elif kind == "extra":
        write(web / "assets/unlisted.js", "extra")
    elif kind == "hidden":
        write(web / ".env", PRIVATE)
    elif kind == "link":
        target = tmp_path / "private"
        (web / "assets/index.js").rename(target)
        (web / "assets/index.js").symlink_to(target)
    else:
        worker = web / "service-worker.js"
        entries = json.loads(worker.read_text().removeprefix("const FILES = ").strip()[:-1])
        entries.append(entries[0] if kind == "duplicate" else {"path": "../private.js", "integrity": entries[0]["integrity"]})
        write(worker, "const FILES = " + json.dumps(entries) + ";\n")
    with pytest.raises((ValueError, OSError)):
        preflight.build_integrity(web)


@pytest.mark.parametrize("changes", [{"worker_ports": []}, {"worker_ports": [8101, 8101]},
    {"worker_ports": [True]}, {"worker_ports": [8106]}, {"retention_days": None},
    {"privacy_notice_url": "http://unsafe.com/privacy"}, {"support_email": "support@example.com"},
    {"backup_retention_days": 10000}, {"unexpected_key": PRIVATE}])
def test_incomplete_or_invalid_operational_plan_stays_pending(tmp_path, changes):
    config = configuration(tmp_path)
    operations(config, **changes)
    report = preflight.launch_report(config, build(tmp_path))
    assert report["status"] == "blocked" and report["launch_ready"] is False
    assert PRIVATE not in json.dumps(report)


def test_capacity_is_source_checked_without_importing_model_modules(tmp_path):
    root = tmp_path / "source"
    write(root / "docthinker/server/invite_store.py", "MAX_USERS = 5\nQUERY_DAILY_LIMIT = 51\nGLOBAL_QUERY_DAILY_LIMIT = 250\n")
    write(root / "docthinker/server/invite_workers.py", "MAX_WORKERS = 5\n")
    report = preflight.launch_report(configuration(tmp_path), build(tmp_path), root=root)
    assert check(report, "bounded_beta_capacity")["status"] == "blocked"


@pytest.mark.parametrize("global_limit", ["", "GLOBAL_QUERY_DAILY_LIMIT = 251\n",
    "GLOBAL_QUERY_DAILY_LIMIT = True\n", "GLOBAL_QUERY_DAILY_LIMIT = '250'\n"])
def test_global_capacity_is_not_inferred_from_per_user_limits(tmp_path, global_limit):
    root = tmp_path / "source"
    write(root / "docthinker/server/invite_store.py", "MAX_USERS = 5\nQUERY_DAILY_LIMIT = 50\n" + global_limit)
    write(root / "docthinker/server/invite_workers.py", "MAX_WORKERS = 5\n")
    with pytest.raises(ValueError):
        preflight._capacity_contract(root)


def test_capacity_reports_only_verified_global_limit(tmp_path):
    root = tmp_path / "source"
    write(root / "docthinker/server/invite_store.py", "MAX_USERS = 5\nQUERY_DAILY_LIMIT = 50\nGLOBAL_QUERY_DAILY_LIMIT = 250\n")
    write(root / "docthinker/server/invite_workers.py", "MAX_WORKERS = 5\n")
    result = preflight._capacity_contract(root)
    assert result["max_daily_requests"] == 250
    assert result["hard_cost_cap_verified"] is False


def test_unserved_design_showcase_is_not_confused_with_gateway_assets(tmp_path):
    web = build(tmp_path)
    write(web / "brand/showcase.template.html", "design showcase")
    write(web / "static/brand/catalog.json", "{}")
    result = preflight.build_integrity(web)
    assert result["asset_count"] == 5 and result["unserved_build_files"] == 2
    write(web / "brand/.env", PRIVATE)
    with pytest.raises(ValueError):
        preflight.build_integrity(web)


@pytest.mark.parametrize("resource", ["/assets/missing.js", "https://outside-domain.com/model.js", "//outside-domain.com/model.js", "../private.js"])
def test_index_must_reference_only_present_same_origin_assets(tmp_path, resource):
    web = build(tmp_path)
    write(web / "index.html", '<script src="' + resource + '"></script>')
    worker = web / "service-worker.js"
    entries = json.loads(worker.read_text().removeprefix("const FILES = ").strip()[:-1])
    entries[0]["integrity"] = "sha256-" + base64.b64encode(hashlib.sha256((web / "index.html").read_bytes()).digest()).decode()
    write(worker, "const FILES = " + json.dumps(entries) + ";\n")
    with pytest.raises(ValueError):
        preflight.build_integrity(web)


def test_main_outputs_only_report_and_blocked_exit(tmp_path, capsys):
    assert preflight.main(["--config-dir", str(tmp_path / "missing"), "--web-dir", str(tmp_path / "missing-web")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked" and PRIVATE not in json.dumps(report)
