"""Frozen presentation assets survive integration without freezing new UI code."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_preservation", ROOT / "scripts/check_release_preservation.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.fixture
def release(tmp_path, monkeypatch):
    # Tiny independent fixture; the complete real manifest is checked below.
    files = {
        "logo.png": b"frozen logo",
        "docs/assets/demo.gif": b"frozen demo",
        "clients/mobile/src/static/brand/logo.png": b"frozen mobile identity",
    }
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    monkeypatch.setattr(CHECK, "EXPECTED_SHA256", {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    })
    monkeypatch.setattr(CHECK, "README_IMAGES", ("logo.png", "docs/assets/demo.gif"))
    body = ('<img src="logo.png" alt="logo" />\n'
            '![Original demo](docs/assets/demo.gif)\n'
            '[Current architecture](docs/ARCHITECTURE.md)\n')
    for name in CHECK.README_PATHS:
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_real_release_preserves_all_frozen_assets_and_bilingual_galleries():
    assert CHECK.LEGACY_FILE_COUNT == 76
    assert CHECK.MOBILE_BRAND_COUNT == 113
    assert len(CHECK.EXPECTED_SHA256) == 189
    assert CHECK.verify(ROOT) == (189, 2)
    assert "docthinker/ui/templates/query_modern.html" not in CHECK.EXPECTED_SHA256
    assert "docthinker/ui/promo_graph/frontend/src/gesture/gesture-controller.ts" not in CHECK.EXPECTED_SHA256
    assert {"logo.png", "scripts/brain_memory_demo.py", "scripts/memory_demo.py",
            "docthinker/memory_core/protocols.py"}.issubset(CHECK.EXPECTED_SHA256)
    assert len([name for name in CHECK.EXPECTED_SHA256
                if name.startswith("docs/assets/")]) == 11


def test_checker_works_without_git_checkout_or_network(release):
    assert not (release / ".git").exists()
    assert CHECK.verify(release) == (3, 2)


@pytest.mark.parametrize("name", [
    "logo.png", "docs/assets/demo.gif", "clients/mobile/src/static/brand/logo.png",
])
def test_changed_protected_bytes_fail_closed(release, name):
    (release / name).write_bytes(b"replacement")
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        CHECK.verify(release)


def test_missing_protected_file_fails_closed(release):
    (release / "logo.png").unlink()
    with pytest.raises(ValueError, match="logo"):
        CHECK.verify(release)


def test_symlink_to_identical_protected_content_is_rejected(release):
    target = release / "logo.png"
    twin = release / "same.png"
    target.rename(twin)
    target.symlink_to(twin)
    with pytest.raises(ValueError, match="logo"):
        CHECK.verify(release)


def test_symlinked_parent_of_protected_asset_is_rejected(release):
    assets = release / "docs/assets"
    moved = release / "moved-assets"
    assets.rename(moved)
    assets.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="demo"):
        CHECK.verify(release)


@pytest.mark.parametrize("name", CHECK.README_PATHS)
def test_both_readmes_must_retain_visible_original_demo(release, name):
    path = release / name
    body = path.read_text(encoding="utf-8")
    path.write_text(body.replace("![Original demo](docs/assets/demo.gif)", ""),
                    encoding="utf-8")
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        CHECK.verify(release)


@pytest.mark.parametrize("hidden", [
    "<!-- ![demo](docs/assets/demo.gif) -->",
    '```html\n<img src="docs/assets/demo.gif" />\n```',
    "~~~md\n![demo](docs/assets/demo.gif)\n~~~",
])
def test_comments_and_code_samples_cannot_replace_visible_demo(release, hidden):
    for name in CHECK.README_PATHS:
        path = release / name
        body = path.read_text(encoding="utf-8")
        path.write_text(body.replace("![Original demo](docs/assets/demo.gif)", hidden),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="original logo/demo/architecture"):
        CHECK.verify(release)


def test_readme_prose_and_new_ui_can_evolve_without_asset_replacement(release):
    for name in CHECK.README_PATHS:
        path = release / name
        path.write_text("New personal-agent release\n" + path.read_text(encoding="utf-8"),
                        encoding="utf-8")
    ui = release / "docthinker/ui/templates/query_modern.html"
    ui.parent.mkdir(parents=True)
    ui.write_text("Improved UI with the same protected identity", encoding="utf-8")
    assert CHECK.verify(release) == (3, 2)


def test_logo_remains_in_source_distribution_manifest():
    assert "include logo.png" in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
