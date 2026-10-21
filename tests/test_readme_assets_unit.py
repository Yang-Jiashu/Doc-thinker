"""Offline visual-document checks; no Markdown renderer or browser required.

The checks protect valid links and self-contained artwork, not prose, layout or
marketing wording. Repository documentation links need not all ship in the
explicitly restricted invitation package; README and architecture images do.
"""
from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import struct
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
READMES = ("README.md", "README.zh-CN.md")
HISTORY = "docs/HISTORY.md"
OVERVIEW = "docs/assets/docthinker-v2-overview.svg"
ARCHITECTURE = "docs/ARCHITECTURE.md"
ARCHITECTURE_IMAGES = (
    "docs/assets/architecture-system.svg",
    "docs/assets/architecture-memory.svg",
    "docs/assets/architecture-task-loop.svg",
)
CURRENT_UI_GIFS = (
    "docs/assets/current-ui-workspace.gif",
    "docs/assets/current-ui-memory.gif",
    "docs/assets/current-ui-graph.gif",
    "docs/assets/current-ui-projects.gif",
)
UI_SHOWCASE = "docs/UI_SHOWCASE.md"
UI_MANIFEST = "docs/assets/current-ui-manifest.json"
UI_V2_GIFS = (
    "docs/assets/ui-v2-workspace.gif",
    "docs/assets/ui-v2-memory-observatory.gif",
    "docs/assets/ui-v2-graph-canvas.gif",
)
UI_V2_MANIFEST = "docs/assets/ui-v2-manifest.json"
CURRENT_UI_MODES = {
    path: ("offline-sample" if path.endswith(("workspace.gif", "graph.gif"))
           else "isolated-runtime-local-model-stub")
    for path in CURRENT_UI_GIFS
}
UI_V2_MODES = {path: "isolated-runtime-zero-model" for path in UI_V2_GIFS}
SVG_NS = "http://www.w3.org/2000/svg"
_DESTINATION = r'\]\(\s*(?:<([^>\n]+)>|([^\s)\n]+))'


class _HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value is not None and key in {"src", "href"}:
                self.references.append((value, tag == "img" and key == "src"))

    handle_startendtag = handle_starttag


class _InlineImageGroups(HTMLParser):
    """Keep paragraph rows separate; linked images retain their row identity."""
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.groups = []
        self.parents = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            # A new paragraph also ends a preceding implicitly closed <p>.
            self.current = []
            self.groups.append(self.current)
            self.parents = ["p"]
        elif self.current is not None:
            if tag == "br":
                self.current = []
                self.groups.append(self.current)
            elif tag == "img" and all(parent == "a" for parent in self.parents[1:]):
                source = dict(attrs).get("src")
                if source:
                    self.current.append(source)
            if tag not in self._VOID:
                self.parents.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.current is not None and tag in self.parents:
            last = len(self.parents) - 1 - self.parents[::-1].index(tag)
            self.parents = self.parents[:last]
            if tag == "p":
                self.current = None


def inline_image_groups(body: str) -> list[list[str]]:
    """Find actual inline HTML rows, not paragraph markup quoted in examples."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"(?ms)^\s*(?:```|~~~).*?^\s*(?:```|~~~)[^\n]*$", "", body)
    body = re.sub(r"`[^`\n]+`", "", body)
    parser = _InlineImageGroups()
    parser.feed(body)
    parser.close()
    return [group for group in parser.groups if group]


def raster_dimensions(path: Path) -> tuple[int, int]:
    """Read only the PNG IHDR or GIF logical-screen header, not image frames."""
    with path.open("rb") as stream:
        signature = stream.read(8)
        if signature == b"\x89PNG\r\n\x1a\n":
            header = signature + stream.read(16)  # 24 bytes total, including IHDR dimensions.
            if len(header) != 24 or header[8:16] != b"\x00\x00\x00\rIHDR":
                raise ValueError("Missing PNG dimensions: " + str(path))
            width, height = struct.unpack(">II", header[16:24])
        elif signature[:6] in {b"GIF87a", b"GIF89a"}:
            header = signature + stream.read(2)  # Exactly 10 bytes, even for a large animation.
            if len(header) != 10:
                raise ValueError("Missing GIF dimensions: " + str(path))
            width, height = struct.unpack("<HH", header[6:10])
        else:
            raise ValueError("Unsupported raster header: " + str(path))
    if not width or not height:
        raise ValueError("Invalid raster dimensions: " + str(path))
    return width, height


def validate_inline_proportions(root: Path, source: str, body: str) -> None:
    """At equal width, row heights must stay within 1.4x of each other.

    This guards intrinsic diagram proportions, not fixed pixel widths, prose,
    ordering or a particular screenshot. Single images and separate rows are
    independent; SVG and remote assets are not decoded or fetched here.
    """
    for group in inline_image_groups(body):
        heights = []
        for target in group:
            path = local_target(root, source, target)
            if path is not None and path.suffix.lower() in {".png", ".gif"}:
                width, height = raster_dimensions(path)
                heights.append(height / width)
        if len(heights) > 1 and max(heights) / min(heights) > 1.4:
            raise ValueError("Inline image proportions differ by more than 1.4x: " + ", ".join(group))


def validate_capture_manifest(
        value: dict, root: Path, *, inventory: tuple[str, ...] = CURRENT_UI_GIFS,
        modes: dict[str, str] | None = None) -> None:
    """Bind declared capture metadata to shipped GIF bytes and header sizes.

    Each recording generation has an explicit inventory and workflow modes;
    the defaults retain the previous four-recording contract unchanged.
    Frame counts, elapsed duration and source-video hashes remain recording
    metadata; this check does not decode animation or certify model quality.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def positive(value):
        return type(value) is int and value > 0

    def digest(value):
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    require(type(inventory) is tuple and len(inventory) > 0
            and all(type(path) is str and path.startswith("docs/assets/")
                    and "\\" not in path and not {"..", "."}.intersection(path.split("/"))
                    and path.endswith(".gif") for path in inventory)
            and len(set(inventory)) == len(inventory), "Invalid expected capture inventory")
    expected_modes = CURRENT_UI_MODES if modes is None else modes
    require(type(expected_modes) is dict and set(expected_modes) == set(inventory)
            and all(type(mode) is str and mode.strip() for mode in expected_modes.values()),
            "Expected capture modes must match the explicit inventory")
    require(type(value) is dict and set(value) == {
        "schema_version", "source_commit", "client_build", "external_model_calls", "captures"},
        "Unexpected capture manifest envelope")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Unknown capture manifest version")
    require(type(value["source_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]),
            "Missing capture source commit")
    require(type(value["external_model_calls"]) is int and value["external_model_calls"] == 0,
            "Showcase recordings must not claim a paid-model evaluation")
    build = value["client_build"]
    require(type(build) is dict and set(build) == {"pwa_version", "command"}
            and type(build["pwa_version"]) is str and re.fullmatch(r"[0-9a-f]{20}", build["pwa_version"])
            and type(build["command"]) is str and 0 < len(build["command"].strip()) <= 240,
            "Missing client build identity")
    captures = value["captures"]
    require(type(captures) is list and len(captures) == len(inventory),
            "Capture inventory size differs from the expected recordings")
    fields = {"path", "mode", "viewport", "width", "height", "frames", "duration_ms", "sha256",
              "source_video_sha256", "actions"}
    require(all(type(row) is dict and set(row) == fields and type(row["path"]) is str for row in captures),
            "Unexpected capture metadata fields")
    require({row["path"] for row in captures} == set(inventory),
            "Recording paths must be unique and match the shipped inventory")
    for row in captures:
        path = row["path"]
        require(row["mode"] == expected_modes[path], "Capture mode differs from its recorded workflow")
        viewport = row["viewport"]
        require(type(viewport) is dict and set(viewport) == {"width", "height"}
                and all(positive(viewport[key]) for key in viewport), "Invalid capture viewport")
        require(all(positive(row[key]) for key in ("width", "height", "frames", "duration_ms"))
                and row["frames"] > 1, "Missing animation metadata")
        require(digest(row["sha256"]) and digest(row["source_video_sha256"]), "Invalid capture hash")
        require(type(row["actions"]) is list and 0 < len(row["actions"]) <= 64
                and all(type(action) is str and action.strip() and len(action) <= 2000
                        for action in row["actions"]), "Missing recorded actions")
        asset = root / path
        require(asset.is_file() and not asset.is_symlink() and asset.resolve() == asset,
                "Capture asset must be a regular file")
        require(0 < asset.stat().st_size <= 10 * 1024 * 1024, "Recording exceeds normal input budget")
        actual_hash = hashlib.sha256()
        with asset.open("rb") as stream:
            require(stream.read(6) in {b"GIF87a", b"GIF89a"}, "Recording must be a GIF")
            stream.seek(0)
            for block in iter(lambda: stream.read(65536), b""):
                actual_hash.update(block)
        require(actual_hash.hexdigest() == row["sha256"], "Recording bytes differ from capture manifest")
        require(raster_dimensions(asset) == (row["width"], row["height"]),
                "Recording dimensions differ from capture manifest")


def readme_references(body: str) -> list[tuple[str, bool]]:
    """Collect inline Markdown targets and HTML references outside examples."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"(?ms)^\s*(?:```|~~~).*?^\s*(?:```|~~~)[^\n]*$", "", body)
    body = re.sub(r"`[^`\n]+`", "", body)
    targets = [(first or second, False)
               for first, second in re.findall(_DESTINATION, body)]
    targets.extend((first or second, True) for first, second in re.findall(
        r"!\[[^\]\n]*" + _DESTINATION, body))
    parser = _HTMLReferences()
    parser.feed(body)
    targets.extend(parser.references)
    return targets


def local_target(root: Path, source: str, target: str) -> Path | None:
    """Resolve repo-relative paths, ignoring remote URLs and same-page anchors.

    Fragments and queries do not become filesystem names. Percent-encoded local
    filenames are decoded; a path cannot escape the repository.
    """
    value = urlsplit(unescape(target))
    if value.scheme or value.netloc or not value.path:
        return None
    relative = unquote(value.path)
    resolved = ((root / relative.lstrip("/")) if relative.startswith("/")
                else (root / source).parent / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("README target escapes repository: " + target)
    return resolved


def _pack_contract():
    spec = importlib.util.spec_from_file_location("readme_invite_pack", ROOT / "deploy/invite/pack.py")
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    return pack


def validate_svg(source: str) -> None:
    """Require accessible SVG with only in-document resource references."""
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", source, re.I):
        raise ValueError("SVG must not load declarations or entities")
    if re.search(r"<\?(?!xml\s)[\w:-]+", source, re.I):
        raise ValueError("SVG must not load processing instructions")
    root = ET.fromstring(source)
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("Not an SVG document")
    for name in ("title", "desc"):
        entry = root.find(f"{{{SVG_NS}}}{name}")
        if entry is None or not "".join(entry.itertext()).strip():
            raise ValueError("SVG requires a nonempty " + name)
    ids = {node.attrib["id"] for node in root.iter() if "id" in node.attrib}
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag in {"script", "foreignobject"}:
            raise ValueError("SVG must not contain active embedded content")
        for key, value in node.attrib.items():
            attribute = key.rsplit("}", 1)[-1].lower()
            if attribute.startswith("on"):
                raise ValueError("SVG must not contain event handlers")
            if attribute in {"href", "src"} and (
                    not value.startswith("#") or value[1:] not in ids):
                raise ValueError("SVG references must stay inside the document")
    if re.search(r"@(?:import|font-face)\b", source, re.I):
        raise ValueError("SVG must not import external styles or fonts")
    for match in re.finditer(r"url\(\s*(['\"]?)([^)]+?)\1\s*\)", source, re.I):
        target = match.group(2).strip()
        if not target.startswith("#") or target[1:] not in ids:
            raise ValueError("SVG URL must reference an existing internal ID")


class ReadmeAssetsTests(unittest.TestCase):
    def test_bilingual_inline_image_groups_have_compatible_proportions(self):
        for name in (*READMES, HISTORY):
            with self.subTest(readme=name):
                validate_inline_proportions(ROOT, name, (ROOT / name).read_text(encoding="utf-8"))

    def test_inline_groups_preserve_paragraphs_links_and_explicit_line_breaks(self):
        body = ('<p><a href="one.png"><img src="one.png"/></a><img\n src="two.gif"/></p>'
                '<p><img src="portrait.png"/></p><p><img src="wide.gif"/></p>'
                '<p><img src="top.png"/><br/><a href="bottom.gif"><img src="bottom.gif"/></a></p>'
                '<p><img src="implicit-one.png"><p><img src="implicit-two.gif"></p>'
                '<!-- <p><img src="hidden.png"></p> -->\n'
                '```html\n<p><img src="sample.png"></p>\n```\n')
        self.assertEqual(inline_image_groups(body), [
            ["one.png", "two.gif"], ["portrait.png"], ["wide.gif"],
            ["top.png"], ["bottom.gif"], ["implicit-one.png"], ["implicit-two.gif"],
        ])

    def test_raster_header_proportions_accept_similar_rows_and_reject_mixed_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files = {
                "wide.png": b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 800, 400),
                "wide.gif": b"GIF89a" + struct.pack("<HH", 1600, 800),
                "portrait.png": b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 400, 1200),
                "near.gif": b"GIF87a" + struct.pack("<HH", 800, 560),
                "over.gif": b"GIF89a" + struct.pack("<HH", 800, 564),
            }
            for name, header in files.items():
                (root / name).write_bytes(header)
            self.assertEqual(raster_dimensions(root / "wide.png"), (800, 400))
            self.assertEqual(raster_dimensions(root / "wide.gif"), (1600, 800))
            validate_inline_proportions(root, "README.md", '<p><img src="wide.png"><img src="wide.gif"></p>')
            validate_inline_proportions(root, "README.md", '<p><img src="wide.png"><img src="near.gif"></p>')
            validate_inline_proportions(root, "README.md",
                '<p><img src="portrait.png"></p><p><img src="wide.gif"></p>')
            validate_inline_proportions(root, "README.md",
                '<p><img src="portrait.png"><br><img src="wide.gif"></p>')
            for invalid in (
                '<p><img src="portrait.png"><img src="wide.gif"></p>',
                '<p><a href="portrait.png"><img src="portrait.png"></a>'
                '<a href="wide.gif"><img src="wide.gif"></a></p>',
                '<p><img src="wide.png"><img src="over.gif"></p>',
            ):
                with self.subTest(markup=invalid), self.assertRaisesRegex(ValueError, "1.4x"):
                    validate_inline_proportions(root, "README.md", invalid)
            for invalid in (b"GIF89a", b"\x89PNG\r\n\x1a\n", b"not an image",
                            b"GIF89a" + struct.pack("<HH", 0, 800)):
                path = root / "invalid.gif"
                path.write_bytes(invalid)
                with self.subTest(header=invalid), self.assertRaises(ValueError):
                    raster_dimensions(path)

    def test_bilingual_local_links_and_images_exist(self):
        for name in (*READMES, HISTORY):
            references = readme_references((ROOT / name).read_text(encoding="utf-8"))
            self.assertTrue(references)
            for target, _ in references:
                with self.subTest(readme=name, target=target):
                    path = local_target(ROOT, name, target)
                    if path is not None:
                        self.assertTrue(path.exists(), f"Missing local README target: {target}")

    def test_all_readme_images_ship_in_explicit_invitation_package(self):
        pack = _pack_contract()
        self.assertIn(OVERVIEW, pack.FIXED)
        self.assertNotIn(OVERVIEW, pack.ORIGINAL_README_ASSETS)
        for name in (*READMES, HISTORY):
            images = set()
            for target, image in readme_references((ROOT / name).read_text(encoding="utf-8")):
                path = local_target(ROOT, name, target)
                if image and path is not None:
                    relative = path.relative_to(ROOT).as_posix()
                    images.add(relative)
                    with self.subTest(readme=name, image=relative):
                        self.assertTrue(pack.allowed(PurePosixPath(relative)))
            if name in READMES:
                self.assertIn(OVERVIEW, images)

    def test_new_overview_is_accessible_self_contained_svg(self):
        validate_svg((ROOT / OVERVIEW).read_text(encoding="utf-8"))

    def test_overview_and_architecture_are_default_visible_in_both_readmes(self):
        spec = importlib.util.spec_from_file_location(
            "readme_visible_presentation", ROOT / "scripts/check_release_preservation.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        expected = {OVERVIEW, *ARCHITECTURE_IMAGES}
        for name in READMES:
            with self.subTest(readme=name):
                visible = checker._visible_images((ROOT / name).read_text(encoding="utf-8"))
                self.assertTrue(expected.issubset(visible),
                                "README overview and architecture must not be hidden in details")

    def test_previous_ui_recordings_are_visible_in_history_and_explicitly_packaged(self):
        spec = importlib.util.spec_from_file_location(
            "current_ui_visible_presentation", ROOT / "scripts/check_release_preservation.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        pack = _pack_contract()
        for asset in (*CURRENT_UI_GIFS, UI_SHOWCASE, UI_MANIFEST, HISTORY):
            with self.subTest(asset=asset):
                self.assertTrue((ROOT / asset).is_file())
                self.assertIn(asset, pack.FIXED)
                self.assertNotIn(asset, pack.ORIGINAL_README_ASSETS)
        visible_archive = checker._visible_images((ROOT / HISTORY).read_text(encoding="utf-8"))
        visible = set()
        for target in visible_archive:
            path = local_target(ROOT, HISTORY, target)
            if path is not None:
                visible.add(path.relative_to(ROOT).as_posix())
        self.assertTrue(set(CURRENT_UI_GIFS).issubset(visible),
                        "Previous UI recordings must stay visible in the history archive")
        for name in READMES:
            with self.subTest(readme=name):
                visible = checker._visible_images((ROOT / name).read_text(encoding="utf-8"))
                self.assertFalse(set(CURRENT_UI_GIFS).intersection(visible),
                                 "Previous UI must not be promoted as current product imagery")
                self.assertIn(HISTORY, checker._visible_links((ROOT / name).read_text(encoding="utf-8")))

    def test_current_ui_showcase_local_links_and_images_exist(self):
        references = readme_references((ROOT / UI_SHOWCASE).read_text(encoding="utf-8"))
        self.assertTrue(references)
        local_references = set()
        for target, _ in references:
            with self.subTest(document=UI_SHOWCASE, target=target):
                path = local_target(ROOT, UI_SHOWCASE, target)
                if path is not None:
                    self.assertTrue(path.exists(), f"Missing showcase target: {target}")
                    local_references.add(path.relative_to(ROOT).as_posix())
        self.assertTrue(set(CURRENT_UI_GIFS).issubset(local_references))

    def test_current_ui_manifest_matches_shipped_recording_bytes_and_dimensions(self):
        validate_capture_manifest(json.loads((ROOT / UI_MANIFEST).read_text(encoding="utf-8")), ROOT)

    def test_v2_recordings_are_default_visible_and_explicitly_packaged(self):
        spec = importlib.util.spec_from_file_location(
            "v2_ui_visible_presentation", ROOT / "scripts/check_release_preservation.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        pack = _pack_contract()
        for asset in (*UI_V2_GIFS, UI_V2_MANIFEST):
            with self.subTest(asset=asset):
                self.assertTrue((ROOT / asset).is_file(), f"Missing current UI capture asset: {asset}")
                self.assertIn(asset, pack.FIXED)
                self.assertTrue(pack.allowed(PurePosixPath(asset)))
                self.assertNotIn(asset, pack.ORIGINAL_README_ASSETS)
        for name in READMES:
            with self.subTest(readme=name):
                visible = checker._visible_images((ROOT / name).read_text(encoding="utf-8"))
                self.assertTrue(set(UI_V2_GIFS).issubset(visible),
                                "Current UI recordings must be visible without opening details")

    def test_v2_capture_manifest_matches_shipped_recording_bytes_and_dimensions(self):
        self.assertTrue((ROOT / UI_V2_MANIFEST).is_file())
        validate_capture_manifest(json.loads((ROOT / UI_V2_MANIFEST).read_text(encoding="utf-8")),
                                  ROOT, inventory=UI_V2_GIFS, modes=UI_V2_MODES)

    def test_v2_capture_manifest_rejects_false_modes_inventory_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # Header-only fixtures exercise the manifest contract, not actual
            # playback, provider isolation or the actions declared by a recorder.
            header = b"GIF89a" + struct.pack("<HH", 640, 400)
            captures = []
            for path in UI_V2_GIFS:
                asset = root / path
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(header)
                captures.append({
                    "path": path, "mode": UI_V2_MODES[path],
                    "viewport": {"width": 1180, "height": 820}, "width": 640, "height": 400,
                    "frames": 3, "duration_ms": 250, "sha256": hashlib.sha256(header).hexdigest(),
                    "source_video_sha256": "a" * 64, "actions": ["Fixture current UI action"],
                })
            value = {"schema_version": 1, "source_commit": "b" * 40,
                     "client_build": {"pwa_version": "c" * 20, "command": "fixture build"},
                     "external_model_calls": 0, "captures": captures}
            validate_capture_manifest(value, root, inventory=UI_V2_GIFS, modes=UI_V2_MODES)
            # A new recording generation cannot silently inherit the old
            # offline/stub modes, nor pass the old four-recording contract.
            with self.assertRaises(ValueError):
                validate_capture_manifest(value, root)
            with self.assertRaises(ValueError):
                validate_capture_manifest(value, root, inventory=UI_V2_GIFS)
            changes = (
                ("schema_version", True), ("source_commit", "unknown"),
                ("external_model_calls", 1), ("external_model_calls", False),
                ("client_build.pwa_version", "not-a-build"),
                ("captures.0.mode", "offline-sample"),
                ("captures.0.mode", "isolated-runtime-local-model-stub"),
                ("captures.0.mode", "real-model"),
                ("captures.0.path", UI_V2_GIFS[1]),
                ("captures.0.path", CURRENT_UI_GIFS[0]),
                ("captures.0.path", "docs/assets/../outside.gif"),
                ("captures.0.sha256", "0" * 64), ("captures.0.width", 641),
                ("captures.0.height", True), ("captures.0.frames", 1),
                ("captures.0.duration_ms", 0), ("captures.0.viewport.height", 0),
                ("captures.0.source_video_sha256", "invalid"), ("captures.0.actions", []),
            )
            for field, replacement in changes:
                changed = json.loads(json.dumps(value))
                target = changed
                components = field.split(".")
                for part in components[:-1]:
                    target = target[int(part)] if isinstance(target, list) else target[part]
                target[components[-1]] = replacement
                with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                    validate_capture_manifest(changed, root, inventory=UI_V2_GIFS, modes=UI_V2_MODES)
            for inventory, modes in (
                (UI_V2_GIFS, {UI_V2_GIFS[0]: UI_V2_MODES[UI_V2_GIFS[0]]}),
                ((UI_V2_GIFS[0], UI_V2_GIFS[0]), UI_V2_MODES),
                (("docs/assets/../../outside.gif",), {"docs/assets/../../outside.gif": "unused"}),
            ):
                with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                    validate_capture_manifest(value, root, inventory=inventory, modes=modes)
            for change in ("missing-field", "extra-field", "missing-row", "extra-row"):
                changed = json.loads(json.dumps(value))
                if change == "missing-field":
                    del changed["captures"][0]["actions"]
                elif change == "extra-field":
                    changed["captures"][0]["quality_score"] = 1
                elif change == "missing-row":
                    changed["captures"].pop()
                else:
                    changed["captures"].append(changed["captures"][0].copy())
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_capture_manifest(changed, root, inventory=UI_V2_GIFS, modes=UI_V2_MODES)
            first_asset = root / UI_V2_GIFS[0]
            first_asset.unlink()
            first_asset.symlink_to(root / UI_V2_GIFS[1])
            with self.assertRaisesRegex(ValueError, "regular file"):
                validate_capture_manifest(value, root, inventory=UI_V2_GIFS, modes=UI_V2_MODES)

    def test_capture_manifest_rejects_mismatched_metadata_without_decoding_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            captures = []
            # Deliberately header-only fixtures: these tests do not claim to
            # validate animation playback, frame counts or provider behavior.
            header = b"GIF89a" + struct.pack("<HH", 320, 200)
            for path in CURRENT_UI_GIFS:
                asset = root / path
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(header)
                captures.append({"path": path,
                    "mode": "offline-sample" if path.endswith(("workspace.gif", "graph.gif"))
                            else "isolated-runtime-local-model-stub",
                    "viewport": {"width": 640, "height": 400}, "width": 320, "height": 200,
                    "frames": 2, "duration_ms": 200, "sha256": hashlib.sha256(header).hexdigest(),
                    "source_video_sha256": "a" * 64, "actions": ["Fixture UI action"]})
            value = {"schema_version": 1, "source_commit": "b" * 40,
                     "client_build": {"pwa_version": "c" * 20, "command": "fixture build"},
                     "external_model_calls": 0, "captures": captures}
            validate_capture_manifest(value, root)
            changes = (
                ("source_commit", "not-a-commit"), ("external_model_calls", 1),
                ("captures.0.mode", "real-model"), ("captures.0.sha256", "0" * 64),
                ("captures.0.width", 321), ("captures.0.height", True), ("captures.0.frames", 1),
                ("captures.0.path", CURRENT_UI_GIFS[1]), ("captures.0.source_video_sha256", "invalid"),
                ("captures.0.viewport.width", 0), ("captures.0.actions", []),
            )
            for field, replacement in changes:
                changed = json.loads(json.dumps(value))
                target = changed
                components = field.split(".")
                for part in components[:-1]:
                    target = target[int(part)] if isinstance(target, list) else target[part]
                target[components[-1]] = replacement
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_capture_manifest(changed, root)

    def test_architecture_local_links_exist_and_images_ship_in_package(self):
        pack = _pack_contract()
        references = readme_references((ROOT / ARCHITECTURE).read_text(encoding="utf-8"))
        self.assertTrue(references)
        images = set()
        for target, image in references:
            with self.subTest(document=ARCHITECTURE, target=target):
                path = local_target(ROOT, ARCHITECTURE, target)
                if path is not None:
                    self.assertTrue(path.exists(), f"Missing local architecture target: {target}")
                    if image:
                        relative = path.relative_to(ROOT).as_posix()
                        images.add(relative)
                        self.assertTrue(pack.allowed(PurePosixPath(relative)))
        self.assertTrue(set(ARCHITECTURE_IMAGES).issubset(images))
        for name in ARCHITECTURE_IMAGES:
            with self.subTest(image=name):
                self.assertIn(name, pack.FIXED)
                self.assertNotIn(name, pack.ORIGINAL_README_ASSETS)

    def test_architecture_diagrams_are_accessible_self_contained_svgs(self):
        for name in ARCHITECTURE_IMAGES:
            with self.subTest(image=name):
                validate_svg((ROOT / name).read_text(encoding="utf-8"))

    def test_links_handle_nested_images_fragments_and_encoded_names(self):
        body = ('[![diagram](docs/diagram.svg)](docs/guide%20one.md#section)\n'
                '<img src="docs/diagram.svg?version=2#detail"/>\n'
                '[same page](#section) [remote](https://example.invalid/missing.md)\n'
                '<!-- [hidden](missing.md) -->\n```md\n[example](missing.md)\n```\n')
        references = readme_references(body)
        self.assertIn(("docs/diagram.svg", True), references)
        self.assertIn(("docs/guide%20one.md#section", False), references)
        self.assertNotIn(("missing.md", False), references)
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "docs").mkdir()
            guide = root / "docs/guide one.md"
            guide.touch()
            self.assertEqual(local_target(root, "README.md", "docs/guide%20one.md#section"), guide)
            self.assertEqual(local_target(root, "README.md", "/docs/guide%20one.md?plain=1#section"), guide)
            self.assertEqual(local_target(root, "docs/ARCHITECTURE.md", "guide%20one.md#section"), guide)
            self.assertEqual(local_target(root, "docs/ARCHITECTURE.md", "../README.md#start"),
                             root / "README.md")
            self.assertIsNone(local_target(root, "README.md", "#section"))
            self.assertIsNone(local_target(root, "README.md", "https://example.invalid/missing.md"))
            self.assertFalse(local_target(root, "README.md", "docs/missing.md#part").exists())
            with self.assertRaises(ValueError):
                local_target(root, "README.md", "../outside.md")

    def test_svg_rejects_missing_descriptions_active_content_and_dependencies(self):
        start = f'<svg xmlns="{SVG_NS}" xmlns:xlink="http://www.w3.org/1999/xlink">'
        valid = '<title>Title</title><desc>Description</desc><path id="line"/>'
        cases = (
            "<title>Title</title>",
            "<desc>Description</desc>",
            valid + '<script>alert(1)</script>',
            valid + '<foreignObject/>',
            valid + '<image href="https://example.invalid/image.svg"/>',
            valid + '<image xlink:href="local.png"/>',
            valid + '<path onload="alert(1)"/>',
            valid + '<style>@import "https://example.invalid/style.css";</style>',
            valid + '<style>@font-face {font-family:remote}</style>',
            valid + '<path fill="url(https://example.invalid/image.svg)"/>',
            valid + '<path fill="url(#missing)"/>',
        )
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_svg(start + content + "</svg>")
        validate_svg(start + valid + '<use href="#line" fill="url(#line)"/></svg>')
        with self.assertRaises(ValueError):
            validate_svg('<!DOCTYPE svg SYSTEM "https://example.invalid/schema">' + start + valid + "</svg>")
        with self.assertRaises(ValueError):
            validate_svg('<?xml-stylesheet href="remote.css"?>' + start + valid + "</svg>")


if __name__ == "__main__":
    unittest.main()
