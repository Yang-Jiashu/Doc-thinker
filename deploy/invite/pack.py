"""Create a source deployment ZIP from explicit inputs, never runtime data.

This packages Docker build inputs, not a prebuilt image or a public website.
Run from any directory: python3 deploy/invite/pack.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "docthinker-invite"
KNOWLEDGE_PACKS = ("mao-workplace", "research-ideation", "literature-synthesis", "evidence-writing", "decision-review")
ORIGINAL_README_ASSETS = (
    "docs/assets/banner.png", "docs/assets/agentic_memory_architecture.png",
    "docs/assets/usecase_chat.gif", "docs/assets/usecase_kg.gif",
    "docs/assets/memory_kg_observability_demo.gif", "docs/assets/gpu_knowledge_graph_demo.gif",
    "docs/assets/ui_memory_editor_screenshot.png", "docs/assets/pipeline.png",
    "docs/assets/lifecycle.png", "docs/assets/multi_agent_evolution.png", "docs/assets/sparql_cot.png",
)
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
# One immutable original demo exceeds the normal per-file bound. An arbitrary
# file, changed demo or same-name payload cannot use this size exception.
FROZEN_LARGE_ASSETS = {
    "docs/assets/gpu_knowledge_graph_demo.gif": (
        41001259, "a2210b09b40a0dc46da824ec15030ab30dd694a128865bac97b70f8747791557"),
}
FIXED = (
    ".dockerignore", "LICENSE", "CONTRIBUTING.md", "README.md", "README.zh-CN.md", "logo.png",
    *ORIGINAL_README_ASSETS,
    "docs/assets/docthinker-v2-overview.svg",
    "docs/assets/architecture-system.svg", "docs/assets/architecture-memory.svg",
    "docs/assets/architecture-task-loop.svg",
    "docs/assets/current-ui-workspace.gif", "docs/assets/current-ui-memory.gif",
    "docs/assets/current-ui-graph.gif", "docs/assets/current-ui-projects.gif",
    "docs/assets/current-ui-manifest.json", "docs/UI_SHOWCASE.md", "docs/HISTORY.md",
    "docs/assets/ui-v2-workspace.gif", "docs/assets/ui-v2-memory-observatory.gif",
    "docs/assets/ui-v2-graph-canvas.gif", "docs/assets/ui-v2-manifest.json",
    "docs/ARCHITECTURE.md", "docs/PRODUCTIZATION.md", "docs/INVITE_WEB_BETA.md",
    "docs/MOBILE_APP.md", "docs/RESEARCH_GROUNDED_DESIGN.md", "docs/RSI_ROADMAP.md",
    "docs/WORKBENCH_PRODUCT_DESIGN.md", "docs/VOICE_MODELS.md", "docs/WECHAT_MINIPROGRAM.md",
    "docs/LONG_PROJECT_RUNTIME.md",
    "docs/PERFORMANCE_STRUCTURES_2026-10-13.md",
    "docs/MOBILE_INSTALLATION.md", "docs/MOBILE_DESIGN.md", "docs/GESTURE_INTERACTION.md",
    "docs/MODEL_CONFIGURATION.md", "docs/MEMORY_PLUGIN_GUIDE.md", "docs/WECHAT_PRIVATE_DEPLOYMENT.md",
    "docs/QUERY_RUNTIME.md", "docs/SELF_EVOLUTION_EVALUATION.md",
    "docs/INSIGHT_RUNTIME_DESIGN.md", "docs/CODEX_HARNESS_REFERENCE.md", "docs/SUPER_SKILLS.md",
    "docs/RELEASE_ACCEPTANCE_2026-10-12.md",
    "docs/RELEASE_ACCEPTANCE_2026-10-13.md",
    "docs/RELEASE_ACCEPTANCE_2026-10-20.md",
    "benchmarks/runtime_algorithms/long_project_personal_agent_result_20261020.json",
    "benchmarks/runtime_algorithms/long_project_context_benchmark_20261020.json",
    "benchmarks/runtime_algorithms/long_project_full_mainline_final_20261020.xml.gz",
    "benchmarks/runtime_algorithms/personal_agent_full_mainline_final_20261020.xml.gz",
    "benchmarks/runtime_algorithms/project_replan_mobile_20261020.xml.gz",
    "benchmarks/runtime_algorithms/publish_verification_20261020.json",
    "benchmarks/runtime_algorithms/publish_full_mainline_20261020.xml.gz",
    "benchmarks/runtime_algorithms/event_evidence_runtime_20261020.json",
    "benchmarks/runtime_algorithms/event_runtime_full_20261020.xml.gz",
    "benchmarks/runtime_algorithms/goal_evidence_staged_result_20261020.json",
    "benchmarks/runtime_algorithms/goal_evidence_staged_full_20261020.xml.gz",
    "benchmarks/research_real/result_insight_v4_development_review_20261012.md",
    "benchmarks/research_real/result_insight_v5_targeted_review_20261012.md",
    "benchmarks/research_real/plan_insight_component_regression_20261012.md",
    "benchmarks/research_real/result_insight_component_regression_20261012.md",
    "benchmarks/research_real/plan_insight_source_scope_regression_20261012.md",
    "benchmarks/research_real/result_insight_source_scope_regression_20261013.md",
    "benchmarks/research_real/insight_newpapers_smoke_20261013.json",
    "benchmarks/research_real/insight_newpapers_smoke_20261013.registration.json",
    "benchmarks/research_real/plan_insight_newpapers_smoke_20261013.md",
    "benchmarks/research_real/result_insight_newpapers_preflight_v8_20261013.md",
    "benchmarks/research_real/insight_newpapers_routing_v9_20261013.registration.json",
    "benchmarks/research_real/plan_insight_newpapers_routing_regression_v9_20261013.md",
    "benchmarks/research_real/result_insight_newpapers_routing_v9_20261013.md",
    "benchmarks/research_real/plan_insight_natural_evidence_regression_v10_20261013.md",
    "benchmarks/research_real/insight_natural_evidence_v10_20261013.registration.json",
    "benchmarks/research_real/result_insight_natural_evidence_v10_20261013.md",
    "benchmarks/research_real/plan_insight_evaluation_delivery_v11_20261013.md",
    "benchmarks/research_real/insight_evaluation_delivery_v11_20261013.registration.json",
    "benchmarks/research_real/result_insight_evaluation_delivery_v11_20261013.md",
    "benchmarks/research_real/insight_transfer_v11_20261013.json",
    "benchmarks/research_real/plan_insight_transfer_v11_20261013.json",
    "benchmarks/research_real/insight_transfer_v11_20261013.registration.json",
    "benchmarks/research_real/result_insight_transfer_v11_20261013.md",
    "benchmarks/research_real/plan_insight_title_route_v12_20261013.md",
    "benchmarks/research_real/result_insight_title_input_v12_20261013.md",
    "benchmarks/research_real/plan_insight_operation_coverage_v13_20261013.md",
    "benchmarks/research_real/result_insight_operation_input_v13_20261013.md",
    "benchmarks/research_real/plan_insight_operation_repair_v14_20261013.md",
    "benchmarks/research_real/insight_operation_repair_v14_20261013.registration.json",
    "benchmarks/research_real/result_insight_operation_repair_v14_20261013.md",
    "scripts/benchmark_research_assist.py", "scripts/research_benchmark_contract.py",
    "scripts/benchmark_event_reasoning.py",
    "scripts/benchmark_goal_evidence.py", "scripts/benchmark_goal_evidence_staged.py",
    "deploy/invite/Dockerfile",
    "deploy/invite/compose.yaml", "deploy/invite/compose.https.yaml",
    "deploy/invite/Caddyfile.container", "deploy/invite/Caddyfile.example", "deploy/invite/requirements.in",
    "deploy/invite/requirements.txt", "deploy/invite/manage.py", "deploy/invite/pack.py",
    "deploy/invite/preflight.py", "deploy/invite/backup.py",
    "deploy/invite/README.md", "docthinker/knowledge_base_schema.sql",
    "clients/mobile/package.json", "clients/mobile/package-lock.json",
    "clients/mobile/index.html", "clients/mobile/vite.config.js",
    "examples/research_reasoning/energy_chain.txt",
    "examples/research_reasoning/analogy_energy.txt", "examples/research_reasoning/analogy_learning.txt",
    "examples/research_reasoning/induction_a.txt", "examples/research_reasoning/induction_b.txt",
    "examples/research_reasoning/induction_counterexample.txt",
    *(f"docthinker/knowledge_skills/{name}/{filename}"
      for name in KNOWLEDGE_PACKS for filename in ("SKILL.md", "knowledge.json")),
)
SUFFIXES = {".vue", ".js", ".mjs", ".json", ".css", ".svg", ".png", ".html", ".webmanifest"}
BLOCKED = {"node_modules", "__pycache__", ".git", "dist", "data", "uploads", "secrets"}


def _input_size_allowed(name: str, size: int) -> bool:
    return (0 <= size <= MAX_INPUT_BYTES or
            name in FROZEN_LARGE_ASSETS and size == FROZEN_LARGE_ASSETS[name][0])


def _check_input_content(name: str, size: int, digest: str) -> None:
    if (not _input_size_allowed(name, size) or
            size > MAX_INPUT_BYTES and (size, digest) != FROZEN_LARGE_ASSETS.get(name)):
        raise ValueError("Unexpected oversized or changed presentation input")


def allowed(relative: PurePosixPath) -> bool:
    if relative.is_absolute() or ".." in relative.parts or any(part in BLOCKED for part in relative.parts):
        return False
    if str(relative) in FIXED:
        return True
    if any(part.startswith(".") for part in relative.parts):
        return False
    if relative.suffix == ".py" and relative.parts[0] in {"docthinker", "graphcore"}:
        return not (relative.parts[:2] == ("docthinker", "ui") or
                    relative.parts[:3] in {("graphcore", "coregraph", "api"), ("graphcore", "coregraph", "tools")})
    return (relative.parts[:3] in {("clients", "mobile", "src"), ("clients", "mobile", "public"),
                                  ("clients", "mobile", "scripts")}
            and relative.suffix in SUFFIXES)


def build_bundle(root: Path, output: Path) -> tuple[int, str]:
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.is_symlink() or output.exists():
        raise ValueError("Output already exists; choose a new ZIP filename")
    # Enumerate only the named build trees; do not traverse personal data at all.
    files = {Path(name) for name in FIXED}
    for tree in ("docthinker", "graphcore", "clients/mobile/src", "clients/mobile/public", "clients/mobile/scripts"):
        start = root / tree
        if start.is_symlink():
            raise ValueError("Build input directories must not be symlinks")
        for directory, directories, names in os.walk(start, followlinks=False):
            directories[:] = [name for name in directories if name not in BLOCKED and not name.startswith(".")]
            for name in names:
                relative = (Path(directory) / name).relative_to(root)
                if allowed(PurePosixPath(relative.as_posix())):
                    files.add(relative)
    payloads = []
    manifest = {"format": 1, "profile": "default chat/history; opt-in bounded research; no bundled secrets or user data", "files": {}}
    for relative in sorted(files):
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise ValueError("A required build input is missing or not a real regular file")
        name = relative.as_posix()
        if not _input_size_allowed(name, path.stat().st_size):
            raise ValueError("Unexpected oversized build input")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        _check_input_content(name, len(data), digest)
        manifest["files"][name] = digest
        payloads.append((name, data))
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    if sum(len(data) for _, data in payloads) + len(manifest_bytes) > MAX_BUNDLE_BYTES:
        raise ValueError("Oversized bundle inventory")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in payloads:
                archive.writestr(f"{PREFIX}/{name}", data)
            archive.writestr(f"{PREFIX}/PACKAGE-MANIFEST.json", manifest_bytes)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return len(payloads), hashlib.sha256(output.read_bytes()).hexdigest()


def verify_bundle(path: Path) -> tuple[int, str]:
    """Check a bounded archive without extracting or executing its contents.

    SHA256 establishes consistency with the enclosed manifest, not publisher
    authenticity. A replaced ZIP and replaced manifest still require a trusted
    out-of-band digest. No settings, account data or model key is accessed.
    """
    path = Path(path).absolute()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BUNDLE_BYTES:
            raise ValueError("Invalid or oversized bundle")
        try:
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                names = [item.filename for item in members]
                manifest_name = f"{PREFIX}/PACKAGE-MANIFEST.json"
                if (not members or len(members) > 2000 or len(set(names)) != len(names)
                        or names.count(manifest_name) != 1
                        or sum(item.file_size for item in members) > MAX_BUNDLE_BYTES):
                    raise ValueError("Invalid bundle inventory")
                for item in members:
                    name = item.filename
                    relative = PurePosixPath(name)
                    mode = item.external_attr >> 16
                    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                            or relative.as_posix() != name or relative.parts[0] != PREFIX
                            or item.is_dir() or item.flag_bits & 1
                            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                            or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                            or (name != manifest_name and
                                not _input_size_allowed(PurePosixPath(*relative.parts[1:]).as_posix(),
                                                        item.file_size))
                            or (name != manifest_name and not allowed(PurePosixPath(*relative.parts[1:])))):
                        raise ValueError("Unsafe or unsupported bundle entry")
                if archive.getinfo(manifest_name).file_size > 1024 * 1024:
                    raise ValueError("Oversized bundle manifest")
                manifest = json.loads(archive.read(manifest_name))
                files = manifest.get("files") if isinstance(manifest, dict) else None
                if (manifest.get("format") != 1 or not isinstance(files, dict)
                        or not set(FIXED).issubset(files)
                        or set(names) != {manifest_name, *(f"{PREFIX}/{name}" for name in files)}):
                    raise ValueError("Bundle and manifest inventories differ")
                for name, expected in files.items():
                    if (not isinstance(name, str) or not isinstance(expected, str)
                            or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
                        raise ValueError("Bundle content does not match manifest")
                    data = archive.read(f"{PREFIX}/{name}")
                    digest = hashlib.sha256(data).hexdigest()
                    _check_input_content(name, len(data), digest)
                    if digest != expected:
                        raise ValueError("Bundle content does not match manifest")
        except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid bundle or manifest") from error
        stream.seek(0)
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        digest = hasher.hexdigest()
    return len(files), digest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--output", type=Path, default=ROOT / "dist" / "docthinker-invite-deploy.zip")
    operation.add_argument("--verify", type=Path, help="Read-only ZIP inventory and SHA256 verification; no extraction")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            count, digest = verify_bundle(args.verify)
            print(f"Verified {count} build inputs. SHA256: {digest}")
            print("Manifest consistency only, not publisher authentication, container or production acceptance.")
            return 0
        count, digest = build_bundle(ROOT, args.output)
        print(f"Packaged {count} build inputs. SHA256: {digest}")
        print(f"Deployment ZIP: {args.output.absolute()}")
        print("No public service, accounts, provider requests or container image were created.")
        return 0
    except Exception:
        print("Bundle operation failed; check the build inputs, archive or unused output path.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
