"""Offline contract tests for the source deployment bundle and Docker recipe."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("invite_pack", ROOT / "deploy/invite/pack.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


@pytest.mark.parametrize("name", [".env", "docthinker/.env", "clients/mobile/src/.env.local",
    "data/#00001/history.json", "clients/mobile/node_modules/key.js", "secrets/llm-api-key",
    "docthinker/private.db", "docthinker/ui/static/private.js", "../key.py", "/secret.py",
    "benchmarks/research_real/calls.jsonl", "benchmarks/research_real/answers/private.json",
    "benchmarks/research_real/unregistered.json", "benchmarks/research_real/papers/private.pdf",
    "benchmarks/runtime_algorithms/long_project_full_mainline_final_20261020.xml",
    "scripts/private_eval.py"])
def test_bundle_rejects_non_build_inputs(name):
    assert not pack.allowed(PurePosixPath(name))


@pytest.mark.parametrize("name", ["docthinker/server/invite_runtime.py", "graphcore/coregraph/utils.py",
    "clients/mobile/src/pages/studio/index.vue", "clients/mobile/src/static/brand/logo.png", "logo.png",
    "docs/assets/docthinker-v2-overview.svg",
    "docs/assets/architecture-system.svg", "docs/assets/architecture-memory.svg",
    "docs/assets/architecture-task-loop.svg",
    "deploy/invite/requirements.txt", "deploy/invite/backup.py", ".dockerignore", "docs/ARCHITECTURE.md", "docs/PRODUCTIZATION.md",
    "docs/RELEASE_ACCEPTANCE_2026-10-12.md",
    "docs/RELEASE_ACCEPTANCE_2026-10-20.md",
    "benchmarks/runtime_algorithms/long_project_personal_agent_result_20261020.json",
    "benchmarks/runtime_algorithms/long_project_context_benchmark_20261020.json",
    "benchmarks/runtime_algorithms/long_project_full_mainline_final_20261020.xml.gz",
    "benchmarks/runtime_algorithms/event_evidence_runtime_20261020.json",
    "benchmarks/runtime_algorithms/event_runtime_full_20261020.xml.gz",
    "benchmarks/runtime_algorithms/goal_evidence_staged_result_20261020.json",
    "benchmarks/runtime_algorithms/goal_evidence_staged_full_20261020.xml.gz",
    "docs/LONG_PROJECT_RUNTIME.md", "docs/RELEASE_ACCEPTANCE_2026-10-13.md",
    "docthinker/server/project_task_handoff.py",
    "benchmarks/research_real/result_insight_v5_targeted_review_20261012.md",
    "benchmarks/research_real/plan_insight_component_regression_20261012.md",
    "benchmarks/research_real/result_insight_component_regression_20261012.md",
    "benchmarks/research_real/plan_insight_source_scope_regression_20261012.md",
    "benchmarks/research_real/result_insight_source_scope_regression_20261013.md",
    "benchmarks/research_real/insight_newpapers_smoke_20261013.json",
    "benchmarks/research_real/insight_newpapers_routing_v9_20261013.registration.json",
    "benchmarks/research_real/result_insight_newpapers_routing_v9_20261013.md",
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
    "docthinker/server/research_review_stage.py",
    "docthinker/server/project_obligation_ledger.py",
    "docthinker/server/project_evidence_controller.py",
    "docthinker/server/project_analysis_lineage.py",
    "docthinker/harness_runtime.py", "docthinker/server/insight_runtime.py",
    "docthinker/knowledge_skills/literature-synthesis/knowledge.json",
    "docthinker/knowledge_skills/literature-synthesis/SKILL.md"])
def test_bundle_allows_required_inputs(name):
    assert pack.allowed(PurePosixPath(name))


@pytest.mark.parametrize("name", pack.ORIGINAL_README_ASSETS)
def test_bundle_explicitly_retains_original_readme_images(name):
    assert name in pack.FIXED
    assert pack.allowed(PurePosixPath(name))


def fixture_tree(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in pack.FIXED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"build input\n")
    for name, content in {"docthinker/server/demo.py": "pass\n", "graphcore/coregraph/demo.py": "pass\n",
        "clients/mobile/src/main.js": "export {}", "clients/mobile/public/logo.svg": "<svg/>",
        "clients/mobile/scripts/build.mjs": "export {}", ".env": "REAL_PRIVATE_SENTINEL",
        "data/history.json": "REAL_PRIVATE_SENTINEL", "clients/mobile/src/.env": "REAL_PRIVATE_SENTINEL"}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_archive_contains_verified_manifest_without_runtime_data(tmp_path):
    root = fixture_tree(tmp_path)
    output = tmp_path / "package.zip"
    count, digest = pack.build_bundle(root, output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o600
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read(f"{pack.PREFIX}/PACKAGE-MANIFEST.json"))
        assert count == len(manifest["files"])
        assert {"docs/ARCHITECTURE.md", "docs/PRODUCTIZATION.md", "README.zh-CN.md"}.issubset(manifest["files"])
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(f"{pack.PREFIX}/{name}")).hexdigest() == expected
        assert not any(b"REAL_PRIVATE_SENTINEL" in archive.read(name) for name in archive.namelist())


def test_actual_source_bundle_can_load_all_knowledge_and_insight_modules(tmp_path):
    output = tmp_path / "actual-source.zip"
    pack.build_bundle(ROOT, output)
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read(f"{pack.PREFIX}/PACKAGE-MANIFEST.json"))
        for name in ("logo.png", *pack.ORIGINAL_README_ASSETS,
                     "docs/assets/docthinker-v2-overview.svg",
                     "docs/assets/architecture-system.svg", "docs/assets/architecture-memory.svg",
                     "docs/assets/architecture-task-loop.svg"):
            expected = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            assert manifest["files"][name] == expected
            assert hashlib.sha256(archive.read(f"{pack.PREFIX}/{name}")).hexdigest() == expected
        large_name, (large_size, large_hash) = next(iter(pack.FROZEN_LARGE_ASSETS.items()))
        assert archive.getinfo(f"{pack.PREFIX}/{large_name}").file_size == large_size == 41001259
        assert manifest["files"][large_name] == large_hash
        assert sum(item.file_size for item in archive.infolist()) <= 128 * 1024 * 1024
        assert pack.verify_bundle(output)[0] == len(manifest["files"])
        archive.extractall(destination)
    bundle = destination / pack.PREFIX
    script = (
        "import json; from docthinker.server.knowledge_skills import _load_pack, knowledge_skill_catalog; "
        "from docthinker.harness_runtime import HarnessRun; "
        "from docthinker.server.insight_runtime import InsightRuntime; "
        "from docthinker.server.research_component_coverage import functional_definitions; "
        "from docthinker.server.invite_account_lifecycle import export_account, purge_account; "
        "print(json.dumps({item['id']:len(_load_pack(item['id'])) for item in knowledge_skill_catalog()}))"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=bundle,
                            env={"PYTHONPATH": str(bundle), "PYTHON_DOTENV_DISABLED": "1",
                                 "DOCTHINKER_RESEARCH_SEMANTIC": "off"},
                            capture_output=True, text=True, timeout=30, check=True)
    counts = json.loads(result.stdout)
    assert set(counts) == set(pack.KNOWLEDGE_PACKS)
    assert all(count > 0 for count in counts.values())


@pytest.fixture
def small_large_asset(monkeypatch):
    name = "docs/assets/gpu_knowledge_graph_demo.gif"
    content = b"frozen original GIF payload" * 3
    monkeypatch.setattr(pack, "MAX_INPUT_BYTES", 64)
    monkeypatch.setattr(pack, "FROZEN_LARGE_ASSETS", {
        name: (len(content), hashlib.sha256(content).hexdigest()),
    })
    return name, content


def test_only_exact_frozen_large_demo_can_exceed_normal_bound(tmp_path, small_large_asset):
    name, content = small_large_asset
    root = fixture_tree(tmp_path)
    (root / name).write_bytes(content)
    output = tmp_path / "frozen-demo.zip"
    count, digest = pack.build_bundle(root, output)
    assert pack.verify_bundle(output) == (count, digest)


@pytest.mark.parametrize("damage", ["wrong_hash", "wrong_size", "different_path"])
def test_build_rejects_changed_or_other_oversized_input(tmp_path, small_large_asset, damage):
    name, content = small_large_asset
    root = fixture_tree(tmp_path)
    target = root / ("docs/assets/usecase_chat.gif" if damage == "different_path" else name)
    target.write_bytes(b"x" * len(content) if damage == "wrong_hash"
                       else content + b"x" if damage == "wrong_size" else content)
    output = tmp_path / "bad.zip"
    with pytest.raises(ValueError):
        pack.build_bundle(root, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["wrong_hash", "wrong_size", "different_path"])
def test_verify_rejects_oversized_substitution_even_with_matching_manifest(
        tmp_path, small_large_asset, damage):
    name, content = small_large_asset
    root = fixture_tree(tmp_path)
    (root / name).write_bytes(content)
    good = tmp_path / "good.zip"
    pack.build_bundle(root, good)
    bad = tmp_path / "bad.zip"
    selected = "docs/assets/usecase_chat.gif" if damage == "different_path" else name
    replacement = (b"x" * len(content) if damage == "wrong_hash"
                   else content + b"x" if damage == "wrong_size" else content)
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(bad, "w") as target:
        manifest = json.loads(source.read(f"{pack.PREFIX}/PACKAGE-MANIFEST.json"))
        manifest["files"][selected] = hashlib.sha256(replacement).hexdigest()
        for member in source.namelist():
            if member.endswith("/PACKAGE-MANIFEST.json"):
                target.writestr(member, json.dumps(manifest))
            else:
                target.writestr(member, replacement if member == f"{pack.PREFIX}/{selected}"
                                else source.read(member))
    with pytest.raises(ValueError):
        pack.verify_bundle(bad)


def test_total_bundle_bound_is_not_widened_by_frozen_demo_exception(
        tmp_path, monkeypatch, small_large_asset):
    name, content = small_large_asset
    root = fixture_tree(tmp_path)
    (root / name).write_bytes(content)
    monkeypatch.setattr(pack, "MAX_BUNDLE_BYTES", len(content))
    output = tmp_path / "too-large.zip"
    with pytest.raises(ValueError, match="Oversized bundle"):
        pack.build_bundle(root, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    root = fixture_tree(tmp_path)
    output = tmp_path / "package.zip"
    output.write_bytes(b"keep me")
    with pytest.raises(ValueError):
        pack.build_bundle(root, output)
    assert output.read_bytes() == b"keep me"


def test_verify_bundle_is_read_only_and_checks_manifest(tmp_path):
    root = fixture_tree(tmp_path)
    output = tmp_path / "package.zip"
    count, digest = pack.build_bundle(root, output)
    original = output.read_bytes()
    assert pack.verify_bundle(output) == (count, digest)
    assert output.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["package.zip", "source"]


@pytest.mark.parametrize("damage", ["changed", "extra", "duplicate", "escape", "missing", "symlink", "bad_manifest"])
def test_verify_bundle_rejects_modified_or_unsafe_members(tmp_path, damage):
    root = fixture_tree(tmp_path)
    good = tmp_path / "good.zip"
    pack.build_bundle(root, good)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(bad, "w") as target:
        names = source.namelist()
        selected = f"{pack.PREFIX}/README.md"
        for name in names:
            if name == selected and damage == "missing":
                continue
            data = source.read(name)
            if name == selected and damage == "symlink":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                target.writestr(info, data)
            elif name.endswith("/PACKAGE-MANIFEST.json") and damage == "bad_manifest":
                target.writestr(name, b"{\"format\":1,\"files\":null}")
            else:
                target.writestr(name, b"changed" if name == selected and damage == "changed" else data)
        if damage == "extra":
            target.writestr(f"{pack.PREFIX}/.env", b"private")
        elif damage == "duplicate":
            with pytest.warns(UserWarning):
                target.writestr(selected, source.read(selected))
        elif damage == "escape":
            target.writestr(f"{pack.PREFIX}/../escape.py", b"pass")
    with pytest.raises(ValueError):
        pack.verify_bundle(bad)


def test_verify_cli_never_rebuilds_or_extracts(tmp_path, capsys):
    root = fixture_tree(tmp_path)
    output = tmp_path / "package.zip"
    count, digest = pack.build_bundle(root, output)
    assert pack.main(["--verify", str(output)]) == 0
    assert f"Verified {count} build inputs. SHA256: {digest}" in capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["package.zip", "source"]


def test_source_symlink_cannot_leak_private_file(tmp_path):
    root = fixture_tree(tmp_path)
    private = tmp_path / "private"
    private.write_text("private")
    (root / "docthinker/server/leak.py").symlink_to(private)
    output = tmp_path / "package.zip"
    with pytest.raises(ValueError):
        pack.build_bundle(root, output)
    assert not output.exists()


def test_compose_is_local_nonroot_bounded_and_persistent():
    yaml = pytest.importorskip("yaml", reason="Compose YAML validation requires the optional PyYAML test tool")
    configuration = yaml.safe_load((ROOT / "deploy/invite/compose.yaml").read_text())
    gateway = configuration["services"]["gateway"]
    assert gateway["ports"] == ["127.0.0.1:8088:8080"]
    assert gateway["user"] == "10001:10001"
    assert gateway["read_only"] is True
    assert gateway["init"] is True
    assert gateway["restart"] == "no"
    assert gateway["cap_drop"] == ["ALL"]
    assert gateway["pids_limit"] <= 128
    assert gateway["volumes"] == ["invite-data:/var/lib/docthinker"]
    assert gateway["environment"]["LLM_BINDING_API_KEY_FILE"] == "/run/secrets/llm_api_key"
    assert "LLM_BINDING_API_KEY" not in gateway["environment"]
    assert gateway["healthcheck"]["test"][-1] == "--healthcheck"


def test_docker_recipe_keeps_build_tools_out_of_runtime():
    recipe = (ROOT / "deploy/invite/Dockerfile").read_text()
    runtime = recipe.split(" AS runtime", 1)[1]
    assert "USER 10001:10001" in runtime
    assert "COPY . " not in recipe
    assert "--require-hashes" in recipe
    assert "node_modules" not in runtime and "npm ci" not in runtime
    assert "PYTHON_DOTENV_DISABLED=1" in runtime
    assert "install -d -o 10001 -g 10001 -m 0700 /var/lib/docthinker" in runtime
    assert "LLM_BINDING_API_KEY=" not in recipe
    assert ".env" not in recipe


def test_docker_context_denies_personal_state_by_default():
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    patterns = [line for line in lines if line and not line.startswith("#")]
    assert patterns[0] == "**"
    for value in ("**/.env", "**/*.pem", "**/*.key", "**/*.db*", "**/node_modules/**"):
        assert value in patterns
    assert "!clients/mobile/src/**" in patterns
    assert "!docthinker/knowledge_base_schema.sql" in patterns
    assert "!docthinker/knowledge_skills/*/knowledge.json" in patterns
    assert "!docthinker/knowledge_skills/*/SKILL.md" in patterns


def test_chat_dependency_lock_excludes_local_model_and_parser_stacks():
    lines = (ROOT / "deploy/invite/requirements.txt").read_text().splitlines()
    pins = [line.split("==")[0] for line in lines if line and not line.startswith((" ", "#"))]
    assert {"fastapi", "uvicorn", "httpx", "openai"}.issubset(pins)
    assert not {"torch", "transformers", "mineru", "magic-pdf", "pipmaster", "faiss-cpu"}.intersection(pins)
    assert all("==" in line for line in lines if line and not line.startswith((" ", "#")))
