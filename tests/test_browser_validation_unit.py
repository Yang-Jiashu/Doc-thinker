"""Offline regressions for mandatory browser execution and sandbox admission."""

import subprocess

import pytest

from tests.test_query_ui_controls_unit import (
    _assert_linux_sandbox,
    _configured_browser,
    _run_browser,
)


def test_browser_remains_optional_for_ordinary_unit_tests(monkeypatch):
    monkeypatch.delenv("DOCTHINKER_BROWSER_BIN", raising=False)
    monkeypatch.delenv("DOCTHINKER_REQUIRE_BROWSER", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _configured_browser()


def test_required_browser_cannot_silently_skip(monkeypatch):
    monkeypatch.delenv("DOCTHINKER_BROWSER_BIN", raising=False)
    monkeypatch.setenv("DOCTHINKER_REQUIRE_BROWSER", "1")
    with pytest.raises(pytest.fail.Exception, match="DOCTHINKER_BROWSER_BIN"):
        _configured_browser()


def test_explicit_invalid_browser_fails_even_outside_ci(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCTHINKER_BROWSER_BIN", str(tmp_path / "missing-chromium"))
    with pytest.raises(pytest.fail.Exception, match="not an executable"):
        _configured_browser()


@pytest.mark.parametrize("first_layer", ["SUID", "Namespace"])
def test_sandbox_status_accepts_both_supported_first_layers(first_layer):
    _assert_linux_sandbox(f"""<table>
        <tr><td>Layer 1 Sandbox</td><td>{first_layer}</td></tr>
        <tr><td>PID namespaces</td><td>Yes</td></tr>
        <tr><td>Network namespaces</td><td>Yes</td></tr>
        <tr><td>Seccomp-BPF sandbox</td><td>Yes</td></tr>
    </table>""")


@pytest.mark.parametrize("disabled", ["Layer 1 Sandbox", "PID namespaces",
                                     "Network namespaces", "Seccomp-BPF sandbox"])
def test_sandbox_status_rejects_each_missing_layer(disabled):
    statuses = {"Layer 1 Sandbox": "Namespace", "PID namespaces": "Yes",
                "Network namespaces": "Yes", "Seccomp-BPF sandbox": "Yes"}
    statuses[disabled] = "No"
    dom = "<table>" + "".join(f"<tr><td>{name}</td><td>{value}</td></tr>"
                              for name, value in statuses.items()) + "</table>"
    with pytest.raises(AssertionError):
        _assert_linux_sandbox(dom)


def test_browser_launch_failure_propagates_with_diagnostics(monkeypatch, tmp_path):
    def failed_launch(command, **kwargs):
        assert not any(flag in command for flag in
                       ("--no-sandbox", "--disable-setuid-sandbox", "--disable-seccomp-filter-sandbox"))
        return subprocess.CompletedProcess(command, 1, "", "No usable sandbox!")

    monkeypatch.setattr(subprocess, "run", failed_launch)
    monkeypatch.setenv("DOCTHINKER_BROWSER_ARTIFACTS", str(tmp_path / "logs"))
    with pytest.raises(AssertionError, match="No usable sandbox"):
        _run_browser("chromium", tmp_path / "profile", 390, "file:///fixture.html", "failure")
    assert (tmp_path / "logs/failure.stderr.txt").read_text() == "No usable sandbox!"
