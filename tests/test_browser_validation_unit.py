"""Offline regressions for mandatory browser execution and sandbox admission."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_query_ui_controls_unit import (
    _assert_linux_sandbox,
    _configured_browser,
    sandboxed_browser,
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


def _mock_playwright(monkeypatch, launch):
    api = ModuleType("playwright.sync_api")
    api.sync_playwright = lambda: nullcontext(SimpleNamespace(chromium=SimpleNamespace(launch=launch)))
    monkeypatch.setitem(sys.modules, "playwright", ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", api)
    monkeypatch.setattr("tests.test_query_ui_controls_unit._configured_browser", lambda: "chromium")
    monkeypatch.setattr("tests.test_query_ui_controls_unit.os.geteuid", lambda: 1001)


def test_browser_driver_explicitly_enables_sandbox_and_closes(monkeypatch):
    browser = Mock()
    browser.new_page.return_value.content.return_value = (
        "Layer 1 Sandbox Namespace PID namespaces Yes "
        "Network namespaces Yes Seccomp-BPF sandbox Yes"
    )
    launch = Mock(return_value=browser)
    _mock_playwright(monkeypatch, launch)
    fixture = sandboxed_browser.__wrapped__()
    assert next(fixture) is browser
    assert launch.call_args.kwargs["chromium_sandbox"] is True
    assert not any("sandbox" in arg for arg in launch.call_args.kwargs["args"])
    fixture.close()
    browser.close.assert_called_once()


def test_browser_launch_failure_propagates_with_diagnostics(monkeypatch, tmp_path):
    _mock_playwright(monkeypatch, Mock(side_effect=RuntimeError("No usable sandbox!")))
    monkeypatch.setenv("DOCTHINKER_BROWSER_ARTIFACTS", str(tmp_path / "logs"))
    with pytest.raises(RuntimeError, match="No usable sandbox"):
        next(sandboxed_browser.__wrapped__())
    assert (tmp_path / "logs/browser-launch.stderr.txt").read_text() == "No usable sandbox!"
