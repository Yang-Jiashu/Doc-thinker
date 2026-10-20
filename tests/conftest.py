"""Keep research unit tests independent of locally installed model caches."""

import pytest


@pytest.fixture(autouse=True)
def offline_research_unit_environment(request, monkeypatch):
    """Only automatic model discovery is disabled; injected backends still run.

    Service integration tests can explicitly set the semantic mode to ``local``
    or ``auto`` (or delete it to exercise the production default). Direct model
    tests retain their existing opt-in gates and bypass this service setting.
    """
    name = request.path.name
    if not (name.startswith("test_research_") and name.endswith("_unit.py")):
        return
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DOCTHINKER_RESEARCH_SEMANTIC", "off")
