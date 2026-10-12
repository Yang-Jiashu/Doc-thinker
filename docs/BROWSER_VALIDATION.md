# Sandboxed responsive layout validation

The `Sandboxed browser layout` job in `.github/workflows/ci.yml` runs on a
GitHub-hosted Ubuntu 24.04 VM, as the ordinary CI did in run `35417919650`.
It is independent of the Python 3.10/3.11 unit-test matrix and runs on every
pull request to `main`/`master`. No model API, application server or external
page is needed: the test renders a local template, removes remote assets, and
deliberately injects conflicting utility CSS last.

## Browser installation and sandbox

`.github/browser-requirements.txt` pins Playwright 1.63.0, which installs full
Chromium/Chrome for Testing 153.0.8010.12 (revision 1243). Playwright is only an
installer and executable locator; its browser-launch defaults are not used.
The pytest test invokes the executable directly with normal sandbox defaults.
The hosted VM image, browser version and executable/helper SHA-256 digests are
recorded as CI artifacts. The hosted VM and apt dependencies receive upstream
updates; this is a versioned browser recipe, not a byte-identical VM image.
Update the browser pin deliberately as security releases become available.

Ubuntu 24.04's AppArmor user-namespace restrictions can block downloaded
Chromium builds. The job uses the runner's existing
`/opt/google/chrome/chrome-sandbox` through `CHROME_DEVEL_SANDBOX`, following
[Chromium's documented SUID-helper option](https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md).
It checks that this helper is root-owned, executable and mode `4755`, and that
the browser runs as a non-root user. No AppArmor profile, global sysctl,
container privilege setting or sandbox-disabling switch is changed.

Before any viewport runs, a real `chrome://sandbox` launch must report:

- SUID or Namespace first-layer sandbox;
- PID and network namespace isolation;
- Seccomp-BPF sandbox enabled.

Missing layers, incompatible helpers, launch failures and timeouts fail the
job. A helper existing on disk does not constitute successful sandbox validation.
The full browser is required because the headless shell does not provide the
same internal sandbox diagnostics page.

## Execution and evidence

`DOCTHINKER_REQUIRE_BROWSER=1` turns an unset `DOCTHINKER_BROWSER_BIN` into a
pytest failure. An explicitly configured invalid executable always fails.
Each viewport (390, 768, 1280px) is a separate pytest case checking viewport
size, overflow, navigation hit areas and main/sidebar dimensions. The existing
layout assertions are preserved.

Pytest's nonzero exit status propagates directly to CI. A second check parses
JUnit XML and requires exactly the three expected passing cases with no skips,
failures or errors. `sandboxed-browser-layout` artifacts retain the JUnit report,
sandbox DOM, viewport DOMs and browser stderr for seven days, including on
failure. Treat this separate job as a required check in branch protection if
merges must be gated on browser validation.

## Local reproduction

Use a non-root account in a Linux VM with a working Chromium sandbox:

```bash
python -m venv .browser-venv
.browser-venv/bin/python -m pip install -r .github/browser-requirements.txt
.browser-venv/bin/python -m playwright install --with-deps --no-shell chromium
export DOCTHINKER_BROWSER_BIN="$(.browser-venv/bin/python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    print(p.chromium.executable_path)
PY
)"
# On Ubuntu 24.04 with an installed, root-owned Google Chrome SUID helper:
export CHROME_DEVEL_SANDBOX=/opt/google/chrome/chrome-sandbox
export DOCTHINKER_REQUIRE_BROWSER=1
export DOCTHINKER_BROWSER_ARTIFACTS="$PWD/browser-artifacts"
.browser-venv/bin/python -m pytest \
  tests/test_query_ui_controls_unit.py::test_shell_survives_late_utility_css_in_real_browser \
  -v -rA --junitxml=browser-artifacts/layout.xml
```

On Linux hosts supporting unprivileged user namespaces without AppArmor
restrictions, `CHROME_DEVEL_SANDBOX` can be omitted; the same sandbox probe
still must pass. The Linux sandbox status check is Linux-specific.

For ordinary unit tests, leave both browser variables unset and use the normal
test command. Only the optional real-browser cases skip; the offline UI,
JavaScript and mandatory-browser admission regressions still run.

## If the execution environment cannot support the sandbox

Keep the browser job failing and preserve its diagnostics. Do not add
`--no-sandbox`, disable seccomp/AppArmor globally, or turn required-browser mode
off to obtain a green check.

Use a fresh, disposable Linux VM with a non-root account and a supported SUID
helper or working user namespaces; install the same pinned requirements and
run the same pytest/JUnit checks. For automated PR execution, prefer an isolated
ephemeral VM runner with no secrets and read-only repository access, destroyed
after the job. Do not run unreviewed PR code on a persistent privileged
self-hosted runner. Record the commit SHA, image, executable version/digests,
sandbox status and all three viewport results before claiming completion.

A constrained container that cannot download a browser, expose the necessary
namespaces or run non-root cannot validate the real layout. Offline tests in
such an environment verify admission/failure behavior, not browser rendering.
