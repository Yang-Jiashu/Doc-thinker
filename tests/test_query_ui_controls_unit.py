"""Check UI controls offline, with an optional real-browser layout regression."""

import json
import os
import re
import shutil
import subprocess
from html import unescape
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

TEMPLATE = Path(__file__).parents[1] / "docthinker/ui/templates/query_modern.html"


def test_query_template_preserves_accessible_control_bindings():
    source = TEMPLATE.read_text()
    Environment().parse(source)
    for control in ("memory", "context", "cache", "evolution"):
        tag = re.search(rf'<button[^>]+id="control-{control}"[^>]*>', source).group()
        assert 'role="switch"' in tag and 'aria-checked="true"' in tag
    assert source.count('id="evolution-mode"') == 1
    assert source.index('id="evolution-mode"') < source.index('id="query-settings"')
    assert 'aria-describedby="send-keyboard-hint policy-description"' in source
    assert 'aria-labelledby="inspector-title" inert' in source
    assert 'id="request-status"' in source and 'role="status"' in source


@pytest.mark.parametrize("endpoint", ["query_page", "knowledge_graph_page", "config_page"])
def test_shared_shell_keeps_navigation_available_on_small_screens(endpoint):
    environment = Environment(loader=FileSystemLoader(TEMPLATE.parent))
    html = environment.get_template("base_modern.html").render(request={"endpoint": endpoint})
    assert 'style="width:224px' not in html
    assert '@media (max-width: 900px)' in html
    assert '.app-shell { flex-direction: column; }' in html
    for path in ("/query", "/knowledge-graph", "/config"):
        assert f'href="{path}"' in html
    assert html.count('aria-current="page"') == 1
    assert 'aria-label="主导航"' in html


def test_shell_survives_late_utility_css_in_real_browser(tmp_path):
    """Exercise CSS cascade/layout, not just the presence of responsive rules.

    Opt in with DOCTHINKER_BROWSER_BIN in a sandbox-capable environment.
    No browser download, package install, network request, or model call occurs.
    """
    # A browser on PATH is not proof its sandbox is usable (e.g. CI AppArmor
    # restrictions). Keep this explicit opt-in, and never add --no-sandbox or
    # suppress launch/layout failures when a browser was explicitly configured.
    browser = os.environ.get("DOCTHINKER_BROWSER_BIN")
    if not browser:
        pytest.skip("Set DOCTHINKER_BROWSER_BIN to run the optional sandboxed layout check")
    environment = Environment(loader=FileSystemLoader(TEMPLATE.parent))
    page = environment.get_template("base_modern.html").render(request={"endpoint": "query_page"})
    page = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", page)
    page = re.sub(r"<(?:link|img)\b[^>]*>", "", page)
    # Reintroduce the former conflicting utility classes deliberately, and load
    # their CSS last, matching the late-injected Tailwind CDN failure mode.
    page = page.replace('class="app-shell"', 'class="app-shell flex h-full"')
    page = page.replace('class="app-sidebar ', 'class="app-sidebar flex flex-col h-full ')
    page = page.replace('class="app-main ', 'class="app-main flex flex-col h-full ')
    page = page.replace("</head>", """<style>
        .flex { display: flex; } .flex-col { flex-direction: column; }
        .h-full { height: 100%; } .flex-1 { flex: 1 1 0%; }
    </style></head>""")
    page = page.replace("</body>", """<script>
        const rect = selector => {
            const r = document.querySelector(selector).getBoundingClientRect();
            return {x: r.x, y: r.y, width: r.width, height: r.height};
        };
        const report = document.createElement('pre');
        report.id = 'layout-report';
        report.textContent = JSON.stringify({
            viewport: {width: innerWidth, height: innerHeight},
            sidebar: rect('.app-sidebar'), main: rect('.app-main'),
            direction: getComputedStyle(document.querySelector('.app-sidebar')).flexDirection,
            scrollWidth: document.documentElement.scrollWidth,
            links: Array.from(document.querySelectorAll('nav a'), link => {
                const r = link.getBoundingClientRect();
                return {left: r.left, right: r.right, height: r.height};
            })
        });
        document.body.appendChild(report);
    </script></body>""")
    fixture = tmp_path / "responsive-shell.html"
    fixture.write_text(page)
    for width in (390, 768, 1280):
        result = subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-first-run",
             "--disable-background-networking", "--force-device-scale-factor=1",
             f"--user-data-dir={tmp_path / str(width)}", f"--window-size={width},844",
             "--dump-dom", fixture.as_uri()],
            text=True, capture_output=True, check=False, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        report = re.search(r'<pre id="layout-report">(.*?)</pre>', result.stdout, re.DOTALL)
        assert report, result.stderr
        layout = json.loads(unescape(report.group(1)))
        viewport = layout["viewport"]
        assert viewport["width"] == width, layout
        assert layout["scrollWidth"] <= width, layout
        assert layout["main"]["height"] > viewport["height"] * 0.75, layout
        assert len(layout["links"]) == 3
        if width <= 900:
            assert layout["direction"] == "row", layout
            assert 44 <= layout["sidebar"]["height"] < 100, layout
            assert layout["main"]["width"] == width, layout
            assert layout["main"]["y"] == layout["sidebar"]["height"], layout
            assert all(link["height"] >= 44 and link["left"] >= 0 and link["right"] <= width
                       for link in layout["links"]), layout
        else:
            assert layout["direction"] == "column", layout
            assert layout["sidebar"]["width"] == 224, layout
            assert layout["main"]["width"] == width - 224, layout


def test_query_controls_ime_switches_and_mobile_inspector():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the UI JavaScript regression")
    script = re.search(r"<script>([\s\S]*?)</script>", TEMPLATE.read_text()).group(1)
    driver = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(0, 'utf8');
new Function(source);
const elements = new Map();
const listeners = {};
let mobile = false;
function element(id) {
    if (elements.has(id)) return elements.get(id);
    const classes = new Set(id === 'memory-inspector' ? ['is-collapsed'] : []);
    const attributes = {};
    const item = {
        id, value: '', checked: false, disabled: false, inert: true,
        textContent: '', innerHTML: '', className: '', dataset: {}, style: {},
        classList: {
            add: (...names) => names.forEach(name => classes.add(name)),
            remove: (...names) => names.forEach(name => classes.delete(name)),
            contains: name => classes.has(name),
            toggle: (name, state) => { state ??= !classes.has(name); state ? classes.add(name) : classes.delete(name); return state; },
        },
        setAttribute: (key, value) => { attributes[key] = value; },
        getAttribute: key => attributes[key],
        removeAttribute: key => { delete attributes[key]; },
        focus() { document.activeElement = item; },
        contains(target) { return target === item || target === element('inspector-close'); },
        querySelector(selector) { return element(selector === '.welcome-panel' ? 'welcome-panel' : 'inspector-close'); },
        querySelectorAll() { return [element('inspector-close')]; },
        closest() { return element(id + '-section'); },
        appendChild() {},
        remove() { item.removed = true; },
        dispatchEvent() {},
    };
    elements.set(id, item);
    return item;
}
const document = {
    body: {}, activeElement: {},
    getElementById: element,
    createElement: () => element('created-message'),
    querySelectorAll: () => ['quick', 'standard', 'deep'].map(mode => element('mode-' + mode)),
    addEventListener: (event, handler) => { listeners[event] = handler; },
};
const context = vm.createContext({ document, window: { matchMedia: () => ({ matches: mobile }) }, console, Event: class {} });
vm.runInContext(source, context);
const run = expression => vm.runInContext(expression, context);
element('evolution-mode').value = 'auto';
element('adaptive-context').checked = true;
run('updateQueryControls()');
assert.equal(element('settings-summary').textContent, '默认设置');
run("toggleRunControl('memory')");
assert.equal(element('control-memory').getAttribute('aria-checked'), 'false');
assert.equal(run('rememberTurn'), false);
assert.equal(element('remember-toggle-text').textContent, '开启记忆');
run('toggleRememberTurn()');
assert.equal(element('control-memory').getAttribute('aria-checked'), 'true');
assert.equal(run('rememberTurn'), true);
element('path-completion').checked = true;
element('include-discovered').checked = true;
element('evolution-mode').value = 'faithful';
run('updateQueryControls()');
assert.equal(run("getEnabledCheckbox('path-completion')"), false);
assert.equal(run("getEnabledCheckbox('include-discovered')"), false);
element('evolution-mode').value = 'path';
run('updateQueryControls()');
assert.equal(run("getEnabledCheckbox('path-completion')"), true);
run("toggleRunControl('evolution')");
assert.equal(run("getEnabledCheckbox('path-completion')"), false);
run('resetQueryControls()');
assert.equal(element('settings-summary').textContent, '默认设置');
assert.equal(element('mode-standard').getAttribute('aria-pressed'), 'true');

run('var sends = 0; sendMessage = () => { sends += 1; };');
context.event = { key: 'Enter', shiftKey: false, isComposing: true, preventDefault() { throw Error('IME interrupted'); } };
run('handleKeyDown(event)');
context.event = { key: 'Enter', shiftKey: false, keyCode: 229, preventDefault() { throw Error('IME interrupted'); } };
run('handleKeyDown(event)');
context.event = { key: 'Enter', shiftKey: true, preventDefault() { throw Error('Newline interrupted'); } };
run('handleKeyDown(event)');
assert.equal(run('sends'), 0);
context.event = { key: 'Enter', shiftKey: false, preventDefault() {} };
run('handleKeyDown(event)');
assert.equal(run('sends'), 1);

run("renderMemoryInspector({ question_policy: { mode: 'faithful' }, context_budget: { retrieval_limits: { max_total_tokens: 12000, chunk_top_k: 8, max_relations: 16 } }, memory_trace: { events: [{type: 'long_horizon_recall', count: 1}] } })");
assert.ok(element('memory-inspector').classList.contains('is-collapsed'), 'evidence must not take focus automatically');
assert.ok(element('memory-budget').innerHTML.includes('12,000'));
mobile = true;
document.activeElement = element('memory-inspector-toggle');
run('setMemoryInspectorOpen(true)');
assert.equal(element('memory-inspector').inert, false);
assert.equal(element('memory-inspector').getAttribute('aria-modal'), 'true');
assert.equal(document.activeElement, element('inspector-close'));
listeners.keydown({ key: 'Escape', preventDefault() {} });
assert.equal(element('memory-inspector').inert, true);
assert.equal(document.activeElement, element('memory-inspector-toggle'));
run("setRequestStatus('busy', '正在检索')");
assert.equal(element('chat-messages').getAttribute('aria-busy'), 'true');
run("setRequestStatus('done', '已完成')");
assert.equal(element('chat-messages').getAttribute('aria-busy'), 'false');
run("addMessage('first message', 'user')");
assert.equal(element('welcome-panel').removed, true);
run("defaultWelcomeHtml = '<div>Welcome</div>'; renderWelcome()");
assert.equal(element('chat-messages').innerHTML, '<div>Welcome</div>');
"""
    result = subprocess.run(
        [node, "-e", driver], input=script, text=True, capture_output=True,
        check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr
