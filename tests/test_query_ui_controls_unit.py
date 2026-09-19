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
STATIC = TEMPLATE.parent.parent / "static"


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
    assert '{% block sidebar_content %}' in source
    assert 'id="sidebar-session-list"' in source
    assert 'id="current-session-title"' in source
    assert 'class="composer-bar"' in source
    assert 'class="welcome-mark"' not in source
    assert '<style>' not in source, 'Query theme should have one maintained local stylesheet'
    css = (STATIC / 'query-workspace.css').read_text()
    assert 'linear-gradient' not in css
    assert '.chat-column.is-empty' in css


@pytest.mark.parametrize("endpoint", ["query_page", "knowledge_graph_page", "config_page"])
def test_shared_shell_keeps_navigation_available_on_small_screens(endpoint):
    environment = Environment(loader=FileSystemLoader(TEMPLATE.parent))
    html = environment.get_template("base_modern.html").render(request={"endpoint": endpoint})
    assert 'style="width:224px' not in html
    css = (STATIC / "workspace.css").read_text()
    assert '@media (max-width: 900px)' in css
    assert '.app-shell { flex-direction: column; }' in css
    assert "font-size: 16px" in css
    assert "linear-gradient" not in css
    assert "Georgia" not in css
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
    page = page.replace('</head>', '<style>' + (STATIC / 'workspace.css').read_text() + '</style></head>')
    page = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", page)
    page = re.sub(r"<(?:link|img)\b[^>]*>", "", page)
    # Reintroduce the former conflicting utility classes deliberately, and load
    # their CSS last, matching the late-injected Tailwind CDN failure mode.
    page = page.replace('class="app-shell"', 'class="app-shell flex h-full"')
    page = page.replace('class="app-sidebar"', 'class="app-sidebar flex flex-col h-full"')
    page = page.replace('class="app-main"', 'class="app-main flex flex-col h-full"')
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
            assert layout["sidebar"]["width"] == 248, layout
            assert layout["main"]["width"] == width - 248, layout


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
const toasts = [];
let mobile = false;
let createdCount = 0;
function element(id) {
    if (elements.has(id)) return elements.get(id);
    const classes = new Set(id === 'memory-inspector' ? ['is-collapsed'] : []);
    const attributes = {};
    const item = {
        id, value: '', checked: false, disabled: false, inert: true,
        textContent: '', innerHTML: '', className: '', dataset: {}, style: {}, children: [],
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
        appendChild(child) { item.children.push(child); },
        addEventListener(event, handler) { item[event] = handler; },
        remove() { item.removed = true; },
        dispatchEvent() {},
    };
    elements.set(id, item);
    return item;
}
const document = {
    body: {}, activeElement: {},
    getElementById: element,
    createElement: () => element('created-message-' + (++createdCount)),
    querySelectorAll: () => ['quick', 'standard', 'deep'].map(mode => element('mode-' + mode)),
    addEventListener: (event, handler) => { listeners[event] = handler; },
};
const context = vm.createContext({ document, window: { matchMedia: () => ({ matches: mobile }) }, console, showToast: (...args) => toasts.push(args), Event: class {} });
vm.runInContext(source, context);
const run = expression => vm.runInContext(expression, context);
element('evolution-mode').value = 'auto';
element('adaptive-context').checked = true;
run('updateQueryControls()');
assert.equal(element('settings-summary').textContent, '默认设置');
run("toggleRunControl('memory')");
assert.equal(element('control-memory').getAttribute('aria-checked'), 'false');
assert.equal(run('rememberTurn'), false);
assert.equal(element('remember-toggle-text').textContent, '不记录记忆');
assert.equal(element('remember-toggle').disabled, true);
run('toggleRememberTurn()');
assert.equal(element('control-memory').getAttribute('aria-checked'), 'false', 'writeback toggle must not silently enable memory retrieval');
run("toggleRunControl('memory')");
assert.equal(element('control-memory').getAttribute('aria-checked'), 'true');
assert.equal(run('rememberTurn'), true);
assert.equal(element('remember-toggle').disabled, false);
assert.equal(element('remember-toggle-text').textContent, '记录记忆');
run('toggleRememberTurn()');
assert.equal(element('control-memory').getAttribute('aria-checked'), 'true');
assert.equal(run('rememberTurn'), false);
assert.equal(element('remember-toggle-text').textContent, '不记录记忆');
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
assert.ok(element('chat-column').classList.contains('is-empty'));

run("currentSessionId = '#one'; renderSessionSelector([{id:'#one', title:'第一份资料'}, {id:'#two', title:'第二份资料'}], '#one')");
assert.equal(element('current-session-title').textContent, '第一份资料');
assert.equal(element('sidebar-session-list').children.length, 2);
assert.equal(element('sidebar-session-list').children[0].getAttribute('aria-current'), 'page');
assert.equal(element('sidebar-session-list').children[1].textContent, '第二份资料');
element('message-input').value = '';
run("restoreFailedDraft('#one', '失败时保留这个问题')");
assert.equal(element('message-input').value, '失败时保留这个问题');
element('message-input').value = '正在编辑的新草稿';
run("restoreFailedDraft('#one', '旧问题')");
assert.equal(element('message-input').value, '正在编辑的新草稿');
run("restoreFailedDraft('#two', '别的会话')");
assert.equal(element('message-input').value, '正在编辑的新草稿');
run("_isSending = true; handleSessionSwitch('#two')");
assert.equal(run('currentSessionId'), '#one');
assert.equal(element('session-selector').value, '#one');
assert.ok(toasts.length);
run("_isSending = false; loadHistory = async () => {}; sessionDrafts.set('#two', '第二份草稿'); handleSessionSwitch('#two')");
assert.equal(run('currentSessionId'), '#two');
assert.equal(element('message-input').value, '第二份草稿');
assert.equal(run("sessionDrafts.get('#one')"), '正在编辑的新草稿');
const chat = element('chat-messages');
chat.scrollTop = 100; chat.scrollHeight = 2000;
run('followLatestMessage = false; scrollChatToLatest()');
assert.equal(chat.scrollTop, 100, 'new stream chunks must not interrupt reading earlier messages');
run('scrollChatToLatest(true)');
assert.equal(chat.scrollTop, 2000);
"""
    result = subprocess.run(
        [node, "-e", driver], input=script, text=True, capture_output=True,
        check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_shared_toast_is_text_only_and_cdn_failure_is_nonfatal():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the UI JavaScript regression")
    template = (TEMPLATE.parent / "base_modern.html").read_text()
    assert 'fonts.googleapis.com' not in template
    script = '\n'.join(re.findall(r'<script>([\s\S]*?)</script>', template))
    driver = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(0, 'utf8');
function element(tag) {
    return {
        tag, children: [], attributes: {}, textContent: '',
        appendChild(child) { this.children.push(child); },
        setAttribute(name, value) { this.attributes[name] = value; },
        set innerHTML(value) { throw Error('Toast must never interpret HTML'); },
    };
}
const container = element('div');
const context = vm.createContext({
    window: {}, // Tailwind CDN is unavailable.
    document: { createElement: element, getElementById: () => container },
    setTimeout() {},
});
vm.runInContext(source, context);
context.message = '<img src=x onerror="alert(1)">';
vm.runInContext("showToast(message, 'error')", context);
const toast = container.children[0];
assert.equal(toast.attributes.role, 'alert');
assert.equal(toast.children[1].tag, 'span');
assert.equal(toast.children[1].textContent, context.message);
assert.equal(toast.children[1].children.length, 0);
"""
    result = subprocess.run(
        [node, "-e", driver], input=script, text=True, capture_output=True,
        check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr
