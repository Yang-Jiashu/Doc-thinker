"""Run the settings controller offline against controls from its real template.

The small DOM double deliberately does not invent missing IDs. This exercises
event wiring and state changes, not layout or browser-native form validation.
Fetch accepts only the two local settings/catalog routes; no provider is called.
"""

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "docthinker" / "ui"
SCRIPT = UI / "static" / "model-settings.js"
CATALOG = json.loads((UI / "static" / "model_catalog.json").read_text())
SETTINGS = {
    "llm_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-4.1",
    "llm_models": ["gpt-4.1", "gpt-4o-mini"],
    "vlm_model": "gpt-4o",
    "keyword_llm_model": "gpt-4.1-mini",
    "entity_extraction_llm_model": "gpt-4o-mini",
    "extraction_llm_model": "legacy-hidden-override",
    "llm_api_key_masked": "test…llm",
    "embed_base_url": "https://api.openai.com/v1",
    "embed_model": "text-embedding-3-small",
    "embed_dim": 1536,
    "embed_api_key_masked": "test…embed",
    "rerank_base_url": "https://api.jina.ai/v1",
    "rerank_model": "jina-reranker-v3",
    "rerank_api_key_masked": "test…rerank",
    "llm_max_async": 4,
    "embedding_max_async": 4,
    "max_parallel_insert": 2,
    "llm_router_max_concurrency": 4,
    "requires_restart": False,
    "warnings": [],
    "workdir": "test-data",
}


class TemplateControls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append({"tag": tag, "attrs": dict(attrs)})


@pytest.fixture(scope="module")
def rendered_controls():
    env = Environment(loader=FileSystemLoader(UI / "templates"))
    page = env.get_template("config_modern.html").render(
        request={"endpoint": "config_page"}
    )
    parser = TemplateControls()
    parser.feed(page)
    return parser.elements


DOM_RUNNER = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const clone = value => JSON.parse(JSON.stringify(value));
class Element {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase(); this.attrs = {...attrs};
    this.id = attrs.id || ''; this.dataset = {}; this.children = [];
    this.listeners = {}; this.hidden = 'hidden' in attrs;
    this.disabled = 'disabled' in attrs; this.checked = 'checked' in attrs;
    this.open = 'open' in attrs; this.textContent = ''; this._value = attrs.value || '';
    this.classes = new Set();
    this.classList = {toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name)};
    for (const [name, value] of Object.entries(attrs)) {
      if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())] = value;
    }
  }
  get value() { return this._value; }
  set value(value) { this._value = String(value); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  start(type) {
    const event = {target: this, preventDefault() { this.prevented = true; }};
    return (this.listeners[type] || []).map(handler => handler(event));
  }
  async fire(type) { await Promise.all(this.start(type)); }
  showModal() { this.open = true; }
  close() { this.open = false; this.start('close'); }
  focus() { focused = this; }
  scrollIntoView() {}
}
let focused = null;
const elements = input.elements.map(item => new Element(item.tag, item.attrs));
const ids = new Map(elements.filter(item => item.id).map(item => [item.id, item]));
const docListeners = {};
const document = {
  getElementById: id => ids.get(id) || null,
  createElement: tag => new Element(tag),
  querySelectorAll(selector) {
    const clauses = [...selector.matchAll(/\[([^=\]]+)(?:="([^"]*)")?\]/g)];
    assert.ok(clauses.length, 'Unexpected selector: ' + selector);
    return elements.filter(element => clauses.every(([, name, value]) =>
      name in element.attrs && (value === undefined || element.attrs[name] === value)));
  },
  addEventListener(type, handler) { (docListeners[type] ||= []).push(handler); }
};
const $ = id => { assert.ok(ids.has(id), 'Test references a missing template ID: ' + id); return ids.get(id); };
const field = name => $('s-' + name.replaceAll('_', '-'));
const modelButton = name => elements.find(element => element.dataset.modelField === name);
const requests = [];
let postHandler = async () => ({ok: true, json: async () => ({requires_restart: true})});
const context = {
  URL, document, console,
  addEventListener() {},
  async fetch(url, options = {}) {
    requests.push({url, options});
    assert.ok(['/api/v1/settings', '/static/model_catalog.json'].includes(url), 'Unexpected network destination');
    if (url === '/static/model_catalog.json') {
      if (input.failCatalog) throw new Error('catalog unavailable');
      return {ok: true, json: async () => clone(input.catalog)};
    }
    if (options.method === 'POST') return postHandler(JSON.parse(options.body));
    return {ok: true, json: async () => clone(input.settings)};
  }
};
context.window = context;
vm.runInNewContext(input.source, context, {filename: 'model-settings.js'});
const flush = async () => { for (let i = 0; i < 4; i++) await new Promise(setImmediate); };
const posts = () => requests.filter(request => request.options.method === 'POST').map(request => JSON.parse(request.options.body));
const switchProvider = async (group, id) => {
  $('s-' + group + '-provider').value = id;
  await $('s-' + group + '-provider').fire('change');
};
(async () => {
  for (const handler of docListeners.DOMContentLoaded || []) handler();
  await flush();
  assert.equal($('settings-form').hidden, false, $('settings-loading').textContent);
  assert.equal($('settings-loading').hidden, true);
  await (async () => { SCRIPT_BODY })();
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
"""


def _run_controller(rendered_controls, body, *, settings=None, fail_catalog=False):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for offline controller execution")
    result = subprocess.run(
        [node, "-e", DOM_RUNNER.replace("SCRIPT_BODY", body)],
        input=json.dumps({
            "source": SCRIPT.read_text(),
            "catalog": CATALOG,
            "settings": settings or SETTINGS,
            "elements": rendered_controls,
            "failCatalog": fail_catalog,
        }),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_rendered_template_has_unique_controls_and_valid_picker_names(rendered_controls):
    ids = [e["attrs"]["id"] for e in rendered_controls if "id" in e["attrs"]]
    assert len(ids) == len(set(ids))
    assert {"llm-provider-description", "embed-provider-description", "rerank-provider-description"} <= set(ids)
    pickers = [e["attrs"] for e in rendered_controls if "data-model-field" in e["attrs"]]
    assert len(pickers) == 6
    for picker in pickers:
        assert set(picker["aria-labelledby"].split()) <= set(ids)
        assert "s-" + picker["data-model-field"].replace("_", "-") in ids
    keys = [e for e in rendered_controls if e["attrs"].get("id", "").endswith("-api-key")]
    assert len(keys) == 3
    assert all(e["attrs"]["type"] == "password" for e in keys)
    assert any(e["attrs"].get("src", "").startswith("/static/model-settings.js") for e in rendered_controls)


def test_load_preserves_existing_multi_model_and_auxiliary_assignments(rendered_controls):
    _run_controller(rendered_controls, r"""
      assert.equal(field('llm_model').value, 'gpt-4.1, gpt-4o-mini');
      assert.equal(field('vlm_model').value, 'gpt-4o');
      assert.equal(field('keyword_llm_model').value, 'gpt-4.1-mini');
      assert.equal(field('entity_extraction_llm_model').value, 'gpt-4o-mini');
      assert.equal($('s-llm-provider').value, 'openai');
      assert.equal($('s-llm-api-key').value, '');
      assert.equal($('save-btn').disabled, true);
      assert.equal(requests.length, 2);
      field('llm_max_async').value = '7';
      await $('settings-form').fire('submit');
      assert.deepEqual(posts(), [{llm_max_async: 7}]);
      assert.equal(field('llm_model').value, 'gpt-4.1, gpt-4o-mini');
    """)


def test_switch_provider_clears_typed_key_and_replaces_all_llm_roles(rendered_controls):
    _run_controller(rendered_controls, r"""
      $('s-llm-api-key').value = 'typed-for-old-provider';
      await switchProvider('llm', 'deepseek');
      assert.equal($('s-llm-api-key').value, '');
      assert.equal(field('llm_base_url').value, 'https://api.deepseek.com');
      assert.equal(field('llm_model').value, 'deepseek-flash');
      for (const name of ['vlm_model', 'keyword_llm_model', 'entity_extraction_llm_model']) assert.equal(field(name).value, '');
      assert.equal(field('embed_model').value, 'text-embedding-3-small');
      assert.ok($('llm-warning').textContent.includes('未提供独立的抽取预设'));
      await $('settings-form').fire('submit');
      assert.equal(posts().length, 0);
      assert.ok($('settings-notice').textContent.includes('API Key'));
      $('s-llm-api-key').value = 'new-provider-key';
      await $('settings-form').fire('submit');
      const saved = posts()[0];
      assert.equal(saved.llm_api_key, 'new-provider-key');
      assert.equal(saved.extraction_llm_model, '');
      assert.equal(saved.keyword_llm_model, '');
      assert.equal(saved.vlm_model, '');
    """)


def test_custom_mode_preserves_existing_models_and_manual_destination_clears_key(rendered_controls):
    _run_controller(rendered_controls, r"""
      await switchProvider('llm', 'custom');
      assert.equal(field('llm_model').value, 'gpt-4.1, gpt-4o-mini');
      assert.equal(field('vlm_model').value, 'gpt-4o');
      assert.equal($('llm-advanced').open, true);
      $('s-llm-api-key').value = 'not-for-new-address';
      field('llm_base_url').value = 'https://custom.example/v1';
      await field('llm_base_url').fire('input');
      assert.equal($('s-llm-api-key').value, '');
      assert.equal($('s-llm-provider').value, 'custom');
      await modelButton('llm_model').fire('click');
      $('model-custom-id').value = ' custom-main, custom-backup ';
      await $('model-custom-apply').fire('click');
      assert.equal(field('llm_model').value, 'custom-main, custom-backup');
      assert.equal($('model-dialog').open, false);
    """)


def test_picker_filters_roles_and_search_without_showing_embedding_as_chat(rendered_controls):
    _run_controller(rendered_controls, r"""
      const modelIDs = () => $('model-results').children.filter(e => e.tagName === 'BUTTON').map(e => e.children[1].textContent);
      await modelButton('vlm_model').fire('click');
      assert.equal($('model-dialog').open, true);
      assert.ok(modelIDs().includes(''));
      assert.ok(modelIDs().includes('gpt-4o'));
      assert.ok(!modelIDs().some(id => id.startsWith('text-embedding')));
      $('model-search').value = '4.1-MINI';
      await $('model-search').fire('input');
      assert.deepEqual(modelIDs(), ['gpt-4.1-mini']);
      await $('model-results').children[0].fire('click');
      assert.equal(field('vlm_model').value, 'gpt-4.1-mini');
      assert.equal(focused, modelButton('vlm_model'));
      await modelButton('embed_model').fire('click');
      assert.ok(modelIDs().every(id => id.startsWith('text-embedding')));
      assert.ok(!modelIDs().includes(''));
    """)


def test_base_only_migration_sends_existing_models_and_explicit_inheritance(rendered_controls):
    settings = {**SETTINGS, "vlm_model": "", "entity_extraction_llm_model": ""}
    _run_controller(rendered_controls, r"""
      field('llm_base_url').value = 'https://new-gateway.example/v1';
      await field('llm_base_url').fire('input');
      $('s-llm-api-key').value = 'new-gateway-key';
      await $('settings-form').fire('submit');
      assert.deepEqual(posts(), [{
        llm_base_url: 'https://new-gateway.example/v1',
        llm_api_key: 'new-gateway-key',
        llm_model: 'gpt-4.1, gpt-4o-mini',
        vlm_model: '',
        keyword_llm_model: 'gpt-4.1-mini',
        entity_extraction_llm_model: '',
        extraction_llm_model: ''
      }]);
      assert.equal($('settings-notice').classes.has('ms-error'), false);
    """, settings=settings)


def test_unknown_embedding_dimension_is_cleared_and_blocks_save(rendered_controls):
    _run_controller(rendered_controls, r"""
      await switchProvider('embed', 'siliconflow');
      assert.equal(field('embed_dim').value, '');
      $('s-embed-api-key').value = 'new-embedding-key';
      await $('settings-form').fire('submit');
      assert.equal(posts().length, 0);
      assert.ok($('settings-notice').textContent.includes('向量维度'));
      assert.ok($('embed-warning').textContent.includes('重新导入文档'));
      field('embed_dim').value = '1024';
      await $('settings-form').fire('submit');
      assert.equal(posts()[0].embed_dim, 1024);
      assert.equal(posts()[0].embed_model, 'BAAI/bge-large-zh-v1.5');
    """)


def test_workspace_template_endpoint_blocks_save_until_completed(rendered_controls):
    _run_controller(rendered_controls, r"""
      await switchProvider('llm', 'dashscope');
      assert.equal($('llm-advanced').open, true);
      assert.ok($('llm-warning').textContent.includes('{WorkspaceId}'));
      $('s-llm-api-key').value = 'workspace-key';
      await $('settings-form').fire('submit');
      assert.equal(posts().length, 0);
      assert.ok($('settings-notice').textContent.includes('占位符'));
    """)


def test_disabling_rerank_ignores_new_destination_and_unsaved_key(rendered_controls):
    _run_controller(rendered_controls, r"""
      await switchProvider('rerank', 'siliconflow');
      $('s-rerank-api-key').value = 'should-not-be-submitted';
      $('s-rerank-enabled').checked = false;
      await $('s-rerank-enabled').fire('change');
      assert.equal($('rerank-controls').hidden, true);
      await $('settings-form').fire('submit');
      assert.deepEqual(posts(), [{rerank_model: ''}]);
      assert.equal(field('rerank_base_url').value, 'https://api.jina.ai/v1');
      assert.equal($('s-rerank-api-key').value, '');
      assert.equal($('s-rerank-enabled').checked, false);
    """)


def test_save_disables_controls_prevents_duplicate_and_reports_pending_restart(rendered_controls):
    _run_controller(rendered_controls, r"""
      let finish;
      postHandler = () => new Promise(resolve => { finish = resolve; });
      field('llm_max_async').value = '8';
      const inFlight = $('settings-form').start('submit');
      await flush();
      assert.equal($('settings-fields').disabled, true);
      assert.equal($('save-btn').disabled, true);
      assert.equal($('reset-btn').disabled, true);
      assert.equal($('settings-dirty').textContent, '正在保存…');
      await $('settings-form').fire('submit');
      assert.equal(posts().length, 1);
      finish({ok: true, json: async () => ({requires_restart: true, warnings: ['测试重启提示']})});
      await Promise.all(inFlight);
      assert.equal($('settings-fields').disabled, false);
      assert.equal($('save-btn').disabled, true);
      assert.ok($('settings-dirty').textContent.includes('等待后端重启'));
      assert.ok($('settings-notice').textContent.includes('当前运行中的客户端仍使用旧配置'));
      assert.ok($('settings-notice').textContent.includes('测试重启提示'));
      assert.equal(requests.filter(r => !r.options.method && r.url === '/api/v1/settings').length, 1);
    """)


def test_failed_save_preserves_user_input_and_does_not_echo_validation_secret(rendered_controls):
    _run_controller(rendered_controls, r"""
      field('llm_max_async').value = '9';
      $('s-llm-api-key').value = 'typed-key-retained-for-retry';
      postHandler = async () => ({ok: false, json: async () => ({detail: [{msg: '服务器拒绝配置', input: 'private-server-input'}]})});
      await $('settings-form').fire('submit');
      assert.equal(field('llm_max_async').value, '9');
      assert.equal($('s-llm-api-key').value, 'typed-key-retained-for-retry');
      assert.equal($('settings-fields').disabled, false);
      assert.equal($('save-btn').disabled, false);
      assert.ok($('settings-notice').textContent.includes('保存失败：服务器拒绝配置'));
      assert.ok(!$('settings-notice').textContent.includes('private-server-input'));
      assert.ok(!$('settings-notice').textContent.includes('typed-key-retained-for-retry'));
      await $('reset-btn').fire('click');
      assert.equal(field('llm_max_async').value, '4');
      assert.equal($('s-llm-api-key').value, '');
      assert.equal($('save-btn').disabled, true);
    """)


def test_catalog_failure_preserves_custom_configuration_and_save(rendered_controls):
    _run_controller(rendered_controls, r"""
      assert.equal($('s-llm-provider').value, 'custom');
      assert.equal(field('llm_model').value, 'gpt-4.1, gpt-4o-mini');
      assert.ok($('catalog-notice').textContent.includes('预设暂时无法加载'));
      field('llm_max_async').value = '5';
      await $('settings-form').fire('submit');
      assert.deepEqual(posts(), [{llm_max_async: 5}]);
    """, fail_catalog=True)


def test_pure_helpers_preserve_security_boundary_and_minimal_diff(rendered_controls):
    _run_controller(rendered_controls, r"""
      const api = context.ModelSettings;
      assert.equal(api.normalizeEndpoint('https://API.OPENAI.COM:443/v1///'), 'https://api.openai.com/v1');
      assert.equal(api.providerFor(input.catalog, 'llm', 'https://abc-123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')?.id, 'dashscope');
      assert.equal(api.providerFor(input.catalog, 'llm', 'https://abc.cn-beijing.maas.aliyuncs.com.evil.example/compatible-mode/v1'), undefined);
      assert.equal(api.providerFor(input.catalog, 'llm', 'https://api.openai.com:444/v1'), undefined);
      const diff = api.buildPayload(input.settings, {...input.settings, llm_max_async: '7'}, {llm: '  test-key  ', embed: '', rerank: ''});
      assert.deepEqual(clone(diff), {llm_max_async: 7, llm_api_key: 'test-key'});
      const changed = api.buildPayload(input.settings, {llm_model: 'custom'}, {});
      assert.equal(changed.extraction_llm_model, '');
    """)
