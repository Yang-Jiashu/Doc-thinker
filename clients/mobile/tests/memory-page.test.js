import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, effectScope } from 'vue';

const filename = new URL('../src/pages/memory/index.vue', import.meta.url);
const source = readFileSync(filename, 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; }
const record = (id = 'm-1', summary = '助手整理的示例摘要') => ({ id, kind: 'insight', summary, question: '用户当时的问题', source_type: 'assistant_summary', question_source: 'user_message', created_at: 1700000000, truncated_fields: [] });
const page = (items = [record()], next = null) => ({ items, scope: 'session', has_more: next !== null, next_offset: next });

function mount(overrides = {}) {
  const { descriptor, errors } = parse(source, { filename: String(filename) });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: 'memory' });
  const template = compileTemplate({ source: descriptor.template.content, filename: String(filename), id: 'memory', compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(template.errors, []);
  let active = 'a';
  const hooks = {}, calls = { memory: [], navigate: [] };
  const api = {
    sessions: async () => [{ id: 'a', title: 'A' }, { id: 'b', title: 'B' }],
    memory: async (...args) => { calls.memory.push(args); return page(); },
    ...overrides,
  };
  const scope = {
    ref, computed, api, research:{capabilities:async()=>({enabled:false})},
    getPreferences: () => ({ mode: 'demo', baseUrl: '' }),
    getActiveSession: () => active, setActiveSession: value => { active = value; },
    onShow: fn => { hooks.show = fn; }, onHide: fn => { hooks.hide = fn; }, onUnload: fn => { hooks.unload = fn; },
    BrandHeader: {}, WorkspaceBar: {}, ModeBanner: {}, CapabilityBoundary: {}, MemoryObservatory: {},
    uni: { switchTab: value => calls.navigate.push(value) },
  };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const effects = effectScope();
  const ui = effects.run(() => component.setup({}, { expose() {} }));
  return { ui, api, hooks, calls, active: value => { active = value; }, dispose() { hooks.hide?.(); effects.stop(); } };
}

test('memory page opens with only session-scoped read requests and sources remain distinct', async () => {
  const f = mount(); f.hooks.show(); await settle();
  assert.deepEqual(f.calls.memory, [['a', { limit: 20, offset: 0 }]]);
  assert.equal(f.ui.items.value[0].summary, '助手整理的示例摘要');
  assert.deepEqual(f.ui.expandedIds.value, []);
  f.ui.toggleQuestion('m-1'); assert.deepEqual(f.ui.expandedIds.value, ['m-1']);
  f.ui.toggleQuestion('m-1'); assert.deepEqual(f.ui.expandedIds.value, []);
  assert.match(source, /助手整理摘要/); assert.match(source, /用户当时的提问/);
  assert.match(source, /不代表用户确认过上方摘要/);
  assert.doesNotMatch(source, /api\.(ask|upload|delete|update)|startRecording|confidence|v-html/);
  f.dispose();
});

test('unknown explicitly selected session never falls back to another session memory', async () => {
  const f = mount(); f.active('deleted-session'); f.hooks.show(); await settle();
  assert.equal(f.ui.sessionId.value, ''); assert.equal(f.calls.memory.length, 0);
  assert.deepEqual(f.ui.items.value, []); f.dispose();
});

test('empty store is honestly empty and missing backend never turns into demo success', async () => {
  const f = mount({ memory: async () => page([]) }); f.hooks.show(); await settle();
  assert.deepEqual(f.ui.items.value, []); assert.equal(f.ui.error.value, '');
  f.api.memory = async () => { throw new Error('Service unavailable'); };
  await f.ui.reload(); assert.deepEqual(f.ui.items.value, []); assert.equal(f.ui.error.value, 'Service unavailable'); f.dispose();
});

test('late session read and paginated result cannot cross a session or revive a hidden page', async () => {
  const pendingA = deferred(), pendingMore = deferred();
  const f = mount({ memory: (sid, options) => sid === 'a' ? pendingA.promise : options.offset ? pendingMore.promise : Promise.resolve(page([record('b-1', '仅 B 的摘要')], 20)) });
  f.hooks.show(); await settle(); await f.ui.changeSession('b');
  pendingA.resolve(page([record('a-1', '不能显示的 A 摘要')])); await settle();
  assert.deepEqual(f.ui.items.value.map(item => item.id), ['b-1']);
  const more = f.ui.loadMore(); f.hooks.hide(); pendingMore.resolve(page([record('b-2')])); await more;
  assert.equal(f.ui.sessionId.value, ''); assert.deepEqual(f.ui.items.value, []); assert.equal(f.ui.loadingMore.value, false); f.dispose();
});

test('load more is bounded, singleflight, deduplicated and local search does not trigger calls', async () => {
  const pending = deferred(); let loads = 0;
  const f = mount({ memory: async (_sid, options) => { loads += 1; return options.offset ? pending.promise : page([record('first', '可再生能源')], 20); } });
  f.hooks.show(); await settle();
  const load = f.ui.loadMore(); await f.ui.loadMore();
  assert.equal(loads, 2);
  pending.resolve(page([record('first', '重复项'), record('second', '政策工具')])); await load;
  assert.deepEqual(f.ui.items.value.map(item => item.id), ['first', 'second']);
  f.ui.search.value = '政策'; assert.deepEqual(f.ui.filteredItems.value.map(item => item.id), ['second']);
  assert.equal(loads, 2); assert.equal(f.ui.nextOffset.value, null); f.dispose();
});

test('shared scopes and missing provenance fail closed', async () => {
  for (const result of [{ ...page(), scope: 'project' }, { ...page(), scope: 'user' }, page([{ ...record(), source_type: 'verified_fact' }]), page([{ ...record(), question_source: undefined }])]) {
    const f = mount({ memory: async () => result }); f.hooks.show(); await settle();
    assert.deepEqual(f.ui.items.value, []); assert.ok(f.ui.error.value); f.dispose();
  }
});

test('field truncation is explicit and pagination beyond mobile window stops', async () => {
  const f = mount({ memory: async () => ({ ...page([{ ...record(), truncated_fields: ['summary', 'question'] }]), has_more: true, next_offset: 1020 }) });
  f.hooks.show(); await settle();
  assert.deepEqual(f.ui.items.value[0].truncated_fields, ['summary', 'question']);
  assert.equal(f.ui.nextOffset.value, null); assert.equal(f.ui.windowLimited.value, true);
  assert.match(source, /摘要仅展示前 1500 个字符/); assert.match(source, /不能作为完整原话引用/);
  assert.match(source, /不代表全部记忆/); assert.match(source, /只查看，不修改、不删除/);
  f.dispose();
});
