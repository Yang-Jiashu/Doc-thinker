import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import * as Vue from 'vue';

const filename = 'pages/documents/index.vue';
const source = readFileSync(new URL('../src/' + filename, import.meta.url), 'utf8');
const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');
const { descriptor } = parse(source, { filename });
const script = compileScript(descriptor, { id: 'adaptive-reader' });
const template = compileTemplate({
  source: preprocess(descriptor.template.content, { H5: true, MP_WEIXIN: false, APP_PLUS: false }, { type: 'html' }),
  filename, id: 'adaptive-reader', compilerOptions: { bindingMetadata: script.bindings, isCustomElement: () => true },
});
assert.deepEqual(template.errors, []);
const executableTemplate = template.code.replace(/import \{([^}]+)\} from "vue"/, (_, names) => `const {${names.replace(/\bas\b/g, ':')}} = Vue;`)
  .replace('export function render', 'return function render');
const renderTemplate = new Function('Vue', executableTemplate)(Vue);
const flatten = node => [node, ...(Array.isArray(node?.children) ? node.children.flatMap(flatten) : [])];
const hasClass = (node, cls) => String(node.props?.class || '').split(' ').includes(cls);
const deferred = () => { let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve }; };
const settle = () => new Promise(resolve => setImmediate(resolve));
const paper = (id, name = `${id}.pdf`) => ({ id, filename: name, status: 'ready', chunk_count: 23, source_ids: [`${id}-source`], chunks: [] });

function mount({ desktop = false, demo = false, research: overrides = {}, files = [paper('first'), paper('second')] } = {}) {
  let active = '#00001', identity = 'one', timer = 0;
  const hooks = {}, timers = new Map(), scope = Vue.effectScope();
  const calls = { document: [], source: [], scroll: [], focus: [], nearest: [], model: 0 };
  const panel = { scrollTop: 99 };
  const research = {
    capabilities: async () => ({ enabled: !demo, uploads_allowed: !demo }),
    documents: async () => files,
    document: async (sid, id) => {
      calls.document.push([sid, id]);
      return { ...files.find(file => file.id === id), preview_limited: true, chunks: [{ id: `${id}-source`, page: 4, content: `${id} exact preview`, excerpt_limited: true }] };
    },
    source: async (sid, id) => { calls.source.push([sid, id]); return { id, filename: 'Actual paper.pdf', page: 4, content: 'Exact complete source, not a generated summary.' }; },
    ...overrides,
  };
  const context = {
    ref: Vue.ref, computed: Vue.computed, research,
    api: { sessions: async () => [{ id: '#00001' }, { id: '#00002' }], files: async () => files, ask() { calls.model++; throw new Error('No model during reading'); } },
    researchConnectionKey: () => identity, getPreferences: () => ({ mode: demo ? 'demo' : 'connected', connectionKind: demo ? 'private' : 'invite' }),
    getActiveSession: () => active, setActiveSession: value => { active = value; },
    onShow: fn => { hooks.show = fn; }, onHide: fn => { hooks.hide = fn; }, onUnload: fn => { hooks.unload = fn; },
    setTimeout: (fn, delay) => { const id = ++timer; timers.set(id, { fn, delay }); return id; }, clearTimeout: id => timers.delete(id),
    uni: { pageScrollTo: value => calls.scroll.push(value), switchTab() {} },
    window: { matchMedia: query => ({ matches: desktop && query === '(min-width: 1100px)' }) },
    document: { querySelector: selector => selector === '#library-reader' ? panel : { focus: options => calls.focus.push([selector, options]), scrollIntoView: options => calls.nearest.push([selector, options]) } },
    BrandHeader: {}, WorkspaceBar: {}, ModeBanner: {}, CapabilityBoundary: {},
  };
  const code = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(context), code)(...Object.values(context));
  const ui = scope.run(() => component.setup({}, { expose() {} }));
  // Run the compiled template inside a real render context, so input directives
  // and event bindings are retained rather than reducing this to source matching.
  const renderer = Vue.createRenderer({
    createElement: tag => ({ tagName: tag.toUpperCase(), addEventListener() {}, removeEventListener() {} }),
    createText: text => ({ text }), createComment: text => ({ text }),
    insert() {}, remove() {}, setText() {}, setElementText() {}, patchProp() {}, parentNode: () => null, nextSibling: () => null,
  });
  const render = () => {
    const host = {}, root = { render: () => renderTemplate({}, [], {}, Vue.proxyRefs(ui)) };
    renderer.render(Vue.h(root), host);
    const tree = host._vnode.component.subTree;
    renderer.render(null, host);
    return tree;
  };
  return {
    ui, calls, hooks, panel, research, render, nodes: () => flatten(render()),
    changeIdentity: () => { identity = 'another-user'; },
    async show() { hooks.show(); await settle(); },
    async tick() { const entry = [...timers][0]; if (!entry) return; timers.delete(entry[0]); await entry[1].fn(); },
    dispose() { hooks.hide(); scope.stop(); },
  };
}

const button = (f, predicate) => f.nodes().find(node => node.type === 'button' && predicate(node));
const byLabel = (f, label) => button(f, node => node.props?.['aria-label'] === label);
const keyEvent = (key = 'Enter', values = {}) => ({ key, preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...values });
const assertKeyboardButton = node => { assert.equal(node.props.role, 'button'); assert.equal(node.props.tabindex, 0); assert.equal(typeof node.props.onKeydown, 'function'); };

test('the two-pane library starts with an honest empty reader and never prefetches document bodies', async () => {
  const f = mount({ desktop: true }); await f.show();
  assert.ok(f.nodes().some(node => hasClass(node, 'library-catalog')));
  assert.ok(f.nodes().some(node => hasClass(node, 'library-reader-column') && node.props.role === 'region'));
  assert.ok(f.nodes().some(node => hasClass(node, 'reader-empty')));
  assert.equal(f.nodes().filter(node => hasClass(node, 'document-card')).length, 2);
  assert.equal(f.calls.document.length, 0); assert.equal(f.calls.source.length, 0); assert.equal(f.calls.model, 0);
  assert.match(source, /@media \(min-width:1100px\)[\s\S]*grid-template-columns:minmax\(290px, \.9fr\) minmax\(0, 1\.2fr\)/);
  assert.match(source, /\.reader-empty \{ display:none; \}/);
  f.dispose();
});

test('demo cards without research IDs never look selected or offer a nonexistent reader', async () => {
  const f = mount({ demo: true, files: [{ file_name: '示例资料.pdf', processing_status: 'completed' }, { filename: 'Notes.md', status: 'ready' }] });
  await f.show();
  assert.equal(f.nodes().filter(node => hasClass(node, 'document-card')).length, 2);
  assert.equal(f.nodes().filter(node => hasClass(node, 'is-reading')).length, 0);
  assert.equal(f.ui.isReading({}), false);
  assert.ok(!f.nodes().some(node => hasClass(node, 'library-reader-column')));
  assert.equal(f.calls.document.length, 0); assert.equal(f.calls.source.length, 0); assert.equal(f.calls.model, 0);
  f.dispose();
});

test('clicking a document reads only that document and marks its visible card as selected', async () => {
  const f = mount(); await f.show();
  await byLabel(f, '查看 second.pdf').props.onTap();
  assert.deepEqual(f.calls.document, [['#00001', 'second']]); assert.equal(f.calls.source.length, 0);
  assert.equal(byLabel(f, '查看 second.pdf').props['aria-expanded'], true);
  assert.equal(byLabel(f, '查看 first.pdf').props['aria-expanded'], false);
  const selected = f.nodes().filter(node => hasClass(node, 'is-reading'));
  assert.equal(selected.length, 1); assert.equal(selected[0].props.id, 'library-document-1');
  assert.ok(f.nodes().some(node => node.children === '此页只预览前 20 个片段，不是全文导出。'));
  assert.ok(!f.nodes().some(node => hasClass(node, 'reader-empty')));
  await f.tick(); assert.deepEqual(f.calls.scroll, [{ selector: '#document-reader', duration: 0 }]);
  f.dispose();
});

test('a complete source replaces the preview, and back restores it without fetching again', async () => {
  const f = mount(); await f.show(); await byLabel(f, '查看 first.pdf').props.onTap(); await f.tick();
  await button(f, node => node.children === '打开原文片段').props.onTap(); await f.tick();
  assert.deepEqual(f.calls.source, [['#00001', 'first-source']]);
  assert.ok(f.nodes().some(node => node.props?.id === 'source-reader'));
  assert.ok(!f.nodes().some(node => node.props?.id === 'document-reader'));
  assert.ok(f.nodes().some(node => node.children === 'Exact complete source, not a generated summary.'));
  button(f, node => node.children === '← 返回资料预览').props.onTap(); await f.tick();
  assert.ok(f.nodes().some(node => node.props?.id === 'document-reader'));
  assert.equal(f.ui.sourceDetail.value, null); assert.equal(f.calls.document.length, 1); assert.equal(f.calls.source.length, 1);
  assert.equal(f.calls.scroll.at(-1).selector, '#document-reader'); assert.equal(f.calls.model, 0);
  f.dispose();
});

test('reselecting an already open document returns to its preview without duplicate API calls', async () => {
  const f = mount(); await f.show(); await f.ui.inspectDocument(paper('second')); await f.tick();
  await f.ui.readSource('second-source'); await f.tick();
  await byLabel(f, '查看 second.pdf').props.onTap(); await f.tick();
  assert.equal(f.ui.sourceDetail.value, null); assert.equal(f.ui.documentDetail.value.id, 'second');
  assert.equal(f.calls.document.length, 1); assert.equal(f.calls.source.length, 1);
  f.dispose();
});

test('closing the reader returns a phone to the selected card even after filtering', async () => {
  const f = mount(); await f.show(); f.ui.fileQuery.value = 'second';
  await byLabel(f, '查看 second.pdf').props.onTap(); await f.tick();
  byLabel(f, '关闭原文并返回选中的资料').props.onTap(); await f.tick();
  assert.equal(f.ui.documentDetail.value, null); assert.equal(f.ui.sourceDetail.value, null);
  assert.equal(f.calls.scroll.at(-1).selector, '#library-document-0'); assert.equal(f.ui.fileQuery.value, 'second');
  assert.equal(f.calls.document.length, 1); f.dispose();
});

test('desktop reading scrolls the sticky pane, not the page, and closing focuses the selected card', async () => {
  const f = mount({ desktop: true }); await f.show(); await f.ui.inspectDocument(paper('second')); await f.tick();
  assert.equal(f.panel.scrollTop, 0); assert.equal(f.calls.scroll.length, 0);
  assert.deepEqual(f.calls.focus.at(-1), ['#document-reader', { preventScroll: true }]);
  f.panel.scrollTop = 1000; await f.ui.readSource('second-source'); await f.tick();
  assert.equal(f.panel.scrollTop, 0); assert.equal(f.calls.scroll.length, 0);
  f.ui.dismissReader(); await f.tick();
  assert.deepEqual(f.calls.nearest.at(-1), ['#library-document-1', { block: 'nearest', behavior: 'auto' }]);
  assert.deepEqual(f.calls.focus.at(-1), ['#library-document-1', { preventScroll: true }]); f.dispose();
});

test('the entire desktop reading path has focusable buttons activated by Enter and Space', async () => {
  const f = mount({ desktop: true }); await f.show();
  const inspect = byLabel(f, '查看 second.pdf'); assertKeyboardButton(inspect);
  const enter = keyEvent('Enter'); await inspect.props.onKeydown(enter); await f.tick();
  assert.equal(enter.prevented, true); assert.equal(enter.stopped, true);
  assert.deepEqual(f.calls.document, [['#00001', 'second']]);
  const sourceButton = button(f, node => node.children === '打开原文片段'); assertKeyboardButton(sourceButton);
  const space = keyEvent(' '); await sourceButton.props.onKeydown(space); await f.tick();
  assert.equal(space.prevented, true); assert.deepEqual(f.calls.source, [['#00001', 'second-source']]);
  assertKeyboardButton(byLabel(f, '关闭原文并返回选中的资料'));
  const back = button(f, node => node.children === '← 返回资料预览'); assertKeyboardButton(back);
  await back.props.onKeydown(keyEvent('Enter')); await f.tick();
  assert.equal(f.ui.sourceDetail.value, null); assert.equal(f.ui.documentDetail.value.id, 'second');
  const close = byLabel(f, '关闭原文并返回选中的资料'); assertKeyboardButton(close);
  await close.props.onKeydown(keyEvent(' ')); await f.tick();
  assert.equal(f.ui.documentDetail.value, null); assert.equal(f.calls.document.length, 1); assert.equal(f.calls.source.length, 1);
  assert.deepEqual(f.calls.focus.at(-1), ['#library-document-1', { preventScroll: true }]); assert.equal(f.calls.model, 0); f.dispose();
});

test('unpreviewed source buttons also expose keyboard semantics and read only their explicit ID', async () => {
  const f = mount(); await f.show(); await byLabel(f, '查看 first.pdf').props.onKeydown(keyEvent()); await f.tick();
  f.ui.documentDetail.value.source_ids.push('not-previewed-source');
  const sourceButton = button(f, node => node.children === '打开来源 not-previewed-source'); assertKeyboardButton(sourceButton);
  await sourceButton.props.onKeydown(keyEvent(' ')); await f.tick();
  assert.deepEqual(f.calls.source, [['#00001', 'not-previewed-source']]); f.dispose();
});

test('keyboard repeat, shortcuts, IME and events already handled never read a document', async () => {
  const f = mount(); await f.show(); const inspect = byLabel(f, '查看 first.pdf');
  for (const event of [undefined, {}, { key: 'Enter' }, keyEvent('Tab'), keyEvent('a'), keyEvent('Enter', { repeat: true }),
    keyEvent(' ', { repeat: true }), keyEvent('Enter', { ctrlKey: true }), keyEvent('Enter', { metaKey: true }),
    keyEvent('Enter', { altKey: true }), keyEvent('Enter', { shiftKey: true }), keyEvent('Enter', { isComposing: true }),
    keyEvent('Enter', { keyCode: 229 }), keyEvent('Enter', { defaultPrevented: true })]) {
    await inspect.props.onKeydown(event);
    assert.notEqual(event?.prevented, true);
  }
  f.ui.readerKeydown(keyEvent(), 'remove', paper('first'));
  assert.equal(f.calls.document.length, 0); assert.equal(f.calls.source.length, 0); assert.equal(f.ui.documentDetail.value, null);
  await inspect.props.onKeydown(keyEvent()); await f.tick();
  await inspect.props.onKeydown(keyEvent('Enter', { repeat: true }));
  assert.equal(f.calls.document.length, 1); f.dispose();
});

test('pending and hidden reader buttons cannot duplicate requests or close the active result', async () => {
  const wait = deferred(); let reads = 0;
  const f = mount({ research: { source: () => { reads++; return wait.promise; } } }); await f.show();
  await byLabel(f, '查看 first.pdf').props.onKeydown(keyEvent()); await f.tick();
  const sourceButton = button(f, node => node.children === '打开原文片段');
  const reading = sourceButton.props.onKeydown(keyEvent());
  assert.equal(f.ui.busyDocument.value, true);
  const disabled = byLabel(f, '关闭原文并返回选中的资料');
  assert.equal(disabled.props.disabled, true); assert.equal(disabled.props.tabindex, -1);
  await sourceButton.props.onKeydown(keyEvent()); await sourceButton.props.onKeydown(keyEvent('Enter', { repeat: true }));
  await disabled.props.onKeydown(keyEvent()); f.ui.closeSource(); f.ui.dismissReader();
  assert.equal(reads, 1); assert.equal(f.ui.documentDetail.value.id, 'first');
  wait.resolve({ id: 'first-source', content: 'One original source' }); await reading; await f.tick();
  const hiddenClose = byLabel(f, '关闭原文并返回选中的资料'), hiddenBack = button(f, node => node.children === '← 返回资料预览');
  f.hooks.hide(); await hiddenClose.props.onKeydown(keyEvent()); await hiddenBack.props.onKeydown(keyEvent(' '));
  await sourceButton.props.onKeydown(keyEvent()); f.ui.closeSource(); f.ui.dismissReader();
  assert.equal(reads, 1); assert.equal(f.ui.documentDetail.value, null); assert.equal(f.ui.sourceDetail.value, null); f.dispose();
});

test('a changed identity clears the reader before a delayed navigation can reveal private text', async () => {
  const f = mount({ desktop: true }); await f.show(); await f.ui.inspectDocument(paper('first'));
  f.changeIdentity(); await f.tick();
  assert.equal(f.ui.documentDetail.value, null); assert.equal(f.ui.sourceDetail.value, null); assert.equal(f.ui.files.value.length, 0);
  assert.equal(f.calls.focus.length, 0); assert.equal(f.calls.scroll.length, 0); f.dispose();
});

test('a late source response after leaving never restores a reader or changes focus', async () => {
  const wait = deferred(), f = mount({ desktop: true, research: { source: () => wait.promise } });
  await f.show(); await f.ui.inspectDocument(paper('first')); await f.tick(); const before = f.calls.focus.length;
  const reading = f.ui.readSource('first-source');
  assert.equal(byLabel(f, '关闭原文并返回选中的资料').props.disabled, true);
  f.ui.dismissReader(); assert.equal(f.ui.documentDetail.value.id, 'first');
  f.hooks.hide(); wait.resolve({ id: 'first-source', content: 'Private late source' }); await reading; await f.tick();
  assert.equal(f.ui.documentDetail.value, null); assert.equal(f.ui.sourceDetail.value, null); assert.equal(f.calls.focus.length, before);
  f.dispose();
});

test('filenames and wide source text remain literal and the reading columns can shrink', async () => {
  const name = '<script>not a script</script>' + 'extremely-long-paper-title'.repeat(16) + '.pdf';
  const f = mount({ files: [paper('first', name)] }); await f.show(); await byLabel(f, `查看 ${name}`).props.onTap();
  assert.ok(f.nodes().some(node => node.type === 'text' && node.children === name));
  assert.ok(!f.nodes().some(node => ['script', 'img', 'iframe'].includes(node.type)));
  assert.match(source, /\.library-workspace,\.library-catalog,\.library-reader-column \{ min-width:0; \}/);
  assert.match(source, /\.reader-heading \.section-title \{ display:block; overflow-wrap:anywhere; \}/);
  assert.match(source, /\.source-excerpt \{[^}]*white-space:pre-wrap;[^}]*overflow-wrap:anywhere;/);
  f.dispose();
});
