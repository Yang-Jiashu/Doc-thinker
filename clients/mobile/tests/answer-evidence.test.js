import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, watch, reactive, nextTick, effectScope } from 'vue';
import * as Vue from 'vue';
import { MAX_EVIDENCE_ITEMS, MAX_EVIDENCE_CONTENT, MAX_EVIDENCE_TITLE, normalizeAnswerEvidence } from '../src/services/answer-evidence.js';

const filename = new URL('../src/components/AnswerEvidence.vue', import.meta.url);
const source = readFileSync(filename, 'utf8');
const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');
const platformTemplate = platform => preprocess(parse(source).descriptor.template.content,
  { MP_WEIXIN: false, H5: false, APP_PLUS: false, [platform]: true }, { type: 'html' });
const nodes = node => [node, ...(Array.isArray(node.children) ? node.children.flatMap(nodes) : [])];

function mount(sources = [], demo = false) {
  const { descriptor, errors } = parse(source, { filename: String(filename) });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: 'answer-evidence' });
  const template = compileTemplate({ source: platformTemplate('H5'), filename: String(filename), id: 'answer-evidence', compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(template.errors, []);
  const scope = { ref, computed, watch, MAX_EVIDENCE_ITEMS, normalizeAnswerEvidence };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const props = reactive({ sources, demo });
  const effects = effectScope();
  const ui = effects.run(() => component.setup(props, { expose() {} }));
  const executableTemplate = template.code.replace(/import \{([^}]+)\} from "vue"/, (_, names) => `const {${names.replace(/\bas\b/g, ':')}} = Vue;`)
    .replace('export function render', 'return function render');
  const render = new Function('Vue', executableTemplate)(Vue);
  const setup = Vue.proxyRefs(ui), cache = [];
  return { ui, props, render: () => render({}, cache, props, setup), dispose: () => effects.stop() };
}

test('only a nonempty server content string becomes an excerpt; citation metadata cannot create evidence', () => {
  for (const value of [undefined, null, {}, 'text', [{ title: 'Only title', citation: 'Paper A', confidence: 1, verified: true }], [{ content: {} }], [{ content: ' \u0000 ' }]]) {
    assert.deepEqual(normalizeAnswerEvidence(value).items, []);
  }
  assert.deepEqual(normalizeAnswerEvidence([{ content: '  原文待核对  ', confidence: 1, verified: true, citation: '<b>Verified</b>', url: 'javascript:alert(1)' }]).items,
    [{ title: '来源片段 1', content: '原文待核对', truncated: false }]);
});

test('source count, processing window, title and Unicode excerpt length are bounded', () => {
  const malicious = { title: '题'.repeat(20000), content: '🧠'.repeat(20000) };
  const bounded = normalizeAnswerEvidence(Array(10000).fill(malicious));
  assert.equal(bounded.items.length, MAX_EVIDENCE_ITEMS); assert.equal(bounded.limited, true);
  assert.equal(Array.from(bounded.items[0].content).length, MAX_EVIDENCE_CONTENT + 1);
  assert.equal(Array.from(bounded.items[0].title).length, MAX_EVIDENCE_TITLE + 1);
  assert.equal(bounded.items[0].truncated, true); assert.ok(bounded.items[0].content.endsWith('…'));
  const longInvalidPrefix = normalizeAnswerEvidence([...Array(20).fill(null), { content: 'outside bounded scan' }]);
  assert.deepEqual(longInvalidPrefix.items, []); assert.equal(longInvalidPrefix.limited, true);
});

test('S9 and S10 render as labeled excerpts while S11 cannot masquerade as legacy evidence',()=>{
  const sources=Array.from({length:11},(_,index)=>({label:`S${index+1}`,content:`Original ${index+1}`,filename:'Paper.pdf',page:index+1}));
  const result=normalizeAnswerEvidence(sources);
  assert.equal(result.items.length,10);assert.equal(result.limited,true);
  assert.deepEqual(result.items.slice(8).map(item=>item.label),['S9','S10']);
  assert.deepEqual(normalizeAnswerEvidence([sources[10]]).items,[]);
});

test('excerpts remain literal text; display controls are removed and file titles are supported', () => {
  const text = '<img src=x onerror=alert(1)>\n[外链](javascript:alert(1))';
  assert.deepEqual(normalizeAnswerEvidence([{ filename: '资料.pdf', content: '\u202e' + text + '\u0000' }]).items,
    [{ title: '资料.pdf', content: text, truncated: false }]);
  assert.doesNotMatch(source, /v-html|rich-text|web-view|navigator|openURL|request\(|fetch\(|api\./);
  assert.match(source, /尚未逐条核验/); assert.match(source, /演示片段，不是真实文档证据/);
});

test('rendered button expands literal excerpts and changing answer or mode folds the panel', async () => {
  const f = mount([{ content: '回答 A 的来源片段' }]);
  assert.equal(f.ui.expanded.value, false); assert.equal(f.ui.evidence.value.items.length, 1);
  let rendered = nodes(f.render());
  assert.ok(rendered.some(node => node.children === '查看 1 条依据'));
  assert.ok(!rendered.some(node => node.children === '回答 A 的来源片段'));
  rendered.find(node => node.type === 'button').props.onTap();
  rendered = nodes(f.render());
  assert.equal(f.ui.expanded.value, true);
  assert.ok(rendered.some(node => node.type === 'text' && node.children === '回答 A 的来源片段'));
  f.props.sources = [{ content: '回答 B 的来源片段' }]; await nextTick();
  assert.equal(f.ui.expanded.value, false); assert.equal(f.ui.evidence.value.items[0].content, '回答 B 的来源片段');
  f.ui.expanded.value = true; f.props.demo = true; await nextTick();
  assert.equal(f.ui.expanded.value, false);
  rendered = nodes(f.render()); assert.ok(rendered.some(node => node.children === '演示'));
  f.props.sources = []; await nextTick(); assert.deepEqual(f.ui.evidence.value.items, []);
  assert.equal(nodes(f.render()).some(node => node.type === 'button'), false);
  assert.match(source, /v-if="evidence.items.length"/); assert.match(source, /v-if="expanded"/);
  f.dispose();
});

test('H5, WeChat and App templates keep excerpts selectable without rich HTML or external links', () => {
  for (const platform of ['H5', 'MP_WEIXIN', 'APP_PLUS']) {
    const template = platformTemplate(platform);
    const parsed = parse(`<template>${template}</template>`);
    assert.deepEqual(parsed.errors, []);
    const compiled = compileTemplate({ source: template, filename: String(filename), id: platform });
    assert.deepEqual(compiled.errors, []);
    const excerpt = nodes(parsed.descriptor.template.ast).filter(node => node.tag === 'text'
      && node.props.some(prop => prop.type === 6 && prop.name === 'class' && prop.value?.content === 'evidence-excerpt'));
    assert.equal(excerpt.length, 1);
    const expected = platform === 'MP_WEIXIN' ? 'user-select' : 'selectable';
    const forbidden = platform === 'MP_WEIXIN' ? 'selectable' : 'user-select';
    assert.ok(excerpt[0].props.some(prop => prop.type === 6 && prop.name === expected));
    assert.ok(!excerpt[0].props.some(prop => prop.type === 6 && prop.name === forbidden));
  }
});
