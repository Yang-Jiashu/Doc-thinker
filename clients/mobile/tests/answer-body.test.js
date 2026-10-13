import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, watch, reactive, nextTick, effectScope } from 'vue';
import * as Vue from 'vue';
import {
  parseAnswerBody, MAX_ANSWER_FORMAT_CHARS, MAX_ANSWER_BLOCKS,
  MAX_ANSWER_INLINE_TOKENS, MAX_ANSWER_TABLE_ROWS, MAX_ANSWER_TABLE_COLUMNS,
} from '../src/services/answer-body.js';

const filename = new URL('../src/components/AnswerBody.vue', import.meta.url);
const source = readFileSync(filename, 'utf8');
const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');
const { compile: compileMiniProgram } = require('@dcloudio/uni-mp-compiler');
const platformTemplate = platform => preprocess(parse(source).descriptor.template.content,
  { MP_WEIXIN: false, H5: false, APP_PLUS: false, [platform]: true }, { type: 'html' });
const nodes = node => [node, ...(Array.isArray(node.children) ? node.children.flatMap(nodes) : [])];
const textOf = tokens => tokens.map(token => token.text).join('');

function mount(text = '') {
  const { descriptor, errors } = parse(source, { filename: String(filename) });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: 'answer-body' });
  const template = compileTemplate({ source: platformTemplate('H5'), filename: String(filename), id: 'answer-body', compilerOptions: { bindingMetadata: script.bindings, isCustomElement: () => true } });
  assert.deepEqual(template.errors, []);
  const scope = { ref, computed, watch, parseAnswerBody };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const props = reactive({ text });
  const effects = effectScope();
  const ui = effects.run(() => component.setup(props, { expose() {} }));
  const executableTemplate = template.code.replace(/import \{([^}]+)\} from "vue"/, (_, names) => `const {${names.replace(/\bas\b/g, ':')}} = Vue;`)
    .replace('export function render', 'return function render');
  const render = new Function('Vue', executableTemplate)(Vue);
  const setup = Vue.proxyRefs(ui), cache = [];
  return { ui, props, render: () => render({}, cache, props, setup), dispose: () => effects.stop() };
}

test('answer body formats the bounded Markdown subset without altering the original answer', () => {
  const raw = '# 结论\n\n**重要** 与 *补充*，`foo_bar` [S1]。\n保留换行\n\n- 一\n- 二\n\n3. 第三项\n5) 第五项\n\n> 原文\n> 下一行';
  const result = parseAnswerBody(raw);
  assert.equal(result.raw, raw); assert.equal(result.plain, false);
  assert.deepEqual(result.blocks.map(block => block.type), ['heading', 'paragraph', 'list', 'list', 'quote']);
  assert.equal(result.blocks[0].level, 1);
  assert.deepEqual(result.blocks[1].tokens.filter(token => token.kind !== 'text'), [
    { kind: 'strong', text: '重要' }, { kind: 'em', text: '补充' },
    { kind: 'code', text: 'foo_bar' }, { kind: 'citation', text: '[S1]' },
  ]);
  assert.ok(textOf(result.blocks[1].tokens).endsWith('。\n保留换行'));
  assert.deepEqual(result.blocks[3].items.map(item => item.marker), ['3.', '5)']);
  assert.equal(textOf(result.blocks[4].tokens), '原文\n下一行');
});

test('headings, nested list indentation and non-Markdown identifiers remain readable', () => {
  const result = parseAnswerBody('### 小标题\n- 大项\n  - 小项\n\nfoo_bar_baz 与 foo__bar__baz，_强调_');
  assert.equal(result.blocks[0].level, 3);
  assert.deepEqual(result.blocks[1].items.map(item => item.indent), [0, 2]);
  assert.equal(textOf(result.blocks[2].tokens), 'foo_bar_baz 与 foo__bar__baz，强调');
  assert.equal(result.blocks[2].tokens.filter(token => token.kind === 'em').length, 1);
});

test('literal HTML, Markdown images and external links never create executable/remote nodes', () => {
  const raw = '<script>alert(1)</script>\n<img src="https://tracker.example/a" onerror="x()">\n![cover](https://tracker.example/a.png)\n[open](javascript:alert(1))\nhttps://example.test/foo_bar_baz\n[S2](https://example.test/source)';
  const result = parseAnswerBody(raw);
  assert.equal(result.raw, raw);
  assert.equal(textOf(result.blocks[0].tokens), raw);
  assert.equal(result.blocks[0].tokens.some(token => token.kind === 'citation'), false);
  assert.doesNotMatch(source, /v-html|rich-text|web-view|<image|<img|<a[\s>]|navigator|openURL|fetch\(|request\(/);
  const f = mount(raw);
  const rendered = nodes(f.render());
  assert.ok(rendered.some(node => node.type === 'text' && node.children === raw));
  assert.ok(!rendered.some(node => ['script', 'img', 'a', 'image', 'navigator'].includes(node.type)));
  f.dispose();
});

test('source labels are preserved as labels, never promoted to verified evidence or links', () => {
  const result = parseAnswerBody('结论 [S1] [S3, S4] [S9，S10] [1] [S0001]');
  assert.deepEqual(result.blocks[0].tokens.filter(token => token.kind === 'citation').map(token => token.text), ['[S1]', '[S3, S4]', '[S9，S10]']);
  assert.equal(textOf(result.blocks[0].tokens), result.raw);
  assert.doesNotMatch(source, /verified|已核验|已验证|引用正确/);
});

test('fenced code preserves exact CRLF, spaces, trailing newline and inert code content', () => {
  const code = '  <script>alert(1)</script>\r\n\tprint("**literal**")\r\n';
  const result = parseAnswerBody('````python\r\n' + code + '```\r\n````\r\n\n结束');
  assert.equal(result.blocks[0].type, 'code'); assert.equal(result.blocks[0].language, 'python');
  assert.equal(result.blocks[0].text, code + '```\r\n');
  assert.equal(textOf(result.blocks[1].tokens), '结束');
  assert.equal(parseAnswerBody('~~~js\nx()').blocks[0].text, 'x()');
  assert.equal(parseAnswerBody('```\n~~~\n```').blocks[0].text, '~~~\n');
});

test('small tables preserve every cell, inline code pipes, escaped pipes and citation labels', () => {
  const raw = '| 指标 | 内容 |\n| :--- | ---: |\n| **质量** | `a|b` [S2] |\n| token | A\\|B |';
  const table = parseAnswerBody(raw).blocks[0];
  assert.equal(table.type, 'table');
  assert.deepEqual(table.header.map(textOf), ['指标', '内容']);
  assert.deepEqual(table.rows.map(row => row.map(textOf)), [['质量', 'a|b [S2]'], ['token', 'A|B']]);
});

test('ragged, over-wide and over-tall tables remain complete literal text', () => {
  const wide = Array(MAX_ANSWER_TABLE_COLUMNS + 1).fill('cell').join('|');
  const wideSep = Array(MAX_ANSWER_TABLE_COLUMNS + 1).fill('---').join('|');
  for (const raw of ['a|b\n---|---\none|two|three', wide + '\n' + wideSep,
    'a|b\n---|---\n' + Array(MAX_ANSWER_TABLE_ROWS + 1).fill('one|two').join('\n')]) {
    const result = parseAnswerBody(raw);
    assert.equal(result.raw, raw);
    assert.ok(result.blocks.every(block => block.type !== 'table'));
    assert.equal(result.blocks.map(block => textOf(block.tokens)).join('\n'), raw);
  }
});

test('formatting budgets bound nodes without truncating long text, Unicode or late evidence', () => {
  const inputs = [
    '🧠'.repeat(MAX_ANSWER_FORMAT_CHARS) + '\n关键来源末尾 [S99]',
    Array(MAX_ANSWER_BLOCKS + 1).fill('# heading').join('\n'),
    '**point** '.repeat(MAX_ANSWER_INLINE_TOKENS + 1) + '末尾证据',
  ];
  for (const raw of inputs) {
    const result = parseAnswerBody(raw);
    assert.equal(result.raw, raw); assert.equal(result.plain, true);
    assert.equal(result.blocks.length, 1); assert.equal(result.blocks[0].text, raw);
    assert.ok(['length', 'complexity'].includes(result.reason));
  }
});

test('malformed Markdown and incomplete links do not cause unbounded look-ahead or lose content', () => {
  const raw = '['.repeat(MAX_ANSWER_FORMAT_CHARS - 20) + '\n尾部';
  const started = performance.now();
  const result = parseAnswerBody(raw);
  assert.equal(textOf(result.blocks[0].tokens), raw);
  assert.ok(performance.now() - started < 2000, 'bounded input should not cause quadratic incomplete-link scans');
  for (const value of [undefined, null, {}, 17]) assert.deepEqual(parseAnswerBody(value).blocks, []);
  assert.equal(parseAnswerBody(' \n\t ').blocks[0].text, ' \n\t ');
});

test('rendered code copy copies exact code, handles failures and clears state for another answer', async () => {
  const previous = globalThis.uni;
  let request;
  globalThis.uni = { setClipboardData(value) { request = value; } };
  const f = mount('```js\n  example()\n```');
  try {
    nodes(f.render()).find(node => node.type === 'button').props.onTap();
    assert.equal(request.data, '  example()\n'); assert.equal(request.showToast, false);
    request.success(); assert.equal(f.ui.copiedIndex.value, 0);
    assert.ok(nodes(f.render()).some(node => node.type === 'button' && node.children === '已复制'));
    nodes(f.render()).find(node => node.type === 'button').props.onTap(); request.fail();
    assert.equal(f.ui.copyErrorIndex.value, 0);
    assert.ok(nodes(f.render()).some(node => node.children === '复制失败，可以长按选择代码'));
    const oldRequest = request;
    f.props.text = '另一条回答'; await nextTick();
    oldRequest.success(); oldRequest.fail();
    assert.equal(f.ui.copiedIndex.value, -1); assert.equal(f.ui.copyErrorIndex.value, -1);
    assert.ok(nodes(f.render()).some(node => node.children === '另一条回答'));
  } finally { f.dispose(); globalThis.uni = previous; }
});

test('H5, WeChat and App templates keep body text selectable and code/table horizontally scrollable', () => {
  for (const platform of ['H5', 'MP_WEIXIN', 'APP_PLUS']) {
    const template = platformTemplate(platform);
    const parsed = parse(`<template>${template}</template>`);
    assert.deepEqual(parsed.errors, []);
    assert.deepEqual(compileTemplate({ source: template, filename: String(filename), id: platform }).errors, []);
    const all = nodes(parsed.descriptor.template.ast);
    const bodyText = all.filter(node => node.tag === 'text' && node.props.some(prop => prop.type === 6 && prop.name === 'class'
      && /answer-text|answer-code-text/.test(prop.value?.content || '')));
    assert.equal(bodyText.length, 5);
    for (const node of bodyText) {
      assert.ok(node.props.some(prop => prop.type === 6 && prop.name === (platform === 'MP_WEIXIN' ? 'user-select' : 'selectable')));
      assert.ok(!node.props.some(prop => prop.type === 6 && prop.name === (platform === 'MP_WEIXIN' ? 'selectable' : 'user-select')));
    }
    const scrolls = all.filter(node => node.tag === 'scroll-view');
    assert.equal(scrolls.length, 2);
    assert.ok(scrolls.every(node => node.props.some(prop => prop.type === 6 && prop.name === 'scroll-x')));
  }
});

test('late clipboard callbacks cannot overwrite a newer copy or another answer before the next render', async () => {
  const previous = globalThis.uni, requests = [];
  globalThis.uni = { setClipboardData(value) { requests.push(value); } };
  const f = mount('```\nfirst\n```\n```\nsecond\n```');
  try {
    f.ui.copyCode('first\n', 0); f.ui.copyCode('second\n', 1);
    requests[1].success(); requests[0].success(); requests[0].fail();
    assert.equal(f.ui.copiedIndex.value, 1); assert.equal(f.ui.copyErrorIndex.value, -1);
    f.ui.copyCode('second\n', 1); f.props.text = 'new answer';
    requests[2].success(); requests[2].fail();
    assert.equal(f.ui.copiedIndex.value, -1); assert.equal(f.ui.copyErrorIndex.value, -1);
    await nextTick();
    globalThis.uni = { setClipboardData() { throw new Error('clipboard unavailable'); } };
    f.ui.copyCode('new answer', 0); assert.equal(f.ui.copyErrorIndex.value, 0);
  } finally { f.dispose(); globalThis.uni = previous; }
});

test('the actual uni-app mini-program compiler accepts nested table/list and selectable text rendering', () => {
  const { descriptor } = parse(source);
  const script = compileScript(descriptor, { id: 'answer-body-mini-program' });
  const errors = [];
  const result = compileMiniProgram(platformTemplate('MP_WEIXIN'), {
    filename: 'src/components/AnswerBody.vue', bindingMetadata: script.bindings,
    onError: error => errors.push(error),
  });
  assert.deepEqual(errors, []);
  assert.ok(result.code.length > 0);
  assert.match(result.code, /copyCode/);
  assert.doesNotMatch(result.code, /innerHTML|openURL/);
});
