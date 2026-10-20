import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { firstValueCase } from '../src/services/first-value-case.js';

test('first-value case has a complete, stable display contract', () => {
  assert.deepEqual(Object.keys(firstValueCase).sort(), [
    'id', 'version', 'title', 'question', 'answer', 'boundary', 'nextQuestion', 'sources', 'steps',
  ].sort());
  assert.equal(firstValueCase.id, 'more-context-is-not-enough');
  assert.equal(firstValueCase.version, 1);
  for (const key of ['title', 'question', 'answer', 'boundary', 'nextQuestion']) {
    assert.equal(typeof firstValueCase[key], 'string');
    assert.ok(firstValueCase[key].trim());
    assert.ok(firstValueCase[key].length <= 200, `${key} stays readable on a phone`);
  }
  assert.equal(firstValueCase.steps.length, 3);
  assert.equal(firstValueCase.sources.length, 2);
});

test('every displayed source is a version-pinned primary paper with an exact locator', () => {
  const expected = new Map([
    ['rag-2020', { path: '/abs/2005.11401v4', year: 2020 }],
    ['lost-middle-2023', { path: '/abs/2307.03172v3', year: 2023 }],
  ]);
  for (const source of firstValueCase.sources) {
    assert.deepEqual(Object.keys(source).sort(), ['id', 'title', 'authors', 'year', 'url', 'locator', 'quote', 'explanation'].sort());
    const url = new URL(source.url);
    assert.equal(url.protocol, 'https:');
    assert.equal(url.hostname, 'arxiv.org');
    assert.equal(url.pathname, expected.get(source.id).path);
    assert.equal(url.search, '');
    assert.equal(url.username, '');
    assert.equal(url.password, '');
    assert.equal(source.year, expected.get(source.id).year);
    assert.match(source.locator, /摘要（Abstract）/);
    for (const key of ['title', 'authors', 'quote', 'explanation']) assert.ok(source[key].trim());
  }
});

test('short original-language quotations remain separate from Chinese interpretation', () => {
  // Exact excerpts were checked against the pinned abstracts, not generated paraphrases.
  const excerpts = {
    'rag-2020': 'set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures.',
    'lost-middle-2023': 'performance can degrade significantly when changing the position of relevant information',
  };
  for (const source of firstValueCase.sources) {
    assert.equal(source.quote, excerpts[source.id]);
    assert.ok(source.quote.split(/\s+/).length <= 25);
    assert.doesNotMatch(source.quote, /[\u3400-\u9fff]/);
    assert.match(source.explanation, /[\u3400-\u9fff]/);
    assert.notEqual(source.quote, source.explanation);
  }
});

test('steps refer only to existing sources and distinguish observations from synthesis', () => {
  const sourceIds = new Set(firstValueCase.sources.map(source => source.id));
  const stepIds = new Set(firstValueCase.steps.map(step => step.id));
  assert.equal(sourceIds.size, firstValueCase.sources.length);
  assert.equal(stepIds.size, firstValueCase.steps.length);
  for (const step of firstValueCase.steps) {
    assert.deepEqual(Object.keys(step).sort(), ['id', 'title', 'detail', 'status', 'sourceIds'].sort());
    assert.ok(step.title && step.detail);
    assert.ok(step.sourceIds.length > 0);
    assert.equal(new Set(step.sourceIds).size, step.sourceIds.length);
    step.sourceIds.forEach(id => assert.ok(sourceIds.has(id), `unknown source ${id}`));
  }
  assert.deepEqual(firstValueCase.steps.map(step => step.status), ['supported', 'supported', 'synthesis']);
  assert.deepEqual(firstValueCase.steps[2].sourceIds, [...sourceIds]);
  assert.match(firstValueCase.steps[2].detail, /整理者的综合判断/);
});

test('the sample does not misrepresent live generation, product superiority or unlimited memory', () => {
  assert.match(firstValueCase.boundary, /人工整理的真实论文样例/);
  assert.match(firstValueCase.boundary, /非实时模型生成/);
  assert.match(firstValueCase.boundary, /不是本产品效果评测/);
  assert.match(firstValueCase.boundary, /模型、任务和年代不同/);
  assert.match(firstValueCase.boundary, /不能据此断言所有模型都一样/);
  assert.match(firstValueCase.boundary, /没有验证本产品的长期记忆能力/);
  assert.match(firstValueCase.answer, /^不一定/);
  assert.match(firstValueCase.steps[2].detail, /没有证明减少 token 就一定更准确/);
});

test('case data is deeply immutable and its module performs no model, storage or network access', () => {
  const assertFrozen = value => {
    if (!value || typeof value !== 'object') return;
    assert.ok(Object.isFrozen(value));
    Object.values(value).forEach(assertFrozen);
  };
  assertFrozen(firstValueCase);
  assert.throws(() => { firstValueCase.steps[0].sourceIds.push('fake-source'); }, TypeError);
  assert.throws(() => { firstValueCase.sources[0].quote = 'invented'; }, TypeError);
  const moduleSource = readFileSync(new URL('../src/services/first-value-case.js', import.meta.url), 'utf8');
  assert.doesNotMatch(moduleSource, /\bimport\b|\bfetch\s*\(|\bapi\.|\buni\.|localStorage|setStorage|XMLHttpRequest/);
});
