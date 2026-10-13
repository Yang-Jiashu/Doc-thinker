import test from 'node:test';
import assert from 'node:assert/strict';
import { sources, modes, getSource, buildShowcase } from '../src/services/research-showcase.js';

test('showcase labels describe supporting materials rather than separate reasoning modes', () => {
  assert.deepEqual(modes.map(mode => mode.label), ['检查条件', '比较案例', '借鉴思路']);
  assert.deepEqual(modes.map(mode => mode.technicalLabel), ['还缺哪一环', '也看看反例', '保留差异与边界']);
  assert.doesNotMatch(modes.map(mode => `${mode.label} ${mode.technicalLabel}`).join(' '), /推理模式|演绎推理|归纳推理|类比推理/);
});

test('the labelled teaching material has unique sources and all step references resolve', () => {
  assert.deepEqual(modes.map(mode => mode.id), ['deduction', 'induction', 'analogy']);
  assert.equal(new Set(sources.map(source => source.id)).size, sources.length);
  for (const source of sources) {
    assert.match(source.title, /本地构造示例/);
    assert.ok(source.kind && source.quote);
    assert.equal(getSource(source.id), source);
  }
  for (const mode of modes) {
    assert.ok(mode.label && mode.technicalLabel && mode.question);
    for (const includeBridge of [true, false]) {
      const result = buildShowcase(mode.id, { includeBridge });
      for (const field of ['heading', 'answer', 'badge', 'limit', 'nextQuestion']) assert.equal(typeof result[field], 'string');
      assert.match(result.badge, /本地构造示例/);
      assert.ok(result.steps.length);
      assert.equal(new Set(result.steps.map(step => step.id)).size, result.steps.length);
      const usedSources = new Set();
      for (const step of result.steps) {
        assert.ok(step.title && step.detail && step.status && step.sourceIds.length);
        for (const id of step.sourceIds) {
          assert.ok(getSource(id), `Unresolved source ${id}`);
          assert.ok(result.sourceIds.includes(id), `Source ${id} is not visible`);
          usedSources.add(id);
        }
      }
      assert.deepEqual(new Set(result.sourceIds), usedSources);
    }
  }
});

test('removing the bridge removes both its quote and its references and stops the conclusion', () => {
  const result = buildShowcase('deduction', { includeBridge: false });
  assert.equal(result.sourceIds.includes('bridge'), false);
  assert.ok(result.steps.every(step => !step.sourceIds.includes('bridge')));
  assert.ok(!JSON.stringify(result).includes(getSource('bridge').quote));
  const visibleQuotes = result.sourceIds.map(id => getSource(id).quote);
  assert.ok(!visibleQuotes.includes(getSource('bridge').quote));
  assert.match(result.answer, /暂不能推出/);
  assert.ok(result.steps.some(step => step.status === 'gap'));
  assert.ok(result.steps.every(step => step.status !== 'conditional'));
  assert.deepEqual(result.sourceIds, ['investment', 'supply']);
});

test('a visible bridge yields only a conditional conclusion and does not claim causal proof', () => {
  const result = buildShowcase('deduction');
  assert.ok(result.sourceIds.includes('bridge'));
  assert.ok(result.steps.some(step => step.detail === getSource('bridge').quote));
  assert.match(result.answer, /如果/);
  assert.match(result.answer, /其他条件不变/);
  assert.match(result.limit, /不能证明现实因果/);
  assert.match(result.limit, /没有证明这些条件已经满足/);
  assert.ok(result.steps.every(step => step.status !== 'gap'));
  assert.equal(buildShowcase('deduction', { includeBridge: 'true' }).steps.at(-1).status, 'gap');
});

test('induction retains two supporting cases and one explicit counterexample without universality', () => {
  const result = buildShowcase('induction');
  assert.equal(result.steps.filter(step => step.status === 'supported').length, 2);
  const counterexample = result.steps.find(step => step.status === 'counterexample');
  assert.ok(counterexample);
  assert.match(counterexample.title, /反例/);
  assert.match(counterexample.detail, /原有进口没有减少/);
  assert.ok(counterexample.sourceIds.includes('case-counterexample'));
  assert.match(result.limit, /全部是虚构案例/);
  assert.match(result.limit, /不能估算概率或推成普遍规律/);
  assert.deepEqual(buildShowcase('induction', { includeBridge: false }), result);
});

test('analogy keeps its mapping a hypothesis and includes a concrete mismatch', () => {
  const result = buildShowcase('analogy');
  assert.match(result.answer, /待验证的假设/);
  assert.match(result.badge, /待验证假设/);
  assert.match(result.limit, /不能证明科研结论/);
  assert.ok(result.steps.some(step => step.status === 'hypothesis'));
  const mismatch = result.steps.find(step => step.status === 'mismatch');
  assert.ok(mismatch);
  assert.match(mismatch.detail, /共享数据与偏差/);
  assert.ok(mismatch.sourceIds.includes('analogy-mismatch'));
  assert.ok(result.steps.every(step => !['supported', 'conditional'].includes(step.status)));
  assert.deepEqual(buildShowcase('analogy', { includeBridge: false }), result);
});

test('unknown inputs fail closed and callers cannot change later material through returned objects', () => {
  for (const id of [undefined, null, '', 'unknown', 'toString']) {
    assert.throws(() => buildShowcase(id), RangeError);
    assert.equal(getSource(id), null);
  }
  const result = buildShowcase('deduction');
  result.sourceIds.length = 0;
  result.steps[0].sourceIds.length = 0;
  result.steps[0].detail = 'changed by caller';
  const next = buildShowcase('deduction');
  assert.ok(next.sourceIds.length);
  assert.deepEqual(next.steps[0].sourceIds, ['investment']);
  assert.notEqual(next.steps[0].detail, 'changed by caller');
  assert.ok(Object.isFrozen(sources) && Object.isFrozen(sources[0]));
  assert.ok(Object.isFrozen(modes) && Object.isFrozen(modes[0]));
});
