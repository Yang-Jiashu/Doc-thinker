import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript } from '@vue/compiler-sfc';
import { defineComponent, effectScope, h, nextTick, ref, watch } from 'vue';

const source = readFileSync(new URL('../src/components/ComposerJump.vue', import.meta.url), 'utf8');
const { descriptor } = parse(source);
const script = compileScript(descriptor, { id: 'composer-jump' }).content
  .replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');

function mount({ active = false, disabled = false, available = true, target = {}, broken = false } = {}) {
  const observers = [], events = [], hooks = {}, props = ref({ active, disabled }).value;
  let currentTarget = target;
  const scope = { defineComponent, h, nextTick, ref, watch,
    onMounted: fn => { hooks.mounted = fn; }, onBeforeUnmount: fn => { hooks.unmount = fn; },
    document: { getElementById: id => { assert.equal(id, 'chat-composer'); return currentTarget; } },
    window: { IntersectionObserver: available ? class {
      constructor(callback, options) {
        if (broken) throw new Error('Unavailable in this environment');
        this.callback = callback; this.options = options; this.disconnected = false;
        observers.push(this);
      }
      observe(value) { this.target = value; }
      disconnect() { this.disconnected = true; }
      report(isIntersecting, value = this.target) { this.callback([{ target: value, isIntersecting }]); }
    } : undefined },
  };
  const component = new Function(...Object.keys(scope), script)(...Object.values(scope));
  const effects = effectScope();
  const render = effects.run(() => component.setup(props, { emit: (...args) => events.push(args) }));
  hooks.mounted();
  return { props, observers, events, render, target: value => { currentTarget = value; },
    dispose: () => { hooks.unmount(); effects.stop(); } };
}

test('inactive or empty chat has no observer or floating action', async () => {
  const f = mount(); await nextTick();
  assert.equal(f.observers.length, 0); assert.equal(f.render(), null); f.dispose();
});

test('only the composer being fully outside the viewport exposes a native keyboard button', async () => {
  const f = mount({ active: true }); await nextTick();
  assert.equal(f.observers.length, 1);
  assert.deepEqual(f.observers[0].options, { threshold: 0, rootMargin: '0px 0px -86px 0px' });
  assert.equal(f.render(), null);
  f.observers[0].report(false, {}); assert.equal(f.render(), null);
  f.observers[0].report(false);
  const button = f.render();
  assert.equal(button.type, 'button'); assert.equal(button.props.type, 'button');
  assert.equal(button.props['aria-label'], '回到输入框');
  assert.equal(button.props['aria-controls'], 'chat-composer');
  assert.deepEqual(f.events, []);
  button.props.onClick(); assert.deepEqual(f.events, [['jump']]);
  f.observers[0].report(true); assert.equal(f.render(), null);
  button.props.onClick(); assert.equal(f.events.length, 1); f.dispose();
});

test('busy state hides the action and a previously rendered button cannot activate it', async () => {
  const f = mount({ active: true }); await nextTick(); f.observers[0].report(false);
  const button = f.render(); f.props.disabled = true;
  assert.equal(f.render(), null); button.props.onClick(); assert.deepEqual(f.events, []);
  f.props.disabled = false; assert.equal(f.render().type, 'button');
  assert.equal(f.observers.length, 1); f.dispose();
});

test('page hiding disconnects immediately and late observer callbacks cannot resurrect the action', async () => {
  const f = mount({ active: true }); await nextTick(); const old = f.observers[0]; old.report(false);
  const button = f.render(); f.props.active = false;
  assert.equal(old.disconnected, true); old.report(false); button.props.onClick();
  assert.equal(f.render(), null); assert.deepEqual(f.events, []);
  const nextTarget = {}; f.target(nextTarget); f.props.active = true; await nextTick();
  assert.equal(f.observers.length, 2); assert.equal(f.observers[1].target, nextTarget);
  old.report(false); assert.equal(f.render(), null);
  f.observers[1].report(false); assert.equal(f.render().type, 'button');
  old.report(true); assert.equal(f.render().type, 'button'); f.dispose();
});

test('unmounting disconnects observers and ignores callbacks or activation after disposal', async () => {
  const f = mount({ active: true }); await nextTick(); f.observers[0].report(false);
  const button = f.render(); f.dispose();
  assert.equal(f.observers[0].disconnected, true); f.observers[0].report(false);
  button.props.onClick(); assert.equal(f.render(), null); assert.deepEqual(f.events, []);
});

test('unmount or rapid hide before next render cannot install a stale observer', async () => {
  const removed = mount({ active: true }); removed.dispose(); await nextTick();
  assert.equal(removed.observers.length, 0);
  const hidden = mount({ active: true }); hidden.props.active = false; await nextTick();
  assert.equal(hidden.observers.length, 0); hidden.dispose();
});

test('unsupported browsers, observer failures and absent target keep ordinary scrolling intact', async () => {
  for (const options of [{ available: false }, { broken: true }, { target: null }]) {
    const f = mount({ active: true, ...options }); await nextTick();
    assert.equal(f.render(), null); assert.deepEqual(f.events, []); f.dispose();
  }
  const f = mount({ active: true, target: null }); await nextTick();
  f.props.active = false; f.target({}); f.props.active = true; await nextTick();
  assert.equal(f.observers.length, 1); f.dispose();
});

test('jump component contains no send, network, polling, storage or automatic scrolling behavior', () => {
  assert.doesNotMatch(source, /api\.|fetch\(|XMLHttpRequest|setInterval|setTimeout|localStorage|sessionStorage|uni\.|scrollIntoView|addEventListener|emit\('send'/);
  assert.match(source, /min-height:44px/); assert.match(source, /safe-area-inset-bottom/);
  assert.match(source, /:focus-visible/);
});
