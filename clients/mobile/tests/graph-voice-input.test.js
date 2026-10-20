import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, watch, reactive, effectScope } from 'vue';

const root = new URL('../src/', import.meta.url);
const raw = readFileSync(new URL('components/GraphVoiceInput.vue', root), 'utf8');
const flush = async () => { for (let n = 0; n < 24; n += 1) await Promise.resolve(); };
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
function mount(options = {}) {
  const filename = 'GraphVoiceInput.vue';
  const { descriptor, errors } = parse(raw, { filename }); assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: filename });
  const template = compileTemplate({ source: descriptor.template.content, filename, id: filename, compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(template.errors, []);
  const props = reactive({ active: true, disabled: false, resetKey: 'a', ...options.props });
  const prefs = { mode: 'demo', ...options.prefs };
  const events = [], calls = { start: 0, stop: 0, cancel: 0 }, hooks = {};
  const scope = {
    ref, computed, watch, onBeforeUnmount: fn => { hooks.unmount = fn; },
    getPreferences: () => prefs,
    startRecording: async ({ onState }) => { calls.start += 1; calls.onState = onState; },
    stopRecording: async () => { calls.stop += 1; return '  能源安全与进口依赖  '; },
    cancelRecording: () => { calls.cancel += 1; }, ...options.voice,
  };
  const source = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), source)(...Object.values(scope));
  const effects = effectScope();
  const ui = effects.run(() => component.setup(props, { expose() {}, emit: (...event) => events.push(event) }));
  return { ui, props, prefs, calls, events, dispose() { hooks.unmount?.(); effects.stop(); } };
}
function transcripts(f) { return f.events.filter(([name]) => name === 'transcript').map(([, text]) => text); }
function busyEvents(f) { return f.events.filter(([name]) => name === 'busy').map(([, value]) => value); }

test('demo graph voice starts only after a click and explicitly labels its fixed locator example', async () => {
  const f = mount();
  assert.equal(f.calls.start, 0); assert.deepEqual(f.events, []);
  await f.ui.toggle(); assert.equal(f.ui.phase.value, 'recording');
  assert.match(f.ui.stageLabel.value, /演示.*未使用麦克风/);
  assert.match(f.ui.notice.value, /不会开启麦克风或上传/);
  assert.deepEqual(busyEvents(f), [true]); assert.deepEqual(transcripts(f), []);
  await f.ui.toggle(); assert.equal(f.calls.stop, 1);
  assert.deepEqual(transcripts(f), ['可再生能源投资和能源安全有什么联系？']);
  assert.deepEqual(f.events.slice(-2), [['busy', false], ['transcript', '可再生能源投资和能源安全有什么联系？']]);
  assert.match(f.ui.notice.value, /演示定位文字，未开麦/); assert.match(f.ui.notice.value, /检查/); f.dispose();
});
test('connected graph voice keeps every ASR character, even when it resembles a demo example', async () => {
  const text = '  【演示转写，未使用麦克风】可再生能源投资如何影响能源安全？\n';
  const f = mount({ prefs: { mode: 'connected' }, voice: { stopRecording: async () => text } });
  await f.ui.toggle(); await f.ui.toggle();
  assert.deepEqual(transcripts(f), [text]); assert.doesNotMatch(f.ui.notice.value, /演示定位文字/); f.dispose();
});
test('automatic device stop holds audio ready and never uploads until a second explicit action', async () => {
  const f = mount({ prefs: { mode: 'connected' } }); await f.ui.toggle();
  f.calls.onState({ state: 'stopped', reason: 'limit' });
  assert.equal(f.ui.phase.value, 'ready'); assert.equal(f.ui.busy.value, true);
  assert.match(f.ui.notice.value, /尚未上传/); assert.equal(f.calls.stop, 0);
  assert.deepEqual(transcripts(f), []); await f.ui.toggle();
  assert.equal(f.calls.stop, 1); assert.equal(transcripts(f).length, 1); f.dispose();
});
test('ready event during permission start is not overwritten when the start promise resolves', async () => {
  const permission = deferred(); let onState;
  const f = mount({ voice: { startRecording: options => { onState = options.onState; return permission.promise; } } });
  const starting = f.ui.toggle(); onState({ state: 'recording' }); onState({ state: 'stopped' });
  permission.resolve(); await starting; assert.equal(f.ui.phase.value, 'ready'); assert.equal(f.calls.stop, 0); f.dispose();
});
for (const boundary of ['hide', 'disabled', 'session', 'unmount', 'cancel']) {
  for (const phase of ['preparing', 'transcribing']) {
    test(`${boundary} invalidates ${phase} and ignores all late callbacks and text`, async () => {
      const wait = deferred(); let onState;
      const voice = phase === 'preparing'
        ? { startRecording: options => { onState = options.onState; return wait.promise; } }
        : { stopRecording: () => wait.promise };
      const f = mount({ voice });
      let pending = f.ui.toggle();
      if (phase === 'transcribing') { await pending; onState = f.calls.onState; pending = f.ui.toggle(); }
      assert.equal(f.ui.phase.value, phase);
      if (boundary === 'hide') f.props.active = false;
      else if (boundary === 'disabled') f.props.disabled = true;
      else if (boundary === 'session') f.props.resetKey = 'b';
      else if (boundary === 'unmount') f.dispose();
      else f.ui.cancel();
      assert.equal(f.ui.busy.value, false); assert.equal(f.calls.cancel, 1);
      onState?.({ state: 'recording' }); onState?.({ state: 'stopped' });
      wait.resolve('Do not locate this'); await pending;
      assert.equal(f.ui.phase.value, 'idle'); assert.deepEqual(transcripts(f), []);
      assert.deepEqual(busyEvents(f), [true, false]); f.dispose();
    });
  }
}
test('idle lifecycle cleanup does not cancel a different page recording', async () => {
  const f = mount(); f.props.active = false; f.props.disabled = true; f.props.resetKey = 'b';
  f.dispose(); assert.equal(f.calls.cancel, 0); assert.deepEqual(busyEvents(f), []);
});
test('recording and ready audio are cancelled synchronously when navigation changes', async () => {
  for (const ready of [false, true]) {
    const f = mount(); await f.ui.toggle(); if (ready) f.calls.onState({ state: 'stopped' });
    f.props.resetKey = 'b'; assert.equal(f.calls.cancel, 1); assert.equal(f.calls.stop, 0);
    assert.equal(f.ui.busy.value, false); f.dispose();
  }
});
test('inactive, external busy and invitation modes cannot start a voice operation', async () => {
  for (const options of [{ props: { active: false } }, { props: { disabled: true } }, { prefs: { connectionKind: 'invite' } }]) {
    const f = mount(options); await f.ui.toggle(); assert.equal(f.calls.start, 0); assert.equal(f.calls.stop, 0); f.dispose();
  }
  const f = mount(); f.prefs.connectionKind = 'invite'; await f.ui.toggle(); assert.equal(f.calls.start, 0); f.dispose();
});
test('one preparing or transcribing operation cannot be double dispatched', async () => {
  const preparation = deferred(), transcription = deferred(); let starts = 0, stops = 0;
  const f = mount({ voice: { startRecording: () => { starts++; return preparation.promise; }, stopRecording: () => { stops++; return transcription.promise; } } });
  const start = f.ui.toggle(); await f.ui.toggle(); assert.equal(starts, 1);
  preparation.resolve(); await start;
  const stop = f.ui.toggle(); await f.ui.toggle(); assert.equal(stops, 1);
  transcription.resolve('定位'); await stop; f.dispose();
});
test('device stop while transcribing cannot reopen the upload button', async () => {
  const pending = deferred(); const f = mount({ voice: { stopRecording: () => pending.promise } });
  await f.ui.toggle(); const stopping = f.ui.toggle(); f.calls.onState({ state: 'stopped' });
  assert.equal(f.ui.phase.value, 'transcribing'); assert.equal(f.ui.unavailable.value, true);
  pending.resolve('定位'); await stopping; f.dispose();
});
test('errors are recoverable and never display raw provider URLs, tokens or messages', async () => {
  let attempts = 0;
  const f = mount({ voice: { startRecording: async () => { if (!attempts++) throw new Error('https://example.invalid?token=do-not-expose'); } } });
  await f.ui.toggle(); assert.equal(f.ui.phase.value, 'idle'); assert.equal(f.ui.busy.value, false);
  assert.doesNotMatch(f.ui.notice.value, /example|token|expose/); assert.match(f.ui.notice.value, /权限或语音服务/);
  await f.ui.toggle(); assert.equal(f.ui.phase.value, 'recording'); f.dispose();
});
test('a shared-service busy preflight does not cancel the existing recording owner', async () => {
  const f = mount({ voice: { startRecording: async () => { throw new Error('上一段录音尚未结束'); } } });
  await f.ui.toggle(); assert.equal(f.ui.busy.value, false); assert.equal(f.calls.cancel, 0);
  assert.deepEqual(transcripts(f), []); f.dispose(); assert.equal(f.calls.cancel, 0);
});
test('late callbacks and errors from a cancelled attempt cannot stop its successor', async () => {
  const old = deferred(); const callbacks = []; let starts = 0;
  const f = mount({ voice: { startRecording: ({onState}) => { callbacks.push(onState); return starts++ ? Promise.resolve() : old.promise; } } });
  const first = f.ui.toggle(); f.ui.cancel(); await f.ui.toggle();
  assert.equal(f.ui.phase.value, 'recording'); const cancellations = f.calls.cancel;
  callbacks[0]({state:'error'}); callbacks[0]({state:'stopped'}); old.reject(new Error('old failure')); await first;
  assert.equal(f.ui.phase.value, 'recording'); assert.equal(f.calls.cancel, cancellations);
  await f.ui.toggle(); assert.equal(transcripts(f).length, 1); f.dispose();
});
test('device error drops operation and late text even if stop promise subsequently resolves', async () => {
  const pending = deferred(); const f = mount({ voice: { stopRecording: () => pending.promise } });
  await f.ui.toggle(); const stopping = f.ui.toggle(); f.calls.onState({ state: 'error', message: 'PRIVATE TOKEN' });
  pending.resolve('stale'); await stopping; assert.deepEqual(transcripts(f), []);
  assert.doesNotMatch(f.ui.notice.value, /PRIVATE/); assert.equal(f.calls.cancel, 1); f.dispose();
});
test('empty results do not alter the parent draft; long results are emitted without clipping', async () => {
  for (const text of ['', '  ', '全文'.repeat(1500)]) {
    const f = mount({ prefs: { mode: 'connected' }, voice: { stopRecording: async () => text } });
    await f.ui.toggle(); await f.ui.toggle();
    assert.deepEqual(transcripts(f), text.trim() ? [text] : []); assert.equal(f.ui.busy.value, false); f.dispose();
  }
});
test('cancel stays usable during preparation or transcription and reports the remote-processing boundary', async () => {
  const pending = deferred(); const f = mount({ voice: { stopRecording: () => pending.promise } });
  await f.ui.toggle(); const stopping = f.ui.toggle();
  f.ui.onKeydown({ key: ' ', preventDefault() {} }, 'cancel');
  assert.equal(f.calls.cancel, 1); assert.match(f.ui.notice.value, /已经提交服务端/);
  pending.resolve('ignored'); await stopping; assert.deepEqual(transcripts(f), []); f.dispose();
});
test('keyboard accepts only unmodified non-repeated Enter/Space and preserves IME input', async () => {
  const f = mount(); let prevented = 0;
  for (const extra of [{ key: 'a' }, { repeat: true }, { isComposing: true }, { keyCode: 229 }, { ctrlKey: true }, { metaKey: true }, { altKey: true }, { shiftKey: true }, { defaultPrevented: true }]) {
    f.ui.onKeydown({ key: 'Enter', preventDefault() { prevented++; }, ...extra }, 'toggle');
  }
  assert.equal(f.calls.start, 0); assert.equal(prevented, 0);
  f.ui.onKeydown({ key: 'Enter', preventDefault() { prevented++; } }, 'toggle'); await flush();
  assert.equal(f.calls.start, 1); assert.equal(prevented, 1);
  f.ui.onKeydown({ key: ' ', preventDefault() {} }, 'toggle'); await flush();
  assert.equal(f.calls.stop, 1); f.dispose();
});

function preprocess(source, platform) {
  const stack = [true], lines = [];
  for (const line of source.split('\n')) {
    const conditional = line.match(/^\s*\/\/\s*#(ifdef|ifndef)\s+(\S+)/);
    if (conditional) stack.push(stack.at(-1) && (conditional[1] === 'ifdef' ? conditional[2] === platform : conditional[2] !== platform));
    else if (/^\s*\/\/\s*#endif/.test(line)) stack.pop();
    else if (stack.at(-1)) lines.push(line);
  }
  assert.equal(stack.length, 1);
  return lines.join('\n').replace(/^import .*;\s*$/gm, '').replaceAll('export ', '');
}
// Actual shared voice service, fake final platform I/O. These tests never use
// a microphone, network, model, or real credentials on any target platform.
function platformVoice(platform) {
  const calls = { mic: 0, starts: 0, stopped: 0, uploads: 0, removed: [] };
  const callbacks = {}, prefs = { mode: 'connected', baseUrl: 'https://offline.invalid' };
  const native = {
    onStart: fn => { callbacks.start = fn; }, onStop: fn => { callbacks.stop = fn; }, onError: fn => { callbacks.error = fn; },
    start() { calls.starts++; queueMicrotask(() => callbacks.start()); },
    stop() { calls.stopped++; queueMicrotask(() => callbacks.stop({ tempFilePath: '/offline-recording.mp3', fileSize: 3 })); },
  };
  const track = { stop() { calls.stopped++; } };
  class Recorder {
    static isTypeSupported() { return true; }
    state = 'inactive';
    start() { calls.starts++; this.state = 'recording'; }
    stop() { this.state = 'inactive'; queueMicrotask(() => { this.ondataavailable({ data: new Blob(['fake']) }); this.onstop(); }); }
  }
  const context = {
    Blob, File, queueMicrotask, setTimeout, clearTimeout,
    URL: { createObjectURL: () => 'blob:offline-recording', revokeObjectURL: url => calls.removed.push(url) },
    navigator: { mediaDevices: { getUserMedia: async () => { calls.mic++; return { getTracks: () => [track] }; } } }, MediaRecorder: Recorder,
    getPreferences: () => prefs,
    connection: () => ({ url: 'https://offline.invalid/api', header: { Authorization: 'offline-only' } }),
    request: async () => ({ asr: { available: true } }),
    uploadFile: async () => { calls.uploads++; return { text: '  能源安全  ' }; },
    ensureWeChatPrivacy: async () => {}, ensureWeChatRecordingPermission: async () => {},
    uni: {
      showModal: options => options.success({ confirm: true }), getRecorderManager: () => native,
      getFileSystemManager: () => ({ unlink: options => calls.removed.push(options.filePath) }),
    },
    plus: { io: { resolveLocalFileSystemURL: (path, done) => done({ isFile: true, remove: () => calls.removed.push(path) }) } },
  };
  const voiceRaw = readFileSync(new URL('services/voice.js', root), 'utf8');
  vm.runInNewContext(preprocess(voiceRaw, platform) + '\nglobalThis.voice = { startRecording, stopRecording, cancelRecording };', context);
  return { voice: context.voice, calls };
}
for (const platform of ['H5', 'MP-WEIXIN', 'APP-PLUS']) {
  test(`${platform}: component and real voice state machine do one explicit upload and cleanup`, async () => {
    const service = platformVoice(platform), f = mount({ prefs: { mode: 'connected' }, voice: service.voice });
    try {
      await f.ui.toggle(); assert.equal(f.ui.phase.value, 'recording'); assert.equal(service.calls.uploads, 0);
      await f.ui.toggle(); assert.deepEqual(transcripts(f), ['  能源安全  ']);
      assert.equal(service.calls.uploads, 1); assert.equal(service.calls.starts, 1);
      assert.ok(service.calls.removed.length > 0); assert.equal(f.ui.busy.value, false);
    } finally { f.dispose(); }
  });
  test(`${platform}: hiding cancels actual platform capture without upload or locator emission`, async () => {
    const service = platformVoice(platform), f = mount({ prefs: { mode: 'connected' }, voice: service.voice });
    try {
      await f.ui.toggle(); f.props.active = false; await flush();
      assert.equal(service.calls.uploads, 0); assert.ok(service.calls.stopped > 0);
      assert.deepEqual(transcripts(f), []); assert.equal(f.ui.busy.value, false);
    } finally { f.dispose(); }
  });
}
