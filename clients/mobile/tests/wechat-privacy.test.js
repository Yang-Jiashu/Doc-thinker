import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const raw = fs.readFileSync(new URL('../src/services/wechat-privacy.js', import.meta.url), 'utf8');
const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };

function harness(platform = 'MP-WEIXIN') {
  let source = raw.replaceAll('export ', '');
  if (platform !== 'MP-WEIXIN') source = source.replace(/\/\/ #ifdef MP-WEIXIN[\s\S]*?\/\/ #endif/g, '');
  const calls = [], timers = new Map(); let timerId = 0;
  const handlers = {
    getPrivacySetting: options => options.success({needAuthorization:true}),
    requirePrivacyAuthorize: options => options.success({}),
    getSetting: options => options.success({authSetting:{}}),
    authorize: options => options.success({}),
  };
  const wx = Object.fromEntries(Object.keys(handlers).map(method => [method, options => {
    calls.push({method,scope:options.scope}); handlers[method](options);
  }]));
  const context = {
    wx,
    setTimeout: fn => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: id => timers.delete(id),
  };
  vm.runInNewContext(source + '\nglobalThis.api = {ensureWeChatPrivacy, ensureWeChatRecordingPermission};', context);
  return {...context.api,context,handlers,calls,timers};
}

test('demo and non-WeChat platforms do not access any WeChat authorization APIs', async () => {
  for (const platform of ['MP-WEIXIN','H5','APP-PLUS']) {
    const h = harness(platform);
    delete h.context.wx;
    await h.ensureWeChatPrivacy();
    await h.ensureWeChatRecordingPermission({mode:'demo'});
    if (platform !== 'MP-WEIXIN') {
      await h.ensureWeChatPrivacy({mode:'connected'});
      await h.ensureWeChatRecordingPermission({mode:'connected'});
    }
    assert.equal(h.calls.length,0);
  }
});

test('privacy requires platform consent only when WeChat reports a pending policy', async () => {
  const h = harness();
  await h.ensureWeChatPrivacy({mode:'connected',action:'上传资料'});
  assert.deepEqual(h.calls.map(call => call.method),['getPrivacySetting','requirePrivacyAuthorize']);
  h.handlers.getPrivacySetting = options => options.success({needAuthorization:false});
  await h.ensureWeChatPrivacy({mode:'connected'});
  assert.equal(h.calls.length,3);
  assert.equal(h.timers.size,0);
});

test('privacy consent is never cached in application state', async () => {
  const h = harness();
  h.handlers.getPrivacySetting = options => options.success({needAuthorization:false});
  await h.ensureWeChatPrivacy({mode:'connected'});
  h.handlers.getPrivacySetting = options => options.success({needAuthorization:true});
  await h.ensureWeChatPrivacy({mode:'connected'});
  assert.deepEqual(h.calls.map(call => call.method),['getPrivacySetting','getPrivacySetting','requirePrivacyAuthorize']);
});

for (const method of ['getPrivacySetting','requirePrivacyAuthorize']) {
  test(`missing ${method} fails closed instead of using the old unguarded recorder`, async () => {
    const h = harness(); delete h.context.wx[method];
    await assert.rejects(h.ensureWeChatPrivacy({mode:'connected'}),/升级微信/);
    assert.equal(h.calls.length,0);
  });
  test(`${method} failure is understandable and is not retried`, async () => {
    const h = harness(); h.handlers[method] = options => options.fail({errMsg:'private SDK detail'});
    await assert.rejects(h.ensureWeChatPrivacy({mode:'connected',action:'上传资料'}),error => /隐私/.test(error.message) && !error.message.includes('private SDK'));
    assert.equal(h.calls.filter(call => call.method === method).length,1);
    assert.equal(h.timers.size,0);
  });
}

test('missing privacy state cannot be treated as already authorized', async () => {
  const h = harness(); h.handlers.getPrivacySetting = options => options.success({});
  await assert.rejects(h.ensureWeChatPrivacy({mode:'connected'}),/状态不明确/);
  assert.equal(h.calls.length,1);
});

test('cancelling during privacy-state lookup cannot open a late authorization dialog', async () => {
  const h = harness(); let callbacks, active = true;
  h.handlers.getPrivacySetting = options => { callbacks = options; };
  const result = h.ensureWeChatPrivacy({mode:'connected',isCurrent:()=>active});
  active = false; callbacks.success({needAuthorization:true});
  await assert.rejects(result,/取消/);
  assert.equal(h.calls.length,1);
});

test('a late consent success does not revive an operation cancelled by navigation', async () => {
  const h = harness(); let callbacks, active = true;
  h.handlers.requirePrivacyAuthorize = options => { callbacks = options; };
  const result = h.ensureWeChatPrivacy({mode:'connected',isCurrent:()=>active});
  await flush(); active = false; callbacks.success({});
  await assert.rejects(result,/取消/);
});

test('a platform dialog timeout rejects once and ignores a later success', async () => {
  const h = harness(); let callbacks;
  h.handlers.requirePrivacyAuthorize = options => { callbacks = options; };
  const result = h.ensureWeChatPrivacy({mode:'connected'});
  await flush(); for (const fn of [...h.timers.values()]) fn();
  await assert.rejects(result,/超时/);
  callbacks.success({});
  assert.equal(h.calls.length,2);
});

test('microphone scope requests are separate from the privacy agreement', async () => {
  const h = harness();
  await h.ensureWeChatRecordingPermission({mode:'connected'});
  assert.deepEqual(h.calls.map(call => call.method),['getSetting','authorize']);
  assert.equal(h.calls[1].scope,'scope.record');
});

test('a previously denied microphone is not repeatedly requested or automatically opened in settings', async () => {
  const h = harness();
  h.handlers.getSetting = options => options.success({authSetting:{'scope.record':false}});
  await assert.rejects(h.ensureWeChatRecordingPermission({mode:'connected'}),/设置.*手动开始/);
  assert.equal(h.calls.length,1);
});

test('new microphone rejection and unavailable scope state fail closed', async () => {
  const h = harness(); h.handlers.authorize = options => options.fail({});
  await assert.rejects(h.ensureWeChatRecordingPermission({mode:'connected'}),/没有开始录音/);
  h.handlers.getSetting = options => options.success({});
  await assert.rejects(h.ensureWeChatRecordingPermission({mode:'connected'}),/状态不明确/);
});
