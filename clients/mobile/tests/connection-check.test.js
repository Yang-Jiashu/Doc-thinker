import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../src/services/api.js',import.meta.url),'utf8')
  .replace(/^import .*;\s*$/gm,'').replaceAll('export ','');
const contract = () => ({audience:'private-development',customer_auth:false,realtime_voice:false,
  voice:'/mobile/voice/capabilities',max_upload_bytes:20 * 1024 * 1024});

function harness({demo = false} = {}) {
  const storage = new Map(), calls = [], timers = new Map(); let timerId = 0, aborts = 0;
  const context = {
    setTimeout: (callback,delay) => {timers.set(++timerId,{callback,delay}); return timerId;},
    clearTimeout: id => timers.delete(id),
    uni: {
      getStorageSync:key => storage.get(key),setStorageSync:(key,value) => storage.set(key,value),
      request: options => {calls.push(options); return {abort() {aborts += 1; options.fail({errMsg:'request:fail abort'});}};},
      uploadFile: () => {throw new Error('diagnostic must not upload');},
      getRecorderManager: () => {throw new Error('diagnostic must not record');},
    },
  };
  vm.runInNewContext(source + '\nglobalThis.service = {api,checkConnection,cancelConnectionCheck,savePreferences,getPreferences};',context);
  const service = context.service;
  if (!demo) service.savePreferences({mode:'connected',baseUrl:'https://private.example',accessToken:'private-test-code'});
  return {...service,context,storage,calls,timers,aborts:()=>aborts};
}

test('connection check is explicit and makes one authenticated capability GET only', async () => {
  const h = harness(); assert.equal(h.calls.length,0);
  const checking = h.api.checkConnection();
  assert.equal(h.calls.length,1); const request = h.calls[0];
  assert.equal(request.url,'https://private.example/api/v1/mobile/capabilities');
  assert.equal(request.method,'GET'); assert.equal(request.timeout,8000); assert.equal(request.dataType,'json');
  assert.equal(request.header.Authorization,'Bearer private-test-code');
  assert.equal('data' in request,false); assert.equal(JSON.stringify([...h.storage.values()]).includes('private-test-code'),false);
  request.success({statusCode:200,data:contract()});
  assert.equal((await checking).connected,true); assert.equal(h.timers.size,0); assert.equal(h.aborts(),0);
});

test('demo and incomplete saved credentials never make a diagnostic request', async () => {
  const h = harness({demo:true});
  assert.equal((await h.checkConnection()).reason,'configuration');
  h.savePreferences({mode:'connected',baseUrl:'https://private.example',accessToken:''});
  assert.equal((await h.checkConnection()).reason,'configuration');
  assert.equal(h.calls.length,0);
});

test('singleflight blocks duplicate diagnostics and permits only a later manual check', async () => {
  const h = harness(), first = h.checkConnection();
  assert.equal((await h.checkConnection()).reason,'busy'); assert.equal(h.calls.length,1);
  h.calls[0].success({statusCode:200,data:contract()}); await first;
  assert.equal(h.calls.length,1); const second = h.checkConnection(); assert.equal(h.calls.length,2);
  h.calls[1].success({statusCode:200,data:contract()}); await second;
});

test('8-second deadline aborts once and discards late success without retry', async () => {
  const h = harness(), checking = h.checkConnection();
  const timer = [...h.timers.values()][0]; assert.equal(timer.delay,8000); timer.callback();
  const result = await checking; assert.equal(result.connected,false); assert.equal(result.reason,'timeout');
  assert.equal(h.aborts(),1); assert.equal(h.timers.size,0);
  h.calls[0].success({statusCode:200,data:contract()}); assert.equal(h.calls.length,1);
});

test('cancel aborts and settles a pending check; late callbacks do not affect a newer check', async () => {
  const h = harness(), first = h.checkConnection(); h.cancelConnectionCheck();
  assert.equal((await first).cancelled,true); assert.equal(h.aborts(),1);
  const second = h.checkConnection(); h.calls[0].success({statusCode:200,data:contract()});
  assert.equal((await h.checkConnection()).reason,'busy');
  h.calls[1].success({statusCode:200,data:contract()}); assert.equal((await second).connected,true);
  assert.equal(h.calls.length,2);
});

test('page invalidation blocks a request at entry and discards an in-flight response', async () => {
  const h = harness(); let visible = false;
  assert.equal((await h.checkConnection({isCurrent:()=>visible})).cancelled,true); assert.equal(h.calls.length,0);
  visible = true; const checking = h.checkConnection({isCurrent:()=>visible}); visible = false;
  h.calls[0].success({statusCode:200,data:contract()}); assert.equal((await checking).cancelled,true);
});

for (const field of ['baseUrl','accessToken','mode','same-values']) {
  test(`saving ${field} immediately cancels the check, even before its late response`, async () => {
    const h = harness(), checking = h.checkConnection();
    const next = {...h.getPreferences()};
    if (field !== 'same-values') next[field] = field === 'mode' ? 'demo' : field === 'baseUrl' ? 'https://new.example' : 'new-code';
    h.savePreferences(next);
    assert.equal((await checking).cancelled,true); assert.equal(h.aborts(),1);
    h.calls[0].success({statusCode:200,data:contract()}); assert.equal(h.calls.length,1);
  });
}

for (const [status,reason] of [[401,'unauthorized'],[403,'unauthorized'],[404,'unavailable'],[503,'not-configured'],[500,'http'],[302,'http']]) {
  test(`HTTP ${status} produces a fixed safe reason without echoing the response or retrying`, async () => {
    const h = harness(), checking = h.checkConnection();
    h.calls[0].success({statusCode:status,data:{message:'https://host?token=SECRET Bearer SECRET'}});
    const result = await checking; assert.equal(result.reason,reason); assert.equal(result.connected,false);
    assert.doesNotMatch(JSON.stringify(result),/SECRET|https:/); assert.equal(h.calls.length,1);
  });
}

for (const [name,data] of [
  ['HTML','<html>another service</html>'],['missing',{}],['array',[]],['null',null],
  ['wrong audience',{...contract(),audience:'public'}],['customer auth',{...contract(),customer_auth:true}],
  ['string boolean',{...contract(),customer_auth:'false'}],['voice redirect',{...contract(),voice:'https://another.example'}],
  ['realtime voice',{...contract(),realtime_voice:true}],['size string',{...contract(),max_upload_bytes:'20971520'}],
  ['size zero',{...contract(),max_upload_bytes:0}],['size too large',{...contract(),max_upload_bytes:21 * 1024 * 1024}],
]) {
  test(`incompatible ${name} capability response is not reported as a successful connection`, async () => {
    const h = harness(), checking = h.checkConnection(); h.calls[0].success({statusCode:200,data});
    const result = await checking; assert.equal(result.connected,false); assert.equal(result.reason,'contract');
    assert.equal(h.calls.length,1);
  });
}

test('optional new backend fields are compatible but not reflected as model readiness', async () => {
  const h = harness(), checking = h.checkConnection();
  h.calls[0].success({statusCode:200,data:{...contract(),document_rag:false,uploads:false,memory_write_enabled:false,raw_message:'SECRET'}});
  const result = await checking; assert.deepEqual(Object.keys(result),['connected']); assert.equal(result.connected,true);
});

test('network and synchronous transport failures return safe fixed reasons without retry', async () => {
  const h = harness(), checking = h.checkConnection();
  h.calls[0].fail({errMsg:'network contains Bearer SECRET'});
  assert.equal((await checking).reason,'network'); assert.equal(h.calls.length,1);
  const timeout = h.checkConnection(); h.calls[1].fail({errMsg:'request:fail timeout https://secret'});
  assert.equal((await timeout).reason,'timeout'); assert.equal(h.calls.length,2);
  h.context.uni.request = () => {throw new Error('secret transport error');};
  assert.equal((await h.checkConnection()).reason,'network'); assert.equal(h.timers.size,0);
});
