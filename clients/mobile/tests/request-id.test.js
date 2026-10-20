import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';

const require=createRequire(import.meta.url);
const {preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
const source=readFileSync(new URL('../src/services/api.js',import.meta.url),'utf8');
const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
function harness(platform='MP_WEIXIN',{crypto,random}={}) {
  const calls=[],timers=new Map();let timerId=0;
  const context={crypto,ArrayBuffer,Uint8Array,
    wx:{getRandomValues:options=>{calls.push(options);random?.(options);}},
    setTimeout:(callback,delay)=>{timers.set(++timerId,{callback,delay});return timerId;},clearTimeout:key=>timers.delete(key)};
  const code=preprocess(source,{H5:false,MP_WEIXIN:false,APP_PLUS:false,[platform]:true},{type:'js'}).replace(/^import[^;]+;\s*$/gm,'').replaceAll('export ','');
  vm.runInNewContext(code+'\nglobalThis.ids={createQueryRequestId,createQueryRequestIdAsync,queryRequestIdSupported};',context);
  return {...context.ids,calls,timers,wx:context.wx};
}

test('H5 preserves synchronous Web Crypto IDs and async helper never calls a native random adapter',async()=>{
  const h=harness('H5',{crypto:{randomUUID:()=>id}});assert.equal(h.queryRequestIdSupported(),true);assert.equal(h.createQueryRequestId(),id);assert.equal(await h.createQueryRequestIdAsync(),id);assert.equal(h.calls.length,0);assert.equal(h.timers.size,0);
});
test('WeChat gets exactly sixteen SDK cryptographic bytes and formats a protocol-compatible UUID v4',async()=>{
  const h=harness();assert.equal(h.queryRequestIdSupported(),true);assert.equal(h.calls.length,0);
  const pending=h.createQueryRequestIdAsync();assert.equal(h.calls.length,1);assert.equal(h.calls[0].length,16);assert.equal([...h.timers.values()][0].delay,8000);
  const bytes=new Uint8Array(Array.from({length:16},(_,index)=>index));h.calls[0].success({randomValues:bytes.buffer});
  const result=await pending;assert.equal(result,'00010203-0405-4607-8809-0a0b0c0d0e0f');assert.equal(h.timers.size,0);assert.equal(bytes[6],6,'the SDK-returned buffer is not mutated');
  h.calls[0].fail({errMsg:'late callback'});assert.equal(h.calls.length,1);
});
test('missing native API or unsupported App/H5 fails closed without weak IDs or native callbacks',async()=>{
  for(const platform of ['H5','APP_PLUS','MP_WEIXIN']) {
    const h=harness(platform);if(platform==='MP_WEIXIN') h.wx.getRandomValues=undefined;
    assert.equal(h.queryRequestIdSupported(),false);await assert.rejects(h.createQueryRequestIdAsync(),/本版尚未接入.*没有发送问题/);assert.equal(h.calls.length,0);assert.equal(h.timers.size,0);
  }
});
test('WeChat malformed random values, native failures and thrown errors do not create usable IDs or expose raw errors',async()=>{
  for(const result of [null,{}, {randomValues:new Uint8Array(16)}, {randomValues:new ArrayBuffer(15)}, {randomValues:new ArrayBuffer(17)}, {randomValues:'secret'}]) {
    const h=harness(),pending=h.createQueryRequestIdAsync();h.calls[0].success(result);await assert.rejects(pending,/没有发送问题/);assert.equal(h.timers.size,0);
  }
  for(const action of [options=>options.fail({errMsg:'secret-native-path'}),()=>{throw new Error('secret-native-path');}]) {
    const h=harness('MP_WEIXIN',{random:action});await assert.rejects(h.createQueryRequestIdAsync(),error=>/没有发送问题/.test(error.message)&&!error.message.includes('secret-native-path'));assert.equal(h.calls.length,1);assert.equal(h.timers.size,0);
  }
});
test('native RNG deadline rejects once and ignores late success without retrying or creating a question',async()=>{
  const h=harness(),pending=h.createQueryRequestIdAsync();const timer=[...h.timers.values()][0];timer.callback();await assert.rejects(pending,/没有发送问题/);
  h.calls[0].success({randomValues:new ArrayBuffer(16)});assert.equal(h.calls.length,1);assert.equal(h.timers.size,0);
});
test('lifecycle checks reject before native entry or after its callback without sending any question',async()=>{
  const before=harness();await assert.rejects(before.createQueryRequestIdAsync({isCurrent:()=>false}),/页面、身份或草稿/);assert.equal(before.calls.length,0);
  for(const check of [()=>false,()=>{throw new Error('Page gone');}]) {
    const h=harness();let current=true;const pending=h.createQueryRequestIdAsync({isCurrent:()=>current ? true : check()});current=false;h.calls[0].success({randomValues:new ArrayBuffer(16)});await assert.rejects(pending,/页面、身份或草稿/);assert.equal(h.calls.length,1);assert.equal(h.timers.size,0);
  }
});
test('secure random APIs are never replaced by Math.random or clocks on any compiled platform',()=>{
  for(const platform of ['H5','MP_WEIXIN','APP_PLUS']) {
    const compiled=preprocess(source,{H5:false,MP_WEIXIN:false,APP_PLUS:false,[platform]:true},{type:'js'});
    const ids=compiled.slice(compiled.indexOf('function queryIdFromBytes'),compiled.indexOf('function trackedQueryResponse'));
    assert.doesNotMatch(ids,/Math\.random|Date\.now/);if(platform!=='MP_WEIXIN') assert.doesNotMatch(ids,/wx\.getRandomValues/);
  }
});
