import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';
import {File} from 'node:buffer';

const require=createRequire(import.meta.url);
const {preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
const raw=readFileSync(new URL('../src/services/api.js',import.meta.url),'utf8');
const code=preprocess(raw,{H5:true},{type:'js'}).replace(/^import[^;]+;\s*$/gm,'').replaceAll('export ','');
const path='/research/documents?session_id=%2300001';
const settle=()=>new Promise(resolve=>setImmediate(resolve));
function harness({invite=true}={}) {
  const storage=new Map(),calls=[],timers=new Map();let timerId=0;
  const context={
    ensureWeChatPrivacy:async()=>{},AbortController,FormData,File,URL,Uint8Array,ArrayBuffer,Promise,
    setTimeout:fn=>{const id=++timerId;timers.set(id,fn);return id;},clearTimeout:id=>timers.delete(id),
    window:{location:{origin:'https://invite.example'}},
    uni:{getStorageSync:key=>storage.get(key),setStorageSync:(key,value)=>storage.set(key,value),
      request:options=>{calls.push({url:options.url,options,finish:(data={},status=200)=>options.success({statusCode:status,data})});return {abort:()=>options.fail({})};},
      uploadFile:options=>{calls.push({url:options.url,options,finish:(data={},status=200)=>options.success({statusCode:status,data})});return {abort:()=>options.fail({})};}},
    fetch:(url,options)=>new Promise((resolve,reject)=>{
      const call={url,options,aborted:false,finish:(data={},status=200)=>resolve({ok:status>=200&&status<300,status,json:async()=>data})};
      options.signal.addEventListener('abort',()=>{call.aborted=true;reject(new Error('Aborted'));},{once:true});calls.push(call);
    }),
  };
  vm.runInNewContext(code+'\nglobalThis.service={request,researchRequest,researchStatusRequest,researchUpload,uploadFile,activateInviteSession,savePreferences,api};',context);
  const service=context.service;
  const login=()=>service.activateInviteSession({access_token:'synthetic-test-token-only',expires_in:3600,user:{id:'test-user',username:'tester'}});
  if (invite) login();else service.savePreferences({mode:'connected',baseUrl:'https://private.example',accessToken:'synthetic-test-code'});
  return {service,calls,timers,login};
}

test('a tab navigation waits for the background GET response before opening the next service request',async()=>{
  const h=harness(),poll=h.service.researchStatusRequest(path),next=h.service.request('/capabilities');
  assert.equal(h.calls.length,1);assert.match(h.calls[0].url,/research\/documents/);assert.equal(h.calls[0].options.method,'GET');
  h.calls[0].finish({documents:[]});await poll;await settle();assert.equal(h.calls.length,2);assert.match(h.calls[1].url,/capabilities$/);
  h.calls[1].finish({enabled:true});await next;assert.equal(h.calls.length,2);assert.equal(h.timers.size,0);
});

test('a failed status read releases the next request once, without reissuing either request',async()=>{
  const h=harness(),poll=h.service.researchStatusRequest(path),failure=assert.rejects(poll,/服务尚未配置好/),next=h.service.request('/sessions');
  h.calls[0].finish({},503);await failure;await settle();assert.equal(h.calls.length,2);h.calls[1].finish({sessions:[]});await next;assert.equal(h.calls.length,2);
});

test('two reads of the same status are single-flight and another session cannot race them',async()=>{
  const h=harness(),first=h.service.researchStatusRequest(path),second=h.service.researchStatusRequest(path);
  assert.equal(first,second);await assert.rejects(h.service.researchStatusRequest('/research/documents?session_id=%2300002'),/仍在读取/);
  assert.equal(h.calls.length,1);h.calls[0].finish({documents:[]});await Promise.all([first,second]);assert.equal(h.calls.length,1);
});

test('identity replacement cancels old reads and drops queued calls rather than sending them as a new user',async()=>{
  const h=harness(),poll=h.service.researchStatusRequest(path),next=h.service.request('/sessions');
  const old=assert.rejects(poll,/身份或连接已改变/),queued=assert.rejects(next,/身份或连接已改变/);h.login();await Promise.all([old,queued]);
  assert.equal(h.calls.length,1);assert.equal(h.calls[0].aborted,true);
  const fresh=h.service.request('/capabilities');assert.equal(h.calls.length,2);h.calls[1].finish({});await fresh;
});

test('a queued answer POST preserves its original request ID and sends only once',async()=>{
  const h=harness(),poll=h.service.researchStatusRequest(path);
  const payload={session_id:'#00001',question:'Keep this exact question',request_id:'a1445267-f7e2-4945-bacc-bf5617038e55',use_memory:false};
  const answer=h.service.request('/query',payload,'POST');assert.equal(h.calls.length,1);
  h.calls[0].finish({documents:[]});await poll;await settle();assert.equal(h.calls.length,2);assert.equal(h.calls[1].options.method,'POST');
  assert.deepEqual(JSON.parse(h.calls[1].options.body),payload);h.calls[1].finish({request_status:'completed',answer:'Answer'});await answer;assert.equal(h.calls.length,2);
});

test('research uploads also wait for the read and send the explicitly selected file only once',async()=>{
  const h=harness(),poll=h.service.researchStatusRequest(path),file=new File(['User selected material'],'notes.txt');
  const upload=h.service.researchUpload('blob:test',{session_id:'#00001'},file);assert.equal(h.calls.length,1);
  h.calls[0].finish({documents:[]});await poll;await settle();assert.equal(h.calls.length,2);assert.match(h.calls[1].url,/research\/upload$/);
  assert.equal(h.calls[1].options.body.get('session_id'),'#00001');assert.equal(await h.calls[1].options.body.get('file').text(),'User selected material');
  h.calls[1].finish({document:{status:'queued'}});await upload;assert.equal(h.calls.length,2);
});

test('private native requests and uploads obey the same opt-in status barrier',async()=>{
  const h=harness({invite:false}),poll=h.service.researchStatusRequest(path),next=h.service.request('/sessions');assert.equal(h.calls.length,1);
  h.calls[0].finish({documents:[]});await poll;await settle();assert.equal(h.calls.length,2);h.calls[1].finish({sessions:[]});await next;
  const secondPoll=h.service.researchStatusRequest(path),upload=h.service.uploadFile('/upload','/selected/file',{session_id:'#00001'});assert.equal(h.calls.length,3);
  h.calls[2].finish({documents:[]});await secondPoll;await settle();assert.equal(h.calls.length,4);h.calls[3].finish({status:'queued'});await upload;
});

test('leaving a page while its upload waits for status prevents the later file transmission',async()=>{
  for(const invite of [true,false]) {
    const h=harness({invite}),poll=h.service.researchStatusRequest(path);let visible=true;
    const file=new File(['Private selected file'],'notes.txt');
    const upload=invite ? h.service.researchUpload('blob:test',{session_id:'#00001'},file,{isCurrent:()=>visible})
      : h.service.uploadFile('/upload','/selected/file',{session_id:'#00001'},file,{isCurrent:()=>visible});
    const cancelled=assert.rejects(upload,/没有继续上传/);visible=false;h.calls[0].finish({documents:[]});await poll;await cancelled;assert.equal(h.calls.length,1);
  }
});

test('ordinary reads stay ordinary and the background endpoint cannot submit any mutation',async()=>{
  const h=harness(),first=h.service.researchRequest(path),second=h.service.request('/capabilities');assert.equal(h.calls.length,2);
  for(const invalid of ['/query','/research/documents/delete','/research/source?chunk_id=x','/research/documents?session_id=other','/research/documents?session_id=%2300001&extra=yes']) await assert.rejects(h.service.researchStatusRequest(invalid),/只支持/);
  assert.equal(h.calls.length,2);h.calls[0].finish({documents:[]});h.calls[1].finish({});await Promise.all([first,second]);
});

test('a cancelled operation is rejected before entering the status barrier or touching the network',async()=>{
  for(const invite of [true,false]) {
    const h=harness({invite});
    await assert.rejects(h.service.request('/voice/tts',{text:'No audio requested'},'POST','arraybuffer',{isCurrent:()=>false}),{code:'REQUEST_CANCELLED'});
    assert.equal(h.calls.length,0);
    const poll=h.service.researchStatusRequest(path);
    await assert.rejects(h.service.request('/voice/tts',{text:'No audio requested'},'POST','arraybuffer',{isCurrent:()=>false}),{code:'REQUEST_CANCELLED'});
    assert.equal(h.calls.length,1);h.calls[0].finish({documents:[]});await poll;await settle();assert.equal(h.calls.length,1);
  }
});

test('cancelling speech while it waits for status prevents the later synthesis POST',async()=>{
  for(const invite of [true,false]) {
    const h=harness({invite}),poll=h.service.researchStatusRequest(path);let current=true;
    const speech=h.service.request('/voice/tts',{text:'Read only if still requested'},'POST','arraybuffer',{isCurrent:()=>current});
    const cancelled=assert.rejects(speech,{code:'REQUEST_CANCELLED',message:'操作已取消，没有发送请求。'});
    current=false;h.calls[0].finish({documents:[]});await poll;await cancelled;
    assert.equal(h.calls.length,1);assert.equal(h.timers.size,0);
  }
});

test('identity replacement drops a guarded queued operation even if its caller remains current',async()=>{
  const h=harness(),poll=h.service.researchStatusRequest(path);
  const speech=h.service.request('/voice/tts',{text:'Original account only'},'POST','arraybuffer',{isCurrent:()=>true});
  const old=assert.rejects(poll,/身份或连接已改变/),queued=assert.rejects(speech,/身份或连接已改变/);
  h.login();await Promise.all([old,queued]);assert.equal(h.calls.length,1);assert.equal(h.calls[0].aborted,true);
});

test('a current queued operation dispatches exactly once with unchanged payload and response type',async()=>{
  const h=harness({invite:false}),poll=h.service.researchStatusRequest(path);let checks=0;
  const payload={text:'One explicit playback'};
  const speech=h.service.request('/voice/tts',payload,'POST','arraybuffer',{isCurrent:()=>{checks+=1;return true;}});
  assert.equal(checks,1);assert.equal(h.calls.length,1);
  h.calls[0].finish({documents:[]});await poll;await settle();
  assert.equal(checks,2);assert.equal(h.calls.length,2);
  assert.equal(h.calls[1].options.method,'POST');assert.equal(h.calls[1].options.responseType,'arraybuffer');assert.equal(h.calls[1].options.data,payload);
  const audio=new ArrayBuffer(2);h.calls[1].finish(audio);assert.equal(await speech,audio);assert.equal(h.calls.length,2);
});

test('operation guards fail closed on callback errors before and after the barrier',async()=>{
  const h=harness();
  const broken=()=>{throw new Error('Untrusted diagnostic details');};
  await assert.rejects(h.service.request('/voice/tts',{},'POST','text',{isCurrent:broken}),{code:'REQUEST_CANCELLED',message:'操作已取消，没有发送请求。'});
  assert.equal(h.calls.length,0);
  const poll=h.service.researchStatusRequest(path);let current=true;
  const speech=h.service.request('/voice/tts',{},'POST','text',{isCurrent:()=>current ? true : broken()});
  const cancelled=assert.rejects(speech,{code:'REQUEST_CANCELLED',message:'操作已取消，没有发送请求。'});
  current=false;h.calls[0].finish({documents:[]});await poll;await cancelled;assert.equal(h.calls.length,1);
});

test('truthy non-boolean operation guard results cannot authorize a request',async()=>{
  const h=harness();
  for(const value of [undefined,null,1,'true',Promise.resolve(true)]) {
    await assert.rejects(h.service.request('/voice/tts',{},'POST','text',{isCurrent:()=>value}),{code:'REQUEST_CANCELLED'});
  }
  assert.equal(h.calls.length,0);
});
