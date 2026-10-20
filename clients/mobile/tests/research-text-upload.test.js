import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';
import {File} from 'node:buffer';
import {validResearchExport} from '../src/services/research-export.js';

const require = createRequire(import.meta.url);
const {preprocess} = require('@dcloudio/uni-cli-shared/lib/preprocess');
const raw = readFileSync(new URL('../src/services/research.js',import.meta.url),'utf8');
const SID = '#00001', ID = 'b6fe2407-a253-445d-ac69-c69271138aa7';
const caps = {enabled:true,uploads_allowed:true,max_upload_bytes:10485760,supported_upload_types:['pdf','txt','md'],memory_write_enabled:true};
const doc = () => ({id:ID,session_id:SID,filename:'我的资料.txt',status:'queued',chunk_count:0,byte_size:30});
const clone = value => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return {promise,resolve}; };
const tick = () => new Promise(resolve => setImmediate(resolve));

function harness(platform = 'H5') {
  let key = 'identity-a', prefs = {mode:'connected',baseUrl:'https://research.example',connectionKind:'private'};
  const calls = [], uploads = [], dialogs = [], created = [], revoked = [];
  const handlers = {
    request:async path => path === '/research/capabilities' ? clone(caps) : {},
    upload:async () => ({document:doc()}),
    modal:options => options.success({confirm:true}),
  };
  const context = {
    getPreferences:() => ({...prefs}), researchConnectionKey:() => key,
    researchRequest:async (...args) => { calls.push(args); return handlers.request(...args); },
    researchStatusRequest:async () => { throw new Error('unexpected status request'); },
    researchUpload:async (...args) => { uploads.push(args); return handlers.upload(...args); },
    ensureWeChatPrivacy:async () => {}, validResearchExport,setTimeout,clearTimeout,File,
    URL:{createObjectURL:file => { const path = `blob:local-test-${created.length}`; created.push({path,file}); return path; },revokeObjectURL:path => revoked.push(path)},
    uni:{showModal:options => { dialogs.push(options); handlers.modal(options); },chooseFile:() => { throw new Error('pasted text must not open a file picker'); }},
  };
  const code = preprocess(raw,{[platform]:true},{type:'js'}).replace(/^import[^;]+;\s*$/gm,'').replaceAll('export ','');
  vm.runInNewContext(code+'\nglobalThis.service = research;',context);
  return {service:context.service,context,calls,uploads,dialogs,created,revoked,handlers,
    key:value => { key = value; }, preferences:value => { prefs = {...prefs,...value}; }};
}

test('explicit text submission reuses the real research multipart upload and does not ask a model',async () => {
  const h = harness();
  const result = await h.service.uploadText(SID,{title:'我的资料',text:'A sourced passage.'});
  assert.equal(result.document.id,ID); assert.equal(result.document.status,'queued');
  assert.deepEqual(h.calls.map(([path]) => path),['/research/capabilities']);
  assert.equal(h.uploads.length,1); assert.equal(h.dialogs.length,1);
  const [path,form,file,options] = h.uploads[0];
  assert.equal(path,'blob:local-test-0');
  assert.deepEqual({...form},{session_id:SID,original_filename:'我的资料.txt'});
  assert.ok(file instanceof File); assert.equal(file.name,'我的资料.txt');
  assert.equal(file.type,'text/plain;charset=utf-8'); assert.equal(await file.text(),'A sourced passage.');
  assert.equal(options.isCurrent(),true);
  assert.deepEqual(h.revoked,[path]);
});

test('original whitespace, CRLF, Unicode and punctuation are preserved byte-for-byte',async () => {
  const h = harness(), text = '  原文🙂\r\n\t第二段，“不代表因果”。\n  ';
  await h.service.uploadText(SID,{title:'摘录',text});
  assert.equal(await h.uploads[0][2].text(),text);
  assert.deepEqual(new Uint8Array(await h.uploads[0][2].arrayBuffer()),new TextEncoder().encode(text));
});

test('the consent modal names the destination, character count and later model use',async () => {
  const h = harness();
  await h.service.uploadText(SID,{title:'摘录',text:'🙂中a'});
  assert.match(h.dialogs[0].title,/当前研究会话/);
  assert.match(h.dialogs[0].content,/3 字符/);
  assert.match(h.dialogs[0].content,/https:\/\/research\.example/);
  assert.match(h.dialogs[0].content,/不会自动提问或调用模型/);
  assert.match(h.dialogs[0].content,/之后提问时发送给模型/);
});

test('cancelling consent performs no upload and creates no object URL',async () => {
  const h = harness(); h.handlers.modal = options => options.success({confirm:false});
  const result = await h.service.uploadText(SID,{title:'摘录',text:'Do not upload.'});
  assert.equal(result.cancelled,true); assert.equal(h.uploads.length,0);
  assert.equal(h.created.length,0); assert.equal(h.revoked.length,0);
});

test('every text submission requires new confirmation',async () => {
  const h = harness();
  await h.service.uploadText(SID,{title:'摘录',text:'One.'});
  await h.service.uploadText(SID,{title:'摘录',text:'Two.'});
  assert.equal(h.dialogs.length,2); assert.equal(h.uploads.length,2);
  assert.equal(h.calls.length,1,'only capabilities can be cached, never consent');
});

test('invalid text rejects before capability, confirmation or upload',async () => {
  for (const text of ['', ' \n\t ', undefined, null, 12, {}, '\ud800', '\udc00', 'bad\0text', 'x'.repeat(12001)]) {
    const h = harness();
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text}),/有效原文/);
    assert.equal(h.calls.length,0); assert.equal(h.dialogs.length,0); assert.equal(h.uploads.length,0);
  }
});

test('the text limit counts Unicode characters rather than UTF-16 code units',async () => {
  const h = harness();
  await h.service.uploadText(SID,{title:'资料',text:'🙂'.repeat(12000)});
  assert.equal(h.uploads[0][2].size,48000);
  assert.match(h.dialogs[0].content,/12000 字符/);
  await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'🙂'.repeat(12001)}),/有效原文/);
  assert.equal(h.uploads.length,1);
});

test('invalid titles are rejected, not silently trimmed, truncated or converted from paths',async () => {
  for (const title of ['', undefined, null, {}, '.', '..', ' name', 'name ', 'a/b', 'a\\b', 'bad\0name', 'bad\nname', 'bad\x7fname', '\ud800', 'x'.repeat(117), 'x'.repeat(121)]) {
    const h = harness();
    await assert.rejects(h.service.uploadText(SID,{title,text:'Original'}),/标题/);
    assert.equal(h.calls.length,0); assert.equal(h.uploads.length,0);
  }
});

test('a TXT extension is added only when absent, with the final filename bounded to 120 characters',async () => {
  for (const [title,name] of [['原文','原文.txt'],['notes.txt','notes.txt'],['notes.TXT','notes.TXT'],['paper.pdf','paper.pdf.txt'],['x'.repeat(116),'x'.repeat(116)+'.txt']]) {
    const h = harness(); await h.service.uploadText(SID,{title,text:'Source'});
    assert.equal(h.uploads[0][2].name,name); assert.equal(h.uploads[0][1].original_filename,name);
  }
});

test('invalid sessions cannot upload a document',async () => {
  const h = harness();
  for (const sid of ['', '#demo1', '#000001', '../foreign', undefined]) {
    await assert.rejects(h.service.uploadText(sid,{title:'资料',text:'Original'}),/研究会话/);
  }
  assert.equal(h.calls.length,0); assert.equal(h.uploads.length,0);
});

test('demo and pending invitation cannot submit real text or fabricate a sample result',async () => {
  for (const prefs of [{mode:'demo'},{mode:'connected',connectionKind:'invite',invitePending:true}]) {
    const h = harness(); h.preferences(prefs);
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Original'}),/尚未开放/);
    assert.equal(h.calls.length,0); assert.equal(h.uploads.length,0); assert.equal(h.dialogs.length,0);
  }
});

test('disabled service, denied uploads or missing TXT support cannot be bypassed',async () => {
  for (const response of [{enabled:false},{...caps,uploads_allowed:false},{...caps,supported_upload_types:['pdf']}]) {
    const h = harness(); h.handlers.request = async () => response;
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Original'}),/未开放/);
    assert.equal(h.uploads.length,0); assert.equal(h.dialogs.length,0);
  }
});

test('UTF-8 bytes are checked against the actual advertised upload budget before confirmation',async () => {
  const h = harness(); h.handlers.request = async () => ({...caps,max_upload_bytes:5});
  await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'中文'}),/字节限制/);
  assert.equal(h.dialogs.length,0); assert.equal(h.uploads.length,0);
  const exact = harness(); exact.handlers.request = async () => ({...caps,max_upload_bytes:6});
  await exact.service.uploadText(SID,{title:'资料',text:'中文'});
  assert.equal(exact.uploads[0][2].size,6);
});

test('page or identity changes during capability reads cannot reach confirmation',async () => {
  for (const changeIdentity of [false,true]) {
    const h = harness(), wait = deferred(); let visible = true;
    h.handlers.request = () => wait.promise;
    const operation = h.service.uploadText(SID,{title:'资料',text:'Private',isCurrent:() => visible});
    await tick(); if (changeIdentity) h.key('identity-b'); else visible = false;
    wait.resolve(clone(caps)); await assert.rejects(operation,/身份|页面/);
    assert.equal(h.dialogs.length,0); assert.equal(h.uploads.length,0);
  }
});

test('page or identity changes while confirming cannot transmit text',async () => {
  for (const changeIdentity of [false,true]) {
    const h = harness(); let visible = true;
    h.handlers.modal = options => { if (changeIdentity) h.key('identity-b'); else visible = false; options.success({confirm:true}); };
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Private',isCurrent:() => visible}),/页面、身份/);
    assert.equal(h.uploads.length,0); assert.equal(h.created.length,0);
  }
});

test('an inactive or throwing current-context callback fails closed before network access',async () => {
  for (const isCurrent of [() => false, () => undefined, () => { throw new Error('bad state'); }]) {
    const h = harness();
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Private',isCurrent}),/页面、身份/);
    assert.equal(h.calls.length,0); assert.equal(h.uploads.length,0);
  }
});

test('the upload adapter receives a live context guard for the existing background-read barrier',async () => {
  const h = harness(); let visible = true;
  h.handlers.upload = async (_path,_form,_file,options) => {
    assert.equal(options.isCurrent(),true); visible = false; assert.equal(options.isCurrent(),false);
    throw new Error('Existing upload barrier cancelled');
  };
  await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Private',isCurrent:() => visible}),/barrier cancelled/);
  assert.deepEqual(h.revoked,['blob:local-test-0']);
});

test('stale completed responses are discarded and their temporary URLs are released',async () => {
  const h = harness(), wait = deferred(); h.handlers.upload = () => wait.promise;
  const operation = h.service.uploadText(SID,{title:'资料',text:'Private'});
  await tick(); assert.equal(h.uploads.length,1); h.key('identity-b');
  wait.resolve({document:doc()}); await assert.rejects(operation,/页面、身份/);
  assert.deepEqual(h.revoked,['blob:local-test-0']);
});

test('upload failure frees its object URL and never automatically retries',async () => {
  const h = harness(); h.handlers.upload = async () => { throw new Error('Network interrupted'); };
  await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Source'}),/Network interrupted/);
  assert.equal(h.uploads.length,1); assert.deepEqual(h.revoked,['blob:local-test-0']);
});

test('malformed and foreign-session upload responses are rejected and temporary URLs are released',async () => {
  for (const result of [{}, {document:{...doc(),status:'fake-ready'}}, {document:{...doc(),session_id:'#00002'}}]) {
    const h = harness(); h.handlers.upload = async () => result;
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Source'}),/格式不符合/);
    assert.deepEqual(h.revoked,['blob:local-test-0']);
  }
});

test('a modal failure cannot send text',async () => {
  const h = harness(); h.handlers.modal = options => options.fail();
  await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Source'}),/未能确认/);
  assert.equal(h.uploads.length,0); assert.equal(h.created.length,0);
});

test('browsers without File or temporary URL support reject without requests',async () => {
  for (const missing of ['File','URL','createObjectURL','revokeObjectURL']) {
    const h = harness();
    if (missing === 'File' || missing === 'URL') delete h.context[missing]; else delete h.context.URL[missing];
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Source'}),/当前浏览器/);
    assert.equal(h.calls.length,0); assert.equal(h.uploads.length,0);
  }
});

test('native File construction or URL creation failure cannot produce a fake successful result',async () => {
  const fileFail = harness(); fileFail.context.File = class { constructor() { throw new Error('unavailable'); } };
  await assert.rejects(fileFail.service.uploadText(SID,{title:'资料',text:'Source'}),/无法创建文本资料/);
  assert.equal(fileFail.uploads.length,0); assert.equal(fileFail.dialogs.length,0);
  const urlFail = harness(); urlFail.context.URL.createObjectURL = () => { throw new Error('URL unavailable'); };
  await assert.rejects(urlFail.service.uploadText(SID,{title:'资料',text:'Source'}),/URL unavailable/);
  assert.equal(urlFail.uploads.length,0);
});

test('a URL cleanup failure never repeats the already-completed upload',async () => {
  const h = harness(); h.context.URL.revokeObjectURL = () => { throw new Error('cleanup failed'); };
  const result = await h.service.uploadText(SID,{title:'资料',text:'Source'});
  assert.equal(result.document.id,ID); assert.equal(h.uploads.length,1);
});

test('WeChat and App builds explicitly reject pasted-text upload before any request or demo behavior',async () => {
  for (const platform of ['MP-WEIXIN','APP-PLUS']) {
    const h = harness(platform);
    await assert.rejects(h.service.uploadText(SID,{title:'资料',text:'Source'}),/仅支持在浏览器/);
    assert.equal(h.calls.length,0); assert.equal(h.uploads.length,0); assert.equal(h.dialogs.length,0); assert.equal(h.created.length,0);
  }
});
