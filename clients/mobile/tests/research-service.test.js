import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';
import {File} from 'node:buffer';
import {knowledgeSkillCatalog} from '../src/services/knowledge-skills.js';
import {validResearchExport} from '../src/services/research-export.js';

const require = createRequire(import.meta.url);
const {preprocess} = require('@dcloudio/uni-cli-shared/lib/preprocess');
const raw = readFileSync(new URL('../src/services/research.js',import.meta.url),'utf8');
const code = preprocess(raw,{H5:true},{type:'js'}).replace(/^import[^;]+;\s*$/gm,'').replaceAll('export ','');
const docId = 'b6fe2407-a253-445d-ac69-c69271138aa7', chunkId = 'c9d9ff40-6ec6-462a-b75f-afdcb1b6d1cd';
const memoryId = 'c1f2dfe9-99d1-4ac2-924c-6e42379d1a19';
const capabilities = {enabled:true,uploads_allowed:true,max_upload_bytes:10485760,supported_upload_types:['pdf','txt','md'],memory_write_enabled:true};
const document = () => ({id:docId,filename:'Research.pdf',status:'ready',chunk_count:3,byte_size:1024});
const memory = () => ({id:memoryId,text:'Only correlation has been observed.',kind:'finding',status:'active',confirmed_by_user:true,revision:1,source_ids:[chunkId]});
const clone = value=>JSON.parse(JSON.stringify(value));
const deferred = () => {let resolve;const promise = new Promise(done=>{resolve = done;});return {promise,resolve};};
function harness() {
  let key = 'identity-a', prefs = {mode:'connected',baseUrl:'https://private.example',connectionKind:'invite'}, confirmation = true;
  const calls = [], uploads = [], dialogs = [], pickers = [];
  const handlers = {
    request:async(path) => path === '/research/capabilities' ? clone(capabilities) : path.startsWith('/research/documents?') ? {documents:[document()]} : path.startsWith('/research/document?') ? {document:{...document(),chunks:[{id:chunkId,content:'Original excerpt',page:2,ordinal:0,filename:'Research.pdf'}]}} : path.startsWith('/research/source?') ? {chunk:{id:chunkId,content:'Original excerpt',page:2,filename:'Research.pdf'}} : path.startsWith('/research/memory?') ? {items:[memory()]} : path === '/research/memory' ? {memory:memory()} : path === '/research/memory/delete' ? {deleted:true} : path === '/research/documents/delete' ? {removed:true} : path.startsWith('/research/export?') ? {markdown:'# Research\nPrivate content'} : {},
    picker:options=>options.success({tempFiles:[Object.assign(new File(['x'.repeat(1024)],'Research.pdf',{type:'application/pdf'}),{path:'blob:local-selected-file'})]}),
    modal:options=>options.success({confirm:confirmation}),
    upload:async()=>({document:{...document(),status:'queued'}}),
  };
  const context = {
    getPreferences:()=>({...prefs}), researchConnectionKey:()=>key,
    researchRequest:async(...args)=>{calls.push(args);return handlers.request(...args);},
    researchStatusRequest:async path=>{calls.push([path,undefined,'GET','background']);return handlers.request(path);},
    researchUpload:async(...args)=>{uploads.push(args);return handlers.upload(...args);},
    ensureWeChatPrivacy:async()=>{},knowledgeSkillCatalog,validResearchExport,setTimeout,clearTimeout,File,
    uni:{chooseFile:options=>{pickers.push(options);handlers.picker(options);},showModal:options=>{dialogs.push(options);handlers.modal(options);}},
  };
  vm.runInNewContext(code+'\nglobalThis.service = research;',context);
  return {service:context.service,calls,uploads,dialogs,pickers,handlers,key:value=>{key=value;},preferences:value=>{prefs={...prefs,...value};},confirm:value=>{confirmation=value;}};
}

test('demo and pending invitation never probe research or select files',async()=>{
  const h = harness();h.preferences({mode:'demo'});assert.equal((await h.service.capabilities()).enabled,false);assert.equal(h.calls.length,0);
  await assert.rejects(h.service.upload('#00001'),/尚未开放/);assert.equal(h.pickers.length,0);assert.equal(h.uploads.length,0);
  h.preferences({mode:'connected',invitePending:true});assert.equal((await h.service.capabilities()).enabled,false);assert.equal(h.calls.length,0);
});
test('capabilities cache is ephemeral, copied, single-flight and renewed across identities',async()=>{
  const h=harness(), wait=deferred();h.handlers.request=()=>wait.promise;
  const first=h.service.capabilities(),second=h.service.capabilities();assert.equal(h.calls.length,1);wait.resolve(clone(capabilities));
  const a=await first,b=await second;a.supported_upload_types.push('exe');assert.deepEqual([...b.supported_upload_types],['pdf','txt','md']);
  assert.deepEqual([...(await h.service.capabilities()).supported_upload_types],['pdf','txt','md']);assert.equal(h.calls.length,1);
  h.key('identity-b');h.handlers.request=async()=>clone(capabilities);await h.service.capabilities();assert.equal(h.calls.length,2);
});
test('knowledge catalog is capability gated, authenticated and stays inert',async()=>{
  const old=harness();assert.deepEqual([...(await old.service.knowledgeSkills())],[]);assert.equal(old.calls.length,1);
  const h=harness();h.handlers.request=async path=>path==='/research/capabilities'?{...capabilities,knowledge_skills:true}:{items:[{id:'mao-workplace',name:'职场方法 · 毛选启发',version:'2',description:'方法查询',instructions:'execute secret'}]};
  assert.equal((await h.service.capabilities()).knowledge_skills,true);const items=await h.service.knowledgeSkills();assert.equal(items[0].id,'mao-workplace');assert.equal(items[0].instructions,undefined);assert.equal(h.calls.at(-1)[0],'/research/knowledge-skills');assert.equal(h.calls.length,2);
});
test('knowledge catalog cannot be inherited from a changed identity',async()=>{
  const h=harness(),wait=deferred();h.handlers.request=async path=>path==='/research/capabilities'?{...capabilities,knowledge_skills:true}:wait.promise;
  await h.service.capabilities();const pending=h.service.knowledgeSkills();await new Promise(resolve=>setImmediate(resolve));h.key('different');wait.resolve({items:[{id:'mao-workplace',name:'职场方法 · 毛选启发',version:'2',description:'x'}]});await assert.rejects(pending,/身份或连接/);
});
test('automatic knowledge catalog retains all builtins through the authenticated service without querying each pack',async()=>{
  const packs=[['mao-workplace','2'],['research-ideation','1'],['literature-synthesis','1'],['evidence-writing','1'],['decision-review','2']];
  const h=harness();h.handlers.request=async path=>path==='/research/capabilities'?{...capabilities,knowledge_skills:true}:{items:packs.map(([id,version])=>({id,version,name:id,description:'只按需查询的方法库',script:'ignored'}))};
  await h.service.capabilities();const result=await h.service.knowledgeSkills();assert.deepEqual([...result].map(item=>item.id),packs.map(([id])=>id));assert.equal(h.calls.length,2);assert.ok([...result].every(item=>!('script' in item)));
});
test('disabled or older endpoints block feature reads; outage is never a fake enabled profile',async()=>{
  const h=harness();h.handlers.request=async()=>({enabled:false});
  await assert.rejects(h.service.documents('#00001'),/尚未开放/);assert.equal(h.calls.length,1);
  await assert.rejects(h.service.saveMemory('#00001',{text:'Do not save'}),/尚未开放/);assert.equal(h.calls.length,1);
  const old=harness();old.handlers.request=async()=>{throw Object.assign(new Error('Not found'),{status:404});};assert.equal((await old.service.capabilities()).enabled,false);
  const down=harness();down.handlers.request=async()=>{throw Object.assign(new Error('Unavailable'),{status:503});};await assert.rejects(down.service.capabilities(),/Unavailable/);
});
test('malformed capabilities and stale identity response fail closed',async()=>{
  for(const value of [{...capabilities,max_upload_bytes:20*1048576},{...capabilities,supported_upload_types:['png']},{...capabilities,max_upload_bytes:'10485760'}]) {
    const h=harness();h.handlers.request=async()=>value;await assert.rejects(h.service.capabilities(),/格式不符合/);assert.equal(h.uploads.length,0);
  }
  const h=harness(),wait=deferred();h.handlers.request=()=>wait.promise;const pending=h.service.capabilities();h.key('identity-b');wait.resolve(clone(capabilities));await assert.rejects(pending,/身份或连接/);
});
test('documents and sources are encoded, scoped and manually requested with valid status and exact identity',async()=>{
  const h=harness();const docs=await h.service.documents('#00001');assert.equal(docs[0].status,'ready');assert.equal(docs[0].size_bytes,1024);
  assert.match(h.calls[1][0],/session_id=%2300001/);assert.equal(h.calls.length,2);
  const detail=await h.service.document('#00001',docId);assert.equal(detail.chunks[0].content,'Original excerpt');
  const source=await h.service.source('#00001',chunkId);assert.equal(source.page,2);assert.equal(h.calls.length,4);
  const before=h.calls.length;await assert.rejects(h.service.source('#00001','../foreign'),/编号/);await assert.rejects(h.service.documents('other-user'),/研究会话/);assert.equal(h.calls.length,before);
  h.handlers.request=async()=>({chunk:{id:docId,content:'Wrong source'}});await assert.rejects(h.service.source('#00001',chunkId),/格式不符合/);
});
test('only explicit document status checks opt in to the background read barrier',async()=>{
  const h=harness();await h.service.documents('#00001');assert.equal(h.calls.at(-1).length,3);
  await h.service.documents('#00001',{background:true});assert.equal(h.calls.at(-1)[3],'background');assert.match(h.calls.at(-1)[0],/session_id=%2300001/);
  await h.service.document('#00001',docId);assert.equal(h.calls.at(-1).length,3);
});
test('unknown statuses and excessive lists are not represented as ready material',async()=>{
  const h=harness();await h.service.capabilities();
  h.handlers.request=async()=>({documents:[{...document(),status:'completed'}]});await assert.rejects(h.service.documents('#00001'),/格式不符合/);
  h.handlers.request=async()=>({documents:Array.from({length:21},document)});await assert.rejects(h.service.documents('#00001'),/格式不符合/);
});
test('file picker and send are explicit and each file needs fresh consent',async()=>{
  const h=harness();await h.service.capabilities();assert.equal(h.pickers.length,0);
  const result=await h.service.upload('#00001');assert.equal(result.document.status,'queued');assert.equal(h.pickers.length,1);assert.equal(h.dialogs.length,1);assert.equal(h.uploads.length,1);
  assert.deepEqual({...h.uploads[0][1]},{session_id:'#00001',original_filename:'Research.pdf'});assert.match(h.dialogs[0].content,/解析后的片段可能.*发送给模型/);
  h.confirm(false);assert.equal((await h.service.upload('#00001')).cancelled,true);assert.equal(h.uploads.length,1);
});
test('actual uni H5 native File with added path and wrapped File both preserve selected bytes for invitation FormData',async()=>{
  for(const wrapped of [false,true]) {
    const h=harness();const file=Object.assign(new File(['Selected original bytes'],'Research.txt',{type:'text/plain'}),{path:'blob:local-selected-file'});
    h.handlers.picker=options=>options.success({tempFiles:[wrapped ? {name:file.name,size:file.size,path:file.path,file} : file]});
    await h.service.upload('#00001');assert.equal(h.uploads.length,1);assert.equal(h.uploads[0][0],file.path);assert.equal(h.uploads[0][2],file);
    assert.equal(await h.uploads[0][2].text(),'Selected original bytes');assert.equal(h.dialogs.length,1);
  }
});
test('enabled reading profile never implies upload permission or reparsing rights',async()=>{
  const h=harness();h.handlers.request=async()=>({...capabilities,uploads_allowed:false});
  await assert.rejects(h.service.upload('#00001'),/未开放资料上传/);await assert.rejects(h.service.retryDocument('#00001',docId),/未开放资料处理/);
  assert.equal(h.calls.length,1);assert.equal(h.pickers.length,0);assert.equal(h.uploads.length,0);
});
test('oversized, empty and unsupported files reject before upload or consent',async()=>{
  for(const file of [{name:'Large.pdf',size:10485761},{name:'Empty.md',size:0},{name:'Document.docx',size:1000},{name:'Unknown.pdf'},{name:'a'.repeat(117)+'.pdf',size:100},{name:' File.pdf',size:100},{name:'Bad\0Name.txt',size:100}]) {
    const h=harness();h.handlers.picker=options=>options.success({tempFiles:[{...file,path:'blob:local-file'}]});await assert.rejects(h.service.upload('#00001'));
    assert.equal(h.uploads.length,0);assert.equal(h.dialogs.length,0);
  }
});
test('identity or page changes while selecting/confirming cannot transmit the file',async()=>{
  const h=harness();let selected;h.handlers.picker=options=>{selected=options;};const pending=h.service.upload('#00001');await new Promise(resolve=>setImmediate(resolve));
  h.key('identity-b');selected.success({tempFiles:[{name:'Private.pdf',size:100,path:'blob:local-file'}]});await assert.rejects(pending,/页面、身份或服务/);assert.equal(h.uploads.length,0);
  const next=harness();let current=true;next.handlers.modal=options=>{current=false;options.success({confirm:true});};await assert.rejects(next.service.upload('#00001',{isCurrent:()=>current}),/页面、身份或服务/);assert.equal(next.uploads.length,0);
});
test('confirmed memory distinguishes user confirmation, source loss and optimistic revisions',async()=>{
  const h=harness();const items=await h.service.memory('#00001');assert.equal(items[0].confirmed_by_user,true);
  await h.service.saveMemory('#00001',{text:'Updated conclusion',kind:'finding',source_ids:[chunkId],memory_id:memoryId,expected_revision:1});
  const sent=h.calls.at(-1);assert.equal(sent[2],'POST');assert.equal(sent[1].expected_revision,1);assert.equal(sent[1].memory_id,memoryId);assert.equal(sent[1].text,'Updated conclusion');
  h.handlers.request=async()=>({items:[{...memory(),status:'invalidated'}]});assert.equal((await h.service.memory('#00001'))[0].status,'invalidated');
  h.handlers.request=async()=>({items:[{...memory(),confirmed_by_user:false}]});await assert.rejects(h.service.memory('#00001'),/格式不符合/);
});
test('temporal memory search, cursor and version pages stay scoped, bounded and read only',async()=>{
  const h=harness();await h.service.capabilities();h.handlers.request=async()=>({items:[{...memory(),valid_from:10,valid_until:20,recorded_at:11,archived:true}],next_cursor:'opaque-cursor',total:321});
  const page=await h.service.memoryPage('#00001',{query:'old decision',as_of:15,include_archived:true});assert.equal(page.total,321);assert.equal(page.next_cursor,'opaque-cursor');assert.equal(page.items[0].valid_from,10);assert.equal(page.items[0].archived,true);
  assert.match(h.calls.at(-1)[0],/session_id=%2300001.*query=old%20decision.*as_of=15.*include_archived=true/);assert.equal(h.calls.at(-1)[2],'GET');
  await h.service.memoryPage('#00001',{cursor:page.next_cursor});assert.match(h.calls.at(-1)[0],/cursor=opaque-cursor/);
  const history=await h.service.memoryHistory('#00001',memoryId);assert.equal(history.items[0].recorded_at,11);assert.match(h.calls.at(-1)[0],/memory\/history\?session_id=%2300001&memory_id=/);
  const before=h.calls.length;for(const value of [{as_of:Infinity},{as_of:'10'},{cursor:''},{query:'x'.repeat(501)},{limit:101}]) await assert.rejects(h.service.memoryPage('#00001',value));assert.equal(h.calls.length,before);
  h.handlers.request=async()=>({items:[memory()],next_cursor:{raw:'bad'},total:321});await assert.rejects(h.service.memoryPage('#00001'),/格式不符合/);
});
test('validity and reversible archival require exact user-selected revision and preserve zero epoch',async()=>{
  const h=harness();await h.service.capabilities();
  await h.service.saveMemory('#00001',{text:'Historic preference',valid_from:0,valid_until:86400});assert.equal(h.calls.at(-1)[1].valid_from,0);assert.equal(h.calls.at(-1)[1].valid_until,86400);
  const before=h.calls.length;await assert.rejects(h.service.saveMemory('#00001',{text:'Impossible interval',valid_from:20,valid_until:10}));assert.equal(h.calls.length,before);
  h.handlers.request=async()=>({memory:{...memory(),archived:true,revision:2}});const result=await h.service.archiveMemory('#00001',memoryId,1,true);assert.equal(result.archived,true);assert.deepEqual({...h.calls.at(-1)[1]},{session_id:'#00001',memory_id:memoryId,expected_revision:1,archived:true});assert.equal(h.calls.at(-1)[2],'POST');
});
test('version history redactions never expose removed text or source ids while archived rows remain readable',async()=>{
  const h=harness();await h.service.capabilities();h.handlers.request=async()=>({items:[{...memory(),status:'deleted',redacted:true,text:'stale text must be discarded'}],next_cursor:null,total:1});
  const history=await h.service.memoryHistory('#00001',memoryId);assert.equal(history.items[0].redacted,true);assert.equal(history.items[0].text,'');assert.deepEqual([...history.items[0].source_ids],[]);
  h.handlers.request=async()=>({items:[{...memory(),status:'archived',archived:true}],next_cursor:null,total:1});assert.equal((await h.service.memoryPage('#00001',{include_archived:true})).items[0].archived,true);
  h.handlers.request=async()=>({items:[{...memory(),status:'superseded'}],next_cursor:null,total:1});assert.equal((await h.service.memoryHistory('#00001',memoryId)).items[0].status,'superseded');
  h.handlers.request=async()=>({items:[{...memory(),status:'invalidated',redacted:true,text:''}],next_cursor:null,total:1});assert.equal((await h.service.memoryPage('#00001')).items[0].redacted,true);
});
test('memory invalid input and disabled writing never cause a write request',async()=>{
  const h=harness();await h.service.capabilities();const before=h.calls.length;
  for(const value of [{text:'a'.repeat(2001)},{text:'',kind:'finding'},{text:'\ud800'},{text:'Hello\0World'},{text:'Hello',kind:'guessed_fact'},{text:'Hello',source_ids:['../foreign']},{text:'Hello',memory_id:memoryId}]) await assert.rejects(h.service.saveMemory('#00001',value));
  await assert.rejects(h.service.deleteMemory('#00001',memoryId,0));assert.equal(h.calls.length,before);
  const noWrite=harness();noWrite.handlers.request=async()=>({...capabilities,memory_write_enabled:false});await assert.rejects(noWrite.service.saveMemory('#00001',{text:'Not allowed'}),/未开放.*写入/);assert.equal(noWrite.calls.length,1);
});
test('answer-linked memory draft remains blank, one-shot, private and project scoped without any POST',async()=>{
  const h=harness();await h.service.capabilities();h.service.stageMemoryDraft({session_id:'#00001',source_ids:[chunkId]});
  assert.equal(h.calls.length,1);assert.equal(h.service.consumeMemoryDraft('#00002'),null);
  const draft=h.service.consumeMemoryDraft('#00001');assert.equal(draft.text,'');assert.deepEqual([...draft.source_ids],[chunkId]);assert.equal(draft.kind,'finding');assert.equal(h.service.consumeMemoryDraft('#00001'),null);
  h.service.stageMemoryDraft({session_id:'#00001',text:'Private editable draft'});h.key('identity-b');assert.equal(h.service.consumeMemoryDraft('#00001'),null);
  assert.throws(()=>h.service.stageMemoryDraft({session_id:'#00001',text:'a'.repeat(2001)}),/草稿不符合/);
});
test('deletion/export must confirm their response contract and never retry the operation',async()=>{
  const h=harness();await h.service.deleteDocument('#00001',docId);await h.service.deleteMemory('#00001',memoryId,1);assert.equal(await h.service.export('#00001'),'# Research\nPrivate content');
  assert.equal(h.calls.filter(call=>call[2]==='POST').length,2);
  h.handlers.request=async()=>({removed:false});await assert.rejects(h.service.deleteDocument('#00001',docId),/格式不符合/);
  assert.equal(h.calls.filter(call=>call[0]==='/research/documents/delete').length,2);
});
test('project export accepts all server-bounded ASCII bytes and rejects oversized UTF-8 without truncation',async()=>{
  const h=harness();await h.service.capabilities();const markdown='x'.repeat(800000);
  h.handlers.request=async()=>({markdown});assert.equal(await h.service.export('#00001'),markdown);
  h.handlers.request=async()=>({markdown:'🧠'.repeat(200001)});await assert.rejects(h.service.export('#00001'),/格式/);
});
test('failed document retry is exactly one explicit POST and returned document identity must match',async()=>{
  const h=harness();await h.service.capabilities();h.handlers.request=async()=>({document:{...document(),status:'queued'}});
  assert.equal((await h.service.retryDocument('#00001',docId)).status,'queued');assert.equal(h.calls.length,2);assert.equal(h.calls[1][2],'POST');assert.equal(h.calls[1][0],'/research/documents/retry');
  h.handlers.request=async()=>({document:{...document(),id:chunkId,status:'queued'}});await assert.rejects(h.service.retryDocument('#00001',docId),/格式不符合/);assert.equal(h.calls.length,3);
});
