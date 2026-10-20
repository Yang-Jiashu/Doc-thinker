import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {ref,computed,effectScope} from 'vue';
import {researchExportParts,downloadResearchMarkdown} from '../src/services/research-export.js';

const filename='pages/documents/index.vue';
const source=readFileSync(new URL('../src/'+filename,import.meta.url),'utf8');
const {descriptor}=parse(source,{filename});
const script=compileScript(descriptor,{id:'documents-workflow'});
const template=compileTemplate({source:descriptor.template.content,filename,id:'documents-workflow',compilerOptions:{bindingMetadata:script.bindings}});
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};
const document=(id='one',status='queued')=>({id,filename:`${id}.pdf`,status,chunk_count:0,chunks:[],source_ids:[]});

function mount(overrides={}) {
  let active='#00001',now=0,timerId=0,identity=0;
  const timers=new Map(),hooks={},calls={documents:[],upload:[],retry:[],remove:[],scroll:[],ask:0},effects=effectScope();
  const research={
    capabilities:async()=>({enabled:true,uploads_allowed:true}),
    documents:async sid=>{calls.documents.push(sid);return [document()];},
    upload:async(...args)=>{calls.upload.push(args);return {document:document('new')};},
    retryDocument:async(...args)=>{calls.retry.push(args);return document('one');},
    deleteDocument:async(...args)=>{calls.remove.push(args);return {removed:true};},
    document:async()=>({...document('one','ready'),chunks:[{id:'source-1',content:'Original material'}]}),
    source:async()=>({id:'source-1',content:'Original material'}),
    export:async()=>'Private project export',...overrides.research,
  };
  const context={ref,computed,research,researchExportParts,downloadResearchMarkdown,
    api:{sessions:async()=>[{id:'#00001'},{id:'#00002'}],ask:()=>{calls.ask++;throw new Error('No model on document page');}},
    researchConnectionKey:()=>String(identity),getPreferences:()=>({mode:'connected',connectionKind:'invite'}),getActiveSession:()=>active,setActiveSession:value=>{active=value;},
    onShow:fn=>{hooks.show=fn;},onHide:fn=>{hooks.hide=fn;},onUnload:fn=>{hooks.unload=fn;},
    setTimeout:(fn,delay)=>{const id=++timerId;timers.set(id,{fn,at:now+delay,delay});return id;},clearTimeout:id=>timers.delete(id),
    Date:{now:()=>now},uni:{showModal:options=>options.success({confirm:true}),pageScrollTo:options=>calls.scroll.push(options),switchTab:()=>{}},
    BrandHeader:{},WorkspaceBar:{},ModeBanner:{},CapabilityBoundary:{},
  };
  const code=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(context),code)(...Object.values(context));
  const ui=effects.run(()=>component.setup({}, {expose(){}}));
  return {ui,hooks,research,calls,timers,changeIdentity:()=>{identity++;},
    async tick() {const entry=[...timers].sort((a,b)=>a[1].at-b[1].at)[0];if (!entry) return;timers.delete(entry[0]);now=entry[1].at;await entry[1].fn();await settle();},
    dispose(){hooks.hide();effects.stop();},
  };
}

test('document workflow compiles and waits for capabilities without a false unavailable warning',async()=>{
  assert.deepEqual(template.errors,[]);
  const wait=deferred(),f=mount({research:{capabilities:()=>wait.promise}});f.hooks.show();
  assert.equal(f.ui.loading.value,true);assert.equal(f.ui.capabilityReady.value,false);
  assert.match(source,/v-if="loading && !capabilityReady"/);assert.match(source,/CapabilityBoundary v-else-if/);
  wait.resolve({enabled:true,uploads_allowed:true});await settle();assert.equal(f.ui.capabilityReady.value,true);f.dispose();
});

test('queued documents get bounded status reads, stop on ready, and never retry processing or ask',async()=>{
  const f=mount();f.hooks.show();await settle();assert.equal(f.ui.pendingCount.value,1);assert.equal(f.ui.trackingState.value,'watching');
  let reads=0;f.research.documents=async sid=>{reads++;assert.equal(sid,'#00001');return [document('one','ready')];};
  await f.tick();assert.equal(reads,1);assert.equal(f.ui.readyCount.value,1);assert.equal(f.timers.size,0);assert.match(f.ui.notice.value,/已准备好/);
  assert.equal(f.calls.ask,0);assert.equal(f.calls.retry.length,0);assert.equal(f.calls.upload.length,0);f.dispose();
});

test('status checking stops after six reads rather than polling indefinitely or inventing completion',async()=>{
  const f=mount();f.hooks.show();await settle();for(let i=0;i<6;i++) await f.tick();
  assert.equal(f.calls.documents.length,7);assert.equal(f.ui.trackingState.value,'paused');assert.equal(f.ui.readyCount.value,0);assert.equal(f.timers.size,0);
  await f.ui.reload();assert.equal(f.ui.trackingState.value,'watching');assert.equal(f.timers.size,1);f.dispose();
});

test('a status network failure pauses reads and preserves the actual pending state without retrying',async()=>{
  const f=mount();f.hooks.show();await settle();let reads=0;f.research.documents=async()=>{reads++;throw new Error('Network failed');};
  await f.tick();assert.equal(reads,1);assert.equal(f.ui.trackingState.value,'error');assert.equal(f.ui.pendingCount.value,1);assert.equal(f.ui.readyCount.value,0);assert.equal(f.timers.size,0);assert.equal(f.calls.retry.length,0);f.dispose();
});

test('a hidden page discards late status responses and removes timers and document contents',async()=>{
  const f=mount();f.hooks.show();await settle();const wait=deferred();f.research.documents=()=>wait.promise;
  const checking=f.tick();f.hooks.hide();wait.resolve([document('private','ready')]);await checking;
  assert.equal(f.ui.files.value.length,0);assert.equal(f.ui.refreshingStatuses.value,false);assert.equal(f.timers.size,0);f.dispose();
});

test('user operations cannot race a pending status read through a single-request gateway',async()=>{
  const f=mount();f.hooks.show();await settle();const wait=deferred();f.research.documents=sid=>sid === '#00001' ? wait.promise : Promise.resolve([document('other','ready')]);
  const checking=f.tick();await f.ui.upload();await f.ui.removeDocument(document());await f.ui.retryDocument(document('one','failed'));await f.ui.previewExport();await f.ui.changeSession('#00002');await f.ui.reload();
  assert.equal(f.ui.sessionId.value,'#00001');assert.equal(f.calls.upload.length,0);assert.equal(f.calls.retry.length,0);assert.equal(f.calls.remove.length,0);
  wait.resolve([document('old','ready')]);await checking;await f.ui.changeSession('#00002');
  assert.equal(f.ui.sessionId.value,'#00002');assert.deepEqual(f.ui.files.value.map(file=>file.id),['other']);assert.equal(f.timers.size,0);f.dispose();
});

test('identity change before a scheduled read clears the private view and makes no new request',async()=>{
  const f=mount();f.hooks.show();await settle();f.changeIdentity();await f.tick();
  assert.equal(f.calls.documents.length,1);assert.equal(f.ui.files.value.length,0);assert.equal(f.ui.sessionId.value,'');assert.equal(f.timers.size,0);assert.match(f.ui.error.value,/身份或连接已改变/);f.dispose();
});

test('identity change during a status read discards the result and clears private contents',async()=>{
  const f=mount();f.hooks.show();await settle();const wait=deferred();f.research.documents=()=>wait.promise;const checking=f.tick();f.changeIdentity();wait.resolve([document('private','ready')]);await checking;
  assert.equal(f.ui.files.value.length,0);assert.equal(f.ui.refreshingStatuses.value,false);assert.equal(f.timers.size,0);assert.match(f.ui.error.value,/身份或连接已改变/);f.dispose();
});

test('status reads defer during a user action and never run alongside a blocked page forever',async()=>{
  const f=mount();f.hooks.show();await settle();f.ui.busyDocument.value=true;
  for(let i=0;i<45 && f.timers.size;i++) await f.tick();
  assert.equal(f.calls.documents.length,1);assert.equal(f.timers.size,0);assert.equal(f.ui.trackingState.value,'paused');f.dispose();
});

test('upload result appears immediately and a returned ready status needs no redundant full reload',async()=>{
  const f=mount({research:{documents:async()=>[],upload:async()=>({document:document('uploaded','ready')})}});f.hooks.show();await settle();await f.ui.upload();
  assert.deepEqual(f.ui.files.value.map(file=>file.id),['uploaded']);assert.equal(f.ui.readyCount.value,1);assert.equal(f.ui.uploading.value,false);assert.match(f.ui.notice.value,/已准备好/);assert.equal(f.timers.size,0);f.dispose();
});

test('an old upload finishing after hide and return cannot unblock the next upload',async()=>{
  const first=deferred(),second=deferred();let uploads=0;
  const f=mount({research:{documents:async()=>[],upload:()=>++uploads === 1 ? first.promise : second.promise}});f.hooks.show();await settle();
  const old=f.ui.upload();f.hooks.hide();f.hooks.show();await settle();const current=f.ui.upload();
  first.resolve({document:document('old','ready')});await old;assert.equal(f.ui.uploading.value,true);assert.equal(f.ui.files.value.length,0);
  second.resolve({document:document('new','ready')});await current;assert.equal(f.ui.uploading.value,false);assert.deepEqual(f.ui.files.value.map(file=>file.id),['new']);f.dispose();
});

test('local name and status filters make no requests and reset across projects',async()=>{
  const f=mount({research:{documents:async()=>[document('Paper','ready'),document('Note','failed'),document('Uncertain','mystery')]}});f.hooks.show();await settle();
  f.ui.fileQuery.value='paper';assert.deepEqual(f.ui.visibleFiles.value.map(file=>file.id),['Paper']);
  f.ui.fileQuery.value='';f.ui.fileFilter.value='attention';assert.equal(f.ui.visibleFiles.value.length,2);
  await f.ui.changeSession('#00002');assert.equal(f.ui.fileFilter.value,'all');assert.equal(f.ui.fileQuery.value,'');assert.equal(f.calls.ask,0);f.dispose();
});

test('source reader navigation is explicit, while hidden page readers never trigger a late scroll',async()=>{
  const f=mount({research:{documents:async()=>[document('one','ready')]}});f.hooks.show();await settle();assert.equal(f.calls.scroll.length,0);
  await f.ui.inspectDocument(document('one','ready'));await f.tick();assert.deepEqual(f.calls.scroll,[{selector:'#document-reader',duration:0}]);
  await f.ui.readSource('source-1');f.hooks.hide();await f.tick();assert.equal(f.calls.scroll.length,1);assert.equal(f.ui.sourceDetail.value,null);f.dispose();
});

test('a background list detecting removal also clears the old reader and export',async()=>{
  const f=mount({research:{documents:async()=>[document('one','ready'),document('pending')]}});f.hooks.show();await settle();
  await f.ui.inspectDocument(document('one','ready'));await f.tick();await f.ui.readSource('source-1');await f.tick();await f.ui.previewExport();
  f.research.documents=async()=>[document('pending','ready')];await f.tick();assert.equal(f.ui.documentDetail.value,null);assert.equal(f.ui.sourceDetail.value,null);assert.equal(f.ui.exportMarkdown.value,'');f.dispose();
});
