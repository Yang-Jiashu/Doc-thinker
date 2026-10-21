import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {ref,computed,effectScope} from 'vue';
import {researchExportParts,downloadResearchMarkdown} from '../src/services/research-export.js';

const root = new URL('../src/',import.meta.url), settle = () => new Promise(resolve=>setImmediate(resolve));
const deferred = () => {let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
const docId='b6fe2407-a253-445d-ac69-c69271138aa7',chunkId='c9d9ff40-6ec6-462a-b75f-afdcb1b6d1cd',memoryId='c1f2dfe9-99d1-4ac2-924c-6e42379d1a19';
const doc = () => ({id:docId,filename:'Research.pdf',status:'ready',chunk_count:2,chunks:[],source_ids:[]});
const item = () => ({id:memoryId,text:'We observed correlation, not causation.',kind:'finding',revision:1,status:'active',source_ids:[chunkId],confirmed_by_user:true});
function mount(page, overrides = {}) {
  const filename=`pages/${page}/index.vue`,source=readFileSync(new URL(filename,root),'utf8');
  const {descriptor,errors}=parse(source,{filename});assert.deepEqual(errors,[]);
  const script=compileScript(descriptor,{id:page});
  const template=compileTemplate({source:descriptor.template.content,filename,id:page,compilerOptions:{bindingMetadata:script.bindings}});assert.deepEqual(template.errors,[]);
  let active='#00001', confirm=true,preferences={mode:'connected',connectionKind:'invite',baseUrl:'https://private.example'};
  const hooks={},calls={capabilities:0,documents:[],document:[],source:[],upload:[],deleteDocument:[],retryDocument:[],memory:[],saveMemory:[],deleteMemory:[],export:[],clipboard:[],navigation:[],modal:[],sessions:0};
  const research={
    capabilities:async()=>{calls.capabilities++;return {enabled:true,uploads_allowed:true,memory_write_enabled:true};},
    documents:async sid=>{calls.documents.push(sid);return [doc()];},
    document:async(...args)=>{calls.document.push(args);return {...doc(),chunks:[{id:chunkId,content:'Exact original source',filename:'Research.pdf',page:3,ordinal:0}]};},
    source:async(...args)=>{calls.source.push(args);return {id:chunkId,content:'Exact original source',filename:'Research.pdf',page:3};},
    upload:async(...args)=>{calls.upload.push(args);return {document:doc()};},
    deleteDocument:async(...args)=>{calls.deleteDocument.push(args);return {removed:true};},
    retryDocument:async(...args)=>{calls.retryDocument.push(args);return {...doc(),status:'queued'};},
    memory:async sid=>{calls.memory.push(sid);return [item()];},
    memoryPage:async(sid,options)=>{const items=await research.memory(sid);return {items,next_cursor:null,total:items.length};},
    memoryHistory:async()=>({items:[item()],next_cursor:null,total:1}),
    archiveMemory:async(_sid,_id,revision,archived)=>({...item(),revision:revision+1,archived}),
    saveMemory:async(...args)=>{calls.saveMemory.push(args);return {...item(),...args[1],revision:args[1].memory_id ? args[1].expected_revision+1 : 1};},
    deleteMemory:async(...args)=>{calls.deleteMemory.push(args);return {deleted:true};},
    export:async(...args)=>{calls.export.push(args);return '# Research\nOnly this project';},
    consumeMemoryDraft:()=>null,...overrides.research,
  };
  const api={sessions:async()=>{calls.sessions++;return [{id:'#00001',title:'Project A'},{id:'#00002',title:'Project B'}];},
    files:()=>{throw new Error('Do not call legacy files');},memory:()=>{throw new Error('Do not call legacy automatic memory');},ask:()=>{throw new Error('No model on research pages');},...overrides.api};
  const uni={switchTab:options=>calls.navigation.push(options),showModal:options=>{calls.modal.push(options);options.success({confirm});},setClipboardData:options=>{calls.clipboard.push(options.data);options.success();},...overrides.uni};
  const scope={ref,computed,api,research,researchExportParts,downloadResearchMarkdown:overrides.downloadResearchMarkdown || downloadResearchMarkdown,researchConnectionKey:()=>JSON.stringify(preferences),getPreferences:()=>({...preferences}),getActiveSession:()=>active,setActiveSession:value=>{active=value;},
    onShow:fn=>{hooks.show=fn;},onHide:fn=>{hooks.hide=fn;},onUnload:fn=>{hooks.unload=fn;},
    BrandHeader:{},ModeBanner:{},WorkspaceBar:{},CapabilityBoundary:{},MemoryObservatory:{},uni};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope)),effects=effectScope();
  const ui=effects.run(()=>component.setup({}, {expose(){}}));
  return {ui,calls,hooks,source,research,confirm:value=>{confirm=value;},active:value=>{active=value;},preferences:value=>{preferences={...preferences,...value};},dispose:()=>{hooks.hide?.();effects.stop();}};
}

test('research docs open with capability/session/list only and ready contents remain scoped',async()=>{
  const f=mount('documents');f.hooks.show();await settle();assert.equal(f.ui.researchEnabled.value,true);assert.deepEqual(f.calls.documents,['#00001']);assert.equal(f.ui.readyCount.value,1);
  assert.equal(f.calls.document.length,0);assert.equal(f.calls.source.length,0);assert.equal(f.calls.upload.length,0);assert.equal(f.calls.export.length,0);assert.equal(f.calls.clipboard.length,0);
  f.ui.openReadyChat();assert.deepEqual(f.calls.navigation,[{url:'/pages/chat/index'}]);f.dispose();
});
test('unopened research profile in invitation has no sessions or document/memory reads',async()=>{
  for(const page of ['documents','memory']) {const f=mount(page,{research:{capabilities:async()=>({enabled:false})}});f.hooks.show();await settle();assert.equal(f.calls.sessions,0);assert.equal(f.calls.documents.length,0);assert.equal(f.calls.memory.length,0);assert.equal(f.ui.loading.value,false);f.dispose();}
});
test('source review is explicit and deletion requires a separate irreversible confirmation',async()=>{
  const f=mount('documents');f.hooks.show();await settle();await f.ui.inspectDocument(f.ui.files.value[0]);assert.equal(f.calls.document.length,1);assert.equal(f.ui.documentDetail.value.chunks[0].content,'Exact original source');assert.equal(f.calls.source.length,0);
  await f.ui.readSource(chunkId);assert.equal(f.calls.source.length,1);assert.equal(f.ui.sourceDetail.value.page,3);
  f.confirm(false);await f.ui.removeDocument(doc());assert.equal(f.calls.deleteDocument.length,0);assert.equal(f.ui.files.value.length,1);
  f.confirm(true);await f.ui.removeDocument(doc());assert.deepEqual(f.calls.deleteDocument,[['#00001',docId]]);assert.equal(f.ui.files.value.length,0);assert.equal(f.ui.sourceDetail.value,null);assert.match(f.calls.modal.at(-1).content,/备份副本不自动删除/);f.dispose();
});
test('failed document retries only on confirmed click and success returns to queued rather than fake ready',async()=>{
  const failed={...doc(),status:'failed'},f=mount('documents',{research:{documents:async()=>[failed]}});f.hooks.show();await settle();assert.equal(f.calls.retryDocument.length,0);assert.equal(f.ui.readyCount.value,0);
  f.confirm(false);await f.ui.retryDocument(failed);assert.equal(f.calls.retryDocument.length,0);f.confirm(true);await f.ui.retryDocument(failed);assert.deepEqual(f.calls.retryDocument,[['#00001',docId]]);assert.equal(f.ui.files.value[0].status,'queued');assert.equal(f.ui.readyCount.value,0);assert.match(f.ui.notice.value,/没有调用模型/);f.dispose();
});
test('hidden page cannot approve a delayed deletion or display another project source',async()=>{
  const wait=deferred();let modal;
  const f=mount('documents',{research:{source:()=>wait.promise},uni:{showModal:options=>{modal=options;}}});f.hooks.show();await settle();const pending=f.ui.readSource(chunkId);f.hooks.hide();wait.resolve({id:chunkId,content:'Private old source'});await pending;assert.equal(f.ui.sourceDetail.value,null);
  f.hooks.show();await settle();const deletion=f.ui.removeDocument(doc());f.hooks.hide();modal.success({confirm:true});await deletion;assert.equal(f.calls.deleteDocument.length,0);f.dispose();
});
test('document upload is user initiated and stale selection cannot proceed under page identity predicate',async()=>{
  const wait=deferred();let current;
  const f=mount('documents',{research:{upload:async(_sid,options)=>{current=options.isCurrent;return wait.promise;}}});f.hooks.show();await settle();const pending=f.ui.upload();assert.equal(current(),true);f.hooks.hide();assert.equal(current(),false);wait.resolve({document:doc()});await pending;assert.equal(f.ui.files.value.length,0);assert.equal(f.ui.uploading.value,false);f.dispose();
});
test('research export is preview first, clipboard explicit, no public link or automatic write',async()=>{
  const f=mount('documents');f.hooks.show();await settle();await f.ui.previewExport();assert.equal(f.calls.export.length,1);assert.equal(f.calls.clipboard.length,0);assert.match(f.ui.exportPreview.value,/Only this project/);
  await f.ui.copyExport();assert.deepEqual(f.calls.clipboard,['# Research\nOnly this project']);assert.match(f.ui.notice.value,/隐私内容/);f.dispose();
});
test('long project export supports deliberate lossless ordered copy rather than disabling delivery',async()=>{
  const markdown='A'.repeat(39999)+'🧠'+'B'.repeat(60000),f=mount('documents',{research:{export:async()=>markdown}});
  f.hooks.show();await settle();await f.ui.previewExport();assert.equal(f.calls.clipboard.length,0);
  assert.equal(f.ui.exportParts.value.join(''),markdown);assert.equal(f.ui.exportParts.value.length,3);
  for(let index=0;index<f.ui.exportParts.value.length;index++) await f.ui.copyExport(index);
  assert.equal(f.calls.clipboard.join(''),markdown);assert.equal(f.ui.exportPartIndex.value,0);
  await f.ui.copyExport(99);assert.equal(f.calls.clipboard.length,3);f.dispose();
});
test('explicit download requires confirmation and reports browser handoff, not verified disk persistence',async()=>{
  const downloads=[],f=mount('documents',{downloadResearchMarkdown:(...args)=>downloads.push(args)});
  f.hooks.show();await settle();await f.ui.previewExport();assert.equal(downloads.length,0);
  f.confirm(false);await f.ui.downloadExport();assert.equal(downloads.length,0);
  f.confirm(true);await f.ui.downloadExport();assert.equal(downloads.length,1);assert.equal(downloads[0][1],'#00001');
  assert.match(f.calls.modal.at(-1).content,/不会自动上传或公开分享/);assert.match(f.ui.notice.value,/已请求浏览器下载/);f.dispose();
});
test('export download/clipboard failures keep complete text and expose selectable lossless pieces',async()=>{
  const markdown='private'.repeat(10000),f=mount('documents',{research:{export:async()=>markdown},downloadResearchMarkdown:()=>{throw new Error('SENSITIVE backend detail');},uni:{setClipboardData:options=>options.fail()}});
  f.hooks.show();await settle();await f.ui.previewExport();await f.ui.downloadExport();
  assert.equal(f.ui.exportMarkdown.value,markdown);assert.equal(f.ui.exportPartsOpen.value,true);assert.doesNotMatch(f.ui.error.value,/SENSITIVE/);
  await f.ui.copyExport(1);assert.match(f.ui.error.value,/手动选择/);assert.equal(f.ui.exportParts.value.join(''),markdown);f.dispose();
});
test('identity changes before copy or while download confirmation is open prevent old data delivery',async()=>{
  let modal;const downloads=[],f=mount('documents',{downloadResearchMarkdown:(...args)=>downloads.push(args),uni:{showModal:options=>{modal=options;}}});
  f.hooks.show();await settle();await f.ui.previewExport();const pending=f.ui.downloadExport();
  f.preferences({baseUrl:'https://different.example'});modal.success({confirm:true});await pending;assert.equal(downloads.length,0);
  await f.ui.copyExport();assert.equal(f.calls.clipboard.length,0);f.dispose();
});
test('refresh after another client removes a document cannot keep its stale source reader or export preview',async()=>{
  const f=mount('documents');f.hooks.show();await settle();await f.ui.inspectDocument(doc());await f.ui.readSource(chunkId);await f.ui.previewExport();
  f.research.documents=async()=>[];await f.ui.reload();assert.equal(f.ui.files.value.length,0);
  assert.equal(f.ui.documentDetail.value,null);assert.equal(f.ui.sourceDetail.value,null);assert.equal(f.ui.exportPreview.value,'');assert.equal(f.ui.exportMarkdown.value,'');f.dispose();
});
test('confirmed memory opens read-only until user submits and confirms editable content',async()=>{
  const f=mount('memory');f.hooks.show();await settle();assert.equal(f.ui.researchEnabled.value,true);assert.equal(f.ui.confirmedItems.value[0].text,item().text);assert.equal(f.calls.saveMemory.length,0);
  f.ui.startMemory();f.ui.editorText.value='The mechanism still needs an intervention test.';assert.equal(f.calls.saveMemory.length,0);
  f.confirm(false);await f.ui.saveMemory();assert.equal(f.calls.saveMemory.length,0);assert.equal(f.ui.editorOpen.value,true);
  f.confirm(true);await f.ui.saveMemory();assert.equal(f.calls.saveMemory.length,1);assert.equal(f.calls.saveMemory[0][0],'#00001');assert.equal(f.calls.saveMemory[0][1].kind,'finding');assert.deepEqual(f.calls.saveMemory[0][1].source_ids,[]);assert.equal(f.ui.editorOpen.value,false);f.dispose();
});
test('memory search covers server project, stable cursor filters and stale reads cannot cross pages',async()=>{
  const reads=[],other={...item(),id:docId,text:'Older archived decision',archived:true};
  const f=mount('memory',{research:{memoryPage:async(sid,options)=>{reads.push({sid,...options});return options.cursor ? {items:[item(),other],next_cursor:null,total:201} : {items:[item()],next_cursor:'cursor-1',total:201};}}});
  f.hooks.show();await settle();assert.equal(f.ui.confirmedTotal.value,201);assert.equal(f.ui.confirmedItems.value.length,1);
  f.ui.confirmedQuery.value='historic energy';await f.ui.searchConfirmed();assert.equal(reads.at(-1).query,'historic energy');
  f.ui.confirmedQuery.value='unsent draft';await f.ui.loadMoreConfirmed();assert.equal(reads.at(-1).query,'historic energy');assert.equal(reads.at(-1).cursor,'cursor-1');assert.equal(f.ui.confirmedItems.value.length,2);assert.equal(f.ui.confirmedCursor.value,null);
  await f.ui.selectConfirmedDate({detail:{value:'2026-01-01'}});assert.equal(reads.at(-1).as_of,new Date('2026-01-01T23:59:59').getTime()/1000);f.ui.editMemory(item());assert.equal(f.ui.editorOpen.value,false);await f.ui.removeMemory(item());assert.equal(f.calls.deleteMemory.length,0);
  const wait=deferred();f.research.memoryPage=()=>wait.promise;const pending=f.ui.searchConfirmed();f.hooks.hide();wait.resolve({items:[item()],next_cursor:null,total:1});await pending;assert.deepEqual(f.ui.confirmedItems.value,[]);f.dispose();
});
test('memory versions and archive stay deliberate, reversible and blocked by stale revision',async()=>{
  const f=mount('memory');f.hooks.show();await settle();assert.equal(f.ui.memoryHistoryId.value,'');await f.ui.openMemoryHistory(item());assert.equal(f.ui.memoryVersions.value.length,1);
  f.confirm(false);await f.ui.archiveMemory(item());assert.equal(f.ui.confirmedItems.value.length,1);f.confirm(true);await f.ui.archiveMemory(item());assert.equal(f.ui.confirmedItems.value.length,0);assert.match(f.ui.notice.value,/已归档/);assert.equal(f.ui.memoryHistoryId.value,'');
  f.research.archiveMemory=async()=>{throw Object.assign(new Error('Conflict'),{status:409});};await f.ui.archiveMemory({...item(),archived:true});assert.match(f.ui.error.value,/已有更新/);assert.equal(f.calls.saveMemory.length,0);f.dispose();
});
test('editing text starts a new effective version now and preserves precise expiry unless explicitly changed',async()=>{
  const f=mount('memory');f.hooks.show();await settle();const original={...item(),valid_from:1735723456.125,valid_until:4102444800.5};f.ui.editMemory(original);assert.equal(f.ui.editorFrom.value,'');f.ui.editorText.value='Only text changed';await f.ui.saveMemory();assert.equal('valid_from' in f.calls.saveMemory[0][1],false);assert.equal(f.calls.saveMemory[0][1].valid_until,original.valid_until);
  f.ui.startMemory();f.ui.editorText.value='Invalid interval';f.ui.editorFrom.value='2026-10-10';f.ui.editorUntil.value='2026-01-01';await f.ui.saveMemory();assert.equal(f.calls.saveMemory.length,1);assert.match(f.ui.error.value,/结束日期/);f.dispose();
});
test('pending capability probe displays loading instead of a false unavailable memory boundary',async()=>{
  const wait=deferred(),f=mount('memory',{research:{capabilities:()=>wait.promise}});f.hooks.show();assert.equal(f.ui.loading.value,true);assert.equal(f.ui.capabilityReady.value,false);assert.match(f.source,/v-if="loading && !capabilityReady"/);assert.match(f.source,/CapabilityBoundary v-else-if/);wait.resolve({enabled:true,memory_write_enabled:true});await settle();assert.equal(f.ui.capabilityReady.value,true);assert.equal(f.ui.researchEnabled.value,true);f.dispose();
});
test('memory update binds expected revision and conflict preserves draft instead of overwriting',async()=>{
  const f=mount('memory',{research:{saveMemory:async()=>{throw Object.assign(new Error('Raw conflict detail'),{status:409});}}});f.hooks.show();await settle();f.ui.editMemory(item());f.ui.editorText.value='Reviewed edit';await f.ui.saveMemory();assert.equal(f.ui.editingRevision.value,1);assert.equal(f.ui.editorText.value,'Reviewed edit');assert.match(f.ui.error.value,/已有更新/);assert.doesNotMatch(f.ui.error.value,/Raw/);f.dispose();
  const g=mount('memory');g.hooks.show();await settle();g.ui.editMemory(item());g.ui.editorText.value='Reviewed edit';await g.ui.saveMemory();assert.equal(g.calls.saveMemory[0][1].expected_revision,1);assert.equal(g.calls.saveMemory[0][1].memory_id,memoryId);assert.equal(g.ui.confirmedItems.value[0].revision,2);g.dispose();
});
test('invalidated sources remain visibly invalidated and only create an unlinked editable draft',async()=>{
  const lost={...item(),status:'invalidated',revision:2};const f=mount('memory',{research:{memory:async()=>[lost]}});f.hooks.show();await settle();f.ui.editMemory(lost);assert.equal(f.ui.editingId.value,'');assert.deepEqual(f.ui.editorSources.value,[]);assert.equal(f.ui.editorText.value,lost.text);assert.equal(f.calls.saveMemory.length,0);assert.match(f.ui.notice.value,/旧记忆仍然失效/);f.dispose();
});
test('memory deletion is explicit and late confirmation cannot mutate a hidden page',async()=>{
  const f=mount('memory');f.hooks.show();await settle();f.confirm(false);await f.ui.removeMemory(item());assert.equal(f.calls.deleteMemory.length,0);f.confirm(true);await f.ui.removeMemory(item());assert.deepEqual(f.calls.deleteMemory,[['#00001',memoryId,1]]);assert.equal(f.ui.confirmedItems.value.length,0);f.dispose();
  let modal;const g=mount('memory',{uni:{showModal:options=>{modal=options;}}});g.hooks.show();await settle();const pending=g.ui.removeMemory(item());g.hooks.hide();modal.success({confirm:true});await pending;assert.equal(g.calls.deleteMemory.length,0);assert.equal(g.ui.confirmedItems.value.length,0);g.dispose();
});
test('answer draft never copies a model conclusion or writes automatically; source review remains manual',async()=>{
  const f=mount('memory',{research:{consumeMemoryDraft:()=>({text:'',kind:'finding',source_ids:[chunkId],origin:'answer_sources'})}});f.hooks.show();await settle();assert.equal(f.ui.editorOpen.value,true);assert.equal(f.ui.editorText.value,'');assert.deepEqual(f.ui.editorSources.value,[chunkId]);assert.equal(f.calls.saveMemory.length,0);assert.equal(f.calls.source.length,0);await f.ui.readConfirmedSource(chunkId);assert.equal(f.calls.source.length,1);f.dispose();
});
test('memory character budget blocks save without truncation and page hide wipes private drafts',async()=>{
  const f=mount('memory');f.hooks.show();await settle();f.ui.startMemory();f.ui.editorText.value='🧠'.repeat(2001);assert.equal(f.ui.canSaveMemory.value,false);await f.ui.saveMemory();assert.equal(f.calls.modal.length,0);assert.equal(f.calls.saveMemory.length,0);assert.equal(Array.from(f.ui.editorText.value).length,2001);f.hooks.hide();assert.equal(f.ui.editorText.value,'');assert.equal(f.ui.confirmedItems.value.length,0);f.dispose();
});
