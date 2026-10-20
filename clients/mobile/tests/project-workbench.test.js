import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {ref,computed,watch,effectScope,nextTick} from 'vue';
import {createWorkbenchClient,taskRecipes,recipeDraft,briefDraft,requestReceipt} from '../src/services/project-workbench.js';
import {isFeedbackAnswerId} from '../src/services/api.js';
const id='b6fe2407-a253-445d-ac69-c69271138aa7',sid='#00001';
const brief=(revision=0)=>({session_id:sid,goal:'A goal',output_format:'A memo',constraints:'Do not guess',revision,updated_at:null});
const artifact=()=>({answer_id:id,title:'项目成果',available:true,answer:{id,role:'assistant',content:'A reviewable answer',research_core:true,sources:[],citation_validation:{warnings:[]}}});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
const source=readFileSync(new URL('../src/components/ProjectWorkbench.vue',import.meta.url),'utf8');
function service() {
  let key='owner-a',preferences={mode:'connected'},active=0,maximum=0;const calls=[];
  const handlers={request:async(path,data)=>path==='/capabilities'?{workbench:{brief:true,artifacts:true}}:path.startsWith('/workbench/brief')?{brief:brief(data?data.expected_revision+1:0)}:path==='/workbench/artifacts/delete'?{removed:true}:path==='/workbench/artifacts'||path.startsWith('/workbench/artifact?')?{artifact:artifact()}:{items:[artifact()]}};
  const client=createWorkbenchClient({identity:()=>key,preferences:()=>preferences,request:async(...args)=>{calls.push(args);active++;maximum=Math.max(active,maximum);try{return await handlers.request(...args);}finally{active--;}}});
  return {client,calls,handlers,key:value=>{key=value;},preferences:value=>{preferences=value;},maximum:()=>maximum};
}
test('live workbench artifacts retain ten research sources without expanding the ordinary five-fragment contract',async()=>{
  const h=service();await h.client.load(sid);
  const sources=Array.from({length:11},(_,index)=>({id:`fb27e5ae-5cb9-4650-9dfd-${String(index+1).padStart(12,'0')}`,label:`S${index+1}`,content:`Original ${index+1}`}));
  h.handlers.request=async()=>({artifact:{...artifact(),answer:{...artifact().answer,sources}}});
  const value=await h.client.artifact(sid,id);
  assert.equal(value.answer.sources.length,10);assert.equal(value.answer.sources[9].label,'S10');
  h.handlers.request=async()=>({artifact:{...artifact(),answer:{...artifact().answer,research_core:false,sources}}});
  assert.equal((await h.client.artifact(sid,id)).answer.sources.length,5);
});
test('recipes are deterministic draft-only prompts with ordinary task names, not reasoning modes',()=>{
  assert.deepEqual(taskRecipes.map(item=>item.title),['对比资料','带依据写作','继续项目','复核结论']);
  assert.equal(recipeDraft('constructor'),'');assert.equal(recipeDraft('https://foreign'),'');
  for(const item of taskRecipes)assert.equal(recipeDraft(item.id),item.prompt);
  assert.match(briefDraft(brief()),/由我提供.*不是资料证据/);assert.equal(briefDraft({goal:' ',constraints:''}),'');
});
test('receipts distinguish transport waiting from server state without invented stage progress',()=>{
  assert.match(requestReceipt('running').title,/服务确认/);assert.match(requestReceipt('running',{asking:true}).title,/等待服务/);
  assert.match(requestReceipt(null,{preparing:true}).detail,/没有宣称模型已开始/);assert.match(requestReceipt('completed').detail,/不代表结论已核验/);
  assert.match(requestReceipt('not_found').detail,/不等于模型未执行/);assert.equal(requestReceipt('retrieving'),null);
});
test('opening loads authenticated capability, brief and live artifacts sequentially with encoded scope',async()=>{
  const h=service();assert.equal(h.calls.length,0);const result=await h.client.load(sid);
  assert.equal(result.enabled,true);assert.equal(result.brief.revision,0);assert.equal('answer' in result.items[0],false);
  assert.deepEqual(h.calls.map(call=>call[0]),['/capabilities','/workbench/brief?session_id=%2300001','/workbench/artifacts?session_id=%2300001']);assert.equal(h.maximum(),1);
});
test('disabled capabilities never read project data and cannot write; demo does not access network',async()=>{
  const h=service();h.handlers.request=async()=>({workbench:{brief:false,artifacts:false}});assert.equal((await h.client.load(sid)).enabled,false);assert.equal(h.calls.length,1);
  await assert.rejects(h.client.saveBrief(sid,brief()),/未开放/);await assert.rejects(h.client.saveArtifact(sid,id,'Title'),/未开放/);assert.equal(h.calls.length,1);
  h.preferences({mode:'demo'});assert.equal((await h.client.load(sid)).enabled,false);assert.equal(h.calls.length,1);
});
test('CAS brief writes preserve all explicit fields and never attach them to a model request',async()=>{
  const h=service();await h.client.load(sid);await h.client.saveBrief(sid,brief());
  assert.deepEqual(h.calls.at(-1),['/workbench/brief',{session_id:sid,expected_revision:0,goal:'A goal',output_format:'A memo',constraints:'Do not guess'},'POST']);
  await assert.rejects(h.client.saveBrief(sid,{...brief(),goal:'x'.repeat(1201)}),/超出范围/);assert.equal(h.calls.length,4);
  h.handlers.request=async()=>{throw Object.assign(new Error('Conflict'),{status:409});};await assert.rejects(h.client.saveBrief(sid,brief()),error=>error.status===409);
});
test('artifact writes contain references only, unavailable responses cannot expose stale snapshots',async()=>{
  const h=service();await h.client.load(sid);await h.client.saveArtifact(sid,id,'项目成果');
  assert.deepEqual(h.calls.at(-1),['/workbench/artifacts',{session_id:sid,answer_id:id,title:'项目成果'},'POST']);
  h.handlers.request=async(path)=>path==='/capabilities'?{workbench:{brief:false,artifacts:true}}:{items:[{...artifact(),available:false,answer:{...artifact().answer,content:'STALE PRIVATE CONTENT'}}]};
  assert.equal('answer' in (await h.client.load(sid)).items[0],false);
  h.handlers.request=async()=>({artifact:{...artifact(),available:false,answer:{content:'STALE PRIVATE CONTENT'}}});assert.equal((await h.client.artifact(sid,id)).answer,null);
});
test('changing identity or project while capabilities return cannot read a later private endpoint',async()=>{
  const h=service(),wait=deferred();h.handlers.request=()=>wait.promise;const pending=h.client.load(sid);h.key('owner-b');wait.resolve({workbench:{brief:true,artifacts:true}});
  await assert.rejects(pending,/身份或项目/);assert.equal(h.calls.length,1);
  const second=service();let current=true;second.handlers.request=async()=>{current=false;return {workbench:{brief:true}};};await assert.rejects(second.client.load(sid,{isCurrent:()=>current}),/身份或项目/);assert.equal(second.calls.length,1);
});
test('source revocation metadata survives hydration and malformed references fail closed',async()=>{
  const h=service();await h.client.load(sid);h.handlers.request=async()=>({artifact:{...artifact(),answer:{...artifact().answer,content:'[Source removed]',citation_validation:{warnings:[{code:'source_revoked'}]}}}});
  assert.equal((await h.client.artifact(sid,id)).answer.citation_validation.warnings[0].code,'source_revoked');
  h.handlers.request=async()=>({artifact:{...artifact(),answer:{...artifact().answer,id:'foreign'}}});await assert.rejects(h.client.artifact(sid,id),/不符合/);
});

function mount(overrides={}) {
  const {descriptor}=parse(source),script=compileScript(descriptor,{id:'workbench'});
  const props=ref({sessionId:sid,identity:'owner-a',demo:false,disabled:false,answers:[{id,role:'assistant',content:'Never use this as title'}]}).value;
  const hooks={},events=[],calls=[],dialogs=[];let key='owner-a',confirmation=true;
  const client={load:async()=>({enabled:true,artifactsEnabled:true,brief:brief(),items:[{answer_id:id,title:'项目成果',available:true}]}),artifact:async()=>artifact(),saveBrief:async(_session,value)=>({...value,revision:value.revision+1}),saveArtifact:async()=>artifact(),removeArtifact:async()=>true,...overrides};
  for(const name of Object.keys(client)){const original=client[name];client[name]=async(...args)=>{calls.push([name,...args]);return original(...args);};}
  const scope={ref,computed,watch,taskRecipes,recipeDraft,briefDraft,createWorkbenchClient:()=>client,isFeedbackAnswerId,researchConnectionKey:()=>key,AnswerBody:{},AnswerEvidence:{},AnswerDeliverable:{},ReasoningEvidence:{},onHide:fn=>{hooks.hide=fn;},onUnload:fn=>{hooks.unload=fn;},uni:{showModal:options=>{dialogs.push(options);options.success({confirm:confirmation});}}};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope)),effects=effectScope();
  const ui=effects.run(()=>component.setup(props,{expose(){},emit:(...args)=>events.push(args)}));
  return {ui,props,hooks,events,calls,dialogs,key:value=>{key=value;},confirm:value=>{confirmation=value;},dispose:()=>{hooks.hide?.();effects.stop();}};
}
test('workbench starts collapsed and makes no reads, writes, or model calls before user action',async()=>{
  const f=mount();assert.equal(f.ui.opened.value,false);assert.equal(f.calls.length,0);f.ui.chooseRecipe('compare');assert.deepEqual(f.events,[['draft',recipeDraft('compare')]]);assert.equal(f.calls.length,0);
  f.props.disabled=true;f.ui.chooseRecipe('write');await f.ui.toggle();assert.equal(f.calls.length,0);f.dispose();
});
test('collapse and reopen preserves unsaved brief; session and identity changes clear private values',async()=>{
  const f=mount();await f.ui.toggle();f.ui.editor.value.goal='Private unsaved goal';await f.ui.toggle();assert.equal(f.ui.opened.value,false);await f.ui.toggle();assert.equal(f.ui.editor.value.goal,'Private unsaved goal');assert.equal(f.calls.length,1);
  f.props.sessionId='#00002';await nextTick();assert.equal(f.ui.editor.value.goal,'');assert.equal(f.ui.artifacts.value.length,0);assert.equal(f.ui.opened.value,false);f.dispose();
});
test('tab navigation retains an unsaved brief only in the same project and rechecks capability before reopening',async()=>{
  const f=mount();await f.ui.toggle();f.ui.editor.value.goal='My unsaved project goal';await nextTick();f.hooks.hide();
  assert.equal(f.ui.opened.value,false);assert.equal(f.ui.enabled.value,false);assert.equal(f.ui.artifacts.value.length,0);assert.equal(f.ui.editor.value.goal,'My unsaved project goal');
  f.props.sessionId='';await nextTick();f.props.sessionId=sid;await nextTick();await f.ui.toggle();
  assert.equal(f.calls.filter(call=>call[0]==='load').length,2);assert.equal(f.ui.editor.value.goal,'My unsaved project goal');assert.equal(f.ui.briefDirty.value,true);assert.equal(f.ui.conflict.value,false);
  f.hooks.hide();f.props.identity='other';f.key('other');await nextTick();assert.equal(f.ui.editor.value.goal,'');assert.equal(f.ui.briefDirty.value,false);f.dispose();
});
test('restoring an unsaved brief never overwrites a remotely revised brief or bypasses the conflict check',async()=>{
  let revision=0;const f=mount({load:async()=>({enabled:true,artifactsEnabled:true,brief:brief(revision),items:[]})});
  await f.ui.toggle();f.ui.editor.value.goal='Local changes';f.hooks.hide();revision=1;await f.ui.toggle();
  assert.equal(f.ui.editor.value.goal,'Local changes');assert.equal(f.ui.conflict.value,true);await f.ui.saveBrief();assert.equal(f.calls.some(call=>call[0]==='saveBrief'),false);
  f.hooks.unload();assert.equal(f.ui.editor.value.goal,'');f.dispose();
});
test('saving and including brief are separate actions and no recipe overwrites parent drafts itself',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.saveBrief();assert.equal(f.ui.editor.value.revision,1);assert.equal(f.events.some(event=>event[0]==='draft'),false);
  f.ui.includeBrief();assert.match(f.events.at(-1)[1],/由我提供/);assert.deepEqual(f.calls.map(call=>call[0]),['load','saveBrief']);f.dispose();
});
test('editing a saved brief shows an unsaved marker and clears only the previous brief save notice',async()=>{
  const f=mount();await f.ui.toggle();assert.equal(f.ui.briefDirty.value,false);await f.ui.saveBrief();await nextTick();assert.match(f.ui.notice.value,/简报已保存/);
  f.ui.editor.value.goal='A newer unsaved goal';await nextTick();assert.equal(f.ui.briefDirty.value,true);assert.equal(f.ui.notice.value,'');assert.match(source,/v-if="briefDirty"[^>]*>有未保存的简报编辑/);
  await f.ui.toggle();await f.ui.toggle();assert.equal(f.ui.briefDirty.value,true);assert.equal(f.ui.editor.value.goal,'A newer unsaved goal');
  await f.ui.saveBrief();await nextTick();assert.equal(f.ui.briefDirty.value,false);assert.match(f.ui.notice.value,/简报已保存/);
  f.ui.notice.value='已收藏回答编号';f.ui.editor.value.goal='Another edit';await nextTick();assert.equal(f.ui.briefDirty.value,true);assert.equal(f.ui.notice.value,'已收藏回答编号');f.dispose();
});
test('CAS conflict retains complete draft and blocks further save until explicit confirmed reload',async()=>{
  const f=mount({saveBrief:async()=>{throw Object.assign(new Error('Conflict'),{status:409});}});await f.ui.toggle();f.ui.editor.value.goal='Keep my full edit';await f.ui.saveBrief();assert.equal(f.ui.conflict.value,true);assert.equal(f.ui.editor.value.goal,'Keep my full edit');await f.ui.saveBrief();assert.equal(f.calls.filter(call=>call[0]==='saveBrief').length,1);
  f.confirm(false);await f.ui.refresh();assert.equal(f.ui.editor.value.goal,'Keep my full edit');assert.equal(f.calls.filter(call=>call[0]==='load').length,1);
  f.confirm(true);await f.ui.refresh();assert.equal(f.ui.conflict.value,false);assert.equal(f.ui.editor.value.goal,'A goal');f.dispose();
});
test('disabled server features do not permit synthetic writes',async()=>{
  const f=mount({load:async()=>({enabled:false,artifactsEnabled:false,brief:null,items:[]})});await f.ui.toggle();await f.ui.saveBrief();await f.ui.saveArtifact();f.ui.includeBrief();assert.deepEqual(f.calls.map(call=>call[0]),['load']);assert.equal(f.events.some(event=>event[0]==='draft'),false);f.dispose();
});
test('artifact defaults to a neutral title and each review obtains fresh hydrated data',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.saveArtifact();assert.equal(f.calls.at(-1)[0],'saveArtifact');assert.equal(f.calls.at(-1)[3],'项目成果');assert.equal('answer' in f.ui.artifacts.value[0],false);await f.ui.openArtifact(id);assert.equal(f.calls.at(-1)[0],'artifact');assert.equal(f.ui.selectedArtifact.value.answer.content,'A reviewable answer');f.hooks.hide();assert.equal(f.ui.selectedArtifact.value,null);assert.equal(f.ui.artifacts.value.length,0);f.dispose();
});
test('private responses after hide or identity change cannot reappear; busy operations serialize',async()=>{
  const wait=deferred(),f=mount({load:()=>wait.promise});const pending=f.ui.toggle();assert.equal(f.ui.busy.value,true);await f.ui.toggle();await f.ui.saveBrief();assert.equal(f.calls.length,1);f.hooks.hide();wait.resolve({enabled:true,artifactsEnabled:true,brief:brief(),items:[artifact()]});await pending;assert.equal(f.ui.editor.value.goal,'');assert.equal(f.ui.artifacts.value.length,0);assert.equal(f.ui.busy.value,false);assert.deepEqual(f.events.filter(event=>event[0]==='busy'),[['busy',true],['busy',false]]);f.dispose();
});
test('deletion confirmation cannot mutate hidden project and cancelled deletion preserves artifact',async()=>{
  const f=mount();await f.ui.toggle();f.confirm(false);await f.ui.removeArtifact(artifact());assert.equal(f.calls.filter(call=>call[0]==='removeArtifact').length,0);assert.equal(f.ui.artifacts.value.length,1);f.confirm(true);await f.ui.removeArtifact(artifact());assert.equal(f.ui.artifacts.value.length,0);f.dispose();
});
test('component compiles for web, WeChat and native without browser storage or fake stages',()=>{
  const require=createRequire(import.meta.url),{preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
  for(const platform of ['H5','MP_WEIXIN','APP_PLUS']){const context={H5:false,MP_WEIXIN:false,APP_PLUS:false,[platform]:true};const content=preprocess(source,context,{type:'html'}),{descriptor,errors}=parse(content);assert.deepEqual(errors,[]);const script=compileScript(descriptor,{id:platform});assert.deepEqual(compileTemplate({source:descriptor.template.content,id:platform,compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);}
  assert.doesNotMatch(source,/localStorage|setStorage|api\.ask|setInterval|setTimeout/);assert.match(source,/min-height:44px/);
});
