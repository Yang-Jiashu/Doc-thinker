import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, watch, nextTick, effectScope } from 'vue';
import { createQueryRequestIdAsync, queryRequestIdSupported, questionByteLength, INVITE_MAX_QUESTION_BYTES, researchAnswerMetadata } from '../src/services/api.js';
import {requestReceipt} from '../src/services/project-workbench.js';
import {boundedAnswerSources} from '../src/services/research-source-contract.js';
import {planGraphNavigation, GRAPH_NAVIGATION_LIMITS} from '../src/services/graph-navigation.js';
import {buildEvidenceCheckQuestion} from '../src/services/evidence-check.js';
import {researchExportParts,downloadResearchMarkdown} from '../src/services/research-export.js';

const root = new URL('../src/', import.meta.url);
const settle = () => new Promise(resolve => setImmediate(resolve));
test('long project operations share the chat network gate and phase opening is explicitly delegated',async()=>{
  const f=mount('chat');f.hooks.show();await settle();let opened=null;f.ui.taskPanel.value={openTask:async id=>{opened=id;}};
  f.ui.setLongProjectBusy(true);assert.equal(f.ui.networkBusy.value,true);f.ui.draft.value='Keep this draft';await f.ui.send();assert.equal(f.calls.ask.length,0);await f.ui.openLongProjectTask('phase-task');assert.equal(opened,null);
  f.ui.setLongProjectBusy(false);await f.ui.openLongProjectTask('phase-task');assert.equal(opened,'phase-task');assert.equal(f.ui.draft.value,'Keep this draft');f.dispose();
});
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; }

/** Execute the actual SFC setup with offline API/voice and captured page hooks. */
function mount(page, overrides = {}) {
  const filename = `pages/${page}/index.vue`;
  const { descriptor, errors } = parse(readFileSync(new URL(filename, root), 'utf8'), { filename });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: page });
  const template = compileTemplate({ source: descriptor.template.content, filename, id: page, compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(template.errors, []);
  const hooks = {}, calls = { ask: [], start: 0, stop: 0, cancel: 0, stopSpeaking: 0, save: 0, connectionChecks:0, cancelChecks:0, inviteChecks:0, authentications:[], inviteCancels:0, navigation:[], logouts:[], reloads:[],memoryDrafts:[] };
  let active = 'a';
  let identity = null;
  let connectionRevision = 0;
  let preferences = { mode: 'demo', baseUrl: '', accessToken: '', useMemory: true, useContext: true, useCache: true, useEvolution: false };
  const api = {
    checkConnection: async () => { calls.connectionChecks += 1; return {connected:true}; },
    sessions: async () => [{ id: 'a', title: 'Session A' }, { id: 'b', title: 'Session B' }],
    history: async () => [], createSession: async () => ({ id: 'b', title: 'New' }),
    ask: async (...args) => { calls.ask.push(args); return { answer: 'Offline answer' }; },
    files: async () => [{ filename: 'Demo.pdf', status: 'completed' }], upload: async () => ({ status: 'demo' }),
    graph: async () => ({ nodes: [{ id: 'same-id', label: 'Shared name' }], edges: [] }),
    chunks: async () => [{ content: 'Source excerpt' }], ...overrides.api,
  };
  const scope = {
    ref, computed, watch, nextTick, api, createQueryRequestIdAsync, queryRequestIdSupported, questionByteLength, INVITE_MAX_QUESTION_BYTES, researchAnswerMetadata,requestReceipt,boundedAnswerSources,ProjectWorkbench:{},ProjectTasks:{},LongProjects:{},ComposerJump:{},EvidenceIntake:{},buildEvidenceCheckQuestion,researchExportParts,downloadResearchMarkdown,
    researchConnectionKey:()=>String(connectionRevision),...overrides.requestIds,
    research:{capabilities:async()=>({enabled:false}),stageMemoryDraft:value=>calls.memoryDrafts.push(value),...overrides.research},
    isFeedbackAnswerId: value => typeof value === 'string' && (/^[0-9a-f]{8}-[0-9a-f-]{27}$/.test(value) || /^demo-answer-\d+$/.test(value)),
    getPreferences: () => ({ ...preferences }),
    savePreferences: value => { calls.save += 1; preferences = { ...value }; },
    cancelConnectionCheck: () => { calls.cancelChecks += 1; },
    getActiveSession: () => active, setActiveSession: value => { active = value; },
    onShow: callback => { hooks.show = callback; }, onHide: callback => { hooks.hide = callback; }, onUnload: callback => { hooks.unload = callback; },
    startRecording: async (options = {}) => { calls.start += 1; calls.onVoiceState = options.onState; }, stopRecording: async () => { calls.stop += 1; return '模拟语音草稿'; },
    cancelRecording: () => { calls.cancel += 1; }, speak: async () => {}, stopSpeaking: () => { calls.stopSpeaking += 1; },
    uni:{reLaunch:options=>calls.navigation.push(options),switchTab:options=>calls.navigation.push(options),navigateTo:options=>calls.navigation.push(options)},
    enterInvitation:()=>{}, getInviteIdentity:()=>identity,
    checkInvitationService:async()=>{calls.inviteChecks++;return true;},
    authenticateInvitation:async(fields,options)=>{calls.authentications.push({fields,options});identity={id:'u1',username:fields.username};return identity;},
    cancelInviteRequest:()=>{calls.inviteCancels++;},validateInvitationForm:()=>'',
    logoutInvitation:async options=>{calls.logouts.push(options);identity=null;},reloadInvitation:options=>calls.reloads.push(options),
    planGraphNavigation, GRAPH_NAVIGATION_LIMITS, GraphVoiceInput:{}, ModeBanner: {}, WorkspaceBar: {}, BrandHeader: {}, InstallGuide: {}, AnswerBody: {}, AnswerEvidence: {}, ReasoningEvidence: {}, KnowledgeSkillEvidence:{}, AnswerDeliverable: {}, AnswerFeedback: {}, CapabilityBoundary: {}, ProductGuide: {}, ...overrides.voice, ...overrides.invite,
  };
  const source = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), source)(...Object.values(scope));
  const effects = effectScope();
  const ui = effects.run(() => component.setup({}, { expose() {} }));
  return { ui, hooks, calls, api, preferences: value => { connectionRevision++;preferences = { ...preferences, ...value }; }, active: value => { active = value; }, dispose: () => { hooks.hide?.(); effects.stop(); } };
}

test('chat, documents, graph and settings compile and opening them never records, speaks or submits a question', async () => {
  for (const name of ['chat', 'documents', 'graph', 'settings']) {
    const f = mount(name); f.hooks.show(); await settle();
    assert.equal(f.calls.start, 0); assert.equal(f.calls.ask.length, 0); assert.equal(f.calls.save, 0);
    f.dispose();
  }
});

test('project workbench activity blocks sending, voice and session changes and delays history reads',async()=>{
  const f=mount('chat');f.hooks.show();await settle();f.ui.draft.value='Keep my draft';f.ui.setWorkbenchBusy(true);assert.equal(f.ui.canSend.value,false);await f.ui.send();await f.ui.toggleRecording();await f.ui.changeSession('b');assert.equal(f.ui.sessionId.value,'a');assert.equal(f.calls.ask.length,0);assert.equal(f.calls.start,0);
  await f.ui.reload();assert.equal(f.ui.sessionId.value,'a');f.ui.setWorkbenchBusy(false);await settle();assert.equal(f.ui.draft.value,'Keep my draft');assert.equal(f.ui.canSend.value,true);f.dispose();
});

test('return to composer preserves the draft and makes no question, and deactivates off page',async()=>{
  const f=mount('chat');
  try {
    assert.equal(f.ui.pageVisible.value,false);
    f.hooks.show();await settle();f.ui.draft.value='Keep this phone draft';
    assert.equal(f.ui.pageVisible.value,true);
    await f.ui.focusComposer();
    assert.equal(f.ui.composerFocused.value,true);
    assert.equal(f.ui.draft.value,'Keep this phone draft');
    assert.equal(f.calls.ask.length,0);
    f.hooks.hide();assert.equal(f.ui.pageVisible.value,false);
    await f.ui.focusComposer();assert.equal(f.ui.composerFocused.value,false);
    assert.equal(f.calls.ask.length,0);
  } finally {f.dispose();}
});
test('workbench recipes and explicit brief additions use the existing safe draft choice without calling a model',async()=>{
  const f=mount('chat');f.hooks.show();await settle();f.ui.draft.value='My original';f.ui.fillDraft('Project brief');assert.equal(f.ui.draft.value,'My original');assert.equal(f.ui.draftSuggestion.value,'Project brief');assert.equal(f.calls.ask.length,0);f.ui.resolveDraftSuggestion('append');assert.equal(f.ui.draft.value,'My original\n\nProject brief');assert.equal(f.calls.ask.length,0);f.dispose();
});
test('feedback and workbench share a network gate so neither overlaps a paid question',async()=>{
  const f=mount('chat');f.hooks.show();await settle();f.ui.draft.value='Question';f.ui.setFeedbackBusy({answerId:'saved-answer',active:true});assert.equal(f.ui.feedbackBusy.value,true);assert.equal(f.ui.networkBusy.value,true);await f.ui.send();await f.ui.toggleRecording();assert.equal(f.calls.ask.length,0);assert.equal(f.calls.start,0);
  f.ui.setWorkbenchBusy(true);f.ui.setFeedbackBusy({answerId:'saved-answer',active:false});assert.equal(f.ui.canSend.value,false);f.ui.setWorkbenchBusy(false);assert.equal(f.ui.canSend.value,true);f.dispose();
});

test('chat keeps bounded server fragments with the answer instead of dropping its sources', async () => {
  const sources = Array.from({length:8}, (_, index) => ({content:`Returned fragment ${index}`,confidence:1,verified:true}));
  const f = mount('chat',{api:{ask:async()=>({answer:'An answer with source fragments',sources})}});
  f.hooks.show(); await settle(); f.ui.draft.value = 'Question'; await f.ui.send();
  const answer = f.ui.messages.value.at(-1);
  assert.equal(answer.role,'assistant'); assert.equal(answer.content,'An answer with source fragments');
  assert.deepEqual(answer.sources,sources.slice(0,5));
  f.dispose();
});
test('enabled research answers preserve ten sources and checked metadata without claiming entailment',async()=>{
  const sources=Array.from({length:11},(_,index)=>({id:`fb27e5ae-5cb9-4650-9dfd-${String(index+1).padStart(12,'0')}`,label:`S${index+1}`,content:`Original excerpt ${index+1}`,filename:'Research.pdf',page:index+1}));
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,memory_write_enabled:true})},api:{ask:async(_sid,_question,options)=>({request_id:options.requestId,request_status:'completed',answer:'A bounded answer [S1]',research_core:true,answer_mode:'evidence',used_memory_count:1,sources,citation_validation:{entailment_verified:true,missing_citations:false,unknown_labels:[]}})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'offline-test-token'});f.hooks.show();await settle();assert.equal(f.ui.researchEnabled.value,true);
  f.ui.draft.value='Compare original evidence';await f.ui.send();const result=f.ui.messages.value.at(-1);assert.equal(result.sources.length,10);assert.equal(result.sources[9].label,'S10');assert.equal(result.used_memory_count,1);assert.equal(result.citation_validation.entailment_verified,false);assert.match(f.ui.researchAnswerNote(result),/不代表已经验证/);f.dispose();
});
test('research history and explicit request recovery preserve S9/S10 without admitting S11',async()=>{
  const sources=Array.from({length:11},(_,index)=>({id:`fb27e5ae-5cb9-4650-9dfd-${String(index+1).padStart(12,'0')}`,label:`S${index+1}`,content:`Original ${index+1}`}));
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{
    history:async()=>[{role:'assistant',content:'Historical answer [S10]',research_core:true,sources}],
    ask:async(_sid,_question,options)=>({request_id:options.requestId,request_status:'running'}),
    queryStatus:async(_sid,id)=>({request_id:id,request_status:'completed',answer:'Recovered answer [S9] [S10]',research_core:true,sources}),
  }});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();
  assert.equal(f.ui.messages.value[0].sources.length,10);assert.equal(f.ui.messages.value[0].sources[9].label,'S10');
  f.ui.draft.value='Continue the evidence review';await f.ui.send();await f.ui.checkPendingQuestion();
  const recovered=f.ui.messages.value.at(-1);assert.equal(recovered.sources.length,10);assert.deepEqual(recovered.sources.slice(8).map(item=>item.label),['S9','S10']);f.dispose();
});
test('answer to confirmed memory stages only a blank scoped draft after a click, never writes or asks again',async()=>{
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,memory_write_enabled:true})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'offline-test-token'});f.hooks.show();await settle();assert.equal(f.calls.memoryDrafts.length,0);
  const answer={role:'assistant',research_core:true,content:'A claim still requiring review',sources:[{id:'fb27e5ae-5cb9-4650-9dfd-000000000001',label:'S1',content:'Original source'}]};f.ui.prepareMemory(answer);
  assert.deepEqual(f.calls.memoryDrafts,[{session_id:'a',text:'',kind:'finding',source_ids:['fb27e5ae-5cb9-4650-9dfd-000000000001']}]);assert.deepEqual(f.calls.navigation,[{url:'/pages/memory/index'}]);assert.equal(f.calls.ask.length,0);assert.equal(f.calls.save,0);
  f.hooks.hide();f.ui.prepareMemory(answer);f.ui.openResearchLibrary();assert.equal(f.calls.memoryDrafts.length,1);assert.equal(f.calls.navigation.length,1);f.dispose();
});
test('research memory switch is per-question input and turning it off starts no model call itself',async()=>{
  let options;
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,memory_write_enabled:true})},api:{ask:async(_sid,_question,value)=>{options=value;return {request_id:value.requestId,request_status:'completed',answer:'No memory used',research_core:true,answer_mode:'general',used_memory_count:0,sources:[]};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'offline-test-token'});f.hooks.show();await settle();f.ui.useProjectMemory.value=false;assert.equal(options,undefined);f.ui.draft.value='This question excludes memory';await f.ui.send();assert.equal(options.useProjectMemory,false);f.dispose();
});
test('saved memory-off preference remains off when the research page first loads',async()=>{
  let options;
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,memory_write_enabled:true})},api:{ask:async(_sid,_question,value)=>{options=value;return {request_id:value.requestId,request_status:'completed',answer:'Memory stays off',research_core:true,sources:[]};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'offline-test-token',useMemory:false});f.hooks.show();await settle();
  assert.equal(f.ui.useProjectMemory.value,false,'the research page must not turn saved memory off back on');
  f.ui.draft.value='Continue without saved memory';await f.ui.send();assert.equal(options.useProjectMemory,false);f.dispose();
});
test('research local memory choice survives refresh but changes when saved memory preference changes',async()=>{
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,memory_write_enabled:true})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',useMemory:true});f.hooks.show();await settle();
  f.ui.useProjectMemory.value=false;await f.ui.reload();assert.equal(f.ui.useProjectMemory.value,false);
  f.preferences({useMemory:false});await f.ui.reload();assert.equal(f.ui.useProjectMemory.value,false);
  f.preferences({useMemory:true});await f.ui.reload();assert.equal(f.ui.useProjectMemory.value,true);assert.equal(f.calls.ask.length,0);assert.equal(f.calls.save,0);f.dispose();
});
test('private research 202 creates one tracked query and recovers only by explicit status read',async()=>{
  let queries=0,statusReads=0;
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{ask:async(_sid,_question,options)=>{queries++;return {request_id:options.requestId,request_status:'running'};},
    queryStatus:async(_sid,id)=>{statusReads++;return {request_id:id,request_status:'completed',answer:'Recovered research [S1]',research_core:true,sources:[]};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Original question';await f.ui.send();
  assert.match(f.ui.pendingQuestion.value.requestId,/^[0-9a-f-]{36}$/);assert.equal(f.ui.pendingQuestion.value.status,'running');assert.equal(f.ui.messages.value.length,0);assert.equal(f.ui.draft.value,'Original question');
  await f.ui.send();await settle();assert.equal(queries,1);assert.equal(statusReads,0);
  await f.ui.checkPendingQuestion();assert.equal(queries,1);assert.equal(statusReads,1);assert.equal(f.ui.pendingQuestion.value,null);assert.equal(f.ui.messages.value.at(-1).research_core,true);assert.equal(f.ui.draft.value,'');f.dispose();
});

test('known terminal model failure preserves draft and requires explicit return before a fresh request',async()=>{
  let queries=0;
  const failure={code:'model_response_truncated',message:'模型达到本次输出上限，未完成回答。没有自动重试；草稿已保留。',automatic_retry:false};
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{ask:async(_sid,_question,options)=>{queries++;return {request_id:options.requestId,request_status:'failed',request_error:failure};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Keep the original';await f.ui.send();
  const original=f.ui.pendingQuestion.value.requestId;
  assert.equal(f.ui.pendingQuestion.value.status,'failed');assert.deepEqual(f.ui.pendingQuestion.value.failure,failure);
  assert.match(f.ui.receipt.value.title,/已结束/);assert.equal(f.ui.messages.value.length,0);assert.equal(f.ui.draft.value,'Keep the original');assert.equal(f.ui.canSend.value,false);
  await f.ui.send();await f.ui.continuePendingQuestion();assert.equal(queries,1);
  await f.ui.releasePendingQuestion();assert.equal(queries,1);assert.equal(f.ui.pendingQuestion.value,null);assert.equal(f.ui.draft.value,'Keep the original');assert.match(f.ui.requestNotice.value,/没有重新发送/);
  await f.ui.send();assert.equal(queries,2);assert.notEqual(f.ui.pendingQuestion.value.requestId,original);f.dispose();
});

test('status recovery turns an uncertain request into known failure without assistant text or resubmission',async()=>{
  let queries=0,statusReads=0;
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{ask:async()=>{queries++;throw new Error('Disconnected');},
    queryStatus:async(_sid,id)=>{statusReads++;return {request_id:id,request_status:'failed',request_error:{code:'model_response_empty',message:'模型没有提供正文；没有自动重试。',automatic_retry:false}};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Retain question';await f.ui.send();assert.equal(f.ui.pendingQuestion.value.status,'unconfirmed');
  await f.ui.checkPendingQuestion();assert.equal(f.ui.pendingQuestion.value.status,'failed');assert.equal(statusReads,1);assert.equal(queries,1);assert.equal(f.ui.messages.value.length,0);assert.equal(f.ui.draft.value,'Retain question');f.dispose();
});
test('private research interrupted request can explicitly continue only the original ID, question and four controls',async()=>{
  const queries=[];
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{ask:async(...args)=>{queries.push(args);if(queries.length===1) throw new Error('Disconnected');return {request_id:args[2].requestId,request_status:'completed',answer:'Recovered'};},
    queryStatus:async(_sid,id)=>({request_id:id,request_status:'not_found'})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',useMemory:false,useContext:false,useCache:false,useEvolution:false});f.hooks.show();await settle();
  f.ui.draft.value='Original';await f.ui.send();assert.equal(f.ui.pendingQuestion.value.status,'unconfirmed');await f.ui.send();assert.equal(queries.length,1);
  await f.ui.checkPendingQuestion();f.preferences({useMemory:true,useContext:true,useCache:true,useEvolution:true});await f.ui.reload();f.ui.draft.value='Edited after disconnect';f.ui.useProjectMemory.value=true;
  await f.ui.continuePendingQuestion();assert.equal(queries.length,2);assert.equal(queries[1][1],'Original');assert.deepEqual(queries[0][2],queries[1][2]);
  for(const key of ['useProjectMemory','useConversationContext','useLLMCache','useSelfEvolution']) assert.equal(queries[1][2][key],false);f.dispose();
});
test('normal chat has no reasoning selector and historical memory binds to its original continuation',async()=>{
  const queries=[];const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{ask:async(...args)=>{queries.push(args);if(queries.length===1) throw new Error('Disconnected');return {request_id:args[2].requestId,request_status:'completed',answer:'Recovered'};},queryStatus:async(_sid,id)=>({request_id:id,request_status:'not_found'})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();assert.equal(f.ui.reasoningMode,undefined);assert.equal(f.ui.reasoningModes,undefined);f.ui.setMemoryDate({detail:{value:'2025-01-15'}});assert.equal(queries.length,0);f.ui.draft.value='请参考旧案例，帮我写一段讨论';await f.ui.send();assert.equal('reasoningMode' in queries[0][2],false);assert.equal(queries[0][2].memoryAsOf,new Date('2025-01-15T23:59:59').getTime()/1000);
  await f.ui.checkPendingQuestion();f.ui.memoryDate.value='2026-02-10';await f.ui.continuePendingQuestion();assert.deepEqual(queries[1][2],queries[0][2]);f.dispose();
  const page=readFileSync(new URL('pages/chat/index.vue',root),'utf8');assert.doesNotMatch(page,/reasoning-options|reasoning-choice|v-for="option in reasoningModes"/);assert.match(page,/直接提问、讨论或写作/);
});
test('knowledge support defaults to automatic task routing and freezes its choice in a same-ID continuation',async()=>{
  const queries=[];
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,knowledge_skills:true})},api:{ask:async(...args)=>{queries.push(args);if(queries.length===1)throw new Error('Interrupted');return{request_id:args[2].requestId,request_status:'completed',answer:'Finished'};},queryStatus:async(_sid,id)=>({request_id:id,request_status:'not_found'})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();assert.equal(f.ui.knowledgeSkillMode.value,'auto');assert.equal(f.ui.knowledgeSkillsEnabled.value,true);assert.equal(queries.length,0);
  f.ui.draft.value='帮我梳理职场沟通';await f.ui.send();assert.equal(queries[0][2].knowledgeSkillMode,'auto');assert.equal(f.ui.pendingQuestion.value.selectedKnowledgeMode,'auto');
  await f.ui.checkPendingQuestion();f.ui.knowledgeSkillMode.value='off';await f.ui.continuePendingQuestion();assert.deepEqual(queries[1][2],queries[0][2]);f.dispose();
});
test('knowledge override resets across sessions and identities; explicit off is sent and never asks itself',async()=>{
  const queries=[];
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true,knowledge_skills:true})},api:{ask:async(...args)=>{queries.push(args);return{request_id:args[2].requestId,request_status:'completed',answer:'Finished'};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.toggleKnowledgeSkills();assert.equal(f.ui.knowledgeSkillMode.value,'off');await f.ui.reload();assert.equal(f.ui.knowledgeSkillMode.value,'off');
  await f.ui.changeSession('b');assert.equal(f.ui.knowledgeSkillMode.value,'auto');f.ui.toggleKnowledgeSkills();f.preferences({accessToken:'changed'});await f.ui.reload();assert.equal(f.ui.knowledgeSkillMode.value,'auto');
  f.ui.toggleKnowledgeSkills();assert.equal(queries.length,0);f.ui.draft.value='普通问题';await f.ui.send();assert.equal(queries[0][2].knowledgeSkillMode,'off');f.dispose();
});
test('older research services make no knowledge catalog read and cannot receive a new skills field',async()=>{
  let catalogs=0;const queries=[];const f=mount('chat',{research:{capabilities:async()=>({enabled:true}),knowledgeSkills:async()=>{catalogs++;return[];}},api:{ask:async(...args)=>{queries.push(args);return{request_id:args[2].requestId,request_status:'completed',answer:'Answer'};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='职场问题';await f.ui.send();assert.equal(catalogs,0);assert.equal('knowledgeSkillMode' in queries[0][2],false);f.dispose();
});
test('K-only replies are not hidden by an empty user-source list and explain methods are not user facts',()=>{
  const f=mount('chat');const note=f.ui.researchAnswerNote({sources:[],knowledge_skill:{status:'ready'}});assert.match(note,/方法知识参考/);assert.match(note,/未附你的资料原文/);assert.match(note,/不代表模型已正确遵循/);
  const source=readFileSync(new URL('pages/chat/index.vue',root),'utf8');assert.match(source,/<KnowledgeSkillEvidence v-if="message.role === 'assistant' && message.research_core"/);assert.doesNotMatch(source,/<KnowledgeSkillEvidence[^>]+sources\?\.length/);f.dispose();
});
test('an already captured legacy mode is preserved only for explicit same-ID continuation',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc',queries=[];
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{ask:async(...args)=>{queries.push(args);return {request_id:id,request_status:'completed',answer:'Recovered legacy answer'};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();
  f.ui.pendingQuestion.value={requestId:id,question:'Original legacy question',session:'a',status:'not_found',contextEnabled:false,memoryEnabled:false,cacheEnabled:false,evolutionEnabled:false,selectedMode:'deduction',memoryAsOf:0};
  await f.ui.continuePendingQuestion();assert.equal(queries.length,1);assert.equal(queries[0][1],'Original legacy question');assert.equal(queries[0][2].reasoningMode,'deduction');assert.equal(queries[0][2].requestId,id);assert.equal(queries[0][2].memoryAsOf,0);f.dispose();
});
test('private research restores unresolved history references without inventing original prompts or resending',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';let queries=0;
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},api:{pendingQueries:()=>[{request_id:id,request_status:'unknown'}],ask:async()=>{queries++;},queryStatus:async()=>({request_id:id,request_status:'unknown'})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();assert.equal(f.ui.pendingQuestion.value.requestId,id);assert.equal(f.ui.pendingQuestion.value.question,'');
  await f.ui.continuePendingQuestion();await f.ui.checkPendingQuestion();assert.equal(queries,0);assert.equal(f.ui.pendingQuestion.value.status,'unknown');f.dispose();
});
test('capability network failure cannot silently downgrade an opted research query to untracked legacy sending',async()=>{
  const f=mount('chat',{research:{capabilities:async()=>{throw new Error('Capability read unavailable');}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Do not pay without safe protocol';
  assert.equal(f.ui.sessionId.value,'');assert.equal(f.ui.canSend.value,false);await f.ui.send();assert.equal(f.calls.ask.length,0);assert.match(f.ui.error.value,/Capability read unavailable/);f.dispose();
});
test('chat capability, session and history reads respect a gateway allowing only one active request per user',async()=>{
  let active=0,maximum=0;const order=[];
  const read=async(name,result)=>{if(active) throw Object.assign(new Error('Gateway rejects concurrent reads'),{status:429});active++;maximum=Math.max(maximum,active);order.push(name);try {await settle();return result;}finally {active--;}};
  const f=mount('chat',{research:{capabilities:()=>read('capabilities',{enabled:true})},api:{sessions:()=>read('sessions',[{id:'a',title:'Project A'}]),history:()=>read('history',[{role:'assistant',content:'Existing private history'}])}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();
  for(let turn=0;turn<5;turn++) await settle();
  assert.deepEqual(order,['capabilities','sessions','history']);assert.equal(maximum,1);assert.equal(f.ui.error.value,'');assert.equal(f.ui.messages.value[0].content,'Existing private history');assert.equal(f.calls.ask.length,0);f.dispose();
});
test('private research invalid UTF-8 or oversized question retains the complete editable draft before a model request',async()=>{
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();
  for(const text of ['文'.repeat(3000),'\ud800']) { f.ui.draft.value=text;assert.equal(f.ui.inviteQuestionOverLimit.value,true);await f.ui.send();assert.equal(f.ui.draft.value,text); }
  assert.equal(f.calls.ask.length,0);f.dispose();
});
test('async secure ID preparation permits only one send and blocks voice/session mutation before the question is sent',async()=>{
  const wait=deferred();let preparations=0;let current;
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},requestIds:{queryRequestIdSupported:()=>true,createQueryRequestIdAsync:options=>{preparations++;current=options.isCurrent;return wait.promise;}},
    api:{ask:async(...args)=>{f.calls.ask.push(args);return {request_id:args[2].requestId,request_status:'completed',answer:'One answer'};}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Explicit question';
  const sending=f.ui.send();assert.equal(f.ui.preparingRequest.value,true);assert.equal(f.ui.canSend.value,false);await f.ui.send();await f.ui.toggleRecording();await f.ui.createSession();assert.equal(preparations,1);assert.equal(f.calls.ask.length,0);assert.equal(f.calls.start,0);assert.equal(current(),true);
  wait.resolve('ed0b7a48-cb27-430d-95f1-02b2757a69cc');await sending;assert.equal(f.calls.ask.length,1);assert.equal(f.ui.preparingRequest.value,false);assert.equal(f.ui.messages.value.at(-1).content,'One answer');f.dispose();
});
test('late secure ID cannot send a hidden page, changed connection, changed project or changed draft',async()=>{
  for(const action of ['hide','connection','project','draft']) {
    const wait=deferred();let current;
    const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},requestIds:{queryRequestIdSupported:()=>true,createQueryRequestIdAsync:options=>{current=options.isCurrent;return wait.promise;}}});
    f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Original';const sending=f.ui.send();
    if(action==='hide') f.hooks.hide();else if(action==='connection') f.preferences({baseUrl:'https://different.example'});else if(action==='project') f.ui.sessionId.value='b';else f.ui.draft.value='Edited';
    assert.equal(current(),false);wait.resolve('ed0b7a48-cb27-430d-95f1-02b2757a69cc');await sending;assert.equal(f.calls.ask.length,0);assert.equal(f.ui.pendingQuestion.value,null);assert.equal(f.ui.preparingRequest.value,false);f.dispose();
  }
});
test('missing native crypto disables new research sends with an editable draft, while demo and ordinary legacy chat stay available',async()=>{
  for(const mode of ['research','demo','legacy']) {
    let preparations=0;
    const f=mount('chat',{research:{capabilities:async()=>({enabled:mode==='research'})},requestIds:{queryRequestIdSupported:()=>false,createQueryRequestIdAsync:async()=>{preparations++;throw new Error('Unsupported native crypto');}}});
    if(mode!=='demo') f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Editable draft';
    assert.equal(f.ui.requestIdUnavailable.value,mode==='research');assert.equal(f.ui.canSend.value,mode!=='research');await f.ui.send();assert.equal(preparations,0);assert.equal(f.calls.ask.length,mode==='research'?0:1);if(mode==='research') assert.equal(f.ui.draft.value,'Editable draft');f.dispose();
  }
});
test('secure native ID failure preserves the draft and never starts a question or an automatic retry',async()=>{
  let preparations=0;const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},requestIds:{queryRequestIdSupported:()=>true,createQueryRequestIdAsync:async()=>{preparations++;throw new Error('raw-native-secret');}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();f.ui.draft.value='Keep original';await f.ui.send();assert.equal(preparations,1);assert.equal(f.calls.ask.length,0);assert.equal(f.ui.preparingRequest.value,false);assert.equal(f.ui.draft.value,'Keep original');assert.match(f.ui.error.value,/没有发送问题/);assert.doesNotMatch(f.ui.error.value,/raw-native-secret/);f.dispose();
});
test('research source revocation and incomplete references remain warnings, not verified answer badges',async()=>{
  const f=mount('chat');assert.match(f.ui.researchAnswerNote({citation_validation:{warnings:[{code:'source_revoked'}]}}),/来源已被删除/);
  assert.match(f.ui.researchAnswerNote({citation_validation:{unknown_labels:['S99']}}),/引用不完整/);
  assert.match(f.ui.researchAnswerNote({answer_mode:'insufficient_evidence'}),/没有请求模型/);f.dispose();
});
test('research chat explains missing evidence and local retrieval limitations in ordinary language',()=>{
  const f=mount('chat');
  const checks=[['evidence_facets_uncovered',/部分问题要点还缺少具体原文/],
    ['evidence_anchors_uncovered',/不能用背景介绍代替/],
    ['complete_statement_exceeds_budget',/限定条件未能放入/],
    ['semantic_retrieval_unavailable',/语义检索暂不可用/],
    ['semantic_retrieval_windowed',/不代表已查完全部资料/],
    ['literal_coverage_scan_limited',/没有查完全部资料/],
    ['source_boundary_unverified',/标点完整也不等于前提完整/],
    ['source_antecedent_unverified',/不能从下一实验猜前提/],
    ['uncovered_query_terms',/不是概念缺失或全文结论/],
    ['evidence_coverage_proxy',/不代表已找齐关键证据或验证结论/]];
  for (const [code,expected] of checks) {
    const note=f.ui.researchAnswerNote({retrieval_gaps:[{code,message:'untrusted-server-text'}]});
    assert.match(note,expected);assert.doesNotMatch(note,/untrusted-server-text/);
  }
  assert.match(f.ui.researchAnswerNote({retrieval_gaps:[{code:'evidence_coverage_proxy'},{code:'evidence_facets_uncovered'}]}),/缺少具体原文/);
  f.dispose();
});
test('chat feedback binds only a persisted answer; history supplies existing votes with no extra GET',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  const feedback={answer_id:id,rating:'inaccurate',note:'已有反馈'};
  const f=mount('chat',{api:{history:async()=>[{id,role:'assistant',content:'Saved answer',feedback}],
    feedback:()=>{throw new Error('Do not issue extra requests to load feedback');},ask:async()=>({answer:'Next answer',answer_id:id})}});
  f.hooks.show();await settle();assert.deepEqual(f.ui.messages.value[0].feedback,feedback);
  f.ui.draft.value='Next question';await f.ui.send();assert.equal(f.ui.messages.value.at(-1).id,id);f.dispose();
  const legacy=mount('chat');legacy.hooks.show();await settle();legacy.ui.draft.value='Legacy';await legacy.ui.send();
  assert.equal(legacy.ui.messages.value.at(-1).id,'');legacy.dispose();
});

test('invitation page opens with status only; registration is an explicit form submit and wipes secrets',async () => {
  const f = mount('invite');f.hooks.show();await settle();
  assert.equal(f.calls.inviteChecks,1);assert.equal(f.calls.authentications.length,0);assert.equal(f.calls.ask.length,0);assert.equal(f.calls.start,0);
  f.ui.setRegister(true);f.ui.username.value='invited_user';f.ui.password.value='correct password 123';f.ui.inviteCode.value='i'.repeat(40);
  assert.equal(f.calls.authentications.length,0);await f.ui.submit();
  assert.equal(f.calls.authentications.length,1);assert.equal(f.calls.authentications[0].options.register,true);
  assert.equal(f.calls.authentications[0].fields.username,'invited_user');assert.equal(f.ui.password.value,'');assert.equal(f.ui.inviteCode.value,'');
  assert.deepEqual(f.calls.navigation,[{url:'/pages/chat/index'}]);assert.equal(f.calls.save,0);f.dispose();
});
test('invitation status error blocks submit; explicit retry never sends credentials or prompts',async () => {
  let checks=0;const f=mount('invite',{invite:{checkInvitationService:async()=>{checks++;throw new Error('当前网站不是兼容的邀请入口');}}});
  f.hooks.show();await settle();await f.ui.submit();assert.equal(f.calls.authentications.length,0);
  assert.equal(f.ui.available.value,false);assert.match(f.ui.error.value,/不是兼容/);
  await f.ui.checkService();assert.equal(checks,2);assert.equal(f.calls.ask.length,0);f.dispose();
});
test('invitation page hide clears fields and discards a late auth response without navigating',async () => {
  const wait=deferred();const f=mount('invite',{invite:{authenticateInvitation:()=>wait.promise}});
  f.hooks.show();await settle();f.ui.username.value='invited_user';f.ui.password.value='correct password 123';f.ui.inviteCode.value='i'.repeat(40);
  const pending=f.ui.submit();f.hooks.hide();wait.resolve({id:'u1',username:'invited_user'});await pending;
  assert.equal(f.ui.password.value,'');assert.equal(f.ui.inviteCode.value,'');assert.equal(f.ui.username.value,'');assert.equal(f.calls.navigation.length,0);assert.ok(f.calls.inviteCancels>0);f.dispose();
});
test('invitation logout stops audio and reloads login; demo is a separate explicit leave action',async () => {
  const f=mount('invite');f.hooks.show();await settle();await f.ui.signOut();
  assert.equal(f.calls.logouts.length,1);assert.equal(f.calls.reloads.length,1);assert.ok(f.calls.cancel>0);assert.ok(f.calls.stopSpeaking>0);f.dispose();
  const demo=mount('invite');demo.hooks.show();await settle();await demo.ui.viewDemo();
  assert.deepEqual(demo.calls.logouts,[{leave:true}]);assert.deepEqual(demo.calls.reloads,[{leave:true}]);demo.dispose();
});
test('invitation settings expose account management not private connection checks; controls save without model calls',async () => {
  const f=mount('settings');f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'u1',username:'invited_user'},baseUrl:'https://invite.example',accessToken:'user-token',useMemory:false,useEvolution:false});
  f.hooks.show();assert.equal(f.ui.canCheckConnection.value,false);await f.ui.testConnection();assert.equal(f.calls.connectionChecks,0);
  f.ui.form.value.useContext=false;await f.ui.save();assert.equal(f.calls.save,1);assert.match(f.ui.notice.value,/回答设置已保存/);assert.equal(f.calls.ask.length,0);
  f.ui.openInvite();assert.deepEqual(f.calls.navigation,[{url:'/pages/invite/index'}]);f.dispose();
});
test('invitation pages do not read unopened graph/files and chat rejects voice before any microphone or TTS call',async () => {
  for(const page of ['graph','documents']) {
    const f=mount(page,{api:{sessions:()=>{throw new Error('No feature reads in invite mode');}}});
    f.preferences({mode:'connected',connectionKind:'invite'});f.hooks.show();await settle();assert.equal(f.ui.loading.value,false);assert.equal(f.ui.error.value,'');f.dispose();
  }
  let speaks=0;const f=mount('chat',{voice:{speak:async()=>{speaks++;}}});f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'}});f.hooks.show();await settle();
  await f.ui.toggleRecording();await f.ui.readAnswer('hello',0);assert.equal(f.calls.start,0);assert.equal(speaks,0);assert.match(f.ui.voiceNotice.value,/暂未开放/);
  f.ui.draft.value='previous user private draft';f.hooks.hide();f.preferences({inviteUser:{id:'b'}});f.hooks.show();await settle();assert.equal(f.ui.draft.value,'');f.dispose();
});

test('voice requires two explicit taps, fills draft, and only Send submits it', async () => {
  const f = mount('chat'); f.hooks.show(); await settle();
  await f.ui.toggleRecording(); assert.equal(f.ui.recording.value, true);
  await f.ui.toggleRecording(); assert.equal(f.ui.draft.value, '模拟语音草稿');
  assert.equal(f.calls.ask.length, 0); assert.equal(f.calls.start, 1); assert.equal(f.calls.stop, 1);
  await f.ui.send(); assert.deepEqual(f.calls.ask, [['a', '模拟语音草稿']]); assert.equal(f.ui.draft.value, '');
  f.dispose();
});

test('voice is visibly a draft input and can undo only its own untouched addition',async()=>{
  const f=mount('chat');f.hooks.show();await settle();f.ui.draft.value='  Original draft\n';
  await f.ui.toggleRecording();assert.equal(f.ui.voiceStage.value,'recording');
  await f.ui.toggleRecording();await settle();
  assert.equal(f.ui.voiceStage.value,'review');assert.equal(f.ui.composerFocused.value,true);
  assert.equal(f.ui.canUndoVoiceDraft.value,true);assert.equal(f.calls.ask.length,0);
  f.ui.undoVoiceDraft();assert.equal(f.ui.draft.value,'  Original draft\n');assert.equal(f.ui.canUndoVoiceDraft.value,false);assert.equal(f.ui.composerRevision.value,1);
  assert.equal(f.calls.ask.length,0);assert.match(f.ui.voiceNotice.value,/原草稿保留/);f.dispose();
});

test('undo never overwrites subsequent edits and voice draft metadata does not cross page hides',async()=>{
  const f=mount('chat');f.hooks.show();await settle();
  await f.ui.toggleRecording();await f.ui.toggleRecording();f.ui.draft.value += '\nMy correction';
  assert.equal(f.ui.canUndoVoiceDraft.value,false);f.ui.undoVoiceDraft();assert.match(f.ui.draft.value,/My correction/);
  f.hooks.hide();assert.equal(f.ui.lastVoiceDraft.value,null);assert.equal(f.ui.voiceStage.value,'');
  f.hooks.show();await settle();assert.match(f.ui.draft.value,/My correction/);assert.equal(f.ui.canUndoVoiceDraft.value,false);f.dispose();
});

test('new voice operations share the chat busy gate without starting capture or ASR',async()=>{
  const f=mount('chat');f.hooks.show();await settle();
  for(const name of ['switching','creating','checkingRequest','preparingRequest','asking','loading']) {
    f.ui[name].value=true;await f.ui.toggleRecording();assert.equal(f.ui.voiceActionBlocked.value,true);f.ui[name].value=false;
  }
  f.ui.pendingQuestion.value={status:'unknown'};await f.ui.toggleRecording();f.ui.pendingQuestion.value=null;
  f.ui.draftSuggestion.value='Choice unresolved';await f.ui.toggleRecording();f.ui.draftSuggestion.value='';
  f.hooks.hide();await f.ui.toggleRecording();assert.equal(f.calls.start,0);assert.equal(f.calls.stop,0);f.dispose();
});

test('research byte budget blocks voice capture and ready-audio transcription without dropping either draft or audio',async()=>{
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();
  for(const text of ['文'.repeat(2730)+'aa','文'.repeat(3000),'\ud800']) {
    f.ui.draft.value=text;await f.ui.toggleRecording();assert.equal(f.calls.start,0);assert.equal(f.ui.draft.value,text);assert.match(f.ui.voiceNotice.value,/字节预算/);
  }
  f.ui.draft.value='';await f.ui.toggleRecording();f.calls.onVoiceState({state:'stopped',reason:'limit'});
  f.ui.draft.value='文'.repeat(3000);await f.ui.toggleRecording();assert.equal(f.calls.stop,0);assert.equal(f.ui.audioReady.value,true);assert.match(f.ui.voiceNotice.value,/录音仍保留/);
  f.ui.draft.value='Short';await f.ui.toggleRecording();assert.equal(f.calls.stop,1);f.dispose();
});

test('a byte-oversized ASR result stays complete but does not claim it is ready to send',async()=>{
  const transcript='文'.repeat(3000);
  const f=mount('chat',{research:{capabilities:async()=>({enabled:true})},voice:{stopRecording:async()=>transcript}});
  f.preferences({mode:'connected',baseUrl:'https://private.example'});f.hooks.show();await settle();
  await f.ui.toggleRecording();await f.ui.toggleRecording();
  assert.equal(f.ui.draft.value,transcript);assert.equal(f.ui.draftOverLimit.value,false);assert.equal(f.ui.canSend.value,false);
  assert.match(f.ui.voiceNotice.value,/完整保留/);assert.match(f.ui.voiceNotice.value,/字节预算/);assert.match(f.ui.voiceNotice.value,/没有自动提交/);assert.equal(f.calls.ask.length,0);f.dispose();
});

test('cancelling transcribing keeps the draft and states the already-submitted processing boundary',async()=>{
  const wait=deferred(),f=mount('chat',{voice:{stopRecording:()=>wait.promise}});
  f.hooks.show();await settle();f.ui.draft.value='Keep';await f.ui.toggleRecording();const pending=f.ui.toggleRecording();
  assert.equal(f.ui.voiceStage.value,'transcribing');f.ui.cancelVoice();assert.equal(f.ui.draft.value,'Keep');
  assert.match(f.ui.voiceNotice.value,/不能撤回已开始/);wait.resolve('Ignore this result');await pending;
  assert.equal(f.ui.draft.value,'Keep');assert.equal(f.calls.ask.length,0);assert.equal(f.ui.lastVoiceDraft.value,null);f.dispose();
});

test('voice keyboard activation ignores IME/repeats/shortcuts, while cancel remains usable when other work is busy',async()=>{
  const f=mount('chat');f.hooks.show();await settle();
  for(const extra of [{repeat:true},{isComposing:true},{ctrlKey:true},{metaKey:true},{altKey:true},{shiftKey:true},{keyCode:229},{defaultPrevented:true}]) f.ui.voiceKeydown({key:'Enter',...extra},'record');
  assert.equal(f.calls.start,0);f.ui.voiceKeydown({key:'Enter',preventDefault(){}},'record');await settle();assert.equal(f.calls.start,1);
  f.ui.setWorkbenchBusy(true);assert.equal(f.ui.voiceActionBlocked.value,true);
  f.ui.voiceKeydown({key:' ',preventDefault(){}},'cancel');assert.equal(f.ui.recording.value,false);assert.equal(f.calls.stop,0);
  f.ui.setWorkbenchBusy(false);f.dispose();
});

test('stopping current speech is always available even if another operation became busy',async()=>{
  const wait=deferred();let speaks=0;
  const f=mount('chat',{voice:{speak:()=>{speaks++;return wait.promise;}}});f.hooks.show();await settle();
  const reading=f.ui.readAnswer('A short answer',0);assert.equal(f.ui.speakingIndex.value,0);
  f.ui.setWorkbenchBusy(true);await f.ui.readAnswer('A short answer',0);assert.equal(f.ui.speakingIndex.value,-1);
  await f.ui.readAnswer('Another answer',1);assert.equal(speaks,1);
  wait.resolve();await reading;f.ui.setWorkbenchBusy(false);f.dispose();
});

test('late transcription is ignored after the page hides', async () => {
  const pending = deferred(); const f = mount('chat', { voice: { stopRecording: () => pending.promise } });
  f.hooks.show(); await settle(); await f.ui.toggleRecording();
  const processing = f.ui.toggleRecording(); f.hooks.hide(); pending.resolve('do not send this'); await processing;
  assert.equal(f.ui.draft.value, ''); assert.equal(f.ui.recording.value, false); assert.ok(f.calls.cancel > 0); assert.ok(f.calls.stopSpeaking > 0); assert.equal(f.calls.ask.length, 0);
  f.dispose();
});

test('a full or oversized draft blocks recording before any ASR work', async () => {
  const f = mount('chat'); f.hooks.show(); await settle();
  for (const size of [6000, 6001]) {
    f.ui.draft.value = '字'.repeat(size); await f.ui.toggleRecording();
    assert.equal(f.calls.start, 0); assert.equal(f.calls.stop, 0); assert.equal(f.ui.recording.value, false);
    assert.match(f.ui.voiceNotice.value, /先编辑缩短再录音/);
  }
  assert.equal(f.ui.canSend.value, false); await f.ui.send(); assert.equal(f.calls.ask.length, 0); f.dispose();
});

test('oversized transcription preserves every character and stays editable but unsendable', async () => {
  const original = '  原草稿\n' + '文'.repeat(5980) + '  ';
  const transcript = '  ' + '完整转写'.repeat(20) + '\n';
  const f = mount('chat', { voice: { stopRecording: async () => transcript } });
  f.hooks.show(); await settle(); f.ui.draft.value = original;
  await f.ui.toggleRecording(); await f.ui.toggleRecording();
  assert.equal(f.ui.draft.value, `${original}\n${transcript}`);
  assert.equal(f.ui.draftOverLimit.value, true); assert.equal(f.ui.canSend.value, false);
  assert.equal(f.ui.recording.value, false); assert.equal(f.ui.voiceBusy.value, false);
  assert.match(f.ui.voiceNotice.value, /完整保留/); assert.match(f.ui.voiceNotice.value, /编辑缩短/);
  await f.ui.send(); assert.equal(f.calls.ask.length, 0);
  f.ui.draft.value = '人工确认后缩短的问题'; assert.equal(f.ui.canSend.value, true);
  const source = readFileSync(new URL('pages/chat/index.vue', root), 'utf8');
  assert.match(source, /:maxlength="-1"/); // Native textarea must not truncate the retained text.
  f.dispose();
});

test('a long transcript with no previous draft is also preserved and Unicode counts match characters', async () => {
  const transcript = '🌱'.repeat(6001);
  const f = mount('chat', { voice: { stopRecording: async () => transcript } });
  f.hooks.show(); await settle(); await f.ui.toggleRecording(); await f.ui.toggleRecording();
  assert.equal(f.ui.draft.value, transcript); assert.equal(f.ui.draftLength.value, 6001); assert.equal(f.ui.canSend.value, false);
  f.ui.draft.value = '🌱'.repeat(6000); assert.equal(f.ui.draftLength.value, 6000); assert.equal(f.ui.canSend.value, true); f.dispose();
});

test('automatic recorder stop presents an explicit transcribe action without calling ASR or asking', async () => {
  const f = mount('chat'); f.hooks.show(); await settle(); await f.ui.toggleRecording();
  f.calls.onVoiceState({ state: 'stopped', reason: 'limit' });
  assert.equal(f.ui.recording.value, false); assert.equal(f.ui.audioReady.value, true); assert.equal(f.ui.voiceBusy.value, false);
  assert.match(f.ui.voiceNotice.value, /录音已停止/); assert.match(f.ui.voiceNotice.value, /转成文字/);
  assert.equal(f.calls.stop, 0); assert.equal(f.calls.ask.length, 0);
  await f.ui.toggleRecording(); assert.equal(f.calls.stop, 1); assert.equal(f.ui.audioReady.value, false);
  assert.equal(f.ui.draft.value, '模拟语音草稿'); assert.equal(f.calls.ask.length, 0); f.dispose();
});

test('ready audio is retained without paid transcription while the editable draft is full', async () => {
  const f = mount('chat'); f.hooks.show(); await settle(); await f.ui.toggleRecording();
  f.calls.onVoiceState({ state: 'stopped', reason: 'device' }); f.ui.draft.value = '文'.repeat(6000);
  await f.ui.toggleRecording(); assert.equal(f.calls.stop, 0); assert.equal(f.ui.audioReady.value, true);
  assert.match(f.ui.voiceNotice.value, /录音仍保留/);
  f.ui.draft.value = '已缩短'; await f.ui.toggleRecording(); assert.equal(f.calls.stop, 1); assert.equal(f.ui.audioReady.value, false); f.dispose();
});

test('recorder failure immediately clears the recording UI and leaves draft untouched', async () => {
  const f = mount('chat'); f.hooks.show(); await settle(); f.ui.draft.value = 'Original'; await f.ui.toggleRecording();
  f.calls.onVoiceState({ state: 'error', reason: 'interrupted', message: '系统中断录音，请重试' });
  assert.equal(f.ui.recording.value, false); assert.equal(f.ui.audioReady.value, false); assert.equal(f.ui.voiceBusy.value, false);
  assert.equal(f.ui.voiceNotice.value, '系统中断录音，请重试'); assert.equal(f.ui.draft.value, 'Original'); assert.equal(f.calls.stop, 0); f.dispose();
});

test('early stop notification is not overwritten when startRecording resolves afterward', async () => {
  const f = mount('chat', { voice: { startRecording: async ({ onState }) => { onState({ state: 'stopped', reason: 'device' }); } } });
  f.hooks.show(); await settle(); await f.ui.toggleRecording();
  assert.equal(f.ui.recording.value, false); assert.equal(f.ui.audioReady.value, true); assert.match(f.ui.voiceNotice.value, /录音已停止/); f.dispose();
});

test('late recorder callbacks after hiding cannot revive recording or audio-ready state', async () => {
  const f = mount('chat'); f.hooks.show(); await settle(); await f.ui.toggleRecording(); const oldCallback = f.calls.onVoiceState;
  f.hooks.hide(); oldCallback({ state: 'stopped', reason: 'limit' });
  assert.equal(f.ui.recording.value, false); assert.equal(f.ui.audioReady.value, false); assert.equal(f.ui.voiceNotice.value, '');
  f.hooks.show(); await settle(); await f.ui.toggleRecording(); oldCallback({ state: 'error', message: 'Old failure' });
  assert.equal(f.ui.recording.value, true); assert.doesNotMatch(f.ui.voiceNotice.value, /Old failure/); f.dispose();
});

test('failed questions retain draft and tell the user to inspect history before retrying', async () => {
  const f = mount('chat', { api: { ask: async () => { throw new Error('连接中断'); } } });
  f.hooks.show(); await settle(); f.ui.draft.value = 'Keep this question'; await f.ui.send();
  assert.equal(f.ui.draft.value, 'Keep this question'); assert.equal(f.ui.messages.value.length, 0);
  assert.match(f.ui.error.value, /刷新历史/); assert.equal(f.ui.asking.value, false); f.dispose();
});

test('invitation draft budget prevents paid requests before sending and does not truncate the draft',async()=>{
  const f=mount('chat');f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'}});
  f.hooks.show();await settle();const question='文'.repeat(2731);f.ui.draft.value=question;
  assert.equal(f.ui.inviteQuestionOverLimit.value,true);assert.equal(f.ui.canSend.value,false);await f.ui.send();
  assert.equal(f.calls.ask.length,0);assert.equal(f.ui.draft.value,question);f.dispose();
});
test('202 running is not an answer, does not clear a draft, and manual status fetch recovers without asking again',async()=>{
  let queries=0,statusReads=0;
  const f=mount('chat',{api:{ask:async(sid,question,options)=>{queries++;return {request_id:options.requestId,request_status:'running'};},
    queryStatus:async(sid,id)=>{statusReads++;return {request_id:id,request_status:'completed',answer:'Recovered answer',answer_id:'ed0b7a48-cb27-430d-95f1-02b2757a69cc',sources:[]};}}});
  f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'}});f.hooks.show();await settle();f.ui.draft.value='Question';await f.ui.send();
  assert.equal(f.ui.pendingQuestion.value.status,'running');assert.equal(f.ui.messages.value.length,0);assert.equal(f.ui.draft.value,'Question');assert.equal(f.ui.canSend.value,false);
  await f.ui.send();assert.equal(queries,1);assert.equal(statusReads,0);
  await f.ui.checkPendingQuestion();assert.equal(queries,1);assert.equal(statusReads,1);assert.equal(f.ui.pendingQuestion.value,null);
  assert.equal(f.ui.messages.value.at(-1).content,'Recovered answer');assert.equal(f.ui.draft.value,'');assert.match(f.ui.requestNotice.value,/没有再次请求模型/);f.dispose();
});
test('unknown outcome blocks plain send; not_found can explicitly continue the original id and context',async()=>{
  const calls=[];
  const f=mount('chat',{api:{ask:async(...args)=>{calls.push(args);if(calls.length===1) throw new Error('Network disconnected');return {answer:'Answer',request_status:'completed',request_id:args[2].requestId};},
    queryStatus:async(sid,id)=>({request_id:id,request_status:'not_found'})}});
  f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'},useContext:false});f.hooks.show();await settle();f.ui.draft.value='Original';await f.ui.send();
  assert.equal(f.ui.pendingQuestion.value.status,'unconfirmed');await f.ui.send();assert.equal(calls.length,1);
  await f.ui.checkPendingQuestion();f.preferences({useContext:true});f.ui.draft.value='Edited draft';await f.ui.continuePendingQuestion();
  assert.equal(calls.length,2);assert.equal(calls[1][1],'Original');assert.deepEqual(calls[0][2],calls[1][2]);assert.equal(calls[1][2].useConversationContext,false);f.dispose();
});
test('history request references survive a refresh without inventing a lost question or starting a model',async()=>{
  let queries=0;const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  const f=mount('chat',{api:{pendingQueries:()=>[{request_id:id,request_status:'unknown'}],ask:async()=>{queries++;},queryStatus:async()=>({request_id:id,request_status:'unknown'})}});
  f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'}});f.hooks.show();await settle();
  assert.equal(f.ui.pendingQuestion.value.requestId,id);assert.equal(f.ui.pendingQuestion.value.question,'');assert.equal(f.ui.messages.value.length,0);
  await f.ui.continuePendingQuestion();await f.ui.checkPendingQuestion();assert.equal(queries,0);assert.equal(f.ui.pendingQuestion.value.status,'unknown');f.dispose();
});
test('late status response cannot alter a hidden page or restore another identity',async()=>{
  const wait=deferred(),id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  const f=mount('chat',{api:{pendingQueries:()=>[{request_id:id,request_status:'unknown'}],queryStatus:()=>wait.promise}});
  f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'}});f.hooks.show();await settle();const checking=f.ui.checkPendingQuestion();
  f.hooks.hide();f.preferences({inviteUser:{id:'b'}});f.hooks.show();await settle();wait.resolve({request_id:id,request_status:'completed',answer:'Previous user answer'});await checking;
  assert.equal(f.ui.messages.value.length,0);assert.equal(f.ui.requestNotice.value,'');assert.equal(f.ui.checkingRequest.value,false);f.dispose();
});
test('starting a new question requires an explicit confirmation and never resends or cancels the old one',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';let confirmations=0,confirm=false;
  const f=mount('chat',{api:{pendingQueries:()=>[{request_id:id,request_status:'unknown'}]},invite:{uni:{showModal:options=>{confirmations++;options.success({confirm});}}}});
  f.preferences({mode:'connected',connectionKind:'invite',inviteUser:{id:'a'}});f.hooks.show();await settle();
  await f.ui.releasePendingQuestion();assert.equal(f.ui.pendingQuestion.value.requestId,id);assert.equal(f.calls.ask.length,0);
  confirm=true;await f.ui.releasePendingQuestion();assert.equal(f.ui.pendingQuestion.value,null);assert.equal(confirmations,2);assert.equal(f.calls.ask.length,0);assert.match(f.ui.requestNotice.value,/没有取消或重发/);f.dispose();
});

test('assistant starter prompts only fill an editable draft and make no task or memory claims', async () => {
  const f = mount('chat'); f.hooks.show(); await settle();
  f.ui.fillDraft('先梳理我的目标和困难。'); assert.equal(f.ui.draft.value,'先梳理我的目标和困难。');
  assert.equal(f.calls.ask.length,0); assert.equal(f.calls.start,0);
  const source = readFileSync(new URL('pages/chat/index.vue',root),'utf8');
  assert.match(source,/原文支持吗/); assert.match(source,/一起做研究/);
  assert.doesNotMatch(source,/从资料，/); f.dispose();
});

test('starter preserves an occupied draft until an explicit append, replace or cancel decision', async () => {
  const f = mount('chat'); f.hooks.show(); await settle();
  f.ui.draft.value = '我的原始想法'; f.ui.fillDraft('快捷起草');
  assert.equal(f.ui.draft.value,'我的原始想法'); assert.equal(f.ui.draftSuggestion.value,'快捷起草');
  assert.equal(f.ui.canSend.value,false); await f.ui.send(); assert.equal(f.calls.ask.length,0);
  f.ui.resolveDraftSuggestion('cancel'); assert.equal(f.ui.draft.value,'我的原始想法'); assert.equal(f.ui.draftSuggestion.value,'');
  f.ui.fillDraft('追加问题'); f.ui.draft.value += '，仍然可编辑'; f.ui.resolveDraftSuggestion('append');
  assert.equal(f.ui.draft.value,'我的原始想法，仍然可编辑\n\n追加问题');
  f.ui.fillDraft('显式替换'); assert.notEqual(f.ui.draft.value,'显式替换'); f.ui.resolveDraftSuggestion('replace');
  assert.equal(f.ui.draft.value,'显式替换'); assert.equal(f.calls.ask.length,0); assert.equal(f.calls.start,0); f.dispose();
});

test('appending a starter never truncates an oversized draft or submits it', async () => {
  const f=mount('chat'); f.hooks.show(); await settle();
  const original='🌱'.repeat(5998); f.ui.draft.value=original; f.ui.fillDraft('完整保留'); f.ui.resolveDraftSuggestion('append');
  assert.equal(f.ui.draft.value,original+'\n\n完整保留'); assert.equal(f.ui.draftOverLimit.value,true);
  assert.equal(f.ui.canSend.value,false); await f.ui.send(); assert.equal(f.calls.ask.length,0); f.dispose();
});

test('starter cannot mutate audio or unresolved query state, and leaving clears only its pending choice', async () => {
  const f=mount('chat'); f.hooks.show(); await settle(); f.ui.draft.value='保留草稿';
  for(const flag of ['asking','recording','audioReady','voiceBusy','loading','creating']) {
    f.ui[flag].value=true; f.ui.fillDraft('不能替换'); assert.equal(f.ui.draftSuggestion.value,''); assert.equal(f.ui.draft.value,'保留草稿'); f.ui[flag].value=false;
  }
  f.ui.pendingQuestion.value={status:'unknown'}; f.ui.fillDraft('不能替换'); assert.equal(f.ui.draftSuggestion.value,'');
  f.ui.pendingQuestion.value=null; f.ui.fillDraft('等待选择'); f.hooks.hide(); await settle();
  assert.equal(f.ui.draftSuggestion.value,''); assert.equal(f.ui.composerFocused.value,false); assert.equal(f.ui.draft.value,'保留草稿');
  f.ui.fillDraft('隐藏页面不得改变'); assert.equal(f.ui.draft.value,'保留草稿'); assert.equal(f.calls.ask.length,0); f.dispose();
});

test('keyboard send requires explicit activation, respects busy state and cannot send from a hidden page', async () => {
  const f=mount('chat'); f.hooks.show(); await settle(); f.ui.draft.value='键盘确认问题'; let prevented=0;
  for(const event of [{key:'a'},{key:'Enter',repeat:true},{key:'Enter',ctrlKey:true},{key:' ',metaKey:true}]) f.ui.sendFromKey({...event,preventDefault:()=>prevented++});
  assert.equal(f.calls.ask.length,0); assert.equal(prevented,0);
  f.ui.asking.value=true; f.ui.sendFromKey({key:'Enter',preventDefault:()=>prevented++}); assert.equal(f.calls.ask.length,0); f.ui.asking.value=false;
  f.ui.sendFromKey({key:'Enter',preventDefault:()=>prevented++}); await settle(); assert.equal(f.calls.ask.length,1);
  f.ui.draft.value='隐藏后不要发送'; f.hooks.hide(); f.ui.sendFromKey({key:' ',preventDefault:()=>prevented++}); await f.ui.send(); await settle();
  assert.equal(f.calls.ask.length,1); f.dispose();
});

test('document file selection is cancelled before upload after the page leaves', async () => {
  let checkCurrent;
  const pending = deferred();
  const f = mount('documents',{api:{upload:async (_sid,options) => {checkCurrent = options.isCurrent; return pending.promise;}}});
  f.hooks.show(); await settle();
  const upload = f.ui.upload(); assert.equal(checkCurrent(),true);
  f.hooks.hide(); assert.equal(checkCurrent(),false);
  pending.resolve({status:'success'}); await upload;
  assert.equal(f.ui.notice.value,''); f.dispose();
});

test('settings describe development privacy limitations instead of promising incognito', () => {
  const source = readFileSync(new URL('pages/settings/index.vue',root),'utf8');
  assert.match(source,/关闭记忆不等于无痕/); assert.match(source,/删除会话不等于彻底删除/);
  assert.match(source,/私人部署不等于完全离线/); assert.match(source,/游客模式/);
});

test('connection diagnostics are manual, saved-connected only and do not save or ask a question', async () => {
  const f = mount('settings'); f.hooks.show(); await settle();
  assert.equal(f.ui.canCheckConnection.value,false); await f.ui.testConnection();
  assert.equal(f.calls.connectionChecks,0);
  f.ui.form.value = {...f.ui.form.value,mode:'connected',baseUrl:'https://private.example',accessToken:'private-code'};
  assert.equal(f.ui.canCheckConnection.value,false); await f.ui.testConnection();
  assert.equal(f.calls.connectionChecks,0);
  await f.ui.save(); assert.equal(f.calls.connectionChecks,0); assert.equal(f.ui.canCheckConnection.value,true);
  await f.ui.testConnection(); assert.equal(f.calls.connectionChecks,1); assert.equal(f.calls.save,1);
  assert.match(f.ui.connectionNotice.value,/服务连通，访问码验证通过/);
  assert.match(f.ui.connectionNotice.value,/不代表模型.*多用户隔离已就绪/);
  assert.equal(f.calls.ask.length,0); assert.equal(f.calls.start,0);
  f.dispose();
});

test('settings connection check is singleflight and drops the result when hidden', async () => {
  const pending = deferred(); let count = 0, current;
  const f = mount('settings',{api:{checkConnection:options => {count += 1; current = options.isCurrent; return pending.promise;}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'code'}); f.hooks.show();
  const check = f.ui.testConnection(); await f.ui.testConnection();
  assert.equal(count,1); assert.equal(f.ui.checking.value,true); assert.equal(current(),true);
  f.hooks.hide(); assert.equal(current(),false); assert.equal(f.ui.canCheckConnection.value,false);
  pending.resolve({connected:true}); await check;
  assert.equal(f.ui.connectionNotice.value,''); assert.equal(f.ui.checking.value,false); f.dispose();
});

for (const key of ['mode','baseUrl','accessToken','useMemory']) {
  test(`editing ${key} invalidates a connection check even if the value is later restored`, async () => {
    const pending = deferred(); let current;
    const f = mount('settings',{api:{checkConnection:options => {current = options.isCurrent; return pending.promise;}}});
    f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'code'}); f.hooks.show();
    const check = f.ui.testConnection(), original = f.ui.form.value[key];
    f.ui.form.value[key] = key === 'mode' ? 'demo' : key === 'baseUrl' ? 'https://another.example' : key === 'accessToken' ? 'different-code' : false;
    assert.equal(current(),false); assert.equal(f.ui.canCheckConnection.value,false);
    f.ui.form.value[key] = original;
    assert.equal(current(),false); pending.resolve({connected:true}); await check;
    assert.equal(f.ui.connectionNotice.value,''); assert.equal(f.ui.connectionError.value,''); f.dispose();
  });
}

test('saving invalidates an in-flight check and never starts another automatically', async () => {
  const pending = deferred(); let count = 0;
  const f = mount('settings',{api:{checkConnection:() => {count += 1; return pending.promise;}}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'code'}); f.hooks.show();
  const check = f.ui.testConnection(); await f.ui.save(); pending.resolve({connected:true}); await check;
  assert.equal(f.calls.save,1); assert.equal(count,1); assert.equal(f.ui.connectionNotice.value,''); f.dispose();
});

test('connection failure UI accepts only fixed reason labels, never remote error text', async () => {
  const f = mount('settings',{api:{checkConnection:async () => ({connected:false,reason:'timeout',message:'Bearer SECRET https://host?token=SECRET'})}});
  f.preferences({mode:'connected',baseUrl:'https://private.example',accessToken:'code'}); f.hooks.show();
  await f.ui.testConnection(); assert.match(f.ui.connectionError.value,/8 秒/);
  assert.doesNotMatch(f.ui.connectionError.value,/SECRET|https:\/\//);
  f.api.checkConnection = async () => {throw new Error('secret response');};
  await f.ui.testConnection(); assert.doesNotMatch(f.ui.connectionError.value,/secret/);
  const source = readFileSync(new URL('pages/settings/index.vue',root),'utf8');
  assert.match(source,/:disabled="!canCheckConnection"/); assert.match(source,/检查连接（不调用模型）/); f.dispose();
});

test('late history cannot overwrite the newly selected session', async () => {
  const pending = deferred(); const f = mount('chat', { api: { history: id => id === 'a' ? pending.promise : Promise.resolve([{ role: 'assistant', content: 'Only B' }]) } });
  f.hooks.show(); await settle(); await f.ui.changeSession('b');
  pending.resolve([{ role: 'assistant', content: 'Old A' }]); await settle();
  assert.equal(f.ui.sessionId.value, 'b'); assert.equal(f.ui.messages.value[0].content, 'Only B'); f.dispose();
});

test('switching projects protects an unsent draft until explicit discard, without model calls', async () => {
  let confirm=false, dialogs=0;
  const f=mount('chat',{invite:{uni:{showModal:options=>{dialogs++;options.success({confirm});}}}});
  f.hooks.show();await settle();f.ui.draft.value='My unfinished work';
  await f.ui.changeSession('b');assert.equal(f.ui.sessionId.value,'a');assert.equal(f.ui.draft.value,'My unfinished work');
  confirm=true;await f.ui.changeSession('b');assert.equal(f.ui.sessionId.value,'b');assert.equal(f.ui.draft.value,'');
  assert.equal(dialogs,2);assert.equal(f.calls.ask.length,0);f.dispose();
});

test('unsaved project brief also protects navigation, and missing confirmation fails closed', async () => {
  const f=mount('chat');f.hooks.show();await settle();f.ui.workbenchDirty.value=true;
  await f.ui.changeSession('b');assert.equal(f.ui.sessionId.value,'a');await f.ui.createSession();assert.equal(f.ui.sessionId.value,'a');
  assert.equal(f.calls.ask.length,0);f.dispose();
});

test('late navigation confirmation cannot change hidden pages or a changed service', async () => {
  for (const action of ['hide','identity']) {
    let modal, created=0;
    const f=mount('chat',{api:{createSession:async()=>{created++;return {id:'b'};}},invite:{uni:{showModal:options=>{modal=options;}}}});
    f.hooks.show();await settle();f.ui.draft.value='Original';
    const changing=f.ui.changeSession('b');assert.equal(f.ui.switching.value,true);assert.equal(f.ui.canSend.value,false);
    await f.ui.changeSession('b');await f.ui.createSession();await f.ui.send();
    if(action==='hide')f.hooks.hide();else f.preferences({baseUrl:'https://new.example'});
    modal.success({confirm:true});await changing;
    assert.equal(f.ui.sessionId.value,'a');assert.equal(f.ui.draft.value,'Original');assert.equal(created,0);assert.equal(f.calls.ask.length,0);f.dispose();
  }
});

test('creating a first conversation preserves the first draft and does not automatically send', async () => {
  let created=false;
  const f=mount('chat',{api:{sessions:async()=>created?[{id:'b'}]:[],createSession:async()=>{created=true;return {id:'b'};}}});
  f.hooks.show();await settle();f.ui.draft.value='The first question';assert.match(f.ui.sendBlockReason.value,/先点「新建」/);
  await f.ui.createSession();assert.equal(f.ui.sessionId.value,'b');assert.equal(f.ui.draft.value,'The first question');assert.equal(f.calls.ask.length,0);f.dispose();
});

test('composer shortcut requires Ctrl or Meta Enter and never sends while Chinese IME is composing', async () => {
  const f=mount('chat');f.hooks.show();await settle();f.ui.draft.value='中文输入';let prevented=0;
  const key=(values={})=>({key:'Enter',preventDefault:()=>prevented++,...values});
  for(const event of [key(),key({ctrlKey:true,isComposing:true}),key({metaKey:true,keyCode:229}),key({ctrlKey:true,repeat:true}),key({ctrlKey:true,shiftKey:true})])f.ui.composerKeydown(event);
  assert.equal(f.calls.ask.length,0);assert.equal(prevented,0);
  f.ui.composerKeydown(key({metaKey:true}));await settle();assert.equal(f.calls.ask.length,1);assert.equal(prevented,1);
  f.ui.draft.value='Do not send while hidden';f.hooks.hide();f.ui.composerKeydown(key({ctrlKey:true}));await settle();assert.equal(f.calls.ask.length,1);f.dispose();
});

test('draft survives returning to the same session but never crosses a service, mode, or session', async () => {
  const f = mount('chat'); f.hooks.show(); await settle(); f.ui.draft.value = 'Private A draft';
  f.hooks.hide(); f.hooks.show(); await settle(); assert.equal(f.ui.draft.value, 'Private A draft');
  f.hooks.hide(); f.preferences({ mode: 'connected', baseUrl: 'https://one.example', accessToken: 'secret' }); f.hooks.show(); await settle(); assert.equal(f.ui.draft.value, '');
  f.ui.draft.value = 'Only for one.example'; f.hooks.hide(); f.preferences({ baseUrl: 'https://two.example' }); f.hooks.show(); await settle(); assert.equal(f.ui.draft.value, '');
  f.ui.draft.value = 'Only session A'; f.hooks.hide(); f.active('b'); f.hooks.show(); await settle(); assert.equal(f.ui.draft.value, ''); assert.equal(f.calls.ask.length, 0); f.dispose();
});

test('session load failure clears the old active target and prevents sending', async () => {
  const f = mount('chat'); f.hooks.show(); await settle(); f.ui.draft.value = 'Old draft';
  f.api.sessions = async () => { throw new Error('Cannot read sessions'); }; await f.ui.reload();
  assert.equal(f.ui.sessionId.value, ''); assert.equal(f.ui.canSend.value, false); assert.equal(f.ui.draft.value, 'Old draft'); f.dispose();
});

test('same-named entities in different sessions never reuse stale evidence', async () => {
  const pending = deferred(); const f = mount('graph', { api: { chunks: id => id === 'a' ? pending.promise : Promise.resolve([{ content: 'B evidence' }]) } });
  f.hooks.show(); await settle(); const old = f.ui.selectNode(f.ui.nodes.value[0]);
  await f.ui.changeSession('b'); await f.ui.selectNode(f.ui.nodes.value[0]);
  pending.resolve([{ content: 'A evidence' }]); await old;
  assert.equal(f.ui.chunks.value[0].content, 'B evidence'); f.dispose();
});

test('documents expose unknown statuses instead of claiming successful parsing', async () => {
  const f = mount('documents'); f.hooks.show(); await settle();
  assert.equal(f.ui.files.value[0].filename, 'Demo.pdf'); assert.equal(f.ui.statusLabel('completed'), '可提问');
  assert.equal(f.ui.statusLabel('unknown-status'), '待确认'); assert.equal(f.ui.statusLabel('failed'), '解析失败'); f.dispose();
});

test('document summaries respect processing_status and never invent a ready count', async () => {
  const f = mount('documents', { api: { files: async () => [
    { filename: 'report.pdf', processing_status: 'completed', file_size: 2048 },
    { filename: 'notes.md', status: 'processing' },
    { filename: 'unknown', status: 'unrecognized' },
  ] } });
  f.hooks.show(); await settle();
  assert.equal(f.ui.readyCount.value, 1);
  assert.equal(f.ui.fileExtension(f.ui.files.value[0]), 'PDF');
  assert.equal(f.ui.fileSize(f.ui.files.value[0]), '2 KB');
  assert.equal(f.ui.fileSize({ file_size: -1 }), '');
  assert.equal(f.ui.fileExtension({ filename: 'no extension' }), 'FILE');
  f.dispose();
});

test('graph adjacency uses actual endpoints, distinguishes inferred edges, and follows neighbors', async () => {
  const evidenceRequests = [];
  const f = mount('graph', { api: {
    graph: async () => ({ nodes: [
      { id: 'a', label: '气候目标' }, { id: 'b', label: '政策工具', is_discovered: 'false' },
      { id: 'c', label: '补充线索', is_discovered: true }, { id: 'd', label: '另一个实体' },
    ], edges: [
      { source: 'a', target: 'b', description: '文档中的关联', is_discovered: false },
      { source: { id: 'c' }, target: { id: 'a' }, description: '待核验的补充', is_discovered: true },
      { source: 'c', target: 'd', description: '与当前节点无关' },
      { source: 'a', target: '', description: '缺失端点' },
    ] }),
    chunks: async (sid, nodeId) => { evidenceRequests.push([sid, nodeId]); return [{ content: nodeId }]; },
  } });
  f.hooks.show(); await settle(); await f.ui.selectNode(f.ui.nodes.value[0]);
  assert.equal(f.ui.edgeCount.value, 3);
  assert.deepEqual(f.ui.selectedRelations.value.map(item => item.label), ['政策工具', '补充线索']);
  assert.deepEqual(f.ui.selectedRelations.value.map(item => item.is_discovered), [false, true]);
  assert.equal(f.ui.nodes.value[1].is_discovered, false);
  await f.ui.selectNeighbor(f.ui.selectedRelations.value[0]);
  assert.equal(f.ui.selected.value.id, 'b');
  assert.deepEqual(evidenceRequests, [['a', 'a'], ['a', 'b']]);
  const source = readFileSync(new URL('pages/graph/index.vue', root), 'utf8');
  assert.match(source, /不表示因果方向/);
  assert.match(source, /不一定支持每一条关系/);
  f.dispose();
});

test('graph limits adjacency rendering and discards old relations across sessions', async () => {
  const f = mount('graph', { api: { graph: async sid => sid === 'a' ? {
    nodes: [{ id: 'center', label: '中心' }],
    edges: Array.from({ length: 12 }, (_, i) => ({ source: 'center', target: `outside-${i}` })),
  } : { nodes: [{ id: 'center', label: '同名但不同会话' }], edges: [] } } });
  f.hooks.show(); await settle(); await f.ui.selectNode(f.ui.nodes.value[0]);
  assert.equal(f.ui.selectedRelations.value.length, 12); assert.equal(f.ui.displayedRelations.value.length, 8);
  assert.equal(f.ui.selectedRelations.value[0].node, undefined);
  await f.ui.selectNeighbor(f.ui.selectedRelations.value[0]);
  assert.equal(f.ui.selected.value.id, 'center');
  f.ui.relationLimit.value += 8; assert.equal(f.ui.displayedRelations.value.length, 12);
  await f.ui.changeSession('b');
  assert.equal(f.ui.selected.value, null); assert.equal(f.ui.selectedRelations.value.length, 0);
  assert.equal(f.ui.edges.value.length, 0); assert.equal(f.ui.relationLimit.value, 8);
  f.dispose();
});

test('secondary pages use the shared brand and keep installation instructions H5-only', () => {
  for (const page of ['documents', 'graph', 'settings']) {
    const source = readFileSync(new URL(`pages/${page}/index.vue`, root), 'utf8');
    assert.match(source, /<BrandHeader eyebrow=/);
  }
  const settings = readFileSync(new URL('pages/settings/index.vue', root), 'utf8');
  assert.match(settings, /<!-- #ifdef H5 -->\s*<InstallGuide :invitation="savedPreferences\.connectionKind === 'invite'" \/>\s*<!-- #endif -->/);
  assert.match(settings, /\/\/ #ifdef H5\s*import InstallGuide/);
  assert.match(settings, /不是模型厂商 API Key/);
  const install = readFileSync(new URL('components/InstallGuide.vue', root), 'utf8');
  assert.match(install, /邀请账号重新打开后仍须联网登录/);
  assert.match(install, /不会缓存你的对话，也不会自动切成演示回答/);
});

test('settings validate a private origin and code, save only on request, and can clear the code', async () => {
  const f = mount('settings'); f.hooks.show();
  f.ui.form.value.mode = 'connected'; f.ui.form.value.baseUrl = 'https://private.example/api'; f.ui.form.value.accessToken = 'private-code';
  await f.ui.save(); assert.equal(f.calls.save, 0); assert.match(f.ui.error.value, /不含 \/api/);
  f.ui.form.value.baseUrl = 'https://private.example'; await f.ui.save(); assert.equal(f.calls.save, 1); assert.equal(f.ui.savedPreferences.value.mode, 'connected');
  assert.equal(f.calls.ask.length, 0); await f.ui.returnToDemo(); assert.equal(f.ui.savedPreferences.value.accessToken, ''); assert.equal(f.ui.savedPreferences.value.mode, 'demo'); f.dispose();
});
