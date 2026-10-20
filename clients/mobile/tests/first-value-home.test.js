import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, nextTick, effectScope } from 'vue';
import {buildEvidenceCheckQuestion} from '../src/services/evidence-check.js';
import {boundedAnswerSources} from '../src/services/research-source-contract.js';
const source=readFileSync(new URL('../src/pages/chat/index.vue',import.meta.url),'utf8');
const settle=()=>new Promise(resolve=>setImmediate(resolve));
function mount(t,{mode='connected',connectionKind='private',enabled=true}={}) {
  const {descriptor,errors}=parse(source);assert.deepEqual(errors,[]);
  const script=compileScript(descriptor,{id:'first-value-home'});
  assert.deepEqual(compileTemplate({source:descriptor.template.content,filename:'chat.vue',id:'first-value-home',compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);
  const hooks={},calls={navigation:[],ask:[],create:0,history:0};let active='a';
  const components=Object.fromEntries([...script.content.matchAll(/^import (\w+) from ['"][^'"]+\.vue['"];$/gm)].map(match=>[match[1],{}]));
  const scope={...components,ref,computed,nextTick,buildEvidenceCheckQuestion,boundedAnswerSources,
    api:{sessions:async()=>[{id:'a',title:'User project'}],history:async()=>{calls.history++;return [];},ask:async(sid,q,opts)=>{calls.ask.push({sid,q,opts});return {answer:'Answer from service',request_id:opts?.requestId,request_status:'completed'};},createSession:async()=>{calls.create++;return{id:'b'};}},
    research:{capabilities:async()=>({enabled,memory_write_enabled:enabled,uploads_allowed:true})},
    getPreferences:()=>({mode,connectionKind,baseUrl:'https://offline.invalid',useMemory:true}),getActiveSession:()=>active,setActiveSession:value=>active=value,
    isFeedbackAnswerId:()=>false,createQueryRequestIdAsync:async()=> '12345678-1234-4234-8234-123456789abc',queryRequestIdSupported:()=>true,
    questionByteLength:value=>new TextEncoder().encode(value).length,INVITE_MAX_QUESTION_BYTES:8192,researchAnswerMetadata:value=>value,researchConnectionKey:()=>`${mode}:${connectionKind}`,
    startRecording:async()=>{},stopRecording:async()=>'',cancelRecording:()=>{},speak:async()=>{},stopSpeaking:()=>{},requestReceipt:()=>null,
    onShow:fn=>hooks.show=fn,onHide:fn=>hooks.hide=fn,onUnload:fn=>hooks.unload=fn,
    uni:{navigateTo:opts=>calls.navigation.push(opts),switchTab:opts=>calls.navigation.push(opts)},
  };
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope));const effects=effectScope();
  const ui=effects.run(()=>component.setup({},{expose(){}}));t.after(()=>{hooks.unload?.();effects.stop();});
  return {ui,hooks,calls,template:descriptor.template.content};
}
test('first use accepts the user own materials and question, not a fixed task or demo answer',t=>{
  const f=mount(t);assert.match(f.template,/原文支持吗/);assert.match(f.template,/<EvidenceIntake/);assert.match(f.template,/v-model="draft"/);
  assert.doesNotMatch(f.template,/30 秒|预先整理|first-value-example|openStudio|更多资料交给 AI/);
  assert.deepEqual(f.calls.ask,[]);assert.deepEqual(f.calls.navigation,[]);
});
test('explicit recipe preserves the original judgment in a visible editable draft and never sends itself',async t=>{
  const f=mount(t);f.hooks.show();await settle();const text='我们自己的实验仅在低温下有效？\n不要省略这个限制。';f.ui.draft.value=text;
  f.ui.prepareEvidenceCheck();assert.equal(f.ui.draft.value,buildEvidenceCheckQuestion(text));assert.equal(f.calls.ask.length,0);
  f.ui.prepareEvidenceCheck();assert.equal(f.ui.draft.value,buildEvidenceCheckQuestion(text));
  await f.ui.send();assert.equal(f.calls.ask.length,1);assert.equal(f.calls.ask[0].q,buildEvidenceCheckQuestion(text));assert.ok(f.calls.ask[0].opts.requestId);
  assert.equal(f.calls.ask[0].opts.reasoningMode,undefined);
});
test('intake shares paid-request and navigation gates without being disabled by its own busy state',async t=>{
  const f=mount(t);f.hooks.show();await settle();f.ui.draft.value='Keep my question';f.ui.setEvidenceBusy(true);
  assert.equal(f.ui.networkBusy.value,true);assert.equal(f.ui.canSend.value,false);assert.equal(f.ui.intakeBlocked.value,false);
  await f.ui.send();f.ui.prepareEvidenceCheck();f.ui.navigateEvidence('settings');assert.equal(f.calls.ask.length,0);assert.equal(f.calls.navigation.length,0);assert.equal(f.ui.draft.value,'Keep my question');
  const history=f.calls.history;await f.ui.reload();assert.equal(f.calls.history,history);f.ui.setEvidenceBusy(false);await settle();assert.equal(f.ui.networkBusy.value,false);assert.equal(f.ui.draft.value,'Keep my question');
});
test('demo or disabled service cannot convert a draft into a falsely grounded question',async t=>{
  for(const options of [{mode:'demo'}, {mode:'connected',enabled:false}]){const f=mount(t,options);f.hooks.show();await settle();f.ui.draft.value='My own claim';f.ui.prepareEvidenceCheck();assert.equal(f.ui.draft.value,'My own claim');assert.equal(f.calls.ask.length,0);}
});
test('all busy states and hidden pages prevent drafting or navigation',async t=>{
  const f=mount(t);f.hooks.show();await settle();f.ui.draft.value='Original';
  const states={switching:true,workbenchBusy:true,loading:true,preparingRequest:true,asking:true,creating:true,checkingRequest:true,pendingQuestion:{status:'running'},recording:true,audioReady:true,voiceBusy:true,speakingIndex:0,workbenchDirty:true,draftSuggestion:'Pending choice'};
  for(const [key,value] of Object.entries(states)){const previous=f.ui[key].value;f.ui[key].value=value;f.ui.prepareEvidenceCheck();f.ui.navigateEvidence('settings');assert.equal(f.ui.draft.value,'Original',key);assert.equal(f.calls.navigation.length,0,key);f.ui[key].value=previous;}
  f.hooks.hide();f.ui.prepareEvidenceCheck();assert.equal(f.ui.draft.value,'Original');
});
test('recipe does not hide over-budget text or bypass existing byte limits',async t=>{
  const f=mount(t);f.hooks.show();await settle();const original='论'.repeat(2700);f.ui.draft.value=original;f.ui.prepareEvidenceCheck();assert.ok(f.ui.draft.value.includes(original));assert.equal(f.ui.inviteQuestionOverLimit.value,true);assert.equal(f.ui.canSend.value,false);await f.ui.send();assert.equal(f.calls.ask.length,0);
});
test('unsubmitted original blocks paid sending, remains editable, and participates in leaving confirmation',async t=>{
  const f=mount(t);f.hooks.show();await settle();f.ui.draft.value='Question';f.ui.intakeDirty.value=true;
  assert.equal(f.ui.canSend.value,false);assert.equal(f.ui.intakeBlocked.value,false);assert.match(f.ui.sendBlockReason.value,/未提交的原文/);
  await f.ui.send();assert.equal(f.calls.ask.length,0);assert.equal(await f.ui.confirmLeavingDraft(),false);
  assert.match(f.template,/v-show="!messages.length && !loading"/);assert.match(f.template,/:session-id="intakeSessionId"/);
  f.ui.intakeDirty.value=false;assert.equal(f.ui.canSend.value,true);
});
test('intake retains its stable session during a same-project reload instead of receiving an empty transient id',async t=>{
  const f=mount(t);f.hooks.show();await settle();assert.equal(f.ui.intakeSessionId.value,'a');
  const pending=f.ui.reload();assert.equal(f.ui.sessionId.value,'');assert.equal(f.ui.intakeSessionId.value,'a');await pending;assert.equal(f.ui.intakeSessionId.value,'a');
});
test('keyboard recipe activation excludes IME and modifiers; navigation uses fixed internal destinations',async t=>{
  const f=mount(t);f.hooks.show();await settle();f.ui.draft.value='A claim';
  for(const extra of [{isComposing:true},{keyCode:229},{ctrlKey:true},{metaKey:true},{altKey:true},{shiftKey:true},{repeat:true}])f.ui.firstValueKeydown({key:'Enter',...extra},'check');assert.equal(f.ui.draft.value,'A claim');
  f.ui.firstValueKeydown({key:'Enter',preventDefault(){}},'check');assert.match(f.ui.draft.value,/核对以下问题/);
  f.ui.navigateEvidence('https://bad.invalid');assert.equal(f.calls.navigation.length,0);f.ui.navigateEvidence('settings');assert.equal(f.calls.navigation[0].url,'/pages/settings/index');
});
