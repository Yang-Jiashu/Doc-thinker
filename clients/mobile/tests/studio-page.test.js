import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, nextTick, effectScope } from 'vue';
import { firstValueCase } from '../src/services/first-value-case.js';

function mount(t, options={}) {
  const filename=new URL('../src/pages/studio/index.vue',import.meta.url);
  const {descriptor,errors}=parse(readFileSync(filename,'utf8'),{filename:String(filename)});
  assert.deepEqual(errors,[]);
  const script=compileScript(descriptor,{id:'first-value'});
  const template=compileTemplate({source:descriptor.template.content,filename:String(filename),id:'first-value',compilerOptions:{bindingMetadata:script.bindings}});
  assert.deepEqual(template.errors,[]);
  const hooks={},calls={save:[],clear:[],scroll:[],navigation:[],clipboard:[]};
  let stored=options.stored || null, preferences={mode:'demo',...options.preferences},session=options.session || '';
  const scope={ref,computed,nextTick,firstValueCase,BrandHeader:{},
    getPreferences:()=>({...preferences}),getActiveSession:()=>session,
    loadFirstValueProgress:()=>stored,
    saveFirstValueProgress:record=>{calls.save.push(record);if(options.saveFails)return {ok:false};stored={...record,savedAt:Date.now()};return {ok:true,record:stored};},
    clearFirstValueProgress:record=>{calls.clear.push(record);if(options.clearFails)return {ok:false};stored=null;return {ok:true};},
    onShow:fn=>hooks.show=fn,onHide:fn=>hooks.hide=fn,onUnload:fn=>hooks.unload=fn,
    uni:{pageScrollTo:value=>calls.scroll.push(value),switchTab:value=>calls.navigation.push(value),navigateTo:value=>calls.navigation.push(value),setClipboardData:value=>calls.clipboard.push(value)},
  };
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope));
  const effects=effectScope();const ui=effects.run(()=>component.setup({},{expose(){}}));
  t.after(()=>{hooks.unload?.();effects.stop();});
  hooks.show();
  return {ui,hooks,calls,template:descriptor.template.content,setPreferences:value=>preferences={...value},setSession:value=>session=value};
}
test('first use is a real-source curated sample without model calls or automatic persistence',t=>{
  const f=mount(t);
  assert.equal(f.ui.sample,firstValueCase);assert.equal(f.ui.activeSource.value,null);
  assert.deepEqual(f.calls.save,[]);assert.deepEqual(f.calls.navigation,[]);
  assert.match(f.template,/真实论文 · 预先整理 · 不调用模型/);
  assert.match(f.template,/不是当前模型现场生成/);
  assert.match(f.template,/不写入你的项目记忆/);
  assert.doesNotMatch(f.template,/切换材料组织示例|以下是例子，不是聊天模式|v-html/);
  const source=readFileSync(new URL('../src/pages/studio/index.vue',import.meta.url),'utf8');
  assert.doesNotMatch(source,/api\.(ask|upload|createSession)|savePreferences|research\./);
});
test('one action reveals a precise source and the selected route survives a return',async t=>{
  const f=mount(t);f.ui.inspectFirst();await nextTick();
  assert.equal(f.ui.activeStep.value.id,'context-position');
  assert.equal(f.ui.activeSource.value.id,'lost-middle-2023');
  assert.match(f.ui.activeSource.value.locator,/Abstract/);
  assert.equal(f.calls.scroll.at(-1).selector,'#sample-source');
  f.ui.backToRoute();await nextTick();
  assert.equal(f.calls.scroll.at(-1).selector,'#sample-route');
  assert.equal(f.ui.activeStep.value.id,'context-position');
  assert.deepEqual(f.calls.save,[]);
});
test('synthesis exposes both sources and a source from another step cannot leak in',t=>{
  const f=mount(t);f.ui.selectStep('bounded-conclusion');
  assert.equal(f.ui.stepSources.value.length,2);
  f.ui.selectSource('lost-middle-2023');assert.equal(f.ui.activeSource.value.id,'lost-middle-2023');
  f.ui.selectSource('arbitrary');assert.equal(f.ui.activeSource.value.id,'lost-middle-2023');
  f.ui.selectStep('retrieved-evidence');assert.equal(f.ui.activeSource.value.id,'rag-2020');
  f.ui.selectSource('lost-middle-2023');assert.equal(f.ui.activeSource.value.id,'rag-2020');
  f.ui.selectStep('unknown');assert.equal(f.ui.activeStep.value.id,'retrieved-evidence');
});
test('save is explicit, bounded to sample ids and resumes in a fresh page without account memory',t=>{
  const f=mount(t);f.ui.savePlace();assert.equal(f.calls.save.length,0);
  f.ui.selectStep('bounded-conclusion');f.ui.selectSource('lost-middle-2023');f.ui.savePlace();
  assert.deepEqual(f.calls.save,[{caseId:firstValueCase.id,version:firstValueCase.version,stepId:'bounded-conclusion',sourceId:'lost-middle-2023'}]);
  const g=mount(t,{stored:f.ui.savedProgress.value});assert.equal(g.ui.activeSource.value,null);
  g.ui.resume();assert.equal(g.ui.activeSource.value.id,'lost-middle-2023');assert.equal(g.ui.activeStep.value.id,'bounded-conclusion');
  assert.equal(g.calls.save.length,0);g.ui.forgetPlace();assert.equal(g.ui.savedProgress.value,null);
});
test('invalid membership in saved progress is ignored',t=>{
  const f=mount(t,{stored:{caseId:firstValueCase.id,version:1,stepId:'retrieved-evidence',sourceId:'lost-middle-2023'}});
  assert.equal(f.ui.savedProgress.value,null);f.ui.resume();assert.equal(f.ui.activeSource.value,null);
});
test('storage failures are visible and never reported as successful persistence',t=>{
  const f=mount(t,{saveFails:true});f.ui.inspectFirst();f.ui.savePlace();assert.equal(f.ui.savedProgress.value,null);assert.match(f.ui.notice.value,/未能保存/);
  const g=mount(t,{stored:{caseId:firstValueCase.id,version:1,stepId:'context-position',sourceId:'lost-middle-2023'},clearFails:true});
  g.ui.forgetPlace();assert.ok(g.ui.savedProgress.value);assert.match(g.ui.notice.value,/未能清除/);
});
test('own materials handoff does not disguise demo upload as real or bypass invitation login',t=>{
  for(const [preferences,session,route] of [
    [{mode:'demo'},'','/pages/settings/index'],
    [{mode:'connected',connectionKind:'invite',invitePending:true},'','/pages/invite/index'],
    [{mode:'connected'},'','/pages/chat/index'],
    [{mode:'connected'},'#00001','/pages/documents/index'],
    [{mode:'connected',connectionKind:'invite',invitePending:false},'#00001','/pages/documents/index'],
  ]){const f=mount(t,{preferences,session});f.ui.continueWithOwn();assert.equal(f.calls.navigation.length,0);f.ui.showOwnMaterials();assert.equal(f.calls.navigation.length,0);f.ui.continueWithOwn();assert.deepEqual(f.calls.navigation,[{url:route}]);}
});
test('handoff rechecks changed identity instead of using stale destination',t=>{
  const f=mount(t,{preferences:{mode:'connected'},session:'#00001'});f.ui.showOwnMaterials();
  f.setPreferences({mode:'connected',connectionKind:'invite',invitePending:true});f.ui.continueWithOwn();
  assert.deepEqual(f.calls.navigation,[{url:'/pages/invite/index'}]);assert.equal(f.calls.save.length,0);
});
test('hidden page cannot navigate, save or revive clipboard notices',async t=>{
  const f=mount(t);f.ui.inspectFirst();f.ui.copySourceLink();f.ui.showOwnMaterials();f.hooks.hide();
  f.calls.clipboard[0].success();f.calls.clipboard[0].fail();f.ui.savePlace();f.ui.continueWithOwn();f.ui.openChat();f.ui.selectStep('retrieved-evidence');await nextTick();
  assert.equal(f.ui.notice.value,'');assert.deepEqual(f.calls.save,[]);assert.deepEqual(f.calls.navigation,[]);assert.deepEqual(f.calls.scroll,[]);
  assert.equal(f.ui.activeStep.value.id,'context-position');f.hooks.show();f.ui.openChat();assert.deepEqual(f.calls.navigation,[{url:'/pages/chat/index'}]);
});
test('source changes invalidate delayed clipboard notices',t=>{
  const f=mount(t);f.ui.inspectFirst();f.ui.copySourceLink();f.ui.selectStep('retrieved-evidence');f.calls.clipboard[0].success();assert.equal(f.ui.notice.value,'');
  assert.match(f.template,/target="_blank" rel="noopener noreferrer"/);
});
test('keyboard activation supports Enter and Space but not IME, repeated or modified keys',t=>{
  const f=mount(t);let actions=0,prevented=0;
  for(const event of [{key:'Enter'},{key:' '},{key:'Spacebar'}])f.ui.activate({...event,preventDefault:()=>prevented++},()=>actions++);
  for(const event of [{key:'Tab'},{key:'Enter',repeat:true},{key:'Enter',isComposing:true},{key:'Enter',metaKey:true},{key:'Enter',ctrlKey:true},{key:' ',altKey:true}])f.ui.activate(event,()=>actions++);
  assert.equal(actions,3);assert.equal(prevented,3);
});
