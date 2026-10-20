import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {computed,ref,watch,reactive,nextTick,effectScope} from 'vue';

const source = readFileSync(new URL('../src/components/AnswerFeedback.vue',import.meta.url),'utf8');
const id = 'ed0b7a48-cb27-430d-95f1-02b2757a69cc';
function mount(overrides = {},submit) {
  const {descriptor,errors} = parse(source); assert.deepEqual(errors,[]);
  const script = compileScript(descriptor,{id:'answer-feedback'});
  assert.deepEqual(compileTemplate({source:descriptor.template.content,id:'answer-feedback',compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);
  const calls = [],hooks = {},events=[];
  const props = reactive({sessionId:'#00001',answerId:id,initial:null,demo:false,disabled:false,...overrides});
  const scope = {computed,ref,watch,api:{submitFeedback:async (...args) => {
    calls.push(args); return submit ? submit(...args) : {feedback:{answer_id:args[1],...args[2]}};
  }},onHide:fn=>{hooks.hide=fn;},onUnload:fn=>{hooks.unload=fn;}};
  const executable = script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component = new Function(...Object.keys(scope),executable)(...Object.values(scope));
  const effects = effectScope(),ui = effects.run(()=>component.setup(props,{expose(){},emit:(...args)=>events.push(args)}));
  return {ui,props,calls,hooks,events,dispose:()=>effects.stop()};
}
test('feedback opens only on an explicit choice and only save performs one write',async()=>{
  const f=mount();assert.equal(f.ui.editing.value,false);assert.equal(f.calls.length,0);
  f.ui.choose('inaccurate');f.ui.note.value='漏掉了中间条件';assert.equal(f.calls.length,0);
  await f.ui.save();assert.deepEqual(f.calls,[['#00001',id,{rating:'inaccurate',note:'漏掉了中间条件'}]]);
  assert.equal(f.ui.editing.value,false);assert.match(f.ui.notice.value,/反馈已保存/);
  assert.match(f.ui.notice.value,/不会自动改写/);assert.deepEqual(f.events,[['busy',{answerId:id,active:true}],['busy',{answerId:id,active:false}]]);f.dispose();
});
test('saved history feedback is visible and cancelling an edit restores it without writes',()=>{
  const initial={answer_id:id,rating:'helpful',note:'有清晰的出处'};
  const f=mount({initial});assert.equal(f.ui.rating.value,'helpful');assert.equal(f.ui.note.value,initial.note);
  f.ui.choose('inaccurate');f.ui.note.value='unsaved';f.ui.cancel();
  assert.equal(f.ui.rating.value,'helpful');assert.equal(f.ui.note.value,initial.note);assert.equal(f.calls.length,0);f.dispose();
});
test('Unicode bounds keep every drafted character and prevent oversized submission',async()=>{
  const f=mount();f.ui.choose('inaccurate');f.ui.note.value='🧠'.repeat(501);
  assert.equal(f.ui.noteLength.value,501);assert.equal(f.ui.canSave.value,false);await f.ui.save();assert.equal(f.calls.length,0);
  f.ui.note.value='🧠'.repeat(500);assert.equal(f.ui.canSave.value,true);await f.ui.save();assert.equal(f.calls.length,1);
  assert.match(source,/:maxlength="-1"/);f.dispose();
});
test('unknown response and transport error preserve draft, never report success or echo raw errors',async()=>{
  for(const submit of [async()=>{throw new Error('secret API key');},async()=>({feedback:{answer_id:'other',rating:'inaccurate',note:''}})]) {
    const f=mount({},submit);f.ui.choose('inaccurate');f.ui.note.value='please inspect';await f.ui.save();
    assert.equal(f.ui.editing.value,true);assert.equal(f.ui.note.value,'please inspect');assert.equal(f.ui.notice.value,'');
    assert.match(f.ui.error.value,/未确认保存/);assert.doesNotMatch(f.ui.error.value,/secret/);assert.equal(f.calls.length,1);f.dispose();
  }
});
test('late feedback responses cannot change another answer, session or hidden page',async()=>{
  for(const action of ['answer','session','hide','unload']) {
    let resolve;const pending=new Promise(done=>{resolve=done;});const f=mount({},()=>pending);
    f.ui.choose('inaccurate');const saving=f.ui.save();
    if(action==='answer') f.props.answerId='0e9f63b1-462b-47ca-a43f-3848b285fb56';
    else if(action==='session') f.props.sessionId='#00002';
    else f.hooks[action]();
    await nextTick();resolve({feedback:{answer_id:id,rating:'inaccurate',note:''}});await saving;
    assert.equal(f.ui.notice.value,'');assert.equal(f.ui.busy.value,false);assert.equal(f.ui.editing.value,false);assert.equal(f.ui.note.value,'');f.dispose();
  }
});
test('disabled and busy feedback cannot submit; malformed initial data does not become a confirmed vote',async()=>{
  const f=mount({disabled:true});f.ui.choose('helpful');assert.equal(f.ui.editing.value,false);await f.ui.save();assert.equal(f.calls.length,0);
  f.props.disabled=false;f.ui.choose('helpful');f.ui.busy.value=true;await f.ui.save();assert.equal(f.calls.length,0);f.dispose();
  for(const initial of [{answer_id:'other',rating:'helpful',note:''},{answer_id:id,rating:'unknown',note:''},{answer_id:id,rating:'helpful',note:'x'.repeat(501)}]) {
    const g=mount({initial});assert.equal(g.ui.rating.value,'');assert.equal(g.ui.notice.value,'');g.dispose();
  }
});
test('demo is explicitly local and component cannot modify memory, graph, training or submit questions',async()=>{
  const f=mount({demo:true,answerId:'demo-answer-2'});f.ui.choose('helpful');await f.ui.save();assert.match(f.ui.notice.value,/演示.*没有联网/);
  assert.doesNotMatch(source,/v-html|rich-text|api\.(ask|upload|memory)|fetch\(|request\(/);
  assert.match(source,/不会自动训练模型或修改记忆/);f.dispose();
});
