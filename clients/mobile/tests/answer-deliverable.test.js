import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { computed, ref, watch, reactive, nextTick, effectScope } from 'vue';
import { answerDeliverable, MAX_RECORD_ANSWER } from '../src/services/answer-deliverable.js';

const source = readFileSync(new URL('../src/components/AnswerDeliverable.vue',import.meta.url),'utf8');
function mount(overrides = {}, action = options=>options.success()) {
  const {descriptor,errors} = parse(source); assert.deepEqual(errors,[]);
  const script = compileScript(descriptor,{id:'answer-deliverable'});
  assert.deepEqual(compileTemplate({source:descriptor.template.content,id:'answer-deliverable',compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);
  const calls = [],hooks = {};
  const props = reactive({answer:'Visible answer',question:'Current question',sources:[],demo:false,disabled:false,...overrides});
  const scope = {computed,ref,watch,answerDeliverable,uni:{setClipboardData:options=>{calls.push(options);action(options);}},
    onHide:fn=>{hooks.hide=fn;},onUnload:fn=>{hooks.unload=fn;}};
  const executable = script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component = new Function(...Object.keys(scope),executable)(...Object.values(scope));
  const effects = effectScope(),ui = effects.run(()=>component.setup(props,{expose(){}}));
  return {ui,props,calls,hooks,dispose:()=>effects.stop()};
}

test('local answer record preserves actual visible question and answer, never invents evidence',()=>{
  const answer='  完整回答\n<script>untrusted literal</script>\n  ';
  const record=answerDeliverable({answer,question:'用户当前问题',sources:[{title:'Invented source',confidence:1,verified:true}]});
  assert.equal(record.available,true);assert.ok(record.text.includes(answer));assert.match(record.text,/用户当前问题/);
  assert.match(record.text,/没有附带可展示/);assert.match(record.text,/并非事实核验/);assert.doesNotMatch(record.text,/Invented source/);
});
test('source fragments are bounded, explicitly unverified and truncation is visible',()=>{
  const record=answerDeliverable({answer:'Answer',sources:Array.from({length:11},(_,i)=>({title:`Source ${i}`,content:'文'.repeat(1200),verified:true}))});
  assert.match(record.text,/尚未逐条核验/);assert.match(record.text,/已截短/);assert.match(record.text,/最多 10 条/);
  assert.ok(record.text.includes('Source 9'));assert.ok(!record.text.includes('Source 10'));assert.ok(record.text.length<10500);
});
test('copied research record carries revoked source warnings rather than losing visible provenance failures',()=>{
  const result=answerDeliverable({answer:'Earlier conclusion [S1]. Remaining evidence [S2].',
    sources:[{id:'fb27e5ae-5cb9-4650-9dfd-000000000002',label:'S2',filename:'Remaining.pdf',content:'Remaining evidence'}],
    citationValidation:{warnings:[{code:'source_revoked'}],unknown_labels:[],missing_citations:false},researchCore:true});
  assert.equal(result.available,true);assert.match(result.text,/来源.*(删除|撤销)|原文.*(删除|撤销)/);
});
test('research copy preserves citation and coverage failures using local labels only',()=>{
  const record=answerDeliverable({answer:'Unverified [S99]',researchCore:true,answerMode:'insufficient_evidence',citationValidation:{missing_citations:true,unknown_labels:['S99'],warnings:[{code:'provenance_unavailable',message:'secret-upstream-warning'}]},retrievalGaps:[{code:'bounded_overview'},{code:'context_budget_omitted'}]});
  assert.match(record.text,/依据记录不可用/);assert.match(record.text,/引用缺失/);assert.match(record.text,/没有请求模型/);assert.match(record.text,/不是完整全文总结/);assert.match(record.text,/部分历史或确认记忆未加入/);assert.doesNotMatch(record.text,/secret-upstream-warning/);
});
test('copied evidence selection limits remain visible without claiming proof or copying remote text',()=>{
  const record=answerDeliverable({answer:'A tentative conclusion [S1]',researchCore:true,
    retrievalGaps:['evidence_coverage_proxy','evidence_facets_uncovered','evidence_anchors_uncovered',
      'complete_statement_exceeds_budget','semantic_retrieval_unavailable','semantic_retrieval_windowed']
      .map(code=>({code,message:'untrusted-upstream-message'}))});
  assert.match(record.text,/部分问题要点还缺少具体原文/);
  assert.match(record.text,/限定条件未能放入/);
  assert.match(record.text,/语义检索暂不可用/);
  assert.match(record.text,/不代表已查完全部资料/);
  assert.match(record.text,/不代表已找齐关键证据或验证结论/);
  assert.doesNotMatch(record.text,/untrusted-upstream-message/);
});
test('provenance changes clear copied preview and ignore old clipboard acknowledgements',async()=>{
  let options;const f=mount({researchCore:true,citationValidation:{warnings:[]}},value=>{options=value;});f.ui.toggle();const copying=f.ui.copy();
  f.props.citationValidation.warnings.push({code:'source_revoked'});await nextTick();options.success();await copying;
  assert.equal(f.ui.expanded.value,false);assert.equal(f.ui.notice.value,'');assert.match(f.ui.record.value.text,/来源.*删除/);f.dispose();
});
test('literal coverage and incomplete premise limits survive copying as local fixed messages',()=>{
  const record=answerDeliverable({answer:'Tentative answer.',researchCore:true,
    retrievalGaps:['uncovered_query_terms','literal_coverage_scan_limited','source_boundary_unverified','source_antecedent_unverified']
      .map(code=>({code,message:'untrusted-provider-message'}))});
  assert.match(record.text,/不代表概念缺失/);assert.match(record.text,/未查完全部资料/);
  assert.match(record.text,/句尾完整也不等于前提完整/);assert.match(record.text,/不能从下一实验猜前提/);
  assert.doesNotMatch(record.text,/untrusted-provider-message/);
});
test('graph budget and qualification boundaries survive export as fixed local text',()=>{
  const record=answerDeliverable({answer:'Tentative idea.',researchCore:true,
    retrievalGaps:['graph_capacity_reached','graph_excerpt_windowed','graph_path_conditions_differ']
      .map(code=>({code,message:'untrusted-graph-payload'}))});
  assert.match(record.text,/不能据此声称推理链完整/);
  assert.match(record.text,/关联不等于因果证明/);
  assert.match(record.text,/否定或条件差异/);
  assert.doesNotMatch(record.text,/untrusted-graph-payload/);
});
test('copy boundary disables oversized records rather than silently truncating their answer',()=>{
  for(const answer of ['',null,{},'🧠'.repeat(MAX_RECORD_ANSWER+1)]) assert.equal(answerDeliverable({answer}).available,false);
  const answer='🧠'.repeat(MAX_RECORD_ANSWER);assert.ok(answerDeliverable({answer}).text.includes(answer));
  assert.equal(answerDeliverable({answer:'Valid',question:'字'.repeat(8001)}).available,false);
  assert.match(answerDeliverable({answer:'Synthetic',demo:true}).text,/【演示】/);
});
test('opening and previewing do not write clipboard; copy is a separate deliberate action',async()=>{
  const f=mount();assert.equal(f.ui.expanded.value,false);assert.equal(f.calls.length,0);
  f.ui.toggle();assert.equal(f.ui.expanded.value,true);assert.equal(f.calls.length,0);
  f.ui.preview.value=true;assert.equal(f.calls.length,0);await f.ui.copy();
  assert.equal(f.calls.length,1);assert.equal(f.calls[0].showToast,false);assert.match(f.calls[0].data,/Visible answer/);
  assert.match(f.ui.notice.value,/已复制/);assert.match(f.ui.notice.value,/不是事实核验/);f.dispose();
});
test('failed or unavailable clipboard keeps selectable text and never reports success or raw errors',async()=>{
  for(const action of [options=>options.fail({errMsg:'secret-key'}),()=>{throw new Error('secret-key');}]) {
    const f=mount({},action);f.ui.toggle();await f.ui.copy();assert.equal(f.calls.length,1);
    assert.equal(f.ui.notice.value,'');assert.match(f.ui.error.value,/手动复制/);assert.doesNotMatch(f.ui.error.value,/secret/);
    assert.equal(f.ui.preview.value,true);f.dispose();
  }
});
test('busy and disabled buttons never perform extra writes',async()=>{
  const f=mount({disabled:true});f.ui.toggle();await f.ui.copy();assert.equal(f.calls.length,0);
  f.props.disabled=false;f.ui.toggle();f.ui.busy.value=true;await f.ui.copy();assert.equal(f.calls.length,0);f.dispose();
});
test('late clipboard confirmation cannot restore another answer or a hidden page',async()=>{
  for(const action of ['answer','hide','unload']) {
    let options;const f=mount({},value=>{options=value;});f.ui.toggle();const copying=f.ui.copy();
    if(action==='answer') f.props.answer='New answer';else f.hooks[action]();
    await nextTick();options.success();await copying;
    assert.equal(f.ui.notice.value,'');assert.equal(f.ui.busy.value,false);assert.equal(f.ui.expanded.value,false);f.dispose();
  }
});
test('record copy has an 8-second deadline and late acknowledgement cannot turn failure into success',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});let options;
  const f=mount({},value=>{options=value;});f.ui.toggle();const copying=f.ui.copy();
  t.mock.timers.tick(8000);await copying;assert.match(f.ui.error.value,/未能确认/);assert.equal(f.ui.busy.value,false);
  options.success();await nextTick();assert.equal(f.ui.notice.value,'');assert.equal(f.calls.length,1);f.dispose();
});
test('demo record is labelled and all platforms use literal selectable text with no network or sharing API',async()=>{
  const f=mount({demo:true});f.ui.toggle();await f.ui.copy();assert.match(f.ui.notice.value,/演示.*没有联网/);
  assert.match(f.calls[0].data,/演示回答记录/);assert.doesNotMatch(source,/v-html|rich-text|api\.|fetch\(|upload|shareApp|localStorage/);
  assert.match(source,/user-select/);assert.match(source,/selectable/);assert.match(source,/role="button"/);f.dispose();
});
