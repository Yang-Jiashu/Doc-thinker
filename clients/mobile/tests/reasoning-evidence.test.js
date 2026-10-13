import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {ref,computed,watch,reactive,effectScope,nextTick} from 'vue';
import {reasoningEvidence,reasoningRecord,sanitizeReasoningReceipts} from '../src/services/reasoning-evidence.js';
import {researchAnswerMetadata} from '../src/services/api.js';
import {answerDeliverable} from '../src/services/answer-deliverable.js';
const id='c9d9ff40-6ec6-462a-b75f-afdcb1b6d1cd';
const original='🌿 原文明确只观察到相关性。';
const source={id,label:'S1',filename:'研究.txt',page:2,excerpt_start:20,content:original};
const refValue={label:'S1',source_id:id,quote:original,start:20,end:20+Array.from(original).length};
const audit=()=>({status:'quoted_only',claims:[{id:'C1',text:original,status:'exact_quote',source_refs:[refValue],issues:[]}]});
const reasoning=()=>({mode:'analogy',status:'hypothesis',conclusions:[{text:'新场景可能出现相同机制，需要干预验证。',kind:'hypothesis',mappings:[{from:'论文中的共同上游',to:'供应商共同上游'}],differences:['目标场景的替代渠道尚未证实。']}],gaps:[{code:'condition_missing',message:'需要核对供应方是否共享上游。'}]});
const secondId='95e86b7f-341f-4278-8f0a-548733e11fb2';
const secondSource={...source,id:secondId,label:'S2',filename:'关联材料.txt'};
const receipt=()=>({
  graph_retrieval:{status:'expanded',algorithm:'bounded_evidence_graph_v1',bridge_source_ids:[secondId],kept_source_ids:[id,secondId],model_calls:0,max_hops:2,semantic_entailment_verified:false},
  hypothesis_support:{status:'ready',algorithm:'grounded_hypothesis_support_v1',source_ids:[id,secondId],algorithm_modes:['analogy','deduction','induction'],model_calls:0},
});
test('bounded audit omissions remain explicit in the UI and copied record without trusting remote prose',()=>{
  const response={research_core:true,sources:[source],claim_audit:{...audit(),summary:{omitted_statement_count:34,message:'raw-untrusted-audit'}}};
  const metadata=researchAnswerMetadata(response);
  assert.deepEqual(metadata.claim_audit.summary,{omitted_statement_count:34});
  const options={sources:[source],claimAudit:metadata.claim_audit};
  assert.match(reasoningEvidence(options).note,/另有 34 段.*未检查/);
  assert.match(reasoningRecord(options),/另有 34 段.*未检查/);
  assert.doesNotMatch(reasoningRecord(options),/raw-untrusted-audit/);
  for(const omitted_statement_count of [null,undefined,-1,1.2,100001,'34',true]) {
    const unknown=researchAnswerMetadata({...response,claim_audit:{...audit(),summary:{omitted_statement_count}}});
    assert.equal(unknown.claim_audit.summary.omitted_statement_count,null);
    assert.match(reasoningEvidence({...options,claimAudit:unknown.claim_audit}).note,/不能视为全文核对/);
  }
  const withdrawn=reasoningEvidence({...options,citationValidation:{warnings:[{code:'source_revoked'}]}});
  assert.doesNotMatch(withdrawn.note,/34|未检查其引用/);
});
test('citation displacement signals are displayed as review issues, never repaired or promoted to verified',()=>{
  const message='多个英文原词未在所引节选出现，却在本次其他节选出现；请核对具体引用。这仅是词面信号，不是语义判定。';
  const packet={...audit(),status:'review_needed',summary:{omitted_statement_count:0},claims:[{...audit().claims[0],status:'unverified',issues:[{code:'citation_anchor_drift',message}]}]};
  const metadata=researchAnswerMetadata({research_core:true,sources:[source],claim_audit:packet});
  const view=reasoningEvidence({sources:[source],claimAudit:metadata.claim_audit});
  assert.equal(view.claims[0].status,'unverified');assert.equal(view.claims[0].label,'结论待核对');
  assert.deepEqual(view.claims[0].issues,[message]);assert.equal(view.claims[0].refs[0].label,'S1');
  assert.match(reasoningRecord({sources:[source],claimAudit:metadata.claim_audit}),/仅是词面信号/);
});
test('all ten final sources survive graph and hypothesis receipts, but an eleventh is rejected',()=>{
  const sources=Array.from({length:10},(_,index)=>({...source,id:`fb27e5ae-5cb9-4650-9dfd-${String(index+1).padStart(12,'0')}`,label:`S${index+1}`}));
  const ids=sources.map(item=>item.id);
  const trace={
    graph_retrieval:{...receipt().graph_retrieval,kept_source_ids:ids,bridge_source_ids:ids.slice(8)},
    hypothesis_support:{...receipt().hypothesis_support,source_ids:ids},
  };
  assert.deepEqual(sanitizeReasoningReceipts(trace,sources),trace);
  const view=reasoningEvidence({reasoning:{mode:'context_assist',trace},sources});
  assert.deepEqual(view.graphRetrieval.bridgeLabels,['S9','S10']);assert.equal(view.hypothesisSupport.sourceLabels.length,10);
  const extra={...source,id:'fb27e5ae-5cb9-4650-9dfd-000000000011',label:'S11'};
  assert.equal(sanitizeReasoningReceipts({hypothesis_support:{...trace.hypothesis_support,source_ids:[...ids,extra.id]}},[...sources,extra]),undefined);
  assert.equal(sanitizeReasoningReceipts({hypothesis_support:{...trace.hypothesis_support,source_ids:[extra.id]}},[extra]),undefined);
});
test('S9/S10 quote and proof paths retain codepoint checks and source revocation; S11 is not a quote',()=>{
  for(const label of ['S9','S10']) {
    const evidence={...source,label},reference={...refValue,label};
    const response={research_core:true,sources:[evidence],reasoning:{mode:'context_assist',conclusions:[{text:'Conditional result',source_ids:[id],proof:[{text:'Premise',source_ids:[id]}]}]},
      claim_audit:{...audit(),claims:[{...audit().claims[0],source_refs:[reference]}]}};
    const metadata=researchAnswerMetadata(response);
    const options={sources:[evidence],reasoning:metadata.reasoning,claimAudit:metadata.claim_audit};
    const view=reasoningEvidence(options);
    assert.equal(view.claims[0].status,'exact_quote');assert.equal(view.claims[0].refs[0].label,label);
    assert.deepEqual(view.conclusions[0].proof[0].sources,[label]);
    assert.match(reasoningRecord(options),new RegExp(`\\[${label}\\]`));
    assert.equal(reasoningEvidence({...options,sources:[{...evidence,excerpt_start:19}]}).claims[0].status,'unverified');
    assert.equal(reasoningEvidence({...options,citationValidation:{warnings:[{code:'source_revoked'}]}}).claims[0].refs.length,0);
  }
  const invalid={...refValue,label:'S11'};
  const metadata=researchAnswerMetadata({research_core:true,claim_audit:{...audit(),claims:[{...audit().claims[0],source_refs:[invalid]}]}});
  assert.deepEqual(metadata.claim_audit.claims[0].source_refs,[]);
  assert.equal(reasoningEvidence({sources:[{...source,label:'S11'}],claimAudit:{...audit(),claims:[{...audit().claims[0],source_refs:[invalid]}]}}).claims[0].status,'unverified');
});
test('public graph and hypothesis receipts show source labels and limitations, not generated claims or raw traces',()=>{
  const trace=receipt();trace.secret='private raw context';trace.graph_retrieval.confidence=1;trace.hypothesis_support.hypothesis_count=400;trace.hypothesis_support.idea='untrusted generated idea';
  const safe=sanitizeReasoningReceipts(trace,[source,secondSource]);assert.deepEqual(safe,receipt());
  const options={reasoning:{mode:'context_assist',status:'ready',trace},sources:[source,secondSource]};
  const view=reasoningEvidence(options);assert.equal(view.available,true);assert.deepEqual(view.conclusions,[]);
  assert.deepEqual(view.graphRetrieval.sourceLabels,['S1','S2']);assert.deepEqual(view.graphRetrieval.bridgeLabels,['S2']);
  assert.match(view.graphRetrieval.note,/不代表因果证明/);assert.deepEqual(view.hypothesisSupport.operators,['类比找参照','演绎查条件','归纳找共性']);
  assert.match(view.hypothesisSupport.note,/不表示均已得到结论/);assert.match(view.hypothesisSupport.note,/模型背景知识不是本次资料依据/);
  const record=reasoningRecord(options);assert.match(record,/关联补充：\[S2\]/);assert.match(record,/假设需验证/);assert.doesNotMatch(record,/400|private raw context|untrusted generated idea|confidence|model_calls/);
});
test('public receipts fail closed on malformed, unsupported or untraceable source metadata',()=>{
  for(const change of [
    graph=>({...graph,status:'unavailable'}),graph=>({...graph,status:'perfect'}),graph=>({...graph,algorithm:'new_algorithm'}),
    graph=>({...graph,model_calls:1}),graph=>({...graph,max_hops:3}),graph=>({...graph,semantic_entailment_verified:true}),
    graph=>({...graph,semantic_entailment_verified:undefined}),graph=>({...graph,kept_source_ids:[]}),
    graph=>({...graph,bridge_source_ids:[]}),graph=>({...graph,kept_source_ids:[id]}),
    graph=>({...graph,kept_source_ids:[id,secondId,id]}),graph=>({...graph,bridge_source_ids:[secondId,secondId]}),
    graph=>({...graph,bridge_source_ids:['unknown']}),graph=>({...graph,kept_source_ids:Array(9).fill(id)}),
  ]) assert.equal(sanitizeReasoningReceipts({graph_retrieval:change(receipt().graph_retrieval)},[source,secondSource]),undefined);
  for(const change of [
    hypothesis=>({...hypothesis,status:'generated'}),hypothesis=>({...hypothesis,algorithm:'unknown'}),
    hypothesis=>({...hypothesis,model_calls:'0'}),hypothesis=>({...hypothesis,semantic_entailment_verified:true}),
    hypothesis=>({...hypothesis,source_ids:[]}),hypothesis=>({...hypothesis,source_ids:['unknown']}),
    hypothesis=>({...hypothesis,algorithm_modes:['analogy','unknown']}),hypothesis=>({...hypothesis,algorithm_modes:null}),
    hypothesis=>({...hypothesis,algorithm_modes:['analogy','analogy']}),hypothesis=>({...hypothesis,algorithm_modes:[{}]}),
    hypothesis=>({...hypothesis,algorithm_modes:['__proto__']}),
  ]) assert.equal(sanitizeReasoningReceipts({hypothesis_support:change(receipt().hypothesis_support)},[source,secondSource]),undefined);
  for(const sources of [[source],[source,{...secondSource,content:''}],[source,{...secondSource,label:'S1'}],[source,{...secondSource,label:'S99'}],[source,secondSource,secondSource]]) {
    assert.equal(sanitizeReasoningReceipts(receipt(),sources),undefined);
    assert.equal(reasoningEvidence({reasoning:{mode:'context_assist',trace:receipt()},sources}).available,false);
  }
});
test('ready composition support without parseable operator claims does not imply the three algorithms produced results',()=>{
  const trace={hypothesis_support:{...receipt().hypothesis_support,algorithm_modes:[]}};
  const response={research_core:true,reasoning:{mode:'context_assist',status:'hypothesis_support',trace},sources:[source,secondSource]};
  const metadata=researchAnswerMetadata(response);assert.deepEqual(metadata.reasoning.trace,trace);
  const options={reasoning:metadata.reasoning,sources:response.sources},view=reasoningEvidence(options);
  assert.equal(view.available,true);assert.deepEqual(view.hypothesisSupport.operators,[]);
  assert.equal(view.hypothesisSupport.summary,'本次未得到可解析的三类算法线索；仍按原文组织待验证候选。');
  assert.match(reasoningRecord(options),/本次未得到可解析的三类算法线索/);
  assert.doesNotMatch(reasoningRecord(options),/类比找参照|演绎查条件|归纳找共性/);
});
test('unchanged graph receipts are honest about no additions and source withdrawal hides every receipt and copied note',()=>{
  const trace=receipt();trace.graph_retrieval.status='unchanged';trace.graph_retrieval.bridge_source_ids=[];
  const options={reasoning:{mode:'context_assist',trace},sources:[source,secondSource]};
  assert.match(reasoningEvidence(options).graphRetrieval.summary,/没有加入额外原文/);
  for(const unavailable of [{citationValidation:{warnings:[{code:'source_revoked'}]}},{citationValidation:{warnings:[{code:'provenance_unavailable'}]}},{reasoning:{...options.reasoning,status:'source_unavailable'}},{claimAudit:{status:'unavailable'}}]) {
    const value={...options,...unavailable},view=reasoningEvidence(value);
    assert.equal(view.graphRetrieval,null);assert.equal(view.hypothesisSupport,null);assert.equal(view.unavailable,true);
    const record=reasoningRecord(value);assert.doesNotMatch(record,/关联补充|类比找参照|假设边界|S1|S2/);assert.match(record,/旧推断不再/);
  }
});
test('API preserves only source-backed public receipts and never leaks raw trace into rendered or copied answers',()=>{
  const trace={...receipt(),secret:'never expose this internal trace'};
  const response={research_core:true,reasoning:{mode:'context_assist',status:'ready',trace},sources:[source,secondSource]};
  const metadata=researchAnswerMetadata(response);assert.deepEqual(metadata.reasoning.trace,receipt());
  const options={reasoning:metadata.reasoning,sources:response.sources};
  assert.equal(reasoningEvidence(options).graphRetrieval.bridgeLabels[0],'S2');
  const record=answerDeliverable({...options,answer:'An ordinary answer [S1]',researchCore:true});
  assert.match(record.text,/假设需验证/);assert.doesNotMatch(record.text,/never expose|model_calls|bounded_evidence_graph_v1/);
  for(const result of [{...response,sources:[source]},{...response,citation_validation:{warnings:[{code:'source_revoked'}]}}]) {
    assert.equal(researchAnswerMetadata(result).reasoning.trace,undefined);
  }
});
test('exact quote display validates original codepoint offsets, not only a server badge or substring',()=>{
  const valid=reasoningEvidence({claimAudit:audit(),sources:[source]});assert.equal(valid.claims[0].status,'exact_quote');assert.equal(valid.claims[0].refs[0].quote,original);
  for(const value of [{...source,id:'other'},{...source,excerpt_start:19},{...source,content:'原文并没有支持这句话。'},{...source,label:'S2'}]) {
    const result=reasoningEvidence({claimAudit:audit(),sources:[value]});assert.equal(result.claims[0].status,'unverified');assert.equal(result.claims[0].refs.length,0);
  }
  const paraphrase=audit();paraphrase.claims[0].text='已经发现了因果关系。';assert.equal(reasoningEvidence({claimAudit:paraphrase,sources:[source]}).claims[0].status,'unverified');
});
test('multi-source exact badges fail closed when any reference is lost, contradictory or outside the audit window',()=>{
  const validSecond={...refValue,label:'S2',source_id:secondId};
  const sources=[source,secondSource];
  const packet=refs=>({...audit(),claims:[{...audit().claims[0],source_refs:refs}]});
  const both=packet([refValue,validSecond]);
  assert.equal(reasoningEvidence({claimAudit:both,sources}).claims[0].status,'exact_quote');
  for(const refs of [[refValue,{...validSecond,label:'S11'}],[refValue,{...validSecond,end:9999}],
    [refValue,{...validSecond,quote:'Not the asserted quotation.'}],Array(4).fill(refValue),[refValue,refValue]]) {
    const direct=reasoningEvidence({claimAudit:packet(refs),sources});
    assert.equal(direct.claims[0].status,'unverified');
    const sanitized=researchAnswerMetadata({research_core:true,sources,claim_audit:packet(refs)}).claim_audit;
    assert.equal(reasoningEvidence({claimAudit:sanitized,sources}).claims[0].status,'unverified');
  }
  for(const issues of [[{code:'quote_context_review',message:'Condition needs review.'}],[{}],undefined]) {
    const contradicted={...both,claims:[{...both.claims[0],issues}]};
    assert.equal(reasoningEvidence({claimAudit:contradicted,sources}).claims[0].status,'unverified');
    assert.equal(researchAnswerMetadata({research_core:true,sources,claim_audit:contradicted}).claim_audit.claims[0].status,'unverified');
  }
  const dropped=researchAnswerMetadata({research_core:true,sources,claim_audit:packet([refValue,{...validSecond,label:'S11'}])}).claim_audit;
  assert.equal(dropped.claims[0].source_refs.length,1);assert.equal(dropped.claims[0].status,'unverified');
});
test('inferences retain hypothesis and difference labels, and revoked sources suppress advice and quotes in UI and export',()=>{
  const result=reasoningEvidence({reasoning:reasoning(),claimAudit:audit(),sources:[source]});assert.equal(result.conclusions[0].label,'待验证的迁移假设');assert.match(result.conclusions[0].mappings[0],/共同上游.*→/);assert.equal(result.gaps.length,1);
  const options={reasoning:reasoning(),claimAudit:audit(),sources:[source],citationValidation:{warnings:[{code:'source_revoked'}]}};
  const revoked=reasoningEvidence(options);assert.equal(revoked.unavailable,true);assert.deepEqual(revoked.conclusions,[]);assert.deepEqual(revoked.claims[0].refs,[]);assert.equal(revoked.claims[0].status,'source_unavailable');
  const exported=reasoningRecord(options);assert.doesNotMatch(exported,/新场景可能出现/);assert.match(exported,/旧推断不再/);
  const record=answerDeliverable({...options,answer:'Old answer',researchCore:true});assert.match(record.text,/来源.*撤销/);assert.doesNotMatch(record.text,/新场景可能出现/);
});
test('API metadata bounds model content and ignores claimed entailment, confidence and raw trace',()=>{
  const result=researchAnswerMetadata({research_core:true,reasoning:{...reasoning(),trace:{secret:'raw-data'},conclusions:Array(100).fill({...reasoning().conclusions[0],text:'甲'.repeat(4000)})},claim_audit:{...audit(),entailment_verified:true,confidence:1}});
  assert.equal(result.reasoning.conclusions.length,12);assert.equal(Array.from(result.reasoning.conclusions[0].text).length,800);assert.equal(result.reasoning.trace,undefined);assert.equal(result.claim_audit.entailment_verified,false);assert.equal(result.claim_audit.confidence,undefined);
  assert.match(reasoningEvidence({reasoning:result.reasoning}).note,/仅展示部分/);
  const revoked=researchAnswerMetadata({research_core:true,reasoning:reasoning(),claim_audit:audit(),citation_validation:{warnings:[{code:'source_revoked'}]}});assert.deepEqual(revoked.reasoning.conclusions,[]);assert.deepEqual(revoked.claim_audit.claims[0].source_refs,[]);
});
test('actual structural mapping and coded difference contract yields readable hypotheses and proof steps',()=>{
  const result=researchAnswerMetadata({research_core:true,reasoning:{mode:'analogy',status:'hypothesis',conclusions:[{text:'Compare mechanisms',source_ids:[id],mappings:[{base:'supplier',target:'cloud'}],differences:[{code:'different_conditions',base_conditions:['独立'],target_conditions:['共用上游']},{code:'role_types_not_fully_specified'}],candidate_inferences:[{text:'cloud 可能受共同上游故障影响',conditions:['待验证']}],proof:[{text:'供应方共享上游。',kind:'observation',source_ids:[id]}]}]}});
  const view=reasoningEvidence({reasoning:result.reasoning,sources:[source]});assert.equal(view.conclusions[0].mappings[0],'supplier → cloud');assert.match(view.conclusions[0].differences[0],/独立.*共用上游/);assert.match(view.conclusions[0].differences[1],/角色类型/);assert.equal(view.conclusions[0].hypotheses[0],'cloud 可能受共同上游故障影响；沿用条件：待验证；目标场景尚未验证。');assert.deepEqual(view.conclusions[0].proof[0].sources,['S1']);
  assert.match(reasoningRecord({reasoning:result.reasoning,sources:[source]}),/沿用条件：待验证；目标场景尚未验证/);
});
test('source IDs do not upgrade organized premises or unknown proof steps into literal source facts',()=>{
  const result=researchAnswerMetadata({research_core:true,reasoning:{mode:'context_assist',conclusions:[{
    text:'These materials offer directions for checking the answer.',kind:'deduced',source_ids:[id],
    proof:[
      {text:'这句话不在关联原文中。',kind:'observation',source_ids:[id]},
      {text:'只有条件成立才可继续推导。',kind:'rule',source_ids:[id]},
      {text:'未知类型也不是原文认证。',kind:'unknown',source_ids:[id]},
      {text:'因此可继续检查一个假设。',kind:'deduced',source_ids:[id]},
    ],
  }]},claim_audit:audit()});
  const options={reasoning:result.reasoning,claimAudit:result.claim_audit,sources:[source]},view=reasoningEvidence(options);
  assert.deepEqual(view.conclusions[0].proof.map(step=>step.label),[
    '材料整理的前提 · 待核对','参考条件 · 待核对','辅助线索 · 待核对','中间推断',
  ]);
  assert.ok(view.conclusions[0].proof.every(step=>step.sources.includes('S1')),'references remain available for checking');
  assert.equal(view.claims[0].label,'原文逐字对应','exact quotes keep their separately validated label');
  const record=reasoningRecord(options);
  assert.match(record,/材料整理的前提 · 待核对 \[S1\]/);
  assert.match(record,/辅助线索 · 待核对 \[S1\]/);
  assert.doesNotMatch(record,/原文中的前提|原文中的条件规则/);
});
test('background analysis is bounded supplementary context, never a standalone mode or a proof badge',()=>{
  const result=researchAnswerMetadata({research_core:true,reasoning:{...reasoning(),mode:'context_assist',trace:{algorithm_modes:['analogy'],model_calls:0,secret:'do-not-display'},conclusions:[
    {text:'Under the explicit premises, inspect the next step.',kind:'deduced',source_ids:[id],proof:[{text:original,kind:'observation',source_ids:[id]}]},
    {text:'A similar case offers a possible explanation.',kind:'analogy_hypothesis',mappings:[{base:'supplier',target:'cloud'}],candidate_inferences:[{text:'Possible transfer',conditions:['low load']}]},
    {text:'Two reports agree and one report differs.',kind:'inductive_candidate',support_count:2,counterexample_count:1}
  ]},claim_audit:audit()});
  assert.equal(result.reasoning.mode,'context_assist');assert.equal(result.reasoning.trace,undefined);
  const options={reasoning:result.reasoning,claimAudit:result.claim_audit,sources:[source]},view=reasoningEvidence(options);
  assert.equal(view.title,'参考线索与核对');assert.match(view.note,/不是新的原文事实/);assert.match(view.note,/参考案例不能证明结论/);
  assert.deepEqual(view.conclusions.map(item=>item.label),['基于已列前提的推断','待验证的迁移假设','有限样本中的候选规律']);assert.equal(view.conclusions[2].counterexampleCount,1);assert.equal(view.claims[0].status,'exact_quote');
  const record=answerDeliverable({...options,answer:'A natural discussion [S1]',researchCore:true});assert.match(record.text,/A natural discussion/);assert.match(record.text,/参考线索与逐条依据/);assert.match(record.text,/沿用条件：low load/);assert.doesNotMatch(record.text,/自动选择|推理模式|do-not-display/);
  const revoked=researchAnswerMetadata({research_core:true,reasoning:result.reasoning,citation_validation:{warnings:[{code:'source_revoked'}]}});assert.deepEqual(revoked.reasoning.conclusions,[]);
});
test('ordinary conversation without material does not show an empty analysis control',()=>{
  const options={reasoning:{mode:'context_assist',status:'not_requested',conclusions:[],gaps:[]},claimAudit:{status:'no_claims',claims:[]}};
  assert.equal(reasoningEvidence(options).available,false);assert.equal(reasoningRecord(options),'');
  const record=answerDeliverable({...options,answer:'你好，一起聊聊。',researchCore:true});assert.doesNotMatch(record.text,/参考线索与逐条依据/);
  assert.equal(reasoningEvidence({...options,citationValidation:{warnings:[{code:'source_revoked'}]}}).available,true,'revocation remains visible even with no safe conclusions');
});
test('historical memory timestamp remains explicit in answer metadata and copied record',()=>{
  const metadata=researchAnswerMetadata({research_core:true,memory_as_of:0});assert.equal(metadata.memory_as_of,0);
  const record=answerDeliverable({researchCore:true,answer:'Historic context',memoryAsOf:metadata.memory_as_of});assert.match(record.text,/1970-01-01T00:00:00.000Z/);assert.match(record.text,/不代表当时已经获知/);
  assert.equal(researchAnswerMetadata({research_core:true,memory_as_of:'date'}).memory_as_of,null);
});
test('reasoning panel compiles for mobile and collapses on source revocation without a model call',async()=>{
  const filename='ReasoningEvidence.vue',sourceText=readFileSync(new URL('../src/components/'+filename,import.meta.url),'utf8');
  const {descriptor,errors}=parse(sourceText,{filename});assert.deepEqual(errors,[]);const script=compileScript(descriptor,{id:'reasoning'});
  assert.deepEqual(compileTemplate({source:descriptor.template.content,filename,id:'reasoning',compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);
  const scope={computed,ref,watch,reasoningEvidence},executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope)),effects=effectScope();
  const props=reactive({reasoning:reasoning(),claimAudit:audit(),sources:[source],citationValidation:{}});const ui=effects.run(()=>component.setup(props,{expose(){}}));
  assert.equal(ui.expanded.value,false,'analysis details must not distract from the natural answer');ui.expanded.value=true;assert.equal(ui.view.value.claims[0].status,'exact_quote');props.citationValidation={warnings:[{code:'source_revoked'}]};await nextTick();assert.equal(ui.expanded.value,false);assert.equal(ui.view.value.unavailable,true);assert.equal(ui.view.value.conclusions.length,0);effects.stop();
  assert.doesNotMatch(sourceText,/v-html|rich-text|fetch\(|api\.ask|\.request\(/);assert.match(sourceText,/user-select/);assert.match(sourceText,/selectable/);
});
