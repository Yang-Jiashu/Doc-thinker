import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import * as Vue from 'vue';
import {knowledgeSkillMetadata,knowledgeSkillNote,knowledgeSkillRecord,KNOWLEDGE_SKILL_BOUNDARY} from '../src/services/knowledge-skills.js';
import {researchAnswerMetadata} from '../src/services/api.js';
import {answerDeliverable} from '../src/services/answer-deliverable.js';

const reference = () => ({id:'literature-synthesis',name:'论文综合 · 证据对照',version:'1',status:'ready',
  sources:[{id:'compare-conditions',label:'K1',title:'对齐条件再比较',source_title:'DocThinker · 方法设计',
    reference_kind:'project_original',source_url:'',excerpt:'',commentary:'不同口径不能直接合并。',
    boundaries:['结合上传原文核对。'],required_facts:['实验条件是否一致'],stop_conditions:['实验条件不同'],reasoning_role:'deduction'}]});
function methodPlan(skill = reference()) {
  return {kind:'method_plan',version:1,skill_id:skill.id,
    methods:skill.sources.map(source=>({id:source.id,label:source.label,
      reasoning_role:source.reasoning_role || 'unspecified',requirement_count:source.required_facts?.length || 0,
      requirements_status:'pending_review',stop_conditions_status:'pending_review'})),
    source_labels:['S1','S2'],evidence:{selected_sources:2,usable_sources:2,unique_excerpts:2,distinct_documents:2,
      truncated_sources:[],duplicate_excerpts:[]},structural_gaps:[],decision:'review_requirements',
    semantic_entailment_verified:false,model_calls:0,method_execution_claimed:false,normal_answer_allowed:true};
}
function plannedReference() { const value=reference();value.method_plan=methodPlan(value);return value; }
function collectReference() {
  const value=plannedReference();Object.assign(value.method_plan.evidence,{unique_excerpts:1,distinct_documents:1,
    duplicate_excerpts:['S2'],truncated_sources:['S1']});
  Object.assign(value.method_plan,{structural_gaps:['insufficient_distinct_documents'],decision:'collect_evidence'});
  return value;
}

test('bounded structural plans retain actual K cards and never claim verification or execution',()=>{
  const input=plannedReference();input.method_plan.instructions='RUN_REMOTE_SECRET';input.method_plan.verified=true;
  input.method_plan.methods[0].title='FORGED_SOURCE';input.method_plan.evidence.raw_text='PRIVATE_RAW_TEXT';
  const result=knowledgeSkillMetadata(input), plan=result.method_plan;
  assert.equal(plan.kind,'method_plan');assert.equal(plan.skill_id,result.id);
  assert.deepEqual(plan.methods,[{id:result.sources[0].id,label:'K1',reasoning_role:'deduction',requirement_count:1,
    requirements_status:'pending_review',stop_conditions_status:'pending_review'}]);
  assert.equal(plan.semantic_entailment_verified,false);assert.equal(plan.method_execution_claimed,false);assert.equal(plan.model_calls,0);
  assert.doesNotMatch(JSON.stringify(plan),/RUN_REMOTE_SECRET|FORGED_SOURCE|PRIVATE_RAW_TEXT|normal_answer_allowed|"verified"/);
  assert.deepEqual(researchAnswerMetadata({research_core:true,knowledge_skill:input,sources:[]}).knowledge_skill.method_plan,plan);
  input.method_plan.source_labels.push('S3');input.method_plan.methods[0].label='K99';
  assert.deepEqual(plan.source_labels,['S1','S2']);assert.equal(plan.methods[0].label,'K1');
});

test('old ready records without a plan remain compatible and non-ready records discard stale plans',()=>{
  const old=knowledgeSkillMetadata(reference());assert.equal(old.method_plan,undefined);
  assert.equal(knowledgeSkillNote(old.status,old.method_plan),knowledgeSkillNote('ready'));
  for (const status of ['not_applicable','no_match','budget_omitted','unavailable']) {
    const result=knowledgeSkillMetadata({...collectReference(),status});
    assert.deepEqual(result.sources,[]);assert.equal(result.method_plan,undefined);
    assert.doesNotMatch(knowledgeSkillRecord(result),/资料检查|先补充或核对材料|S2/);
  }
  assert.equal(knowledgeSkillMetadata(null),null);
  assert.deepEqual(researchAnswerMetadata({knowledge_skill:plannedReference()}),{});
});

test('invalid plans are dropped without losing backward-compatible method references',()=>{
  const changes = [plan=>{plan.kind='verified_plan';},plan=>{plan.version='1';},plan=>{plan.skill_id='research-ideation';},
    plan=>{plan.semantic_entailment_verified=true;},plan=>{plan.method_execution_claimed=true;},plan=>{plan.model_calls=1;},
    plan=>{delete plan.semantic_entailment_verified;},plan=>{plan.methods=[];},plan=>{plan.methods=Array(4).fill(plan.methods[0]);},
    plan=>{plan.methods[0].id='invented-method';},plan=>{plan.methods[0].label='S1';},plan=>{plan.methods[0].label='K2';},
    plan=>{plan.methods[0].reasoning_role='analogy';},plan=>{plan.methods[0].requirement_count=2;},
    plan=>{plan.methods[0].requirements_status='verified';},plan=>{plan.methods[0].stop_conditions_status='passed';},
    plan=>{plan.source_labels=['S1','S1'];},plan=>{plan.source_labels=['S0'];},plan=>{plan.source_labels=['S10000'];},plan=>{plan.source_labels=['S999999999999999999'];},
    plan=>{plan.source_labels=['K1'];},plan=>{plan.source_labels=Array(13).fill('S1');},plan=>{plan.source_labels=Array(2);},
    plan=>{plan.evidence=null;},plan=>{delete plan.evidence.truncated_sources;},
    plan=>{plan.evidence.truncated_sources=['S3'];},plan=>{plan.evidence.truncated_sources=['S1','S1'];},
    plan=>{plan.evidence.duplicate_excerpts=['K1'];},plan=>{plan.evidence.duplicate_excerpts=['S1'];},
    plan=>{plan.evidence.selected_sources=1;},plan=>{plan.evidence.usable_sources=3;},
    plan=>{plan.evidence.unique_excerpts=3;},plan=>{plan.evidence.distinct_documents=3;},plan=>{plan.evidence.distinct_documents=0;},
    plan=>{plan.structural_gaps=['invented_verified_gap'];},plan=>{plan.structural_gaps=['__proto__'];},
    plan=>{plan.structural_gaps=Array(11).fill('empty_excerpts');},plan=>{plan.structural_gaps=[{code:'empty_excerpts'}];},
    plan=>{plan.structural_gaps=['empty_excerpts','empty_excerpts'];plan.decision='collect_evidence';},
    plan=>{plan.decision='execute_method';},plan=>{plan.decision='collect_evidence';},
    plan=>{plan.structural_gaps=['empty_excerpts'];},plan=>{plan.structural_gaps=['no_usable_excerpts'];plan.decision='collect_evidence';}];
  for (const key of ['selected_sources','usable_sources','unique_excerpts','distinct_documents']) {
    for (const invalid of [-1,13,1.5,true,'2',NaN,Infinity]) changes.push(plan=>{plan.evidence[key]=invalid;});
  }
  for (const change of changes) {
    const input=plannedReference();change(input.method_plan);
    const result=knowledgeSkillMetadata(input);
    assert.ok(result);assert.equal(result.sources[0].id,'compare-conditions');assert.equal(result.method_plan,undefined);
    assert.doesNotMatch(knowledgeSkillRecord(input),/资料检查|invented_verified_gap|execute_method/);
  }
  for (const invalid of [null,[],true,'method_plan']) {
    assert.equal(knowledgeSkillMetadata({...reference(),method_plan:invalid}).method_plan,undefined);
  }
});

test('bounded source labels preserve the source grammar without relabeling or displaying new provenance',()=>{
  const input=plannedReference();input.method_plan.source_labels=['S13','S99'];
  input.method_plan.evidence.truncated_sources=['S13'];
  const result=knowledgeSkillMetadata(input);
  assert.deepEqual(result.method_plan.source_labels,['S13','S99']);
  assert.deepEqual(result.sources.map(source=>source.label),['K1']);
  assert.doesNotMatch(knowledgeSkillRecord(result),/S13|S99|\[S\d+\]/);
});

test('every supplied method must match the real card id, label, role and required facts',()=>{
  const ids=['mao-workplace','research-ideation','literature-synthesis','evidence-writing','decision-review'];
  for(const id of ids) {
    const pack=JSON.parse(readFileSync(new URL(`../../../docthinker/knowledge_skills/${id}/knowledge.json`,import.meta.url),'utf8'));
    for(const entry of pack.entries) {
      const skill={id:pack.id,name:pack.title,version:pack.version,status:'ready',sources:[{...entry,label:'K1'}]};
      skill.method_plan=methodPlan(skill);
      assert.equal(knowledgeSkillMetadata(skill).method_plan.methods[0].id,entry.id);
    }
  }
  const skill=reference();skill.sources.push({...skill.sources[0],id:'check-counterexample',label:'K2'});
  skill.method_plan=methodPlan(skill);assert.equal(knowledgeSkillMetadata(skill).method_plan.methods.length,2);
  skill.method_plan.methods.reverse();assert.equal(knowledgeSkillMetadata(skill).method_plan,undefined);
});

test('material notes and copied records describe limits without exposing algorithms or fabricating citations',()=>{
  const skill=knowledgeSkillMetadata(collectReference()),note=knowledgeSkillNote(skill.status,skill.method_plan);
  assert.match(note,/资料检查：1份文档、1段不同节选；方法前提仍待核对/);
  assert.match(note,/先补充或核对材料/);assert.match(note,/不足两份不同文档/);
  assert.match(note,/尚未检查内容是否支持结论/);assert.match(note,/文档数不代表独立证据/);
  assert.match(note,/方法步骤并未执行/);assert.match(note,/部分节选已截短/);assert.match(note,/重复节选/);
  const copied=answerDeliverable({answer:'仍需核对实验条件 [K1]',researchCore:true,sources:[],knowledgeSkill:skill}).text;
  assert.match(copied,/资料检查：1份文档、1段不同节选/);assert.match(copied,/方法前提仍待核对/);
  assert.match(copied,/K 方法参考不能替代用户事实/);
  assert.doesNotMatch(copied,/\[S\d+\]|method_plan|pending_review|collect_evidence|model_calls|insufficient_distinct_documents|BM25/);
  const empty=plannedReference();Object.assign(empty.method_plan,{source_labels:[],structural_gaps:['no_usable_excerpts'],decision:'collect_evidence'});
  Object.assign(empty.method_plan.evidence,{selected_sources:0,usable_sources:0,unique_excerpts:0,distinct_documents:0});
  assert.match(knowledgeSkillRecord(empty),/没有结构完整的可用节选/);
});

function mount(value) {
  const source=readFileSync(new URL('../src/components/KnowledgeSkillEvidence.vue',import.meta.url),'utf8');
  const {descriptor,errors}=parse(source);assert.deepEqual(errors,[]);
  const script=compileScript(descriptor,{id:'method-plan-evidence'});
  const compiled=compileTemplate({source:descriptor.template.content,filename:'KnowledgeSkillEvidence.vue',id:'method-plan-evidence',
    compilerOptions:{bindingMetadata:script.bindings,isCustomElement:()=>true}});assert.deepEqual(compiled.errors,[]);
  const props=Vue.reactive({value}),scope={computed:Vue.computed,ref:Vue.ref,watch:Vue.watch,
    knowledgeSkillMetadata,knowledgeSkillNote,KNOWLEDGE_SKILL_BOUNDARY};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope));
  const effects=Vue.effectScope(),ui=effects.run(()=>component.setup(props,{expose(){}}));
  const executableTemplate=compiled.code.replace(/import \{([^}]+)\} from "vue"/,(_,names)=>`const {${names.replace(/\bas\b/g,':')}} = Vue;`)
    .replace('export function render','return function render');
  const render=new Function('Vue',executableTemplate)(Vue),setup=Vue.proxyRefs(ui),cache=[];
  return {ui,props,source,render:()=>render({},cache,props,setup),dispose:()=>effects.stop()};
}
const visibleText=node=>typeof node==='string'?node:Array.isArray(node)?node.map(visibleText).join('\n')
  :typeof node?.children==='string'?node.children:visibleText(node?.children||[]);
test('plan stays inside the existing folded reference card and disappears on withdrawal',async()=>{
  const f=mount(collectReference());
  assert.equal(f.ui.expanded.value,false);assert.doesNotMatch(visibleText(f.render()),/资料检查/);
  f.ui.expanded.value=true;let text=visibleText(f.render());
  assert.match(text,/资料检查：1份文档、1段不同节选/);assert.match(text,/先补充或核对材料/);
  assert.doesNotMatch(f.source,/v-html|<(?:a|web-view|image|navigator)\b|:src=|openURL|navigateTo/);
  assert.doesNotMatch(text,/method_plan|pending_review|collect_evidence|model_calls|\[S\d+\]/);
  f.props.value={...collectReference(),status:'unavailable'};await Vue.nextTick();
  assert.equal(f.ui.expanded.value,false);assert.equal(f.ui.skill.value.method_plan,undefined);
  f.ui.expanded.value=true;assert.doesNotMatch(visibleText(f.render()),/资料检查|先补充或核对材料/);
  f.props.value=reference();await Vue.nextTick();f.ui.expanded.value=true;
  assert.match(visibleText(f.render()),/方法参考/);assert.doesNotMatch(visibleText(f.render()),/资料检查/);
  f.props.value=null;await Vue.nextTick();assert.equal(f.ui.skill.value,null);
  assert.doesNotMatch(visibleText(f.render()),/资料检查|方法参考/);f.dispose();
});
