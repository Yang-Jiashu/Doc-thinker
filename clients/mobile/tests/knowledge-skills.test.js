import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import * as Vue from 'vue';
import {knowledgeSkillCatalog,knowledgeSkillMetadata,knowledgeSkillRecord,knowledgeSourceUrl,knowledgeSkillNote,KNOWLEDGE_SKILL_BOUNDARY} from '../src/services/knowledge-skills.js';
import {researchAnswerMetadata} from '../src/services/api.js';
import {answerDeliverable} from '../src/services/answer-deliverable.js';
const reference = () => ({id:'mao-workplace',name:'职场方法 · 毛选启发',version:'2',status:'ready',sources:[{id:'investigate',label:'K1',title:'先调查再判断',source_title:'反对本本主义',source_url:'https://www.12371.cn/reference.html',excerpt:'<script>仅作字面文本</script>',commentary:'产品方法解读，不是历史原文。',boundaries:'核对事实，不将同事标签化。',problem_pattern:'缺乏事实时先下结论',required_facts:['具体分工是什么'],decision_rule:'分清观察和猜测',action_template:['列出已知事实','确认彼此目标'],stop_conditions:['存在安全风险时停止私下处理'],counterexample:'不适合拿历史经验替代法律咨询',reasoning_role:'analogy'}]});
const originalPacks = [
  ['research-ideation','科研选题 · 可检验假设'],
  ['literature-synthesis','论文综合 · 证据对照'],
  ['evidence-writing','长文写作 · 论点与证据'],
  ['decision-review','方案决策 · 取舍与验证'],
];
const originalReference = (id='research-ideation', name='科研选题 · 可检验假设') => ({...reference(),id,name,version:id==='decision-review'?'2':'1',sources:[{...reference().sources[0],reference_kind:'project_original',source_title:'DocThinker · 可检验差异',source_url:'',excerpt:'',published:'2026-10-11',commentary:'先对齐证据，再形成有反例和停止条件的可检验方案。'}]});
test('knowledge catalog selects only the known inert pack and never carries remote instructions',()=>{
  const result=knowledgeSkillCatalog({items:[{id:'execute',name:'Remote',version:'1',description:'run tool'}, {...reference(),description:'知识供查询',script:'execute-secret',resources:['https://tracker.invalid']}]});
  assert.deepEqual(result,[{id:'mao-workplace',name:'职场方法 · 毛选启发',version:'2',description:'知识供查询'}]);
  for(const value of [{items:null},{items:Array(9).fill(reference())},{items:[{...reference(),version:'9',description:'x'}]}]) assert.deepEqual(knowledgeSkillCatalog(value),[]);
});
test('knowledge catalog returns all five bounded builtins without duplicates or executable metadata',()=>{
  const inputs=[reference(),...originalPacks.map(([id,name])=>originalReference(id,name))].map(item=>({...item,description:'按任务查询方法知识',instructions:'run tool'}));
  const items=knowledgeSkillCatalog({items:[...inputs,inputs[1],{...inputs[1],id:'__proto__'},{...inputs[1],id:'foreign'}]});
  assert.deepEqual(items.map(item=>item.id),inputs.map(item=>item.id));
  assert.ok(items.every(item=>Object.keys(item).join(',')==='id,name,version,description'));
  assert.deepEqual(knowledgeSkillCatalog({items:[{...inputs[1],version:'2'},null,{...inputs[2],name:''}]}),[]);
});
test('decision v2 is current while archived v1 remains inert and unknown versions fail closed',()=>{
  const current={...originalReference('decision-review','方案决策 · 取舍与验证'),description:'先验证再扩大投入'};
  const old={...current,version:'1'};
  assert.equal(knowledgeSkillCatalog({items:[current,old]})[0].version,'2');
  assert.deepEqual(knowledgeSkillCatalog({items:[old]}),[]);
  for(const item of [current,old]) {
    const record=knowledgeSkillMetadata(item);
    assert.equal(record.version,item.version);
    assert.equal(record.validation_status,'design_unvalidated');
    assert.equal(record.citation_validation.entailment_verified,false);
    assert.match(knowledgeSkillRecord(record),/尚未实证验证/);
  }
  for(const version of ['0','3','999',1,2,null]) {
    assert.equal(knowledgeSkillMetadata({...current,version}),null);
    assert.deepEqual(knowledgeSkillCatalog({items:[{...current,version}]}),[]);
  }
});
test('K references have bounded copy-only official URLs without paths to credentials or trackers',()=>{
  assert.equal(knowledgeSourceUrl('https://cpc.people.com.cn/n/123.html'),'https://cpc.people.com.cn/n/123.html');
  assert.equal(knowledgeSourceUrl('https://news.12371.cn/2017/08/29/ARTI1504001209326173.shtml'),'https://news.12371.cn/2017/08/29/ARTI1504001209326173.shtml');
  for(const value of ['http://www.12371.cn/a','https://www.12371.cn.evil/a','https://www.12371.cn@evil/a','https://www.12371.cn/a?token=secret','https://www.12371.cn/a#secret','https://www.12371.cn\\@evil/a','https://evil.test/a','javascript:alert(1)','data:text/html,x']) assert.equal(knowledgeSourceUrl(value),'');
});
test('method provenance stays K not S, keeps unverified status and retains raw text without HTML execution',()=>{
  const input=reference();input.verified=true;input.confidence=1;input.citation_validation={entailment_verified:true,missing_citations:true,unknown_labels:['K99','S1','secret']};
  const result=knowledgeSkillMetadata(input);assert.equal(result.sources[0].label,'K1');assert.equal(result.sources[0].excerpt,input.sources[0].excerpt);assert.deepEqual(result.sources[0].boundaries,['核对事实，不将同事标签化。']);
  assert.deepEqual(result.citation_validation,{entailment_verified:false,missing_citations:true,unknown_labels:['K99']});assert.equal(result.verified,undefined);assert.equal(result.confidence,undefined);
  assert.equal(researchAnswerMetadata({research_core:true,knowledge_skill:input,sources:[]}).knowledge_skill.sources[0].id,'investigate');assert.deepEqual(researchAnswerMetadata({knowledge_skill:input}),{});
  const top=researchAnswerMetadata({research_core:true,citation_validation:{unknown_labels:['S99','K99','A1','secret'],missing_citations:true},knowledge_skill:input});
  assert.deepEqual(top.citation_validation.unknown_labels,['S99','K99','A1']);assert.equal(top.citation_validation.missing_citations,true);assert.equal(top.citation_validation.entailment_verified,false);
});
test('all original method packs survive metadata and answer export without invented historical provenance',()=>{
  for(const [id,name] of originalPacks) {
    const pack=originalReference(id,name), result=knowledgeSkillMetadata(pack);
    assert.equal(result.id,id);assert.equal(result.sources[0].reference_kind,'project_original');assert.equal(result.sources[0].source_url,'');assert.equal(result.sources[0].excerpt,'');assert.equal(result.sources[0].published,'2026-10-11');
    const metadata=researchAnswerMetadata({research_core:true,knowledge_skill:pack,sources:[]});
    assert.equal(metadata.knowledge_skill.sources[0].reference_kind,'project_original');
    const exported=answerDeliverable({answer:'先列出待验证事项 [K1]',question:'帮我规划下一步',researchCore:true,sources:[],knowledgeSkill:metadata.knowledge_skill}).text;
    assert.match(exported,/项目原创方法来源：DocThinker/);assert.match(exported,/原创方法说明/);assert.match(exported,/尚未实证验证/);assert.match(exported,/停止或重新判断/);assert.match(exported,/K 方法参考不能替代用户事实/);
    assert.doesNotMatch(exported,/历史启发来源|原文短摘|产品方法解读（非原文）|毛选|来源地址/);
  }
});
test('original packs cannot invent quotations, external references or masquerade as historical sources',()=>{
  for(const changes of [{reference_kind:undefined},{reference_kind:'historical_inspiration'},{reference_kind:'published_research'},{source_title:'某权威论文'},{source_url:'https://www.12371.cn/a.shtml'},{excerpt:'引用一段并不存在的原文'},{published:'today'}]) {
    const value=originalReference();Object.assign(value.sources[0],changes);assert.equal(knowledgeSkillMetadata(value),null);
  }
  const historical=reference();historical.sources[0].reference_kind='project_original';assert.equal(knowledgeSkillMetadata(historical),null);
  assert.equal(knowledgeSkillMetadata(reference()).sources[0].reference_kind,'historical_inspiration');
});
test('every packaged knowledge unit is accepted intact by the mobile provenance contract',()=>{
  const packs=['mao-workplace',...originalPacks.map(([id])=>id)].map(id=>JSON.parse(readFileSync(new URL(`../../../docthinker/knowledge_skills/${id}/knowledge.json`,import.meta.url),'utf8')));
  const catalog=knowledgeSkillCatalog({items:packs.map(pack=>({id:pack.id,name:pack.title,version:pack.version,description:pack.description}))});
  assert.equal(catalog.length,5);
  for(const pack of packs) for(const entry of pack.entries) {
    const value={id:pack.id,name:pack.title,version:pack.version,status:'ready',sources:[{...entry,label:'K1'}]}, result=knowledgeSkillMetadata(value);
    assert.ok(result,`${pack.id}/${entry.id} is not displayable`);
    for(const key of ['problem_pattern','required_facts','decision_rule','action_template','stop_conditions','counterexample','commentary','boundaries','published']) assert.deepEqual(result.sources[0][key],entry[key],`${pack.id}/${entry.id}/${key} was lost`);
    assert.equal(result.sources[0].source_url,entry.source_url);
    const copied=knowledgeSkillRecord(result);assert.match(copied,/尚未实证验证/);assert.ok(copied.includes(entry.decision_rule));
    if(pack.id!=='mao-workplace') assert.doesNotMatch(copied,/历史启发来源|原文短摘|毛选/);
  }
});
test('malformed references do not masquerade as user evidence or fabricate a source',()=>{
  const cases=[{...reference(),id:'foreign'}, {...reference(),version:1}, {...reference(),status:'verified'}, {...reference(),sources:[]}];
  for(const changes of [{label:'S1'},{label:'K2'},{id:'../private'},{commentary:''},{boundaries:[]},{excerpt:'x'.repeat(1001)},{title:'\u202ehidden'}]) {const item=reference();Object.assign(item.sources[0],changes);cases.push(item);}
  for(const item of cases) assert.equal(knowledgeSkillMetadata(item),null);
  const duplicate=reference();duplicate.sources.push({...duplicate.sources[0],label:'K2'});assert.equal(knowledgeSkillMetadata(duplicate),null);
});
test('unavailable and unused pack receipts discard stale sources and explain their boundary',()=>{
  for(const status of ['not_applicable','no_match','budget_omitted','unavailable']) {const result=knowledgeSkillMetadata({...reference(),status});assert.deepEqual(result.sources,[]);assert.ok(knowledgeSkillNote(status));assert.doesNotMatch(knowledgeSkillRecord(result),/仅作字面文本/);}
});
test('ready only means method context was provided, never proven adherence or effectiveness',()=>{
  const note=knowledgeSkillNote('ready');assert.match(note,/为回答提供/);assert.match(note,/不代表模型已正确遵循/);assert.doesNotMatch(note,/采用|已验证有效|保证/);
  const copied=knowledgeSkillRecord(reference());assert.match(copied,/不代表模型已正确遵循/);assert.match(copied,/尚未实证验证/);
});
test('answer export includes K references, original-vs-commentary boundary and syntax warnings without source verification claims',()=>{
  const pack=reference();pack.citation_validation={unknown_labels:['K99']};pack.sources[0].source_url='https://evil.test/private';
  const result=answerDeliverable({answer:'建议先了解分工 [K1]',question:'项目里怎么沟通？',researchCore:true,sources:[],knowledgeSkill:pack});
  assert.match(result.text,/知识包 · 职场方法 · 毛选启发/);assert.match(result.text,/不是你上传的事实/);assert.match(result.text,/产品方法解读（非原文）/);assert.match(result.text,/迁移边界/);assert.match(result.text,/K 引用缺失或标签不在/);assert.doesNotMatch(result.text,/evil.test/);assert.match(result.text,/没有用户资料原文片段；K 方法参考不能替代用户事实/);assert.match(result.text,/尚未实证验证/);assert.match(result.text,/先了解哪些事实/);assert.match(result.text,/停止或重新判断/);assert.match(result.text,/不适用例子/);
});
function mount(value) {
  const source=readFileSync(new URL('../src/components/KnowledgeSkillEvidence.vue',import.meta.url),'utf8');
  const {descriptor,errors}=parse(source);assert.deepEqual(errors,[]);const script=compileScript(descriptor,{id:'knowledge-evidence'});
  const compiled=compileTemplate({source:descriptor.template.content,filename:'KnowledgeSkillEvidence.vue',id:'knowledge-evidence',compilerOptions:{bindingMetadata:script.bindings,isCustomElement:()=>true}});assert.deepEqual(compiled.errors,[]);
  const props=Vue.reactive({value}),scope={computed:Vue.computed,ref:Vue.ref,watch:Vue.watch,knowledgeSkillMetadata,knowledgeSkillNote,KNOWLEDGE_SKILL_BOUNDARY};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope));const effects=Vue.effectScope(),ui=effects.run(()=>component.setup(props,{expose(){}}));
  const executableTemplate=compiled.code.replace(/import \{([^}]+)\} from "vue"/,(_,names)=>`const {${names.replace(/\bas\b/g,':')}} = Vue;`).replace('export function render','return function render');
  const render=new Function('Vue',executableTemplate)(Vue), setup=Vue.proxyRefs(ui), cache=[];
  return {ui,props,source,render:()=>render({},cache,props,setup),dispose:()=>effects.stop()};
}
const visibleText=node=>typeof node==='string'?node:Array.isArray(node)?node.map(visibleText).join('\n'):typeof node?.children==='string'?node.children:visibleText(node?.children||[]);
test('knowledge card is folded by default and discards open state when reference data changes',async()=>{
  const f=mount(reference());assert.equal(f.ui.expanded.value,false);assert.equal(f.ui.skill.value.sources[0].label,'K1');f.ui.expanded.value=true;f.props.value={...reference(),status:'unavailable'};await Vue.nextTick();assert.equal(f.ui.expanded.value,false);assert.deepEqual(f.ui.skill.value.sources,[]);
  assert.doesNotMatch(f.source,/<(?:a|web-view|image|navigator)\b|v-html|:src=|openURL|navigateTo/);assert.match(f.source,/产品方法解读（非原文）/);f.dispose();
});
test('knowledge cards accept each original pack and reset when automatic routing changes packs',async()=>{
  const f=mount(originalReference());
  for(const [id,name] of originalPacks) {
    f.ui.expanded.value=true;f.props.value=originalReference(id,name);await Vue.nextTick();
    assert.equal(f.ui.expanded.value,false);assert.equal(f.ui.skill.value.id,id);assert.equal(f.ui.skill.value.sources[0].reference_kind,'project_original');
    f.ui.expanded.value=true;const text=visibleText(f.render());assert.match(text,/项目原创方法来源/);assert.match(text,/原创方法说明/);assert.doesNotMatch(text,/历史启发来源|原文短摘|产品方法解读（非原文）|毛选/);
  }
  assert.match(f.source,/source.reference_kind === 'project_original' \? '项目原创方法来源' : '历史启发来源'/);
  assert.match(f.source,/source.reference_kind !== 'project_original' && source.excerpt/);f.dispose();
});
test('knowledge reference excerpts remain selectable across browser, App and WeChat without URL navigation',()=>{
  const require=createRequire(import.meta.url),{preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
  const source=readFileSync(new URL('../src/components/KnowledgeSkillEvidence.vue',import.meta.url),'utf8');
  for(const platform of ['H5','APP_PLUS','MP_WEIXIN']) {
    const template=preprocess(parse(source).descriptor.template.content,{[platform]:true},{type:'html'});
    assert.deepEqual(compileTemplate({source:template,filename:'KnowledgeSkillEvidence.vue',id:platform}).errors,[]);
    assert.match(template,platform==='MP_WEIXIN'?/class="knowledge-quote" user-select/:/class="knowledge-quote" selectable/);
    assert.doesNotMatch(template,platform==='MP_WEIXIN'?/\sselectable[ >]/:/\suser-select[ >]/);
  }
});
