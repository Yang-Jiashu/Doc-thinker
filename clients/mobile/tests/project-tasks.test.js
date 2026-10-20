import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {ref,computed,watch,effectScope,nextTick} from 'vue';
import {createProjectTasksClient,taskPlan,taskStateLabel} from '../src/services/project-tasks.js';

const sid='#00001', id='a039ad02-a3c6-4794-94da-f1a0a087b424', runId='b039ad02-a3c6-4794-94da-f1a0a087b424';
const plan=()=>({goal:'Compare our own archive strategies',constraints:'Keep original assumptions',questions:[{id:'q1',title:'Evidence',question:'What evidence supports the proposed strategy?'}]});
const task=(status='ready')=>({id,session_id:sid,...plan(),status,revision:1,generation:1,run_id:null,run_remaining:0,
  steps:[{id:'inventory',kind:'inventory',title:'资料核对',question:'',document_ids:null,depends_on:[],status:'ready',revision:1,generation:1,reason:null,artifact_version:0},
    {id:'q1',kind:'analysis',title:'Evidence',question:plan().questions[0].question,document_ids:null,depends_on:['inventory'],status:'ready',revision:1,generation:1,reason:null,artifact_version:0},
    {id:'report',kind:'report',title:'汇集报告',question:'',document_ids:null,depends_on:['q1'],status:'ready',revision:1,generation:1,reason:null,artifact_version:0}],
  artifact_versions:[{version:1,status:'current'},{version:2,status:'stale'}]});
const artifact=(status='current')=>({task_id:id,step_id:'report',version:1,status,available:status==='current',content:'# Current report\nOriginal evidence remains attributed.',sources:[]});
const documentId='c039ad02-a3c6-4794-94da-f1a0a087b424', unavailableId='d039ad02-a3c6-4794-94da-f1a0a087b424';
const documents=()=>[{id:documentId,filename:'Archive observations.pdf',status:'ready'},{id:unavailableId,filename:'Pending notes.pdf',status:'processing'}];
const memoryId='e039ad02-a3c6-4794-94da-f1a0a087b424';
const memory=(overrides={})=>({id:memoryId,text:'本周先完成客户方案，预算不超过两小时。',kind:'decision',status:'active',revision:2,confirmed_by_user:true,version_status:'active',...overrides});
const memoryPage=()=>({items:[memory()],total:1,next_cursor:null});
const personalPlan=()=>({...plan(),questions:[{...plan().questions[0],input_kind:'personal',memory_ids:[memoryId]}]});
const personalTask=(status='ready')=>({...task(status),steps:task(status).steps.map(step=>({...step,input_kind:'personal',memory_ids:[memoryId]}))});
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};
const source=readFileSync(new URL('../src/components/ProjectTasks.vue',import.meta.url),'utf8');
const serviceSource=readFileSync(new URL('../src/services/project-tasks.js',import.meta.url),'utf8');
const proposalSha='a'.repeat(64);
function proposalFor(value,overrides={}){
  const ids=value.questions.map(item=>item.id),added=ids.filter(id=>id!=='q1'),retired=ids.includes('q1')?[]:['q1'];
  return {task_id:id,base_revision:1,definition_version:1,proposal_sha256:proposalSha,added,retired,changed:[],
    affected:[...new Set([...added,...retired,'report'])].sort(),unaffected_completed:[],blocked:false,blocked_reason:null,
    modified:true,active_analysis_count:ids.length,max_analysis_count:6,execution_authorized:false,...overrides};
}
function evidenceProgress(overrides={}){
  const binding={task_id:id,step_id:'q1',artifact_version:0,step_generation:1,answer_id:null,
    input_sha256:'d'.repeat(64),receipt_sha256:null,requirements_sha256:'e'.repeat(64)};
  const obligations=['b','c'].map((letter,index)=>({id:letter.repeat(64),owner_step_id:'q1',kind:'task_requirement',
    requirement_id:index?'G1':'Q1',semantic_id:(index?'f':'a').repeat(64),status:index?'contradicted':'unresolved',
    query:index?'Check the original goal against the absence roster.':'Retrieve the original workload change log.',
    lookup_query:index?'Find the absence roster.':'Find the change log.',gaps:['missing_original_record'],
    binding:{...binding},requirement_sha256:(index?'f':'a').repeat(64)}));
  const work_items=obligations.map((row,index)=>({id:String(index+1).repeat(64),owner_step_id:'q1',kind:row.kind,
    query:row.query,lookup_query:row.lookup_query,status:'unresolved',attempts:0,round:0,binding:{...binding},obligation_ids:[row.id]}));
  return {schema_version:1,task_id:id,task_revision:1,fingerprint:'9'.repeat(64),obligations,work_items,
    coverage:{total:2,ready_for_review:0,unresolved:1,contradicted:1,not_assessed:0,goal_verified:false},
    next_action:'local_lookup',semantic_verified:false,goal_verified:false,...overrides};
}
function evidenceRepair(overrides={}){
  const value=plan();value.questions[0].question+='\n\n核对缺失原始记录，并保留替代解释。';
  return {task_id:id,base_revision:1,ledger_fingerprint:'9'.repeat(64),plan:value,
    changed_steps:['q1'],deferred_work_count:1,capacity_steps:[],execution_authorized:false,automatic_execution:false,
    next_action:'preview_replan',proposal:proposalFor(value,{changed:['q1'],affected:['q1','report']}),...overrides};
}
function pairedPlan(count=1,personal=false){
  const questions=[];
  for(let index=1;index<=count;index++){
    const question={id:'q'+index,title:index===1?'Evidence':'Question '+index,question:index===1?plan().questions[0].question:'Original complete requirement '+index,
      purpose:'compose',depends_on:index>1?['v_q'+(index-1)]:[],...(personal?{input_kind:'personal',memory_ids:[memoryId]}:{document_ids:[documentId]})};
    questions.push(question,{...question,id:'v_q'+index,title:'核对证据',purpose:'verify',review_of:question.id,depends_on:[question.id]});
  }
  return {...plan(),questions};
}
function pairedTask(count=1,personal=false){const value=pairedPlan(count,personal),base=task();return {...base,goal:value.goal,constraints:value.constraints,
  steps:[base.steps[0],...value.questions.map(row=>({...row,kind:'analysis',status:'ready',revision:1,generation:1,artifact_version:0,
    document_ids:row.document_ids || null,depends_on:row.depends_on.length?row.depends_on:['inventory']})),{...base.steps[2],depends_on:value.questions.map(row=>row.id)}]};}
function auditRepair(overrides={}){
  const value=plan();value.questions.push({...value.questions[0],id:'v_q1',title:'核对证据',purpose:'verify',review_of:'q1',depends_on:['q1']});
  return evidenceRepair({plan:value,changed_steps:['v_q1'],deferred_work_count:0,
    proposal:proposalFor(value,{added:['v_q1'],changed:[],affected:['v_q1','report'],max_analysis_count:12}),...overrides});
}
function recomposeRepair(original=plan(),overrides={}){
  const value=structuredClone(original),owner=value.questions[0],writer={...owner,id:'lookup_q1',title:'补查与重新整理',refine_of:owner.id,depends_on:[owner.id]};
  delete writer.purpose;delete writer.review_of;
  const reviewer={...writer,id:'verify_lookup',title:'核对新版草稿',purpose:'verify',review_of:writer.id,depends_on:[writer.id]};delete reviewer.refine_of;
  value.questions.push(writer,reviewer);
  return evidenceRepair({plan:value,changed_steps:[writer.id,'verify_lookup'],deferred_work_count:0,
    proposal:proposalFor(value,{added:[writer.id,'verify_lookup'],changed:[],affected:[writer.id,'verify_lookup','report'],max_analysis_count:12}),...overrides});
}

function service(){
  let key='owner-a',prefs={mode:'connected'},active=0,maximum=0;
  const calls=[],handlers={request:async path=>path==='/capabilities'?{workbench:{tasks:true}}:path.startsWith('/workbench/tasks?')?{items:[task()]}:path.startsWith('/workbench/task/artifact?')?{artifact:artifact()}:{task:task()}};
  const client=createProjectTasksClient({identity:()=>key,preferences:()=>prefs,request:async(...args)=>{calls.push(args);active++;maximum=Math.max(active,maximum);try{return await handlers.request(...args);}finally{active--;}}});
  return {client,calls,handlers,key:value=>{key=value;},preferences:value=>{prefs=value;},maximum:()=>maximum};
}
test('list opens capabilities and the explicitly scoped list sequentially, never a run',async()=>{
  const h=service();assert.equal(h.calls.length,0);const value=await h.client.load(sid);
  assert.equal(value.enabled,true);assert.equal(value.items[0].steps[0].artifact_version,null);
  assert.deepEqual(h.calls.map(row=>row[0]),['/capabilities','/workbench/tasks?session_id=%2300001']);assert.equal(h.maximum(),1);
  assert.equal('input_snapshot' in value.items[0].steps[0],false);assert.equal('questions' in value.items[0],false);
});
test('demo, pending invitation and missing capabilities cannot create or run',async()=>{
  const h=service();h.preferences({mode:'demo'});assert.equal((await h.client.load(sid)).enabled,false);assert.equal(h.calls.length,0);
  h.preferences({mode:'connected',invitePending:true});await h.client.load(sid);assert.equal(h.calls.length,0);
  h.preferences({mode:'connected'});h.handlers.request=async()=>({workbench:{tasks:false}});await h.client.load(sid);
  await assert.rejects(h.client.create(sid,plan()),/未开放/);await assert.rejects(h.client.run(sid,id,1,runId,1),/未开放/);assert.equal(h.calls.length,1);
});
test('late capability response for another owner cannot read a task endpoint',async()=>{
  const h=service(),wait=deferred();h.handlers.request=()=>wait.promise;const pending=h.client.load(sid);h.key('other');wait.resolve({workbench:{tasks:true}});
  await assert.rejects(pending,/身份或项目/);assert.equal(h.calls.length,1);
});
test('permissions are scoped to a session, mode and identity, not cached globally',async()=>{
  const h=service();await h.client.load(sid);await assert.rejects(h.client.detail('#00002',id),/未开放/);
  h.preferences({mode:'demo'});await assert.rejects(h.client.create(sid,plan()),/未开放/);
  h.preferences({mode:'connected'});h.key('other');await assert.rejects(h.client.detail(sid,id),/未开放/);assert.equal(h.calls.length,2);
});
test('create/update save original drafts with CAS and do not request model execution',async()=>{
  const h=service();await h.client.load(sid);await h.client.create(sid,plan());
  assert.deepEqual(h.calls.at(-1),['/workbench/tasks',{session_id:sid,...plan()},'POST']);
  await h.client.update(sid,id,7,plan());assert.deepEqual(h.calls.at(-1),['/workbench/tasks/update',{session_id:sid,task_id:id,expected_revision:7,...plan()},'POST']);
  assert.equal(h.calls.some(row=>row[0].endsWith('/run')),false);
});
test('six user questions maximum, nontruncating limits and nonempty explicit scopes',()=>{
  const p=plan();p.questions[0].question='🙂'.repeat(1800);assert.equal(taskPlan(p).questions[0].question,p.questions[0].question);
  p.questions[0].question+='🙂';assert.throws(()=>taskPlan(p),/1800/);
  assert.throws(()=>taskPlan({...plan(),questions:Array.from({length:7},(_,i)=>({...plan().questions[0],id:'q'+i}))}),/1–6/);
  p.questions=[{...plan().questions[0],document_ids:[]}];assert.throws(()=>taskPlan(p),/范围/);
  p.questions=[{...plan().questions[0],title:''}];assert.equal('title' in taskPlan(p).questions[0],false);
});
test('run uses an explicit UUID, exact revision and bounded approval; transport failures are not retried',async()=>{
  const h=service();await h.client.load(sid);h.handlers.request=async()=>{throw new Error('network unknown');};
  await assert.rejects(h.client.run(sid,id,7,runId,3),/unknown/);
  assert.deepEqual(h.calls.at(-1),['/workbench/tasks/run',{session_id:sid,task_id:id,expected_revision:7,request_id:runId,max_calls:3},'POST']);
  assert.equal(h.calls.filter(row=>row[0].endsWith('/run')).length,1);
  for(const cap of [0,7,1.5,true])await assert.rejects(h.client.run(sid,id,7,runId,cap),/额度/);
  await assert.rejects(h.client.run(sid,id,7,'guess',1),/编号/);assert.equal(h.calls.length,3);
});
test('simultaneous UI requests are not queued into later side effects',async()=>{
  const h=service();await h.client.load(sid);const wait=deferred();h.handlers.request=()=>wait.promise;const pending=h.client.detail(sid,id);
  await assert.rejects(h.client.run(sid,id,1,runId,1),/重复提交/);wait.resolve({task:task()});await pending;assert.equal(h.calls.length,3);
});
test('detail and artifact fail closed on wrong identity or foreign version',async()=>{
  const h=service();await h.client.load(sid);h.handlers.request=async()=>({task:{...task(),session_id:'#00002'}});await assert.rejects(h.client.detail(sid,id),/不符合/);
  h.handlers.request=async()=>({artifact:{...artifact(),version:2}});await assert.rejects(h.client.artifact(sid,id,1),/不符合/);
  h.handlers.request=async()=>({task:{...task(),status:'successfully_researched'}});await assert.rejects(h.client.detail(sid,id),/不符合/);
});
test('server-created step questions accept the complete 2000-character API bound without expanding draft limits',async()=>{
  const h=service();await h.client.load(sid);const value=task();value.steps[1].question='🙂'.repeat(2000);
  h.handlers.request=async()=>({task:value});const result=await h.client.detail(sid,id);
  assert.equal(result.steps[1].question,value.steps[1].question);assert.equal(Array.from(result.steps[1].question).length,2000);
  assert.throws(()=>taskPlan({...plan(),questions:[{...plan().questions[0],question:value.steps[1].question}]}),/1800/);
  value.steps[1].question+='🙂';await assert.rejects(h.client.detail(sid,id),/不符合/);
});
test('stale/withdrawn artifact responses never expose supplied stale content',async()=>{
  const h=service();await h.client.load(sid);
  for(const status of ['stale','withdrawn']){h.handlers.request=async()=>({artifact:{...artifact(status),available:true,content:'STALE PRIVATE CONTENT'}});const a=await h.client.artifact(sid,id,1);assert.equal(a.available,false);assert.equal(a.content,null);}
  h.handlers.request=async()=>({artifact:{...artifact(),content:null}});await assert.rejects(h.client.artifact(sid,id,1),/不符合/);
});
test('late artifacts cannot return to a hidden or changed project',async()=>{
  const h=service();await h.client.load(sid);let current=true;const wait=deferred();h.handlers.request=()=>wait.promise;const pending=h.client.artifact(sid,id,1,{isCurrent:()=>current});current=false;wait.resolve({artifact:artifact()});await assert.rejects(pending,/身份或项目/);
});
test('control permits only explicit pause/cancel/refresh with the current revision',async()=>{
  const h=service();await h.client.load(sid);for(const action of ['pause','cancel','refresh']){await h.client.control(sid,id,8,action);assert.equal(h.calls.at(-1)[1].expected_revision,8);assert.equal(h.calls.at(-1)[1].action,action);}
  await assert.rejects(h.client.control(sid,id,8,'retry'),/不支持/);assert.equal(h.calls.length,5);
});
test('retry_failed is a separate metadata-only control using the exact revision, never a run',async()=>{
  const h=service();await h.client.load(sid);await h.client.control(sid,id,9,'retry_failed');
  assert.deepEqual(h.calls.at(-1),['/workbench/tasks/control',{session_id:sid,task_id:id,expected_revision:9,action:'retry_failed'},'POST']);
  assert.equal(h.calls.some(row=>row[0].endsWith('/run')),false);
});
test('document scope reads only metadata after capability approval and rejects foreign/malformed identities',async()=>{
  const h=service();await assert.rejects(h.client.documents(sid),/未开放/);assert.equal(h.calls.length,0);
  await h.client.load(sid);assert.equal(h.calls.length,2);
  h.handlers.request=async()=>({documents:documents().map(doc=>({...doc,session_id:sid,chunks:[{content:'DO NOT RETAIN BODY'}],source_ids:['raw']}))});
  assert.deepEqual(await h.client.documents(sid),documents());assert.equal(h.calls.at(-1)[0],'/research/documents?session_id=%2300001');
  h.handlers.request=async()=>({documents:[{...documents()[0],session_id:'#00002'}]});await assert.rejects(h.client.documents(sid),/不符合/);
  h.handlers.request=async()=>({documents:[documents()[0],documents()[0]]});await assert.rejects(h.client.documents(sid),/不符合/);
  h.handlers.request=async()=>({documents:[{...documents()[0],status:'automatically_trusted'}]});await assert.rejects(h.client.documents(sid),/不符合/);
});
test('a late document list cannot cross a changed connection or hidden page',async()=>{
  for(const changed of ['identity','hidden']){
    const h=service();await h.client.load(sid);const pendingResult=deferred();let active=true;h.handlers.request=()=>pendingResult.promise;
    const pending=h.client.documents(sid,{isCurrent:()=>active});if(changed==='identity')h.key('other');else active=false;pendingResult.resolve({documents:documents()});
    await assert.rejects(pending,/身份或项目/);assert.equal(h.calls.length,3);
  }
});

function mount(overrides={}){
  const {descriptor}=parse(source),script=compileScript(descriptor,{id:'tasks'});
  const props=ref({sessionId:sid,identity:'owner-a',active:true,demo:false,disabled:false}).value;
  const hooks={},events=[],calls=[],dialogs=[],clipboard=[],downloads=[];let key='owner-a',confirmation=true;
  const handlers={load:async()=>({enabled:true,items:[task()]}),detail:async()=>task(),create:async()=>task(),update:async()=>({...task(),revision:2}),
    previewReplan:async(_sid,_id,_revision,value)=>proposalFor(value),replan:async()=>({...task('stale'),revision:2}),
    previewEvidenceRepair:async()=>evidenceRepair(),
    run:async(_sid,_id,_revision,request)=>({...task('running'),run_id:request,run_remaining:2}),control:async()=>task('paused'),artifact:async()=>artifact(),documents:async()=>documents(),memories:async()=>memoryPage(),...overrides};
  const client=Object.fromEntries(Object.entries(handlers).map(([name,handler])=>[name,async(...args)=>{calls.push([name,...args]);return handler(...args);} ]));
  const scope={ref,computed,watch,createProjectTasksClient:()=>client,taskPlan,taskStateLabel,researchConnectionKey:()=>key,
    createQueryRequestIdAsync:async()=>runId,downloadResearchMarkdown:(...args)=>downloads.push(args),AnswerBody:{},
    onHide:fn=>hooks.hide=fn,onUnload:fn=>hooks.unload=fn,onBeforeUnmount:fn=>hooks.unmount=fn,
    uni:{showModal:options=>{dialogs.push(options);options.success({confirm:confirmation});},setClipboardData:options=>{clipboard.push(options.data);options.success({});}}};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope)),effects=effectScope();
  const ui=effects.run(()=>component.setup(props,{expose(){},emit:(...args)=>events.push(args)}));
  return {ui,props,hooks,events,calls,dialogs,clipboard,downloads,key:value=>{key=value;},confirm:value=>{confirmation=value;},dispose:()=>{hooks.unmount();effects.stop();}};
}
test('panel does not read, schedule, generate or show toy data before a user opens it',async()=>{
  const f=mount();assert.equal(f.ui.opened.value,false);assert.equal(f.calls.length,0);assert.equal(f.ui.items.value.length,0);
  await f.ui.toggle();assert.deepEqual(f.calls.map(row=>row[0]),['load']);assert.equal(f.ui.task.value,null);f.dispose();
});
test('creating an editable plan is separate from a quota-approved run',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();f.ui.editor.value=plan();await f.ui.savePlan();
  assert.deepEqual(f.calls.map(row=>row[0]),['load','create']);assert.match(f.ui.notice.value,/没有调用模型/);
  f.confirm(false);await f.ui.runTask();assert.equal(f.calls.some(row=>row[0]==='run'),false);
  f.confirm(true);f.ui.setBudget({detail:{value:2}});await f.ui.runTask();assert.equal(f.calls.at(-1)[0],'run');assert.equal(f.calls.at(-1)[5],3);
  assert.match(f.dialogs.at(-1).content,/预扣 3.*未用完不退/);assert.match(f.dialogs.at(-1).content,/不是金额上限/);assert.equal(f.ui.task.value.status,'running');f.dispose();
});
test('new questions preview explicit dependencies, never call a planning model',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();f.ui.editor.value.questions[0].question='First check';f.ui.addQuestion();
  assert.deepEqual(f.ui.editor.value.questions[1].depends_on,['v_q1']);f.ui.setDependency(f.ui.editor.value.questions[1],1,{detail:{value:0}});assert.deepEqual(f.ui.editor.value.questions[1].depends_on,[]);
  f.ui.addQuestion();f.ui.removeQuestion(1);assert.deepEqual(f.ui.editor.value.questions[1].depends_on,[]);assert.deepEqual(f.calls.map(row=>row[0]),['load']);f.dispose();
});
test('editing a saved plan preserves user question IDs and strips system inventory dependency',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();
  assert.equal(f.ui.editor.value.questions[0].id,'q1');assert.deepEqual(f.ui.editor.value.questions[0].depends_on,[]);
  f.ui.editor.value.constraints='New explicit constraint';await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'update');assert.equal(f.calls.at(-1)[3],1);assert.equal(f.ui.task.value.revision,2);f.dispose();
});
test('document scopes are user-opened, select only ready metadata, and preserve dependent question structure on save',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();assert.equal(f.calls.some(row=>row[0]==='documents'),false);
  const item=f.ui.editor.value.questions[0];await f.ui.toggleDocumentScope(item);assert.deepEqual(f.ui.documents.value,documents());
  await f.ui.setDocumentScope(item,{detail:{value:[unavailableId]}});assert.equal(item.document_ids,null);
  await f.ui.setDocumentScope(item,{detail:{value:[documentId]}});assert.deepEqual(item.document_ids,[documentId]);await f.ui.savePlan();
  assert.deepEqual(f.calls.at(-1)[4].questions[0],{id:'q1',title:'Evidence',question:item.question,document_ids:[documentId],depends_on:[]});
  assert.equal(f.calls.some(row=>row[0]==='run'),false);await f.ui.toggleDocumentScope(item);assert.equal(f.ui.documents.value.length,0);assert.equal(f.calls.filter(row=>row[0]==='documents').length,1);f.dispose();
});
test('clearing the final document requires explicit confirmation of broadening to all project materials',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();const item=f.ui.editor.value.questions[0];await f.ui.toggleDocumentScope(item);
  await f.ui.setDocumentScope(item,{detail:{value:[documentId]}});f.confirm(false);await f.ui.setDocumentScope(item,{detail:{value:[]}});assert.deepEqual(item.document_ids,[documentId]);
  assert.match(f.dialogs.at(-1).content,/全项目.*以后新增/);f.confirm(true);await f.ui.setDocumentScope(item,{detail:{value:[]}});assert.equal(item.document_ids,null);
  assert.equal(f.calls.some(row=>['create','update','run'].includes(row[0])),false);f.dispose();
});
test('unavailable selected documents are never silently dropped or converted to whole-project scope',async()=>{
  const missing='e039ad02-a3c6-4794-94da-f1a0a087b424';const f=mount();await f.ui.toggle();await f.ui.newPlan();const item=f.ui.editor.value.questions[0];item.document_ids=[unavailableId,missing];await f.ui.toggleDocumentScope(item);
  assert.deepEqual(f.ui.unavailableScope(item),[unavailableId,missing]);await f.ui.setDocumentScope(item,{detail:{value:[documentId,unavailableId]}});assert.deepEqual(new Set(item.document_ids),new Set([documentId,unavailableId,missing]));
  f.confirm(false);await f.ui.removeUnavailableDocument(item,missing);assert.equal(item.document_ids.includes(missing),true);
  f.confirm(true);await f.ui.removeUnavailableDocument(item,missing);assert.equal(item.document_ids.includes(missing),false);assert.equal(item.document_ids.includes(unavailableId),true);
  assert.equal(f.calls.some(row=>['create','update','run'].includes(row[0])),false);f.dispose();
});
test('hidden document pickers clear filenames and reject their late responses without clearing a saved scope',async()=>{
  const wait=deferred(),f=mount({documents:()=>wait.promise});await f.ui.toggle();await f.ui.newPlan();const item=f.ui.editor.value.questions[0];item.document_ids=[documentId];
  const pending=f.ui.toggleDocumentScope(item);f.hooks.hide();wait.resolve(documents());await pending;
  assert.equal(f.ui.documents.value.length,0);assert.equal(f.ui.documentsLoaded.value,false);assert.equal(f.ui.documentScopeId.value,'');assert.deepEqual(item.document_ids,[documentId]);assert.equal(f.calls.length,2);f.dispose();
});
test('a lost run response does not permit automatic or repeated execution; refresh is read-only',async()=>{
  const f=mount({run:async()=>{throw new Error('Unknown result');},detail:async()=>({...task('blocked'),run_id:runId})});
  await f.ui.toggle();f.ui.task.value=task();await f.ui.runTask();assert.equal(f.ui.uncertainRun.value,runId);await f.ui.runTask();assert.equal(f.calls.filter(row=>row[0]==='run').length,1);
  await f.ui.refreshTask();assert.equal(f.calls.at(-1)[0],'detail');assert.equal(f.ui.uncertainRun.value,'');assert.equal(f.ui.canRun.value,false);f.dispose();
});
test('pause and cancel require user confirmation and do not generate',async()=>{
  const f=mount();await f.ui.toggle();f.ui.task.value=task('running');f.confirm(false);await f.ui.controlTask('pause');assert.equal(f.calls.some(row=>row[0]==='control'),false);
  f.confirm(true);await f.ui.controlTask('pause');assert.equal(f.calls.at(-1)[4],'pause');assert.match(f.dialogs.at(-1).content,/可能仍会完成并计费/);assert.equal(f.ui.task.value.status,'paused');f.dispose();
});
test('a paused unknown provider outcome blocks both run and failed-step reset; GET reconciliation is free',async()=>{
  const value=task('paused');value.steps[1].status='blocked';value.steps[1].reason='provider_outcome_unknown';
  const f=mount({detail:async()=>value});await f.ui.toggle();await f.ui.openTask(id);
  assert.equal(f.ui.hasUnknownOutcome.value,true);assert.equal(f.ui.canRun.value,false);assert.equal(f.ui.canRetryFailed.value,false);
  await f.ui.runTask();await f.ui.controlTask('retry_failed');await f.ui.refreshTask();
  assert.deepEqual(f.calls.map(row=>row[0]),['load','detail','detail']);assert.equal(f.dialogs.length,0);f.dispose();
});
test('known failures require explicit reset approval followed by a different explicit run approval',async()=>{
  const failed=task('paused');failed.steps[1].status='blocked';failed.steps[1].reason='provider_failed';
  const f=mount({detail:async()=>failed,control:async()=>({...task(),revision:3})});await f.ui.toggle();await f.ui.openTask(id);
  assert.equal(f.ui.canRun.value,false);assert.equal(f.ui.canRetryFailed.value,true);
  f.confirm(false);await f.ui.controlTask('retry_failed');assert.equal(f.calls.some(row=>row[0]==='control'),false);
  f.confirm(true);await f.ui.controlTask('retry_failed');assert.equal(f.calls.at(-1)[4],'retry_failed');
  assert.match(f.dialogs.at(-1).content,/可能已经计费.*不返还.*重新批准.*另耗额度/);assert.equal(f.calls.some(row=>row[0]==='run'),false);assert.equal(f.ui.canRun.value,true);
  f.confirm(false);await f.ui.runTask();assert.equal(f.calls.some(row=>row[0]==='run'),false);
  f.confirm(true);await f.ui.runTask();assert.equal(f.calls.filter(row=>row[0]==='run').length,1);assert.equal(f.calls.at(-1)[3],3);f.dispose();
});
test('unknown outcome or active work anywhere prevents resetting another known failed step',async()=>{
  for(const block of ['provider_outcome_unknown','running','completed','cancelled','stale']){
    const f=mount();await f.ui.toggle();const value=task('paused');value.steps[1].status='blocked';value.steps[1].reason='provider_failed';
    if(block==='provider_outcome_unknown')value.steps[2].reason=block;else if(block==='running')value.steps[2].status=block;else value.status=block;
    f.ui.task.value=value;assert.equal(f.ui.canRetryFailed.value,false);await f.ui.controlTask('retry_failed');assert.equal(f.calls.length,1);f.dispose();
  }
});
test('explicit keyboard activation ignores nonbuttons, disabled controls, IME, modifiers and repeats',()=>{
  const f=mount();let clicked=0,prevented=0;
  const button={hasAttribute:()=>false,getAttribute:()=>null,click:()=>clicked++};
  const event={key:'Enter',target:{closest:()=>button},preventDefault:()=>prevented++};
  f.ui.activateButton(event);assert.equal(clicked,1);assert.equal(prevented,1);
  for(const patch of [{repeat:true},{isComposing:true},{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{key:'Tab'},{target:{closest:()=>null}}])f.ui.activateButton({...event,...patch});
  button.hasAttribute=()=>true;f.ui.activateButton(event);assert.equal(clicked,1);
  button.hasAttribute=()=>false;button.getAttribute=()=> 'true';f.ui.activateButton(event);assert.equal(clicked,1);f.dispose();
});
test('a report is rehydrated before every copy or local download, never read from browser storage',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.openTask(id);await f.ui.viewArtifact(1);assert.equal(f.clipboard.length,0);
  await f.ui.deliverArtifact('copy');assert.equal(f.calls.filter(row=>row[0]==='artifact').length,2);assert.equal(f.clipboard[0],artifact().content);
  await f.ui.deliverArtifact('download');assert.equal(f.calls.filter(row=>row[0]==='artifact').length,3);assert.equal(f.downloads.length,1);f.dispose();
});
test('revoked report cannot use a previously displayed private body for copy',async()=>{
  let current=true;const f=mount({artifact:async()=>current?artifact():{...artifact('withdrawn'),available:false,content:null}});
  await f.ui.toggle();await f.ui.openTask(id);await f.ui.viewArtifact(1);current=false;await f.ui.deliverArtifact('copy');assert.equal(f.clipboard.length,0);assert.equal(f.ui.artifact.value.content,null);f.dispose();
});
test('hidden page and unmounted component discard late private data and cannot copy it',async()=>{
  for(const hook of ['hide','unmount']){const wait=deferred(),f=mount({artifact:()=>wait.promise});await f.ui.toggle();await f.ui.openTask(id);const pending=f.ui.viewArtifact(1);f.hooks[hook]();wait.resolve(artifact());await pending;assert.equal(f.ui.artifact.value,null);assert.equal(f.ui.items.value.length,0);assert.equal(f.ui.busy.value,false);assert.equal(f.clipboard.length,0);f.dispose();}
});
test('identity change clears drafts, reports, plan state and late callbacks without new reads',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();f.ui.editor.value=plan();f.props.identity='owner-b';f.key('owner-b');await nextTick();assert.equal(f.ui.editor.value.goal,'');assert.equal(f.ui.task.value,null);assert.equal(f.ui.opened.value,false);assert.equal(f.calls.length,1);f.dispose();
});
test('CAS conflict retains full editable draft and does not resubmit',async()=>{
  const f=mount({update:async()=>{throw Object.assign(new Error('Conflict'),{status:409});}});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();f.ui.editor.value.goal='My full amended goal';await f.ui.savePlan();assert.equal(f.ui.editor.value.goal,'My full amended goal');assert.match(f.ui.error.value,/版本或状态/);assert.equal(f.calls.filter(row=>row[0]==='update').length,1);f.dispose();
});
test('phone/web/native templates compile; model calls and private persistence are absent',()=>{
  const require=createRequire(import.meta.url),{preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
  for(const platform of ['H5','MP_WEIXIN','APP_PLUS']){const context={H5:false,MP_WEIXIN:false,APP_PLUS:false,[platform]:true};const content=preprocess(source,context,{type:'html'}),{descriptor,errors}=parse(content);assert.deepEqual(errors,[]);const script=compileScript(descriptor,{id:platform});assert.deepEqual(compileTemplate({source:descriptor.template.content,id:platform,compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);}
  assert.doesNotMatch(source+serviceSource,/localStorage|setStorage|api\.ask|setInterval|setTimeout/);
  assert.match(source,/min-height:44px/);assert.match(source,/@media\(max-width:680px\)/);assert.match(source,/按依赖顺序/);
});
test('H5 keyboard delegation uses a native DOM wrapper, not a uni-view normalized event target',()=>{
  const require=createRequire(import.meta.url),{preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
  const h5=preprocess(source,{H5:true,MP_WEIXIN:false,APP_PLUS:false},{type:'html'});
  assert.match(h5,/<div class="project-tasks" @keydown="activateButton">/);assert.doesNotMatch(h5,/<view class="project-tasks"/);
  for(const platform of ['MP_WEIXIN','APP_PLUS']){const native=preprocess(source,{H5:false,MP_WEIXIN:false,APP_PLUS:false,[platform]:true},{type:'html'});assert.match(native,/<view class="project-tasks">/);assert.doesNotMatch(native,/<div class="project-tasks"/);}
});
test('local steps and fixed reasons use plain Chinese while analysis titles remain user authored',()=>{
  const f=mount();f.ui.task.value=task();
  assert.equal(f.ui.stepTitle({...task().steps[0],title:'Material inventory'}),'核对项目资料');
  assert.equal(f.ui.stepTitle({...task().steps[2],title:'Evidence report'}),'汇集研究报告');
  assert.equal(f.ui.stepTitle({...task().steps[1],title:'My exact authored comparison'}),'My exact authored comparison');
  assert.equal(f.ui.dependencyNames(task().steps[1]),'核对项目资料');
  assert.match(f.ui.reasonLabel('source_changed'),/资料或计划.*变化/);assert.match(f.ui.reasonLabel('provider_outcome_unknown'),/可能已经计费.*不要重试/);
  for(const value of ['source_changed','private_backend_detail_should_not_show','__proto__','toString']){const label=f.ui.reasonLabel(value);assert.equal(typeof label,'string');assert.equal(label.includes(value),false);}
  assert.doesNotMatch(source,/\{\{\s*step\.reason\s*\}\}/);assert.equal(f.calls.length,0);f.dispose();
});
test('personal plans preserve only explicit memory IDs and permit instruction-only work without documents',()=>{
  const p=personalPlan();assert.deepEqual(taskPlan(p),p);
  p.questions[0].memory_ids=[];assert.deepEqual(taskPlan(p).questions[0].memory_ids,[]);
  assert.equal('document_ids' in taskPlan(p).questions[0],false);
  assert.deepEqual(taskPlan(plan()),plan());
  assert.deepEqual(taskPlan({...plan(),questions:[{...plan().questions[0],input_kind:'documents',memory_ids:[]}]}),plan());
});
test('personal memory scopes reject duplicates, excess, unknown kinds and document mixing rather than silently dropping fields',()=>{
  for(const memory_ids of [null,[memoryId,memoryId],['not-a-memory'],Array.from({length:9},(_,i)=>`${i}039ad02-a3c6-4794-94da-f1a0a087b424`)])assert.throws(()=>taskPlan({...personalPlan(),questions:[{...personalPlan().questions[0],memory_ids}]}),/8 条/);
  assert.throws(()=>taskPlan({...personalPlan(),questions:[{...personalPlan().questions[0],document_ids:[documentId]}]}),/个人任务/);
  assert.throws(()=>taskPlan({...plan(),questions:[{...plan().questions[0],memory_ids:[memoryId]}]}),/8 条/);
  assert.throws(()=>taskPlan({...plan(),questions:[{...plan().questions[0],input_kind:'all_memory'}]}),/输入方式/);
});
test('personal task save and read round-trip identity without leaking private context snapshots',async()=>{
  const h=service();await h.client.load(sid);h.handlers.request=async()=>({task:{...personalTask(),steps:personalTask().steps.map(step=>({...step,context_snapshot:{[memoryId]:2},text:'PRIVATE SNAPSHOT'}))}});
  const value=await h.client.create(sid,personalPlan());assert.deepEqual(h.calls.at(-1)[1],{session_id:sid,...personalPlan()});
  assert.deepEqual(value.steps[1].memory_ids,[memoryId]);assert.equal(value.steps[1].input_kind,'personal');
  assert.equal('context_snapshot' in value.steps[1],false);assert.equal(JSON.stringify(value).includes('PRIVATE SNAPSHOT'),false);
  const malformed=personalTask();malformed.steps[1].document_ids=[documentId];h.handlers.request=async()=>({task:malformed});await assert.rejects(h.client.detail(sid,id),/不符合/);
});
test('memory picker is explicitly scoped and retains only currently active confirmed text, never withdrawn hidden data',async()=>{
  const h=service();await assert.rejects(h.client.memories(sid),/未开放/);assert.equal(h.calls.length,0);await h.client.load(sid);
  const rows=[memory(),memory({id:documentId,status:'invalidated',text:'WITHDRAWN SECRET',redacted:true}),memory({id:unavailableId,status:'archived',text:'ARCHIVED SECRET'}),memory({id:runId,confirmed_by_user:false,text:'UNCONFIRMED SECRET'}),memory({id,redacted:true,text:'HIDDEN SECRET'})];
  h.handlers.request=async()=>({items:rows,total:5,next_cursor:'next / page'});
  const value=await h.client.memories(sid);assert.deepEqual(value,{items:[{id:memoryId,text:memory().text,kind:'decision',revision:2}],next_cursor:'next / page'});
  assert.equal(h.calls.at(-1)[0],'/research/memory?session_id=%2300001&limit=20');
  await h.client.memories(sid,{cursor:value.next_cursor});assert.match(h.calls.at(-1)[0],/cursor=next%20%2F%20page/);
  h.handlers.request=async()=>({items:[memory({session_id:'#00002'})],total:1,next_cursor:null});await assert.rejects(h.client.memories(sid),/不符合/);
  h.handlers.request=async()=>({items:[memory(),memory()],total:2,next_cursor:null});await assert.rejects(h.client.memories(sid),/不符合/);
  h.handlers.request=async()=>({items:[memory({redacted:'true'})],total:1,next_cursor:null});await assert.rejects(h.client.memories(sid),/不符合/);
});
test('late memory pages cannot cross owner, session permission or hide boundaries',async()=>{
  for(const changed of ['identity','hidden']){
    const h=service();await h.client.load(sid);const wait=deferred();let current=true;h.handlers.request=()=>wait.promise;
    const pending=h.client.memories(sid,{isCurrent:()=>current});if(changed==='identity')h.key('other');else current=false;
    wait.resolve(memoryPage());await assert.rejects(pending,/身份或项目/);
  }
  const h=service();await h.client.load(sid);await assert.rejects(h.client.memories('#00002'),/未开放/);assert.equal(h.calls.length,2);
});
test('personal mode needs no upload or memory read and every added question starts with empty explicit context',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();assert.equal(f.ui.inputKind.value,'documents');await f.ui.setInputKind('personal');
  f.ui.editor.value.goal='安排本周客户交付';f.ui.editor.value.questions[0].question='先整理计划与需要我确认的事项';f.ui.addQuestion();f.ui.editor.value.questions[1].question='基于前一步列出交付检查表';
  for(const item of f.ui.editor.value.questions){assert.equal(item.input_kind,'personal');assert.deepEqual(item.memory_ids,[]);assert.equal('document_ids' in item,false);}
  await f.ui.savePlan();const value=f.calls.find(row=>row[0]==='create')[2];assert.equal(value.questions.length,4);assert.equal(value.questions.every(item=>item.input_kind==='personal' && item.memory_ids.length===0),true);
  assert.equal(f.calls.some(row=>['documents','memories','run'].includes(row[0])),false);f.dispose();
});
test('personal picker reads only on click, never auto-selects, enforces eight explicit choices and retains unlisted saved IDs',async()=>{
  const rows=Array.from({length:9},(_,i)=>({id:`${i}039ad02-a3c6-4794-94da-f1a0a087b424`,text:'确认内容'+i,kind:'preference',revision:1}));
  const f=mount({memories:async()=>({items:rows,next_cursor:null})});await f.ui.toggle();await f.ui.newPlan();await f.ui.setInputKind('personal');const item=f.ui.editor.value.questions[0];
  assert.equal(f.calls.some(row=>row[0]==='memories'),false);await f.ui.toggleMemoryScope(item);assert.deepEqual(item.memory_ids,[]);
  f.ui.setMemoryScope(item,{detail:{value:rows.map(row=>row.id)}});assert.deepEqual(item.memory_ids,[]);
  f.ui.setMemoryScope(item,{detail:{value:rows.slice(0,8).map(row=>row.id)}});assert.equal(item.memory_ids.length,8);
  f.ui.setMemoryScope(item,{detail:{value:['unknown']}});assert.equal(item.memory_ids.length,8);
  item.memory_ids=[memoryId];f.ui.setMemoryScope(item,{detail:{value:[rows[0].id]}});assert.deepEqual(item.memory_ids,[rows[0].id,memoryId]);
  f.confirm(false);await f.ui.removeUnlistedMemory(item,memoryId);assert.equal(item.memory_ids.includes(memoryId),true);
  f.confirm(true);await f.ui.removeUnlistedMemory(item,memoryId);assert.equal(item.memory_ids.includes(memoryId),false);f.dispose();
});
test('changing input kinds confirms scope removal and never implicitly converts a selected context',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();const item=f.ui.editor.value.questions[0];item.document_ids=[documentId];
  f.confirm(false);await f.ui.setInputKind('personal');assert.deepEqual(item.document_ids,[documentId]);assert.equal(f.ui.inputKind.value,'documents');
  f.confirm(true);await f.ui.setInputKind('personal');assert.equal('document_ids' in item,false);item.memory_ids=[memoryId];
  f.confirm(false);await f.ui.setInputKind('documents');assert.deepEqual(item.memory_ids,[memoryId]);
  f.confirm(true);await f.ui.setInputKind('documents');assert.equal('memory_ids' in item,false);assert.equal('input_kind' in item,false);f.dispose();
});
test('editing existing personal and mixed tasks preserves per-step identities instead of silently broadening inputs',async()=>{
  const mixed=personalTask();mixed.steps.splice(2,0,{...task().steps[1],id:'q2',depends_on:['q1']});
  const f=mount({detail:async()=>mixed});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();assert.equal(f.ui.inputKind.value,'mixed');
  const p=taskPlan(f.ui.editor.value);assert.equal(p.questions[0].input_kind,'personal');assert.deepEqual(p.questions[0].memory_ids,[memoryId]);assert.equal('input_kind' in p.questions[1],false);
  await f.ui.savePlan();assert.deepEqual(f.calls.at(-1)[4].questions,p.questions);f.dispose();
});
test('closing/hiding a memory picker clears text and late responses but preserves only draft IDs',async()=>{
  const wait=deferred(),f=mount({memories:()=>wait.promise});await f.ui.toggle();await f.ui.newPlan();await f.ui.setInputKind('personal');const item=f.ui.editor.value.questions[0];item.memory_ids=[memoryId];
  const pending=f.ui.toggleMemoryScope(item);f.hooks.hide();wait.resolve(memoryPage());await pending;
  assert.deepEqual(f.ui.memories.value,[]);assert.equal(f.ui.memoryCursor.value,null);assert.deepEqual(item.memory_ids,[memoryId]);assert.equal(f.ui.memoryScopeId.value,'');
  f.props.identity='owner-b';f.key('owner-b');await nextTick();assert.equal(f.ui.editor.value.questions[0].memory_ids,undefined);f.dispose();
});
test('personal memory pagination is manual and completion explicitly requires user acceptance',async()=>{
  let page=0;const f=mount({memories:async()=>++page===1?{items:[{id:memoryId,text:'当前确认内容',kind:'decision',revision:2}],next_cursor:'page2'}:{items:[{id:documentId,text:'第二页确认内容',kind:'preference',revision:1}],next_cursor:null}});
  await f.ui.toggle();await f.ui.newPlan();await f.ui.setInputKind('personal');const item=f.ui.editor.value.questions[0];await f.ui.toggleMemoryScope(item);assert.equal(page,1);
  await f.ui.loadMoreMemories(item);assert.equal(page,2);assert.equal(f.ui.memories.value.length,2);assert.deepEqual(item.memory_ids,[]);
  assert.equal(f.ui.stepTitle({...personalTask().steps[0]}),'核对个人输入与记忆版本');
  assert.match(source,/流程已完成，待你验收/);assert.match(source,/不代表目标已验证/);assert.match(source,/个人任务（无需上传文件）/);f.dispose();
});
test('report acceptance metadata is preserved and an asserted verified goal or partial declaration is rejected',async()=>{
  const h=service();await h.client.load(sid);h.handlers.request=async()=>({artifact:{...artifact(),acceptance_status:'review_required',goal_verified:false}});
  const value=await h.client.artifact(sid,id,1);assert.equal(value.acceptance_status,'review_required');assert.equal(value.goal_verified,false);
  for(const metadata of [{acceptance_status:'review_required',goal_verified:true},{acceptance_status:'accepted',goal_verified:false},{goal_verified:false},{acceptance_status:'review_required'}]){h.handlers.request=async()=>({artifact:{...artifact(),...metadata}});await assert.rejects(h.client.artifact(sid,id,1),/不符合/);}
  h.handlers.request=async()=>({artifact:artifact()});assert.equal((await h.client.artifact(sid,id,1)).available,true);
});
test('only overall completed tasks say awaiting acceptance; individual step state labels retain their meaning',()=>{
  const f=mount();assert.equal(f.ui.taskCompletionLabel('completed'),'流程完成，待验收');assert.equal(taskStateLabel('completed'),'已完成');
  for(const state of ['ready','running','paused','blocked'])assert.equal(f.ui.taskCompletionLabel(state),taskStateLabel(state));
  assert.match(source,/报告已生成，待你验收/);assert.match(source,/artifact\.goal_verified === false/);f.dispose();
});
test('explicit phase opening uses the existing task permission and editor rather than executing a task',async()=>{
  const f=mount();assert.equal(f.ui.opened.value,false);await f.ui.openLinkedTask(id);assert.equal(f.ui.opened.value,true);assert.deepEqual(f.calls.map(row=>row[0]),['load','detail']);assert.equal(f.ui.task.value.id,id);assert.equal(f.calls.some(row=>row[0]==='run'),false);f.dispose();
});

const expandedPlan=()=>({...plan(),questions:[...plan().questions,{id:'q2',title:'Follow-up',question:'Prepare the private follow-up.',depends_on:['q1'],input_kind:'personal',memory_ids:[memoryId]}]});
test('replan preview and apply send complete exact drafts, CAS and reviewed SHA without run or refresh',async()=>{
  const h=service();await h.client.load(sid);const draft=expandedPlan();
  h.handlers.request=async path=>path.endsWith('/preview')?{proposal:proposalFor(draft)}:{task:{...task('stale'),revision:2}};
  const p=await h.client.previewReplan(sid,id,1,draft);
  assert.deepEqual(h.calls.at(-1),['/workbench/tasks/replan/preview',{session_id:sid,task_id:id,expected_revision:1,...taskPlan(draft)},'POST']);
  assert.deepEqual(p.added,['q2']);assert.equal(p.execution_authorized,false);
  await h.client.replan(sid,id,1,draft,p.proposal_sha256);
  assert.deepEqual(h.calls.at(-1),['/workbench/tasks/replan',{session_id:sid,task_id:id,expected_revision:1,...taskPlan(draft),proposal_sha256:proposalSha},'POST']);
  assert.equal(h.calls.some(row=>row[0].endsWith('/run') || row[0].endsWith('/control')),false);
  await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/重新预览/);
});
test('replan is capability gated and requires one matching complete reviewed draft',async()=>{
  const h=service(),draft=expandedPlan();
  await assert.rejects(h.client.previewReplan(sid,id,1,draft),/未开放/);assert.equal(h.calls.length,0);
  await h.client.load(sid);await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/重新预览/);
  h.handlers.request=async()=>({proposal:proposalFor(draft)});await h.client.previewReplan(sid,id,1,draft);
  const count=h.calls.length;
  for(const [revision,value,sha] of [[1,{...draft,constraints:'Changed full constraint'},proposalSha],[2,draft,proposalSha],[1,draft,'b'.repeat(64)]])
    await assert.rejects(h.client.replan(sid,id,revision,value,sha),/重新预览/);
  assert.equal(h.calls.length,count);
});
for(const [name,mutate] of [
  ['foreign task',p=>p.task_id=runId],['wrong base revision',p=>p.base_revision=2],['invalid SHA',p=>p.proposal_sha256='guess'],
  ['duplicate list IDs',p=>p.added=['q2','q2']],['unbounded IDs',p=>p.affected=['X'.repeat(90)]],
  ['added absent from new plan',p=>p.added=['not_present']],['retired still active',p=>p.retired=['q1']],
  ['affected omitted',p=>p.affected=['report']],['completed also invalidated',p=>p.unaffected_completed=['q2']],
  ['blocked reason missing',p=>p.blocked=true],['unexpected authorization',p=>p.execution_authorized=true],
  ['wrong node count',p=>p.active_analysis_count=6],['nonboolean modified',p=>p.modified=1],['unexpected metadata body',p=>p.content='old private body'],
])test(`replan preview rejects ${name} instead of approving a partial or misleading plan`,async()=>{
  const h=service(),draft=expandedPlan();await h.client.load(sid);const p=proposalFor(draft);mutate(p);
  h.handlers.request=async()=>({proposal:p});await assert.rejects(h.client.previewReplan(sid,id,1,draft),/不符合/);
  await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/重新预览/);assert.equal(h.calls.length,3);
});
test('blocked previews are displayed but can never apply or implicitly reconcile unknown work',async()=>{
  const h=service(),draft=expandedPlan();await h.client.load(sid);
  h.handlers.request=async()=>({proposal:proposalFor(draft,{blocked:true,blocked_reason:'unresolved_outcome'})});
  assert.equal((await h.client.previewReplan(sid,id,1,draft)).blocked_reason,'unresolved_outcome');
  await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/重新预览/);assert.equal(h.calls.length,3);
});
test('late replan previews cannot authorize a changed identity, session or hidden editor',async()=>{
  for(const change of ['identity','session','hidden']){
    const h=service(),draft=expandedPlan(),wait=deferred();await h.client.load(sid);let current=true;
    h.handlers.request=()=>wait.promise;const pending=h.client.previewReplan(sid,id,1,draft,{isCurrent:()=>current});
    if(change==='identity')h.key('other');else current=false;
    wait.resolve({proposal:proposalFor(draft)});await assert.rejects(pending,/身份或项目/);
    await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/未开放|重新预览/);assert.equal(h.calls.length,3);
  }
});
test('an unknown apply response consumes its local preview and is never automatically resubmitted',async()=>{
  const h=service(),draft=expandedPlan();await h.client.load(sid);h.handlers.request=async()=>({proposal:proposalFor(draft)});
  await h.client.previewReplan(sid,id,1,draft);h.handlers.request=async()=>{throw new Error('Unknown save outcome');};
  await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/Unknown/);
  await assert.rejects(h.client.replan(sid,id,1,draft,proposalSha),/重新预览/);
  assert.equal(h.calls.filter(row=>row[0]==='/workbench/tasks/replan').length,1);
});
test('dependency drafts reject missing, duplicate, self and circular refs without dropping them',()=>{
  for(const deps of [['missing'],['q1'],['q2','q2']])assert.throws(()=>taskPlan({...expandedPlan(),questions:[{...plan().questions[0],depends_on:deps},expandedPlan().questions[1]]}),/依赖/);
  const circular=expandedPlan();circular.questions[0].depends_on=['q2'];assert.throws(()=>taskPlan(circular),/循环/);
});
test('malformed server dependency metadata is rejected rather than silently replaced with no dependencies',async()=>{
  const h=service();await h.client.load(sid);
  for(const deps of ['q1',['q1','q1'],[{}]]){const value=task();value.steps[1].depends_on=deps;h.handlers.request=async()=>({task:value});await assert.rejects(h.client.detail(sid,id),/不符合/);}
});
test('editing a saved task can add a fresh ID, preview then separately confirm replan without model work',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();await f.ui.addQuestion();
  const added=f.ui.editor.value.questions[1];assert.match(added.id,/^node_[0-9a-f]{24}$/);assert.notEqual(added.id,'q2');
  added.question='Prepare a second private review.';const original=JSON.stringify(f.ui.editor.value);
  await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'previewReplan');assert.equal(f.ui.replanPreview.value.added[0],added.id);
  assert.equal(f.calls.some(row=>['update','replan','run','control'].includes(row[0])),false);
  f.confirm(false);await f.ui.applyReplan();assert.equal(f.calls.some(row=>row[0]==='replan'),false);
  f.confirm(true);await f.ui.applyReplan();const call=f.calls.at(-1);assert.equal(call[0],'replan');assert.equal(call[3],1);assert.equal(call[5],proposalSha);
  assert.equal(JSON.stringify(f.ui.editor.value),original);assert.equal(f.ui.task.value.status,'stale');assert.equal(f.ui.replanPreview.value,null);
  assert.equal(f.calls.some(row=>['run','control','update'].includes(row[0])),false);f.dispose();
});
test('editing any draft field invalidates a preview immediately and requires a fresh preview',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();await f.ui.addQuestion();
  f.ui.editor.value.questions[1].question='Prepare a second private review.';await f.ui.savePlan();assert.ok(f.ui.replanPreview.value);
  f.ui.editor.value.constraints+=' Full new condition.';assert.equal(f.ui.replanPreview.value,null);
  await f.ui.applyReplan();assert.equal(f.calls.some(row=>row[0]==='replan'),false);await f.ui.savePlan();assert.equal(f.calls.filter(row=>row[0]==='previewReplan').length,2);f.dispose();
});
test('retiring an existing node requires confirmation and previews downstream change without dropping other scope refs',async()=>{
  const value=personalTask();value.steps.splice(2,0,{...task().steps[1],id:'q2',title:'Second',document_ids:[documentId],depends_on:['q1']});
  value.steps.at(-1).depends_on=['q1','q2'];const f=mount({detail:async()=>value});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();
  f.confirm(false);await f.ui.removeQuestion(0);assert.equal(f.ui.editor.value.questions.length,2);
  f.confirm(true);await f.ui.removeQuestion(0);const remaining=f.ui.editor.value.questions[0];assert.equal(remaining.id,'q2');assert.deepEqual(remaining.document_ids,[documentId]);assert.deepEqual(remaining.depends_on,[]);
  await f.ui.removeQuestion(0);assert.equal(f.ui.editor.value.questions.length,1);await f.ui.savePlan();assert.deepEqual(f.ui.replanPreview.value.retired,['q1']);
  assert.equal(f.calls.some(row=>['update','replan','run','control'].includes(row[0])),false);f.dispose();
});
test('existing multi-parent dependencies are preserved and explicit cycle edits are rejected',async()=>{
  const value=task();value.steps.splice(2,0,{...task().steps[1],id:'q2',title:'Second',depends_on:['inventory']},{...task().steps[1],id:'q3',title:'Join',depends_on:['inventory','q1','q2']});value.steps.at(-1).depends_on=['q1','q2','q3'];
  const f=mount({detail:async()=>value});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();const [first,second,join]=f.ui.editor.value.questions;
  assert.deepEqual(join.depends_on,['q1','q2']);f.ui.setDependencies(first,{detail:{value:['q3']}});assert.deepEqual(first.depends_on,[]);assert.match(f.ui.error.value,/循环/);assert.deepEqual(join.depends_on,['q1','q2']);
  f.ui.setDependencies(join,{detail:{value:['q2']}});assert.deepEqual(join.depends_on,['q2']);assert.equal(f.ui.topologyChanged.value,true);await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'previewReplan');f.dispose();
});
test('blocked unknown-work preview cannot apply and draft remains intact for manual reconciliation',async()=>{
  const f=mount({previewReplan:async(_sid,_id,_revision,value)=>proposalFor(value,{blocked:true,blocked_reason:'unresolved_outcome'})});
  await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();await f.ui.addQuestion();f.ui.editor.value.questions[1].question='A new check.';
  await f.ui.savePlan();assert.equal(f.ui.replanPreview.value.blocked,true);assert.match(f.ui.replanBlockLabel('unresolved_outcome'),/可能已计费/);await f.ui.applyReplan();assert.equal(f.calls.some(row=>row[0]==='replan'),false);assert.equal(f.ui.editor.value.questions.length,2);f.dispose();
});
test('late or changed-draft UI previews cannot reappear after hide or session switch',async()=>{
  for(const change of ['hide','session','draft']){
    const wait=deferred(),f=mount({previewReplan:()=>wait.promise});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();await f.ui.addQuestion();f.ui.editor.value.questions[1].question='A new check.';
    const draft=taskPlan(f.ui.editor.value),pending=f.ui.savePlan();if(change==='hide')f.hooks.hide();else if(change==='session'){f.props.sessionId='#00002';await nextTick();}else f.ui.editor.value.goal='Changed full goal';
    wait.resolve(proposalFor(draft));await pending;assert.equal(f.ui.replanPreview.value,null);assert.equal(f.calls.some(row=>row[0]==='replan'),false);f.dispose();
  }
});
test('replan CAS conflict retains the complete draft, clears stale preview and never auto-refreshes or executes',async()=>{
  const f=mount({replan:async()=>{throw Object.assign(new Error('Conflict'),{status:409});}});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();await f.ui.addQuestion();f.ui.editor.value.questions[1].question='Keep this complete new requirement.';
  await f.ui.savePlan();const draft=JSON.stringify(f.ui.editor.value);await f.ui.applyReplan();assert.equal(f.ui.replanPreview.value,null);assert.equal(f.ui.saveUncertain.value,false);assert.equal(JSON.stringify(f.ui.editor.value),draft);assert.match(f.ui.error.value,/版本或状态/);assert.equal(f.calls.filter(row=>row[0]==='detail').length,1);assert.equal(f.calls.some(row=>['run','control'].includes(row[0])),false);f.dispose();
});

test('current goal coverage and missing work survive client decoding instead of completed becoming accepted',async()=>{
  const h=service();await h.client.load(sid);const value={...task('completed'),evidence_progress:evidenceProgress()};
  h.handlers.request=async()=>({task:value});const result=await h.client.detail(sid,id);
  assert.deepEqual(result.evidence_progress,value.evidence_progress);
  assert.equal(result.evidence_progress.goal_verified,false);assert.equal(result.evidence_progress.coverage.contradicted,1);
  result.evidence_progress.obligations[0].gaps.push('local display');assert.equal(value.evidence_progress.obligations[0].gaps.length,1);
});

test('grouped goal checks preserve the host group hash without duplicating project acceptance requirements',async()=>{
  const h=service();await h.client.load(sid);const progress=evidenceProgress();
  progress.obligations[1].binding.review_group_sha256='8'.repeat(64);
  progress.work_items[1].binding.review_group_sha256='8'.repeat(64);
  h.handlers.request=async()=>({task:{...task(),evidence_progress:progress}});
  const result=await h.client.detail(sid,id);assert.equal(result.evidence_progress.obligations[1].binding.review_group_sha256,'8'.repeat(64));
  assert.equal(result.evidence_progress.obligations.filter(row=>row.requirement_id==='G1').length,1);
});

test('unavailable evidence is a separate fail-closed state; old services may omit the new field',async()=>{
  const h=service();await h.client.load(sid);
  h.handlers.request=async()=>({task:task()});assert.equal('evidence_progress' in await h.client.detail(sid,id),false);
  const unavailable={schema_version:1,task_id:id,task_revision:1,status:'unavailable',next_action:'collect_evidence',semantic_verified:false,goal_verified:false};
  h.handlers.request=async()=>({task:{...task(),evidence_progress:unavailable}});
  assert.deepEqual((await h.client.detail(sid,id)).evidence_progress,unavailable);
  unavailable.coverage={ready_for_review:100};await assert.rejects(h.client.detail(sid,id),/不符合/);
});

for(const [label,mutate] of [
  ['foreign task',v=>v.task_id=runId],['stale task revision',v=>v.task_revision=2],
  ['claimed goal proof',v=>v.goal_verified=true],['claimed semantic proof',v=>v.semantic_verified=true],
  ['miscounted coverage',v=>v.coverage.ready_for_review=1],['private answer body',v=>v.answer='PRIVATE'],
  ['foreign owner',v=>v.obligations[0].owner_step_id='foreign'],['duplicate obligations',v=>v.obligations.push({...v.obligations[0]})],
  ['missing work coverage',v=>v.work_items.pop()],['wrong artifact version',v=>v.obligations[0].binding.artifact_version=1],
  ['source body in binding',v=>v.obligations[0].binding.content='PRIVATE'],['invalid group checksum',v=>v.obligations[0].binding.review_group_sha256='wrong'],
  ['foreign work reference',v=>v.work_items[0].obligation_ids=['0'.repeat(64)]],['duplicate work identity',v=>v.work_items[1].id=v.work_items[0].id],
  ['lookup as verification',v=>v.work_items[0].status='ready_for_review'],['invalid attempt count',v=>v.work_items[0].attempts=-1],
  ['oversized UTF8 hint',v=>v.obligations[0].lookup_query='证'.repeat(200)],['ready with missing proof',v=>v.obligations[0].status='ready_for_review'],
])test(`evidence progress rejects ${label} instead of displaying a stale positive badge`,async()=>{
  const h=service();await h.client.load(sid);const progress=evidenceProgress();mutate(progress);
  h.handlers.request=async()=>({task:{...task(),evidence_progress:progress}});await assert.rejects(h.client.detail(sid,id),/不符合/);
});

test('report delivery gap status remains distinct from acceptance and cannot assert verified completion',async()=>{
  const h=service();await h.client.load(sid);
  for(const delivery_status of ['draft_with_evidence_gaps','ready_for_user_review']){
    h.handlers.request=async()=>({artifact:{...artifact(),acceptance_status:'review_required',goal_verified:false,delivery_status}});
    assert.equal((await h.client.artifact(sid,id,1)).delivery_status,delivery_status);
  }
  h.handlers.request=async()=>({artifact:{...artifact(),acceptance_status:'review_required',goal_verified:false,delivery_status:'goal_verified'}});
  await assert.rejects(h.client.artifact(sid,id,1),/不符合/);
});

test('evidence repair uses the real metadata preview and existing reviewed CAS apply, never execution',async()=>{
  const h=service();await h.client.load(sid);const repair=evidenceRepair();
  h.handlers.request=async path=>path==='/workbench/tasks/evidence/preview'?{repair}:{task:task('stale')};
  const result=await h.client.previewEvidenceRepair(sid,id,1);
  assert.deepEqual(h.calls.at(-1),['/workbench/tasks/evidence/preview',{session_id:sid,task_id:id,expected_revision:1},'POST']);
  assert.equal(result.execution_authorized,false);assert.deepEqual(result.plan,taskPlan(repair.plan));
  await h.client.replan(sid,id,1,result.plan,result.proposal.proposal_sha256);
  assert.equal(h.calls.at(-1)[0],'/workbench/tasks/replan');assert.equal(h.calls.some(row=>row[0].endsWith('/run')),false);
  await assert.rejects(h.client.replan(sid,id,1,result.plan,result.proposal.proposal_sha256),/重新预览/);
});

for(const [label,mutate] of [
  ['wrong task',v=>v.task_id=runId],['stale revision',v=>v.base_revision=2],['automatic execution',v=>v.automatic_execution=true],
  ['new authorization',v=>v.execution_authorized=true],['bad fingerprint',v=>v.ledger_fingerprint='wrong'],
  ['hidden added node',v=>v.proposal.added=['q1']],['lost changed steps',v=>v.changed_steps=[]],
  ['contradictory capacity',v=>v.capacity_steps=['q1']],['unexpected private body',v=>v.answer='PRIVATE'],
])test(`repair preview rejects ${label} without arming a later apply`,async()=>{
  const h=service();await h.client.load(sid);const repair=evidenceRepair();mutate(repair);
  h.handlers.request=async()=>({repair});await assert.rejects(h.client.previewEvidenceRepair(sid,id,1),/不符合/);
  await assert.rejects(h.client.replan(sid,id,1,evidenceRepair().plan,proposalSha),/重新预览/);
  assert.equal(h.calls.some(row=>row[0]==='/workbench/tasks/replan'),false);
});

test('late evidence repair previews cannot arm replan for another owner or hidden page',async()=>{
  for(const change of ['identity','hidden']){
    const h=service();await h.client.load(sid);const wait=deferred();let current=true;h.handlers.request=()=>wait.promise;
    const pending=h.client.previewEvidenceRepair(sid,id,1,{isCurrent:()=>current});
    if(change==='identity')h.key('other');else current=false;wait.resolve({repair:evidenceRepair()});
    await assert.rejects(pending,/身份或项目/);assert.equal(h.calls.length,3);
  }
});

test('repair UI shows actual gaps then loads editable CAS preview without saving or running',async()=>{
  const value={...task('completed'),evidence_progress:evidenceProgress()},f=mount({detail:async()=>value});
  await f.ui.toggle();await f.ui.openTask(id);assert.equal(f.ui.evidenceGaps.value.length,2);
  assert.equal(f.ui.evidenceRequirementLabel(f.ui.evidenceGaps.value[1]),'原始目标');
  await f.ui.repairEvidence();assert.equal(f.calls.at(-1)[0],'previewEvidenceRepair');
  assert.equal(f.ui.editorOpen.value,true);assert.equal(f.ui.editor.value.goal,value.goal);
  assert.equal(f.ui.editor.value.constraints,value.constraints);assert.match(f.ui.editor.value.questions[0].question,/核对缺失原始记录/);
  assert.equal(f.ui.evidenceRepair.value.deferred_work_count,1);assert.equal(f.ui.replanPreview.value.proposal_sha256,proposalSha);
  assert.equal(f.ui.requiresReplan.value,true);assert.equal(f.calls.some(row=>['replan','update','run','control'].includes(row[0])),false);
  f.confirm(false);await f.ui.applyReplan();assert.equal(f.calls.some(row=>row[0]==='replan'),false);
  f.confirm(true);await f.ui.applyReplan();assert.equal(f.calls.at(-1)[0],'replan');
  assert.equal(f.calls.some(row=>['update','run','control'].includes(row[0])),false);f.dispose();
});

test('editing a repair draft invalidates old CAS and saving requires a fresh preview rather than update',async()=>{
  const f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()})});await f.ui.toggle();await f.ui.openTask(id);await f.ui.repairEvidence();
  f.ui.editor.value.questions[0].question+=' Preserve this new condition.';assert.equal(f.ui.replanPreview.value,null);
  await f.ui.applyReplan();assert.equal(f.calls.some(row=>row[0]==='replan'),false);
  await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'previewReplan');assert.equal(f.calls.some(row=>['update','run'].includes(row[0])),false);f.dispose();
});

test('repair capacity and exhausted evidence keep original requirements and do not enter a retry loop',async()=>{
  const repair=evidenceRepair({plan:plan(),changed_steps:[],deferred_work_count:4,capacity_steps:['q1'],next_action:'collect_evidence',
    proposal:proposalFor(plan(),{changed:[],affected:[],modified:false})});
  const f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()}),previewEvidenceRepair:async()=>repair});
  await f.ui.toggle();await f.ui.openTask(id);const original=JSON.stringify(f.ui.editor.value);await f.ui.repairEvidence();
  assert.equal(JSON.stringify(f.ui.editor.value),original);assert.equal(f.ui.replanPreview.value,null);
  assert.deepEqual(f.ui.evidenceRepair.value.capacity_steps,['q1']);assert.match(f.ui.notice.value,/4 项.*原要求没有截断/);
  assert.equal(f.calls.filter(row=>row[0]==='previewEvidenceRepair').length,1);assert.equal(f.calls.some(row=>['replan','update','run'].includes(row[0])),false);f.dispose();
});

test('unsaved user drafts require explicit replacement approval before preparing a repair',async()=>{
  const f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()})});await f.ui.toggle();await f.ui.openTask(id);
  await f.ui.newPlan();f.ui.editor.value.goal='Keep my unsaved task';f.confirm(false);await f.ui.repairEvidence();
  assert.equal(f.calls.some(row=>row[0]==='previewEvidenceRepair'),false);assert.equal(f.ui.editor.value.goal,'Keep my unsaved task');
  f.confirm(true);await f.ui.repairEvidence();assert.equal(f.calls.at(-1)[0],'previewEvidenceRepair');assert.equal(f.ui.editor.value.goal,plan().goal);f.dispose();
});

test('late repair drafts cannot overwrite hidden, switched or newly edited private state',async()=>{
  for(const change of ['hide','session','draft']){
    const wait=deferred(),f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()}),previewEvidenceRepair:()=>wait.promise});
    await f.ui.toggle();await f.ui.openTask(id);const pending=f.ui.repairEvidence();
    if(change==='hide')f.hooks.hide();else if(change==='session'){f.props.sessionId='#00002';await nextTick();}else f.ui.editor.value.goal='New unsaved goal';
    wait.resolve(evidenceRepair());await pending;assert.equal(f.ui.evidenceRepair.value,null);assert.equal(f.ui.replanPreview.value,null);
    assert.equal(f.calls.some(row=>['replan','run','update'].includes(row[0])),false);f.dispose();
  }
});

for(const [label,change] of [
  ['goal',p=>p.goal='An easier substituted goal'],['constraint',p=>p.constraints='Discard original assumptions'],
  ['original question',p=>p.questions[0].question='Do a different easier task'],
  ['document scope',p=>p.questions[0].document_ids=[documentId]],
  ['input kind',p=>{p.questions[0].input_kind='personal';p.questions[0].memory_ids=[];}],
  ['dependency',p=>p.questions[0].depends_on=['unapproved']],
])test(`repair UI rejects changed ${label} before replacing the user draft`,async()=>{
  const repair=evidenceRepair();change(repair.plan);
  const f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()}),previewEvidenceRepair:async()=>repair});
  await f.ui.toggle();await f.ui.openTask(id);const original=JSON.stringify(f.ui.editor.value);await f.ui.repairEvidence();
  assert.equal(JSON.stringify(f.ui.editor.value),original);assert.equal(f.ui.evidenceRepair.value,null);
  assert.match(f.ui.error.value,/改变了原目标.*没有替换、保存或执行/);assert.equal(f.calls.some(row=>['replan','update','run'].includes(row[0])),false);f.dispose();
});

test('six explicit compose/verify pairs preserve all requirements and twelve disclosed calls; legacy plans stay single stage',()=>{
  const value=pairedPlan(6),result=taskPlan(value);
  assert.equal(result.questions.length,12);assert.equal(result.questions.filter(row=>row.purpose==='compose').length,6);
  assert.equal(result.questions.filter(row=>row.purpose==='verify').length,6);
  for(const row of result.questions.filter(row=>row.purpose==='verify')){
    const target=result.questions.find(item=>item.id===row.review_of);assert.equal(row.question,target.question);
    assert.deepEqual(row.document_ids,target.document_ids);assert.deepEqual(row.depends_on,[target.id]);
  }
  assert.deepEqual(taskPlan(plan()),plan());
  assert.throws(()=>taskPlan(pairedPlan(7)),/1–6/);
});
for(const [label,change] of [
  ['unknown purpose',p=>p.questions[1].purpose='auto_prove'],
  ['missing target',p=>delete p.questions[1].review_of],
  ['foreign target',p=>p.questions[1].review_of='q_other'],
  ['missing target dependency',p=>p.questions[1].depends_on=[]],
  ['shortened original question',p=>p.questions[1].question='Check only a small fragment'],
  ['broader document scope',p=>delete p.questions[1].document_ids],
  ['different memory scope',p=>{for(const row of p.questions){delete row.document_ids;row.input_kind='personal';row.memory_ids=[memoryId];}p.questions[1].memory_ids=[];}],
  ['review field on compose',p=>p.questions[0].review_of='q1'],
])test(`explicit audit plan rejects ${label} without silently repairing or broadening it`,()=>{
  const value=pairedPlan();change(value);assert.throws(()=>taskPlan(value));
});
test('review targets follow DAG dependencies, not display array order; cyclic reverse references still fail closed',async()=>{
  const value=pairedPlan();value.questions.reverse();assert.deepEqual(taskPlan(value).questions.map(row=>row.id),['v_q1','q1']);
  const h=service();await h.client.load(sid);const current=pairedTask();[current.steps[1],current.steps[2]]=[current.steps[2],current.steps[1]];
  h.handlers.request=async()=>({task:current});const result=await h.client.detail(sid,id);assert.equal(result.steps[1].review_of,'q1');
  current.steps[2].depends_on=['v_q1'];await assert.rejects(h.client.detail(sid,id),/不符合/);
  value.questions[1].depends_on=['v_q1'];assert.throws(()=>taskPlan(value),/循环/);
});

test('task decoder accepts twelve explicit stages and full report dependencies while preserving role/target and legacy behavior',async()=>{
  const h=service();await h.client.load(sid);const value=pairedTask(6);h.handlers.request=async()=>({task:value});
  const result=await h.client.detail(sid,id);assert.equal(result.steps.length,14);assert.equal(result.steps[2].purpose,'verify');
  assert.equal(result.steps[2].review_of,'q1');assert.equal(result.steps.at(-1).depends_on.length,12);
  h.handlers.request=async()=>({task:task()});const legacy=await h.client.detail(sid,id);
  assert.equal('purpose' in legacy.steps[1],false);assert.equal('review_of' in legacy.steps[1],false);
});
for(const [label,change] of [
  ['scope mismatch',value=>value.steps[2].document_ids=null],
  ['nonexistent reviewed step',value=>value.steps[2].review_of='foreign'],
  ['missing dependency',value=>value.steps[2].depends_on=['inventory']],
  ['different question',value=>value.steps[2].question='Ignore the original conditions'],
  ['wrapper as a reviewer',value=>{value.steps[0].purpose='verify';value.steps[0].review_of='q1';}],
])test(`task decoder rejects malformed audit ${label}`,async()=>{
  const h=service();await h.client.load(sid);const value=pairedTask();change(value);h.handlers.request=async()=>({task:value});
  await assert.rejects(h.client.detail(sid,id),/不符合/);
});

test('partially searched evidence remains unresolved/contradicted and does not masquerade as exhausted or proven coverage',async()=>{
  const h=service();await h.client.load(sid);const progress=evidenceProgress();progress.work_items[0].status='partially_searched';progress.work_items[0].attempts=1;progress.work_items[0].round=1;
  h.handlers.request=async()=>({task:{...task('completed'),evidence_progress:progress}});
  const result=await h.client.detail(sid,id);assert.equal(result.evidence_progress.work_items[0].status,'partially_searched');
  assert.equal(result.evidence_progress.coverage.ready_for_review,0);assert.equal(result.evidence_progress.coverage.unresolved,1);
  assert.equal(result.evidence_progress.goal_verified,false);progress.work_items[0].attempts=0;progress.work_items[0].round=0;
  await assert.rejects(h.client.detail(sid,id),/不符合/);
});

test('new ordinary plans explicitly disclose compose then verify, save both nodes, and still need a separate bounded grant',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();f.ui.editor.value.goal=plan().goal;f.ui.editor.value.constraints=plan().constraints;
  f.ui.editor.value.questions[0].question=plan().questions[0].question;
  assert.equal(f.ui.editor.value.questions.length,1);assert.equal(f.ui.editor.value.reviews.length,1);assert.equal(f.ui.plannedCalls.value,2);
  await f.ui.savePlan();const sent=f.calls.find(row=>row[0]==='create')[2];assert.deepEqual(sent.questions.map(row=>row.purpose),['compose','verify']);
  assert.equal(sent.questions[1].review_of,sent.questions[0].id);assert.equal(sent.questions[1].question,sent.questions[0].question);
  assert.equal(f.calls.some(row=>row[0]==='run'),false);assert.equal(f.ui.budgetOptions.length,6);
  f.confirm(false);await f.ui.runTask();assert.equal(f.calls.some(row=>row[0]==='run'),false);
  assert.match(source,/写作 \/ 整理 → 核对证据/);assert.match(source,/完整计划至多/);assert.match(source,/单批仍最多批准 6 次/);f.dispose();
});

test('new plan editing synchronizes exact question and explicit personal scope into every visible audit without duplicating editable questions',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.newPlan();await f.ui.setInputKind('personal');
  f.ui.editor.value.goal='Prepare my week';const first=f.ui.editor.value.questions[0];first.question='完整保留所有条件，不发送消息。';first.memory_ids=[memoryId];
  await f.ui.addQuestion();f.ui.editor.value.questions[1].question='基于核对后的第一步安排优先级。';
  assert.equal(f.ui.editor.value.questions.length,2);assert.equal(f.ui.plannedCalls.value,4);
  assert.deepEqual(f.ui.editor.value.questions[1].depends_on,['v_q1']);
  first.question+=' 预算两小时。';const result=f.ui.planForEditor(f.ui.editor.value);
  assert.deepEqual(result.questions.map(row=>row.id),['q1','v_q1','q2','v_q2']);
  assert.equal(result.questions[1].question,first.question);assert.deepEqual(result.questions[1].memory_ids,[memoryId]);
  assert.equal(result.questions.every(row=>row.input_kind==='personal'),true);assert.equal(f.calls.some(row=>row[0]==='memories'),false);f.dispose();
});

test('editing an existing audited plan preserves original order and roles while only compose questions are editable',async()=>{
  const value=pairedTask(2),f=mount({detail:async()=>value});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();
  assert.equal(f.ui.editor.value.questions.length,2);assert.equal(f.ui.editor.value.reviews.length,2);
  assert.deepEqual(f.ui.editor.value.order,['q1','v_q1','q2','v_q2']);
  f.ui.editor.value.questions[0].question+=' Keep the alternative explanation unknown.';
  f.ui.editor.value.questions[0].document_ids=[unavailableId];const result=f.ui.planForEditor(f.ui.editor.value);
  assert.equal(result.questions[1].question,result.questions[0].question);assert.deepEqual(result.questions[1].document_ids,[unavailableId]);
  assert.deepEqual(result.questions[2].depends_on,['v_q1']);assert.equal(f.ui.topologyChanged.value,false);
  await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'update');assert.equal(f.calls.at(-1)[4].questions.length,4);
  assert.equal(f.calls.some(row=>row[0]==='run'),false);f.dispose();
});

test('legacy editing never silently adds audit nodes or changes one-call questions into two-call work',async()=>{
  const f=mount();await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();assert.equal(f.ui.editor.value.reviews,undefined);
  await f.ui.addQuestion();f.ui.editor.value.questions[1].question='Keep this legacy follow-up.';
  const result=f.ui.planForEditor(f.ui.editor.value);assert.equal(result.questions.length,2);assert.equal(result.questions.some(row=>row.purpose==='verify'),false);
  assert.equal(f.ui.plannedCalls.value,2);await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'previewReplan');assert.equal(f.calls.some(row=>row[0]==='run'),false);f.dispose();
});

test('retiring a compose retires its attached audit lineage and dependencies, never reuses stale reviewer identities',async()=>{
  const f=mount({detail:async()=>pairedTask(2)});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();await f.ui.removeQuestion(0);
  assert.deepEqual(f.ui.editor.value.questions.map(row=>row.id),['q2']);assert.deepEqual(f.ui.editor.value.reviews.map(row=>row.id),['v_q2']);
  assert.deepEqual(f.ui.editor.value.questions[0].depends_on,[]);assert.equal(f.ui.plannedCalls.value,2);
  assert.equal(f.ui.planForEditor(f.ui.editor.value).questions.some(row=>['q1','v_q1'].includes(row.id)),false);f.dispose();
});

test('new repair adds an explicit same-scope verify rather than changing writer requirements, then arms only the reviewed CAS draft',async()=>{
  const h=service();await h.client.load(sid);const repair=auditRepair();h.handlers.request=async path=>path.endsWith('/evidence/preview')?{repair}:{task:{...task('stale'),revision:2}};
  const result=await h.client.previewEvidenceRepair(sid,id,1);assert.deepEqual(result.proposal.added,['v_q1']);assert.equal(result.plan.questions[0].question,plan().questions[0].question);
  await h.client.replan(sid,id,1,result.plan,result.proposal.proposal_sha256);assert.equal(h.calls.at(-1)[0],'/workbench/tasks/replan');
  assert.equal(h.calls.some(row=>row[0].endsWith('/run')),false);
});
for(const [label,change] of [
  ['unpaired new compose',repair=>{repair.plan.questions[1].purpose='compose';delete repair.plan.questions[1].review_of;}],
  ['scope expansion',repair=>repair.plan.questions[1].document_ids=[documentId]],
  ['writer question change',repair=>{repair.plan.questions[0].question+='\n\nDiscard old requirements';repair.plan.questions[1].question=repair.plan.questions[0].question;}],
  ['writer dependency change',repair=>repair.plan.questions[0].depends_on=['v_q1']],
  ['foreign reviewed target',repair=>repair.plan.questions[1].review_of='foreign'],
  ['substituted goal',repair=>repair.plan.goal='A different easier goal'],
])test(`new repair decoder rejects ${label} before permitting any CAS apply`,async()=>{
  const h=service();await h.client.load(sid);const repair=auditRepair();change(repair);h.handlers.request=async()=>({repair});
  await assert.rejects(h.client.previewEvidenceRepair(sid,id,1));await assert.rejects(h.client.replan(sid,id,1,plan(),proposalSha),/重新预览/);
  assert.equal(h.calls.some(row=>row[0].endsWith('/run') || row[0].endsWith('/replan')),false);
});

test('repair scope uses immutable last-read metadata rather than caller-mutated task data or an unknown revision',async()=>{
  const h=service(),loaded=await h.client.load(sid);loaded.items[0].goal='Mutated display object';const repair=auditRepair();repair.plan.goal='Mutated display object';
  h.handlers.request=async()=>({repair});await assert.rejects(h.client.previewEvidenceRepair(sid,id,1),/不符合/);
  const before=h.calls.length;await assert.rejects(h.client.previewEvidenceRepair(sid,id,2),/读取最新任务/);assert.equal(h.calls.length,before);
});

test('repair UI adds only a read-only audit card, preserves writer text and explicitly separates CAS save from run',async()=>{
  const repair=auditRepair(),f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()}),previewEvidenceRepair:async()=>repair});
  await f.ui.toggle();await f.ui.openTask(id);await f.ui.repairEvidence();assert.equal(f.ui.editor.value.questions.length,1);
  assert.equal(f.ui.editor.value.questions[0].question,plan().questions[0].question);assert.equal(f.ui.editor.value.reviews.length,1);
  assert.equal(f.ui.plannedCalls.value,2);assert.deepEqual(f.ui.replanPreview.value.added,['v_q1']);
  assert.deepEqual(f.ui.planForEditor(f.ui.editor.value),taskPlan(repair.plan));await f.ui.applyReplan();assert.equal(f.calls.at(-1)[0],'replan');
  assert.equal(f.calls.some(row=>['run','control','update'].includes(row[0])),false);f.dispose();
});

test('read-only chained verify stages synchronize recursively and retain the exact pre-reviewed CAS ordering',async()=>{
  const value=pairedTask(),prior=value.steps[2],extra={...prior,id:'v_second',review_of:prior.id,depends_on:[prior.id]};
  value.steps.splice(3,0,extra);value.steps.at(-1).depends_on.push(extra.id);
  const f=mount({detail:async()=>value});await f.ui.toggle();await f.ui.openTask(id);await f.ui.editPlan();
  assert.equal(f.ui.editor.value.questions.length,1);assert.equal(f.ui.editor.value.reviews.length,2);
  assert.equal(f.ui.reviewsFor('q1').length,2);assert.equal(f.ui.plannedCalls.value,3);
  f.ui.editor.value.questions[0].question+=' Keep every earlier condition and source attribution.';
  f.ui.editor.value.questions[0].document_ids=[unavailableId];const result=f.ui.planForEditor(f.ui.editor.value);
  assert.deepEqual(result.questions.map(row=>row.id),['q1','v_q1','v_second']);
  assert.equal(result.questions.every(row=>row.question===result.questions[0].question),true);
  assert.equal(result.questions.every(row=>JSON.stringify(row.document_ids)===JSON.stringify([unavailableId])),true);
  assert.equal(result.questions[2].review_of,'v_q1');assert.equal(f.ui.taskCallCounts.value.total,3);f.dispose();
});

test('partial search and total stage cost stay visible on task detail without upgrading goal acceptance',async()=>{
  const progress=evidenceProgress();progress.work_items[0]={...progress.work_items[0],status:'partially_searched',attempts:1,round:1};
  const f=mount({detail:async()=>({...pairedTask(),evidence_progress:progress})});await f.ui.toggle();await f.ui.openTask(id);
  assert.equal(f.ui.partiallySearched.value,true);assert.deepEqual(f.ui.taskCallCounts.value,{compose:1,verify:1,total:2});
  assert.equal(f.ui.task.value.evidence_progress.goal_verified,false);assert.equal(f.ui.task.value.evidence_progress.coverage.ready_for_review,0);
  assert.match(source,/只查看了部分候选，不代表资料已查尽/);assert.match(source,/保存计划不代表授权执行/);f.dispose();
});

test('host review lineage and work-only lookup policy round-trip without being mistaken for a proof mismatch',async()=>{
  const h=service();await h.client.load(sid);const progress=evidenceProgress();
  for(const row of progress.obligations)row.binding.review_lineage_sha256='7'.repeat(64);
  for(const row of progress.work_items){row.binding.review_lineage_sha256='7'.repeat(64);row.binding.lookup_policy='goal_evidence_lookup_v2';}
  h.handlers.request=async()=>({task:{...task(),evidence_progress:progress}});const result=await h.client.detail(sid,id);
  assert.equal(result.evidence_progress.obligations[0].binding.review_lineage_sha256,'7'.repeat(64));
  assert.equal(result.evidence_progress.work_items[0].binding.lookup_policy,'goal_evidence_lookup_v2');
  assert.equal('lookup_policy' in result.evidence_progress.obligations[0].binding,false);
  assert.equal(result.evidence_progress.goal_verified,false);
  progress.work_items[0].binding.review_lineage_sha256='8'.repeat(64);await assert.rejects(h.client.detail(sid,id),/不符合/);
});

test('lineage policies cannot insert private data, unknown execution policy or stale malformed hashes into evidence metadata',async()=>{
  const h=service();await h.client.load(sid);
  for(const mutate of [
    progress=>progress.obligations[0].binding.review_lineage_sha256='guess',
    progress=>progress.work_items[0].binding.lookup_policy='auto_execute_and_verify',
    progress=>progress.obligations[0].binding.lookup_policy='goal_evidence_lookup_v2',
    progress=>progress.work_items[0].binding.raw_source='private source body',
  ]){const progress=evidenceProgress();mutate(progress);h.handlers.request=async()=>({task:{...task(),evidence_progress:progress}});await assert.rejects(h.client.detail(sid,id),/不符合/);}
});

for(const inputKind of ['whole-project','scoped-documents','selected-memory'])test(`repair safely admits a disclosed retrieve/recompose then verify pair with unchanged ${inputKind} input`,async()=>{
  const h=service();await h.client.load(sid);const before=task(),original=plan();
  if(inputKind==='scoped-documents'){before.steps[1].document_ids=[documentId];original.questions[0].document_ids=[documentId];}
  if(inputKind==='selected-memory'){Object.assign(before.steps[1],{input_kind:'personal',memory_ids:[memoryId]});Object.assign(original.questions[0],{input_kind:'personal',memory_ids:[memoryId]});}
  h.handlers.request=async()=>({task:before});await h.client.detail(sid,id);const repair=recomposeRepair(original);
  h.handlers.request=async path=>path.endsWith('/evidence/preview')?{repair}:{task:{...before,revision:2,status:'stale'}};
  const result=await h.client.previewEvidenceRepair(sid,id,1);
  assert.deepEqual(result.proposal.added,['lookup_q1','verify_lookup']);assert.deepEqual(result.changed_steps,['lookup_q1','verify_lookup']);
  assert.equal(result.plan.questions[1].question,result.plan.questions[0].question);
  assert.deepEqual(result.plan.questions[1].depends_on,['q1']);assert.equal(result.plan.questions[2].review_of,'lookup_q1');
  assert.equal(result.execution_authorized,false);await h.client.replan(sid,id,1,result.plan,result.proposal.proposal_sha256);
  assert.equal(h.calls.at(-1)[0],'/workbench/tasks/replan');assert.equal(h.calls.some(row=>row[0].endsWith('/run')),false);
});

for(const [label,change] of [
  ['different requirement',r=>{r.plan.questions[1].question='A smaller easier substituted task';r.plan.questions[2].question=r.plan.questions[1].question;}],
  ['broadened scope',r=>{r.plan.questions[1].document_ids=[documentId];r.plan.questions[2].document_ids=[documentId];}],
  ['borrowed personal memory',r=>{for(const row of r.plan.questions.slice(1)){row.input_kind='personal';row.memory_ids=[memoryId];}}],
  ['missing original owner',r=>r.plan.questions[1].depends_on=['foreign']],
  ['missing typed replacement',r=>delete r.plan.questions[1].refine_of],
  ['foreign typed replacement',r=>r.plan.questions[1].refine_of='foreign'],
  ['extra predecessor',r=>r.plan.questions[1].depends_on=['q1','verify_lookup']],
  ['missing review',r=>{r.plan.questions.pop();r.changed_steps=['lookup_q1'];r.proposal.added=['lookup_q1'];r.proposal.affected=['lookup_q1','report'];r.proposal.active_analysis_count=2;}],
  ['reviewing only old draft',r=>{r.plan.questions[2].review_of='q1';r.plan.questions[2].depends_on=['q1'];}],
  ['old definition changed',r=>{r.plan.questions[0].question+='\n\nChanged original';r.plan.questions[1].question=r.plan.questions[0].question;r.plan.questions[2].question=r.plan.questions[0].question;}],
  ['old node in changed list',r=>{r.changed_steps.push('q1');r.proposal.changed=['q1'];r.proposal.affected.push('q1');}],
])test(`repair retrieval pair rejects ${label} rather than inventing authority or weakening requirements`,async()=>{
  const h=service();await h.client.load(sid);const repair=recomposeRepair();change(repair);h.handlers.request=async()=>({repair});
  await assert.rejects(h.client.previewEvidenceRepair(sid,id,1));await assert.rejects(h.client.replan(sid,id,1,plan(),proposalSha),/重新预览/);
  assert.equal(h.calls.some(row=>row[0].endsWith('/run') || row[0].endsWith('/replan')),false);
});

test('repair UI distinguishes two new paid stages from checking old windows, preserves original task, and requires independent grant',async()=>{
  const repair=recomposeRepair(),f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()}),previewEvidenceRepair:async()=>repair});
  await f.ui.toggle();await f.ui.openTask(id);await f.ui.repairEvidence();
  assert.equal(f.ui.editor.value.questions.length,2);assert.equal(f.ui.editor.value.reviews.length,1);
  assert.equal(f.ui.editor.value.questions[0].question,plan().questions[0].question);
  assert.equal(f.ui.evidenceRepair.value.added_composition_count,1);assert.equal(f.ui.evidenceRepair.value.added_review_count,1);
  assert.equal(f.ui.plannedCalls.value,3);assert.deepEqual(f.ui.planForEditor(f.ui.editor.value),taskPlan(repair.plan));
  assert.match(source,/重新检索并整理新版草稿，再单独核对/);assert.match(source,/明确新建下一阶段/);
  await f.ui.applyReplan();assert.equal(f.calls.at(-1)[0],'replan');assert.equal(f.calls.some(row=>['run','control','update'].includes(row[0])),false);
  assert.equal(f.ui.budgetOptions.length,6);f.dispose();
});

test('repair UI refuses a recompose pair with altered owner requirement before replacing the draft',async()=>{
  const repair=recomposeRepair();repair.plan.questions[1].question='Do an unrelated task';repair.plan.questions[2].question=repair.plan.questions[1].question;
  const f=mount({detail:async()=>({...task('completed'),evidence_progress:evidenceProgress()}),previewEvidenceRepair:async()=>repair});
  await f.ui.toggle();await f.ui.openTask(id);const original=JSON.stringify(f.ui.editor.value);await f.ui.repairEvidence();
  assert.equal(JSON.stringify(f.ui.editor.value),original);assert.equal(f.ui.evidenceRepair.value,null);
  assert.match(f.ui.error.value,/没有替换、保存或执行/);assert.equal(f.calls.some(row=>['replan','update','run'].includes(row[0])),false);f.dispose();
});

test('typed refinements round-trip as compose without guessing replacement from repeated question text',async()=>{
  const repair=recomposeRepair(),value=task(),extra=repair.plan.questions.slice(1).map(row=>({...row,kind:'analysis',status:'ready',revision:1,generation:1,artifact_version:0,document_ids:null}));
  value.steps.splice(2,0,...extra);value.steps.at(-1).depends_on=['q1','lookup_q1','verify_lookup'];
  const h=service();await h.client.load(sid);h.handlers.request=async()=>({task:value});const decoded=await h.client.detail(sid,id);
  assert.equal(decoded.steps[2].refine_of,'q1');assert.equal(decoded.steps[2].purpose,undefined);
  assert.equal('refine_of' in decoded.steps[3],false);assert.equal(taskPlan(repair.plan).questions[1].refine_of,'q1');
  const plain={...repair.plan.questions[1]};delete plain.refine_of;
  assert.equal('refine_of' in taskPlan({...plan(),questions:[plan().questions[0],plain]}).questions[1],false);
  extra[0].refine_of='foreign';await assert.rejects(h.client.detail(sid,id),/不符合/);
});

test('refinement can target an old verify, synchronizes inherited requirements recursively, and retains explicit lineage on save',async()=>{
  const original=pairedPlan(),repair=recomposeRepair(original);repair.plan.questions[2].refine_of='v_q1';repair.plan.questions[2].depends_on=['v_q1'];
  const f=mount({detail:async()=>({...pairedTask(),status:'completed',evidence_progress:evidenceProgress()}),previewEvidenceRepair:async()=>repair});
  await f.ui.toggle();await f.ui.openTask(id);await f.ui.repairEvidence();assert.equal(f.ui.error.value,'');
  const [owner,refiner]=f.ui.editor.value.questions;assert.equal(refiner.refine_of,'v_q1');
  owner.question+=' Keep this new full qualification.';owner.document_ids=[unavailableId];
  assert.equal(f.ui.inheritedQuestion(refiner),owner.question);const sent=f.ui.planForEditor(f.ui.editor.value);
  assert.equal(sent.questions.every(row=>row.question===owner.question),true);
  assert.equal(sent.questions.every(row=>JSON.stringify(row.document_ids)===JSON.stringify([unavailableId])),true);
  assert.equal(sent.questions.find(row=>row.id==='lookup_q1').refine_of,'v_q1');
  assert.equal(f.ui.replanPreview.value,null);await f.ui.savePlan();assert.equal(f.calls.at(-1)[0],'previewReplan');
  assert.equal(f.calls.some(row=>row[0]==='run'),false);assert.match(source,/沿用其完整问题、资料或已选记忆/);f.dispose();
});

for(const [label,change] of [
  ['review claiming replacement',p=>p.questions[2].refine_of='q1'],
  ['replacement changing task',p=>{p.questions[1].question='A different task';p.questions[2].question='A different task';}],
  ['replacement omitting original dependency',p=>p.questions[1].depends_on=[]],
  ['replacement broadening selected scope',p=>{p.questions[0].document_ids=[documentId];}],
])test(`typed replacement validation rejects ${label} without dropping the field`,()=>{
  const value=recomposeRepair().plan;change(value);assert.throws(()=>taskPlan(value));
});
