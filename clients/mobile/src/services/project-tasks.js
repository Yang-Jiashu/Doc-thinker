import {getPreferences, request, researchConnectionKey} from './api.js';
import {validResearchExport} from './research-export.js';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const STATES = new Set(['pending','todo','ready','running','completed','blocked','paused','cancelled','stale']);
const bounded = (value, max) => typeof value === 'string' && Array.from(value).length <= max
  && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\ud800-\udfff]/u.test(value);
const integer = (value, min = 0) => Number.isSafeInteger(value) && value >= min;
const digest = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const exact = (value, fields) => value && typeof value === 'object' && !Array.isArray(value)
  && Object.keys(value).length === fields.length && fields.every(key=>Object.hasOwn(value,key));
const utf8Length = value => Array.from(value).reduce((total,char)=>total+(char.codePointAt(0)<=0x7f?1:char.codePointAt(0)<=0x7ff?2:char.codePointAt(0)<=0xffff?3:4),0);
const malformed = () => new Error('长任务响应不符合约定，没有展示旧内容，请手动读取状态。');
const sessionId = value => {if (typeof value !== 'string' || !/^#\d{5}$/.test(value)) throw new Error('请先选择当前服务的项目');return value;};
const taskId = value => {if (!UUID.test(value || '')) throw new Error('长任务编号无效');return value;};
const version = value => {if (!integer(value,1)) throw new Error('报告版本无效');return value;};
function revision(value) {if (!integer(value)) throw new Error('请先读取最新任务版本');return value;}

export const taskStateLabel = value => ({pending:'待开始',todo:'待开始',ready:'等待授权',running:'执行中',completed:'已完成',blocked:'受阻，需确认',paused:'已暂停',cancelled:'已取消',stale:'资料或计划已变化'}[value] || '状态未知');

export function taskPlan(value) {
  if (!value || !bounded(value.goal,1200) || !value.goal.trim() || !bounded(value.constraints,1200)
    || !Array.isArray(value.questions) || !value.questions.length || value.questions.length > 12) {
    throw new Error('填写目标、约束和 1–6 个原始问题，可另有至多 6 个明确列出的证据核对；目标和约束各最多 1200 字符，完整草稿不会截断。');
  }
  const ids = new Set();
  const questions = value.questions.map(item => {
    if (!item || !bounded(item.id,32) || !/^[a-z][a-z0-9_-]*$/.test(item.id) || ['inventory','report'].includes(item.id) || ids.has(item.id)
      || !bounded(item.title ?? '',120) || !bounded(item.question,1800) || !item.question.trim()) {
      throw new Error('每个问题需要唯一编号和 1–1800 字符内容，标题最多 120 字符；请编辑完整草稿。');
    }
    ids.add(item.id);
    const result = {id:item.id,question:item.question,...(item.title?.trim() ? {title:item.title} : {})};
    if(item.purpose!==undefined && !['compose','verify'].includes(item.purpose))throw new Error('问题阶段无效');
    if(item.purpose==='verify'){
      if(!bounded(item.review_of,32) || !item.review_of || item.refine_of!==undefined)throw new Error('证据核对必须指定前序成果，不能同时替换正文');
      result.purpose='verify';result.review_of=item.review_of;
    }else{
      if(item.review_of!==undefined)throw new Error('只有证据核对可指定前序成果');
      if(item.purpose==='compose')result.purpose='compose';
      if(item.refine_of!==undefined){if(!bounded(item.refine_of,32) || !item.refine_of)throw new Error('补查更新必须明确原成果编号');result.refine_of=item.refine_of;}
    }
    if (item.input_kind !== undefined && !['documents','personal'].includes(item.input_kind)) throw new Error('任务输入方式无效');
    const personal = item.input_kind === 'personal', memories = item.memory_ids === undefined ? [] : item.memory_ids;
    if (!Array.isArray(memories) || memories.length > 8 || memories.some(id=>!UUID.test(id))
      || new Set(memories).size !== memories.length || (!personal && memories.length)) throw new Error('请明确选择最多 8 条确认记忆');
    if (personal) {
      if (item.document_ids !== undefined && item.document_ids !== null) throw new Error('个人任务不能同时借用资料范围');
      result.input_kind = 'personal';result.memory_ids = [...memories];
    }
    if (item.document_ids !== undefined && item.document_ids !== null) {
      if (!Array.isArray(item.document_ids) || !item.document_ids.length || item.document_ids.length > 100
        || item.document_ids.some(id=>!UUID.test(id)) || new Set(item.document_ids).size !== item.document_ids.length) throw new Error('问题的资料范围无效，没有扩大为全项目');
      result.document_ids = [...item.document_ids];
    }
    if (item.depends_on !== undefined) {
      if (!Array.isArray(item.depends_on) || item.depends_on.length > 8 || item.depends_on.some(id=>!bounded(id,80))) throw new Error('问题依赖无效');
      result.depends_on = [...item.depends_on];
    }
    return result;
  });
  if(questions.filter(row=>row.purpose!=='verify').length>6 || questions.filter(row=>row.purpose==='verify').length>6)
    throw new Error('最多 1–6 个原始问题和 6 个明确列出的证据核对；不会静默增加调用。');
  for(const row of questions)if(row.purpose==='verify' || row.refine_of!==undefined){
    const target=questions.find(item=>item.id===(row.review_of || row.refine_of));
    if(!target || !row.depends_on?.includes(target.id) || row.question!==target.question
      || (row.input_kind || 'documents')!==(target.input_kind || 'documents')
      || JSON.stringify(row.document_ids || null)!==JSON.stringify(target.document_ids || null)
      || JSON.stringify(row.memory_ids || [])!==JSON.stringify(target.memory_ids || []))
      throw new Error('证据核对或补查更新必须依赖明确的前序成果，并逐字保留原问题与相同输入范围；没有扩大授权。');
  }
  const graph = new Map(questions.map(item=>[item.id,item.depends_on || []])), active = new Set(), done = new Set();
  function visit(id) {
    if(active.has(id))throw new Error('问题依赖形成循环，请先调整；没有保存或执行');
    if(done.has(id))return;
    active.add(id);
    const deps=graph.get(id);
    if(deps.length>6 || new Set(deps).size!==deps.length || deps.some(dep=>dep===id || !graph.has(dep)))throw new Error('问题依赖必须指向本计划中不同的有效问题');
    for(const dep of deps)visit(dep);
    active.delete(id);done.add(id);
  }
  for(const id of graph.keys())visit(id);
  return {goal:value.goal,constraints:value.constraints,questions};
}

function readProposal(value,id,expectedRevision,plan) {
  const fields=['task_id','base_revision','definition_version','proposal_sha256','added','retired','changed','affected',
    'unaffected_completed','blocked','blocked_reason','modified','active_analysis_count','max_analysis_count','execution_authorized'];
  const reasons=new Set(['task_cancelled','active_step','unresolved_outcome','active_run','definition_capacity']);
  const normal=value=>typeof value==='string' && /^[a-z][a-z0-9_-]{0,31}$/.test(value) && !['inventory','report'].includes(value);
  if(!value || Object.keys(value).length!==fields.length || fields.some(key=>!Object.hasOwn(value,key))
    || value.task_id!==id || value.base_revision!==expectedRevision || !integer(value.definition_version,1) || value.definition_version>32
    || typeof value.proposal_sha256!=='string' || !/^[0-9a-f]{64}$/.test(value.proposal_sha256)
    || typeof value.blocked!=='boolean' || typeof value.modified!=='boolean' || value.execution_authorized!==false
    || (value.blocked ? !reasons.has(value.blocked_reason) : value.blocked_reason!==null)
    || value.active_analysis_count!==plan.questions.length || ![6,12].includes(value.max_analysis_count)
    || value.active_analysis_count>value.max_analysis_count || (value.max_analysis_count===6 && plan.questions.some(row=>row.purpose==='verify')))throw malformed();
  for(const key of ['added','retired','changed','affected','unaffected_completed']){
    const list=value[key],max=key==='affected'?26:12;
    if(!Array.isArray(list) || list.length>max || new Set(list).size!==list.length
      || list.some(id=>!normal(id) && !(key==='affected' && ['inventory','report'].includes(id))))throw malformed();
  }
  const ids=new Set(plan.questions.map(item=>item.id)), affected=new Set(value.affected);
  if(value.added.some(id=>!ids.has(id)) || value.retired.some(id=>ids.has(id))
    || [...value.added,...value.retired,...value.changed].some(id=>!affected.has(id))
    || value.unaffected_completed.some(id=>!ids.has(id) || affected.has(id) || value.added.includes(id))
    || (value.modified && !affected.has('report'))
    || (!value.modified && ['added','retired','changed','affected'].some(key=>value[key].length)))throw malformed();
  return {...value,...Object.fromEntries(['added','retired','changed','affected','unaffected_completed'].map(key=>[key,[...value[key]]]))};
}

function readStep(item) {
  if (!item || !bounded(item.id,80) || !item.id || !['inventory','analysis','report'].includes(item.kind)
    || !STATES.has(item.status) || !bounded(item.title ?? '',120) || !bounded(item.question ?? '',2000)
    || !integer(item.revision) || !integer(item.generation)) throw malformed();
  const docs = item.document_ids;
  if (docs !== null && docs !== undefined && (!Array.isArray(docs) || docs.length > 100 || docs.some(id=>!UUID.test(id)))) throw malformed();
  const personal = item.input_kind === 'personal', memories = item.memory_ids === undefined ? [] : item.memory_ids;
  if ((item.input_kind !== undefined && !['documents','personal'].includes(item.input_kind))
    || !Array.isArray(memories) || memories.length > 48 || memories.some(id=>!UUID.test(id))
    || new Set(memories).size !== memories.length || (!personal && memories.length)
    || (personal && docs !== null && docs !== undefined) || (personal && item.kind === 'analysis' && memories.length > 8)) throw malformed();
  if (item.artifact_version !== null && item.artifact_version !== undefined && !integer(item.artifact_version)) throw malformed();
  if(item.depends_on!==undefined && (!Array.isArray(item.depends_on) || item.depends_on.length>(item.kind==='report'?12:8)
    || item.depends_on.some(id=>!bounded(id,80) || !id) || new Set(item.depends_on).size!==item.depends_on.length))throw malformed();
  if(item.purpose!==undefined && (item.kind!=='analysis' || !['compose','verify'].includes(item.purpose)))throw malformed();
  if(item.purpose==='verify' ? !bounded(item.review_of,32) || !item.review_of : item.review_of!==undefined && item.review_of!==null)throw malformed();
  if(item.refine_of!==undefined && item.refine_of!==null && (item.kind!=='analysis' || item.purpose==='verify' || !bounded(item.refine_of,32) || !item.refine_of))throw malformed();
  return {id:item.id,kind:item.kind,title:item.title || '',question:item.question || '',status:item.status,
    revision:item.revision,generation:item.generation,document_ids:docs ? [...docs] : null,
    depends_on:[...(item.depends_on || [])],
    reason:bounded(item.reason,300) ? item.reason : '',artifact_version:item.artifact_version || null,
    ...(item.purpose!==undefined ? {purpose:item.purpose} : {}),...(item.purpose==='verify' ? {review_of:item.review_of} : {}),
    ...(item.refine_of ? {refine_of:item.refine_of} : {}),
    ...(personal ? {input_kind:'personal',memory_ids:[...memories]} : {})};
}
function readEvidenceProgress(value, task) {
  const base=['schema_version','task_id','task_revision','next_action','semantic_verified','goal_verified'];
  if(!value || value.schema_version!==1 || value.task_id!==task.id || value.task_revision!==task.revision
    || value.semantic_verified!==false || value.goal_verified!==false)throw malformed();
  if(value.status==='unavailable'){
    if(!exact(value,[...base,'status']) || value.next_action!=='collect_evidence')throw malformed();
    return {...value};
  }
  const states=['ready_for_review','unresolved','contradicted','not_assessed'];
  const kinds=new Set(['task_requirement','event_relation','event_event','event_path','event_alternatives','event_review']);
  if(!exact(value,[...base,'fingerprint','obligations','work_items','coverage']) || !digest(value.fingerprint)
    || !['review','await_analysis','local_lookup','collect_evidence'].includes(value.next_action)
    || !Array.isArray(value.obligations) || !value.obligations.length || value.obligations.length>600
    || !Array.isArray(value.work_items) || value.work_items.length>600
    || !exact(value.coverage,['total',...states,'goal_verified']) || value.coverage.goal_verified!==false)throw malformed();
  const owners=new Map(task.steps.filter(step=>step.kind==='analysis').map(step=>[step.id,step]));
  function binding(raw,owner,kind,work=false){
    const fields=['task_id','step_id','artifact_version','step_generation','answer_id','input_sha256','receipt_sha256'];
    if(kind==='task_requirement')fields.push('requirements_sha256');
    if(raw && Object.hasOwn(raw,'review_group_sha256'))fields.push('review_group_sha256');
    if(raw && Object.hasOwn(raw,'review_lineage_sha256'))fields.push('review_lineage_sha256');
    if(work && raw && Object.hasOwn(raw,'lookup_policy'))fields.push('lookup_policy');
    const step=owners.get(owner);
    if(!step || !exact(raw,fields) || raw.task_id!==task.id || raw.step_id!==owner
      || !integer(raw.artifact_version) || raw.artifact_version>32 || raw.artifact_version!==(step.artifact_version || 0)
      || raw.step_generation!==step.generation || !integer(raw.step_generation,1)
      || !(raw.answer_id===null || UUID.test(raw.answer_id || '')) || !digest(raw.input_sha256)
      || !(raw.receipt_sha256===null || digest(raw.receipt_sha256))
      || (kind==='task_requirement' && !digest(raw.requirements_sha256))
      || (raw.review_group_sha256!==undefined && !digest(raw.review_group_sha256))
      || (raw.review_lineage_sha256!==undefined && !digest(raw.review_lineage_sha256))
      || (raw.lookup_policy!==undefined && raw.lookup_policy!=='goal_evidence_lookup_v2'))throw malformed();
    return Object.fromEntries(fields.map(key=>[key,raw[key]]));
  }
  const seen=new Map(),counts=Object.fromEntries(states.map(state=>[state,0]));
  const obligations=value.obligations.map(row=>{
    const fields=['id','owner_step_id','kind','requirement_id','semantic_id','status','query','lookup_query','gaps','binding','requirement_sha256'];
    if(!exact(row,fields) || !digest(row.id) || seen.has(row.id) || !owners.has(row.owner_step_id)
      || !kinds.has(row.kind) || !states.includes(row.status) || !bounded(row.requirement_id,32) || !row.requirement_id
      || !bounded(row.semantic_id,128) || !row.semantic_id || !bounded(row.query,2500) || !row.query.trim()
      || !bounded(row.lookup_query,400) || !row.lookup_query.trim() || utf8Length(row.lookup_query)>400
      || !Array.isArray(row.gaps) || row.gaps.length>13 || row.gaps.some(code=>!bounded(code,64) || !code)
      || (row.kind==='task_requirement' ? !digest(row.requirement_sha256) : row.requirement_sha256!==null))throw malformed();
    const bound=binding(row.binding,row.owner_step_id,row.kind);
    if(row.status==='ready_for_review' && (row.gaps.length || !bound.answer_id || !bound.receipt_sha256
      || owners.get(row.owner_step_id).status!=='completed'))throw malformed();
    const result={...row,gaps:[...row.gaps],binding:bound};seen.set(row.id,result);counts[row.status]++;return result;
  });
  if(value.coverage.total!==obligations.length || states.some(state=>!integer(value.coverage[state]) || value.coverage[state]!==counts[state]))throw malformed();
  const covered=new Set(),workIds=new Set();
  const work=value.work_items.map(row=>{
    if(!exact(row,['id','owner_step_id','kind','query','lookup_query','status','attempts','round','binding','obligation_ids'])
      || !digest(row.id) || workIds.has(row.id) || !kinds.has(row.kind) || !owners.has(row.owner_step_id)
      || !bounded(row.query,2500) || !row.query.trim() || !bounded(row.lookup_query,400) || !row.lookup_query.trim()
      || utf8Length(row.lookup_query)>400 || !['unresolved','exhausted','candidates_found','partially_searched'].includes(row.status)
      || !integer(row.attempts) || row.attempts>16 || row.round!==row.attempts
      || (row.status==='unresolved' ? row.attempts!==0 : row.attempts<1)
      || !Array.isArray(row.obligation_ids) || !row.obligation_ids.length || row.obligation_ids.length>600
      || new Set(row.obligation_ids).size!==row.obligation_ids.length)throw malformed();
    const bound=binding(row.binding,row.owner_step_id,row.kind,true),obligationBinding={...bound};delete obligationBinding.lookup_policy;
    for(const id of row.obligation_ids){const obligation=seen.get(id);if(!obligation || covered.has(id)
      || obligation.status==='ready_for_review' || obligation.owner_step_id!==row.owner_step_id || obligation.kind!==row.kind
      || obligation.query!==row.query || obligation.lookup_query!==row.lookup_query
      || JSON.stringify(obligation.binding)!==JSON.stringify(obligationBinding))throw malformed();covered.add(id);}
    workIds.add(row.id);return {...row,binding:bound,obligation_ids:[...row.obligation_ids]};
  });
  if(obligations.some(row=>row.status!=='ready_for_review' && !covered.has(row.id))
    || (work.length===0)!==(value.next_action==='review'))throw malformed();
  return {...value,obligations,work_items:work,coverage:{...value.coverage}};
}
function readTask(value, session, expectedId = null, detail = false) {
  if (!value || !UUID.test(value.id || '') || (expectedId && value.id !== expectedId) || value.session_id !== session
    || !bounded(value.goal,1200) || !bounded(value.constraints,1200) || !STATES.has(value.status)
    || !integer(value.revision) || !integer(value.generation)
    || (value.steps !== undefined && (!Array.isArray(value.steps) || value.steps.length > 14))
    || (detail && !Array.isArray(value.steps))) throw malformed();
  const steps = (value.steps || []).map(readStep);
  if (new Set(steps.map(item=>item.id)).size !== steps.length || steps.filter(item=>item.kind === 'analysis' && item.purpose!=='verify').length > 6
    || steps.filter(item=>item.purpose==='verify').length>6) throw malformed();
  for(const step of steps)if(step.purpose==='verify' || step.refine_of){
    const target=steps.find(row=>row.id===(step.review_of || step.refine_of) && row.kind==='analysis');
    if(!target || !step.depends_on.includes(target.id) || step.question!==target.question
      || (step.input_kind || 'documents')!==(target.input_kind || 'documents')
      || JSON.stringify(step.document_ids)!==JSON.stringify(target.document_ids)
      || JSON.stringify(step.memory_ids || [])!==JSON.stringify(target.memory_ids || []))throw malformed();
  }
  const graph=new Map(steps.map(row=>[row.id,row.depends_on])),active=new Set(),done=new Set();
  function visit(id){if(active.has(id))throw malformed();if(done.has(id))return;active.add(id);
    for(const dependency of graph.get(id)){if(!graph.has(dependency) || dependency===id)throw malformed();visit(dependency);}
    active.delete(id);done.add(id);}
  for(const id of graph.keys())visit(id);
  const versions = value.artifact_versions ?? [];
  if (!Array.isArray(versions) || versions.length > 200 || versions.some(item=>!integer(item?.version,1) || !['current','stale','withdrawn'].includes(item.status))
    || new Set(versions.map(item=>item.version)).size !== versions.length) throw malformed();
  const result={id:value.id,session_id:session,goal:value.goal,constraints:value.constraints,status:value.status,
    reason:bounded(value.reason,300) ? value.reason : '',
    revision:value.revision,generation:value.generation,steps,artifact_versions:versions.map(item=>({version:item.version,status:item.status})),
    run_id:UUID.test(value.run_id || '') ? value.run_id : null,
    run_remaining:integer(value.run_remaining) && value.run_remaining <= 6 ? value.run_remaining : null};
  if(value.evidence_progress!==undefined)result.evidence_progress=readEvidenceProgress(value.evidence_progress,result);
  return result;
}
function readArtifact(value, id, requestedVersion) {
  if (!value || value.task_id !== id || value.step_id !== 'report' || value.version !== requestedVersion || !['current','stale','withdrawn'].includes(value.status)
    || typeof value.available !== 'boolean') throw malformed();
  const available = value.available && value.status === 'current';
  if (available && !validResearchExport(value.content)) throw malformed();
  const hasAcceptance = value.acceptance_status !== undefined || value.goal_verified !== undefined;
  if (hasAcceptance && (value.acceptance_status !== 'review_required' || value.goal_verified !== false)) throw malformed();
  if(value.delivery_status!==undefined && (!hasAcceptance || !['draft_with_evidence_gaps','ready_for_user_review'].includes(value.delivery_status)))throw malformed();
  // Never fall back to cached text, nor reuse private content on an invalidated version.
  return {task_id:id,version:requestedVersion,status:value.status,available,content:available ? value.content : null,
    ...(hasAcceptance ? {acceptance_status:'review_required',goal_verified:false} : {}),
    ...(value.delivery_status!==undefined ? {delivery_status:value.delivery_status} : {})};
}
function readEvidenceRepair(value,id,expectedRevision,original) {
  if(!exact(value,['plan','changed_steps','deferred_work_count','capacity_steps','execution_authorized','automatic_execution','next_action',
    'task_id','base_revision','ledger_fingerprint','proposal']) || value.task_id!==id || value.base_revision!==expectedRevision
    || !digest(value.ledger_fingerprint) || value.execution_authorized!==false || value.automatic_execution!==false
    || !integer(value.deferred_work_count) || value.deferred_work_count>600
    || !['preview_replan','collect_evidence','review'].includes(value.next_action))throw malformed();
  const plan=taskPlan(value.plan),ids=new Set(plan.questions.map(row=>row.id));
  for(const key of ['changed_steps','capacity_steps'])if(!Array.isArray(value[key]) || value[key].length>12
    || new Set(value[key]).size!==value[key].length || value[key].some(id=>!ids.has(id)))throw malformed();
  const proposal=readProposal(value.proposal,id,expectedRevision,plan);
  const changed=[...proposal.changed,...proposal.added],prior=original?.steps.filter(row=>row.kind==='analysis') || [];
  if(!original || original.revision!==expectedRevision || plan.goal!==original.goal || plan.constraints!==original.constraints
    || proposal.retired.length || (proposal.max_analysis_count===12 && proposal.changed.length)
    || value.changed_steps.some(id=>!changed.includes(id)) || changed.some(id=>!value.changed_steps.includes(id))
    || value.capacity_steps.some(id=>value.changed_steps.includes(id))
    || Boolean(value.changed_steps.length)!==(value.next_action==='preview_replan'))throw malformed();
  for(const before of prior){const after=plan.questions.find(row=>row.id===before.id);
    if(!after || (before.title || '')!==(after.title || '') || (before.purpose || 'compose')!==(after.purpose || 'compose')
      || (before.review_of || null)!==(after.review_of || null)
      || (before.refine_of || null)!==(after.refine_of || null)
      || (after.question!==before.question && !(proposal.max_analysis_count===6 && after.question.startsWith(before.question+'\n\n')))
      || (before.input_kind || 'documents')!==(after.input_kind || 'documents')
      || JSON.stringify(before.document_ids || null)!==JSON.stringify(after.document_ids || null)
      || JSON.stringify(before.memory_ids || [])!==JSON.stringify(after.memory_ids || [])
      || JSON.stringify(before.depends_on.filter(id=>id!=='inventory'))!==JSON.stringify(after.depends_on || []))throw malformed();
  }
  const additions=plan.questions.filter(row=>!prior.some(before=>before.id===row.id));
  if(additions.some(row=>!proposal.added.includes(row.id)) || proposal.added.some(id=>!additions.some(row=>row.id===id)))throw malformed();
  for(const row of additions)if(row.purpose!=='verify'){
    const deps=row.depends_on || [],owner=prior.find(before=>before.id===row.refine_of);
    if(deps.length!==1 || deps[0]!==row.refine_of || !owner || row.question!==owner.question
      || (row.input_kind || 'documents')!==(owner.input_kind || 'documents')
      || JSON.stringify(row.document_ids || null)!==JSON.stringify(owner.document_ids || null)
      || JSON.stringify(row.memory_ids || [])!==JSON.stringify(owner.memory_ids || [])
      || !additions.some(review=>review.purpose==='verify' && review.review_of===row.id
        && JSON.stringify(review.depends_on || [])===JSON.stringify([row.id])))throw malformed();
  }
  return {...value,plan,proposal,changed_steps:[...value.changed_steps],capacity_steps:[...value.capacity_steps]};
}
function readDocuments(value, session) {
  if (!Array.isArray(value?.documents) || value.documents.length > 20) throw malformed();
  const items=value.documents.map(item=>{
    if (!item || !UUID.test(item.id || '') || !bounded(item.filename,240) || !['queued','processing','ready','failed','deleted'].includes(item.status)
      || (item.session_id !== undefined && item.session_id !== session)) throw malformed();
    // The scope picker needs identity and state, never chunks or private previews.
    return {id:item.id,filename:item.filename,status:item.status};
  });
  if (new Set(items.map(item=>item.id)).size !== items.length) throw malformed();
  return items;
}
function readMemories(value, session) {
  if (!Array.isArray(value?.items) || value.items.length > 20 || !integer(value.total)
    || value.total < value.items.length || !(value.next_cursor === null || (bounded(value.next_cursor,2048) && value.next_cursor))) throw malformed();
  const seen = new Set(), items = [];
  for (const item of value.items) {
    if (!item || !UUID.test(item.id || '') || seen.has(item.id)
      || (item.session_id !== undefined && item.session_id !== session)
      || ['redacted','archived','historical'].some(key=>item[key] !== undefined && typeof item[key] !== 'boolean')
      || !['active','archived','invalidated','deleted','superseded'].includes(item.status)) throw malformed();
    seen.add(item.id);
    // Ineligible records may carry old text. Never retain it in the picker.
    if (item.status !== 'active' || item.confirmed_by_user !== true || item.redacted === true
      || item.archived === true || item.historical === true || (item.version_status !== undefined && !['active','current'].includes(item.version_status))) continue;
    if (!bounded(item.text,2000) || !item.text.trim() || !['finding','preference','decision'].includes(item.kind)
      || !integer(item.revision,1)) throw malformed();
    items.push({id:item.id,text:item.text,kind:item.kind,revision:item.revision});
  }
  return {items,next_cursor:value.next_cursor};
}

export function createProjectTasksClient(dependencies = {}) {
  const preferences = dependencies.preferences || getPreferences, identity = dependencies.identity || researchConnectionKey;
  const send = dependencies.request || request;
  let permission = null, inFlight = false, reviewed = null;
  const knownTasks=new Map(),remember=(...args)=>{const task=readTask(...args);knownTasks.set(task.id,JSON.parse(JSON.stringify(task)));return task;};
  const guard = (key,current) => {if (key !== identity() || current() !== true) throw new Error('页面、身份或项目已改变，没有继续操作');};
  const allowed = (session,key) => {const p=preferences();if(p.mode !== 'connected' || p.invitePending || permission?.key !== key || permission.session !== session || permission.enabled !== true) throw new Error('当前服务或项目尚未开放长任务，请先读取任务列表');};
  async function once(work) {if (inFlight) throw new Error('正在处理上一次任务操作，请勿重复提交');inFlight=true;try{return await work();}finally{inFlight=false;}}
  async function invoke(path,data,method,key,current) {guard(key,current);const result=await send(path,data,method);guard(key,current);return result;}
  const query = (session,id) => `session_id=${encodeURIComponent(session)}${id ? '&task_id='+encodeURIComponent(id) : ''}`;
  return {
    async load(session,{isCurrent=()=>true}={}) {sessionId(session);return once(async()=>{
      const key=identity(),p=preferences();permission=null;reviewed=null;knownTasks.clear();guard(key,isCurrent);
      if(p.mode !== 'connected' || p.invitePending)return {enabled:false,items:[]};
      let caps;try{caps=await invoke('/capabilities',undefined,'GET',key,isCurrent);}catch(error){if([403,404].includes(error?.status))return {enabled:false,items:[]};throw error;}
      permission={key,session,enabled:caps?.workbench?.tasks === true};
      if(!permission.enabled)return {enabled:false,items:[]};
      const value=await invoke('/workbench/tasks?'+query(session),undefined,'GET',key,isCurrent);
      if(!Array.isArray(value?.items) || value.items.length > 100)throw malformed();
      return {enabled:true,items:value.items.map(item=>remember(item,session))};
    });},
    async detail(session,id,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);return once(async()=>{const key=identity();allowed(session,key);reviewed=null;return remember((await invoke('/workbench/task?'+query(session,id),undefined,'GET',key,isCurrent))?.task,session,id,true);});},
    async documents(session,{isCurrent=()=>true}={}) {sessionId(session);return once(async()=>{const key=identity();allowed(session,key);return readDocuments(await invoke('/research/documents?'+query(session),undefined,'GET',key,isCurrent),session);});},
    async memories(session,{cursor=null,isCurrent=()=>true}={}) {sessionId(session);if(cursor !== null && (!bounded(cursor,2048) || !cursor))throw malformed();return once(async()=>{const key=identity();allowed(session,key);return readMemories(await invoke('/research/memory?'+query(session)+'&limit=20'+(cursor ? '&cursor='+encodeURIComponent(cursor) : ''),undefined,'GET',key,isCurrent),session);});},
    async create(session,value,{isCurrent=()=>true}={}) {sessionId(session);const plan=taskPlan(value);return once(async()=>{const key=identity();allowed(session,key);return remember((await invoke('/workbench/tasks',{session_id:session,...plan},'POST',key,isCurrent))?.task,session,null,true);});},
    async update(session,id,expectedRevision,value,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);revision(expectedRevision);const plan=taskPlan(value);return once(async()=>{const key=identity();allowed(session,key);return remember((await invoke('/workbench/tasks/update',{session_id:session,task_id:id,expected_revision:expectedRevision,...plan},'POST',key,isCurrent))?.task,session,id,true);});},
    async previewReplan(session,id,expectedRevision,value,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);if(!integer(expectedRevision,1))throw malformed();const plan=taskPlan(value);return once(async()=>{const key=identity();allowed(session,key);reviewed=null;const result=readProposal((await invoke('/workbench/tasks/replan/preview',{session_id:session,task_id:id,expected_revision:expectedRevision,...plan},'POST',key,isCurrent))?.proposal,id,expectedRevision,plan);reviewed={key,session,id,revision:expectedRevision,plan:JSON.stringify(plan),proposal:result};return result;});},
    async previewEvidenceRepair(session,id,expectedRevision,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);if(!integer(expectedRevision,1))throw malformed();return once(async()=>{const key=identity();allowed(session,key);reviewed=null;const original=knownTasks.get(id);if(!original?.steps.length || original.revision!==expectedRevision)throw new Error('请先读取最新任务版本，再准备证据补齐草稿');const result=readEvidenceRepair((await invoke('/workbench/tasks/evidence/preview',{session_id:session,task_id:id,expected_revision:expectedRevision},'POST',key,isCurrent))?.repair,id,expectedRevision,original);reviewed={key,session,id,revision:expectedRevision,plan:JSON.stringify(result.plan),proposal:result.proposal};return result;});},
    async replan(session,id,expectedRevision,value,proposalSha256,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);if(!integer(expectedRevision,1))throw malformed();const plan=taskPlan(value);return once(async()=>{const key=identity();allowed(session,key);guard(key,isCurrent);if(!reviewed || reviewed.key!==key || reviewed.session!==session || reviewed.id!==id || reviewed.revision!==expectedRevision || reviewed.plan!==JSON.stringify(plan) || reviewed.proposal.proposal_sha256!==proposalSha256 || reviewed.proposal.blocked)throw new Error('草稿或计划状态已变化，请重新预览影响范围');reviewed=null;return remember((await invoke('/workbench/tasks/replan',{session_id:session,task_id:id,expected_revision:expectedRevision,...plan,proposal_sha256:proposalSha256},'POST',key,isCurrent))?.task,session,id,true);});},
    async run(session,id,expectedRevision,requestId,maxCalls,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);revision(expectedRevision);if(!UUID.test(requestId || '') || !integer(maxCalls,1) || maxCalls > 6)throw new Error('需要安全请求编号和 1–6 次明确调用额度');return once(async()=>{const key=identity();allowed(session,key);return remember((await invoke('/workbench/tasks/run',{session_id:session,task_id:id,expected_revision:expectedRevision,request_id:requestId,max_calls:maxCalls},'POST',key,isCurrent))?.task,session,id,true);});},
    async control(session,id,expectedRevision,action,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);revision(expectedRevision);if(!['pause','cancel','refresh','retry_failed'].includes(action))throw new Error('不支持这个任务操作');return once(async()=>{const key=identity();allowed(session,key);return remember((await invoke('/workbench/tasks/control',{session_id:session,task_id:id,expected_revision:expectedRevision,action},'POST',key,isCurrent))?.task,session,id,true);});},
    async artifact(session,id,reportVersion,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);version(reportVersion);return once(async()=>{const key=identity();allowed(session,key);return readArtifact((await invoke('/workbench/task/artifact?'+query(session,id)+'&version='+reportVersion,undefined,'GET',key,isCurrent))?.artifact,id,reportVersion);});},
  };
}
