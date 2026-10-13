import {getPreferences, request, researchConnectionKey} from './api.js';
import {validResearchExport} from './research-export.js';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const STATES = new Set(['pending','todo','ready','running','completed','blocked','paused','cancelled','stale']);
const bounded = (value, max) => typeof value === 'string' && Array.from(value).length <= max
  && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\ud800-\udfff]/u.test(value);
const integer = (value, min = 0) => Number.isSafeInteger(value) && value >= min;
const malformed = () => new Error('长任务响应不符合约定，没有展示旧内容，请手动读取状态。');
const sessionId = value => {if (typeof value !== 'string' || !/^#\d{5}$/.test(value)) throw new Error('请先选择当前服务的项目');return value;};
const taskId = value => {if (!UUID.test(value || '')) throw new Error('长任务编号无效');return value;};
const version = value => {if (!integer(value,1)) throw new Error('报告版本无效');return value;};
function revision(value) {if (!integer(value)) throw new Error('请先读取最新任务版本');return value;}

export const taskStateLabel = value => ({pending:'待开始',todo:'待开始',ready:'等待授权',running:'执行中',completed:'已完成',blocked:'受阻，需确认',paused:'已暂停',cancelled:'已取消',stale:'资料或计划已变化'}[value] || '状态未知');

export function taskPlan(value) {
  if (!value || !bounded(value.goal,1200) || !value.goal.trim() || !bounded(value.constraints,1200)
    || !Array.isArray(value.questions) || !value.questions.length || value.questions.length > 6) {
    throw new Error('填写目标、约束和 1–6 个研究问题；目标和约束各最多 1200 字符，完整草稿不会截断。');
  }
  const ids = new Set();
  const questions = value.questions.map(item => {
    if (!item || !bounded(item.id,32) || !/^[a-z][a-z0-9_-]*$/.test(item.id) || ['inventory','report'].includes(item.id) || ids.has(item.id)
      || !bounded(item.title ?? '',120) || !bounded(item.question,1800) || !item.question.trim()) {
      throw new Error('每个问题需要唯一编号和 1–1800 字符内容，标题最多 120 字符；请编辑完整草稿。');
    }
    ids.add(item.id);
    const result = {id:item.id,question:item.question,...(item.title?.trim() ? {title:item.title} : {})};
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
  return {goal:value.goal,constraints:value.constraints,questions};
}

function readStep(item) {
  if (!item || !bounded(item.id,80) || !item.id || !['inventory','analysis','report'].includes(item.kind)
    || !STATES.has(item.status) || !bounded(item.title ?? '',120) || !bounded(item.question ?? '',2000)
    || !integer(item.revision) || !integer(item.generation)) throw malformed();
  const docs = item.document_ids;
  if (docs !== null && docs !== undefined && (!Array.isArray(docs) || docs.length > 100 || docs.some(id=>!UUID.test(id)))) throw malformed();
  if (item.artifact_version !== null && item.artifact_version !== undefined && !integer(item.artifact_version)) throw malformed();
  return {id:item.id,kind:item.kind,title:item.title || '',question:item.question || '',status:item.status,
    revision:item.revision,generation:item.generation,document_ids:docs ? [...docs] : null,
    depends_on:Array.isArray(item.depends_on) && item.depends_on.length <= 8 && item.depends_on.every(id=>bounded(id,80)) ? [...item.depends_on] : [],
    reason:bounded(item.reason,300) ? item.reason : '',artifact_version:item.artifact_version || null};
}
function readTask(value, session, expectedId = null, detail = false) {
  if (!value || !UUID.test(value.id || '') || (expectedId && value.id !== expectedId) || value.session_id !== session
    || !bounded(value.goal,1200) || !bounded(value.constraints,1200) || !STATES.has(value.status)
    || !integer(value.revision) || !integer(value.generation)
    || (value.steps !== undefined && (!Array.isArray(value.steps) || value.steps.length > 8))
    || (detail && !Array.isArray(value.steps))) throw malformed();
  const steps = (value.steps || []).map(readStep);
  if (new Set(steps.map(item=>item.id)).size !== steps.length || steps.filter(item=>item.kind === 'analysis').length > 6) throw malformed();
  const versions = value.artifact_versions ?? [];
  if (!Array.isArray(versions) || versions.length > 200 || versions.some(item=>!integer(item?.version,1) || !['current','stale','withdrawn'].includes(item.status))
    || new Set(versions.map(item=>item.version)).size !== versions.length) throw malformed();
  return {id:value.id,session_id:session,goal:value.goal,constraints:value.constraints,status:value.status,
    reason:bounded(value.reason,300) ? value.reason : '',
    revision:value.revision,generation:value.generation,steps,artifact_versions:versions.map(item=>({version:item.version,status:item.status})),
    run_id:UUID.test(value.run_id || '') ? value.run_id : null,
    run_remaining:integer(value.run_remaining) && value.run_remaining <= 6 ? value.run_remaining : null};
}
function readArtifact(value, id, requestedVersion) {
  if (!value || value.task_id !== id || value.step_id !== 'report' || value.version !== requestedVersion || !['current','stale','withdrawn'].includes(value.status)
    || typeof value.available !== 'boolean') throw malformed();
  const available = value.available && value.status === 'current';
  if (available && !validResearchExport(value.content)) throw malformed();
  // Never fall back to cached text, nor reuse private content on an invalidated version.
  return {task_id:id,version:requestedVersion,status:value.status,available,content:available ? value.content : null};
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

export function createProjectTasksClient(dependencies = {}) {
  const preferences = dependencies.preferences || getPreferences, identity = dependencies.identity || researchConnectionKey;
  const send = dependencies.request || request;
  let permission = null, inFlight = false;
  const guard = (key,current) => {if (key !== identity() || current() !== true) throw new Error('页面、身份或项目已改变，没有继续操作');};
  const allowed = (session,key) => {const p=preferences();if(p.mode !== 'connected' || p.invitePending || permission?.key !== key || permission.session !== session || permission.enabled !== true) throw new Error('当前服务或项目尚未开放长任务，请先读取任务列表');};
  async function once(work) {if (inFlight) throw new Error('正在处理上一次任务操作，请勿重复提交');inFlight=true;try{return await work();}finally{inFlight=false;}}
  async function invoke(path,data,method,key,current) {guard(key,current);const result=await send(path,data,method);guard(key,current);return result;}
  const query = (session,id) => `session_id=${encodeURIComponent(session)}${id ? '&task_id='+encodeURIComponent(id) : ''}`;
  return {
    async load(session,{isCurrent=()=>true}={}) {sessionId(session);return once(async()=>{
      const key=identity(),p=preferences();permission=null;guard(key,isCurrent);
      if(p.mode !== 'connected' || p.invitePending)return {enabled:false,items:[]};
      let caps;try{caps=await invoke('/capabilities',undefined,'GET',key,isCurrent);}catch(error){if([403,404].includes(error?.status))return {enabled:false,items:[]};throw error;}
      permission={key,session,enabled:caps?.workbench?.tasks === true};
      if(!permission.enabled)return {enabled:false,items:[]};
      const value=await invoke('/workbench/tasks?'+query(session),undefined,'GET',key,isCurrent);
      if(!Array.isArray(value?.items) || value.items.length > 100)throw malformed();
      return {enabled:true,items:value.items.map(item=>readTask(item,session))};
    });},
    async detail(session,id,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);return once(async()=>{const key=identity();allowed(session,key);return readTask((await invoke('/workbench/task?'+query(session,id),undefined,'GET',key,isCurrent))?.task,session,id,true);});},
    async documents(session,{isCurrent=()=>true}={}) {sessionId(session);return once(async()=>{const key=identity();allowed(session,key);return readDocuments(await invoke('/research/documents?'+query(session),undefined,'GET',key,isCurrent),session);});},
    async create(session,value,{isCurrent=()=>true}={}) {sessionId(session);const plan=taskPlan(value);return once(async()=>{const key=identity();allowed(session,key);return readTask((await invoke('/workbench/tasks',{session_id:session,...plan},'POST',key,isCurrent))?.task,session,null,true);});},
    async update(session,id,expectedRevision,value,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);revision(expectedRevision);const plan=taskPlan(value);return once(async()=>{const key=identity();allowed(session,key);return readTask((await invoke('/workbench/tasks/update',{session_id:session,task_id:id,expected_revision:expectedRevision,...plan},'POST',key,isCurrent))?.task,session,id,true);});},
    async run(session,id,expectedRevision,requestId,maxCalls,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);revision(expectedRevision);if(!UUID.test(requestId || '') || !integer(maxCalls,1) || maxCalls > 6)throw new Error('需要安全请求编号和 1–6 次明确调用额度');return once(async()=>{const key=identity();allowed(session,key);return readTask((await invoke('/workbench/tasks/run',{session_id:session,task_id:id,expected_revision:expectedRevision,request_id:requestId,max_calls:maxCalls},'POST',key,isCurrent))?.task,session,id,true);});},
    async control(session,id,expectedRevision,action,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);revision(expectedRevision);if(!['pause','cancel','refresh','retry_failed'].includes(action))throw new Error('不支持这个任务操作');return once(async()=>{const key=identity();allowed(session,key);return readTask((await invoke('/workbench/tasks/control',{session_id:session,task_id:id,expected_revision:expectedRevision,action},'POST',key,isCurrent))?.task,session,id,true);});},
    async artifact(session,id,reportVersion,{isCurrent=()=>true}={}) {sessionId(session);taskId(id);version(reportVersion);return once(async()=>{const key=identity();allowed(session,key);return readArtifact((await invoke('/workbench/task/artifact?'+query(session,id)+'&version='+reportVersion,undefined,'GET',key,isCurrent))?.artifact,id,reportVersion);});},
  };
}
