import { getPreferences, request, researchConnectionKey, researchAnswerMetadata, isFeedbackAnswerId } from './api.js';
import { boundedAnswerSources } from './research-source-contract.js';

// These recipes only prepare editable text. They never run a model or a tool.
export const taskRecipes = Object.freeze([
  {id:'compare',title:'对比资料',symbol:'⌘',detail:'观点、证据与分歧，一起看',prompt:'请比较当前项目资料中的主要观点、方法和证据。按相同维度组织，保留实验条件与局限，标明原文出处；没有找到的内容直接说明，不用背景介绍补足结论。'},
  {id:'write',title:'带依据写作',symbol:'↗',detail:'写成可审阅、有出处的初稿',prompt:'请帮我围绕当前项目写一份初稿。先确认读者、用途和篇幅，再使用当前资料组织论点与依据；把原文支持、我的要求和待验证推测分开，不编造引用。'},
  {id:'continue',title:'继续项目',symbol:'→',detail:'从已确认的进展继续推进',prompt:'请根据这段对话及允许使用的已确认项目记忆，梳理目前目标、已完成的进展、仍缺的依据和下一步。缺少的信息先问我，不把未经确认的旧回答当作事实，也不要声称已执行任何外部操作。'},
  {id:'review',title:'复核结论',symbol:'◎',detail:'找条件、反证和遗漏的一环',prompt:'请帮我复核一个结论。先让我提供要核对的文字，再逐条检查资料支持、适用条件、反例和关键证据缺口。类比只能提供参考，不能证明结论；无法核验的部分明确保留。'},
].map(item=>Object.freeze(item)));
export function recipeDraft(id) { return taskRecipes.find(item=>item.id === id)?.prompt || ''; }
export function briefDraft(value) {
  const fields = [['项目目标',value?.goal],['预期交付',value?.output_format],['约束与边界',value?.constraints]];
  const parts = fields.filter(([,text])=>typeof text === 'string' && text.trim()).map(([label,text])=>`${label}：${text.trim()}`);
  return parts.length ? `【由我提供的项目简报，不是资料证据】\n${parts.join('\n')}\n\n请结合以上要求帮助我推进本次问题；信息不足时先确认。` : '';
}
export function requestReceipt(status, {asking = false, preparing = false, demo = false} = {}) {
  if (preparing) return {title:'正在准备请求',detail:'还在本机准备安全编号，没有宣称模型已开始。'};
  if (asking) return {title:demo ? '正在组织模拟回答' : '正在等待服务响应',detail:demo ? '演示不调用真实模型。' : '已发送本次问题；服务没有提供细分进度，不推测正在执行哪个步骤。'};
  const states = {
    running:{title:'服务确认仍在处理',detail:'可以查询同一请求的状态；不会重新请求模型。'},
    completed:{title:'已收到回答',detail:'生成完成不代表结论已核验，请审阅回答和依据。'},
    failed:{title:'本次提问已结束 · 未生成完整回答',detail:'服务已收到模型结果，但未作为有效回答保存；不会自动重发，同一编号不会再次调用模型。'},
    not_found:{title:'暂未找到执行记录',detail:'不等于模型未执行，请先核对历史与请求状态。'},
    unknown:{title:'执行结果尚未确认',detail:'可能断线或重启，系统不会自动重发。'},
    unconfirmed:{title:'执行结果尚未确认',detail:'可能断线或重启，系统不会自动重发。'},
  };
  return states[status] || null;
}
const bounded = (text,max) => typeof text === 'string' && Array.from(text).length <= max
  && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\ud800-\udfff]/u.test(text);
const malformed = () => new Error('项目工作台响应不符合约定，请刷新后重试。');
const sid = value => { if (typeof value !== 'string' || !/^#\d{5}$/.test(value)) throw new Error('请选择当前服务的项目'); return value; };
const fields = value => {
  if (!value || !bounded(value.goal,1200) || !bounded(value.output_format,400) || !bounded(value.constraints,1200)) throw new Error('简报字段超出范围或含无效字符，完整草稿仍保留');
  return {goal:value.goal,output_format:value.output_format,constraints:value.constraints};
};
function readBrief(value,session) {
  if (!value || value.session_id !== session || !Number.isSafeInteger(value.revision) || value.revision < 0) throw malformed();
  let content; try { content = fields(value); } catch { throw malformed(); }
  return {...content,session_id:session,revision:value.revision,updated_at:typeof value.updated_at === 'number' && Number.isFinite(value.updated_at) ? value.updated_at : null};
}
function readArtifactMetadata(value) {
  if (!value || !isFeedbackAnswerId(value.answer_id,false) || !bounded(value.title,120) || !value.title.trim() || typeof value.available !== 'boolean') throw malformed();
  return {answer_id:value.answer_id,title:value.title,available:value.available};
}
function readArtifact(value) {
  const metadata=readArtifactMetadata(value);
  const raw = value.answer;
  if (value.available && (!raw || raw.id !== value.answer_id || raw.role !== 'assistant' || !bounded(raw.content,100000))) throw malformed();
  return {...metadata,
    answer:value.available ? {id:raw.id,role:'assistant',content:raw.content,sources:boundedAnswerSources(raw.sources,raw.research_core === true),...researchAnswerMetadata(raw)} : null};
}
export function createWorkbenchClient(dependencies = {}) {
  const preferences = dependencies.preferences || getPreferences, identity = dependencies.identity || researchConnectionKey;
  const send = dependencies.request || request;
  let permission = null;
  const guard = (key,current) => { if (key !== identity() || current() !== true) throw new Error('页面、身份或项目已改变，没有继续操作'); };
  const check = (name,key) => { if (permission?.key !== key || permission[name] !== true) throw new Error('当前服务未开放此工作台功能，没有修改内容'); };
  async function invoke(path,data,method,key,current) { guard(key,current); const value = await send(path,data,method);guard(key,current);return value; }
  return {
    async load(session,{isCurrent = () => true} = {}) {
      sid(session); const key = identity(), p = preferences(); permission = null;
      if (p.mode !== 'connected' || p.invitePending) return {enabled:false,artifactsEnabled:false,brief:null,items:[]};
      let caps;
      try { caps = await invoke('/capabilities',undefined,'GET',key,isCurrent); }
      catch(error) { if ([403,404].includes(error?.status)) return {enabled:false,artifactsEnabled:false,brief:null,items:[]}; throw error; }
      permission = {key,brief:caps?.workbench?.brief === true,artifacts:caps?.workbench?.artifacts === true};
      const result = {enabled:permission.brief,artifactsEnabled:permission.artifacts,brief:null,items:[]};
      if (result.enabled) result.brief = readBrief((await invoke('/workbench/brief?session_id='+encodeURIComponent(session),undefined,'GET',key,isCurrent)).brief,session);
      if (result.artifactsEnabled) {
        const response = await invoke('/workbench/artifacts?session_id='+encodeURIComponent(session),undefined,'GET',key,isCurrent);
        if (!Array.isArray(response.items) || response.items.length > 100) throw malformed();
        result.items = response.items.map(readArtifactMetadata);
      }
      return result;
    },
    async artifact(session,answerId,{isCurrent = () => true} = {}) {
      sid(session);const key=identity();check('artifacts',key);if(!isFeedbackAnswerId(answerId,false))throw new Error('成果编号无效');
      const result=readArtifact((await invoke('/workbench/artifact?session_id='+encodeURIComponent(session)+'&answer_id='+encodeURIComponent(answerId),undefined,'GET',key,isCurrent)).artifact);
      if(result.answer_id!==answerId)throw malformed();return result;
    },
    async saveBrief(session,value,{isCurrent = () => true} = {}) {
      sid(session); const key=identity();check('brief',key);const content=fields(value);
      if (!Number.isSafeInteger(value.revision) || value.revision < 0) throw new Error('请先读取简报版本');
      return readBrief((await invoke('/workbench/brief',{session_id:session,expected_revision:value.revision,...content},'POST',key,isCurrent)).brief,session);
    },
    async saveArtifact(session,answerId,title,{isCurrent = () => true} = {}) {
      sid(session);const key=identity();check('artifacts',key);
      if (!isFeedbackAnswerId(answerId,false) || !bounded(title,120) || !title.trim()) throw new Error('请选择已保存的回答，并填写 1–120 字符标题');
      return readArtifact((await invoke('/workbench/artifacts',{session_id:session,answer_id:answerId,title:title.trim()},'POST',key,isCurrent)).artifact);
    },
    async removeArtifact(session,answerId,{isCurrent = () => true} = {}) {
      sid(session);const key=identity();check('artifacts',key);if (!isFeedbackAnswerId(answerId,false)) throw new Error('成果编号无效');
      const result = await invoke('/workbench/artifacts/delete',{session_id:session,answer_id:answerId},'POST',key,isCurrent);
      if (result?.removed !== true) throw malformed();return true;
    },
  };
}
