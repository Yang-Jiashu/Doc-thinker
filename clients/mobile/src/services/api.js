import { ensureWeChatPrivacy } from './wechat-privacy.js';
import { sanitizeReasoningReceipts } from './reasoning-evidence.js';
import { knowledgeSkillMetadata } from './knowledge-skills.js';
import { MAX_RESEARCH_SOURCES, isResearchSourceLabel, boundedAnswerSources } from './research-source-contract.js';

const STORAGE = 'docthinker.mobile.preferences.v1';
const defaults = { mode: 'demo', baseUrl: '', accessToken: '', useMemory: true, useContext: true, useCache: true, useEvolution: false };
let accessToken = '';
let settingsRevision = 0;
let connectionCheck = null;
let inviteSession = null;
let inviteActiveSession = '';
let identityRevision = 0;
const pendingRequests = new Set();
// A status read can outlive the documents tab. The gateway has one ordinary
// request slot per user, so the next page must wait for that read to settle.
// This is only a barrier for opt-in background document GETs, not a global queue.
let backgroundResearchRead = null;
const queryTracking = new Map();
export const INVITE_MAX_QUESTION_BYTES = 8192;
const QUERY_REQUEST_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
export function questionByteLength(value) {
  if (typeof value !== 'string') return Infinity;
  let size = 0;
  for (const character of value) {
    const point = character.codePointAt(0);
    if (point >= 0xd800 && point <= 0xdfff) return Infinity;
    size += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return size;
}
function queryIdFromBytes(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length !== 16) throw new Error('安全随机数响应不符合协议，没有发送问题。');
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
  const value = [...bytes].map(item=>item.toString(16).padStart(2,'0')).join('');
  return `${value.slice(0,8)}-${value.slice(8,12)}-${value.slice(12,16)}-${value.slice(16,20)}-${value.slice(20)}`;
}
function weChatQueryRandom() {
  // Tencent's SDK >=2.15.0 API supplies cryptographically secure ArrayBuffer
  // bytes asynchronously. Compile it out of H5/App; do not guess App APIs.
  // #ifdef MP-WEIXIN
  if (typeof wx !== 'undefined' && typeof wx.getRandomValues === 'function') return wx;
  // #endif
  return null;
}
export function queryRequestIdSupported() {
  const random = globalThis.crypto;
  return typeof random?.randomUUID === 'function' || typeof random?.getRandomValues === 'function' || weChatQueryRandom() !== null;
}
export function createQueryRequestId() {
  // This opaque marker is not an authentication token. Invitation runs on an
  // HTTPS H5 origin (or localhost), as does private research; never silently
  // weaken the random source when another client platform lacks this API.
  const random = globalThis.crypto;
  if (typeof random?.randomUUID === 'function') {
    const value = random.randomUUID();
    if (typeof value === 'string' && QUERY_REQUEST_ID.test(value)) return value;
  }
  if (typeof random?.getRandomValues === 'function') {
    const bytes = random.getRandomValues(new Uint8Array(16));
    return queryIdFromBytes(bytes);
  }
  throw new Error('当前浏览器无法创建安全请求编号，请使用现代浏览器打开 HTTPS 网站。没有发送问题。');
}
export async function createQueryRequestIdAsync({isCurrent = () => true} = {}) {
  const active = () => { try { return isCurrent() === true; } catch { return false; } };
  const cancelled = () => new Error('页面、身份或草稿已改变，没有发送问题。');
  if (!active()) throw cancelled();
  const random = globalThis.crypto;
  if (typeof random?.randomUUID === 'function' || typeof random?.getRandomValues === 'function') return createQueryRequestId();
  const native = weChatQueryRandom();
  if (!native) throw new Error('本版尚未接入当前环境的安全请求编号，请使用现代浏览器打开 HTTPS 网站。草稿已保留，没有发送问题。');
  return new Promise((resolve,reject) => {
    let done = false;
    const finish = (callback,value) => { if (done) return; done = true; clearTimeout(timer); callback(value); };
    const failed = () => finish(reject,new Error('安全请求编号未能生成，请稍后手动重试或通过浏览器使用。草稿已保留，没有发送问题。'));
    const timer = setTimeout(failed,8000);
    try {
      native.getRandomValues({length:16,success:result => {
        if (!active()) return finish(reject,cancelled());
        if (!(result?.randomValues instanceof ArrayBuffer) || result.randomValues.byteLength !== 16) return failed();
        try { finish(resolve,queryIdFromBytes(new Uint8Array(result.randomValues).slice())); }
        catch { failed(); }
      },fail:failed});
    } catch { failed(); }
  });
}
function knownQueryFailure(value) {
  // Display only known local copy, never provider text or retry instructions.
  const messages = {
    model_response_truncated:'模型达到本次输出上限，未完成回答。没有自动重试；草稿已保留。',
    model_response_empty:'模型已返回，但没有提供可用的回答正文。没有自动重试；草稿已保留。',
    model_response_invalid:'模型返回的结果不符合完整文本回答格式。没有自动重试；草稿已保留。',
  };
  const code = value?.code;
  if (typeof code !== 'string' || !Object.hasOwn(messages,code) || value?.automatic_retry !== false) {
    throw new Error('请求失败回执不符合协议，请查询状态；没有自动重发。');
  }
  return {code,message:messages[code],automatic_retry:false};
}
function trackedQueryResponse(result, requestId) {
  if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error('请求状态响应不符合协议，请查看历史；没有自动重发。');
  if (result.request_id !== requestId || !['running','unknown','not_found','completed','failed'].includes(result.request_status)
    || (result.request_status === 'completed' && (typeof result.answer !== 'string' || !result.answer.trim() || result.answer.length > 100000))) {
    throw new Error('请求状态响应不符合协议，请查看历史；没有自动重发。');
  }
  const data = {request_id:requestId,request_status:result.request_status};
  if (result.request_status === 'failed') {
    // Display only known local copy. Upstream details/partial answers never
    // become assistant messages, HTML, retry authority or provider diagnostics.
    data.request_error = knownQueryFailure(result.request_error);
  }
  if (result.request_status === 'completed') {
    data.answer = result.answer; data.sources = boundedAnswerSources(result.sources,result.research_core === true);
    Object.assign(data, researchAnswerMetadata(result));
    if (isFeedbackAnswerId(result.answer_id)) data.answer_id = result.answer_id;
    if (typeof result.context_windowed === 'boolean') data.context_windowed = result.context_windowed;
  }
  return data;
}
export function researchAnswerMetadata(result) {
  if (result?.research_core !== true) return {};
  const value = result.citation_validation;
  const gapCodes = ['bounded_overview','excerpt_truncated','uncovered_query_terms','documents_not_selected',
    'literal_coverage_scan_limited','source_boundary_unverified','source_antecedent_unverified',
    'context_budget_omitted','confirmed_memory_budget_omitted','complete_statement_exceeds_budget',
    'semantic_retrieval_unavailable','semantic_retrieval_windowed','evidence_coverage_proxy',
    'evidence_facets_uncovered','evidence_anchors_uncovered','insufficient_evidence','duplicate_chunks_removed',
    'graph_association_not_proof','graph_window_limited','graph_hub_omitted','graph_search_limited',
    'graph_no_source_seed','graph_no_connected_path','graph_capacity_reached','graph_source_unavailable',
    'graph_counterevidence_in_path','graph_path_conditions_differ','graph_parser_is_partial',
    'graph_excerpt_windowed','graph_path_window_incomplete'];
  const gaps = Array.isArray(result.retrieval_gaps) ? [...new Set(result.retrieval_gaps.slice(0,32)
    .map(gap=>gap?.code).filter(code=>gapCodes.includes(code)))].map(code=>({code})) : [];
  const knowledgeSkill = knowledgeSkillMetadata(result.knowledge_skill);
  return {research_core:true,...researchReasoningMetadata(result),...(knowledgeSkill ? {knowledge_skill:knowledgeSkill} : {}),
    memory_as_of:typeof result.memory_as_of === 'number' && Number.isFinite(result.memory_as_of) && result.memory_as_of >= 0 && result.memory_as_of <= 253402300799 ? result.memory_as_of : null,
    answer_mode:['evidence','explore','general','memory','insufficient_evidence'].includes(result.answer_mode) ? result.answer_mode : 'general',
    used_memory_count:Number.isInteger(result.used_memory_count) && result.used_memory_count >= 0 && result.used_memory_count <= 8 ? result.used_memory_count : 0,
    citation_validation:{missing_citations:value?.missing_citations === true, entailment_verified:false,
      unknown_labels:Array.isArray(value?.unknown_labels) ? value.unknown_labels.filter(label=>typeof label === 'string' && /^[SK][0-9]{1,16}$/i.test(label)).slice(0,32) : [],
      warnings:Array.isArray(value?.warnings) ? value.warnings.filter(warning=>['source_revoked','provenance_unavailable'].includes(warning?.code)).map(warning=>({code:warning.code})).slice(0,8) : []},
    retrieval_gaps:gaps};
}
// A compact display contract, never a semantic-proof badge. Quotations must
// still match a live source and its absolute offsets when rendered or copied.
export function researchReasoningMetadata(result) {
  const text = (value,max=800) => typeof value === 'string' ? Array.from(value.slice(0,max*2)).slice(0,max).join('').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u202a-\u202e\u2066-\u2069]/g,'') : '';
  const list = (value,max) => Array.isArray(value) ? value.slice(0,max) : [];
  const stale = list(result?.citation_validation?.warnings,8).some(item=>['source_revoked','provenance_unavailable'].includes(item?.code));
  const reasoning = result?.reasoning, audit = result?.claim_audit;
  const receipts = !stale && sanitizeReasoningReceipts(reasoning?.trace,result.sources);
  const out = {};
  if (reasoning && ['context_assist','auto','evidence','analogy','deduction','induction'].includes(reasoning.mode)) out.reasoning = {
    mode:reasoning.mode,status:stale ? 'source_unavailable' : text(reasoning.status,60),
    ...(receipts ? {trace:receipts} : {}),
    display_limited:reasoning.display_limited === true || (Array.isArray(reasoning.conclusions) && reasoning.conclusions.length > 12)
      || list(reasoning.conclusions,12).some(item=>(Array.isArray(item?.proof) && item.proof.length > 24) || (Array.isArray(item?.mappings) && item.mappings.length > 8)),
    conclusions:stale ? [] : list(reasoning.conclusions,12).map(item=>({id:text(item?.id,80),text:text(item?.text),kind:text(item?.kind,60),
      premise_ids:list(item?.premise_ids,12).map(value=>text(value,80)),source_ids:list(item?.source_ids,MAX_RESEARCH_SOURCES).filter(value=>typeof value === 'string' && QUERY_REQUEST_ID.test(value)),
      mappings:list(item?.mappings,8).map(value=>typeof value === 'string' ? text(value,300) : {from:text(value?.from ?? value?.source ?? value?.base,160),to:text(value?.to ?? value?.target,160)}),
      differences:list(item?.differences,8).map(value=>typeof value === 'string' ? text(value,300) : text(value?.text ?? value?.message,300) || ({unmatched_base_relation:'来源中有关系尚未在目标案例观察到。',different_conditions:`成立条件不同：来源 ${list(value?.base_conditions,4).map(x=>text(x,80)).join('、') || '未指定'}；目标 ${list(value?.target_conditions,4).map(x=>text(x,80)).join('、') || '未指定'}。`,role_types_not_fully_specified:'部分角色类型没有明确给出，结构相似不能保证可迁移。'})[value?.code] || ''),
      candidate_inferences:list(item?.candidate_inferences,4).map(value=>({text:text(value?.text),conditions:list(value?.conditions,4).map(x=>text(x,160))})).filter(value=>value.text),
      proof:list(item?.proof,24).map(value=>({text:text(value?.text,500),kind:text(value?.kind,40),source_ids:list(value?.source_ids,MAX_RESEARCH_SOURCES).filter(id=>typeof id === 'string' && QUERY_REQUEST_ID.test(id))})).filter(value=>value.text),
      support_count:Number.isSafeInteger(item?.support_count) && item.support_count >= 0 ? item.support_count : null,
      counterexample_count:Number.isSafeInteger(item?.counterexample_count) && item.counterexample_count >= 0 ? item.counterexample_count : null})).filter(item=>item.text),
    gaps:list(reasoning.gaps,8).map(item=>({code:text(item?.code,80),message:text(item?.message,500)})).filter(item=>item.message),
  };
  if (audit && ['quoted_only','review_needed','unavailable','no_claims'].includes(audit.status)) out.claim_audit = {
    status:stale ? 'unavailable' : audit.status,entailment_verified:false,scope:'literal_and_consistency_checks',
    summary:{omitted_statement_count:Number.isSafeInteger(audit.summary?.omitted_statement_count)
      && audit.summary.omitted_statement_count >= 0 && audit.summary.omitted_statement_count <= 100000
      ? audit.summary.omitted_statement_count : null},
    claims:list(audit.claims,12).map(item=>{
      const refs=stale ? [] : list(item?.source_refs,3).filter(value=>isResearchSourceLabel(value?.label) && QUERY_REQUEST_ID.test(value?.source_id)
        && typeof value?.quote === 'string' && value.quote.length > 0 && value.quote.length <= 2000 && Number.isSafeInteger(value.start) && Number.isSafeInteger(value.end) && value.start >= 0 && value.end > value.start)
        .map(value=>({label:value.label,source_id:value.source_id,quote:value.quote,start:value.start,end:value.end}));
      // Dropping a bad or out-of-window reference must never make the remaining
      // single reference look like a fully checked multi-source quotation.
      const complete=Array.isArray(item?.source_refs) && item.source_refs.length > 0 && item.source_refs.length <= 3
        && refs.length === item.source_refs.length && new Set(refs.map(value=>value.label)).size === refs.length
        && Array.isArray(item?.issues) && item.issues.length === 0;
      return {id:text(item?.id,80),text:text(item?.text),
        status:stale ? 'source_unavailable' : item?.status === 'exact_quote' ? complete ? 'exact_quote' : 'unverified'
          : ['unverified','source_unavailable'].includes(item?.status) ? item.status : 'unverified',
        source_refs:refs,issues:list(item?.issues,8).map(value=>({code:text(value?.code,80),message:text(value?.message,300)})).filter(value=>value.message)};
    }).filter(item=>item.text),
  };
  return out;
}
const fallback = new Map();
const read = key => { try { return typeof uni !== 'undefined' ? uni.getStorageSync(key) || fallback.get(key) : fallback.get(key); } catch { return fallback.get(key); } };
const write = (key, value) => { if (typeof uni !== 'undefined') uni.setStorageSync(key, value); fallback.set(key, value); };

export function normalizeBaseUrl(value) {
  const base = String(value || '').trim().replace(/\/+$/, '');
  if (!base) return '';
  // No credentials, path/query, fragments, unicode host ambiguity or token-in-URL.
  const match = /^(https?):\/\/([a-z0-9.-]+)(?::([0-9]{1,5}))?$/i.exec(base);
  if (!match || (match[3] && (+match[3] < 1 || +match[3] > 65535))) throw new Error('请填写服务地址，例如 https://example.com，不带路径或密钥');
  if (match[1].toLowerCase() === 'http' && !['localhost', '127.0.0.1'].includes(match[2].toLowerCase())) throw new Error('手机连接请使用 HTTPS；HTTP 仅允许本机开发');
  return base.toLowerCase();
}

export function getPreferences() {
  const stored = read(STORAGE);
  const value = stored && typeof stored === 'object' ? stored : {};
  if (value.connectionKind === 'invite') {
    let baseUrl = '';
    try { baseUrl = inviteOrigin(); } catch { /* Pending login must never fall back to another service. */ }
    const active = inviteSession && inviteSession.expiresAt > Date.now() && inviteSession.origin === baseUrl;
    return { ...defaults, ...value, mode:'connected', connectionKind:'invite', baseUrl,
      accessToken:active ? inviteSession.token : '', invitePending:!active,
      inviteUser:active ? {...inviteSession.user} : null, useMemory:false,useEvolution:false,useCache:false };
  }
  return { ...defaults, ...value, mode: value.mode === 'connected' ? 'connected' : 'demo', accessToken };
}

// Only this web origin can receive invited users' credentials or content.
export function inviteOrigin() {
  // #ifdef H5
  if (typeof window !== 'undefined' && window.location?.origin) return normalizeBaseUrl(window.location.origin);
  // #endif
  throw new Error('邀请登录仅支持 HTTPS 网页或本机开发网页，请用浏览器打开邀请入口');
}
function invalidateIdentity() {
  identityRevision += 1;
  queryTracking.clear();
  cancelConnectionCheck();
  for (const cancel of [...pendingRequests]) cancel();
}
export function selectInviteMode() {
  const origin = inviteOrigin(), before = getPreferences();
  if (before.connectionKind === 'invite') return before;
  invalidateIdentity(); accessToken = ''; inviteSession = null; inviteActiveSession = '';
  const next = {mode:'connected',connectionKind:'invite',baseUrl:origin};
  for (const key of ['useMemory','useContext','useCache','useEvolution']) next[key] = before[key];
  write(STORAGE,next); settingsRevision += 1;
  return getPreferences();
}
export function activateInviteSession({access_token, expires_in, user}) {
  const origin = inviteOrigin();
  if (typeof access_token !== 'string' || !/^[A-Za-z0-9._~-]{16,4096}$/.test(access_token)
    || !Number.isInteger(expires_in) || expires_in < 1 || expires_in > 604800
    || typeof user?.id !== 'string' || !user.id || user.id.length > 128
    || typeof user.username !== 'string' || !/^[A-Za-z0-9_-]{3,32}$/.test(user.username)) throw new Error('登录响应不符合协议，请联系运营者');
  selectInviteMode(); invalidateIdentity(); inviteActiveSession = '';
  inviteSession = {token:access_token,expiresAt:Date.now()+expires_in*1000,origin,user:{id:user.id,username:user.username}};
  return getPreferences();
}
export function clearInviteSession({leave = false} = {}) {
  invalidateIdentity(); inviteSession = null; inviteActiveSession = ''; accessToken = ''; settingsRevision += 1;
  if (leave) {
    const before = getPreferences(), next = {mode:'demo',baseUrl:''};
    for (const key of ['useMemory','useContext','useCache','useEvolution']) next[key] = before[key];
    write(STORAGE,next);
  }
}

export function savePreferences(values) {
  const before = getPreferences();
  if (before.connectionKind === 'invite') {
    if ((values.mode && values.mode !== 'connected') || (values.baseUrl && values.baseUrl !== before.baseUrl)
      || (values.accessToken && values.accessToken !== before.accessToken)) throw new Error('邀请账号只能连接当前网站，请先在账号页退出邀请模式');
    const next = {mode:'connected',connectionKind:'invite',baseUrl:before.baseUrl};
    for (const key of ['useMemory','useContext','useCache','useEvolution']) next[key] = typeof values[key] === 'boolean' ? values[key] : before[key];
    write(STORAGE,next); settingsRevision += 1; cancelConnectionCheck();
    return getPreferences();
  }
  const baseUrl = normalizeBaseUrl(values.baseUrl ?? before.baseUrl);
  const mode = values.mode === 'connected' ? 'connected' : 'demo';
  let nextToken = String(values.accessToken ?? accessToken).trim();
  if (baseUrl !== before.baseUrl && nextToken === accessToken) nextToken = '';
  if (mode !== before.mode || baseUrl !== before.baseUrl || nextToken !== accessToken) invalidateIdentity();
  const next = { mode, baseUrl };
  for (const key of ['useMemory','useContext','useCache','useEvolution']) next[key] = typeof values[key] === 'boolean' ? values[key] : before[key];
  write(STORAGE, next); // Device access code stays in memory, never persisted.
  accessToken = mode === 'connected' ? nextToken : '';
  settingsRevision += 1;
  cancelConnectionCheck();
  if (mode !== before.mode || baseUrl !== before.baseUrl) setActiveSession('');
  return getPreferences();
}

const sessionKey = () => { const p = getPreferences(); return `docthinker.mobile.session.${p.mode}.${p.baseUrl}`; };
export function getActiveSession() { return getPreferences().connectionKind === 'invite' ? inviteActiveSession : read(sessionKey()) || ''; }
export function setActiveSession(id) { if (getPreferences().connectionKind === 'invite') inviteActiveSession = String(id || ''); else write(sessionKey(), String(id || '')); }
export function connection() {
  const prefs = getPreferences();
  if (prefs.connectionKind === 'invite' && (prefs.invitePending || prefs.baseUrl !== inviteOrigin())) throw new Error('邀请账号尚未登录或登录已过期，请前往设置中的邀请账号重新登录');
  const base = normalizeBaseUrl(prefs.baseUrl);
  if (prefs.mode !== 'connected' || !base || !prefs.accessToken) throw new Error('请在设置中填写私人服务地址和访问码；也可以使用演示模式');
  return { url: base + '/api/v1/mobile', header: { Authorization: 'Bearer ' + prefs.accessToken } };
}
const query = values => Object.entries(values).map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`).join('&');
function responseError(status) {
  const invited = getPreferences().connectionKind === 'invite';
  const labels = { 400:'输入不符合要求', 401:invited ? '登录已过期或账号不可用，请前往账号页重新登录' : '访问码无效，请重新填写',
    403:'此功能尚未开放或没有访问权限', 404:'移动服务未启用或会话不存在', 408:'请求超时，请先查看历史记录再决定是否重试',
    409:'请求编号与原内容不一致，请先查看历史或原请求状态',
    413:'输入或文件过大，请缩短输入或选择更小的文件', 415:'文件格式不支持', 422:'输入不符合要求',
    429:'服务忙碌或使用额度已达上限，请稍后再试；没有自动重发',
    502:'服务暂时无法完成请求，请先查看历史记录；没有自动重发',
    503:'服务尚未配置好', 504:'处理超时，请先检查历史记录，不要连续重发' };
  const code = Number.isInteger(status) && status >= 400 && status <= 599 ? status : 500;
  const error = new Error(labels[code] || '服务暂时无法完成请求，请稍后手动重试');
  // Only our fixed labels may survive a fetch catch. Never render upstream text.
  error.mobileSafe = true;
  error.status = code;
  return error;
}
export function request(path, data, method = 'GET', responseType = 'text', {isCurrent = () => true} = {}) {
  const cancelled = requestCancellation(isCurrent);
  if (cancelled) return Promise.reject(cancelled);
  return afterBackgroundResearchRead(() => {
    // Cancellation while waiting must not dispatch a now-unwanted operation.
    // This is a pre-dispatch guard, not a promise to undo an upstream request.
    const cancelled = requestCancellation(isCurrent);
    return cancelled ? Promise.reject(cancelled) : requestNow(path,data,method,responseType);
  });
}
function requestCancellation(isCurrent) {
  try { if (isCurrent() === true) return null; } catch { /* Failure is not consent. */ }
  const error = new Error('操作已取消，没有发送请求。');
  error.code = 'REQUEST_CANCELLED';
  return error;
}
function afterBackgroundResearchRead(start) {
  const pending = backgroundResearchRead, revision = identityRevision;
  if (!pending || pending.revision !== revision) return start();
  return pending.settled.then(() => {
    if (revision !== identityRevision) throw new Error('身份或连接已改变，操作已取消');
    return afterBackgroundResearchRead(start);
  });
}
function requestNow(path, data, method = 'GET', responseType = 'text') {
  const config = connection();
  if (getPreferences().connectionKind === 'invite') return inviteRequest(config,path,{method,data,responseType});
  return guardedRequest((resolve, reject) => uni.request({
    url: config.url + path, method, data, header: { ...config.header, 'Content-Type':'application/json' }, responseType, timeout:120000,
    success: res => {
      const status = Number(res.statusCode);
      if (!Number.isInteger(status) || status < 200 || status >= 300) return reject(responseError(res.statusCode));
      if (path === '/query' && method === 'POST' && data?.request_id !== undefined) {
        const valid = status === 200 ? ['completed','failed'].includes(res.data?.request_status)
          : status === 202 && ['running','unknown'].includes(res.data?.request_status);
        if (!valid) return reject(new Error('请求回执不符合协议，请查询状态或查看历史；没有自动重发。'));
      }
      if (path.startsWith('/query/status?') && status !== 200) return reject(new Error('请求状态不符合协议，请查看历史；没有自动重发。'));
      resolve(res.data);
    },
    fail: () => reject(new Error('连接中断。请检查网络或历史记录，系统不会自动重复提交。')),
  }));
}

// Research is a separately probed capability, not a blanket invitation bypass.
// Server-side feature/owner checks remain authoritative for every operation.
export function researchConnectionKey() {
  const p = getPreferences();
  return `${identityRevision}:${settingsRevision}:${p.mode}:${p.connectionKind || 'private'}:${p.baseUrl}:${p.inviteUser?.id || ''}:${p.invitePending === true}`;
}
export function researchRequest(path, data, method = 'GET') {
  if (getPreferences().mode !== 'connected' || !/^\/research\/[a-z][a-z/?=&%0-9_.-]*$/i.test(path)) {
    return Promise.reject(new Error('请选择已连接的研究服务'));
  }
  return request(path, data, method);
}
export function researchStatusRequest(path) {
  if (getPreferences().mode !== 'connected' || !/^\/research\/documents\?session_id=%23\d{5}$/.test(path)) {
    return Promise.reject(new Error('后台状态查询只支持当前研究会话的资料列表'));
  }
  const pending = backgroundResearchRead;
  if (pending?.revision === identityRevision) {
    if (pending.path === path) return pending.result;
    return Promise.reject(new Error('上一次资料状态仍在读取，请稍后手动刷新'));
  }
  const record = {revision:identityRevision,path,result:requestNow(path),settled:null};
  backgroundResearchRead = record;
  const finish = () => { if (backgroundResearchRead === record) backgroundResearchRead = null; };
  // Both outcomes release the barrier; neither outcome resends a request.
  record.settled = record.result.then(finish,finish);
  return record.result;
}
export function researchUpload(filePath, formData, file, {isCurrent = () => true} = {}) {
  return afterBackgroundResearchRead(() => { assertUploadCurrent(isCurrent); return researchUploadNow(filePath,formData,file); });
}
function assertUploadCurrent(isCurrent) {
  let active = false;
  try { active = isCurrent() === true; } catch { /* Failure is not consent. */ }
  if (!active) throw new Error('页面、身份或服务已改变，没有继续上传');
}
function researchUploadNow(filePath, formData, file) {
  const config = connection();
  if (getPreferences().connectionKind === 'invite') {
    if (typeof FormData === 'undefined' || !file) return Promise.reject(new Error('请重新选择需要上传的文件'));
    const form = new FormData();
    for (const [key, value] of Object.entries(formData || {})) form.append(key, String(value));
    form.append('file', file, file.name || 'document.txt');
    return inviteRequest(config, '/research/upload', {method:'POST', form});
  }
  return guardedRequest((resolve, reject) => uni.uploadFile({
    url:config.url + '/research/upload', filePath, ...(file ? {file} : {}), name:'file', formData,
    header:config.header, timeout:120000,
    success:res => {
      if (Number(res.statusCode) < 200 || Number(res.statusCode) >= 300) return reject(responseError(res.statusCode));
      try { resolve(typeof res.data === 'string' ? JSON.parse(res.data) : res.data); }
      catch { reject(new Error('上传结果无法读取，请查看资料列表')); }
    },
    fail:() => reject(new Error('上传未确认，请刷新资料列表；没有自动重传')),
  }));
}

function guardedRequest(start) {
  const revision = identityRevision;
  return new Promise((resolve,reject) => {
    let task, done = false;
    const finish = (callback,value) => { if (done) return; done = true; pendingRequests.delete(cancel); callback(value); };
    const cancel = () => { finish(reject,new Error('身份或连接已改变，操作已取消')); try { task?.abort?.(); } catch { /* no retry */ } };
    pendingRequests.add(cancel);
    try { task = start(value => revision === identityRevision ? finish(resolve,value) : cancel(), error => finish(reject,error)); }
    catch { finish(reject,new Error('连接未完成，请稍后手动重试')); }
  });
}
function inviteRequest(config,path,{method = 'GET',data,responseType = 'text',form} = {}) {
  if (!/^\/[a-z]/.test(path) || path.includes('://') || path.includes('\\')) return Promise.reject(new Error('请求路径无效'));
  return guardedRequest((resolve,reject) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(),120000);
    const header = {...config.header};
    if (!form) header['Content-Type'] = 'application/json';
    fetch(config.url+path,{method,headers:header,body:form || (data === undefined ? undefined : JSON.stringify(data)),
      credentials:'omit',redirect:'error',cache:'no-store',signal:controller.signal})
      .then(async response => {
        if (!response.ok) throw responseError(response.status);
        if (responseType === 'arraybuffer') return response.arrayBuffer();
        const result = await response.json();
        if (path === '/query' && method === 'POST') {
          const valid = response.status === 200 ? ['completed','failed'].includes(result?.request_status)
            : response.status === 202 && ['running','unknown'].includes(result?.request_status);
          if (!valid) throw new Error('Query receipt is incompatible');
        }
        if (path.startsWith('/query/status?') && response.status !== 200) throw new Error('Query status is incompatible');
        return result;
      }).then(resolve,error => reject(error?.mobileSafe === true ? error : new Error('请求未完成。请检查登录状态或历史记录；不会自动重复提交。')))
      .finally(() => clearTimeout(timeout));
    return {abort:() => { clearTimeout(timeout); controller.abort(); }};
  });
}

export function cancelConnectionCheck() {
  connectionCheck?.finish({cancelled:true}, true);
}

// A deliberately narrow probe: no session read, model request or voice check.
// Only fixed result codes cross into the UI; untrusted response data never does.
export function checkConnection({isCurrent = () => true} = {}) {
  if (getPreferences().connectionKind === 'invite') return Promise.resolve({connected:false,reason:'invitation-only'});
  if (connectionCheck) return Promise.resolve({connected:false,reason:'busy'});
  let destination;
  try { destination = connection(); }
  catch { return Promise.resolve({connected:false,reason:'configuration'}); }
  const revision = settingsRevision;
  const active = () => {
    try {
      if (!isCurrent() || revision !== settingsRevision) return false;
      const current = connection();
      return current.url === destination.url && current.header.Authorization === destination.header.Authorization;
    } catch { return false; }
  };
  if (!active()) return Promise.resolve({cancelled:true});
  return new Promise(resolve => {
    const operation = {done:false,task:null,timer:null};
    operation.finish = (result, abort = false) => {
      if (operation.done) return;
      operation.done = true;
      clearTimeout(operation.timer);
      if (connectionCheck === operation) connectionCheck = null;
      if (abort) { try { operation.task?.abort?.(); } catch { /* no retry */ } }
      resolve(active() ? result : {cancelled:true});
    };
    connectionCheck = operation;
    operation.timer = setTimeout(() => operation.finish({connected:false,reason:'timeout'}, true), 8000);
    try {
      operation.task = uni.request({
        url:destination.url + '/capabilities', method:'GET', header:destination.header,
        timeout:8000, dataType:'json',
        success: response => {
          const status = Number(response?.statusCode);
          if (status < 200 || status >= 300 || !Number.isFinite(status)) {
            const reason = status === 401 || status === 403 ? 'unauthorized'
              : status === 404 ? 'unavailable' : status === 503 ? 'not-configured' : 'http';
            operation.finish({connected:false,reason}); return;
          }
          const data = response.data;
          const compatible = data && typeof data === 'object' && !Array.isArray(data)
            && data.audience === 'private-development' && data.customer_auth === false
            && data.realtime_voice === false && data.voice === '/mobile/voice/capabilities'
            && Number.isInteger(data.max_upload_bytes) && data.max_upload_bytes > 0
            && data.max_upload_bytes <= 20 * 1024 * 1024;
          operation.finish(compatible ? {connected:true} : {connected:false,reason:'contract'});
        },
        fail: error => operation.finish({connected:false,reason:/timeout/i.test(String(error?.errMsg || '')) ? 'timeout' : 'network'}),
      });
    } catch { operation.finish({connected:false,reason:'network'}); }
  });
}
export function uploadFile(path, filePath, formData = {}, file, {isCurrent = () => true} = {}) {
  requireExtendedFeatures();
  return afterBackgroundResearchRead(() => { assertUploadCurrent(isCurrent); return uploadFileNow(path,filePath,formData,file); });
}
function uploadFileNow(path, filePath, formData, file) {
  requireExtendedFeatures();
  const config = connection();
  return guardedRequest((resolve, reject) => uni.uploadFile({
    url:config.url + path, filePath, ...(file ? {file} : {}), name:'file', formData, header:config.header, timeout:120000,
    success: res => {
      if (Number(res.statusCode) < 200 || Number(res.statusCode) >= 300) return reject(responseError(res.statusCode));
      try { resolve(typeof res.data === 'string' ? JSON.parse(res.data) : res.data); } catch { reject(new Error('服务返回的内容无法读取')); }
    },
    fail: () => reject(new Error('上传未完成，请检查网络和资料列表')),
  }));
}

const demoSessions = [{id:'#demo1',title:'能源安全 · 示例资料'}];
const demoHistory = new Map();
const demoFiles = new Map();
const demoFeedback = new Map();
const demoNodes = ['气候目标','政策工具','可再生能源投资','国内供给增加','进口依赖下降','能源安全'];
const demoMemories = [
  {id:'demo-memory-1',kind:'project_state',summary:'【示例记忆】这个研究会话关注能源转型如何影响能源安全，需要区分原文证据与假设。',question:'【示例原话】比较论文中的观点时，请同时指出证据和局限。',source_type:'assistant_summary',question_source:'user_message',created_at:null,truncated_fields:[]},
  {id:'demo-memory-2',kind:'user_preference',summary:'【示例偏好，待核对】讨论研究方案时，先梳理目标和约束，再提出下一步。此条并不是系统从你身上学到的偏好。',question:'【示例原话】先问清楚我的研究目标和目前遇到的困难。',source_type:'assistant_summary',question_source:'user_message',created_at:null,truncated_fields:[]},
];
const clone = value => JSON.parse(JSON.stringify(value));
function demoSid(sid) { if (!demoSessions.some(s => s.id === sid)) throw new Error('请选择当前模式中的会话'); }
const isDemo = () => getPreferences().mode === 'demo';
function requireExtendedFeatures() { if (getPreferences().connectionKind === 'invite') throw new Error('邀请内测目前仅开放对话和历史；资料、图谱、长期记忆与语音尚未开放'); }
export function isFeedbackAnswerId(value, demo = false) {
  return typeof value === 'string' && (demo ? /^demo-answer-[0-9]{1,6}$/.test(value)
    : /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value));
}
function validFeedback(value, demo) {
  return value && typeof value === 'object' && isFeedbackAnswerId(value.answer_id,demo)
    && ['helpful','inaccurate'].includes(value.rating) && typeof value.note === 'string'
    && Array.from(value.note).length <= 500;
}

export const api = {
  checkConnection,
  async sessions() { return isDemo() ? clone(demoSessions) : (await request('/sessions')).sessions; },
  async createSession(title = '新对话') {
    if (!isDemo()) return (await request('/sessions',{title},'POST')).session;
    const item = {id:`#demo${demoSessions.length+1}`,title}; demoSessions.unshift(item); return clone(item);
  },
  async history(sid) {
    if (!isDemo()) {
      const result = await request('/history?' + query({session_id:sid}));
      const items = Array.isArray(result?.query_requests) ? result.query_requests.slice(0,5) : [];
      queryTracking.set(sid,items.filter(item => QUERY_REQUEST_ID.test(item?.request_id)
        && ['running','unknown'].includes(item?.request_status)).map(item=>({request_id:item.request_id,request_status:item.request_status})));
      return result.history;
    }
    demoSid(sid);
    const feedback = demoFeedback.get(sid) || new Map();
    return clone((demoHistory.get(sid) || []).map(item => item.role === 'assistant' && feedback.has(item.id)
      ? {...item,feedback:feedback.get(item.id)} : item));
  },
  async files(sid) {
    requireExtendedFeatures();
    if (!isDemo()) return (await request('/files?' + query({session_id:sid}))).files;
    demoSid(sid); return clone(demoFiles.get(sid) || (sid === '#demo1' ? [{filename:'能源转型研究（模拟）.pdf',status:'completed',file_size:245760}] : []));
  },
  pendingQueries(sid) { return !isDemo() ? (queryTracking.get(sid) || []).map(item=>({...item})) : []; },
  async queryStatus(sid, requestId) {
    if (isDemo() || typeof sid !== 'string' || !/^#\d{5}$/.test(sid) || typeof requestId !== 'string' || !QUERY_REQUEST_ID.test(requestId)) throw new Error('请选择当前会话中的有效请求');
    const result = await request('/query/status?' + query({session_id:sid,request_id:requestId}));
    return trackedQueryResponse(result,requestId);
  },
  async ask(sid, question, {requestId, useConversationContext, useProjectMemory, useLLMCache, useSelfEvolution, reasoningMode, memoryAsOf, knowledgeSkillMode} = {}) {
    if (typeof question !== 'string' || !question.trim()) throw new Error('请先输入问题');
    if (!isDemo()) {
      connection(); // Validate current identity before protocol/input diagnostics; no network.
      const p = getPreferences(), invited = p.connectionKind === 'invite', tracked = invited || requestId !== undefined;
      if (tracked && questionByteLength(question) > INVITE_MAX_QUESTION_BYTES) throw new Error('本次问题最多 8192 UTF-8 字节，请缩短输入。完整草稿仍保留，没有发送问题。');
      if (tracked && useConversationContext !== undefined && typeof useConversationContext !== 'boolean') throw new Error('回答上下文设置不符合协议');
      const id = tracked ? requestId === undefined ? createQueryRequestId() : requestId : null;
      if (tracked && (typeof id !== 'string' || !QUERY_REQUEST_ID.test(id) || typeof sid !== 'string' || !/^#\d{5}$/.test(sid))) throw new Error('请求编号或会话不符合协议，没有发送问题');
      if (useProjectMemory !== undefined && typeof useProjectMemory !== 'boolean') throw new Error('确认记忆设置不符合协议');
      if (tracked && [useLLMCache,useSelfEvolution].some(value=>value !== undefined && typeof value !== 'boolean')) throw new Error('回答控制设置不符合协议');
      if (reasoningMode !== undefined && !['auto','evidence','analogy','deduction','induction'].includes(reasoningMode)) throw new Error('推理方式不符合协议');
      if (memoryAsOf !== undefined && memoryAsOf !== null && !(typeof memoryAsOf === 'number' && Number.isFinite(memoryAsOf) && memoryAsOf >= 0 && memoryAsOf <= 253402300799)) throw new Error('记忆时间不符合协议');
      if (knowledgeSkillMode !== undefined && !['auto','off'].includes(knowledgeSkillMode)) throw new Error('知识辅助设置不符合协议，没有发送问题');
      const result = await request('/query',{session_id:sid,question,use_memory:useProjectMemory === undefined ? p.useMemory : useProjectMemory,
        use_conversation_context:tracked && useConversationContext !== undefined ? useConversationContext : p.useContext,
        use_llm_cache:tracked && useLLMCache !== undefined ? useLLMCache : p.useCache,
        use_self_evolution:tracked && useSelfEvolution !== undefined ? useSelfEvolution : p.useEvolution,...(tracked ? {request_id:id} : {}),
        ...(reasoningMode !== undefined ? {reasoning_mode:reasoningMode} : {}),...(memoryAsOf !== undefined ? {memory_as_of:memoryAsOf} : {}),
        ...(knowledgeSkillMode !== undefined ? {knowledge_skill_mode:knowledgeSkillMode} : {})},'POST');
      return tracked ? trackedQueryResponse(result,id) : result?.request_status === 'failed'
        ? {request_status:'failed',request_error:knownQueryFailure(result.request_error)} : result;
    }
    demoSid(sid);
    const answer = /论文|研究|资料|证据/.test(question)
      ? '这是演示回答，没有调用模型，也没有读取真实文档。\n\n研究讨论可以先确定：你要验证的问题、现有证据、不同论文的分歧，以及能做的最小实验。\n\n连接私人服务后才能分析你上传的资料。这段演示没有形成真实研究结论或启动后台任务。'
      : '这是演示回答，没有调用真实模型。\n\n我们可以先从三件事聊起：你想完成什么？目前做到哪一步？最困扰你的是什么？\n\n这只是交互示例，并没有自动学到你的偏好、保存新记忆或创建长期任务。';
    const records = demoHistory.get(sid) || [], answerId = `demo-answer-${records.length + 2}`;
    records.push({role:'user',content:question},{id:answerId,role:'assistant',content:answer}); demoHistory.set(sid, records);
    return {answer,answer_id:answerId,sources:[]};
  },
  async feedback(sid) {
    if (isDemo()) { demoSid(sid); return {items:clone([...(demoFeedback.get(sid) || new Map()).values()])}; }
    const result = await request('/feedback?' + query({session_id:sid}));
    if (!Array.isArray(result?.items) || result.items.length > 500 || !result.items.every(item => validFeedback(item,false))) throw new Error('反馈记录不符合协议');
    return {items:result.items.map(item => ({answer_id:item.answer_id,rating:item.rating,note:item.note}))};
  },
  async submitFeedback(sid, answerId, {rating,note = ''} = {}) {
    const value = {answer_id:answerId,rating,note};
    if (!validFeedback(value,isDemo())) throw new Error('请选择反馈类型，说明不超过 500 字符');
    if (!isDemo()) return request('/feedback',{session_id:sid,...value},'POST');
    demoSid(sid);
    if (!(demoHistory.get(sid) || []).some(item => item.role === 'assistant' && item.id === answerId)) throw new Error('请选择当前会话的已保存回答');
    const records = demoFeedback.get(sid) || new Map(); records.set(answerId,value); demoFeedback.set(sid,records);
    return {feedback:clone(value),demo:true};
  },
  async memory(sid, {limit = 20, offset = 0} = {}) {
    requireExtendedFeatures();
    if (!Number.isInteger(limit) || limit < 1 || limit > 20 || !Number.isInteger(offset) || offset < 0 || offset > 1000) throw new Error('记忆分页参数无效');
    if (!isDemo()) return request('/memory?' + query({session_id:sid,limit,offset}));
    demoSid(sid);
    const items = sid === '#demo1' ? demoMemories : [];
    const page = items.slice(offset, offset + limit);
    const hasMore = offset + page.length < items.length;
    return {items:clone(page),scope:'session',limit,offset,has_more:hasMore,next_offset:hasMore ? offset + page.length : null};
  },
  async upload(sid, {kind = 'document', isCurrent = () => true} = {}) {
    requireExtendedFeatures();
    if (isDemo()) { demoSid(sid); const items = await this.files(sid); items.push({filename:`示例上传-${items.length+1}.txt`,status:'completed',file_size:2048}); demoFiles.set(sid,items); return {status:'demo',message:'仅添加演示记录，没有读取或上传真实文件'}; }
    const destination = connection();
    const active = () => {
      if (!isCurrent() || getPreferences().mode !== 'connected') return false;
      try { const current = connection(); return current.url === destination.url && current.header.Authorization === destination.header.Authorization; }
      catch { return false; }
    };
    const checkActive = () => { if (!active()) throw new Error('操作已取消或服务配置已变化，没有继续上传'); };
    checkActive();
    await ensureWeChatPrivacy({mode:getPreferences().mode,action:'选择和上传资料',isCurrent:active});
    checkActive();
    const selected = await new Promise((resolve,reject) => {
      const options = {count:1,success:resolve,fail:() => reject(new Error('已取消选择文件'))};
      // #ifdef MP-WEIXIN
      if (kind === 'image') uni.chooseImage({...options,sourceType:['album'],sizeType:['original']});
      else wx.chooseMessageFile({...options,type:'file',extension:['pdf','txt','md','docx','png','jpg','jpeg']});
      // #endif
      // #ifdef H5
      uni.chooseFile({...options,type:'all',extension:['.pdf','.txt','.md','.docx','.png','.jpg','.jpeg']});
      // #endif
      // #ifdef APP-PLUS
      uni.chooseImage({...options,sourceType:['album']});
      // #endif
    });
    checkActive();
    const file = selected.tempFiles?.[0];
    if (file?.size > 20 * 1024 * 1024) throw new Error('请选择小于 20 MB 的文件');
    const path = file?.path || selected.tempFilePaths?.[0];
    if (!path) throw new Error('无法读取所选文件');
    const name = String(file?.name || '').split(/[\\/]/).pop();
    if (name && !/\.(pdf|txt|md|docx|png|jpe?g)$/i.test(name)) throw new Error('请选择 PDF、TXT、MD、DOCX 或 PNG/JPG 图片');
    const confirmation = await new Promise((resolve, reject) => uni.showModal({
      title:'确认上传资料', content:`${name || '所选资料'}\n将发送到 ${getPreferences().baseUrl}，供当前会话处理。`,
      confirmText:'上传', cancelText:'取消', success:resolve, fail:()=>reject(new Error('无法确认上传，没有发送资料')),
    }));
    if (!confirmation.confirm) return {cancelled:true};
    checkActive();
    return uploadFile('/upload',path,{session_id:sid,...(name ? {original_filename:name} : {})},file?.file,{isCurrent:active});
  },
  async graph(sid) {
    requireExtendedFeatures();
    if (!isDemo()) return request('/graph?' + query({session_id:sid}));
    demoSid(sid); if (sid !== '#demo1') return {nodes:[],edges:[]};
    return {nodes:demoNodes.map((label,i)=>({id:label,label,type:'概念',description:'仅用于展示交互的模拟知识节点。',degree:i===0||i===5?1:2})),edges:demoNodes.slice(1).map((label,i)=>({source:demoNodes[i],target:label,description:'示例关联，不代表已证实的因果关系'}))};
  },
  async chunks(sid, entity_id) { requireExtendedFeatures(); if (!isDemo()) return (await request('/chunks?' + query({session_id:sid,entity_id}))).chunks; demoSid(sid); return [{content:'模拟来源：可再生能源投资有助于增加国内供给。真实结论需要回到文档证据核验。'}]; },
};
