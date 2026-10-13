import { getPreferences, researchRequest, researchStatusRequest, researchUpload, researchConnectionKey } from './api.js';
import { ensureWeChatPrivacy } from './wechat-privacy.js';
import { knowledgeSkillCatalog } from './knowledge-skills.js';
import { validResearchExport } from './research-export.js';

// Research is opt-in at the authenticated service boundary, not a client switch.
// Cache only the capability envelope in memory, never documents, memory or tokens.
const disabled = () => ({ enabled:false, uploads_allowed:false, max_upload_bytes:10 * 1024 * 1024, supported_upload_types:['pdf','txt','md'] });
let capabilityCache = null;
let stagedMemory = null;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const session = value => { if (!/^#\d{5}$/.test(value)) throw new Error('请选择当前服务中的研究会话'); return value; };
const identifier = value => { if (typeof value !== 'string' || !uuid.test(value)) throw new Error('资料或来源编号不符合要求'); return value; };
const query = values => Object.entries(values).map(([key,value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`).join('&');
const malformed = () => new Error('研究服务返回的格式不符合要求，请刷新后重试。');
const copyCapabilities = value => ({...value,supported_upload_types:[...value.supported_upload_types]});
const currentKey = () => researchConnectionKey();
const checkKey = key => { if (currentKey() !== key) throw new Error('身份或连接已改变，没有继续操作'); };
const boundedText = (value, max) => typeof value === 'string' && Array.from(value).length <= max
  && Array.from(value).every(character=>{const point=character.codePointAt(0);return point < 0xd800 || point > 0xdfff;});
const safeStatuses = ['queued','processing','ready','failed','deleted'];
function readDocument(value, sid) {
  if (!value || !uuid.test(value.id) || !boundedText(value.filename,240) || !safeStatuses.includes(value.status)) throw malformed();
  if (sid && value.session_id !== undefined && value.session_id !== sid) throw malformed();
  return {id:value.id,filename:value.filename,status:value.status,
    chunk_count:Number.isInteger(value.chunk_count) && value.chunk_count >= 0 ? value.chunk_count : 0,
    size_bytes:Number.isInteger(value.size_bytes ?? value.byte_size) && (value.size_bytes ?? value.byte_size) >= 0 ? (value.size_bytes ?? value.byte_size) : undefined,
    error_code:['no_extractable_text','utf8_required','encrypted_pdf','page_limit','text_limit','chunk_limit','extraction_resource_limit','extraction_failed','invalid_pdf','page_stream_limit','parser_capacity_unavailable'].includes(value.error_code) ? value.error_code : '',
    source_ids:Array.isArray(value.source_ids) ? value.source_ids.filter(item=>typeof item === 'string' && uuid.test(item)).slice(0,20) : [],
    preview_limited:value.preview_limited === true,
    chunks:Array.isArray(value.chunks) ? value.chunks.slice(0,20).map(chunk=>{
      const source = readSource(chunk), characters = Array.from(source.content);
      return {...source,content:characters.slice(0,1200).join(''),excerpt_limited:characters.length > 1200};
    }) : []};
}
function readSource(value) {
  const id = value?.id || value?.chunk_id;
  if (!value || !uuid.test(id) || (value.content !== undefined && !boundedText(value.content,20000))) throw malformed();
  return {id,content:typeof value.content === 'string' ? value.content : '',
    filename:boundedText(value.filename,240) ? value.filename : '',
    page:Number.isInteger(value.page) && value.page > 0 ? value.page : null,
    ordinal:Number.isInteger(value.ordinal) && value.ordinal >= 0 ? value.ordinal : null};
}
const memoryKinds = ['finding','preference','decision'];
const epoch = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 253402300799;
function readMemory(value,{allowRedacted = false,allowHistorical = false} = {}) {
  const sources = value?.source_ids ?? value?.source_chunk_ids;
  const redacted = allowRedacted && value?.redacted === true;
  if (!value || !uuid.test(value.id) || !boundedText(value.text,2000) || (!value.text.trim() && !redacted)
    || !memoryKinds.includes(value.kind) || !Number.isInteger(value.revision) || value.revision < 1
    || value.confirmed_by_user !== true || !['active','invalidated','archived',...(redacted ? ['deleted'] : []),...(allowHistorical ? ['superseded'] : [])].includes(value.status) || !Array.isArray(sources) || sources.length > 20
    || !sources.every(id=>typeof id === 'string' && uuid.test(id))) throw malformed();
  return {id:value.id,text:redacted ? '' : value.text,kind:value.kind,status:value.status,revision:value.revision,confirmed_by_user:true,source_ids:redacted ? [] : [...sources],redacted,
    valid_from:epoch(value.valid_from) ? value.valid_from : null, valid_until:epoch(value.valid_until) ? value.valid_until : null,
    recorded_at:epoch(value.recorded_at ?? value.updated_at) ? (value.recorded_at ?? value.updated_at) : null, archived:value.archived === true};
}
async function requireEnabled() {
  const capabilities = await research.capabilities();
  if (!capabilities.enabled) throw new Error('当前服务尚未开放研究资料和确认记忆；没有提交任何内容');
  return capabilities;
}
async function scopedRequest(path, data, method = 'GET', background = false) {
  await requireEnabled(); const key = currentKey();
  const result = await (background ? researchStatusRequest(path) : researchRequest(path,data,method)); checkKey(key); return result;
}
function modal(options) {
  return new Promise((resolve,reject) => {
    let done = false;
    const finish = (error,value) => { if (done) return; done = true; clearTimeout(timer); error ? reject(error) : resolve(value); };
    const timer = setTimeout(()=>finish(new Error('确认等待超时，没有继续操作')),60000);
    try { uni.showModal({...options,success:value=>finish(null,value),fail:()=>finish(new Error('未能确认操作，没有继续'))}); }
    catch { finish(new Error('当前平台无法确认操作，没有继续')); }
  });
}

export const research = {
  async capabilities({refresh = false} = {}) {
    const preferences = getPreferences();
    if (preferences.mode !== 'connected' || preferences.invitePending) { capabilityCache = null; stagedMemory = null; return disabled(); }
    const key = currentKey();
    if (stagedMemory && stagedMemory.key !== key) stagedMemory = null;
    if (!refresh && capabilityCache?.key === key) return capabilityCache.promise ? copyCapabilities(await capabilityCache.promise) : copyCapabilities(capabilityCache.value);
    const record = {key,promise:null,value:null}; capabilityCache = record;
    record.promise = (async () => {
      try {
        const result = await researchRequest('/research/capabilities'); checkKey(key);
        if (result?.enabled !== true) return disabled();
        if (!Number.isInteger(result.max_upload_bytes) || result.max_upload_bytes < 1 || result.max_upload_bytes > 10 * 1024 * 1024
          || !Array.isArray(result.supported_upload_types) || result.supported_upload_types.length < 1
          || !result.supported_upload_types.every(value=>['pdf','txt','md'].includes(String(value).toLowerCase()))) throw malformed();
        return {enabled:true,max_upload_bytes:result.max_upload_bytes,supported_upload_types:result.supported_upload_types.map(value=>value.toLowerCase()),
          uploads_allowed:result.uploads_allowed === true,memory_write_enabled:result.memory_write_enabled === true, automatic_learning:false,
          ...(result.knowledge_skills === true ? {knowledge_skills:true} : {})};
      } catch (error) { if ([403,404].includes(error?.status)) return disabled(); throw error; }
    })();
    try { record.value = await record.promise; return copyCapabilities(record.value); }
    catch (error) { if (capabilityCache === record) capabilityCache = null; throw error; }
    finally { record.promise = null; }
  },
  async knowledgeSkills() {
    const capability = await requireEnabled();
    if (capability.knowledge_skills !== true) return [];
    return knowledgeSkillCatalog(await scopedRequest('/research/knowledge-skills'));
  },
  async documents(sid,{background = false} = {}) {
    const result = await scopedRequest('/research/documents?' + query({session_id:session(sid)}),undefined,'GET',background === true);
    if (!Array.isArray(result?.documents) || result.documents.length > 20) throw malformed();
    return result.documents.map(value=>readDocument(value,sid));
  },
  async document(sid,id) {
    const result = await scopedRequest('/research/document?' + query({session_id:session(sid),document_id:identifier(id)}));
    const document = readDocument(result?.document,sid); if (document.id !== id) throw malformed(); return document;
  },
  async source(sid,id) {
    let result;
    try { result = await scopedRequest('/research/source?' + query({session_id:session(sid),chunk_id:identifier(id)})); }
    catch (error) { if (error?.status !== 404) throw error; const issue = new Error('该原文来源已移除或不属于当前会话，请刷新并核对关联。'); issue.status = 404; throw issue; }
    const source = readSource(result?.chunk || result?.source); if (source.id !== id) throw malformed(); return source;
  },
  async upload(sid,{isCurrent = () => true} = {}) {
    session(sid); const capabilities = await requireEnabled(), key = currentKey();
    if (!capabilities.uploads_allowed) throw new Error('当前研究服务未开放资料上传，没有选择或发送文件');
    const active = () => isCurrent() && currentKey() === key;
    const check = () => { if (!active()) throw new Error('页面、身份或服务已改变，没有继续上传'); };
    check(); await ensureWeChatPrivacy({mode:'connected',action:'选择研究资料',isCurrent:active}); check();
    const selected = await new Promise((resolve,reject) => {
      const options = {count:1,success:resolve,fail:()=>resolve({cancelled:true})};
      try {
        // #ifdef H5
        uni.chooseFile({...options,type:'all',extension:capabilities.supported_upload_types.map(value=>'.'+value)});
        // #endif
        // #ifdef MP-WEIXIN
        wx.chooseMessageFile({...options,type:'file',extension:capabilities.supported_upload_types});
        // #endif
        // #ifdef APP-PLUS
        reject(new Error('本版 App 尚未接入系统文档选择，请使用浏览器或微信测试版上传 PDF / TXT / MD'));
        // #endif
      } catch { reject(new Error('当前平台无法选择研究资料，请使用浏览器上传')); }
    });
    check(); if (selected.cancelled) return {cancelled:true};
    const file = selected.tempFiles?.[0], path = file?.path || selected.tempFilePaths?.[0];
    const name = String(file?.name || '').split(/[\\/]/).pop(), suffix = name.split('.').pop()?.toLowerCase();
    if (!path || !name || !capabilities.supported_upload_types.includes(suffix)) throw new Error('请选择 PDF / TXT / MD 文件；没有上传');
    if (!boundedText(name,120) || name !== name.trim() || /[\u0000-\u001f\u007f]/.test(name)) throw new Error('文件名需不超过 120 字符，不能包含控制字符或首尾空格；请重命名后选择，没有上传');
    if (!Number.isInteger(file?.size) || file.size <= 0 || file.size > capabilities.max_upload_bytes) throw new Error(`请选择不超过 ${Math.floor(capabilities.max_upload_bytes / 1024 / 1024)} MB 的非空文件；没有上传`);
    const confirmation = await modal({title:'确认上传到当前研究会话',content:`${name}\n将发送到当前服务 ${getPreferences().baseUrl}。仅做本地解析，不会自动调用模型；解析后的片段可能在你提问时发送给模型。`,confirmText:'确认上传',cancelText:'取消'});
    check(); if (!confirmation.confirm) return {cancelled:true};
    // uni H5 returns native File objects with an added .path. Some adapters
    // wrap the native File in .file instead; the invitation FormData path
    // needs the actual bytes in either case, not just its blob URL.
    const nativeFile = file?.file || (typeof File !== 'undefined' && file instanceof File ? file : undefined);
    const result = await researchUpload(path,{session_id:sid,original_filename:name},nativeFile,{isCurrent:active}); check();
    return {document:readDocument(result?.document,sid)};
  },
  async uploadText(sid,{title,text,isCurrent = () => true} = {}) {
    // A pasted excerpt is a real document, not a demo answer or an implicit
    // question. Reuse the same consent, upload queue and source provenance.
    // #ifndef H5
    throw new Error('当前版本仅支持在浏览器中提交粘贴原文；请使用浏览器添加 TXT 资料');
    // #endif
    // #ifdef H5
    session(sid);
    if (!boundedText(text,12000) || !text.trim() || text.includes('\0')) throw new Error('请输入 1–12000 字符的有效原文，不能包含空字符；没有上传');
    if (!boundedText(title,120) || !title || title !== title.trim() || title === '.' || title === '..'
      || /[\\/\u0000-\u001f\u007f]/.test(title)) throw new Error('资料标题不能包含路径、控制字符或首尾空格，且不能超过 120 字符；没有上传');
    const name = /\.txt$/i.test(title) ? title : `${title}.txt`;
    if (!boundedText(name,120)) throw new Error('标题加上 .txt 后不能超过 120 字符，请缩短标题；没有上传');
    if (typeof File === 'undefined' || typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function' || typeof URL.revokeObjectURL !== 'function') {
      throw new Error('当前浏览器不能创建文本资料，请更新浏览器或选择 TXT 文件上传');
    }
    const key = currentKey();
    const active = () => {
      try { return isCurrent() === true && currentKey() === key; } catch { return false; }
    };
    const check = () => { if (!active()) throw new Error('页面、身份或服务已改变，没有继续上传'); };
    check(); const capabilities = await requireEnabled(); check();
    if (!capabilities.uploads_allowed || !capabilities.supported_upload_types.includes('txt')) throw new Error('当前研究服务未开放 TXT 资料上传，没有发送原文');
    let file;
    try { file = new File([text],name,{type:'text/plain;charset=utf-8'}); }
    catch { throw new Error('当前浏览器无法创建文本资料，请选择 TXT 文件上传'); }
    if (!Number.isInteger(file.size) || file.size <= 0 || file.size > capabilities.max_upload_bytes) throw new Error('原文超过当前研究服务的上传字节限制，请缩短后重试；没有上传');
    const confirmation = await modal({title:'确认加入当前研究会话',
      content:`${name}\n${Array.from(text).length} 字符，将发送到当前服务 ${getPreferences().baseUrl}。仅做本地解析，不会自动提问或调用模型；相关原文片段可能在你之后提问时发送给模型。`,
      confirmText:'确认提交',cancelText:'取消'});
    check(); if (!confirmation.confirm) return {cancelled:true};
    let path = '';
    try {
      path = URL.createObjectURL(file); check();
      const result = await researchUpload(path,{session_id:sid,original_filename:name},file,{isCurrent:active}); check();
      return {document:readDocument(result?.document,sid)};
    } finally {
      if (path) { try { URL.revokeObjectURL(path); } catch { /* Temporary browser URL only; never repeat an upload. */ } }
    }
    // #endif
  },
  async deleteDocument(sid,id) {
    const result = await scopedRequest('/research/documents/delete',{session_id:session(sid),document_id:identifier(id)},'POST');
    if (result?.removed !== true && !(result?.document_id === id && result?.status === 'deleted')) throw malformed(); return result;
  },
  async retryDocument(sid,id) {
    if (!(await requireEnabled()).uploads_allowed) throw new Error('当前研究服务未开放资料处理，没有重新解析');
    const result = await scopedRequest('/research/documents/retry',{session_id:session(sid),document_id:identifier(id)},'POST');
    const document = readDocument(result?.document,sid); if (document.id !== id) throw malformed(); return document;
  },
  async memory(sid) {
    const result = await scopedRequest('/research/memory?' + query({session_id:session(sid)}));
    if (!Array.isArray(result?.items) || result.items.length > 200) throw malformed(); return result.items.map(readMemory);
  },
  async memoryPage(sid,{query:search = '',cursor = null,limit = 20,as_of = null,include_archived = false} = {}) {
    if (!boundedText(search,500) || search.includes('\0') || !Number.isInteger(limit) || limit < 1 || limit > 100
      || (cursor !== null && (!boundedText(cursor,2048) || !cursor)) || (as_of !== null && !epoch(as_of))) throw new Error('记忆搜索或时间条件不符合要求');
    const result = await scopedRequest('/research/memory?' + query({session_id:session(sid),limit,
      ...(search.trim() ? {query:search.trim()} : {}),...(cursor ? {cursor} : {}),...(as_of !== null ? {as_of} : {}),...(include_archived ? {include_archived:true} : {})}));
    if (!Array.isArray(result?.items) || result.items.length > limit || !Number.isInteger(result.total) || result.total < result.items.length
      || !(result.next_cursor === null || (boundedText(result.next_cursor,2048) && result.next_cursor))) throw malformed();
    return {items:result.items.map(value=>readMemory(value,{allowRedacted:true})),next_cursor:result.next_cursor,total:result.total};
  },
  async memoryHistory(sid,id,{cursor = null,limit = 20} = {}) {
    if ((cursor !== null && (!boundedText(cursor,2048) || !cursor)) || !Number.isInteger(limit) || limit < 1 || limit > 100) throw new Error('历史页码不符合要求');
    const result = await scopedRequest('/research/memory/history?' + query({session_id:session(sid),memory_id:identifier(id),limit,...(cursor ? {cursor} : {})}));
    if (!Array.isArray(result?.items) || result.items.length > limit || !Number.isInteger(result.total) || result.total < result.items.length
      || !(result.next_cursor === null || (boundedText(result.next_cursor,2048) && result.next_cursor))) throw malformed();
    return {items:result.items.map(value=>readMemory({...value,id:value.id || id},{allowRedacted:true,allowHistorical:true})),next_cursor:result.next_cursor,total:result.total};
  },
  async archiveMemory(sid,id,revision,archived) {
    if (!uuid.test(id) || !Number.isInteger(revision) || revision < 1 || typeof archived !== 'boolean') throw new Error('记忆版本不符合要求，请刷新后重试');
    if (!(await requireEnabled()).memory_write_enabled) throw new Error('当前服务未开放确认记忆写入，没有修改内容');
    const result = await scopedRequest('/research/memory/archive',{session_id:session(sid),memory_id:id,expected_revision:revision,archived},'POST');
    const memory = readMemory(result?.memory); if (memory.id !== id) throw malformed(); return memory;
  },
  async saveMemory(sid,{text,kind = 'finding',source_ids = [],memory_id,expected_revision,valid_from,valid_until} = {}) {
    if (!boundedText(text,2000) || !text.trim() || text.includes('\0') || !memoryKinds.includes(kind) || !Array.isArray(source_ids) || source_ids.length > 20 || !source_ids.every(id=>typeof id === 'string' && uuid.test(id))) throw new Error('请输入 1–2000 字符的有效确认内容，并核对来源编号');
    if (memory_id !== undefined && (!uuid.test(memory_id) || !Number.isInteger(expected_revision) || expected_revision < 1)) throw new Error('编辑版本不符合要求，请刷新后重试');
    if ([valid_from,valid_until].some(value=>value !== undefined && value !== null && !epoch(value))
      || (epoch(valid_from) && epoch(valid_until) && valid_until <= valid_from)) throw new Error('有效时间不符合要求，结束时间应晚于开始时间');
    if (!(await requireEnabled()).memory_write_enabled) throw new Error('当前服务未开放确认记忆写入，没有保存内容');
    const result = await scopedRequest('/research/memory',{session_id:session(sid),text,kind,source_ids,
      ...(memory_id ? {memory_id,expected_revision} : {}),...(valid_from !== undefined ? {valid_from} : {}),...(valid_until !== undefined ? {valid_until} : {})},'POST');
    const memory = readMemory(result?.memory); if (memory_id && memory.id !== memory_id) throw malformed(); return memory;
  },
  async deleteMemory(sid,id,revision) {
    if (!uuid.test(id) || !Number.isInteger(revision) || revision < 1) throw new Error('记忆版本不符合要求，请刷新后重试');
    if (!(await requireEnabled()).memory_write_enabled) throw new Error('当前服务未开放确认记忆写入，没有删除内容');
    const result = await scopedRequest('/research/memory/delete',{session_id:session(sid),memory_id:id,expected_revision:revision},'POST');
    if (result?.deleted !== true) throw malformed(); return result;
  },
  async export(sid) {
    const result = await scopedRequest('/research/export?' + query({session_id:session(sid)}));
    if (!validResearchExport(result?.markdown)) throw malformed(); return result.markdown;
  },
  stageMemoryDraft({session_id,text = '',kind = 'finding',source_ids = [],origin = 'answer_sources'} = {}) {
    session(session_id);
    if (getPreferences().mode !== 'connected' || getPreferences().invitePending || !boundedText(text,2000) || text.includes('\0')
      || !memoryKinds.includes(kind) || !Array.isArray(source_ids) || source_ids.length > 20
      || !source_ids.every(value=>typeof value === 'string' && uuid.test(value)) || origin !== 'answer_sources') throw new Error('确认草稿不符合要求，没有保存记忆');
    if (capabilityCache?.key !== currentKey() || capabilityCache?.value?.enabled !== true || capabilityCache.value.memory_write_enabled !== true) throw new Error('当前服务未开放确认记忆，没有保存草稿');
    stagedMemory = {key:currentKey(),session_id,text,kind,source_ids:[...source_ids],origin};
  },
  consumeMemoryDraft(sid) {
    session(sid); const key = currentKey();
    if (stagedMemory?.key !== key) { stagedMemory = null; return null; }
    if (stagedMemory?.session_id !== sid) return null;
    const {key:discard,...value} = stagedMemory; stagedMemory = null;
    return {...value,source_ids:[...value.source_ids]};
  },
};
