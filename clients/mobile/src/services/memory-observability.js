import {getPreferences, request, researchConnectionKey} from './api.js';
import {research} from './research.js';

const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value, max = 240) => typeof value === 'string' && value.length <= max * 2 && Array.from(value).length <= max
  && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u202a-\u202e\u2066-\u2069]/u.test(value)
  && Array.from(value).every(character=>{const point=character.codePointAt(0);return point<0xd800 || point>0xdfff;});
const integer = value => Number.isSafeInteger(value) && value >= 0;
const malformed = () => new Error('记忆观测返回格式不符合约定，请手动刷新；未展示旧会话的数据。');
const session = value => { if (typeof value !== 'string' || !/^#\d{5}$/.test(value)) throw new Error('请先选择当前服务中的会话'); return value; };
const count = value => { if (value !== null && !integer(value)) throw malformed(); return value; };
const label = (value, max = 240) => { if (!text(value,max)) throw malformed(); return value; };
const id = value => { if (!text(value,180) || !value) throw malformed(); return value; };
const timestamp = value => { if (value !== null && !(typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 253402300799)) throw malformed(); return value; };
const eventTime = value => { if (value !== null && !(typeof value === 'number' && Number.isFinite(value) && value >= -62135596800 && value <= 253402300799)) throw malformed(); return value; };
function statuses(value) {
  if (value === null) return null;
  if (!object(value) || Object.keys(value).length > 32) throw malformed();
  return Object.fromEntries(Object.entries(value).map(([key,value]) => [label(key,64),count(value)]));
}
function rows(value, sid, read) {
  if (!Array.isArray(value) || value.length > 100) throw malformed();
  const out = value.map(item => { if (!object(item) || (item.session_id !== undefined && item.session_id !== sid)) throw malformed(); return read(item); });
  if (new Set(out.map(item=>item.id)).size !== out.length) throw malformed();
  return out;
}

// A bounded read projection. No summary is promoted to fact, and a missing
// counter remains null instead of becoming a reassuring but invented zero.
export function normalizeMemoryObservation(value, sid, profile = null) {
  session(sid);
  if (!object(value) || value.schema_version !== 1 || value.session_id !== sid || value.read_only !== true
    || !['bounded_research','legacy_session_memory'].includes(value.profile) || (profile && value.profile !== profile)) throw malformed();
  const legacy = value.profile === 'legacy_session_memory';
  const countsScope = legacy ? 'stored_active_session_summaries' : 'stored_session_rows';
  const available = value.availability, raw = value.counts, window = value.window;
  if (!object(available) || !['documents','confirmed_memories','events','event_relations','graph'].every(key=>typeof available[key] === 'boolean')
    || !object(raw) || !object(raw.documents) || !object(raw.chunks) || !object(raw.memories) || !object(raw.events) || !object(raw.relations)
    || !object(window) || !Number.isInteger(window.limit) || window.limit < 1 || window.limit > 100
    || window.counts_scope !== countsScope || window.validity_scope !== 'returned_window'
    || !object(window.returned) || !object(window.limited)) throw malformed();
  const categories = ['documents','memories','events','relations','sources'];
  if (!categories.every(key=>integer(window.returned[key]) && typeof window.limited[key] === 'boolean')) throw malformed();
  const counts = {
    documents:{total:count(raw.documents.total),by_status:statuses(raw.documents.by_status)},
    chunks:{total:count(raw.chunks.total),available:count(raw.chunks.available)},
    memories:{total:count(raw.memories.total),by_status:statuses(raw.memories.by_status),versions:count(raw.memories.versions),source_links:count(raw.memories.source_links)},
    events:{total:count(raw.events.total)}, relations:{total:count(raw.relations.total)},
  };
  const documents = rows(value.documents,sid,item=>({id:id(item.id),filename:label(item.filename ?? item.title ?? '',240),status:label(item.status ?? '',64)}));
  const memories = rows(value.memories,sid,item=>{
    if (typeof item.redacted !== 'boolean' || (!(legacy && item.revision === null) && (!Number.isInteger(item.revision) || item.revision < 1)) || typeof item.archived !== 'boolean'
      || (!legacy && item.confirmed_by_user !== true)
      || !Array.isArray(item.source_ids) || item.source_ids.length > 20) throw malformed();
    return {id:id(item.id),kind:label(item.kind,64),text:item.redacted ? '' : label(item.text,2000),status:label(item.status,64),revision:item.revision,
      source_count:count(item.source_count),source_ids:item.redacted ? [] : item.source_ids.map(id),redacted:item.redacted,archived:item.archived,
      confirmed_by_user:!legacy && item.confirmed_by_user === true,applicable_now:typeof item.applicable_now === 'boolean' ? item.applicable_now : null,
      valid_from:timestamp(item.valid_from ?? null),valid_until:timestamp(item.valid_until ?? null)};
  });
  const events = rows(value.events,sid,item=>({id:id(item.id),title:item.redacted === true ? '' : label(item.action ?? item.title ?? item.summary ?? item.label ?? '',1200),
    status:label(item.status ?? '',64),revision:item.revision === undefined ? null : count(item.revision),
    occurred_at:eventTime(item.occurred_from ?? item.occurred_at ?? item.event_time ?? null),redacted:item.redacted === true}));
  const relations = rows(value.relations,sid,item=>({id:id(item.id),source:item.redacted === true ? null : id(item.source ?? item.source_id ?? item.from_event_id),
    target:item.redacted === true ? null : id(item.target ?? item.target_id ?? item.to_event_id),relation:item.redacted === true ? '' : label(item.relation ?? item.relation_type ?? item.type ?? '',80),
    status:label(item.status ?? '',64),redacted:item.redacted === true,causality_verified:false}));
  const graph = value.graph;
  if (!object(graph) || graph.kind !== (legacy ? 'legacy_memory_projection' : 'provenance_projection') || graph.causality_verified !== false || typeof graph.window_limited !== 'boolean'
    || !Array.isArray(graph.nodes) || graph.nodes.length > 2400 || !Array.isArray(graph.edges) || graph.edges.length > 4000) throw malformed();
  const nodes = graph.nodes.map(node=>{
    if (!object(node) || !['document','source','memory','event'].includes(node.type) || typeof node.redacted !== 'boolean') throw malformed();
    if (node.source_ids !== undefined && (!Array.isArray(node.source_ids) || node.source_ids.length > 20)) throw malformed();
    return {id:id(node.id),record_id:id(node.record_id),type:node.type,label:node.redacted ? '内容已撤销' : label(node.label,1200),
      description:node.redacted ? '' : label(node.description ?? '',500),source_ids:node.redacted ? [] : (node.source_ids || []).map(id),
      applicable_now:typeof node.applicable_now === 'boolean' ? node.applicable_now : null,
      label_truncated:node.label_truncated === true,description_truncated:node.description_truncated === true,sources_limited:node.sources_limited === true,
      status:label(node.status,64),revision:node.revision === undefined ? null : count(node.revision),redacted:node.redacted};
  });
  const nodeIds = new Set(nodes.map(node=>node.id));
  if (nodeIds.size !== nodes.length) throw malformed();
  const edges = graph.edges.map(edge=>{
    if (!object(edge) || edge.causality_verified !== false || !nodeIds.has(edge.source) || !nodeIds.has(edge.target)) throw malformed();
    if ((edge.source_ids !== undefined && (!Array.isArray(edge.source_ids) || edge.source_ids.length > 20))
      || (edge.conditions !== undefined && (!Array.isArray(edge.conditions) || edge.conditions.length > 16))) throw malformed();
    return {id:id(edge.id),source:edge.source,target:edge.target,type:label(edge.type,80),relation:label(edge.relation ?? '',80),
      description:label(edge.description ?? '',500),source_ids:(edge.source_ids || []).map(id),conditions:(edge.conditions || []).map(value=>label(value,256)),
      sources_limited:edge.sources_limited === true,source_scope:label(edge.source_scope ?? '',80),is_discovered:false,causality_verified:false};
  });
  if (new Set(edges.map(edge=>edge.id)).size !== edges.length) throw malformed();
  if (window.states !== undefined && window.states !== null && !object(window.states)) throw malformed();
  const states = window.states === undefined || window.states === null ? null : Object.fromEntries(Object.entries(window.states).map(([key,value])=>[label(key,64),statuses(value)]));
  if (graph.returned !== undefined && (!object(graph.returned) || graph.returned.nodes !== nodes.length || graph.returned.edges !== edges.length)) throw malformed();
  if (graph.totals !== undefined && (!object(graph.totals) || graph.totals.nodes !== null || graph.totals.edges !== null)) throw malformed();
  return {schema_version:1,profile:value.profile,session_id:sid,read_only:true,sampled_at:timestamp(value.sampled_at),
    availability:{...available},counts,window:{limit:window.limit,returned:{...window.returned},limited:{...window.limited},states,counts_scope:countsScope,validity_scope:'returned_window'},
    documents,memories,events,relations,graph:{kind:graph.kind,nodes,edges,window_limited:graph.window_limited,returned:{nodes:nodes.length,edges:edges.length},totals:{nodes:null,edges:null},causality_verified:false}};
}

export function createMemoryObservationClient(dependencies = {}) {
  const preferences = dependencies.preferences || getPreferences, identity = dependencies.identity || researchConnectionKey;
  const capabilities = dependencies.capabilities || (options=>research.capabilities(options)), send = dependencies.request || request;
  let generation = 0;
  return {
    invalidate() { generation += 1; },
    async load(sid,{mode = 'research',limit = 20,isCurrent = () => true} = {}) {
      session(sid);
      if (!['research','legacy'].includes(mode) || !Number.isInteger(limit) || limit < 1 || limit > 100) throw new Error('记忆观测范围不符合要求');
      const token = ++generation, key = identity();
      const current = () => { try { return token === generation && key === identity() && isCurrent() === true; } catch { return false; } };
      const check = () => { if (!current()) throw new Error('页面、身份或会话已改变，已丢弃记忆观测结果'); };
      check(); const p = preferences();
      if (p.mode !== 'connected' || p.invitePending || (mode === 'legacy' && p.connectionKind === 'invite')) throw new Error('当前连接未开放该记忆观测；未发送查询');
      if (mode === 'research') {
        const value = await capabilities(); check();
        if (value?.enabled !== true) throw new Error('当前服务未开放研究记忆观测；未发送查询');
      }
      check();
      const path = `${mode === 'research' ? '/research/observability' : '/memory/dashboard'}?session_id=${encodeURIComponent(sid)}&limit=${limit}`;
      const value = await send(path,undefined,'GET','text',{isCurrent:current}); check();
      return normalizeMemoryObservation(value,sid,mode === 'research' ? 'bounded_research' : 'legacy_session_memory');
    },
  };
}

// Isolated interaction fixture. Never a fallback for a connected service, and
// never persisted, counted as real growth, or sent to a model.
export function sampleMemoryObservation(sid = '#00001') {
  session(sid);
  const doc='sample-document',source='sample-source',memory='sample-memory',event='sample-event';
  return normalizeMemoryObservation({schema_version:1,profile:'bounded_research',session_id:sid,read_only:true,sampled_at:0,
    availability:{documents:true,confirmed_memories:true,events:true,event_relations:true,graph:true},
    counts:{documents:{total:1,by_status:{ready:1}},chunks:{total:1,available:1},memories:{total:1,by_status:{active:1},versions:2,source_links:1},events:{total:1},relations:{total:0}},
    window:{limit:20,returned:{documents:1,memories:1,events:1,relations:0,sources:1},limited:{documents:false,memories:false,events:false,relations:false,sources:false},counts_scope:'stored_session_rows',validity_scope:'returned_window'},
    documents:[{id:doc,filename:'离线样例 · 项目纪要.txt',status:'ready'}],
    memories:[{id:memory,kind:'preference',text:'样例：希望先看结论，再看证据。',status:'active',revision:2,source_count:1,source_ids:[source],redacted:false,archived:false,confirmed_by_user:true}],
    events:[{id:event,title:'样例：确认交付范围',status:'active',revision:1}],relations:[],
    graph:{kind:'provenance_projection',causality_verified:false,window_limited:false,
      nodes:[{id:`document:${doc}`,record_id:doc,type:'document',label:'样例纪要',status:'ready',redacted:false},{id:`source:${source}`,record_id:source,type:'source',label:'样例原文',status:'available',redacted:false},{id:`memory:${memory}`,record_id:memory,type:'memory',label:'样例偏好',status:'active',revision:2,redacted:false},{id:`event:${event}`,record_id:event,type:'event',label:'样例事件',status:'active',revision:1,redacted:false}],
      edges:[{id:'sample-contains',source:`document:${doc}`,target:`source:${source}`,type:'contains',causality_verified:false},{id:'sample-cites',source:`memory:${memory}`,target:`source:${source}`,type:'cites',causality_verified:false}]}},sid);
}
