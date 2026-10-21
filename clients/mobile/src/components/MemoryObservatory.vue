<template>
  <view class="memory-observatory" :aria-busy="busy" @keydown="activateButton">
    <view class="mo-header"><view><text class="mo-eyebrow">MEMORY OBSERVATORY</text><text class="mo-title">记忆，不只是存下来</text><text class="mo-copy">{{ mode === 'legacy' && !demo ? '查看本会话的自动摘要；未提供的来源、事件和版本保持未知。' : '看清资料、确认内容与事件，怎样留下来、怎样停止使用。' }}</text></view><button role="button" tabindex="0" class="mo-toggle" :aria-expanded="opened" aria-label="展开或收起记忆观测" @tap="opened = !opened">{{ opened ? '−' : '＋' }}</button></view>
    <view v-if="opened" class="mo-body">
      <view class="mo-toolbar"><text class="mo-badge" :class="{'mo-sample':demo}">{{ demo ? '离线交互样例 · 非真实增长' : mode === 'legacy' ? '当前会话摘要 · 只读快照' : '当前会话 · 只读快照' }}</text><button role="button" tabindex="0" class="mo-refresh" :disabled="!canRead || busy" @tap="refresh">{{ busy ? '读取中…' : '刷新观测' }}</button></view>
      <text v-if="busy" role="status" class="mo-message">正在读取当前会话；不会生成记忆或调用模型。</text>
      <text v-else-if="error" role="alert" class="mo-error">{{ error }}</text>
      <text v-else-if="!sessionId" class="mo-message">选择会话后，才能看到它自己的记忆与来源。</text>
      <text v-else-if="!data" class="mo-message">尚未读取可用快照。没有数据，不代表记忆数量为零。</text>
      <template v-if="data">
        <view class="mo-metrics"><view v-for="metric in metrics" :key="metric.key" class="mo-metric"><text class="mo-number">{{ counter(metric.value) }}</text><text class="mo-metric-label">{{ metric.label }}</text></view></view>
        <view role="tablist" aria-label="记忆观测面板" class="mo-tabs"><button v-for="tab in tabs" :key="tab.id" role="tab" tabindex="0" :aria-selected="panel === tab.id" :class="{'mo-tab-active':panel===tab.id}" @tap="panel=tab.id">{{ tab.label }}</button></view>
        <view v-if="panel === 'layers'" class="mo-panel">
          <view class="mo-layer-list"><view v-for="layer in layers" :key="layer.key" class="mo-layer"><view class="mo-layer-mark" :class="`mo-${layer.key}`" aria-hidden="true">{{ layer.mark }}</view><view class="mo-layer-content"><text class="mo-row-title">{{ layer.label }} <text class="mo-inline-count">{{ counter(layer.value) }}</text></text><text class="mo-copy">{{ layer.note }}</text></view></view></view>
          <view class="mo-state-section"><text class="mo-section-title">存储状态 · 不是事实可信度</text><view v-if="memoryStates.length" class="mo-states"><view v-for="item in memoryStates" :key="item.id" class="mo-state" :class="{'mo-state-warning':item.id==='invalidated'}"><text>{{ stateLabel(item.id) }}</text><text>{{ counter(item.value) }}</text></view></view><text v-else class="mo-copy">该服务未提供确认记忆的状态统计，不能据此判断全部有效。</text><text class="mo-copy mo-state-note">{{ counter(data.counts.memories.versions) }} 个保存版本 · {{ counter(data.counts.memories.source_links) }} 条来源引用。版本变多不等于知识变正确。</text></view>
          <text v-if="validityStates.length" class="mo-section-title">本次返回窗口 · 已重新核对</text><view v-if="validityStates.length" class="mo-states"><view v-for="item in validityStates" :key="item.id" class="mo-state"><text>{{ item.label }}</text><text>{{ counter(item.value) }}</text></view></view>
        </view>
        <view v-else-if="panel === 'sources'" class="mo-panel">
          <text class="mo-section-title">可追到哪，展示到哪</text><text class="mo-copy">{{ legacy ? '当前为会话摘要投影，不是用户确认记忆。服务没有提供原文引用时，不补画来源。' : '这是当前窗口的来源投影，不是完整语义图，也不把连接当作因果证明。' }}</text>
          <text v-if="!data.availability.graph" class="mo-message">当前服务未提供来源投影。不能把未知画成空图。</text>
          <text v-else-if="!sourceNodes.length" class="mo-message">本次快照没有可展示的来源节点；不会补画模型推测。</text>
          <view v-for="node in sourceNodes" :key="node.id" class="mo-source-row"><view class="mo-source-content"><text class="mo-node-kind">{{ typeLabel(node.type) }}{{ node.revision !== null ? ` · v${node.revision}` : '' }}</text><text class="mo-row-title">{{ node.label || '未提供标题' }}</text><text class="mo-copy">{{ stateLabel(node.status) }} · {{ connectionCount(node.id) }} 条窗口内连接</text></view><button v-if="!demo && !node.redacted && node.source_ids.length" role="button" tabindex="0" class="mo-refresh" :disabled="!canRead || busy" @tap="openSource(node)">查看原文</button></view>
          <text v-if="data.graph.nodes.length > sourceNodes.length" class="mo-copy">面板展示前 {{ sourceNodes.length }} 个节点，完整返回投影可在图谱页查看。</text>
        </view>
        <view v-else class="mo-panel">
          <text class="mo-section-title">事件与显式关联</text><text class="mo-copy">{{ counter(data.counts.events.total) }} 条事件记录 · {{ counter(data.counts.relations.total) }} 条关联。用户登记或材料引用，不等于因果成立。</text>
          <text v-if="!data.availability.events" class="mo-message">该服务未开放事件观测，未提供数量与内容。</text>
          <text v-else-if="!data.events.length" class="mo-message">本次窗口暂无事件记录。不会把模型猜测自动写成经历。</text>
          <view v-for="event in data.events" :key="event.id" class="mo-event"><view class="mo-event-marker" aria-hidden="true"></view><view class="mo-event-content"><text class="mo-row-title">{{ event.redacted ? '事件内容已撤销' : event.title || '事件未提供描述' }}</text><text class="mo-copy">{{ stateLabel(event.status) }}{{ event.revision !== null ? ` · v${event.revision}` : '' }} · {{ eventDate(event.occurred_at) }}</text></view></view>
          <view v-for="relation in data.relations" :key="relation.id" class="mo-relation"><text v-if="relation.redacted">关联依据已失效 · {{ stateLabel(relation.status) }}</text><template v-else><text>{{ eventName(relation.source) }}</text><text class="mo-relation-label">{{ relationLabel(relation.relation) }} →</text><text>{{ eventName(relation.target) }}</text></template></view>
          <text v-if="!data.availability.event_relations" class="mo-copy">事件关联统计未提供，不表示没有关联。</text>
        </view>
        <view class="mo-footnote"><text v-if="limited" class="mo-window-note">窗口有限：只展示此次返回的记录，不能据此断言全部来源或事件有效。</text><text>{{ demo ? '固定交互样例，不是你的记忆，也不是模型质量或长期增长实测。' : `快照 ${snapshotDate} · 手动刷新，不轮询、不调用模型。` }}</text><text>{{ legacy ? '总数仅含本会话活跃摘要。原文、版本与事件未提供时保持未知。' : '总数为当前会话存储记录；可用性仅逐条核对此次窗口。' }}没有连续历史时，不编造增长曲线。</text></view>
      </template>
    </view>
  </view>
</template>

<script setup>
import {computed,ref,watch,onBeforeUnmount} from 'vue';
import {onHide,onUnload} from '@dcloudio/uni-app';
import {researchConnectionKey} from '../services/api.js';
import {createMemoryObservationClient,sampleMemoryObservation} from '../services/memory-observability.js';

const props=defineProps({sessionId:{type:String,default:''},identity:{type:String,default:''},active:{type:Boolean,default:true},enabled:{type:Boolean,default:false},mode:{type:String,default:'research'},demo:{type:Boolean,default:false},disabled:{type:Boolean,default:false},refreshKey:{type:Number,default:0}});
const emit=defineEmits(['projection','source']),client=createMemoryObservationClient();
const opened=ref(true),busy=ref(false),error=ref(''),data=ref(null),panel=ref('layers');
let generation=0,snapshotIdentity='';
const activeNow=()=>props.active && !props.disabled && Boolean(props.sessionId) && (props.demo || props.enabled) && ['research','legacy'].includes(props.mode) && (!props.identity || props.identity===researchConnectionKey());
const canRead=computed(activeNow);
const legacy=computed(()=>data.value?.profile==='legacy_session_memory');
const tabs=[{id:'layers',label:'层级与状态'},{id:'sources',label:'来源连接'},{id:'events',label:'事件关联'}];
const counter=value=>Number.isSafeInteger(value) && value>=0 ? String(value) : '未提供';
const typeLabel=value=>({document:'资料',source:'原文片段',memory:legacy.value?'会话摘要':'确认记忆',event:'事件'}[value] || '记录');
const stateLabel=value=>({active:'保留中',archived:'已归档',invalidated:'已失效',deleted:'已删除',available:'当前可查',unavailable:'不可用',source_unavailable:'来源不可用',stale_endpoint:'事件版本已更新',ready:'解析完成',queued:'等待解析',processing:'解析中',failed:'解析失败',superseded:'已有新版本',expired:'已过适用期'}[value] || (value ? '状态待核对' : '状态未提供'));
const relationLabel=value=>({before:'早于',supersedes:'更新替代',depends_on:'依赖',contradicts:'存在冲突',related_to:'相关',reported_cause:'登记的原因 · 未验证',hypothesis:'假设 · 待验证'}[value] || '显式关联 · 待核对');
const metrics=computed(()=>data.value ? [{key:'documents',label:'资料记录',value:data.value.counts.documents.total},{key:'memories',label:legacy.value?'会话摘要':'确认记忆',value:data.value.counts.memories.total},{key:'events',label:'事件记录',value:data.value.counts.events.total},{key:'invalidated',label:'失效记忆 · 存储状态',value:data.value.counts.memories.by_status?.invalidated ?? null}] : []);
const layers=computed(()=>data.value ? [{key:'documents',mark:'01',label:'资料',value:data.value.counts.documents.total,note:'上传和解析的原始材料，不是模型结论。'},{key:'sources',mark:'02',label:'原文',value:data.value.counts.chunks.available,note:'当前可用的片段总数；不代表这次全部展开。'},{key:'memories',mark:'03',label:legacy.value?'会话摘要':'确认记忆',value:data.value.counts.memories.total,note:legacy.value?'助手自动整理，不是用户原话，也未经用户确认。':'明确保存的内容，保留版本和适用时间。'},{key:'events',mark:'04',label:'事件',value:data.value.counts.events.total,note:'带时间与来源的记录；关联不等于因果。'}] : []);
const memoryStates=computed(()=>Object.entries(data.value?.counts.memories.by_status || {}).map(([id,value])=>({id,value})));
const validityStates=computed(()=>Object.entries(data.value?.window.states || {}).flatMap(([kind,states])=>Object.entries(states || {}).map(([status,value])=>({id:`${kind}:${status}`,label:`${({memories:'记忆',events:'事件',relations:'关联'}[kind] || '记录')} · ${stateLabel(status)}`,value}))));
const sourceNodes=computed(()=>(data.value?.graph.nodes || []).slice(0,24));
const limited=computed(()=>data.value && (data.value.graph.window_limited || Object.values(data.value.window.limited).some(Boolean)));
const snapshotDate=computed(()=>eventDate(data.value?.sampled_at));
function eventDate(value){if(typeof value!=='number' || !Number.isFinite(value))return '时间未提供';const date=new Date(value*1000);return Number.isNaN(date.getTime())?'时间未提供':date.toLocaleString();}
function connectionCount(id){return data.value?.graph.edges.filter(edge=>edge.source===id || edge.target===id).length || 0;}
function eventName(id){const event=data.value?.events.find(event=>event.id===id);return event ? (event.redacted?'已撤销事件':event.title || '未命名事件') : '窗口外事件';}
function activateButton(event){if(!event || event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey || !['Enter',' ','Spacebar'].includes(event.key))return;const target=event.target?.closest?.('[role="button"],[role="tab"]');if(!target || target.hasAttribute?.('disabled') || target.getAttribute?.('aria-disabled')==='true' || typeof target.click!=='function')return;event.preventDefault?.();target.click();}
function openSource(node){if(snapshotIdentity!==researchConnectionKey()){reset();return;}if(activeNow() && !props.demo && !node.redacted && node.source_ids.length)emit('source',node.source_ids[0]);}
function reset(){generation++;snapshotIdentity='';client.invalidate();busy.value=false;error.value='';data.value=null;panel.value='layers';emit('projection',null);}
async function refresh(){
  if(!activeNow() || busy.value)return;
  const token=++generation,sid=props.sessionId,key=researchConnectionKey();
  const current=()=>token===generation && activeNow() && sid===props.sessionId && key===researchConnectionKey();
  busy.value=true;error.value='';data.value=null;emit('projection',null);
  try {
    const value=props.demo ? sampleMemoryObservation('#00001') : await client.load(sid,{mode:props.mode,isCurrent:current});
    if(!current())return;snapshotIdentity=key;data.value=value;emit('projection',{...value.graph,session_id:sid,sample:props.demo});
  }catch(issue){if(current())error.value=issue?.message || '暂时无法读取记忆观测，请手动刷新。';}
  finally{if(current())busy.value=false;}
}
watch(()=>[props.sessionId,props.identity,props.active,props.enabled,props.mode,props.demo,props.refreshKey],()=>{reset();if(canRead.value)void refresh();},{immediate:true});
watch(()=>props.disabled,value=>{if(value){generation++;client.invalidate();busy.value=false;}else if(!data.value && canRead.value)void refresh();});
onHide(reset);onUnload(reset);onBeforeUnmount(reset);
</script>

<style scoped>
.memory-observatory{margin:12px 0 28px;padding:24px 24px 0;border:1px solid #dce5da;border-radius:24px;background:linear-gradient(155deg,#f2f6ee 0%,#fcfdf9 64%);color:#284a3f;overflow:hidden;min-width:0}.mo-header{display:flex;gap:20px;align-items:flex-start;padding-bottom:20px}.mo-eyebrow{display:block;font-size:9px;letter-spacing:2px;color:#779078;margin-bottom:10px}.mo-title{display:block;font-size:21px;font-weight:600;letter-spacing:-.5px;margin-bottom:9px}.mo-copy{display:block;font-size:11px;line-height:1.85;color:#728174;overflow-wrap:anywhere}.mo-toggle{flex:none;width:36px;height:36px;margin:0;padding:0;background:#e7eee1;border-radius:50%;line-height:34px;font-size:23px;color:#456447}.mo-toggle::after,.mo-refresh::after,.mo-tabs button::after{border:0}.mo-body{padding-bottom:20px}.mo-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding-bottom:18px}.mo-badge{font-size:10px;color:#617b63}.mo-sample{color:#907447}.mo-refresh{flex:none;margin:0;min-height:36px;line-height:34px;padding:0 12px;border:1px solid #d8e2d5;border-radius:9px;background:transparent;color:#4e6e53;font-size:11px}.mo-refresh[disabled]{opacity:.5}.mo-message,.mo-error{display:block;padding:15px 0;font-size:12px;line-height:1.8;overflow-wrap:anywhere}.mo-message{color:#728174}.mo-error{color:#94564e}.mo-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;padding:4px 0 23px}.mo-metric{min-width:0}.mo-number{display:block;font-size:29px;line-height:1.2;letter-spacing:-1px;font-weight:400}.mo-metric-label{display:block;font-size:10px;line-height:1.6;color:#7a8877;margin-top:6px}.mo-tabs{display:flex;gap:22px;border-bottom:1px solid #dce4d5}.mo-tabs button{position:relative;flex:none;background:transparent;padding:10px 0;margin:0;font-size:11px;line-height:22px;color:#83917e;border-radius:0}.mo-tabs .mo-tab-active{color:#315b43;font-weight:600}.mo-tab-active::before{position:absolute;bottom:0;left:0;right:0;height:2px;content:'';background:#658461}.mo-panel{padding-top:20px;min-width:0}.mo-layer-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px 28px}.mo-layer{display:flex;gap:13px;align-items:flex-start;min-width:0}.mo-layer-mark{flex:none;display:flex;align-items:center;justify-content:center;width:32px;height:32px;border:1px solid #cfdcca;border-radius:50%;font-size:9px;color:#7d9676;letter-spacing:1px}.mo-sources{border-radius:9px}.mo-memories{background:#e4eddd}.mo-events{border-style:dashed}.mo-layer-content{min-width:0}.mo-row-title{display:block;font-size:12px;line-height:1.7;overflow-wrap:anywhere;word-break:break-word;color:#36513f}.mo-inline-count{font-size:12px;margin-left:5px;color:#70816a}.mo-section-title{display:block;margin-bottom:10px;font-size:11px;font-weight:600;color:#4d6950}.mo-state-section{margin:24px 0 18px;padding-top:18px;border-top:1px dashed #d9e1d1}.mo-states{display:flex;flex-wrap:wrap;gap:8px}.mo-state{display:flex;gap:16px;padding:6px 10px;border-radius:7px;font-size:10px;background:#e9f0e3;color:#637b59}.mo-state-warning{background:#f2e9de;color:#957144}.mo-state-note{margin-top:10px}.mo-footnote{display:flex;flex-direction:column;gap:5px;margin-top:20px;padding-top:14px;border-top:1px solid #e2e8db;color:#899382;font-size:9px;line-height:1.8;overflow-wrap:anywhere}.mo-window-note{color:#9a7c4f}.mo-source-row{display:flex;gap:15px;align-items:center;padding:13px 0;border-bottom:1px solid #e5eadf}.mo-source-content{flex:1;min-width:0}.mo-node-kind{display:block;color:#819277;font-size:9px;margin-bottom:3px}.mo-event{display:flex;gap:14px;margin-top:16px}.mo-event-marker{width:6px;height:6px;flex:none;border:1px solid #8ba17f;border-radius:50%;margin-top:7px}.mo-event-content{min-width:0}.mo-relation{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:14px;padding:10px 12px;border-left:2px solid #d6e0cc;font-size:10px;line-height:1.8;overflow-wrap:anywhere;word-break:break-word}.mo-relation-label{color:#8c977f;font-size:9px}@media(max-width:540px){.memory-observatory{padding:19px 17px 0;border-radius:19px}.mo-title{font-size:19px}.mo-metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;padding-bottom:18px}.mo-number{font-size:27px}.mo-layer-list{grid-template-columns:1fr;gap:17px}.mo-tabs{gap:18px}.mo-tabs button{font-size:10px}.mo-toolbar{gap:6px}.mo-badge{font-size:9px}.mo-source-row{gap:8px}}@media(prefers-reduced-motion:reduce){.memory-observatory *{animation:none!important;transition:none!important}}
</style>
