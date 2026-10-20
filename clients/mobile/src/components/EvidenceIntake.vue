<template>
  <view class="evidence-intake" aria-label="加入自己的真实资料">
    <view class="intake-heading"><view><text class="intake-kicker">01 · 你的原文</text><text class="intake-title">带上资料，再核对你的判断。</text></view><text class="intake-private">只在当前会话</text></view>
    <view v-if="!available" class="intake-boundary"><text>{{ boundaryCopy }}</text><button class="intake-text-button" role="button" :disabled="unavailable" @tap="navigateSetup" @keydown="onKeydown($event, 'setup')">{{ setupLabel }} ↗</button></view>
    <template v-else>
      <view class="intake-actions">
        <button class="intake-add" role="button" :disabled="unavailable || !uploadsAllowed" @tap="uploadFile" @keydown="onKeydown($event, 'file')">{{ phase === 'file' ? '等待选择或上传…' : '＋ 添加资料' }}</button>
        <!-- #ifdef H5 -->
        <button class="intake-paste" role="button" :aria-expanded="pasteOpen" :disabled="unavailable || !uploadsAllowed" @tap="togglePaste" @keydown="onKeydown($event, 'paste')">{{ pasteOpen ? '收起原文草稿' : '粘贴一段原文' }}</button>
        <!-- #endif -->
      </view>
      <text v-if="!uploadsAllowed" class="intake-note">当前服务未开放添加资料；仍可检查已加入的资料。</text>
      <text v-else class="intake-note">PDF / TXT / MD。解析不调用模型；发送问题时，相关片段可能交给模型。</text>
      <!-- #ifdef H5 -->
      <view v-if="pasteOpen" class="intake-paste-panel">
        <text class="intake-field-label">来源名称</text><input v-model="title" class="intake-title-input" aria-label="原文来源名称" placeholder="例如：论文名称 · 第 3 节" :maxlength="-1" :disabled="unavailable" /><text class="intake-note">将保存为 .txt；不填扩展名时最多 116 字符。名称不能有首尾空格、斜线或控制字符。</text>
        <text class="intake-field-label">原文内容</text><textarea v-model="text" class="intake-source-input" aria-label="待加入的原文内容" placeholder="粘贴你有权使用的原文。它不是判断，也不是 AI 回答。" :maxlength="-1" :disabled="unavailable" :show-confirm-bar="false" :adjust-position="true" />
        <view class="intake-counts" :class="{'intake-overflow': !validTitle || !validText}"><text>名称含扩展名 {{ filenameLength }} / 120</text><text>原文 {{ textLength }} / 12000 字符</text></view>
        <text v-if="pasteError" class="intake-error" role="status">{{ pasteError }}</text>
        <text class="intake-note">确认后发送到当前已连接的服务，作为当前会话资料解析。草稿不会自动保存到浏览器。</text>
        <button class="intake-confirm" role="button" :disabled="!canUploadText" @tap="uploadPastedText" @keydown="onKeydown($event, 'text')">{{ phase === 'text' ? '正在加入…' : '确认加入当前会话' }}</button>
        <button class="intake-text-button" role="button" :disabled="unavailable || (!title && !text)" @tap="clearPaste" @keydown="onKeydown($event, 'clear')">清空原文草稿</button>
      </view>
      <!-- #endif -->
      <view class="intake-status-row"><text class="intake-status-copy">{{ statusCopy }}</text><button class="intake-text-button" role="button" :disabled="unavailable" @tap="checkDocuments" @keydown="onKeydown($event, 'check')">{{ phase === 'check' ? '正在检查…' : checked ? '刷新状态' : '检查当前资料' }}</button></view>
      <view v-if="documents.length" class="intake-documents"><view v-for="document in documents.slice(0, 3)" :key="document.id" class="intake-document"><text class="intake-document-name">{{ document.filename }}</text><text class="intake-document-state" :class="{'intake-document-ready': document.status === 'ready' && document.chunk_count > 0}">{{ documentStatus(document) }}</text></view><text v-if="documents.length > 3" class="intake-note">另有 {{ documents.length - 3 }} 份资料，可到资料库查看。</text></view>
      <view v-if="checked && readyCount" class="intake-next"><button class="intake-ask" role="button" :disabled="unavailable" @tap="ask" @keydown="onKeydown($event, 'ask')">写下要核对的判断 <text aria-hidden="true">↓</text></button><text>只定位输入框，不会自动提问。</text></view>
      <button v-if="checked && documents.length" class="intake-text-button intake-library" role="button" :disabled="unavailable" @tap="navigateLibrary" @keydown="onKeydown($event, 'library')">查看资料与原文 ↗</button>
    </template>
    <text v-if="notice" class="intake-notice" role="status" aria-live="polite">{{ notice }}</text><text v-if="error" class="intake-error" role="alert">{{ error }}</text>
  </view>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { onHide } from '@dcloudio/uni-app';
import { research } from '../services/research.js';

const props = defineProps({sessionId:{type:String,default:''},identity:{type:String,default:''},enabled:{type:Boolean,default:false},uploadsAllowed:{type:Boolean,default:false},disabled:{type:Boolean,default:false},active:{type:Boolean,default:true},mode:{type:String,default:'demo'},connectionKind:{type:String,default:'private'},invitePending:{type:Boolean,default:false}});
const emit = defineEmits(['busy','ready','navigate','ask','dirty']);
const documents = ref([]), checked = ref(false), phase = ref('idle'), notice = ref(''), error = ref('');
const pasteOpen = ref(false), title = ref(''), text = ref('');
let generation = 0;
let lastSession = props.sessionId;
const mounted = ref(true);
const busy = computed(()=>phase.value !== 'idle');
const unavailable = computed(()=>!mounted.value || !props.active || props.disabled || busy.value);
const available = computed(()=>props.mode === 'connected' && props.enabled && !props.invitePending && /^#\d{5}$/.test(props.sessionId));
const readyCount = computed(()=>documents.value.filter(item=>item.status === 'ready' && item.chunk_count > 0).length);
const pendingCount = computed(()=>documents.value.filter(item=>['queued','processing'].includes(item.status)).length);
const failedCount = computed(()=>documents.value.filter(item=>item.status === 'failed').length);
const titleLength = computed(()=>Array.from(title.value).length), textLength = computed(()=>Array.from(text.value).length);
const filenameLength = computed(()=>title.value ? titleLength.value + (/\.txt$/i.test(title.value) ? 0 : 4) : 0);
const validUnicode = value => !Array.from(value).some(character=>{const point=character.codePointAt(0);return point === 0 || (point >= 0xd800 && point <= 0xdfff);});
const validTitle = computed(()=>filenameLength.value <= 120 && validUnicode(title.value) && title.value === title.value.trim() && !['.','..'].includes(title.value) && !/[\\/\u0000-\u001f\u007f]/.test(title.value));
const validText = computed(()=>textLength.value <= 12000 && validUnicode(text.value));
const pasteError = computed(()=>!validTitle.value ? '请修改来源名称：含 .txt 最多 120 字符，不能是 . 或 ..，不能有首尾空格、斜线、控制或无效字符。完整名称仍保留。' : !validText.value ? '原文超出 12000 字符或包含无效字符，请编辑；完整内容仍保留。' : '');
const canUploadText = computed(()=>available.value && props.uploadsAllowed && !unavailable.value && validTitle.value && validText.value && Boolean(title.value.trim()) && Boolean(text.value.trim()));
const boundaryCopy = computed(()=>props.mode !== 'connected' ? '这里测试你自己的资料，不播放固定答案。先连接真实服务，演示状态不会上传或核对你的内容。' : props.invitePending ? '请先登录邀请账号，再检查当前服务的资料能力。' : !props.enabled ? '当前服务尚未开放真实资料能力，不能把普通聊天当成资料核对。' : !props.sessionId ? '先建立一个会话，再加入属于它的原文。' : '当前会话不可用于研究资料，请重新选择会话。');
const setupLabel = computed(()=>props.invitePending ? '登录邀请账号' : props.enabled && !props.sessionId ? '先建立会话' : props.connectionKind === 'invite' ? '查看邀请账号' : '连接真实服务');
const statusCopy = computed(()=>!checked.value ? '尚未检查当前会话资料。' : !documents.value.length ? '当前会话还没有资料，从一份原文开始。' : `${readyCount.value} 份可提问${pendingCount.value ? ` · ${pendingCount.value} 份准备中` : ''}${failedCount.value ? ` · ${failedCount.value} 份处理失败` : ''}`);
function setPhase(value) { const before=busy.value;phase.value=value;if(before!==busy.value)emit('busy',busy.value); }
function reportReady() { emit('ready', checked.value ? readyCount.value : 0); }
function current(operation) { return mounted.value && props.active && !props.disabled && operation === generation && available.value && (!['file','text'].includes(phase.value) || props.uploadsAllowed); }
function invalidate({clearDraft=false} = {}) {
  const uploading=['file','text'].includes(phase.value);
  generation++;setPhase('idle');documents.value=[];checked.value=false;error.value='';
  notice.value=clearDraft ? '' : uploading ? '操作环境已改变；已提交的上传可能仍会完成。请先检查资料，不要重复上传。' : notice.value;
  if(clearDraft){title.value='';text.value='';pasteOpen.value=false;}
  reportReady();
}
function begin(value) { if(!available.value || unavailable.value)return null;const operation=++generation;setPhase(value);error.value='';notice.value='';checked.value=false;reportReady();return operation; }
function documentStatus(document) { return document.status === 'ready' ? document.chunk_count > 0 ? '可提问' : '暂无可用片段' : ({queued:'排队中',processing:'解析中',failed:'处理失败'}[document.status] || '状态待确认'); }
async function checkDocuments() {
  const operation=begin('check');if(operation===null)return;
  try { const result=await research.documents(props.sessionId);if(!current(operation))return;documents.value=result.filter(item=>item.status!=='deleted');checked.value=true;reportReady(); }
  catch { if(current(operation))error.value='暂未读到当前资料状态，请手动重试；没有提问或重复上传。'; }
  finally { if(operation===generation)setPhase('idle'); }
}
function applyUpload(result) {
  if(result?.cancelled){notice.value='已取消，没有继续添加资料。';return;}
  const document=result?.document;if(!document){error.value='未能确认资料提交结果，请先检查当前资料，不要重复上传。';return;}
  // An upload receipt is one document, never a complete library snapshot.
  documents.value=[...documents.value.filter(item=>item.id!==document.id),document];
  notice.value=`已收到「${document.filename}」的提交回执：${documentStatus(document)}。请检查当前资料确认最新状态。`;
  checked.value=false;reportReady();
}
async function uploadFile() {
  if(!props.uploadsAllowed)return;
  const operation=begin('file');if(operation===null)return;
  try { const result=await research.upload(props.sessionId,{isCurrent:()=>current(operation)});if(current(operation))applyUpload(result); }
  catch { if(current(operation))error.value='添加未完成或结果尚未确认。请先检查当前资料；没有自动重试上传。'; }
  finally { if(operation===generation)setPhase('idle'); }
}
function togglePaste() { if(!available.value || unavailable.value || !props.uploadsAllowed)return;pasteOpen.value=!pasteOpen.value; }
function clearPaste() { if(unavailable.value)return;title.value='';text.value='';notice.value='未提交的原文草稿已清空，没有移除已上传资料。'; }
async function uploadPastedText() {
  if(!canUploadText.value)return;
  const operation=begin('text');if(operation===null)return;
  const originalTitle=title.value,originalText=text.value;
  try {
    const result=await research.uploadText(props.sessionId,{title:originalTitle,text:originalText,isCurrent:()=>current(operation)});
    if(!current(operation))return;
    applyUpload(result);
    if(result?.document && !result.cancelled && title.value===originalTitle && text.value===originalText){title.value='';text.value='';pasteOpen.value=false;}
  } catch { if(current(operation))error.value='原文提交未完成或结果尚未确认。草稿已保留，请先检查资料，不要重复上传。'; }
  finally { if(operation===generation)setPhase('idle'); }
}
function ask() { if(available.value && checked.value && readyCount.value && !unavailable.value)emit('ask'); }
function navigateLibrary() { if(available.value && !unavailable.value)emit('navigate','documents'); }
function navigateSetup() { if(unavailable.value)return;emit('navigate',props.invitePending ? 'invite' : props.enabled && !props.sessionId ? 'chat' : props.connectionKind==='invite' ? 'invite' : 'settings'); }
function onKeydown(event,action) {
  if(!event || event.defaultPrevented || event.repeat || event.isComposing || event.keyCode===229 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || !['Enter',' ','Spacebar'].includes(event.key))return;
  const actions={check:checkDocuments,file:uploadFile,paste:togglePaste,text:uploadPastedText,clear:clearPaste,ask,library:navigateLibrary,setup:navigateSetup};
  if(!actions[action])return;event.preventDefault?.();void actions[action]();
}
watch(()=>Boolean(title.value || text.value),value=>emit('dirty',value),{flush:'sync'});
watch(()=>[props.identity,props.mode,props.connectionKind,props.invitePending],()=>{lastSession=props.sessionId;invalidate({clearDraft:true});},{flush:'sync'});
watch(()=>props.sessionId,value=>{const changed=Boolean(value && lastSession && value!==lastSession);if(value)lastSession=value;invalidate({clearDraft:changed});},{flush:'sync'});
watch(()=>[props.active,props.enabled],([active,enabled])=>{if(!active || !enabled)invalidate();},{flush:'sync'});
watch(()=>props.disabled,value=>{if(value && busy.value)invalidate();},{flush:'sync'});
watch(()=>props.uploadsAllowed,value=>{if(!value && ['file','text'].includes(phase.value))invalidate();},{flush:'sync'});
onHide(()=>invalidate());
onBeforeUnmount(()=>{mounted.value=false;invalidate({clearDraft:true});});
</script>

<style scoped>
.evidence-intake { padding:18px; border:1px solid #d9e2d1; border-radius:18px; background:#fffefa; color:#243c31; }
.intake-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:15px; }
.intake-kicker { display:block; color:#698054; font-size:10px; letter-spacing:1px; }
.intake-title { display:block; margin-top:6px; font-size:16px; font-weight:650; line-height:1.5; }
.intake-private { color:#71806a; font-size:10px; white-space:nowrap; margin-top:3px; }
.intake-actions { display:flex; gap:10px; }
.intake-add,.intake-paste,.intake-confirm,.intake-ask,.intake-text-button { display:flex; align-items:center; justify-content:center; min-height:44px; margin:0; padding:9px 13px; box-sizing:border-box; border-radius:11px; border:1px solid #d9e3cd; font-size:12px; line-height:24px; }
.intake-add,.intake-paste { flex:1; background:#eff4e7; color:#2f4d31; }
.intake-paste { background:#fffefa; }
.intake-add::after,.intake-paste::after,.intake-confirm::after,.intake-ask::after,.intake-text-button::after { border:0; }
.intake-add[disabled],.intake-paste[disabled],.intake-confirm[disabled],.intake-ask[disabled] { color:#8b9782; background:#f0f2ec; border-color:#e2e7da; }
.intake-note { display:block; margin-top:9px; color:#73816b; font-size:10px; line-height:1.8; }
.intake-status-row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px; border-top:1px solid #ecf0e6; padding-top:7px; }
.intake-status-copy { color:#67785c; font-size:11px; line-height:1.8; }
.intake-text-button { flex-shrink:0; border:0; color:#446336; background:transparent; padding:8px 0; }
.intake-text-button[disabled] { color:#95a08c; background:transparent; }
.intake-documents { display:grid; gap:8px; margin-top:3px; }
.intake-document { display:flex; align-items:center; gap:12px; justify-content:space-between; font-size:11px; }
.intake-document-name { min-width:0; overflow-wrap:anywhere; color:#526249; line-height:1.7; }
.intake-document-state { flex-shrink:0; padding:3px 6px; border-radius:5px; background:#f0f0e6; color:#827755; font-size:10px; }
.intake-document-ready { background:#eaf2df; color:#54713c; }
.intake-boundary { color:#68765f; font-size:12px; line-height:1.8; }
.intake-boundary .intake-text-button { justify-content:flex-start; margin-top:4px; }
.intake-paste-panel { margin-top:14px; padding:14px; border:1px solid #e0e7d5; border-radius:13px; background:#f7f9f1; }
.intake-field-label { display:block; margin:10px 0 7px; color:#516745; font-size:11px; }
.intake-field-label:first-child { margin-top:0; }
.intake-title-input,.intake-source-input { width:100%; box-sizing:border-box; border:1px solid #d9e3cd; border-radius:9px; background:#fffefa; color:#2d4131; padding:11px; font-size:13px; line-height:1.7; }
.intake-title-input { min-height:44px; }
.intake-source-input { min-height:142px; height:142px; }
.intake-counts { display:flex; flex-wrap:wrap; justify-content:space-between; gap:5px; margin-top:7px; color:#7b886f; font-size:10px; }
.intake-overflow,.intake-error { color:#9b4c3d; }
.intake-confirm,.intake-ask { width:100%; margin-top:12px; border-color:#253e30; background:#253e30; color:#e6f4d1; }
.intake-next > text { display:block; margin-top:6px; color:#748169; font-size:10px; }
.intake-ask { justify-content:space-between; }
.intake-library { min-height:44px; }
.intake-notice,.intake-error { display:block; margin-top:10px; font-size:11px; line-height:1.8; overflow-wrap:anywhere; }
.intake-notice { color:#667754; }
.intake-add:focus-visible,.intake-paste:focus-visible,.intake-confirm:focus-visible,.intake-ask:focus-visible,.intake-text-button:focus-visible { outline:2px solid #6b8c4a; outline-offset:3px; }
@media(max-width:390px) { .evidence-intake { padding:15px; } .intake-title { font-size:15px; } .intake-private { font-size:9px; } .intake-actions { gap:8px; } .intake-add,.intake-paste { padding:9px 10px; font-size:11px; } }
</style>
