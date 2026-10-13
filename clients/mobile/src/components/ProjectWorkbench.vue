<template>
  <view class="project-workbench" :class="{'workbench-compact':compact}">
    <button v-if="compact" class="workbench-disclosure" role="button" :disabled="disabled || busy" :aria-expanded="opened" @tap="toggle"><text class="disclosure-title">项目工具<text v-if="briefDirty" class="unsaved-dot"> · 未保存</text></text><text class="disclosure-hint">快捷起草 · 简报 · 成果</text><text aria-hidden="true">{{ opened ? '−' : '＋' }}</text></button>
    <view v-else class="workbench-heading"><view><text class="workbench-kicker">DOCTHINKER / PROJECT DESK</text><text class="workbench-title">把这件事，接着做下去。</text></view><text class="workbench-tag">你的项目 · 你的边界</text></view>
    <view v-if="!compact || opened" class="recipe-grid"><button v-for="recipe in taskRecipes" :key="recipe.id" class="recipe" role="button" :disabled="disabled || busy" @tap="chooseRecipe(recipe.id)"><text class="recipe-symbol" aria-hidden="true">{{ recipe.symbol }}</text><view><text class="recipe-title">{{ recipe.title }}</text><text class="recipe-detail">{{ recipe.detail }}</text></view><text class="recipe-arrow" aria-hidden="true">＋</text></button></view>
    <view v-if="!compact || opened" class="workbench-bottom"><text class="workbench-caption">配方只整理草稿，不自动发送。</text><button v-if="!compact" class="desk-toggle" role="button" :disabled="disabled || busy || !sessionId" :aria-expanded="opened" @tap="toggle">{{ opened ? '收起项目桌面 −' : '简报与成果 ＋' }}</button></view>
    <view v-if="opened" class="desk-panel">
      <view class="desk-panel-heading"><text class="desk-section-title">项目简报</text><button class="desk-link" role="button" :disabled="disabled || busy || demo" @tap="refresh">{{ busy ? '处理中…' : '重新读取' }}</button></view>
      <text class="desk-note">{{ demo ? '演示只体验配方，不保存简报或成果。' : '只在当前服务的当前项目保存。简报是你的要求，不是事实证据，也不是自动使用的记忆。' }}</text>
      <view v-if="error" class="desk-error" role="status">{{ error }}</view><text v-if="notice" class="desk-notice" role="status" aria-live="polite">{{ notice }}</text>
      <template v-if="loaded && !demo">
        <text v-if="!enabled" class="desk-note">当前服务尚未开放项目简报；配方仍可使用，没有保存任何内容。</text>
        <template v-else>
          <view class="brief-grid"><view class="brief-goal"><text class="desk-label">这次想完成什么</text><textarea v-model="editor.goal" class="brief-field" :maxlength="-1" :disabled="disabled || busy" auto-height placeholder="目标，例如：比较两条研究路线，形成决策建议" aria-label="项目目标" /></view><view><text class="desk-label">希望交付什么</text><textarea v-model="editor.output_format" class="brief-field" :maxlength="-1" :disabled="disabled || busy" auto-height placeholder="例如：一页简报，给研究组讨论" aria-label="预期交付" /></view></view>
          <text class="desk-label">必须保留的边界</text><textarea v-model="editor.constraints" class="brief-field brief-constraints" :maxlength="-1" :disabled="disabled || busy" auto-height placeholder="资料范围、时间、不能下的结论，以及需要我确认的事" aria-label="项目约束" />
          <text class="desk-note">目标 {{ Array.from(editor.goal).length }}/1200 · 交付 {{ Array.from(editor.output_format).length }}/400 · 约束 {{ Array.from(editor.constraints).length }}/1200 · 版本 {{ editor.revision }}</text>
          <text v-if="briefDirty" class="desk-unsaved" role="status" aria-live="polite">有未保存的简报编辑</text>
          <view class="desk-actions"><button class="desk-primary" role="button" :disabled="disabled || busy || conflict || !validBrief" @tap="saveBrief">保存简报</button><button class="desk-secondary" role="button" :disabled="disabled || busy || !hasBrief || !validBrief" @tap="includeBrief">加入提问草稿 ↗</button></view><text class="desk-note">保存不会请求模型。加入草稿后可编辑，仍由你点击发送；不会在以后偷偷携带。</text>
          <view v-if="conflict" class="brief-conflict"><text>服务已有更新版本。你的编辑仍保留，不能直接覆盖。</text><button class="desk-secondary" role="button" :disabled="disabled || busy" @tap="refresh">读取最新版本（先确认）</button></view>
        </template>
        <view class="desk-panel-heading artifacts-heading"><text class="desk-section-title">成果架</text><text class="desk-count">{{ artifacts.length }} 份 · 非事实认证</text></view>
        <text class="desk-note">收藏的是回答编号，每次打开重新读取可访问的回答和原文；不把旧片段另存成快照。</text>
        <text v-if="!artifactsEnabled" class="desk-note">当前服务尚未开放成果收藏，没有读取或保存旧成果。</text>
        <template v-else>
          <view v-if="answerChoices.length" class="artifact-save"><picker :range="answerLabels" :value="candidateIndex" :disabled="disabled || busy" @change="candidateIndex = Number($event.detail.value)"><view class="artifact-picker">{{ answerLabels[candidateIndex] || '选择本项目的一次回答' }} ⌄</view></picker><input v-model="artifactTitle" class="artifact-title" :maxlength="120" :disabled="disabled || busy" placeholder="给成果起个名字" aria-label="成果标题" /><button class="desk-secondary" role="button" :disabled="disabled || busy || !artifactTitle.trim()" @tap="saveArtifact">收藏回答</button></view>
          <text v-else class="desk-note">获得真实回答后，可在这里收藏。不会收藏演示或没有持久编号的回答。</text>
          <view v-for="artifact in artifacts" :key="artifact.answer_id" class="artifact-row"><view><text class="artifact-name">{{ artifact.title }}</text><text class="desk-note">{{ artifact.available ? '点击读取并检查出处' : '原回答暂不可访问，不展示旧快照' }}</text></view><view class="artifact-actions"><button class="desk-link" role="button" :disabled="disabled || busy || !artifact.available" @tap="openArtifact(artifact.answer_id)">审阅</button><button class="desk-link" role="button" :disabled="disabled || busy" @tap="removeArtifact(artifact)">移除</button></view></view>
          <view v-if="selectedArtifact?.answer" class="artifact-review"><view class="desk-panel-heading"><text class="desk-section-title">{{ selectedArtifact.title }}</text><button class="desk-link" role="button" @tap="selectedId = ''">收起</button></view><text class="desk-note">来源撤回后会按规则遮蔽相关内容，引用仍需核对；这里不会重新生成回答。</text><text v-if="selectedArtifact.answer.citation_validation?.warnings?.some(item=>['source_revoked','provenance_unavailable'].includes(item.code))" class="desk-error">这份回答的部分原文已撤回或依据记录不可用，不能继续当作有据可查的结论。</text>
            <AnswerBody class="artifact-content" :text="selectedArtifact.answer.content" />
            <AnswerEvidence :sources="selectedArtifact.answer.sources" />
            <ReasoningEvidence v-if="selectedArtifact.answer.research_core" :reasoning="selectedArtifact.answer.reasoning" :claim-audit="selectedArtifact.answer.claim_audit" :sources="selectedArtifact.answer.sources" :citation-validation="selectedArtifact.answer.citation_validation" />
            <AnswerDeliverable :answer="selectedArtifact.answer.content" :sources="selectedArtifact.answer.sources" :research-core="selectedArtifact.answer.research_core === true" :citation-validation="selectedArtifact.answer.citation_validation" :answer-mode="selectedArtifact.answer.answer_mode" :retrieval-gaps="selectedArtifact.answer.retrieval_gaps" :reasoning="selectedArtifact.answer.reasoning" :claim-audit="selectedArtifact.answer.claim_audit" :disabled="disabled || busy" />
          </view>
        </template>
      </template>
    </view>
  </view>
</template>

<script setup>
import {computed, ref, watch} from 'vue';
import {onHide, onUnload} from '@dcloudio/uni-app';
import {taskRecipes, recipeDraft, briefDraft, createWorkbenchClient} from '../services/project-workbench.js';
import {isFeedbackAnswerId, researchConnectionKey} from '../services/api.js';
import AnswerEvidence from './AnswerEvidence.vue';
import AnswerBody from './AnswerBody.vue';
import AnswerDeliverable from './AnswerDeliverable.vue';
import ReasoningEvidence from './ReasoningEvidence.vue';
const props = defineProps({sessionId:{type:String,default:''},identity:{type:String,default:''},demo:{type:Boolean,default:false},compact:{type:Boolean,default:false},disabled:{type:Boolean,default:false},answers:{type:Array,default:()=>[]}});
const emit = defineEmits(['draft','busy','dirty']);
const client = createWorkbenchClient();
const emptyBrief = () => ({goal:'',output_format:'',constraints:'',revision:0});
const opened=ref(false),loaded=ref(false),enabled=ref(false),artifactsEnabled=ref(false),busy=ref(false),error=ref(''),notice=ref(''),conflict=ref(false);
const editor=ref(emptyBrief()),saved=ref(''),artifacts=ref([]),selectedId=ref(''),openedArtifact=ref(null),artifactTitle=ref('项目成果'),candidateIndex=ref(0);
let generation=0;
let scopeIdentity=props.identity,scopeSession=props.sessionId;
const validBrief=computed(()=>Array.from(editor.value.goal).length<=1200 && Array.from(editor.value.output_format).length<=400 && Array.from(editor.value.constraints).length<=1200);
const hasBrief=computed(()=>Boolean(briefDraft(editor.value)));
const briefDirty=computed(()=>Boolean(saved.value) && JSON.stringify(editor.value)!==saved.value);
const answerChoices=computed(()=>props.answers.filter(answer=>answer.role==='assistant' && isFeedbackAnswerId(answer.id,false)).slice(-20).reverse());
const answerLabels=computed(()=>answerChoices.value.map((_,index)=>index===0?'最近一次回答':`往前第 ${index+1} 次回答`));
const selectedArtifact=computed(()=>openedArtifact.value?.answer_id===selectedId.value ? openedArtifact.value : null);
function clear() { generation++;opened.value=false;loaded.value=false;enabled.value=false;artifactsEnabled.value=false;editor.value=emptyBrief();saved.value='';artifacts.value=[];selectedId.value='';openedArtifact.value=null;artifactTitle.value='项目成果';candidateIndex.value=0;conflict.value=false;error.value='';notice.value=''; }
watch(()=>[props.identity,props.sessionId],([identity,session])=>{
  // Chat temporarily clears its session while refreshing. Retain unsaved text
  // only in this component's memory; never reveal it in another project/account.
  if(identity!==scopeIdentity || (session && session!==scopeSession)) {clear();scopeIdentity=identity;scopeSession=session;}
});
watch(()=>answerChoices.value.map(item=>item.id).join(','),()=>{candidateIndex.value=0;});
function allowed() { return !props.disabled && !busy.value; }
function chooseRecipe(id) { if(allowed()) { const draft=recipeDraft(id);if(draft) emit('draft',draft); } }
watch(selectedId,id=>{if(!id)openedArtifact.value=null;});
watch(briefDirty,dirty=>{emit('dirty',dirty);if(dirty && notice.value.startsWith('简报已保存'))notice.value='';});
function includeBrief() { if(allowed() && enabled.value && validBrief.value && hasBrief.value) emit('draft',briefDraft(editor.value)); }
async function operation(action) {
  if(!allowed() || props.demo || !props.sessionId || !opened.value) return;
  const version=generation,identity=props.identity,session=props.sessionId;
  const current=()=>version===generation && identity===props.identity && session===props.sessionId && identity===researchConnectionKey() && opened.value;
  if(!current()) return;
  busy.value=true;emit('busy',true);error.value='';notice.value='';
  try { await action(current,session); }
  catch(issue) { if(current()) { error.value=issue?.status===409?'简报已被其他页面更新。你的草稿仍保留，请读取最新版本后再修改。':String(issue?.message || '工作台操作未确认，请手动刷新。没有自动重试。');if(issue?.status===409)conflict.value=true; } }
  finally { busy.value=false;emit('busy',false); }
}
function adopt(value) { enabled.value=value.enabled;artifactsEnabled.value=value.artifactsEnabled;editor.value=value.brief || emptyBrief();saved.value=JSON.stringify(editor.value);artifacts.value=value.items;loaded.value=true;conflict.value=false;selectedId.value=''; }
function read(current,session,{preserveDraft=false}={}) {
  const local=preserveDraft && briefDirty.value ? {editor:{...editor.value},saved:saved.value} : null;
  return client.load(session,{isCurrent:current}).then(value=>{
    if(!current())return;
    adopt(value);
    if(local && value.enabled) {
      editor.value=local.editor;saved.value=local.saved;
      conflict.value=local.editor.revision!==value.brief.revision;
      notice.value=conflict.value?'服务上的简报已有更新，你未保存的编辑仍保留；请核对后再处理。':'已恢复本页未保存的简报，请保存后再离开项目。';
    }
  });
}
async function toggle() { if(!allowed())return;if(opened.value){opened.value=false;selectedId.value='';return;}opened.value=true;if(!props.demo && !loaded.value)await operation((current,session)=>read(current,session,{preserveDraft:true})); }
async function confirm(options) { return new Promise(resolve=>{try{uni.showModal({...options,success:value=>resolve(value.confirm===true),fail:()=>resolve(false)});}catch{resolve(false);}}); }
async function refresh() {
  if(!allowed())return;
  await operation(async(current,session)=>{
    if(enabled.value && JSON.stringify(editor.value)!==saved.value && !(await confirm({title:'读取最新简报？',content:'当前尚未保存的简报编辑会被最新版本替换。可以先用“加入提问草稿”保留你的文字。',confirmText:'读取最新'})))return;
    if(current())await read(current,session);
  });
}
async function saveBrief() { if(!enabled.value || conflict.value || !validBrief.value)return;await operation(async(current,session)=>{const value=await client.saveBrief(session,{...editor.value},{isCurrent:current});if(current()){editor.value=value;saved.value=JSON.stringify(value);notice.value='简报已保存，没有调用模型，也未加入任何提问。';}}); }
async function saveArtifact() {
  if(!artifactsEnabled.value)return;const candidate=answerChoices.value[candidateIndex.value];if(!candidate)return;
  await operation(async(current,session)=>{const item=await client.saveArtifact(session,candidate.id,artifactTitle.value,{isCurrent:current});if(current()){const metadata={answer_id:item.answer_id,title:item.title,available:item.available};artifacts.value=[metadata,...artifacts.value.filter(value=>value.answer_id!==item.answer_id)].slice(0,100);notice.value='已收藏回答编号，没有复制资料快照或调用模型。';}});
}
async function openArtifact(id) {
  if(!artifactsEnabled.value)return;selectedId.value='';openedArtifact.value=null;
  await operation(async(current,session)=>{const value=await client.artifact(session,id,{isCurrent:current});if(current()){openedArtifact.value=value.available?value:null;selectedId.value=value.available?id:'';if(!selectedId.value)notice.value='该成果已移除或不可访问，没有使用旧快照。';}});
}
async function removeArtifact(item) { if(!artifactsEnabled.value)return;await operation(async(current,session)=>{if(!(await confirm({title:'移除成果收藏？',content:'只移除这条收藏，不删除原回答、资料或确认记忆。',confirmText:'移除收藏'})) || !current())return;await client.removeArtifact(session,item.answer_id,{isCurrent:current});if(current()){artifacts.value=artifacts.value.filter(value=>value.answer_id!==item.answer_id);if(selectedId.value===item.answer_id)selectedId.value='';notice.value='收藏已移除，原回答没有删除。';}}); }
function hide() {
  const local=briefDirty.value ? {editor:{...editor.value},saved:saved.value} : null;
  clear();
  if(local){editor.value=local.editor;saved.value=local.saved;}
}
onHide(hide);onUnload(clear);
</script>

<style scoped>
.project-workbench{max-width:820px;margin:0 auto 24px;padding:22px;border:1px solid #dce3d5;border-radius:20px;background:#fffefa;color:#203d35}.workbench-heading,.workbench-bottom,.desk-panel-heading,.desk-actions,.artifact-row,.artifact-actions{display:flex;align-items:center;justify-content:space-between;gap:12px}.workbench-kicker{display:block;color:#6b7f59;font-size:10px;letter-spacing:1.2px}.workbench-title{display:block;font-size:21px;font-weight:600;margin-top:5px}.workbench-tag{padding:5px 9px;border-radius:7px;background:#edf5da;font-size:10px;white-space:nowrap}.recipe-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:20px}.recipe{display:flex;align-items:center;gap:10px;margin:0;padding:14px;text-align:left;border:1px solid #e2e7da;border-radius:13px;background:#f7f8f1;line-height:1.5;min-height:76px;color:#203d35}.recipe::after,.project-workbench button::after{border:0}.recipe-symbol{display:flex;align-items:center;justify-content:center;width:30px;height:32px;border-radius:9px;background:#e9efdb;font-size:22px;flex-shrink:0}.recipe-title{display:block;font-size:13px;font-weight:600}.recipe-detail{display:block;font-size:11px;color:#63715c;margin-top:3px}.recipe-arrow{margin-left:auto;color:#6d835a;font-size:19px}.workbench-bottom{margin-top:8px}.workbench-caption,.desk-note,.desk-count{display:block;color:#60705a;font-size:11px;line-height:1.85}.desk-toggle,.desk-link{background:transparent;color:#416231;font-size:12px;margin:0;padding:8px 3px;min-height:44px;line-height:1.7}.desk-panel{margin-top:10px;padding-top:16px;border-top:1px solid #e4e9da}.desk-section-title{font-size:16px;font-weight:600}.desk-panel-heading{margin-bottom:7px}.brief-grid{display:grid;grid-template-columns:1.2fr 1fr;gap:12px}.desk-label{display:block;font-size:12px;margin:15px 0 6px;color:#465e3d}.brief-field{width:100%;min-height:90px;max-height:280px;padding:12px;border:1px solid #dce3d2;border-radius:12px;background:#f6f8ef;color:#2e4531;font-size:13px;line-height:1.8;box-sizing:border-box}.brief-constraints{min-height:70px}.desk-actions{justify-content:flex-start;flex-wrap:wrap;margin:13px 0 8px}.desk-primary,.desk-secondary{min-height:44px;margin:0;padding:11px 15px;border-radius:11px;font-size:12px;line-height:1.8;background:#203d35;color:#fff}.desk-secondary{background:#edf4de;border:1px solid #d4dfc2;color:#3c582e}.desk-error,.brief-conflict{padding:12px;margin:10px 0;border:1px solid #e4d1ad;border-radius:10px;background:#fbf5e8;color:#7b5828;font-size:12px;line-height:1.85}.brief-conflict button{margin-top:10px}.desk-notice{display:block;margin:10px 0;font-size:12px;color:#426137}.artifacts-heading{margin-top:25px;padding-top:17px;border-top:1px solid #e3e7db}.artifact-save{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:15px 0}.artifact-picker{min-height:44px;padding:10px 12px;background:#f0f3e7;border-radius:10px;font-size:12px}.artifact-title{flex:1;min-width:120px;min-height:44px;padding:9px 10px;border:1px solid #dce3d2;border-radius:10px;font-size:13px;box-sizing:border-box}.artifact-row{padding:13px 0;border-bottom:1px solid #e6e9df}.artifact-name{display:block;font-size:13px;overflow-wrap:anywhere}.artifact-actions{flex-shrink:0}.artifact-review{padding:17px;margin-top:15px;background:#f3f5ec;border:1px solid #dce4d1;border-radius:14px}.artifact-content{display:block;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.95;margin-top:15px}.project-workbench button[disabled]{opacity:.45}
.desk-unsaved{display:block;margin-top:9px;padding:7px 10px;border-left:3px solid #a77929;background:#faf2df;color:#76501c;font-size:12px;font-weight:600;line-height:1.8}
.workbench-compact{padding:0 16px;margin-bottom:16px;border-color:#e0e4da;background:transparent;border-radius:13px}.workbench-disclosure{display:flex;align-items:center;gap:12px;width:100%;min-height:48px;padding:10px 0;margin:0;background:transparent;color:#42503e;line-height:1.6;text-align:left;font-size:15px}.disclosure-title{font-size:13px;font-weight:600}.disclosure-hint{flex:1;text-align:right;font-size:11px;color:#6b7665}.workbench-compact .recipe-grid{margin-top:2px}.workbench-compact .workbench-bottom{padding-bottom:12px}.workbench-compact .desk-panel{padding-bottom:17px}.unsaved-dot{color:#936421;font-size:11px}
@media(max-width:600px){.project-workbench{padding:17px;border-radius:17px}.workbench-tag{display:none}.workbench-title{font-size:19px}.recipe-grid{gap:7px}.recipe{padding:11px;min-height:92px;align-items:flex-start}.recipe-symbol{width:24px;height:28px;font-size:19px}.recipe-detail{font-size:10px}.recipe-arrow{display:none}.brief-grid{grid-template-columns:1fr}.workbench-caption{max-width:135px}.artifact-save{align-items:stretch}.artifact-save picker{width:100%}.artifact-row{align-items:flex-start}.artifact-review{padding:13px}}
</style>
