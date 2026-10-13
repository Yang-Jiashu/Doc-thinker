<template>
  <view class="mobile-page first-value-page">
    <BrandHeader />
    <view class="fv-topbar"><button class="quiet-button" role="button" tabindex="0" @tap="openChat" @keydown="activate($event, openChat)">← 回到助手</button><text class="fv-sample-label">真实论文 · 预先整理 · 不调用模型</text></view>
    <view class="fv-heading"><text class="fv-eyebrow">先有一个发现，再开始自己的项目</text><text class="fv-title">{{ sample.question }}</text><text class="fv-subtitle">不用先上传。点开一句判断，看看它从哪来。</text></view>

    <view v-if="savedProgress && !resumed" class="fv-resume" role="status"><view><text class="fv-resume-title">上次的思路，还在这里。</text><text class="fv-muted">你在这台设备保存过这个样例的阅读位置。</text></view><button class="secondary-button" role="button" tabindex="0" @tap="resume" @keydown="activate($event, resume)">接着看 ↗</button></view>

    <view class="fv-workspace">
      <view class="fv-main">
        <view class="fv-answer">
          <view class="fv-answer-top"><text>先看结论</text><text class="fv-answer-tag">有依据，也有边界</text></view>
          <text class="fv-answer-title">多给资料，<text class="fv-lime">不等于更懂。</text></text>
          <text class="fv-answer-copy">{{ sample.answer }}</text>
          <button class="fv-proof-button" role="button" tabindex="0" @tap="inspectFirst" @keydown="activate($event, inspectFirst)">看看容易漏掉的一点 <text aria-hidden="true">↗</text></button>
          <text class="fv-answer-note">{{ sample.boundary }}</text>
        </view>
        <view id="sample-route" class="fv-route">
          <view class="fv-section-title"><text>把依据接起来</text><text class="fv-small">点一步，核对一句</text></view>
          <button v-for="(step, index) in sample.steps" :key="step.id" class="fv-step" :class="{'fv-step-active':selectedStepId === step.id}" role="button" tabindex="0" :aria-pressed="selectedStepId === step.id" aria-controls="sample-source" @tap="selectStep(step.id)" @keydown="activate($event, () => selectStep(step.id))"><text class="fv-step-number">0{{ index + 1 }}</text><view class="fv-step-body"><text class="fv-step-title">{{ step.title }}</text><text class="fv-step-detail">{{ step.detail }}</text></view><text class="fv-step-arrow" aria-hidden="true">↗</text></button>
          <text class="fv-route-note">论文摘录与编者解读分开；这条阅读路线不是自动生成的因果证明。</text>
        </view>
      </view>

      <view id="sample-source" class="fv-source" role="region" aria-label="论文原文与解读">
        <template v-if="activeSource">
          <view class="fv-section-title"><text>回到原文</text><text class="fv-source-status">{{ activeSource.year }} · {{ activeSource.locator }}</text></view>
          <view v-if="stepSources.length > 1" class="fv-source-tabs"><button v-for="(source,index) in stepSources" :key="source.id" role="button" tabindex="0" :aria-pressed="selectedSourceId === source.id" :class="{'fv-source-tab-active':selectedSourceId === source.id}" @tap="selectSource(source.id)" @keydown="activate($event, () => selectSource(source.id))">论文 {{ index+1 }}</button></view>
          <text class="fv-paper-title">{{ activeSource.title }}</text>
          <text class="fv-paper-author">{{ activeSource.authors }}</text>
          <text class="fv-quote-label">原文短摘录</text>
          <!-- #ifdef MP-WEIXIN -->
          <text class="fv-quote" user-select>{{ activeSource.quote }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="fv-quote" selectable>{{ activeSource.quote }}</text>
          <!-- #endif -->
          <view class="fv-interpretation"><text class="fv-quote-label">编者解读 · 不是原文逐字翻译</text><text>{{ activeSource.explanation }}</text></view>
          <!-- #ifdef H5 -->
          <a class="fv-paper-link" :href="activeSource.url" target="_blank" rel="noopener noreferrer">打开论文原页 ↗</a>
          <!-- #endif -->
          <!-- #ifndef H5 -->
          <button class="quiet-button" role="button" @tap="copySourceLink">复制论文原页链接 ↗</button>
          <!-- #endif -->
          <button class="fv-return-route quiet-button" role="button" tabindex="0" @tap="backToRoute" @keydown="activate($event, backToRoute)">← 回到刚才这一步</button>
        </template>
        <template v-else><view class="fv-empty-visual" aria-hidden="true"><text>01</text><text>—</text><text>02</text><text>—</text><text>?</text></view><text class="fv-source-empty-title">一个判断，不必只听 AI 说。</text><text class="fv-source-empty-copy">两篇真实论文就在这里。选择左侧的一步，核对它的依据和适用范围。</text><button class="secondary-button" role="button" tabindex="0" @tap="inspectFirst" @keydown="activate($event, inspectFirst)">打开第一处依据</button></template>
      </view>
    </view>

    <view class="fv-next" id="sample-next"><view class="fv-next-copy"><text class="fv-eyebrow">不是看完就散了</text><text class="fv-next-title">留住这一步，下次接着想。</text><text class="fv-next-question">下一问：{{ sample.nextQuestion }}</text><text class="fv-muted">只在本设备保留样例位置，30 天内可继续；不写入你的项目记忆，不上传。</text></view><view class="fv-next-actions"><button class="secondary-button" role="button" tabindex="0" :disabled="!activeStep" @tap="savePlace" @keydown="activate($event, savePlace)">{{ savedProgress ? '更新样例位置' : '留住这一步' }}</button><button v-if="savedProgress" class="quiet-button" role="button" tabindex="0" @tap="forgetPlace" @keydown="activate($event, forgetPlace)">清除样例位置</button></view></view>
    <text v-if="notice" class="fv-notice" role="status" aria-live="polite">{{ notice }}</text>

    <view class="fv-own"><view><text class="fv-own-title">现在，换成你在意的问题。</text><text class="fv-muted">用自己的资料核对观点，答案仍需查证。</text></view><button class="primary-button" role="button" tabindex="0" @tap="showOwnMaterials" @keydown="activate($event, showOwnMaterials)">换成我的资料 ↗</button></view>
    <view v-if="ownMaterialsOpen" id="sample-handoff" class="fv-handoff" role="region" aria-label="开始使用自己的资料"><text class="fv-handoff-title">{{ handoff.title }}</text><text class="fv-muted">{{ handoff.detail }}</text><button class="secondary-button" role="button" tabindex="0" @tap="continueWithOwn" @keydown="activate($event, continueWithOwn)">{{ handoff.label }}</button><text class="fv-small">不会自动切换连接、上传文件或提交问题。</text></view>
    <text class="fv-footnote">这是基于公开论文、预先整理的体验样例，不是当前模型现场生成，也不是 DocThinker 的效果测评。</text>
  </view>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue';
import { onShow, onHide, onUnload } from '@dcloudio/uni-app';
import BrandHeader from '../../components/BrandHeader.vue';
import { firstValueCase } from '../../services/first-value-case.js';
import { loadFirstValueProgress, saveFirstValueProgress, clearFirstValueProgress } from '../../services/first-value-progress.js';
import { getPreferences, getActiveSession } from '../../services/api.js';

const sample = firstValueCase;
const selectedStepId = ref(''), selectedSourceId = ref(''), savedProgress = ref(null), resumed = ref(false), notice = ref(''), ownMaterialsOpen = ref(false);
const preferences = ref({mode:'demo'}), hasSession = ref(false);
let visible = true, copyGeneration = 0;
const scope = {caseId:sample.id,version:sample.version};
const activeStep = computed(() => sample.steps.find(step => step.id === selectedStepId.value) || null);
const stepSources = computed(() => (activeStep.value?.sourceIds || []).map(id => sample.sources.find(source => source.id === id)).filter(Boolean));
const activeSource = computed(() => stepSources.value.find(source => source.id === selectedSourceId.value) || null);
const handoff = computed(() => {
  if (preferences.value.connectionKind === 'invite' && preferences.value.invitePending) return {title:'先登录你的邀请账号',detail:'样例不需要登录。自己的资料需要已开通资料功能的账号；登录后按服务开放范围使用。',label:'去登录',route:'/pages/invite/index',kind:'navigate'};
  if (preferences.value.mode !== 'connected') return {title:'样例已看完，真实资料需要连接服务',detail:'当前是离线体验。连接你的私人服务后，才能上传真实文件；这里不会把演示上传当成真实处理。',label:'去连接服务',route:'/pages/settings/index',kind:'tab'};
  if (!hasSession.value) return {title:'先为你的资料建一个会话',detail:'回到助手新建会话，再到「资料」选择文件。文件归当前会话，不会混入这个样例。',label:'去助手新建会话',route:'/pages/chat/index',kind:'tab'};
  return {title:'把资料放进当前会话',detail:'进入资料页后由你选择文件并确认。能否上传以当前服务权限为准；解析完成后再回助手提问。',label:'打开我的资料库',route:'/pages/documents/index',kind:'tab'};
});
function activate(event, action) { if (!['Enter',' ','Spacebar'].includes(event.key) || event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return; event.preventDefault?.(); action(); }
async function scrollTo(selector) { await nextTick(); if (visible && typeof uni.pageScrollTo === 'function') uni.pageScrollTo({selector,duration:180}); }
function selectStep(id) { const step = sample.steps.find(item => item.id === id); if (!step || !visible) return; copyGeneration++; notice.value=''; selectedStepId.value=id; selectedSourceId.value=step.sourceIds[0] || ''; void scrollTo('#sample-source'); }
function selectSource(id) { if (!visible || !stepSources.value.some(source => source.id === id)) return; copyGeneration++; notice.value=''; selectedSourceId.value=id; }
function inspectFirst() { selectStep(sample.steps[1]?.id || sample.steps[0].id); }
function backToRoute() { void scrollTo('#sample-route'); }
function validProgress(record) { const step = sample.steps.find(item => item.id === record?.stepId); return step && step.sourceIds.includes(record.sourceId) ? record : null; }
function refresh() { visible=true; preferences.value=getPreferences(); hasSession.value=Boolean(getActiveSession()); savedProgress.value=validProgress(loadFirstValueProgress(scope)); }
function resume() { const record=validProgress(savedProgress.value); if (!record || !visible) return; selectedStepId.value=record.stepId; selectedSourceId.value=record.sourceId; resumed.value=true; notice.value='已回到保存的样例位置。'; void scrollTo('#sample-source'); }
function savePlace() { if (!activeStep.value || !activeSource.value || !visible) return; const result=saveFirstValueProgress({...scope,stepId:selectedStepId.value,sourceId:selectedSourceId.value}); if (result.ok) { savedProgress.value=result.record; resumed.value=true; notice.value='已在本设备保存样例位置，下次打开可接着看。没有写入项目记忆。'; } else notice.value='当前设备未能保存位置，你仍可继续阅读。'; }
function forgetPlace() { if (!visible) return; const result=clearFirstValueProgress(scope); if (result.ok) { savedProgress.value=null; resumed.value=false; notice.value='样例位置已清除。'; } else notice.value='未能清除样例位置，请重试。'; }
function copySourceLink() { if (!activeSource.value || !visible) return; const version=++copyGeneration, url=activeSource.value.url; uni.setClipboardData({data:url,success:()=>{if (visible && version===copyGeneration) notice.value='已复制论文原页链接。';},fail:()=>{if (visible && version===copyGeneration) notice.value='未能复制论文链接，请重试。';}}); }
function showOwnMaterials() { if (!visible) return; preferences.value=getPreferences(); hasSession.value=Boolean(getActiveSession()); ownMaterialsOpen.value=true; void scrollTo('#sample-handoff'); }
function continueWithOwn() { if (!visible || !ownMaterialsOpen.value) return; preferences.value=getPreferences(); hasSession.value=Boolean(getActiveSession()); const next=handoff.value; if (next.kind==='navigate') uni.navigateTo({url:next.route}); else uni.switchTab({url:next.route}); }
function openChat() { if (visible) uni.switchTab({url:'/pages/chat/index'}); }
function hide() { visible=false; copyGeneration++; notice.value=''; }
onShow(refresh); onHide(hide); onUnload(hide);
</script>

<style>
@import '../../components/mobile.css';
</style>
<style scoped>
.first-value-page{max-width:1120px;padding-bottom:48px;color:#203d35}.fv-topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:24px}.fv-topbar .quiet-button{margin:0;padding-left:0}.fv-sample-label{font-size:11px;line-height:1.6;color:#637063}.fv-heading{max-width:800px;margin-bottom:28px}.fv-eyebrow{display:block;color:#677659;font-size:11px;letter-spacing:.8px;margin-bottom:10px}.fv-title{display:block;font-size:36px;font-weight:650;letter-spacing:-1.1px;line-height:1.4}.fv-subtitle{display:block;margin-top:12px;color:#62705f;font-size:14px;line-height:1.8}.fv-workspace{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:24px;align-items:start}.fv-main{min-width:0}.fv-answer{padding:28px;border-radius:23px;background:#203d35;color:#f8faef}.fv-answer-top{display:flex;justify-content:space-between;align-items:center;gap:12px;color:#c8d7bc;font-size:11px}.fv-answer-tag{border:1px solid #ffffff30;border-radius:20px;padding:4px 9px;font-size:10px}.fv-answer-title{display:block;margin-top:22px;font-size:29px;font-weight:650;line-height:1.5;letter-spacing:-.6px}.fv-lime{color:#d6fa75}.fv-answer-copy{display:block;margin-top:12px;font-size:14px;line-height:1.9;color:#e0e8db}.fv-proof-button{display:flex;justify-content:space-between;gap:18px;align-items:center;min-height:48px;margin:22px 0 0;padding:13px 16px;border-radius:12px;background:#d6fa75;color:#203d35;font-size:13px;font-weight:600;line-height:1.6}.fv-answer-note{display:block;margin-top:16px;color:#c2d1bc;font-size:11px;line-height:1.8}.fv-route{margin-top:24px}.fv-section-title{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:14px;font-size:15px;font-weight:600}.fv-small{display:block;color:#687460;font-size:11px;line-height:1.8;font-weight:400}.fv-step{display:flex;width:100%;align-items:flex-start;gap:12px;margin:0 0 9px;padding:16px;text-align:left;line-height:1.6;background:#fffefa;border:1px solid #dfe3d7;border-radius:15px;color:#203d35;transition:background .15s,border-color .15s}.fv-step-active{background:#edf4e0;border-color:#829866}.fv-step-number{flex-shrink:0;display:flex;align-items:center;justify-content:center;width:27px;height:27px;border:1px solid #d2dbc6;border-radius:50%;font-size:10px;color:#5b7150}.fv-step-body{min-width:0;flex:1}.fv-step-title{display:block;font-size:13px;font-weight:650}.fv-step-detail{display:block;margin-top:5px;font-size:12px;color:#60705a;line-height:1.8}.fv-step-arrow{font-size:17px;color:#738467}.fv-route-note{display:block;font-size:10px;line-height:1.8;color:#737e68;margin-top:12px}.fv-source{min-width:0;padding:25px;background:#fffefa;border:1px solid #dce2d2;border-radius:23px;position:sticky;top:24px;scroll-margin-top:20px}.fv-source-status{font-size:10px;font-weight:400;line-height:1.6;color:#677659}.fv-paper-title{display:block;font-size:21px;line-height:1.5;font-family:Georgia,serif;overflow-wrap:anywhere}.fv-paper-author{display:block;color:#6b775f;font-size:11px;line-height:1.8;margin-top:8px}.fv-quote-label{display:block;font-size:10px;color:#617553;margin:22px 0 10px}.fv-quote{display:block;border-left:3px solid #a3bb7e;padding:4px 0 4px 15px;font-family:Georgia,serif;color:#304a31;font-size:19px;line-height:1.75;overflow-wrap:anywhere}.fv-interpretation{margin-top:22px;padding:16px;border-radius:12px;background:#f0f3e8;font-size:13px;line-height:1.9;color:#526846}.fv-interpretation .fv-quote-label{margin:0 0 6px}.fv-paper-link{display:flex;align-items:center;min-height:48px;color:#375a32;font-size:12px;text-decoration:underline;text-underline-offset:4px;margin-top:12px}.fv-return-route{margin:6px 0 0;padding-left:0}.fv-source-tabs{display:flex;gap:7px;margin-bottom:18px}.fv-source-tabs button{min-height:44px;margin:0;padding:10px 15px;background:#f3f4ec;border:1px solid #dfe4d6;border-radius:9px;color:#5b6e4d;font-size:12px;line-height:22px}.fv-source-tabs .fv-source-tab-active{background:#e3eecf;border-color:#9fb880}.fv-empty-visual{display:flex;align-items:center;gap:12px;margin:10px 0 30px;color:#a3b38b}.fv-empty-visual text:nth-child(odd){display:flex;align-items:center;justify-content:center;width:46px;height:46px;border:1px solid #d4dfc4;border-radius:50%;color:#537041;font-size:13px}.fv-source-empty-title{display:block;font-size:21px;font-weight:600;line-height:1.6}.fv-source-empty-copy{display:block;margin:12px 0 24px;font-size:13px;color:#65745b;line-height:1.9}.fv-resume{display:flex;align-items:center;justify-content:space-between;gap:16px;border:1px solid #c6d7aa;background:#eef5e2;padding:17px 20px;border-radius:16px;margin-bottom:24px}.fv-resume-title{display:block;font-size:14px;font-weight:600;margin-bottom:5px}.fv-muted{display:block;font-size:12px;line-height:1.9;color:#65735e}.fv-next{display:flex;justify-content:space-between;gap:24px;margin-top:30px;padding:25px;border:1px solid #dce1d2;border-radius:20px;background:#eeefe5}.fv-next-copy{min-width:0;flex:1}.fv-next-title{display:block;font-size:21px;font-weight:600;line-height:1.5}.fv-next-question{display:block;font-size:13px;line-height:1.9;color:#506447;margin:10px 0}.fv-next-actions{display:flex;flex-direction:column;align-items:stretch;justify-content:center;flex-shrink:0;gap:8px}.fv-notice{display:block;background:#eaf2dc;border-radius:12px;padding:14px;margin-top:12px;color:#3e5834;font-size:12px;line-height:1.8}.fv-own{display:flex;justify-content:space-between;align-items:center;gap:22px;margin-top:28px;padding:12px 0}.fv-own-title{display:block;font-size:22px;line-height:1.5;font-weight:600;margin-bottom:8px}.fv-own .primary-button{flex-shrink:0}.fv-handoff{margin-top:14px;padding:22px;background:#fffefa;border:1px solid #dde4d0;border-radius:18px}.fv-handoff-title{display:block;font-size:16px;font-weight:600;margin-bottom:8px}.fv-handoff .secondary-button{margin:16px 0 10px}.fv-footnote{display:block;color:#737e6b;font-size:10px;line-height:1.9;margin-top:24px;border-top:1px solid #dfe3d7;padding-top:18px}.first-value-page button{min-height:44px}.first-value-page button::after{border:0}.first-value-page button:focus-visible,.fv-paper-link:focus-visible{outline:3px solid #7e963b;outline-offset:3px}.first-value-page button[disabled]{opacity:.5}.first-value-page .secondary-button,.first-value-page .primary-button{font-size:13px;white-space:normal;line-height:1.6}.first-value-page .quiet-button{font-size:12px}
@media(max-width:759px){.first-value-page{padding-bottom:35px}.fv-topbar{align-items:flex-start;margin-bottom:20px}.fv-sample-label{max-width:160px;text-align:right;font-size:10px;padding-top:7px}.fv-title{font-size:28px;letter-spacing:-.6px}.fv-heading{margin-bottom:20px}.fv-subtitle{font-size:12px}.fv-workspace{display:flex;flex-direction:column;gap:20px}.fv-main,.fv-source{width:100%;box-sizing:border-box}.fv-answer{padding:22px;border-radius:20px}.fv-answer-title{font-size:25px;margin-top:18px}.fv-answer-copy{font-size:13px}.fv-answer-note{font-size:10px}.fv-source{position:static;padding:22px}.fv-next{flex-direction:column;gap:16px;padding:21px}.fv-next-actions{flex-direction:row;flex-wrap:wrap;justify-content:flex-start;align-items:center}.fv-next-actions button{margin:0}.fv-own{flex-direction:column;align-items:stretch;gap:16px}.fv-own .primary-button{width:100%;box-sizing:border-box}.fv-own-title{font-size:21px}.fv-resume{flex-wrap:wrap;padding:17px}.fv-resume .secondary-button{margin:0}.fv-empty-visual{margin-bottom:20px}.fv-paper-title{font-size:20px}.fv-quote{font-size:18px}}
@media(prefers-reduced-motion:reduce){.fv-step{transition:none}}
</style>
