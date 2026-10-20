<template>
  <view class="mobile-page settings-page">
    <BrandHeader eyebrow="MAKE IT YOURS" title="你的助手，你的节奏。" subtitle="把连接、回答与数据的控制权，留在你手里。" />
    <view class="settings-overview"><view class="overview-status"><view class="overview-dot" aria-hidden="true"></view><text>{{ savedPreferences.connectionKind === 'invite' ? savedPreferences.invitePending ? '邀请账号 · 待登录' : '邀请文字内测' : savedPreferences.mode === 'demo' ? '本地演示 · 不联网' : '私人服务 · 已保存配置' }}</text></view><text class="overview-note">{{ dirty ? '有修改尚未保存' : '设置不会调用模型' }}</text></view>
    <view class="settings-navigation" role="group" aria-label="设置分区">
      <button class="settings-nav-item" role="button" tabindex="0" :class="{ 'settings-nav-active': activeSection === 'start' }" :aria-pressed="activeSection === 'start'" @tap="activeSection = 'start'" @keydown="activateSection($event, 'start')">开始</button>
      <button class="settings-nav-item" role="button" tabindex="0" :class="{ 'settings-nav-active': activeSection === 'connection' }" :aria-pressed="activeSection === 'connection'" @tap="activeSection = 'connection'" @keydown="activateSection($event, 'connection')">{{ savedPreferences.connectionKind === 'invite' ? '账号' : '连接' }}</button>
      <button class="settings-nav-item" role="button" tabindex="0" :class="{ 'settings-nav-active': activeSection === 'thinking' }" :aria-pressed="activeSection === 'thinking'" @tap="activeSection = 'thinking'" @keydown="activateSection($event, 'thinking')">回答<text v-if="dirty" class="nav-change-dot" aria-label="有未保存修改"></text></button>
      <button class="settings-nav-item" role="button" tabindex="0" :class="{ 'settings-nav-active': activeSection === 'privacy' }" :aria-pressed="activeSection === 'privacy'" @tap="activeSection = 'privacy'" @keydown="activateSection($event, 'privacy')">隐私</button>
    </view>
    <view v-if="activeSection === 'start'" class="settings-start">
      <view class="section-introduction"><text class="settings-eyebrow">01 / GET STARTED</text><text class="section-title">少一点配置，多一点开始。</text><text class="settings-intro">先进入当前可用的体验；想改变回答方式，再来这里调整。</text></view>
      <ProductGuide :preferences="savedPreferences" />
      <view class="settings-next-step"><text class="next-step-title">不是所有开关，都需要打开。</text><text class="next-step-copy">按场景选择，不靠堆上下文。回答设置仅随下次提问生效。</text><button class="quiet-button" role="button" tabindex="0" @tap="activeSection = 'thinking'" @keydown="activateSection($event, 'thinking')">调整回答方式 <text aria-hidden="true">↗</text></button></view>
      <ModeBanner :mode="savedPreferences.mode" :connection-kind="savedPreferences.connectionKind" :invite-pending="savedPreferences.invitePending" :show-guide="false" />
    </view>
    <view v-if="activeSection === 'connection' && savedPreferences.connectionKind === 'invite'" class="settings-card account-card">
      <text class="settings-eyebrow">YOUR ACCOUNT</text><text class="section-title">{{ savedPreferences.inviteUser ? savedPreferences.inviteUser.username : '邀请账号尚未登录' }}</text>
      <text class="privacy-summary">仅连接当前网站，不需要设置 API 地址、共享访问码或模型 Key。刷新后需再次登录；当前内测仅开放对话与历史。</text>
      <button class="secondary-button full-width" @tap="openInvite">{{ savedPreferences.invitePending ? '前往登录' : '管理账号 / 退出' }}</button>
    </view>
    <template v-if="activeSection === 'connection' && savedPreferences.connectionKind !== 'invite'">
    <view class="settings-card">
    <view class="section-heading settings-card-heading"><view><text class="settings-eyebrow">01 / CONNECTION</text><text class="section-title">选择使用方式</text></view><text v-if="dirty" class="unsaved-badge">未保存</text></view>
    <view class="mode-selector"><button class="mode-option" :class="{ 'mode-selected': form.mode === 'demo' }" @tap="form.mode = 'demo'"><view class="mode-option-top"><text class="mode-icon" aria-hidden="true">✳</text><view class="mode-radio" :class="{ 'mode-radio-active': form.mode === 'demo' }"></view></view><text class="mode-option-title">先试一试</text><text class="mode-option-note">演示模式 · 无需连接</text></button><button class="mode-option" :class="{ 'mode-selected': form.mode === 'connected' }" @tap="form.mode = 'connected'"><view class="mode-option-top"><text class="mode-icon" aria-hidden="true">↗</text><view class="mode-radio" :class="{ 'mode-radio-active': form.mode === 'connected' }"></view></view><text class="mode-option-title">连接私人服务</text><text class="mode-option-note">使用你自己的后端</text></button></view>
    <view v-if="form.mode === 'demo'" class="settings-note"><text class="note-heading">先认识它，再带上你的资料。</text>当前使用本地模拟数据。问答、上传、图谱与语音转写均为演示，不代表真实模型效果。</view>
    <view v-else>
      <view class="connection-warning"><text class="warning-title">不是正式多用户服务</text><text>尚未接入微信登录或用户隔离。请只连接你自己的私人测试实例，不要把共用访问码发布给其他用户。</text></view>
      <text class="field-label">API 服务地址</text><input v-model="form.baseUrl" class="text-field connection-input" type="text" :maxlength="500" placeholder="https://your-server.example" :adjust-position="true" /><text class="field-help">填服务域名与端口，不含 /api 路径。真机上的 localhost 指手机自身；微信小程序需配置 HTTPS 合法域名。</text>
      <!-- #ifdef MP-WEIXIN -->
      <text class="field-help">服务域名须加入小程序后台的 request 和 uploadFile 合法域名。不要关闭域名校验来代替真机配置。游客模式不能代替真实 AppID 的联调。</text>
      <!-- #endif -->
      <text class="field-label">私人测试访问码</text><input v-model="form.accessToken" class="text-field connection-input" type="text" password :maxlength="1000" placeholder="输入你私人服务的访问码" /><text class="field-help">访问码仅保留在本次运行的内存中，重启后需重新填写；不是微信账号，也不是模型厂商 API Key。更换服务地址后需重新填写。</text>
      <view class="model-note"><text class="model-note-title">模型由你的后端配置</text><text>问答、识别和朗读分别使用服务端配置。本页面不接收模型 API Key，也不会自动切换厂商。</text></view>
      <view class="connection-check">
        <button class="quiet-button full-width" :disabled="!canCheckConnection" @tap="testConnection">{{ checking ? '正在检查连接…' : '检查连接（不调用模型）' }}</button>
        <text class="field-help">{{ dirty ? '请先保存修改，再检查连接。' : '只检查服务连通与访问码，不读取会话或资料；最多等待 8 秒，不自动重试。' }}</text>
        <view v-if="connectionNotice" class="status-box" role="status" aria-live="polite">{{ connectionNotice }}</view>
        <view v-if="connectionError" class="error-box" role="status" aria-live="polite">{{ connectionError }}</view>
      </view>
    </view>
    </view>
    </template>
    <view v-if="activeSection === 'thinking'" class="settings-card experiment-card">
    <view class="section-heading settings-card-heading"><view><text class="settings-eyebrow">02 / THINKING</text><text class="section-title">回答与对比测试</text></view><text class="experiment-badge">实验控制</text></view>
    <text class="settings-intro">保存后随下一次提问生效，不删除已有资料或记忆。演示模式只记录设置，不运行真实算法。</text>
    <view class="setting-row"><view class="setting-copy"><text class="setting-title">记忆</text><text class="setting-description">{{ savedPreferences.connectionKind === 'invite' ? '此开关不控制研究项目的确认记忆。若研究模式已开放，请在助手页选择本次是否使用确认记忆。' : '允许回答参考已保存的记忆；研究模式可在助手页按次切换。' }}</text></view><switch :disabled="savedPreferences.connectionKind === 'invite'" :checked="form.useMemory" color="#203D35" aria-label="记忆" @change="form.useMemory = $event.detail.value" /></view>
    <view class="setting-row"><view class="setting-copy"><text class="setting-title">对话上下文</text><text class="setting-description">带上前面的对话，方便连续追问。</text></view><switch :checked="form.useContext" color="#203D35" aria-label="对话上下文" @change="form.useContext = $event.detail.value" /></view>
    <view class="setting-row"><view class="setting-copy"><text class="setting-title">LLM 缓存</text><text class="setting-description">{{ savedPreferences.connectionKind === 'invite' ? '当前聊天内测未启用应用侧回答缓存。' : '复用已有结果，对比测试时可关闭。' }}</text></view><switch :disabled="savedPreferences.connectionKind === 'invite'" :checked="form.useCache" color="#203D35" aria-label="LLM 缓存" @change="form.useCache = $event.detail.value" /></view>
    <view class="setting-row setting-row-last"><view class="setting-copy"><text class="setting-title">自进化结果 <text class="caution-tag">需核验</text></text><text class="setting-description">{{ savedPreferences.connectionKind === 'invite' ? '邀请内测尚未开放自进化检索。' : '允许使用补充信息，不等于这些信息已经得到证明。' }}</text></view><switch :disabled="savedPreferences.connectionKind === 'invite'" :checked="form.useEvolution" color="#203D35" aria-label="自进化结果" @change="form.useEvolution = $event.detail.value" /></view>
    </view>
    <view v-if="error" class="error-box settings-feedback" role="status" aria-live="polite">{{ error }}</view>
    <view v-if="notice" class="status-box settings-feedback" role="status" aria-live="polite">{{ notice }}</view>
    <view v-if="dirty || activeSection === 'thinking' || (activeSection === 'connection' && savedPreferences.connectionKind !== 'invite')" class="settings-save-area">
      <view class="save-area-heading"><text class="save-area-label">{{ dirty ? '改动还在草稿里' : '保持你的选择' }}</text><text class="save-area-note">{{ dirty ? '切换分区不会丢失' : '保存不会调用模型' }}</text></view>
      <button class="primary-button full-width save-settings" :disabled="saving || !dirty" @tap="save">{{ saving ? '正在保存…' : dirty ? '保存我的设置' : '设置已保存' }}</button>
      <text class="settings-footer">保存不会自动测试连接，也不会调用模型。{{ savedPreferences.connectionKind === 'invite' ? '账号退出或切换，请使用账号分区。' : '可在连接分区单独检查服务，不需要先读取会话。' }}</text>
    </view>
    <button v-if="activeSection === 'connection' && savedPreferences.connectionKind !== 'invite'" class="quiet-button full-width return-demo" :disabled="saving" @tap="returnToDemo">回到演示并清除本次访问码</button>
    <view v-if="activeSection === 'privacy'" class="settings-card privacy-card">
      <text class="settings-eyebrow">03 / YOUR DATA</text><text class="section-title">先说清楚，才放心使用</text>
      <text class="privacy-summary">{{ savedPreferences.connectionKind === 'invite' ? '运营者与云模型服务商可能处理对话内容，不是端到端加密。请勿提交敏感资料。' : '演示不联网，也不使用你的文件或麦克风。连接私人服务后，资料与对话会按该后端配置处理。' }}</text>
      <button class="quiet-button privacy-toggle" :aria-expanded="showDataDetails" @tap="showDataDetails = !showDataDetails">{{ showDataDetails ? '收起数据说明 −' : '查看记忆、删除与外发说明 ＋' }}</button>
      <view v-if="showDataDetails" class="privacy-details">
        <text>关闭记忆不等于无痕。实验开关不会删除聊天历史，也不是禁止所有后台整理的总开关。</text>
        <text>开放研究模式后，记忆页可手动保存当前项目的确认内容；“确认”不代表真实性已核验。旧模式仅展示当前会话的长期记忆摘要，可能理解有误；还不是已验证的跨会话个人画像。</text>
        <text>当前删除会话不等于彻底删除独立记忆和历史修订。测试阶段请勿上传敏感个人资料。</text>
        <text>使用云模型或云端 embedding 时，相关内容会发送给你配置的服务商。私人部署不等于完全离线。</text>
      </view>
    </view>
    <!-- #ifdef MP-WEIXIN -->
    <view v-if="activeSection === 'privacy'" class="settings-card wechat-test-card"><text class="settings-eyebrow">WECHAT / DEVELOPMENT</text><text class="section-title">微信小程序测试版</text><text class="privacy-summary">无 AppID 时，可导入开发者工具用游客模式体验模拟数据。手机扫码预览、体验版及真实隐私授权，需要你自己的小程序 AppID 和平台配置。</text><text class="privacy-summary">已准备：助手对话、资料、会话记忆、图谱和设置。尚未完成：微信登录、客户级隔离及长期任务自动续跑。</text></view>
    <!-- #endif -->
    <view v-if="activeSection === 'privacy'" class="settings-install">
      <!-- #ifdef H5 -->
      <InstallGuide :invitation="savedPreferences.connectionKind === 'invite'" />
      <!-- #endif -->
    </view>
  </view>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { onHide, onShow, onUnload } from '@dcloudio/uni-app';
import { api, cancelConnectionCheck, getPreferences, savePreferences, setActiveSession } from '../../services/api.js';
import ModeBanner from '../../components/ModeBanner.vue';
import BrandHeader from '../../components/BrandHeader.vue';
import ProductGuide from '../../components/ProductGuide.vue';
// #ifdef H5
import InstallGuide from '../../components/InstallGuide.vue';
// #endif

const defaults = { mode: 'demo', baseUrl: '', accessToken: '', useMemory: true, useContext: true, useCache: true, useEvolution: false };
const form = ref({ ...defaults }), savedPreferences = ref({ ...defaults });
const saving = ref(false), error = ref(''), notice = ref(''), showDataDetails = ref(false);
const activeSection = ref('start');
const checking = ref(false), connectionNotice = ref(''), connectionError = ref('');
const visible = ref(false); let checkGeneration = 0;
function activateSection(event, section) {
  if (!['start', 'connection', 'thinking', 'privacy'].includes(section)) return;
  if (!event || !['Enter', ' '].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey || event.repeat) return;
  event.preventDefault?.(); activeSection.value = section;
}
const dirty = computed(() => Object.keys(defaults).some(key => form.value[key] !== savedPreferences.value[key]));
const canCheckConnection = computed(() => visible.value && !saving.value && !checking.value && !dirty.value
  && savedPreferences.value.connectionKind !== 'invite'
  && form.value.mode === 'connected' && savedPreferences.value.mode === 'connected'
  && Boolean(savedPreferences.value.baseUrl && savedPreferences.value.accessToken));
function invalidateConnectionCheck() {
  checkGeneration += 1; cancelConnectionCheck(); checking.value = false;
  connectionNotice.value = ''; connectionError.value = '';
}
watch(() => Object.keys(defaults).map(key => form.value[key]), invalidateConnectionCheck, {flush:'sync'});
function readSettings() { visible.value = true; invalidateConnectionCheck(); savedPreferences.value = { ...defaults, ...getPreferences() }; form.value = { ...savedPreferences.value }; error.value = ''; notice.value = ''; }
async function testConnection() {
  if (!canCheckConnection.value) return;
  const generation = ++checkGeneration;
  const active = () => visible.value && generation === checkGeneration && !dirty.value && !saving.value;
  checking.value = true; connectionNotice.value = ''; connectionError.value = '';
  const labels = {
    configuration:'请先保存完整的私人服务地址和访问码。', busy:'已有连接检查正在进行，请稍候再试。',
    unauthorized:'访问码验证未通过，请核对访问码后重新保存。', unavailable:'未找到移动服务接口，请确认私人后端已启用移动服务。',
    'not-configured':'移动服务尚未配置好，请检查私人后端配置。',
    timeout:'8 秒内未收到有效响应，请检查网络、HTTPS 和微信合法域名配置。没有自动重试。',
    network:'无法连接私人服务，请检查网络、HTTPS 和微信合法域名配置。没有自动重试。',
    contract:'服务有响应，但不符合本客户端的私人测试接口协议；请确认地址和后端版本。',
    http:'服务暂时无法完成连接检查，请稍后手动重试。',
  };
  try {
    const result = await api.checkConnection({isCurrent:active});
    if (!active() || result?.cancelled) return;
    if (result?.connected === true) connectionNotice.value = '服务连通，访问码验证通过。仅完成服务接口检查；不代表模型、语音、微信真机联调或多用户隔离已就绪。';
    else connectionError.value = labels[result?.reason] || labels.http;
  } catch { if (active()) connectionError.value = labels.network; }
  finally { if (generation === checkGeneration) checking.value = false; }
}
function validate(value) {
  if (savedPreferences.value.connectionKind === 'invite') return '';
  if (value.mode !== 'connected') return '';
  if (!/^https?:\/\/[^\s/?#]+\/?$/i.test(value.baseUrl)) return '请填写完整的服务地址，例如 https://your-server.example，不含 /api 路径、查询参数或空格。';
  if (/^https?:\/\/[^/]*@/i.test(value.baseUrl)) return '服务地址不能包含账号或密码，请使用下方的私人测试访问码。';
  if (!value.accessToken) return '连接模式需要填写私人测试访问码。';
  return '';
}
async function save() {
  if (saving.value) return;
  invalidateConnectionCheck();
  const next = { ...form.value, baseUrl: form.value.baseUrl.trim().replace(/\/$/, ''), accessToken: form.value.accessToken.trim() };
  error.value = validate(next); notice.value = ''; if (error.value) return;
  saving.value = true;
  try {
    const changedService = next.mode !== savedPreferences.value.mode || next.baseUrl !== savedPreferences.value.baseUrl || next.accessToken !== savedPreferences.value.accessToken;
    await savePreferences(next);
    if (changedService) setActiveSession('');
    savedPreferences.value = { ...defaults, ...getPreferences() }; form.value = { ...savedPreferences.value };
    notice.value = savedPreferences.value.connectionKind === 'invite' ? '回答设置已保存，随下次提问生效。没有调用模型。' : next.mode === 'demo' ? '已保存，当前为演示模式，不会调用真实服务。' : savedPreferences.value.accessToken ? '已保存私人测试配置。可手动检查连接，不会调用模型。' : '服务地址已保存。为避免将旧访问码发到新地址，请重新填写访问码并保存。';
  } catch (issue) { error.value = issue?.message || '设置保存失败，请重试。'; }
  finally { saving.value = false; }
}
async function returnToDemo() { form.value = { ...form.value, mode: 'demo', baseUrl: '', accessToken: '' }; await save(); }
function openInvite() { uni.reLaunch({url:'/pages/invite/index'}); }
onShow(readSettings);
function hide() { visible.value = false; invalidateConnectionCheck(); }
onHide(hide); onUnload(hide);
</script>

<style>
@import '../../components/mobile.css';
</style>
<style scoped>
.settings-page { background:#f6f5f0; color:#1c2723; padding-bottom:38px; }
.settings-page :deep(.brand-heading) { margin-top:34px; }
.settings-page :deep(.brand-title) { max-width:400px; color:#1c2723; font-size:36px; letter-spacing:-1.4px; line-height:1.25; }
.settings-page :deep(.brand-eyebrow) { color:#59635c; font-size:10px; letter-spacing:2px; }
.settings-page :deep(.brand-subtitle) { max-width:360px; font-size:13px; color:#647168; }
.settings-overview { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:4px 0 20px; }
.overview-status { display:flex; align-items:center; gap:7px; color:#344b3f; font-size:11px; font-weight:600; }
.overview-dot { flex-shrink:0; width:6px; height:6px; border-radius:50%; background:#536d45; }
.overview-note { color:#687369; font-size:10px; text-align:right; }
.settings-navigation { display:flex; padding:5px; gap:3px; border:1px solid #dedfd5; border-radius:16px; background:#e9ebe3; margin-bottom:26px; }
.settings-nav-item { display:flex; align-items:center; justify-content:center; gap:5px; flex:1; min-width:0; min-height:44px; padding:10px 6px; margin:0; border-radius:11px; border:1px solid transparent; color:#5a655d; background:transparent; font-size:13px; font-weight:600; line-height:22px; transition:background .16s ease, color .16s ease; }
.settings-nav-item.settings-nav-active { color:#1c2723; background:#fdfdf7; border-color:#ddded3; box-shadow:0 1px 3px #1c272308; }
.nav-change-dot { width:5px; height:5px; border-radius:50%; background:#967238; }
.settings-eyebrow { display:block; margin-bottom:8px; color:#647263; font-size:10px; letter-spacing:1.5px; font-weight:600; }
.section-introduction { margin-bottom:20px; }
.section-title { display:block; color:#1c2723; font-size:20px; line-height:1.4; letter-spacing:-.4px; font-weight:650; overflow-wrap:anywhere; }
.settings-intro { display:block; margin-top:10px; color:#626e64; font-size:12px; line-height:1.85; }
.settings-card { padding:25px; margin:18px 0; border:1px solid #dedfd5; border-radius:23px; background:#fdfdf9; }
.settings-card-heading { margin:0 0 22px; align-items:flex-start; }
.unsaved-badge, .experiment-badge { flex-shrink:0; padding:5px 8px; border:1px solid #dce2ce; border-radius:7px; color:#4f6545; background:#eef3e3; font-size:10px; line-height:16px; }
.unsaved-badge { color:#816338; background:#faf3e1; border-color:#ebdec5; }
.settings-start :deep(.product-guide) { position:relative; padding:25px; margin:0 0 21px; border:1px solid #1c2723; border-radius:23px; background:#1c2723; color:#f3f5ea; }
.settings-start :deep(.guide-eyebrow) { color:#d6fa75; letter-spacing:1.2px; font-size:10px; }
.settings-start :deep(.guide-title) { color:#fafcf2; font-size:23px; line-height:1.4; letter-spacing:-.4px; }
.settings-start :deep(.guide-status) { color:#d6fa75; background:#30412e; border:1px solid #495a38; border-radius:7px; }
.settings-start :deep(.guide-summary) { color:#c7d0c2; font-size:12px; }
.settings-start :deep(.guide-actions .primary-button) { color:#1c2723; background:#d6fa75; border-color:#d6fa75; padding:12px 18px; border-radius:12px; font-size:13px; }
.settings-start :deep(.guide-actions .quiet-button), .settings-start :deep(.guide-toggle) { color:#d8e1cc; }
.settings-start :deep(.guide-details) { border-color:#475341; }
.settings-start :deep(.guide-available), .settings-start :deep(.guide-step), .settings-start :deep(.guide-boundary) { color:#c8d2bf; }
.settings-start :deep(.guide-step-number) { background:#384834; color:#d6fa75; }
.settings-next-step { border-bottom:1px solid #dedfd5; padding:0 2px 16px; margin-bottom:19px; }
.next-step-title { display:block; color:#263d30; font-size:14px; font-weight:600; }
.next-step-copy { display:block; color:#667264; font-size:12px; line-height:1.8; margin-top:6px; }
.settings-next-step .quiet-button { justify-content:flex-start; gap:10px; padding-left:0; color:#37563e; margin-top:6px; }
.settings-start :deep(.mode-banner) { margin-bottom:0; border-color:#dfe2d4; background:#edefe5; }
.mode-selector { display:flex; gap:10px; }
.mode-option { display:block; flex:1; min-width:0; min-height:143px; margin:0; padding:17px; border:1px solid #dfe2d5; border-radius:16px; text-align:left; background:#f2f3eb; line-height:1.6; transition:background .16s ease, border-color .16s ease; }
.mode-option.mode-selected { border-color:#6c8050; background:#ecf6d3; }
.mode-option-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:17px; }
.mode-icon { color:#38523c; font-size:27px; line-height:1; }
.mode-radio { width:17px; height:17px; border:1px solid #a7b294; border-radius:50%; }
.mode-radio.mode-radio-active { border:5px solid #415d37; background:#d6fa75; }
.mode-option-title { display:block; color:#253b2d; font-size:14px; font-weight:650; }
.mode-option-note { display:block; color:#5f7054; font-size:11px; margin-top:6px; }
.settings-note { margin-top:19px; color:#65735d; font-size:12px; line-height:1.9; }
.note-heading { display:block; color:#344e35; font-size:13px; font-weight:600; margin-bottom:5px; }
.connection-warning { margin-top:20px; padding:16px; background:#faf4e9; border:1px solid #e7dac1; border-radius:13px; color:#806c4b; font-size:12px; line-height:1.85; }
.warning-title { display:block; color:#755b33; font-size:12px; font-weight:650; margin-bottom:5px; }
.field-label { color:#344836; font-size:13px; }
.connection-input { min-height:50px; border-color:#dce0d1; background:#f4f6ed; border-radius:12px; color:#253c2c; font-size:13px; }
.field-help { display:block; margin:9px 2px 0; color:#687261; font-size:12px; line-height:1.85; }
.model-note { margin-top:19px; padding:15px; border-radius:12px; color:#647459; background:#eef2e5; font-size:12px; line-height:1.85; }
.model-note-title { display:block; margin-bottom:5px; color:#415d3a; font-size:12px; font-weight:600; }
.connection-check { margin-top:18px; padding-top:14px; border-top:1px solid #dfe4d5; }
.connection-check button { min-height:48px; border:1px solid #d7dfcc; color:#3e5a36; background:#eef2e5; border-radius:12px; }
.connection-check .status-box, .connection-check .error-box { margin-top:12px; font-size:12px; line-height:1.8; }
.account-card .secondary-button { margin-top:19px; color:#1c2723; background:#d6fa75; border-color:#c6e65f; }
.experiment-card .settings-intro { margin-top:0; padding-bottom:6px; }
.setting-row { display:flex; align-items:center; gap:14px; padding:22px 0; border-bottom:1px solid #e2e6d8; }
.setting-row.setting-row-last { padding-bottom:1px; border-bottom:0; }
.setting-copy { flex:1; min-width:0; }
.setting-title { display:block; color:#2a4230; font-size:15px; font-weight:600; }
.setting-description { display:block; margin-top:6px; color:#63755b; font-size:12px; line-height:1.8; }
.setting-row switch { flex-shrink:0; min-height:44px; display:flex; align-items:center; }
.caution-tag { margin-left:5px; padding:2px 5px; color:#806238; background:#faf1dc; border-radius:5px; font-size:10px; font-weight:500; }
.settings-feedback { margin-top:18px; margin-bottom:0; }
.settings-save-area { border:1px solid #e0e2d7; border-radius:20px; padding:18px; background:#edf0e5; margin-top:22px; }
.save-area-heading { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }
.save-area-label { color:#334c35; font-size:12px; font-weight:600; }
.save-area-note { color:#6c7960; font-size:10px; }
.save-settings { min-height:50px; margin:0; border-radius:12px; background:#1c2723; border-color:#1c2723; color:#faffef; }
.save-settings[disabled] { opacity:1; color:#687660; background:#e0e5d5; border-color:#d6ddca; }
.settings-footer { display:block; color:#68765e; font-size:11px; line-height:1.85; text-align:left; margin:11px 2px 0; }
.return-demo { margin-top:13px; color:#647459; }
.privacy-summary { display:block; margin-top:12px; color:#61715b; font-size:12px; line-height:1.9; }
.privacy-toggle { margin-top:10px; min-height:46px; justify-content:flex-start; padding-left:0; color:#395b36; text-align:left; line-height:1.8; }
.privacy-details { border-top:1px solid #dfe4d5; padding-top:6px; }
.privacy-details text { display:block; margin-top:13px; color:#62725c; font-size:12px; line-height:1.9; }
.settings-install :deep(.install-guide) { padding:22px; border:1px solid #dedfd5; border-radius:20px; background:#fdfdf9; }
@media (max-width:360px) { .settings-page :deep(.brand-title) { font-size:32px; } .settings-card, .settings-start :deep(.product-guide) { padding:20px; } .mode-option { padding:13px; } .mode-option-title { font-size:13px; } .mode-option-note { font-size:10px; } .overview-note { max-width:90px; } .experiment-badge { font-size:9px; } }
@media (prefers-reduced-motion:reduce) { .settings-nav-item, .mode-option { transition:none; } }
</style>
