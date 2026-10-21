<template>
  <view class="mobile-page invite-page">
    <BrandHeader eyebrow="BY INVITATION" title="留一个安静的地方，认真想问题。" subtitle="DocThinker · 邀请内测" />
    <!-- #ifdef H5 -->
    <view class="invite-card">
      <template v-if="identity">
        <text class="invite-kicker">你的工作空间</text><text class="invite-title">你好，{{ identity.username }}</text>
        <text class="invite-copy">已登录当前网站。登录令牌只在本次网页内存中，刷新或重新打开后需再次登录。</text>
        <button role="button" class="primary-button full-width" :disabled="busy" @tap="openWorkspace">进入助手</button>
        <button role="button" class="quiet-button full-width" :disabled="busy" @tap="signOut">{{ busy ? '正在退出…' : '退出账号' }}</button>
      </template>
      <template v-else>
        <view class="invite-tabs"><button role="button" class="invite-tab" :class="{'invite-tab-active': !register}" :disabled="busy" @tap="setRegister(false)">登录</button><button role="button" class="invite-tab" :class="{'invite-tab-active': register}" :disabled="busy" @tap="setRegister(true)">首次受邀使用</button></view>
        <text class="invite-title">{{ register ? '用邀请码开启你的空间' : '欢迎回来' }}</text>
        <text class="invite-copy">{{ register ? '邀请码与邀请人为你预留的用户名配对，只能由受邀者注册。' : '使用受邀账号登录，无需设置 API 地址或模型 Key。' }}</text>
        <text class="field-label">受邀用户名</text><input v-model="username" aria-label="受邀用户名" class="text-field" :disabled="busy" :maxlength="32" placeholder="邀请人为你预留的用户名" autocomplete="username" />
        <text class="field-label">{{ register ? '设置密码' : '密码' }}</text><input v-model="password" :aria-label="register ? '设置密码' : '密码'" class="text-field" password :disabled="busy" :maxlength="256" placeholder="12–128 个字符" :autocomplete="register ? 'new-password' : 'current-password'" />
        <template v-if="register"><text class="field-label">邀请码</text><input v-model="inviteCode" aria-label="邀请码" class="text-field" password :disabled="busy" :maxlength="128" placeholder="粘贴邀请人单独发给你的邀请码" autocomplete="off" /></template>
        <text class="invite-hint">邀请码和密码只在你点击下方按钮后发送到当前网站；不写入网址或应用本地存储。</text>
        <button role="button" class="primary-button full-width" :disabled="busy || !available" @tap="submit">{{ busy ? '正在处理…' : register ? '注册并进入' : '登录并进入' }}</button>
        <button v-if="!available && !checking" role="button" class="quiet-button full-width" :disabled="busy" @tap="checkService">重新检查邀请入口</button>
        <text v-if="checking" class="invite-hint" role="status">正在检查当前网站的邀请服务，不调用模型…</text>
      </template>
      <view v-if="error" class="error-box" role="status" aria-live="polite">{{ error }}</view>
    </view>
    <view class="invite-privacy"><text class="invite-privacy-title">使用前，先说清楚</text><text>这是邀请内测，不是微信账号登录。运营者及配置的云模型服务商可能处理你发送的对话；内容不是端到端加密，请勿提交敏感信息。</text><text>默认开放对话和历史，实际能力以登录后的服务状态为准。运营者开放研究模式后，可上传资料、确认项目记忆，并查看记忆观测与来源／事件图。原完整知识图谱、语音与自动学习在邀请中尚未开放。不会自动调用麦克风，关闭记忆也不等于无痕。</text></view>
    <button role="button" class="quiet-button full-width invite-demo" :disabled="busy" @tap="viewDemo">退出邀请模式，查看离线演示</button>
    <!-- #endif -->
    <!-- #ifndef H5 -->
    <view class="status-box">邀请账号目前仅支持网页。请在浏览器打开邀请人提供的 HTTPS 网页；当前小程序与 App 仍使用演示或私人测试连接。</view>
    <!-- #endif -->
  </view>
</template>
<script setup>
import { ref } from 'vue';
import { onShow, onHide, onUnload } from '@dcloudio/uni-app';
import BrandHeader from '../../components/BrandHeader.vue';
// #ifdef H5
import { authenticateInvitation, cancelInviteRequest, checkInvitationService, enterInvitation, getInviteIdentity, logoutInvitation, reloadInvitation, validateInvitationForm } from '../../services/invite-auth.js';
import { cancelRecording, stopSpeaking } from '../../services/voice.js';
// #endif
const username = ref(''), password = ref(''), inviteCode = ref(''), register = ref(false), identity = ref(null);
const busy = ref(false), checking = ref(false), available = ref(false), error = ref('');
let visible = false, generation = 0;
// #ifdef H5
function setRegister(value) { if (busy.value) return; register.value = value; password.value = ''; inviteCode.value = ''; error.value = ''; }
async function checkService() {
  if (checking.value || busy.value) return;
  const current = ++generation; checking.value = true; available.value = false; error.value = '';
  try { enterInvitation(); await checkInvitationService(); if (visible && current === generation) available.value = true; }
  catch (issue) { if (visible && current === generation) error.value = issue.message; }
  finally { if (current === generation) checking.value = false; }
}
async function submit() {
  if (busy.value || !available.value) return;
  const fields = {username:username.value,password:password.value,invite_code:inviteCode.value};
  error.value = validateInvitationForm(fields,register.value); if (error.value) return;
  const current = ++generation; busy.value = true;
  try {
    await authenticateInvitation(fields,{register:register.value});
    if (!visible || current !== generation) return;
    identity.value = getInviteIdentity(); password.value = ''; inviteCode.value = ''; openWorkspace();
  } catch (issue) { if (visible && current === generation) error.value = issue.message; }
  finally { if (current === generation) {busy.value = false; password.value = ''; inviteCode.value = '';} }
}
function openWorkspace() { uni.reLaunch({url:'/pages/chat/index'}); }
async function signOut() {
  if (busy.value) return; busy.value = true; identity.value = null; password.value = ''; inviteCode.value = ''; username.value = '';
  cancelRecording(); stopSpeaking();
  await logoutInvitation(); reloadInvitation();
}
async function viewDemo() {
  if (busy.value) return; busy.value = true; cancelRecording(); stopSpeaking();
  await logoutInvitation({leave:true}); reloadInvitation({leave:true});
}
onShow(() => { visible = true; identity.value = getInviteIdentity(); if (!identity.value) checkService(); });
function hide() { visible = false; generation += 1; cancelInviteRequest(); password.value = ''; inviteCode.value = ''; username.value = ''; busy.value = false; checking.value = false; }
onHide(hide); onUnload(hide);
// #endif
</script>
<style>
@import '../../components/mobile.css';
.invite-page { padding-top:max(36px,env(safe-area-inset-top)); padding-bottom:40px; }
.invite-card { padding:24px; border:1px solid #e1e7d9; border-radius:24px; background:#fffefa; }
.invite-tabs { display:flex; padding:4px; margin-bottom:26px; gap:4px; background:#eef1e7; border-radius:12px; }
.invite-tab { flex:1; padding:10px 4px; min-height:44px; font-size:13px; line-height:24px; background:transparent; color:#65715e; }
.invite-tab-active { background:#fffefa; color:#203d35; font-weight:600; box-shadow:0 1px 6px #203d3510; }
.invite-title { display:block; font-size:23px; font-weight:650; letter-spacing:-.6px; color:#203d35; line-height:1.4; }
.invite-kicker { display:block; font-size:12px; margin-bottom:12px; color:#697d56; }
.invite-copy { display:block; color:#63725b; font-size:13px; line-height:1.85; margin-top:12px; margin-bottom:24px; }
.invite-hint { display:block; color:#68775f; font-size:12px; line-height:1.8; margin:16px 0; }
.invite-card .text-field { min-height:48px; }
.invite-card .primary-button { margin-top:22px; min-height:50px; }
.invite-privacy { padding:24px 8px 4px; font-size:12px; color:#67775d; line-height:1.9; }
.invite-privacy text { display:block; margin-bottom:12px; }
.invite-privacy-title { color:#3d5439; font-weight:600; font-size:14px; }
.invite-demo { font-size:12px; color:#6e7d61; }
</style>
