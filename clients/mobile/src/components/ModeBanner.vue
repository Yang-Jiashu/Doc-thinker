<template>
  <view class="mode-banner" :class="{ 'mode-banner-connected': mode === 'connected', 'mode-banner-compact':compact }">
    <view class="mode-label"><view class="mode-dot" /><text>{{ connectionKind === 'invite' ? (invitePending ? '请先登录' : '邀请内测') : mode === 'connected' ? '私人测试' : 'DEMO · 演示' }}</text></view>
    <view class="mode-content"><text class="mode-description">{{ compact ? (mode !== 'connected' ? '模拟回答，不调用真实模型' : invitePending ? '登录后才能使用' : researchEnabled ? '项目资料 + 确认记忆' : '能力以当前服务为准') : mode === 'connected' && researchEnabled && !invitePending ? '研究模式已开放 · 使用当前项目原文与手动确认记忆，不自动学习或核验结论。' : connectionKind === 'invite' ? '默认开放对话与历史；资料、确认记忆和来源／事件图以登录后的能力提示为准。原完整知识图谱、自动学习与语音尚未开放。' : mode === 'connected' ? '已选连接模式 · 后端能力需验证，仅供自己的私人实例使用' : '示例资料与模拟回答 · 不调用真实模型或麦克风' }}</text><button v-if="connectionKind === 'invite' && invitePending" class="mode-login" role="button" @tap="openInvite">前往邀请登录</button><button v-else-if="showGuide" class="mode-guide" role="button" @tap="openSettings">{{ compact ? '设置 ↗' : '使用指南 ↗' }}</button></view>
  </view>
</template>

<script setup>
defineProps({ mode: { type: String, default: 'demo' }, compact:{type:Boolean,default:false}, connectionKind:{type:String,default:''}, invitePending:{type:Boolean,default:false}, showGuide:{type:Boolean,default:true}, researchEnabled:{type:Boolean,default:false} });
function openInvite() { uni.reLaunch({url:'/pages/invite/index'}); }
function openSettings() { uni.switchTab({url:'/pages/settings/index'}); }
</script>

<style scoped>
.mode-banner { display:flex; align-items:flex-start; gap:10px; padding:10px 12px; border:1px solid #e0e5d6; background:#ecefe3; border-radius:12px; margin-bottom:18px; }
.mode-label { display:flex; align-items:center; gap:5px; flex-shrink:0; color:#415538; font-size:10px; font-weight:700; line-height:18px; }
.mode-dot { width:5px; height:5px; border-radius:50%; background:#7d944d; }
.mode-description { color:#58664f; font-size:11px; line-height:18px; }
.mode-content { flex:1; min-width:0; }
.mode-banner-connected { background:#e8eeea; }
.mode-login { min-height:44px; margin:6px 0 0; padding:8px 12px; color:#203d35; background:#fffefa; font-size:12px; line-height:24px; }
.mode-guide { display:block; min-height:44px; margin:0 0 -8px; padding:7px 0; color:#4c653e; background:transparent; font-size:11px; line-height:30px; text-align:left; }
.mode-banner-compact{align-items:center;min-height:44px;padding:0 2px;border:0;border-radius:0;background:transparent!important;margin-bottom:8px}.mode-banner-compact .mode-content{display:flex;align-items:center;gap:8px}.mode-banner-compact .mode-description{flex:1;font-size:11px}.mode-banner-compact .mode-guide{flex-shrink:0;margin:0;padding:7px 0;min-height:44px;white-space:nowrap}.mode-banner-compact .mode-login{margin:0;white-space:nowrap}
</style>
