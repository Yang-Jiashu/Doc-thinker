<template>
  <view class="capability-boundary">
    <text class="boundary-state">{{ feature.name }} · 邀请内测暂未开放</text>
    <text class="boundary-title">{{ feature.title }}</text>
    <text class="boundary-description">{{ feature.description }}</text>
    <view class="boundary-available"><text class="boundary-available-label">现在可以用</text><text>文字对话 · 会话历史 · 连续追问</text></view>
    <view class="boundary-actions">
      <button class="primary-button" role="button" @tap="openProductPage(invitePending ? 'invite' : 'chat')">{{ invitePending ? '先登录账号' : '回到助手' }}</button>
      <button v-if="!invitePending" class="quiet-button" role="button" @tap="openProductPage('settings')">查看使用范围</button>
    </view>
    <text class="boundary-footnote">这是产品开放范围，不是你的资料丢失或上传失败。</text>
  </view>
</template>
<script setup>
import { computed } from 'vue';
import { featureBoundary, openProductPage } from '../services/product-capabilities.js';
const props = defineProps({ featureId: { type: String, required: true }, invitePending: { type: Boolean, default: false } });
const feature = computed(() => featureBoundary(props.featureId));
</script>
<style scoped>
.capability-boundary { padding:24px; border:1px solid #dfe5d8; border-radius:22px; background:#fffefa; }
.boundary-state { display:block; color:#718069; font-size:11px; }
.boundary-title { display:block; margin-top:14px; color:#203d35; font-size:22px; font-weight:600; line-height:1.5; }
.boundary-description { display:block; margin-top:12px; color:#64715f; font-size:13px; line-height:1.9; }
.boundary-available { display:flex; flex-direction:column; gap:7px; margin-top:22px; padding:15px; border-radius:12px; background:#eef2e7; color:#42583d; font-size:12px; line-height:1.8; }
.boundary-available-label { font-weight:600; }
.boundary-actions { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:20px; }
.boundary-actions button { min-height:44px; margin:0; padding:10px 16px; font-size:13px; line-height:24px; }
.boundary-actions .primary-button { color:#fff; background:#203d35; border:1px solid #203d35; border-radius:12px; }
.boundary-actions .quiet-button { color:#53694a; background:transparent; border:0; }
.boundary-actions button::after { border:0; }
.boundary-footnote { display:block; margin-top:15px; color:#77826f; font-size:11px; line-height:1.8; }
</style>
