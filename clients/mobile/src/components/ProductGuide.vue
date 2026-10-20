<template>
  <view class="product-guide">
    <view class="guide-heading"><view><text class="guide-eyebrow">开始使用</text><text class="guide-title">{{ experience.label }}</text></view><text class="guide-status">{{ experience.status }}</text></view>
    <text class="guide-summary">{{ experience.summary }}</text>
    <view class="guide-actions"><button class="primary-button" role="button" @tap="openProductPage(experience.primary.page)">{{ experience.primary.label }}</button><button v-if="experience.secondary" class="quiet-button" role="button" @tap="openProductPage(experience.secondary.page)">{{ experience.secondary.label }}</button></view>
    <button class="quiet-button guide-toggle" role="button" :aria-expanded="expanded" @tap="expanded = !expanded">{{ expanded ? '收起使用步骤和边界 −' : '查看使用步骤和功能边界 ＋' }}</button>
    <view v-if="expanded" class="guide-details">
      <text class="guide-available">{{ experience.kind === 'private' ? '客户端提供的入口' : '当前体验范围' }}：{{ experience.available.join(' · ') }}</text>
      <view v-for="(step, index) in experience.steps" :key="index" class="guide-step"><text class="guide-step-number">{{ index + 1 }}</text><text>{{ step }}</text></view>
      <text class="guide-boundary">{{ experience.boundary }}</text>
    </view>
  </view>
</template>
<script setup>
import { computed, ref, watch } from 'vue';
import { productExperience, openProductPage } from '../services/product-capabilities.js';
const props = defineProps({ preferences: { type: Object, default: () => ({ mode: 'demo' }) } });
const experience = computed(() => productExperience(props.preferences));
const expanded = ref(false);
watch(() => [experience.value.kind, props.preferences.invitePending], () => { expanded.value = false; });
</script>
<style scoped>
.product-guide { padding:20px; margin:18px 0; border:1px solid #dce4d3; border-radius:20px; background:#eef3e7; }
.guide-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:10px; }
.guide-eyebrow { display:block; color:#75866a; font-size:10px; margin-bottom:5px; }
.guide-title { display:block; color:#203d35; font-size:17px; font-weight:600; line-height:1.5; }
.guide-status { flex-shrink:0; padding:4px 7px; border-radius:6px; background:#fffefa; color:#617455; font-size:10px; line-height:18px; }
.guide-summary { display:block; margin-top:11px; color:#64735c; font-size:12px; line-height:1.9; }
.guide-actions { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:14px; }
.guide-actions button { min-height:44px; padding:10px 13px; margin:0; font-size:12px; line-height:24px; }
.guide-actions .primary-button { color:#fff; background:#203d35; border:1px solid #203d35; border-radius:12px; }
.guide-actions .quiet-button { color:#53694a; background:transparent; border:0; }
.guide-actions button::after,.guide-toggle::after { border:0; }
.guide-toggle { display:block; margin:8px 0 -7px; padding:10px 0; min-height:44px; color:#53694a; background:transparent; border:0; font-size:11px; text-align:left; }
.guide-details { margin-top:12px; padding-top:14px; border-top:1px solid #dce5d3; }
.guide-available { display:block; color:#4e6644; font-size:11px; line-height:1.9; }
.guide-step { display:flex; align-items:flex-start; gap:9px; margin-top:12px; color:#516249; font-size:12px; line-height:1.9; }
.guide-step-number { display:flex; align-items:center; justify-content:center; flex-shrink:0; width:23px; height:23px; border-radius:50%; background:#e0e9d3; color:#536b42; font-size:11px; }
.guide-boundary { display:block; margin-top:14px; color:#6d795f; font-size:11px; line-height:1.9; }
@media (max-width:360px) { .guide-heading { flex-direction:column; } }
</style>
