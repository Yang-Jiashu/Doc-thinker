<template>
  <view class="brand-header" :style="layout.header">
    <!-- #ifdef H5 -->
    <DesktopNavigation />
    <!-- #endif -->
    <view class="brand-topline" :style="layout.row">
      <view class="brand-signature"><view class="brand-mark-frame"><image class="brand-mark" src="/static/original-brand/logo-64.png" mode="aspectFit" /></view><text class="brand-wordmark">DocThinker<text class="brand-period">.</text></text></view>
      <view class="brand-category"><view class="brand-category-mark" /><text>你的私人助手</text></view>
    </view>
    <view v-if="title" class="brand-heading"><text v-if="eyebrow" class="brand-eyebrow">{{ eyebrow }}</text><text class="brand-title">{{ title }}</text><text v-if="subtitle" class="brand-subtitle">{{ subtitle }}</text></view>
  </view>
</template>
<script setup>
import { ref, onMounted } from 'vue';
import { wechatNavigationLayout } from '../services/navigation-layout.js';
// #ifdef H5
import DesktopNavigation from './DesktopNavigation.vue';
// #endif
defineProps({ eyebrow:{type:String,default:''}, title:{type:String,default:''}, subtitle:{type:String,default:''} });
const layout = ref({ header: {}, row: {} });
// #ifdef MP-WEIXIN
layout.value = wechatNavigationLayout();
onMounted(() => {
  try {
    const info = typeof wx.getWindowInfo === 'function' ? wx.getWindowInfo() : {};
    const capsule = typeof wx.getMenuButtonBoundingClientRect === 'function' ? wx.getMenuButtonBoundingClientRect() : {};
    layout.value = wechatNavigationLayout(info, capsule);
  } catch { /* Conservative layout remains usable on older clients. */ }
});
// #endif
</script>
<style scoped>
.brand-header { margin-bottom:28px; color:var(--dt-ink,#1c2723); }
.brand-topline { display:flex; align-items:center; justify-content:space-between; min-height:48px; gap:12px; }
.brand-signature { display:flex; align-items:center; min-width:0; gap:10px; }
.brand-mark-frame { display:flex; align-items:center; justify-content:center; width:39px; height:39px; flex-shrink:0; border:1px solid var(--dt-border,#dde0d7); border-radius:12px; background:var(--dt-surface,#fffefa); }
.brand-mark { width:29px; height:29px; flex-shrink:0; }
.brand-wordmark { color:var(--dt-ink,#1c2723); font-size:22px; font-weight:700; letter-spacing:-.9px; white-space:nowrap; }
.brand-period { color:#637f25; }
.brand-category { display:flex; align-items:center; flex-shrink:0; gap:8px; color:var(--dt-muted,#626c65); font-size:12px; line-height:1.5; }
.brand-category-mark { width:7px; height:7px; border-radius:50%; background:var(--dt-accent,#d6fa75); border:1px solid #a9c965; }
.brand-heading { margin-top:36px; }
.brand-eyebrow { display:block; color:var(--dt-muted,#626c65); font-size:11px; letter-spacing:1.8px; font-weight:600; margin-bottom:12px; }
.brand-title { display:block; font-size:34px; line-height:1.25; font-weight:650; letter-spacing:-1.1px; color:var(--dt-ink,#1c2723); overflow-wrap:anywhere; }
.brand-subtitle { display:block; margin-top:14px; max-width:600px; color:var(--dt-muted,#626c65); font-size:14px; line-height:1.85; }
@media (min-width:760px) {
  .brand-header { margin-bottom:34px; }
  .brand-topline { padding-bottom:22px; border-bottom:1px solid var(--dt-border,#dde0d7); }
  .brand-wordmark { font-size:24px; }
  .brand-category { font-size:13px; }
  .brand-heading { margin-top:44px; }
  .brand-title { font-size:42px; letter-spacing:-1.5px; }
  .brand-subtitle { font-size:15px; }
}
@media (max-width:360px) {
  .brand-signature { gap:7px; }
  .brand-mark-frame { width:34px; height:34px; border-radius:10px; }
  .brand-mark { width:26px; height:26px; }
  .brand-wordmark { font-size:20px; }
  .brand-category { gap:6px; font-size:11px; }
  .brand-title { font-size:30px; }
}
/* #ifdef H5 */
@supports selector(:has(*)) {
  @media (min-width:1100px) {
    .brand-header:has(.desktop-navigation) .brand-topline { display:none; }
    .brand-header:has(.desktop-navigation):not(:has(.brand-heading)) { margin-bottom:0; }
    .brand-header:has(.desktop-navigation) .brand-heading { margin-top:0; }
  }
}
/* #endif */
/* #ifdef MP-WEIXIN */
.brand-category { display:none; }
.brand-signature { min-width:0; gap:7px; }
.brand-mark-frame { width:32px; height:32px; border-radius:10px; }
.brand-mark { width:25px; height:25px; }
.brand-wordmark { white-space:nowrap; font-size:18px; }
/* #endif */
</style>
