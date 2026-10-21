<template>
  <!-- #ifdef H5 -->
  <nav v-if="visible && currentRoute" class="desktop-navigation" aria-label="主导航">
    <div class="desktop-brand">
      <img class="desktop-brand-mark" src="/static/original-brand/logo-64.png" alt="DocThinker original dog logo" />
      <div><span class="desktop-wordmark">DocThinker<span class="desktop-period">.</span></span><span class="desktop-brand-caption">私人工作台</span></div>
    </div>
    <span class="desktop-section-label">我的空间</span>
    <div class="desktop-navigation-items">
      <NativeButton v-for="item in desktopNavigationItems" :key="item.id" class="desktop-nav-button" :class="{'is-current':currentRoute === item.path}" :aria-current="currentRoute === item.path ? 'page' : undefined" :disabled="navigating" @click="navigate(item.id)">
        <img :src="currentRoute === item.path ? item.activeIcon : item.icon" alt="" class="desktop-nav-icon" />
        <span class="desktop-nav-copy"><span class="desktop-nav-label">{{ item.label }}</span><span class="desktop-nav-detail">{{ item.detail }}</span></span>
        <span v-if="currentRoute === item.path" class="desktop-current-mark" aria-hidden="true" />
      </NativeButton>
    </div>
    <p v-if="navigationError" class="desktop-navigation-error" role="status" aria-live="polite">{{ navigationError }}</p>
    <div class="desktop-navigation-footer"><span class="desktop-footer-title">给想法一个落脚处。</span><p>从讨论开始，带上资料，再把核对过的信息留下。</p></div>
  </nav>
  <!-- #endif -->
</template>

<script setup>
// #ifdef H5
import { h, ref, onMounted, onUnmounted } from 'vue';
import { onShow, onHide } from '@dcloudio/uni-app';
import { desktopNavigationItems, desktopNavigationRoute, switchDesktopPage } from '../services/adaptive-navigation.js';

// uni-app's H5 <button> is a custom element. A native button preserves browser
// keyboard activation, focus, and disabled semantics without synthetic key events.
const NativeButton = (_, { attrs, slots }) => h('button', { ...attrs, type: 'button' }, slots.default?.());
NativeButton.inheritAttrs = false;
const currentRoute = ref(''), visible = ref(true), navigating = ref(false), navigationError = ref('');
let mounted = false;
function readCurrentRoute() {
  if (!mounted) return;
  const hash = typeof window !== 'undefined' ? window.location.hash : '';
  let route = hash;
  if (!route) {
    try {
      const pages = typeof getCurrentPages === 'function' ? getCurrentPages() : [];
      route = pages[pages.length - 1]?.route || '';
    } catch { route = ''; }
  }
  currentRoute.value = desktopNavigationRoute(route);
  navigationError.value = '';
}
function navigate(id) {
  if (!mounted || !visible.value || navigating.value || !currentRoute.value) return;
  navigationError.value = '';
  try {
    const started = switchDesktopPage(id, currentRoute.value, options => {
      navigating.value = true;
      uni.switchTab(options);
    }, {
      success: readCurrentRoute,
      fail: () => { if (mounted && visible.value) navigationError.value = '没有切换成功，请再试一次。'; },
      complete: () => { if (mounted) navigating.value = false; },
    });
    if (!started) navigating.value = false;
  } catch {
    navigating.value = false;
    navigationError.value = '没有切换成功，请再试一次。';
  }
}
onShow(() => { visible.value = true; readCurrentRoute(); });
onHide(() => { visible.value = false; });
onMounted(() => {
  mounted = true;
  readCurrentRoute();
  window.addEventListener('hashchange', readCurrentRoute);
  window.addEventListener('popstate', readCurrentRoute);
});
onUnmounted(() => {
  mounted = false;
  window.removeEventListener('hashchange', readCurrentRoute);
  window.removeEventListener('popstate', readCurrentRoute);
});
// #endif
</script>

<style scoped>
/* #ifdef H5 */
.desktop-navigation { display:none; }
@supports selector(:has(*)) {
  @media (min-width:1100px) {
    .desktop-navigation { position:fixed; display:flex; flex-direction:column; box-sizing:border-box; z-index:25; inset:0 auto 0 0; width:208px; padding:30px 16px 24px; overflow-y:auto; background:#f0f1e9; border-right:1px solid var(--dt-border,#dde0d7); color:var(--dt-ink,#1c2723); }
    .desktop-brand { display:flex; align-items:center; gap:9px; margin:0 0 38px; padding:0 6px; }
    .desktop-brand-mark { width:34px; height:34px; flex-shrink:0; }
    .desktop-wordmark { display:block; font-size:20px; font-weight:700; line-height:1.3; letter-spacing:-.8px; white-space:nowrap; }
    .desktop-period { color:#637f25; }
    .desktop-brand-caption { display:block; margin-top:5px; color:var(--dt-muted,#626c65); font-size:11px; letter-spacing:1px; }
    .desktop-section-label { margin:0 12px 12px; color:#737d73; font-size:11px; letter-spacing:1.6px; }
    .desktop-navigation-items { display:flex; flex-direction:column; gap:7px; }
    .desktop-nav-button { appearance:none; display:flex; align-items:center; gap:12px; width:100%; min-height:66px; box-sizing:border-box; padding:11px 13px; margin:0; border:1px solid transparent; border-radius:13px; background:transparent; color:inherit; text-align:left; font:inherit; line-height:1.4; cursor:pointer; transition:background .16s ease,border-color .16s ease; }
    .desktop-nav-button:hover:not(:disabled) { background:#e8ebdf; }
    .desktop-nav-button.is-current { background:var(--dt-surface,#fffefa); border-color:#dce1d2; box-shadow:0 2px 5px #23351c05; }
    .desktop-nav-button:focus-visible { outline:3px solid #587633; outline-offset:2px; }
    .desktop-nav-button:disabled { cursor:wait; }
    .desktop-nav-icon { width:23px; height:23px; flex-shrink:0; }
    .desktop-nav-copy { display:block; min-width:0; }
    .desktop-nav-label { display:block; font-size:14px; font-weight:600; }
    .desktop-nav-detail { display:block; margin-top:4px; font-size:10px; font-weight:400; color:var(--dt-muted,#626c65); }
    .desktop-current-mark { margin-left:auto; width:5px; height:5px; flex-shrink:0; border-radius:50%; background:#688442; }
    .desktop-navigation-error { margin:16px 10px; color:#8c382c; font-size:12px; line-height:1.7; }
    .desktop-navigation-footer { margin-top:auto; padding:32px 11px 0; }
    .desktop-footer-title { font-size:12px; font-weight:600; }
    .desktop-navigation-footer p { margin:9px 0 0; font-size:11px; line-height:1.9; color:var(--dt-muted,#626c65); }
  }
}
@media (prefers-reduced-motion:reduce) { .desktop-nav-button { transition:none; } }
/* #endif */
</style>
