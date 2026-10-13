<template>
  <view class="install-guide">
    <view class="install-heading"><text class="install-title">添加到手机桌面</text><text class="install-badge">网页 App</text></view>
    <text class="install-copy">不用下载 APK，也无需 Apple 开发者账号。添加后从主屏幕打开；这不是应用商店里的原生安装包。</text>
    <view v-if="state.installed" class="install-status">当前已在独立窗口中运行。</view>
    <button v-else-if="state.canPrompt" class="install-button" :disabled="busy" @tap="install">{{ busy ? '等待系统确认…' : '添加 DocThinker' }}</button>
    <view class="install-steps"><text class="install-platform">安卓 · Chrome</text><text>浏览器右上角菜单 → 安装应用或添加到主屏幕 → 按系统提示确认。</text></view>
    <view class="install-steps"><text class="install-platform">苹果 · Safari</text><text>分享菜单 → 添加到主屏幕 → 如有“作为网页 App 打开”，将它开启 → 添加。微信内置浏览器请先用 Safari 打开。</text></view>
    <text v-if="!state.production" class="install-note">当前是开发预览；离线安装需先构建并打开生产预览或正式 HTTPS 地址。</text>
    <text v-else-if="!state.secure" class="install-note">此地址不满足安全环境要求。手机请使用可信 HTTPS 地址；电脑的 localhost 不能直接当作手机地址。</text>
    <text v-else-if="state.offlineReady" class="install-note">{{ invitation ? '离线界面已准备；邀请账号重新打开后仍须联网登录。不会缓存你的对话，也不会自动切成演示回答。' : '离线界面已准备。关闭本页面后从桌面重开，可使用内置演示；真实问答、上传和语音仍需要联网。' }}</text>
    <text v-else class="install-note">首次安装需联网准备界面。是否显示安装按钮由浏览器决定，也可按上方菜单操作。</text>
    <text v-if="state.updateAvailable" class="install-note">新版已就绪。请先保存或复制草稿，再关闭所有 DocThinker 窗口并重开；不会自动刷新当前页面。</text>
    <text v-if="state.error || message" class="install-note">{{ state.error || message }}</text>
  </view>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue';
defineProps({invitation:{type:Boolean,default:false}});
const state = ref({production:false,secure:false,installed:false,canPrompt:false,offlineReady:false,updateAvailable:false,error:''});
const busy = ref(false), message = ref('');
let installer, unsubscribe;
onMounted(() => {
  installer = window.DocThinkerInstall;
  if (installer) unsubscribe = installer.subscribe(value => { state.value = value; });
});
onUnmounted(() => unsubscribe?.());
async function install() {
  if (!installer || busy.value) return;
  busy.value = true;
  try {
    const outcome = await installer.prompt();
    message.value = outcome === 'accepted' ? '请按系统提示完成添加。' : outcome === 'dismissed' ? '已取消，可稍后从浏览器菜单添加。' : '请使用上方浏览器菜单添加。';
  } finally { busy.value = false; }
}
</script>

<style scoped>
.install-guide { margin-top: 32rpx; padding: 28rpx; border: 1px solid #e4e4e7; border-radius: 20rpx; background: #fafafa; color: #27272b; }
.install-heading { display: flex; gap: 16rpx; align-items: center; margin-bottom: 14rpx; }
.install-title { font-size: 30rpx; font-weight: 600; }.install-badge { font-size: 21rpx; color: #767680; border: 1px solid #dfdfe4; border-radius: 8rpx; padding: 1rpx 10rpx; }
.install-copy, .install-note { display: block; color: #73737d; font-size: 24rpx; line-height: 1.75; }.install-note { margin-top: 20rpx; }
.install-steps { margin-top: 24rpx; font-size: 25rpx; line-height: 1.75; }.install-platform { display: block; margin-bottom: 6rpx; font-weight: 600; }
.install-button { margin-top: 24rpx; padding: 10rpx 20rpx; border: 0; background: #252529; color: #fff; border-radius: 12rpx; font-size: 26rpx; }.install-button::after { border: 0; }
.install-status { margin-top: 20rpx; color: #42624c; font-size: 25rpx; }
</style>
