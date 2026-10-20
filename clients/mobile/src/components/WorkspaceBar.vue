<template>
  <view class="workspace-bar">
    <picker class="workspace-picker" :range="labels" :value="currentIndex" :disabled="disabled || !sessions.length" @change="changeSession">
      <view class="workspace-selection">
        <text class="workspace-caption">当前会话</text>
        <view class="workspace-name"><text class="workspace-title">{{ selectedLabel }}</text><text class="workspace-chevron">⌄</text></view>
      </view>
    </picker>
    <button v-if="allowCreate" class="workspace-create" :class="{ 'workspace-create-disabled': disabled }" :disabled="disabled" @tap="$emit('create')">＋ 新建</button>
  </view>
</template>

<script setup>
import { computed } from 'vue';
const props = defineProps({
  sessions: { type: Array, default: () => [] }, sessionId: { type: String, default: '' },
  disabled: { type: Boolean, default: false }, allowCreate: { type: Boolean, default: false },
});
const emit = defineEmits(['change', 'create']);
const labels = computed(() => props.sessions.map(session => String(session.title || session.name || session.id || '未命名会话')));
const currentIndex = computed(() => Math.max(0, props.sessions.findIndex(session => String(session.id) === props.sessionId)));
const selectedLabel = computed(() => props.sessions.length ? labels.value[currentIndex.value] : '还没有会话');
function changeSession(event) {
  const session = props.sessions[Number(event.detail.value)];
  if (session) emit('change', String(session.id));
}
</script>

<style scoped>
.workspace-bar { display:flex; align-items:center; gap:12px; margin:0 0 18px; padding:12px 14px; border:1px solid #e1e5d9; background:#fffefa; border-radius:18px; }
.workspace-picker { flex: 1; min-width: 0; }
.workspace-selection { min-height:44px; padding:0; }
.workspace-caption { display:block; color:#7d8874; font-size:10px; line-height:18px; }
.workspace-name { display: flex; align-items: center; gap: 12rpx; }
.workspace-title { min-width:0; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:#2e4539; font-size:14px; line-height:26px; font-weight:600; }
.workspace-chevron { flex-shrink: 0; color: #777780; }
.workspace-create { flex-shrink:0; display:flex; align-items:center; min-height:44px; margin:0; padding:0 14px; border:1px solid #e0e5d4; border-radius:12px; background:#f2f4e8; color:#365034; font-size:12px; line-height:1.3; }
.workspace-create::after { border: 0; }
.workspace-create-disabled { opacity: .45; }
</style>
