<template>
  <view v-if="!invitation" class="graph-voice">
    <view class="graph-voice-actions">
      <view class="graph-voice-button" :class="{ 'is-recording': phase === 'recording', 'is-disabled': unavailable }" role="button" :tabindex="unavailable ? -1 : 0" :aria-disabled="unavailable" @click="toggle" @keydown="onKeydown($event, 'toggle')">{{ buttonLabel }}</view>
      <view v-if="busy" class="graph-voice-cancel" role="button" tabindex="0" @click="cancel" @keydown="onKeydown($event, 'cancel')">取消语音</view>
    </view>
    <view class="graph-voice-notice" role="status" aria-live="polite">
      <text v-if="busy" class="graph-voice-stage">{{ stageLabel }}</text>
      <text>{{ notice || '说出要找的节点或联系，转成文字后先检查，再点定位。' }}</text>
    </view>
  </view>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { getPreferences } from '../services/api.js';
import { startRecording, stopRecording, cancelRecording } from '../services/voice.js';

const props = defineProps({ active: { type: Boolean, default: false }, disabled: { type: Boolean, default: false }, resetKey: { type: String, default: '' } });
const emit = defineEmits(['transcript', 'busy']);
const phase = ref('idle'), notice = ref(''), demo = ref(false);
let generation = 0, ownsRecording = false, mounted = true;
const busy = computed(() => phase.value !== 'idle');
const invitation = computed(() => { props.active; props.resetKey; return preferences().connectionKind === 'invite'; });
const unavailable = computed(() => !props.active || props.disabled || phase.value === 'preparing' || phase.value === 'transcribing' || invitation.value);
const buttonLabel = computed(() => ({ preparing: '正在准备语音…', recording: demo.value ? '结束演示并转文字' : '停止并转文字', ready: '将这段录音转文字', transcribing: '正在转文字…' }[phase.value] || '说出要找的内容'));
const stageLabel = computed(() => ({ preparing: '等待语音准备', recording: demo.value ? '演示中 · 未使用麦克风' : '正在录音 · 最长 60 秒', ready: '录音已停止', transcribing: '正在转写' }[phase.value] || ''));

function preferences() { try { return getPreferences() || {}; } catch { return {}; } }
function setPhase(value) {
  const wasBusy = busy.value;
  phase.value = value;
  if (wasBusy !== busy.value) emit('busy', busy.value);
}
function current(value) { return mounted && value === generation && props.active && !props.disabled && preferences().connectionKind !== 'invite'; }
function invalidate(message = '') {
  generation += 1;
  // The shared voice service also serves chat. An inactive graph must not
  // cancel another page's recording just because its own props changed.
  if (ownsRecording) { ownsRecording = false; cancelRecording(); }
  setPhase('idle');
  notice.value = message;
}
function cancel() {
  if (!busy.value) return;
  const transcribing = phase.value === 'transcribing';
  invalidate(transcribing
    ? '已取消接收转写，没有执行定位；已经提交服务端的转写处理可能仍会完成。'
    : '已取消语音，没有修改定位草稿，也没有执行定位。');
}
function failed(value) {
  if (!current(value)) return;
  invalidate('语音暂时不可用，请检查麦克风权限或语音服务后重试，也可以直接输入。');
}
async function toggle() {
  if (!mounted || unavailable.value || preferences().connectionKind === 'invite') return;
  if (phase.value === 'recording' || phase.value === 'ready') {
    const operation = generation;
    setPhase('transcribing');
    notice.value = '只转为可编辑的定位草稿，不会自动查找或提问。';
    try {
      const transcript = await stopRecording();
      if (!current(operation)) return;
      ownsRecording = false;
      setPhase('idle');
      if (typeof transcript !== 'string' || !transcript.trim()) { notice.value = '没有识别到文字，请重试或直接输入。'; return; }
      notice.value = demo.value
        ? '这是演示定位文字，未开麦；请检查或确认替换后再点定位。'
        : '文字已交给定位草稿，请检查或确认替换后再点定位。';
      // Demo shows a graph-specific example instead of the shared chat example.
      // Real ASR preserves its entire result, including whitespace/over-limit text.
      // The parent owns draft reconciliation, validation, and explicit search.
      emit('transcript', demo.value ? '可再生能源投资和能源安全有什么联系？' : transcript);
    } catch { failed(operation); }
    return;
  }
  const operation = ++generation;
  demo.value = preferences().mode === 'demo';
  ownsRecording = true;
  setPhase('preparing');
  notice.value = demo.value ? '这是演示转写，不会开启麦克风或上传音频。' : '允许录音后说出想找的内容；文字由你检查后再定位。';
  try {
    await startRecording({ onState(event) {
      if (!current(operation)) return;
      if (event?.state === 'recording' && phase.value === 'preparing') setPhase('recording');
      if (event?.state === 'stopped' && phase.value !== 'transcribing') {
        setPhase('ready');
        notice.value = '录音已停止，尚未上传转写。点“将这段录音转文字”继续，或取消。';
      }
      if (event?.state === 'error') failed(operation);
    } });
    if (!current(operation)) return;
    if (phase.value === 'preparing') setPhase('recording');
  } catch (error) {
    // This exact service preflight rejection occurs before this component
    // acquires recording resources. Do not cancel the existing owner.
    if (current(operation) && error?.message === '上一段录音尚未结束') ownsRecording = false;
    failed(operation);
  }
}
function onKeydown(event, action) {
  if (!event || event.defaultPrevented || event.repeat || event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
  if (event.key !== 'Enter' && event.key !== ' ' && event.key !== 'Spacebar') return;
  if (action !== 'toggle' && action !== 'cancel') return;
  event.preventDefault?.();
  if (action === 'cancel') cancel(); else toggle();
}
watch(() => [props.active, props.disabled, props.resetKey], ([active, disabled, key], previous) => {
  if (!active || disabled || (previous && key !== previous[2])) invalidate();
}, { flush: 'sync' });
onBeforeUnmount(() => { mounted = false; invalidate(); });
</script>

<style scoped>
.graph-voice { display: flex; flex-direction: column; gap: 8px; }
.graph-voice-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.graph-voice-button, .graph-voice-cancel { min-height: 44px; padding: 0 14px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border-radius: 12px; font-size: 13px; cursor: pointer; }
.graph-voice-button { color: #304b50; background: #edf4f2; border: 1px solid #dce8e5; }
.graph-voice-button.is-recording { color: #8b4334; background: #fff2ec; border-color: #ecd4c9; }
.graph-voice-button.is-disabled { opacity: .55; cursor: default; }
.graph-voice-cancel { color: #596a70; border: 1px solid #dfe6e7; background: #fff; }
.graph-voice-button:focus-visible, .graph-voice-cancel:focus-visible { outline: 2px solid #307369; outline-offset: 3px; }
.graph-voice-notice { display: flex; flex-direction: column; gap: 4px; font-size: 12px; line-height: 1.6; color: #697b82; }
.graph-voice-stage { font-weight: 650; color: #304b50; }
</style>
