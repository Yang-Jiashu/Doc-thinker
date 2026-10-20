<template>
  <view v-if="answerId" class="answer-feedback">
    <view class="feedback-actions">
      <text class="feedback-label">这次回答</text>
      <button role="button" class="feedback-choice" :class="{ selected: rating === 'helpful' }" :aria-pressed="rating === 'helpful'" :disabled="busy || disabled" @tap="choose('helpful')">有帮助</button>
      <button role="button" class="feedback-choice" :class="{ selected: rating === 'inaccurate' }" :aria-pressed="rating === 'inaccurate'" :disabled="busy || disabled" @tap="choose('inaccurate')">有问题</button>
      <text v-if="demo" class="feedback-demo">演示</text>
    </view>
    <view v-if="editing" class="feedback-editor">
      <text class="feedback-explanation">{{ demo ? '仅记录本次演示，不联网，也不保存到真实账号。' : '反馈保存到当前账号的后台，供后续人工核查；不会自动训练模型或修改记忆。' }}</text>
      <textarea v-model="note" class="feedback-note" :maxlength="-1" :disabled="busy || disabled" :auto-height="true" aria-label="回答反馈说明（选填）" placeholder="选填：哪里有帮助，或哪里需要改正？请勿填写敏感信息。" :show-confirm-bar="false" />
      <text class="feedback-count" :class="{ 'over-limit': noteLength > 500 }">{{ noteLength }} / 500 字符</text>
      <view class="feedback-editor-actions">
        <button role="button" class="feedback-save" :disabled="!canSave" @tap="save">{{ busy ? '正在保存…' : '保存反馈' }}</button>
        <button role="button" class="feedback-choice" :disabled="busy" @tap="cancel">取消</button>
      </view>
    </view>
    <text v-if="notice" class="feedback-notice" role="status" aria-live="polite">{{ notice }}</text>
    <text v-if="error" class="feedback-error" role="status" aria-live="polite">{{ error }}</text>
  </view>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { onHide, onUnload } from '@dcloudio/uni-app';
import { api } from '../services/api.js';

const props = defineProps({ sessionId: {type:String, required:true}, answerId: {type:String, default:''},
  initial: {type:Object, default:null}, demo: {type:Boolean, default:false}, disabled: {type:Boolean, default:false} });
const emit = defineEmits(['busy']);
const rating = ref(''), note = ref(''), saved = ref(null), editing = ref(false), busy = ref(false), error = ref(''), notice = ref('');
let generation = 0;
const noteLength = computed(() => Array.from(note.value).length);
const canSave = computed(() => Boolean(props.sessionId && props.answerId && ['helpful','inaccurate'].includes(rating.value)
  && noteLength.value <= 500 && !busy.value && !props.disabled));
function reset() {
  generation += 1; busy.value = false; editing.value = false; error.value = ''; notice.value = '';
  const initial = props.initial;
  saved.value = initial && initial.answer_id === props.answerId && ['helpful','inaccurate'].includes(initial.rating)
    && typeof initial.note === 'string' && Array.from(initial.note).length <= 500
    ? {answer_id:initial.answer_id,rating:initial.rating,note:initial.note} : null;
  rating.value = saved.value?.rating || ''; note.value = saved.value?.note || '';
  if (saved.value) notice.value = props.demo ? '演示反馈已记录' : '反馈已保存，可修改';
}
function choose(value) {
  if (busy.value || props.disabled || !['helpful','inaccurate'].includes(value)) return;
  rating.value = value; editing.value = true; error.value = ''; notice.value = '';
}
function cancel() {
  generation += 1; busy.value = false; editing.value = false; error.value = '';
  rating.value = saved.value?.rating || ''; note.value = saved.value?.note || '';
  notice.value = saved.value ? (props.demo ? '演示反馈已记录' : '反馈已保存，可修改') : '';
}
async function save() {
  if (!canSave.value) return;
  const request = ++generation;
  const session = props.sessionId, answer = props.answerId, values = {rating:rating.value,note:note.value};
  busy.value = true; emit('busy',{answerId:answer,active:true}); error.value = ''; notice.value = '';
  try {
    const result = await api.submitFeedback(session,answer,values);
    if (request !== generation || session !== props.sessionId || answer !== props.answerId) return;
    const feedback = result?.feedback;
    if (!feedback || feedback.answer_id !== answer || feedback.rating !== values.rating || feedback.note !== values.note) throw new Error('反馈保存未得到确认，请稍后手动重试');
    saved.value = {answer_id:answer,...values}; editing.value = false;
    notice.value = props.demo ? '演示反馈已记录，没有联网' : '反馈已保存，不会自动改写答案或记忆';
  } catch (issue) {
    if (request === generation) error.value = issue?.mobileSafe ? issue.message : '反馈未确认保存，说明已保留；可稍后手动重试';
  } finally { if (request === generation) busy.value = false; emit('busy',{answerId:answer,active:false}); }
}
watch(() => [props.sessionId,props.answerId,props.initial,props.demo],reset,{immediate:true});
onHide(cancel); onUnload(cancel);
</script>

<style scoped>
.answer-feedback { margin-top:12px; }
.feedback-actions { display:flex; align-items:center; flex-wrap:wrap; gap:6px; }
.feedback-label { color:#7c8576; font-size:11px; margin-right:4px; }
.feedback-choice { min-height:44px; margin:0; padding:8px 10px; border:1px solid transparent; border-radius:10px; background:transparent; color:#61725c; font-size:12px; line-height:1.7; }
.feedback-choice.selected { background:#eaf0e1; border-color:#cbd9be; color:#354f32; }
.feedback-choice::after,.feedback-save::after { border:0; }
.feedback-demo { color:#7e896f; font-size:10px; }
.feedback-editor { margin-top:6px; padding:12px; border:1px solid #dfe5d9; border-radius:14px; background:#fafbf6; }
.feedback-explanation,.feedback-notice,.feedback-error { display:block; color:#71816a; font-size:11px; line-height:1.8; }
.feedback-note { width:100%; min-height:64px; margin-top:10px; padding:10px; border:1px solid #dce2d6; border-radius:10px; background:#fff; font-size:12px; line-height:1.8; }
.feedback-count { display:block; margin-top:5px; text-align:right; font-size:10px; color:#7b8674; }
.feedback-count.over-limit,.feedback-error { color:#9a513f; }
.feedback-editor-actions { display:flex; gap:8px; margin-top:8px; }
.feedback-save { min-height:44px; margin:0; padding:9px 13px; border-radius:10px; background:#203d35; color:#fff; font-size:12px; line-height:1.7; }
.feedback-notice,.feedback-error { margin-top:5px; }
</style>
