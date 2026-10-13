<template>
  <view v-if="record.available" class="answer-deliverable">
    <button class="record-toggle" role="button" :disabled="disabled || busy" :aria-expanded="expanded" @tap="toggle">
      <text aria-hidden="true">{{ expanded ? '−' : '↗' }}</text><text>{{ expanded ? '收起回答记录' : '带走这份回答' }}</text>
    </button>
    <view v-if="expanded" class="record-panel">
      <text class="record-title">回答、问题与返回依据，一起保留。</text>
      <text class="record-note">只复制当前可见回答与片段，不导出其他会话、不上传或创建公开链接。复制后内容在系统剪贴板中；请核对后再分享。</text>
      <button class="record-copy" role="button" :disabled="disabled || busy" @tap="copy">{{ busy ? '正在复制…' : '复制回答与依据' }}</button>
      <text v-if="notice" class="record-status" role="status" aria-live="polite">{{ notice }}</text>
      <text v-if="error" class="record-error" role="status" aria-live="polite">{{ error }}</text>
      <button class="record-preview-toggle" role="button" :disabled="busy" :aria-expanded="preview" @tap="preview = !preview">{{ preview ? '收起可选文字 −' : '预览可选文字 ＋' }}</button>
      <view v-if="preview" class="record-preview">
        <!-- #ifdef MP-WEIXIN -->
        <text class="record-text" user-select>{{ record.text }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="record-text" selectable>{{ record.text }}</text>
        <!-- #endif -->
      </view>
    </view>
  </view>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { onHide, onUnload } from '@dcloudio/uni-app';
import { answerDeliverable } from '../services/answer-deliverable.js';

const props = defineProps({ answer:{type:String,default:''}, question:{type:String,default:''},
  sources:{type:Array,default:()=>[]}, demo:{type:Boolean,default:false}, disabled:{type:Boolean,default:false},
  researchCore:{type:Boolean,default:false}, citationValidation:{type:Object,default:()=>({})},
  answerMode:{type:String,default:''}, retrievalGaps:{type:Array,default:()=>[]}, reasoning:{type:Object,default:null},claimAudit:{type:Object,default:null},memoryAsOf:{type:Number,default:null},knowledgeSkill:{type:Object,default:null} });
const expanded = ref(false), preview = ref(false), busy = ref(false), notice = ref(''), error = ref('');
const record = computed(() => answerDeliverable(props));
let generation = 0;
function clear() { generation += 1; expanded.value = false; preview.value = false; busy.value = false; notice.value = ''; error.value = ''; }
watch(() => [props.answer, props.question, props.sources, props.demo, props.researchCore,
  props.citationValidation, props.answerMode, props.retrievalGaps,props.reasoning,props.claimAudit,props.memoryAsOf,props.knowledgeSkill], clear, {deep:true});
function toggle() { if (props.disabled || busy.value) return; expanded.value = !expanded.value; preview.value = false; notice.value = ''; error.value = ''; }
async function copy() {
  if (!expanded.value || !record.value.available || props.disabled || busy.value) return;
  const attempt = ++generation;
  const text = record.value.text;
  busy.value = true; notice.value = ''; error.value = '';
  try {
    await new Promise((resolve, reject) => {
      if (typeof uni.setClipboardData !== 'function') return reject(new Error('Clipboard unavailable'));
      let done = false;
      const finish = (callback) => { if (done) return; done = true; clearTimeout(timer); callback(); };
      const timer = setTimeout(() => finish(reject),8000);
      try { uni.setClipboardData({data:text,showToast:false,success:()=>finish(resolve),fail:()=>finish(reject)}); }
      catch { finish(reject); }
    });
    if (attempt === generation) notice.value = props.demo ? '演示记录已复制，没有联网。' : '回答与返回片段已复制；这不是事实核验或公开发布。';
  } catch {
    if (attempt === generation) { error.value = '未能确认复制，请展开可选文字手动复制。没有自动重试。'; preview.value = true; }
  } finally { if (attempt === generation) busy.value = false; }
}
onHide(clear); onUnload(clear);
</script>

<style scoped>
.answer-deliverable { margin-top:10px; }
.record-toggle, .record-preview-toggle { display:flex; align-items:center; gap:7px; min-height:44px; margin:0; padding:8px 10px; border:1px solid #dfe5d8; border-radius:12px; background:#f1f3e9; color:#415b49; font-size:12px; line-height:1.6; text-align:left; }
.record-toggle::after, .record-copy::after, .record-preview-toggle::after { border:0; }
.record-panel { margin-top:8px; padding:14px; border:1px solid #dce4d7; border-radius:16px; background:#fafbf6; }
.record-title { display:block; color:#304d3f; font-size:13px; font-weight:600; line-height:1.8; }
.record-note { display:block; margin-top:6px; color:#6c7768; font-size:11px; line-height:1.9; }
.record-copy { display:block; min-height:44px; margin:12px 0 0; padding:10px 14px; border:0; border-radius:11px; background:#203d35; color:#fff; font-size:12px; line-height:1.8; }
.record-copy[disabled] { opacity:.5; }
.record-status, .record-error { display:block; margin-top:9px; color:#536d4e; font-size:11px; line-height:1.9; }
.record-error { color:#9a6b39; }
.record-preview-toggle { margin-top:9px; border:0; background:transparent; padding-left:0; color:#687960; }
.record-preview { max-height:300px; overflow:auto; margin-top:8px; padding:12px; border:1px solid #e2e7dc; border-radius:10px; background:#fff; }
.record-text { display:block; white-space:pre-wrap; overflow-wrap:anywhere; color:#53634d; font-size:11px; line-height:1.9; }
</style>
