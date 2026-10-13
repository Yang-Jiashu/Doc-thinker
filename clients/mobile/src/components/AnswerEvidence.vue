<template>
  <view v-if="evidence.items.length" class="answer-evidence">
    <button class="evidence-toggle" :aria-expanded="expanded" @tap="expanded = !expanded">
      <text class="evidence-toggle-icon">{{ expanded ? '−' : '＋' }}</text>
      <text>{{ expanded ? '收起依据' : '查看 ' + evidence.items.length + ' 条依据' }}</text>
      <text v-if="demo" class="evidence-demo-label">演示</text>
    </button>
    <view v-if="expanded" class="evidence-panel">
      <text class="evidence-note">{{ demo ? '以下为演示片段，不是真实文档证据。' : evidence.items.some(item => item.label) ? '本次送入模型的原文节选。引用标签只标出处，不代表结论已核验；可与回答逐条比对。' : '服务返回片段，尚未逐条核验，可能包含模型整理内容。' }}</text>
      <view v-for="(item, index) in evidence.items" :key="index" class="evidence-item">
        <view class="evidence-heading"><text class="evidence-number">{{ item.label || index + 1 }}</text><text class="evidence-title">{{ item.title }}{{ item.page ? ' · 第 ' + item.page + ' 页' : '' }}</text></view>
        <!-- #ifdef MP-WEIXIN -->
        <text class="evidence-excerpt" user-select>{{ item.content }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="evidence-excerpt" selectable>{{ item.content }}</text>
        <!-- #endif -->
        <text v-if="item.truncated" class="evidence-truncated">片段已截短</text>
      </view>
      <text v-if="evidence.limited" class="evidence-note evidence-limit">仅展示部分片段，最多 {{ MAX_EVIDENCE_ITEMS }} 条</text>
    </view>
  </view>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { MAX_EVIDENCE_ITEMS, normalizeAnswerEvidence } from '../services/answer-evidence.js';

const props = defineProps({ sources: { type: Array, default: () => [] }, demo: { type: Boolean, default: false } });
const expanded = ref(false);
const evidence = computed(() => normalizeAnswerEvidence(props.sources));
watch(() => [props.sources, props.demo], () => { expanded.value = false; });
</script>

<style scoped>
.answer-evidence { margin-top:10px; }
.evidence-toggle { display:flex; align-items:center; gap:7px; width:auto; min-height:44px; margin:0; padding:8px 10px; border:1px solid #dfe5d8; border-radius:12px; background:#f1f3e9; color:#4b6448; font-size:12px; line-height:1.5; text-align:left; }
.evidence-toggle::after { border:0; }
.evidence-toggle-icon { font-size:16px; color:#6d8459; }
.evidence-demo-label { padding:1px 5px; border-radius:5px; background:#e0e6d4; font-size:10px; }
.evidence-panel { margin-top:8px; padding:12px; border:1px solid #e1e5da; border-radius:14px; background:#fafbf5; }
.evidence-note { display:block; color:#78816f; font-size:11px; line-height:1.7; }
.evidence-item { padding:13px 0 4px; }
.evidence-heading { display:flex; align-items:flex-start; gap:7px; margin-bottom:7px; }
.evidence-number { flex-shrink:0; display:block; min-width:18px; height:18px; border-radius:5px; background:#e8eddf; color:#62794f; font-size:10px; line-height:18px; text-align:center; }
.evidence-title { min-width:0; color:#40543c; font-size:12px; font-weight:600; line-height:18px; overflow-wrap:anywhere; }
.evidence-excerpt { display:block; color:#5a6852; font-size:12px; line-height:1.8; white-space:pre-wrap; overflow-wrap:anywhere; }
.evidence-truncated { display:block; margin-top:5px; color:#878c7d; font-size:10px; }
.evidence-limit { margin-top:10px; }
</style>
