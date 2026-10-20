<template>
  <view class="answer-body">
    <text v-if="body.reason" class="answer-format-note">长回答按原文显示，内容完整保留</text>
    <view v-for="(block, index) in body.blocks" :key="index" class="answer-block" :class="['answer-' + block.type, block.level ? 'answer-heading-' + block.level : '']" :role="block.type === 'heading' ? 'heading' : undefined" :aria-level="block.type === 'heading' ? block.level : undefined">
      <template v-if="block.type === 'plain'">
        <!-- #ifdef MP-WEIXIN -->
        <text class="answer-text" user-select>{{ block.text }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="answer-text" selectable>{{ block.text }}</text>
        <!-- #endif -->
      </template>
      <template v-else-if="block.type === 'code'">
        <view class="answer-code-toolbar">
          <text class="answer-code-language">{{ block.language || '代码' }}</text>
          <button class="answer-code-copy" @tap="copyCode(block.text, index)">{{ copiedIndex === index ? '已复制' : '复制代码' }}</button>
        </view>
        <scroll-view class="answer-code-scroll" scroll-x>
          <!-- #ifdef MP-WEIXIN -->
          <text class="answer-code-text" user-select>{{ block.text }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="answer-code-text" selectable>{{ block.text }}</text>
          <!-- #endif -->
        </scroll-view>
        <text v-if="copyErrorIndex === index" class="answer-copy-error" role="status">复制失败，可以长按选择代码</text>
      </template>
      <view v-else-if="block.type === 'list'" role="list">
        <view v-for="(item, itemIndex) in block.items" :key="itemIndex" class="answer-list-row" role="listitem" :style="{ paddingLeft: item.indent * 5 + 'px' }">
          <text class="answer-list-marker">{{ item.marker }}</text>
          <!-- #ifdef MP-WEIXIN -->
          <text class="answer-text answer-list-content" user-select><text v-for="(token, tokenIndex) in item.tokens" :key="tokenIndex" :class="'answer-inline-' + token.kind">{{ token.text }}</text></text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="answer-text answer-list-content" selectable><text v-for="(token, tokenIndex) in item.tokens" :key="tokenIndex" :class="'answer-inline-' + token.kind">{{ token.text }}</text></text>
          <!-- #endif -->
        </view>
      </view>
      <scroll-view v-else-if="block.type === 'table'" class="answer-table-scroll" scroll-x>
        <view class="answer-table-grid" role="table">
          <view v-for="(row, rowIndex) in [block.header, ...block.rows]" :key="rowIndex" class="answer-table-row" :class="{ 'answer-table-header': rowIndex === 0 }" role="row">
            <view v-for="(cell, cellIndex) in row" :key="cellIndex" class="answer-table-cell" :role="rowIndex === 0 ? 'columnheader' : 'cell'">
              <!-- #ifdef MP-WEIXIN -->
              <text class="answer-text" user-select><text v-for="(token, tokenIndex) in cell" :key="tokenIndex" :class="'answer-inline-' + token.kind">{{ token.text }}</text></text>
              <!-- #endif -->
              <!-- #ifndef MP-WEIXIN -->
              <text class="answer-text" selectable><text v-for="(token, tokenIndex) in cell" :key="tokenIndex" :class="'answer-inline-' + token.kind">{{ token.text }}</text></text>
              <!-- #endif -->
            </view>
          </view>
        </view>
      </scroll-view>
      <template v-else>
        <!-- #ifdef MP-WEIXIN -->
        <text class="answer-text" user-select><text v-for="(token, tokenIndex) in block.tokens" :key="tokenIndex" :class="'answer-inline-' + token.kind">{{ token.text }}</text></text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="answer-text" selectable><text v-for="(token, tokenIndex) in block.tokens" :key="tokenIndex" :class="'answer-inline-' + token.kind">{{ token.text }}</text></text>
        <!-- #endif -->
      </template>
    </view>
  </view>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { parseAnswerBody } from '../services/answer-body.js';

const props = defineProps({ text: { type: String, default: '' } });
const body = computed(() => parseAnswerBody(props.text));
const copiedIndex = ref(-1), copyErrorIndex = ref(-1);
let generation = 0;
watch(() => props.text, () => { generation += 1; copiedIndex.value = -1; copyErrorIndex.value = -1; });
function copyCode(text, index) {
  const current = ++generation, answer = props.text;
  const stillCurrent = () => current === generation && answer === props.text;
  copiedIndex.value = -1; copyErrorIndex.value = -1;
  const failed = () => { if (stillCurrent()) copyErrorIndex.value = index; };
  try {
    uni.setClipboardData({ data: text, showToast: false,
      success: () => { if (stillCurrent()) copiedIndex.value = index; }, fail: failed });
  } catch { failed(); }
}
</script>

<style scoped>
.answer-body { min-width:0; color:inherit; font-size:15px; line-height:1.85; overflow-wrap:anywhere; }
.answer-block + .answer-block { margin-top:14px; }
.answer-text { white-space:pre-wrap; overflow-wrap:anywhere; }
.answer-format-note { display:block; margin-bottom:10px; color:var(--dt-muted, #626c65); font-size:11px; }
.answer-heading { font-size:16px; font-weight:650; line-height:1.55; }
.answer-heading-1 { font-size:22px; }
.answer-heading-2 { font-size:19px; }
.answer-heading-3 { font-size:17px; }
.answer-heading:not(:first-child) { margin-top:24px; }
.answer-quote { padding:9px 14px; border-left:3px solid #9aa98b; border-radius:0 8px 8px 0; background:#f0f3ec; color:#4e6154; }
.answer-inline-strong { font-weight:650; }
.answer-inline-em { font-style:italic; }
.answer-inline-code { padding:1px 4px; border-radius:4px; background:#edf0e8; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.9em; }
.answer-inline-citation { color:#496b4c; font-size:.88em; font-weight:600; white-space:nowrap; }
.answer-list-row { display:flex; align-items:flex-start; gap:9px; }
.answer-list-row + .answer-list-row { margin-top:6px; }
.answer-list-marker { flex:0 0 auto; min-width:15px; color:#62745f; font-variant-numeric:tabular-nums; }
.answer-list-content { flex:1; min-width:0; }
.answer-code { max-width:100%; overflow:hidden; border:1px solid #dfe5d9; border-radius:12px; background:#f5f7f1; }
.answer-code-toolbar { display:flex; justify-content:space-between; align-items:center; gap:10px; min-height:44px; padding:0 12px; border-bottom:1px solid #e0e5da; }
.answer-code-language { min-width:0; color:#697563; font-size:11px; overflow-wrap:anywhere; }
.answer-code-copy { flex-shrink:0; min-height:44px; margin:0; padding:8px 4px; border:0; background:transparent; color:#42633e; font-size:12px; line-height:1.5; }
.answer-code-copy::after { border:0; }
.answer-code-scroll, .answer-table-scroll { width:100%; max-width:100%; }
.answer-code-text { display:block; min-width:max-content; padding:14px; white-space:pre; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; line-height:1.8; }
.answer-copy-error { display:block; padding:0 12px 10px; color:#956344; font-size:11px; }
.answer-table { max-width:100%; overflow:hidden; border:1px solid #dfe5d9; border-radius:12px; }
.answer-table-grid { display:table; min-width:100%; border-collapse:collapse; font-size:13px; line-height:1.7; }
.answer-table-row { display:table-row; }
.answer-table-cell { display:table-cell; min-width:120px; max-width:260px; padding:10px 12px; border-bottom:1px solid #e4e8df; border-right:1px solid #e4e8df; vertical-align:top; }
.answer-table-cell:last-child { border-right:0; }
.answer-table-row:last-child .answer-table-cell { border-bottom:0; }
.answer-table-header { background:#f1f4ec; font-weight:650; color:#40543c; }
</style>
