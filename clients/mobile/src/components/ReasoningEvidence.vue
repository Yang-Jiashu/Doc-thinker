<template>
  <view v-if="view.available" class="reasoning-evidence">
    <button class="reasoning-toggle" :aria-expanded="expanded" @tap="expanded = !expanded"><text>{{ expanded ? '−' : '＋' }}</text><text>{{ view.title }} · {{ expanded ? '收起' : '展开' }}</text></button>
    <view v-if="expanded" class="reasoning-panel">
      <text class="reasoning-note">{{ view.note }}</text>
      <view v-if="view.graphRetrieval" class="reasoning-item">
        <text class="reasoning-label">{{ view.graphRetrieval.title }}</text>
        <text class="reasoning-copy">{{ view.graphRetrieval.summary }}</text>
        <text class="reasoning-detail">保留原文 · {{ view.graphRetrieval.sourceLabels.join(' · ') }}</text>
        <text v-if="view.graphRetrieval.bridgeLabels.length" class="reasoning-detail">关联补充 · {{ view.graphRetrieval.bridgeLabels.join(' · ') }}</text>
        <text class="reasoning-detail">{{ view.graphRetrieval.note }}</text>
      </view>
      <view v-if="view.hypothesisSupport" class="reasoning-gaps">
        <text class="reasoning-label">{{ view.hypothesisSupport.title }}</text>
        <text class="reasoning-copy">{{ view.hypothesisSupport.summary }}</text>
        <text class="reasoning-detail">关联原文 · {{ view.hypothesisSupport.sourceLabels.join(' · ') }}</text>
        <text class="reasoning-detail">{{ view.hypothesisSupport.note }}</text>
      </view>
      <view v-for="(item,index) in view.conclusions" :key="'conclusion-'+index" class="reasoning-item">
        <text class="reasoning-label">{{ item.label }}</text><text class="reasoning-copy">{{ item.text }}</text>
        <text v-for="(mapping,i) in item.mappings" :key="'map-'+i" class="reasoning-detail">对应 · {{ mapping }}</text>
        <text v-for="(difference,i) in item.differences" :key="'diff-'+i" class="reasoning-detail">边界 · {{ difference }}</text>
        <view v-for="(hypothesis,i) in item.hypotheses" :key="'hyp-'+i" class="reasoning-quote"><text class="reasoning-label">待验证的迁移假设</text><text class="reasoning-copy">{{ hypothesis }}</text></view>
        <view v-for="(step,i) in item.proof" :key="'proof-'+i" class="reasoning-proof"><text class="reasoning-detail">{{ i + 1 }} · {{ step.label }} {{ step.sources.join(' · ') }}</text><text class="reasoning-copy">{{ step.text }}</text></view>
        <text v-if="item.sourceLabels.length" class="reasoning-detail">关联原文 · {{ item.sourceLabels.join(' · ') }}</text>
        <text v-if="item.supportCount !== null" class="reasoning-detail">本次找到 {{ item.supportCount }} 个支持案例{{ item.counterexampleCount !== null ? `，${item.counterexampleCount} 个反例` : '' }}；不能代表所有情况。</text>
      </view>
      <view v-if="view.gaps.length" class="reasoning-gaps"><text class="reasoning-label">还缺什么</text><text v-for="(gap,index) in view.gaps" :key="index" class="reasoning-detail">{{ gap }}</text></view>
      <view v-for="(claim,index) in view.claims" :key="'claim-'+index" class="reasoning-item">
        <text class="reasoning-label" :class="{'reasoning-exact':claim.status === 'exact_quote'}">{{ claim.label }}</text><text class="reasoning-copy">{{ claim.text }}</text>
        <view v-for="(ref,i) in claim.refs" :key="i" class="reasoning-quote"><text class="reasoning-detail">{{ ref.label }} · {{ ref.filename }}{{ ref.page ? ` · 第 ${ref.page} 页` : '' }}</text>
          <!-- #ifdef MP-WEIXIN -->
          <text class="reasoning-copy" user-select>{{ ref.quote }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="reasoning-copy" selectable>{{ ref.quote }}</text>
          <!-- #endif -->
        </view>
        <text v-for="(issue,i) in claim.issues" :key="i" class="reasoning-detail">核对 · {{ issue }}</text>
      </view>
    </view>
  </view>
</template>
<script setup>
import {computed,ref,watch} from 'vue';
import {reasoningEvidence} from '../services/reasoning-evidence.js';
const props = defineProps({reasoning:{type:Object,default:null},claimAudit:{type:Object,default:null},sources:{type:Array,default:()=>[]},citationValidation:{type:Object,default:()=>({})}});
const expanded = ref(false), view = computed(()=>reasoningEvidence(props));
watch(()=>[props.reasoning,props.claimAudit,props.sources,props.citationValidation],()=>{expanded.value=false;},{deep:true});
</script>
<style scoped>
.reasoning-evidence { margin-top:10px; }
.reasoning-toggle { display:flex; gap:8px; align-items:center; min-height:44px; padding:10px 12px; margin:0; border:1px solid #dedfce; border-radius:12px; background:#f2f2e8; color:#4b5644; font-size:12px; line-height:1.7; text-align:left; }
.reasoning-toggle::after { border:0; }
.reasoning-panel { padding:16px; margin-top:8px; background:#fbfcf6; border:1px solid #e1e6d8; border-radius:16px; }
.reasoning-note,.reasoning-detail { display:block; font-size:11px; line-height:1.9; color:#697661; overflow-wrap:anywhere; }
.reasoning-item { margin-top:16px; padding-top:14px; border-top:1px solid #e6eadd; }
.reasoning-label { display:block; font-size:11px; color:#8a6a3f; margin-bottom:7px; }
.reasoning-exact { color:#406947; }
.reasoning-copy { display:block; font-size:13px; color:#344c3a; line-height:1.9; white-space:pre-wrap; overflow-wrap:anywhere; }
.reasoning-quote { padding:10px 12px; margin:9px 0; border-left:2px solid #bacdaa; background:#f1f5e9; }
.reasoning-gaps { margin-top:14px; padding:12px; border-radius:10px; background:#f4f0e4; }
.reasoning-proof { margin-top:10px; padding-left:11px; border-left:2px solid #c4d4b5; }
</style>
