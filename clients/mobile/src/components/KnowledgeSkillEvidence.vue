<template>
  <view v-if="skill" class="knowledge-reference">
    <button class="knowledge-toggle" role="button" :aria-expanded="expanded" @tap="expanded = !expanded"><text>{{ expanded ? '−' : '＋' }} 方法参考 · {{ skill.name }}</text><text class="knowledge-count">{{ skill.sources.length ? `${skill.sources.length} 条` : '本次未加入' }}</text></button>
    <view v-if="expanded" class="knowledge-panel">
      <text class="knowledge-boundary">{{ KNOWLEDGE_SKILL_BOUNDARY }}</text>
      <text class="knowledge-note">项目原创方法设计 · 尚未实证验证</text>
      <text class="knowledge-note">{{ knowledgeSkillNote(skill.status,skill.method_plan) }}</text>
      <text v-if="skill.citation_validation.missing_citations || skill.citation_validation.unknown_labels.length" class="knowledge-boundary">本次 K 引用缺失或标签不在返回条目内，请核对。</text>
      <view v-for="source in skill.sources" :key="source.id" class="knowledge-source">
        <text class="knowledge-title">[{{ source.label }}] {{ source.title }}</text>
        <text v-if="source.problem_pattern" class="knowledge-copy">适用问题：{{ source.problem_pattern }}</text>
        <text v-if="source.required_facts?.length" class="knowledge-copy">先了解哪些事实：{{ source.required_facts.join('；') }}</text>
        <text v-if="source.decision_rule" class="knowledge-copy">判断规则：{{ source.decision_rule }}</text>
        <view v-if="source.action_template?.length" class="knowledge-copy"><text>建议行动模板</text><text v-for="(step,index) in source.action_template" :key="index" class="knowledge-step">{{ index + 1 }}. {{ step }}</text></view>
        <text v-if="source.stop_conditions?.length" class="knowledge-boundary">停止或重新判断：{{ source.stop_conditions.join('；') }}</text>
        <text v-if="source.counterexample" class="knowledge-copy">不适用例子：{{ source.counterexample }}</text>
        <text class="knowledge-origin">{{ source.reference_kind === 'project_original' ? '项目原创方法来源' : '历史启发来源' }}：{{ source.source_title }}</text>
        <!-- #ifdef MP-WEIXIN -->
        <text v-if="source.reference_kind !== 'project_original' && source.excerpt" class="knowledge-quote" user-select>原文短摘：{{ source.excerpt }}</text>
        <text class="knowledge-copy" user-select>{{ source.reference_kind === 'project_original' ? '原创方法说明' : '产品方法解读（非原文）' }}：{{ source.commentary }}</text>
        <text class="knowledge-copy" user-select>迁移边界：{{ source.boundaries.join('；') }}</text>
        <text v-if="source.source_url" class="knowledge-url" user-select>来源地址（未在本次打开核验）：{{ source.source_url }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text v-if="source.reference_kind !== 'project_original' && source.excerpt" class="knowledge-quote" selectable>原文短摘：{{ source.excerpt }}</text>
        <text class="knowledge-copy" selectable>{{ source.reference_kind === 'project_original' ? '原创方法说明' : '产品方法解读（非原文）' }}：{{ source.commentary }}</text>
        <text class="knowledge-copy" selectable>迁移边界：{{ source.boundaries.join('；') }}</text>
        <text v-if="source.source_url" class="knowledge-url" selectable>来源地址（未在本次打开核验）：{{ source.source_url }}</text>
        <!-- #endif -->
      </view>
    </view>
  </view>
</template>
<script setup>
import {computed,ref,watch} from 'vue';
import {knowledgeSkillMetadata,knowledgeSkillNote,KNOWLEDGE_SKILL_BOUNDARY} from '../services/knowledge-skills.js';
const props = defineProps({value:{type:Object,default:null}});
const skill = computed(()=>knowledgeSkillMetadata(props.value)), expanded = ref(false);
watch(()=>props.value,()=>{expanded.value=false;},{deep:true});
</script>
<style scoped>
.knowledge-reference{margin-top:10px}.knowledge-toggle{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:44px;margin:0;padding:10px 12px;border:1px solid #dddcca;border-radius:12px;background:#f6f4ec;color:#65583c;font-size:12px;text-align:left;line-height:1.7}.knowledge-toggle::after{border:0}.knowledge-count{font-size:11px;white-space:nowrap;color:#877859}.knowledge-panel{padding:13px;border:1px solid #e7e2d2;border-radius:12px;margin-top:7px;background:#fdfbf5}.knowledge-boundary,.knowledge-note,.knowledge-title,.knowledge-origin,.knowledge-quote,.knowledge-copy,.knowledge-url{display:block;line-height:1.85;overflow-wrap:anywhere;font-size:12px;color:#655d4e}.knowledge-boundary{color:#84633a}.knowledge-note,.knowledge-origin{font-size:11px;color:#857964;margin-top:6px}.knowledge-source{border-top:1px solid #e9e3d5;padding-top:12px;margin-top:12px}.knowledge-title{font-weight:600}.knowledge-quote{border-left:2px solid #c9ba8f;padding-left:10px;margin-top:9px}.knowledge-copy{margin-top:9px}.knowledge-url{margin-top:9px;font-size:10px;color:#877963}
.knowledge-step{display:block;margin-top:4px}.knowledge-source>.knowledge-boundary{margin-top:9px}.knowledge-origin{border-top:1px dashed #e5decb;padding-top:10px;margin-top:12px}
.knowledge-note{white-space:pre-line}
</style>
