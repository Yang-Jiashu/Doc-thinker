// Knowledge packs are inert reference material, never executable client skills.
// Keep their K labels separate from uploaded S evidence and confirmed memory.
const safeText = (value, max) => typeof value === 'string' && value.length <= max * 2
  && Array.from(value).length <= max && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u202a-\u202e\u2066-\u2069]/.test(value)
  && Array.from(value).every(character => { const point = character.codePointAt(0); return point < 0xd800 || point > 0xdfff; });
const builtins = Object.freeze({
  'mao-workplace': {version:'2',referenceKind:'historical_inspiration'},
  'research-ideation': {version:'1',referenceKind:'project_original'},
  'literature-synthesis': {version:'1',referenceKind:'project_original'},
  'evidence-writing': {version:'1',referenceKind:'project_original'},
  'decision-review': {version:'2',referenceKind:'project_original'},
});
// The catalog accepts only current resources. A bounded old decision card can
// still be shown in an archived answer; it is never loaded as today's method.
const known = (value, historical = false) => typeof value?.id === 'string' && Object.hasOwn(builtins,value.id)
  && (value.version === builtins[value.id].version
    || (historical && value.id === 'decision-review' && value.version === '1'));
const hosts = ['www.gov.cn','www.people.com.cn','cpc.people.com.cn','dangshi.people.com.cn','www.12371.cn','news.12371.cn'];
const methodGapNotes = Object.freeze({
  invalid_sources:'部分材料记录不完整',source_limit_exceeded:'本次只检查了部分返回材料',
  invalid_source_labels:'部分材料的出处标记待核对',duplicate_source_labels:'部分材料的出处标记重复',
  empty_excerpts:'部分材料没有可读节选',missing_document_ids:'部分节选无法确认所属文档',
  invalid_source_ids:'部分节选的出处记录待核对',invalid_excerpt_geometry:'部分节选与原文位置记录不一致',
  no_usable_excerpts:'本次没有结构完整的可用节选',
  insufficient_distinct_documents:'本次节选不足两份不同文档，不能据此完成跨文档对照',
});
const boundedCount = value => Number.isInteger(value) && value >= 0 && value <= 12;
function sourceLabels(value, allowed = null) {
  if (!Array.isArray(value) || value.length > 12) return null;
  const labels = Array.from(value);
  if (!labels.every(label=>typeof label === 'string' && label.length <= 5 && /^S[1-9][0-9]{0,3}$/.test(label)
    && (!allowed || allowed.has(label))) || new Set(labels).size !== labels.length) return null;
  return labels;
}
function methodPlanMetadata(value, skill) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || value.kind !== 'method_plan'
    || value.version !== 1 || value.skill_id !== skill.id || value.semantic_entailment_verified !== false
    || value.model_calls !== 0 || value.method_execution_claimed !== false
    || !Array.isArray(value.methods) || value.methods.length !== skill.sources.length) return null;
  const methods = [];
  for (const [index, item] of value.methods.entries()) {
    const source = skill.sources[index], role = source.reasoning_role || 'unspecified';
    if (item?.id !== source.id || item.label !== source.label || item.reasoning_role !== role
      || item.requirement_count !== (source.required_facts?.length || 0)
      || item.requirements_status !== 'pending_review' || item.stop_conditions_status !== 'pending_review') return null;
    methods.push({id:source.id,label:source.label,reasoning_role:role,requirement_count:item.requirement_count,
      requirements_status:'pending_review',stop_conditions_status:'pending_review'});
  }
  const labels = sourceLabels(value.source_labels), evidence = value.evidence;
  if (!labels || !evidence || typeof evidence !== 'object' || Array.isArray(evidence)) return null;
  for (const key of ['selected_sources','usable_sources','unique_excerpts','distinct_documents']) {
    if (!boundedCount(evidence[key])) return null;
  }
  const allowed = new Set(labels), truncated = sourceLabels(evidence.truncated_sources,allowed),
    duplicates = sourceLabels(evidence.duplicate_excerpts,allowed);
  if (!truncated || !duplicates || labels.length > evidence.selected_sources
    || evidence.usable_sources > labels.length || evidence.unique_excerpts > evidence.usable_sources
    || evidence.distinct_documents > evidence.unique_excerpts
    || (evidence.unique_excerpts > 0 && evidence.distinct_documents === 0)
    || duplicates.length !== evidence.usable_sources - evidence.unique_excerpts
    || !Array.isArray(value.structural_gaps) || value.structural_gaps.length > Object.keys(methodGapNotes).length) return null;
  const gaps = Array.from(value.structural_gaps);
  if (!gaps.every(code=>typeof code === 'string' && Object.hasOwn(methodGapNotes,code))
    || new Set(gaps).size !== gaps.length
    || value.decision !== (gaps.length ? 'collect_evidence' : 'review_requirements')
    || (evidence.unique_excerpts === 0) !== gaps.includes('no_usable_excerpts')) return null;
  return {kind:'method_plan',version:1,skill_id:skill.id,methods,source_labels:labels,
    evidence:{selected_sources:evidence.selected_sources,usable_sources:evidence.usable_sources,
      unique_excerpts:evidence.unique_excerpts,distinct_documents:evidence.distinct_documents,
      truncated_sources:truncated,duplicate_excerpts:duplicates},structural_gaps:gaps,decision:value.decision,
    semantic_entailment_verified:false,model_calls:0,method_execution_claimed:false};
}
function methodPlanNote(plan) {
  if (!plan) return '';
  const lines = [`资料检查：${plan.evidence.distinct_documents}份文档、${plan.evidence.unique_excerpts}段不同节选；方法前提仍待核对。`,
    '仅检查本次节选的结构，尚未检查内容是否支持结论；文档数不代表独立证据，方法步骤并未执行。'];
  if (plan.decision === 'collect_evidence') lines.push(`先补充或核对材料：${plan.structural_gaps.map(code=>methodGapNotes[code]).join('；')}。未选入的内容不等于原资料不存在相关证据。`);
  if (plan.evidence.truncated_sources.length) lines.push('部分节选已截短，需要回到原文核对完整内容。');
  if (plan.evidence.duplicate_excerpts.length) lines.push('重复节选没有重复计为不同材料。');
  return lines.join('\n');
}
export const KNOWLEDGE_SKILL_BOUNDARY = 'K 标签是知识包的方法参考，不是你上传的事实，也不证明建议正确。方法适用性需要结合你的 S 原文证据和现实条件核对。';
export function knowledgeSourceUrl(value) {
  if (!safeText(value, 600)) return '';
  const match = /^https:\/\/([a-z0-9.-]+)(\/[^\s?#\\]*)?$/.exec(value);
  return match && hosts.includes(match[1]) ? value : '';
}
export function knowledgeSkillCatalog(value) {
  if (!Array.isArray(value?.items) || value.items.length > 8) return [];
  const seen = new Set(), items = [];
  for (const item of value.items) {
    if (!known(item) || seen.has(item.id) || !safeText(item.name,60) || !item.name.trim() || !safeText(item.description,400)) continue;
    seen.add(item.id);
    items.push({id:item.id,name:item.name,version:item.version,description:item.description});
  }
  return items;
}
export function knowledgeSkillMetadata(value) {
  if (!known(value, true) || !safeText(value.name,60) || !value.name.trim()
    || !['ready','not_applicable','no_match','budget_omitted','unavailable'].includes(value.status)) return null;
  const validation = value.citation_validation;
  const base = {id:value.id,name:value.name,version:value.version,status:value.status,sources:[],authorship:'project_original_method_design',validation_status:'design_unvalidated',citation_validation:{
    unknown_labels:Array.isArray(validation?.unknown_labels) ? validation.unknown_labels.filter(label=>typeof label === 'string' && /^K[0-9]{1,16}$/.test(label)).slice(0,16) : [],
    missing_citations:validation?.missing_citations === true,entailment_verified:false}};
  if (value.status !== 'ready') return base;
  if (!Array.isArray(value.sources) || !value.sources.length || value.sources.length > 3) return null;
  const ids = new Set();
  for (const [index, item] of value.sources.entries()) {
    const referenceKind = item?.reference_kind ?? (value.id === 'mao-workplace' ? 'historical_inspiration' : undefined);
    if (referenceKind !== builtins[value.id].referenceKind) return null;
    const boundaries = typeof item?.boundaries === 'string' ? [item.boundaries] : item?.boundaries;
    if (item?.label !== `K${index+1}` || typeof item.id !== 'string' || !/^[a-z0-9][a-z0-9._:-]{0,95}$/.test(item.id) || ids.has(item.id)
      || !safeText(item.title,160) || !item.title.trim() || !safeText(item.source_title,160) || !item.source_title.trim()
      || !safeText(item.excerpt,1000) || !safeText(item.commentary,1600) || !item.commentary.trim()
      || !Array.isArray(boundaries) || !boundaries.length || boundaries.length > 8 || !boundaries.every(text=>safeText(text,600) && text.trim())) return null;
    if (referenceKind === 'project_original' && (item.source_url !== '' || item.excerpt !== '' || !item.source_title.startsWith('DocThinker · '))) return null;
    if (item.published !== undefined && (typeof item.published !== 'string' || !/^\d{4}-\d{2}(?:-\d{2})?$/.test(item.published))) return null;
    const method = {};
    for (const [key,limit] of [['problem_pattern',200],['decision_rule',300],['counterexample',200]]) {
      if (item[key] === undefined) continue;
      if (!safeText(item[key],limit)) return null;
      method[key] = item[key];
    }
    for (const key of ['required_facts','action_template','stop_conditions']) {
      if (item[key] === undefined) continue;
      if (!Array.isArray(item[key]) || item[key].length > 8 || !item[key].every(text=>safeText(text,150))) return null;
      method[key] = [...item[key]];
    }
    if (['analogy','deduction','induction'].includes(item.reasoning_role)) method.reasoning_role = item.reasoning_role;
    ids.add(item.id);
    base.sources.push({label:item.label,id:item.id,title:item.title,source_title:item.source_title,reference_kind:referenceKind,
      ...(item.published !== undefined ? {published:item.published} : {}),
      source_url:knowledgeSourceUrl(item.source_url),excerpt:item.excerpt,commentary:item.commentary,boundaries:[...boundaries],...method});
  }
  const plan = methodPlanMetadata(value.method_plan,base);
  if (plan) base.method_plan = plan;
  return base;
}
export function knowledgeSkillNote(status, methodPlan = null) {
  const note = ({ready:'本次为回答提供了这些方法参考；是否适用仍需结合具体事实核对，不代表模型已正确遵循。',not_applicable:'本次未启用这个知识包的方法参考。',no_match:'本次检索未找到匹配条目，没有加入方法参考。',
    budget_omitted:'受输入预算限制，本次未加入知识包参考。',unavailable:'本次知识参考记录不可用，不能继续核对 K 引用。'})[status] || '';
  // Only normalized ready metadata reaches this note from the UI/export path.
  return status === 'ready' && methodPlan ? `${note}\n${methodPlanNote(methodPlan)}` : note;
}
export function knowledgeSkillRecord(value) {
  const skill = knowledgeSkillMetadata(value);
  if (!skill) return '';
  const lines = [`知识包 · ${skill.name}（版本 ${skill.version}）`,'项目原创方法设计 · 尚未实证验证',KNOWLEDGE_SKILL_BOUNDARY,knowledgeSkillNote(skill.status,skill.method_plan)];
  if (skill.citation_validation.missing_citations || skill.citation_validation.unknown_labels.length) lines.push('本次 K 引用缺失或标签不在返回条目内，请核对；不代表建议已被验证。');
  for (const source of skill.sources) {
    lines.push(`[${source.label}] ${source.title}`);
    if (source.problem_pattern) lines.push(`适用问题：${source.problem_pattern}`);
    if (source.required_facts?.length) lines.push(`先了解哪些事实：${source.required_facts.join('；')}`);
    if (source.decision_rule) lines.push(`判断规则：${source.decision_rule}`);
    if (source.action_template?.length) lines.push(`建议行动模板：${source.action_template.join('；')}`);
    if (source.stop_conditions?.length) lines.push(`停止或重新判断：${source.stop_conditions.join('；')}`);
    if (source.counterexample) lines.push(`不适用例子：${source.counterexample}`);
    const original = source.reference_kind === 'project_original';
    lines.push(`${original ? '项目原创方法来源' : '历史启发来源'}：${source.source_title}`);
    if (!original && source.excerpt) lines.push(`原文短摘：${source.excerpt}`);
    lines.push(`${original ? '原创方法说明' : '产品方法解读（非原文）'}：${source.commentary}`,`迁移边界：${source.boundaries.join('；')}`);
    if (source.source_url) lines.push(`来源地址（未在本次打开核验）：${source.source_url}`);
  }
  return lines.join('\n\n');
}
