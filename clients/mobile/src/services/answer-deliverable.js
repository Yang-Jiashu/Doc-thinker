import { MAX_EVIDENCE_ITEMS, normalizeAnswerEvidence } from './answer-evidence.js';
import {reasoningRecord} from './reasoning-evidence.js';
import {knowledgeSkillRecord,knowledgeSkillMetadata} from './knowledge-skills.js';

export const MAX_RECORD_ANSWER = 40000;
export const MAX_RECORD_QUESTION = 8000;

function validText(value, limit) {
  if (typeof value !== 'string' || value.length > limit * 2 || !value.trim()) return false;
  let count = 0;
  for (const _character of value) if (++count > limit) return false;
  return true;
}

// A literal local record, not a model-generated report or a verified citation.
// No credentials, hidden history, user IDs or storage handles. A selected K
// knowledge pack may carry a strictly allowlisted HTTPS source as plain text.
export function answerDeliverable({ answer, question = '', sources = [], demo = false,
  researchCore = false, citationValidation = {}, answerMode = '', retrievalGaps = [],reasoning = null,claimAudit = null,memoryAsOf = null,knowledgeSkill = null } = {}) {
  if (!validText(answer, MAX_RECORD_ANSWER)) return { available:false, text:'', reason:'answer-size' };
  if (question && !validText(question, MAX_RECORD_QUESTION)) return { available:false, text:'', reason:'question-size' };
  const evidence = normalizeAnswerEvidence(sources);
  const parts = [demo ? 'DocThinker · 演示回答记录' : 'DocThinker · 回答记录',
    demo ? '【演示】模拟回答与片段，不是真实模型或文档分析。'
      : '【待核对】助手回答并非事实核验；服务返回片段也可能包含模型整理内容。'];
  // Provenance problems must survive copying, not just appear in a UI badge.
  // Use fixed local descriptions only; never copy arbitrary remote warning text.
  if (researchCore === true) {
    const warnings = Array.isArray(citationValidation?.warnings) ? citationValidation.warnings : [];
    const gaps = Array.isArray(retrievalGaps) ? retrievalGaps : [];
    const notes = ['引用标签只说明原文出处，不代表已验证结论正确。'];
    if (warnings.some(item=>item?.code === 'source_revoked')) notes.push('部分来源已被删除或撤销，回答中的相关引用和结论无法继续核对。');
    if (warnings.some(item=>item?.code === 'provenance_unavailable')) notes.push('依据记录不可用，不能将本回答当作已核验的资料结论。');
    if (citationValidation?.missing_citations === true || (Array.isArray(citationValidation?.unknown_labels) && citationValidation.unknown_labels.length)) notes.push('引用缺失或标签不在本次来源内，请核对，不建议直接记为事实。');
    if (answerMode === 'insufficient_evidence') notes.push('资料不足，本次没有请求模型生成资料结论。');
    if (gaps.some(item=>item?.code === 'bounded_overview')) notes.push('这里只使用各资料的开头节选，不是完整全文总结。');
    if (gaps.some(item=>item?.code === 'excerpt_truncated')) notes.push('本次只使用部分原文节选，不是全文证据。');
    if (gaps.some(item=>item?.code === 'uncovered_query_terms')) notes.push('部分问题用词未在本次节选匹配，不代表概念缺失或整篇论文没有相关证据。');
    if (gaps.some(item=>item?.code === 'documents_not_selected')) notes.push('部分资料未进入本次有限上下文，未展示不等于没有相关证据。');
    if (gaps.some(item=>item?.code === 'source_antecedent_unverified')) notes.push('部分节选的回指缺少可见对象，不能从下一实验猜前提；请回到原文核对。');
    if (gaps.some(item=>item?.code === 'source_boundary_unverified')) notes.push('部分节选没有可确认的句尾，可能遗漏条件；句尾完整也不等于前提完整。');
    if (gaps.some(item=>item?.code === 'literal_coverage_scan_limited')) notes.push('词面选证达到扫描上限，未查完全部资料；未找到不代表没有相关内容。');
    if (gaps.some(item=>['evidence_facets_uncovered','evidence_anchors_uncovered'].includes(item?.code))) notes.push('部分问题要点还缺少具体原文，不能用背景介绍代替，也不能据此补出结论。');
    if (gaps.some(item=>item?.code === 'complete_statement_exceeds_budget')) notes.push('部分完整原句或限定条件未能放入本次上下文，请回到原文核对。');
    if (gaps.some(item=>item?.code === 'semantic_retrieval_unavailable')) notes.push('语义检索暂不可用，本次仅按词面查找，可能漏掉同义或跨语言的证据。');
    if (gaps.some(item=>item?.code === 'semantic_retrieval_windowed')) notes.push('语义检索达到扫描上限，不代表已查完全部资料。');
    if (gaps.some(item=>item?.code === 'evidence_coverage_proxy')) notes.push('找到相关片段不代表已找齐关键证据或验证结论。');
    if (gaps.some(item=>['context_budget_omitted','confirmed_memory_budget_omitted'].includes(item?.code))) notes.push('受输入预算限制，部分历史或确认记忆未加入本次回答。');
    if (gaps.some(item=>['graph_capacity_reached','graph_path_window_incomplete'].includes(item?.code))) notes.push('部分图关联所需的完整原文未进入本次上下文，不能据此声称推理链完整。');
    if (gaps.some(item=>['graph_window_limited','graph_search_limited','graph_hub_omitted','graph_parser_is_partial','graph_excerpt_windowed'].includes(item?.code))) notes.push('图检索有解析、搜索和节选范围限制，未找到不代表不存在；关联不等于因果证明。');
    if (gaps.some(item=>['graph_counterevidence_in_path','graph_path_conditions_differ'].includes(item?.code))) notes.push('关联材料包含否定或条件差异，不能直接串成正向因果链。');
    parts.push('本次依据与覆盖边界', notes.join('\n'));
    if (typeof memoryAsOf === 'number' && Number.isFinite(memoryAsOf) && memoryAsOf >= 0 && memoryAsOf <= 253402300799) parts.push('记忆回看时间',new Date(memoryAsOf*1000).toISOString()+'（UTC，有效时间；表示目前所知的历史背景，不代表当时已经获知。）');
  }
  if (question) parts.push('当前问题', question);
  parts.push('助手回答', answer);
  if (researchCore) { const knowledgeText = knowledgeSkillRecord(knowledgeSkill); if (knowledgeText) parts.push(knowledgeText); }
  if (researchCore) { const reasoningText = reasoningRecord({reasoning,claimAudit,sources,citationValidation}); if (reasoningText) parts.push('参考线索与逐条依据',reasoningText); }
  if (!evidence.items.length) parts.push('依据说明', researchCore && knowledgeSkillMetadata(knowledgeSkill)?.status === 'ready'
    ? '本回答没有用户资料原文片段；K 方法参考不能替代用户事实，也不证明建议正确。'
    : '本回答没有附带可展示的来源片段，不能据此声称有文档依据。');
  else {
    parts.push('服务返回片段（尚未逐条核验）');
    evidence.items.forEach((item, index) => {
      parts.push(`${item.label || index + 1}. ${item.title}${item.page ? ' · 第 ' + item.page + ' 页' : ''}`, item.content);
      if (item.truncated) parts.push('【该片段已截短，需回到原资料核对完整内容】');
    });
    if (evidence.limited) parts.push(`【仅包含部分返回片段，最多 ${MAX_EVIDENCE_ITEMS} 条；不是完整资料导出】`);
  }
  return { available:true, text:parts.join('\n\n'), reason:'' };
}
