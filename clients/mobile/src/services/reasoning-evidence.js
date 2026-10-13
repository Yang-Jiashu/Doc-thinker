import { MAX_RESEARCH_SOURCES, isResearchSourceLabel, boundedAnswerSources } from './research-source-contract.js';

const compact = (value,max=800) => typeof value === 'string' ? Array.from(value.slice(0,max*2)).slice(0,max).join('').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u202a-\u202e\u2066-\u2069]/g,'') : '';
const array = (value,max) => Array.isArray(value) ? value.slice(0,max) : [];
const sourceId = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const operatorLabels = {analogy:'类比找参照',deduction:'演绎查条件',induction:'归纳找共性'};

// Transport only the public, source-backed receipt. Raw trace strings, claimed
// confidence and counters are never presentation data or a proof certificate.
export function sanitizeReasoningReceipts(trace, sources = []) {
  if (!trace || typeof trace !== 'object' || Array.isArray(trace)) return undefined;
  const current = boundedAnswerSources(sources,true).filter(item=>typeof (item?.id || item?.chunk_id) === 'string' && sourceId.test(item.id || item.chunk_id)
    && typeof item?.content === 'string' && item.content.trim()
    && item.content.length <= 24000);
  const validIds = value => Array.isArray(value) && value.length <= MAX_RESEARCH_SOURCES && new Set(value).size === value.length
    && value.every(id=>typeof id === 'string' && sourceId.test(id)
      && current.filter(item=>(item.id || item.chunk_id) === id).length === 1
      && current.filter(item=>item.label === current.find(item=>(item.id || item.chunk_id) === id)?.label).length === 1);
  const out = {}, graph = trace.graph_retrieval, hypothesis = trace.hypothesis_support;
  if (graph && ['expanded','unchanged'].includes(graph.status) && graph.algorithm === 'bounded_evidence_graph_v1'
    && graph.model_calls === 0 && graph.max_hops === 2 && graph.semantic_entailment_verified === false
    && validIds(graph.kept_source_ids) && graph.kept_source_ids.length && validIds(graph.bridge_source_ids)
    && graph.bridge_source_ids.every(id=>graph.kept_source_ids.includes(id))
    && (graph.status === 'expanded' ? graph.bridge_source_ids.length > 0 : graph.bridge_source_ids.length === 0)) {
    out.graph_retrieval = {status:graph.status,algorithm:graph.algorithm,model_calls:0,max_hops:2,
      semantic_entailment_verified:false,kept_source_ids:[...graph.kept_source_ids],bridge_source_ids:[...graph.bridge_source_ids]};
  }
  if (hypothesis?.status === 'ready' && hypothesis.algorithm === 'grounded_hypothesis_support_v1'
    && hypothesis.model_calls === 0 && hypothesis.semantic_entailment_verified !== true
    && validIds(hypothesis.source_ids) && hypothesis.source_ids.length
    && Array.isArray(hypothesis.algorithm_modes) && hypothesis.algorithm_modes.length <= 3
    && new Set(hypothesis.algorithm_modes).size === hypothesis.algorithm_modes.length
    && hypothesis.algorithm_modes.every(mode=>typeof mode === 'string' && Object.hasOwn(operatorLabels,mode))) {
    out.hypothesis_support = {status:'ready',algorithm:hypothesis.algorithm,model_calls:0,
      source_ids:[...hypothesis.source_ids],algorithm_modes:[...hypothesis.algorithm_modes]};
  }
  return Object.keys(out).length ? out : undefined;
}

export function reasoningEvidence({reasoning,claimAudit,sources = [],citationValidation = {}} = {}) {
  sources = boundedAnswerSources(sources,true);
  const revoked = reasoning?.status === 'source_unavailable' || array(citationValidation?.warnings,8).some(item=>['source_revoked','provenance_unavailable'].includes(item?.code));
  const literalRef = ref => {
    if (!ref || !isResearchSourceLabel(ref.label) || typeof ref.quote !== 'string' || !ref.quote || ref.quote.length > 2000 || !Number.isSafeInteger(ref.start) || !Number.isSafeInteger(ref.end)) return null;
    const source = sources.find(item=>(item?.id || item?.chunk_id) === ref.source_id && item?.label === ref.label);
    if (!source || typeof source.content !== 'string' || source.content.length > 24000) return null;
    const begin = source.excerpt_start ?? 0, characters = Array.from(source.content);
    if (!Number.isSafeInteger(begin) || begin < 0 || ref.start < begin || ref.end <= ref.start || ref.end > begin + characters.length) return null;
    if (characters.slice(ref.start-begin,ref.end-begin).join('') !== ref.quote) return null;
    return {label:ref.label,quote:ref.quote,filename:compact(source.filename || source.title,120),
      page:Number.isSafeInteger(source.page) && source.page > 0 ? source.page : null,start:ref.start,end:ref.end};
  };
  const claims = array(claimAudit?.claims,12).map(item=>{
    const refs = revoked ? [] : array(item?.source_refs,3).map(literalRef).filter(Boolean);
    const text = compact(item?.text), unavailable = revoked || item?.status === 'source_unavailable' || claimAudit?.status === 'unavailable';
    const exact = !unavailable && item?.status === 'exact_quote' && Array.isArray(item.source_refs)
      && item.source_refs.length > 0 && item.source_refs.length <= 3 && refs.length === item.source_refs.length
      && new Set(refs.map(ref=>ref.label)).size === refs.length && refs.every(ref=>ref.quote === text)
      && Array.isArray(item.issues) && item.issues.length === 0;
    return {text,refs:unavailable ? [] : refs,status:unavailable ? 'source_unavailable' : exact ? 'exact_quote' : 'unverified',
      label:unavailable ? '来源不可用' : exact ? '原文逐字对应' : '结论待核对',
      issues:array(item?.issues,8).map(value=>compact(value?.message,300)).filter(Boolean)};
  }).filter(item=>item.text);
  const omitted = claimAudit?.summary?.omitted_statement_count;
  const auditLimitNote = !revoked && claimAudit && claimAudit.status !== 'unavailable'
    ? Number.isSafeInteger(omitted) && omitted >= 0 && omitted <= 100000
      ? omitted ? `另有 ${omitted} 段超出本次核对窗口，未检查其引用和原文。` : ''
      : '只展示有限核对窗口，服务未提供窗口外的表述数量，不能视为全文核对。'
    : '';
  const unavailable = revoked || reasoning?.status === 'source_unavailable' || claimAudit?.status === 'unavailable';
  const conclusions = unavailable ? [] : array(reasoning?.conclusions,12).map(item=>({
    text:compact(item?.text),label:item?.kind === 'deduced' || reasoning?.mode === 'deduction' ? '基于已列前提的推断' : item?.kind === 'inductive_candidate' || reasoning?.mode === 'induction' ? '有限样本中的候选规律' : item?.kind === 'analogy_hypothesis' || reasoning?.mode === 'analogy' ? '待验证的迁移假设' : '辅助分析线索 · 待核对',
    mappings:array(item?.mappings,8).map(value=>typeof value === 'string' ? compact(value,300) : [compact(value?.from ?? value?.source ?? value?.base,160),compact(value?.to ?? value?.target,160)].filter(Boolean).join(' → ')).filter(Boolean),
    differences:array(item?.differences,8).map(value=>typeof value === 'string' ? compact(value,300) : compact(value?.text ?? value?.message,300)).filter(Boolean),
    hypotheses:array(item?.candidate_inferences,4).map(value=>{
      const text=compact(value?.text), conditions=array(value?.conditions,4).map(item=>compact(item,160)).filter(Boolean);
      return text ? text + (conditions.length ? `；沿用条件：${conditions.join('、')}` : '') + '；目标场景尚未验证。' : '';
    }).filter(Boolean),
    proof:array(item?.proof,24).map(value=>({text:compact(value?.text,500),label:value?.kind === 'deduced' ? '中间推断' : value?.kind === 'rule' ? '参考条件 · 待核对' : value?.kind === 'observation' ? '材料整理的前提 · 待核对' : '辅助线索 · 待核对',
      sources:sources.filter(source=>array(value?.source_ids,MAX_RESEARCH_SOURCES).includes(source?.id || source?.chunk_id)).map(source=>source.label)})).filter(value=>value.text),
    sourceLabels:sources.filter(source=>array(item?.source_ids,MAX_RESEARCH_SOURCES).includes(source?.id || source?.chunk_id)).map(source=>source.label),
    supportCount:Number.isSafeInteger(item?.support_count) && item.support_count >= 0 ? item.support_count : null,
    counterexampleCount:Number.isSafeInteger(item?.counterexample_count) && item.counterexample_count >= 0 ? item.counterexample_count : null,
  })).filter(item=>item.text);
  const gaps = unavailable ? [] : array(reasoning?.gaps,8).map(item=>compact(item?.message,500)).filter(Boolean);
  const receipt = unavailable ? undefined : sanitizeReasoningReceipts(reasoning?.trace,sources);
  const labels = ids => ids.map(id=>sources.find(source=>(source?.id || source?.chunk_id) === id)?.label).filter(Boolean);
  const graph = receipt?.graph_retrieval, hypothesis = receipt?.hypothesis_support;
  const graphRetrieval = graph ? {title:'关联线索',summary:graph.status === 'expanded' ? '已沿材料关联补充检索，并保留以下原文供核对。' : '已检查材料关联，本次没有加入额外原文。',
    sourceLabels:labels(graph.kept_source_ids),bridgeLabels:labels(graph.bridge_source_ids),note:'关联用于找资料，不代表因果证明。'} : null;
  const hypothesisSupport = hypothesis ? {title:'假设边界',operators:hypothesis.algorithm_modes.map(mode=>operatorLabels[mode]),
    summary:hypothesis.algorithm_modes.length ? hypothesis.algorithm_modes.map(mode=>operatorLabels[mode]).join(' · ') : '本次未得到可解析的三类算法线索；仍按原文组织待验证候选。',
    sourceLabels:labels(hypothesis.source_ids),note:'这些方法只在后台辅助组织材料，不表示均已得到结论。假设需验证；模型背景知识不是本次资料依据。'} : null;
  return {available:unavailable || Boolean(conclusions.length || claims.length || gaps.length || graphRetrieval || hypothesisSupport),unavailable,title:unavailable ? '依据已变化，请重新核对' : '参考线索与核对',
    note:unavailable ? '关联来源已撤销或暂不可用，旧推断不再作为可核验的建议展示。' : '以下线索辅助组织回答，不是新的原文事实。参考案例不能证明结论；原文对应只核对文字，分析仍取决于前提、条件与反例。'+(reasoning?.display_limited ? ' 此处仅展示部分分析步骤或结构对应，不是完整证明。' : '')+(auditLimitNote ? ' '+auditLimitNote : ''),
    conclusions,claims,gaps,graphRetrieval,hypothesisSupport};
}

export function reasoningRecord(options) {
  const view = reasoningEvidence(options);
  if (!view.available) return '';
  const lines = [view.title,view.note];
  if (view.graphRetrieval) {
    const graph = view.graphRetrieval;
    lines.push(`${graph.title}：${graph.summary}`,graph.note,'保留原文：'+graph.sourceLabels.map(label=>`[${label}]`).join(' '));
    if (graph.bridgeLabels.length) lines.push('关联补充：'+graph.bridgeLabels.map(label=>`[${label}]`).join(' '));
  }
  if (view.hypothesisSupport) {
    const hypothesis = view.hypothesisSupport;
    lines.push(`${hypothesis.title}：${hypothesis.summary}`,hypothesis.note,'关联原文：'+hypothesis.sourceLabels.map(label=>`[${label}]`).join(' '));
  }
  for (const conclusion of view.conclusions) {
    lines.push(`${conclusion.label}：${conclusion.text}`);
    if (conclusion.mappings.length) lines.push('结构对应：'+conclusion.mappings.join('；'));
    if (conclusion.differences.length) lines.push('差异与边界：'+conclusion.differences.join('；'));
    if (conclusion.hypotheses.length) lines.push('待验证的迁移假设：'+conclusion.hypotheses.join('；'));
    for (const step of conclusion.proof) lines.push(`${step.label} ${step.sources.map(label=>`[${label}]`).join(' ')}：${step.text}`);
    if (conclusion.sourceLabels.length) lines.push('关联原文：'+conclusion.sourceLabels.map(label=>`[${label}]`).join(' '));
  }
  if (view.gaps.length) lines.push('尚缺的条件：'+view.gaps.join('；'));
  for (const claim of view.claims) {
    lines.push(`${claim.label}：${claim.text}`);
    for (const ref of claim.refs) lines.push(`[${ref.label}] ${ref.filename}${ref.page ? ` · 第 ${ref.page} 页` : ''}：${ref.quote}`);
    if (claim.issues.length) lines.push('核对点：'+claim.issues.join('；'));
  }
  return lines.join('\n\n');
}
