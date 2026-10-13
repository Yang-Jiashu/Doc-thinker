// Local navigation, not answer generation: use only the currently loaded graph.
// A visual path is an undirected connection, never proof of a causal conclusion.
export const GRAPH_NAVIGATION_LIMITS = Object.freeze({
  queryChars: 256, nodes: 2000, edges: 8000, candidates: 8,
  hops: 3, visitedNodes: 256, traversedEdges: 2048,
});

const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const text = (value, max = 256) => typeof value === 'string'
  ? value.slice(0, max).replace(/[\u0000-\u001f\u007f\u202a-\u202e\u2066-\u2069]/g, '').trim() : '';
const nameKey = value => text(value).normalize('NFKC').toLowerCase().replace(/\s+/g, ' ');
const discovered = value => value === true || value === 1 || value === '1' || value === 'true';
const none = (reason, message, extra = {}) => ({kind: 'none', reason, message, ...extra});
const unquote = value => {
  const trimmed = value.trim();
  return /^(?:“.*”|「.*」|『.*』|".*"|'[^']*')$/.test(trimmed) ? trimmed.slice(1, -1).trim() : trimmed;
};
const polite = value => value.replace(/^(?:请帮我|麻烦帮我|帮我|请|麻烦)\s*/, '');

function parseIntent(query) {
  const command = polite(query.normalize('NFKC').replace(/[。！？!?]+$/g, '').trim());
  if (/^(?:返回上一步|回到上一步|回上一步|上一步|返回刚才|回到刚才|回到刚才的节点)$/.test(command)) return {kind: 'back'};
  if (/^(?:看|看看|查看|打开|显示|看一下)(?:原文|来源|证据|这条的来源|它的原文)$/.test(command)) return {kind: 'source'};

  // Keep the grammar explicit. Do not infer a command just because a paragraph
  // happens to contain an entity name, nor treat a quoted entity as instructions.
  const relation = command.match(/^(.*?)(?:之间)?(?:有什么|有何|有啥|是什么|的)?(?:联系|关联|关系)(?:吗|呢|呀|一下)?$/);
  if (relation) {
    const body = relation[1].replace(/^(?:看看|查看一下|查看|看一下|查查|找一下)\s*/, '').trim();
    const splits = [];
    for (let index = 0; index < body.length; index += 1) {
      if (!/[和与跟]/.test(body[index])) continue;
      const source = unquote(body.slice(0, index)), target = unquote(body.slice(index + 1));
      if (source && target) splits.push({source, target});
    }
    if (splits.length) return {kind: 'relation', splits};
  }

  const locate = command.match(/^(?:查找一下|搜索一下|定位一下|查看一下|找一下|定位到|查找|找到|找找|搜索|定位|看看|查看|看一下|找|看)\s*(.+)$/);
  const term = unquote((locate ? locate[1].replace(/(?:这个节点|这个实体|的节点)$/, '') : command).trim());
  return {kind: 'locate', term};
}

function prepareNodes(raw) {
  const byId = new Map();
  for (const node of (Array.isArray(raw) ? raw : []).slice(0, GRAPH_NAVIGATION_LIMITS.nodes)) {
    if (!node || typeof node !== 'object') continue;
    const id = text(node.id), label = text(node.label, 160);
    if (!id || !label || node.label.length > 160 || byId.has(id)) continue;
    // IDs are identifiers, not display text. Refuse truncation or sanitization
    // that could make two different identifiers refer to the same node.
    if (id !== node.id) continue;
    byId.set(id, {id, label, lookup: nameKey(label), type: text(node.type, 80), is_discovered: discovered(node.is_discovered)});
  }
  return {byId, nodes: [...byId.values()].sort((a, b) => compare(a.label, b.label) || compare(a.id, b.id)),
    limited: Array.isArray(raw) && raw.length > GRAPH_NAVIGATION_LIMITS.nodes};
}

function closeSpelling(left, right) {
  if (left.length > 64 || right.length > 64 || left.length < 3) return false;
  const threshold = left.length >= 7 ? 2 : 1;
  if (Math.abs(left.length - right.length) > threshold) return false;
  let previous = Array.from({length: right.length + 1}, (_, index) => index);
  for (let i = 1; i <= left.length; i += 1) {
    const row = [i];
    for (let j = 1; j <= right.length; j += 1) row[j] = Math.min(row[j - 1] + 1, previous[j] + 1, previous[j - 1] + (left[i - 1] === right[j - 1] ? 0 : 1));
    if (Math.min(...row) > threshold) return false;
    previous = row;
  }
  return previous[right.length] <= threshold;
}

function matchingNodes(term, nodes) {
  const normalized = nameKey(term);
  if (!normalized) return {exact: false, nodes: []};
  const exact = nodes.filter(node => node.lookup === normalized);
  if (exact.length) return {exact: true, nodes: exact};
  // Substrings and spelling suggestions are *always* presented for confirmation,
  // even if there is only one candidate. Descriptions never participate.
  return {exact: false, nodes: nodes.filter(node => {
    const label = node.lookup;
    return (normalized.length >= 2 && label.includes(normalized)) || closeSpelling(normalized, label);
  })};
}

function resolveNode(term, slot, nodes, resolutions, intent, limited = false) {
  const matches = matchingNodes(term, nodes);
  if (!matches.nodes.length) return {result: limited
    ? none('search_limit', '当前图谱超过本地节点搜索预算，已检查部分节点但未匹配；请缩小范围或直接在列表中选择。', {limited: true})
    : none('node_not_found', `当前图谱里没有找到“${term}”。请使用节点名称，或先在列表中选择。`)};
  const candidates = matches.nodes.slice(0, GRAPH_NAVIGATION_LIMITS.candidates).map(({id, label, type}) => ({id, label, type}));
  const requested = resolutions && typeof resolutions === 'object' && Object.hasOwn(resolutions, slot) ? resolutions[slot] : undefined;
  const chosen = candidates.find(node => node.id === requested);
  if (chosen) return {node: matches.nodes.find(node => node.id === chosen.id)};
  if (matches.exact && matches.nodes.length === 1 && !limited) return {node: matches.nodes[0]};
  return {result: {kind: 'clarify', intent, slot, candidates, totalCandidates: matches.nodes.length,
    hasMore: matches.nodes.length > candidates.length, limited,
    message: `“${term}”${limited ? '只在部分已检查节点中找到候选' : matches.exact ? '有重名节点' : '没有唯一精确匹配'}，请明确选择${slot === 'source' ? '起点' : slot === 'target' ? '终点' : '节点'}。${matches.nodes.length > candidates.length ? '仅列前 8 项，请补充更完整的名称缩小范围。' : ''}`}};
}

function resolveRelationSplit(splits, nodes) {
  // Entity names can themselves contain 和/与/跟. Prefer the unique split whose
  // two names exactly match nodes; otherwise refuse multiple plausible parses.
  const ranked = splits.map(split => {
    const source = matchingNodes(split.source, nodes), target = matchingNodes(split.target, nodes);
    return {split, viable: source.nodes.length > 0 && target.nodes.length > 0, exact: Number(source.exact) + Number(target.exact)};
  });
  const viable = ranked.filter(item => item.viable);
  if (!viable.length) return splits.length === 1 ? splits[0] : null;
  const best = Math.max(...viable.map(item => item.exact)), choices = viable.filter(item => item.exact === best);
  return choices.length === 1 ? choices[0].split : null;
}

function prepareEdges(raw, byId) {
  const adjacency = new Map(), edges = [];
  for (const [index, edge] of (Array.isArray(raw) ? raw : []).slice(0, GRAPH_NAVIGATION_LIMITS.edges).entries()) {
    if (!edge || typeof edge !== 'object' || !byId.has(edge.source) || !byId.has(edge.target) || edge.source === edge.target) continue;
    const key = typeof edge.key === 'string' && edge.key.length <= 256 ? edge.key : Number.isSafeInteger(edge.key) ? edge.key : `edge-${index}`;
    const item = {key, source: edge.source, target: edge.target, description: text(edge.description, 800),
      descriptionTruncated: typeof edge.description === 'string' && edge.description.length > 800,
      is_discovered: discovered(edge.is_discovered)};
    edges.push(item);
  }
  edges.sort((a, b) => Number(a.is_discovered) - Number(b.is_discovered) || compare(a.source, b.source) || compare(a.target, b.target) || compare(String(a.key), String(b.key)) || compare(a.description, b.description));
  for (const edge of edges) {
    for (const [from, to] of [[edge.source, edge.target], [edge.target, edge.source]]) {
      if (!adjacency.has(from)) adjacency.set(from, []);
      adjacency.get(from).push({to, edge});
    }
  }
  return {adjacency, hasDiscovered: edges.some(edge => edge.is_discovered), limited: Array.isArray(raw) && raw.length > GRAPH_NAVIGATION_LIMITS.edges};
}

function pathSearch(source, target, adjacency, originalsOnly, budget) {
  const queue = [{id: source, nodeIds: [source], edges: []}], seen = new Set([source]);
  for (let index = 0; index < queue.length; index += 1) {
    if (budget.nodes >= GRAPH_NAVIGATION_LIMITS.visitedNodes) { budget.exhausted = true; return null; }
    budget.nodes += 1;
    const current = queue[index];
    if (current.edges.length >= GRAPH_NAVIGATION_LIMITS.hops) continue;
    for (const {to, edge} of adjacency.get(current.id) || []) {
      if (budget.edges >= GRAPH_NAVIGATION_LIMITS.traversedEdges) { budget.exhausted = true; return null; }
      budget.edges += 1;
      if ((originalsOnly && edge.is_discovered) || seen.has(to)) continue;
      const next = {id: to, nodeIds: [...current.nodeIds, to], edges: [...current.edges, edge]};
      if (to === target) return {nodeIds: next.nodeIds, edges: next.edges};
      // Bound queue memory too, not just the number of nodes eventually popped.
      if (seen.size >= GRAPH_NAVIGATION_LIMITS.visitedNodes) { budget.exhausted = true; return null; }
      seen.add(to); queue.push(next);
    }
  }
  return null;
}

/**
 * Plan a pure, bounded local UI action. Re-run the same query after a user picks
 * an explicit candidate with resolutions: {node: id} or {source: id, target: id}.
 * No model, network, graph mutation, inferred edge creation, or evidence reads.
 */
export function planGraphNavigation(options = {}) {
  const {query, nodes: rawNodes, edges: rawEdges, selectedId = '', resolutions = {}} = options && typeof options === 'object' ? options : {};
  if (typeof query !== 'string' || !query.trim()) return none('empty_query', '说出或输入一个节点名称，也可以问两个节点有什么联系。');
  if (query.length > GRAPH_NAVIGATION_LIMITS.queryChars) return none('query_too_long', '定位指令请控制在 256 字以内；长问题请在助手中讨论。');
  const cleanQuery = text(query);
  let intent = parseIntent(cleanQuery);
  if (intent.kind === 'back') return {kind: 'back', message: '返回上一步的图谱位置。'};
  const graph = prepareNodes(rawNodes), {nodes, byId} = graph;
  if (intent.kind === 'source') return byId.has(selectedId)
    ? {kind: 'source', nodeId: selectedId, message: '查看当前节点的原文；图谱描述本身不作为原文证据。'}
    : none('selection_required', '先选择一个节点，再查看它的原文。');
  if (!nodes.length) return none('empty_graph', '当前图谱还没有可定位的节点。');
  const literalName = unquote(cleanQuery.normalize('NFKC').replace(/[。！？!?]+$/g, '').trim());
  if (matchingNodes(literalName, nodes).exact) intent = {kind: 'locate', term: literalName};
  if (intent.kind === 'locate') {
    const resolved = resolveNode(intent.term, 'node', nodes, resolutions, 'locate', graph.limited);
    return resolved.result || {kind: 'select', nodeId: resolved.node.id, message: `已定位“${resolved.node.label}”。`, limited: graph.limited};
  }
  const split = resolveRelationSplit(intent.splits, nodes);
  if (!split) return none('ambiguous_expression', '这句话里的节点边界不够明确，请使用“节点 A 与节点 B 有什么联系”的完整名称。');
  const source = resolveNode(split.source, 'source', nodes, resolutions, 'relation', graph.limited);
  if (source.result) return source.result;
  const target = resolveNode(split.target, 'target', nodes, resolutions, 'relation', graph.limited);
  if (target.result) return target.result;
  if (source.node.id === target.node.id) return {kind: 'select', nodeId: source.node.id, message: '两端指向同一节点，已定位；这不是两个实体之间的关系。', limited: graph.limited};
  const prepared = prepareEdges(rawEdges, byId), budget = {nodes: 0, edges: 0, exhausted: false};
  let path = pathSearch(source.node.id, target.node.id, prepared.adjacency, true, budget);
  if (!path && !budget.exhausted && prepared.hasDiscovered) path = pathSearch(source.node.id, target.node.id, prepared.adjacency, false, budget);
  const limited = graph.limited || prepared.limited || budget.exhausted;
  if (!path) return none(limited ? 'search_limit' : 'path_not_found', limited
    ? '已达到本地图谱搜索预算，暂未找到 3 跳以内的路径；不代表两者没有关系。可缩小范围或直接选择节点查看。'
    : '当前图谱没有找到 3 跳以内的现有路径；不代表两者没有关系，也不会自动补造关系。', {sourceId: source.node.id, targetId: target.node.id, limited});
  const containsDiscovered = path.edges.some(edge => edge.is_discovered) || path.nodeIds.some(id => byId.get(id).is_discovered);
  return {kind: 'path', sourceId: source.node.id, targetId: target.node.id,
    path: {...path, requiresVerification: true, containsDiscovered}, limited,
    message: `找到 ${path.edges.length} 跳现有无向联系，不代表因果证明；请逐条核对原文。${containsDiscovered ? '其中包含补充线索，待核验。' : ''}`};
}
