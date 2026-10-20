<template>
  <view class="mobile-page graph-page">
    <BrandHeader eyebrow="CONNECTIONS" title="把思路，接回来" subtitle="说出想找的内容，看清它们如何相连。" />
    <ModeBanner :mode="preferences.mode" :connection-kind="preferences.connectionKind" :invite-pending="preferences.invitePending" />
    <CapabilityBoundary v-if="preferences.connectionKind === 'invite'" feature-id="graph" :invite-pending="preferences.invitePending" />
    <template v-else>
    <WorkspaceBar :sessions="sessions" :session-id="sessionId" :disabled="loading || voiceBusy" @change="changeSession" />
    <view class="graph-controls"><text>当前会话 · 轻量证据视图</text><button class="quiet-button" :disabled="loading || voiceBusy" @tap="refreshGraph">{{ loading ? '读取中' : '刷新图谱 ↻' }}</button></view>
    <view v-if="error" class="error-box">{{ error }}</view>
    <view v-if="loading" class="status-box">正在读取当前会话的图谱…</view>
    <view v-else-if="!sessionId || !nodes.length" class="empty-state"><text class="empty-title">{{ sessionId ? '图谱还没有内容' : '先选择一个会话' }}</text><text class="empty-copy">上传并解析资料后，实体与关系会出现在这里。不同会话的资料不会合并显示。</text><button class="secondary-button" @tap="openDocuments">去资料页</button></view>
    <view v-else>
      <view class="graph-summary"><text>{{ nodes.length }} 个实体 · {{ edgeCount }} 条关系</text><text>关联不等于因果</text></view>
      <view class="graph-locator">
        <text class="locator-eyebrow">不必记住刚才看到哪</text><text class="locator-title">想找什么，直接说</text>
        <text class="locator-copy">找一个知识点，或看看两个知识点之间的联系。语音先转文字，检查后再定位。</text>
        <textarea :key="locatorInputRevision" v-model="locatorDraft" class="locator-input" aria-label="图谱定位指令" placeholder="例如：政策工具和国内供给增加有什么联系？" :maxlength="-1" :auto-height="true" :disabled="loading || voiceBusy" :show-confirm-bar="false" @keydown="locatorKeydown" />
        <view v-if="pendingVoiceDraft" class="voice-draft-choice"><text>转写已完成，原定位文字还在。是否替换为：</text><text class="voice-draft-preview">{{ pendingVoiceDraft }}</text><view class="inline-actions"><button class="quiet-button" :disabled="navigationBlocked" @tap="resolveVoiceDraft(true)">使用这次转写</button><button class="quiet-button" @tap="resolveVoiceDraft(false)">保留原文字</button></view></view>
        <view class="locator-actions"><GraphVoiceInput :active="pageVisible" :disabled="loading || evidenceLoading || Boolean(pendingVoiceDraft)" :reset-key="String(navigationRevision)" @transcript="receiveLocatorTranscript" @busy="voiceBusy = $event" /><button class="primary-button locator-submit" :disabled="navigationBlocked || !locatorDraft.trim() || locatorTooLong || Boolean(pendingVoiceDraft)" role="button" tabindex="0" @tap="runLocator()" @keydown="navigationKeydown($event, 'locate')">定位 / 看联系 ↗</button></view>
        <text v-if="locatorTooLong" class="locator-warning">定位指令最多 {{ GRAPH_NAVIGATION_LIMITS.queryChars }} 字符，文字已完整保留，请缩短后再定位。</text>
        <text v-if="locatorNotice" class="locator-copy" role="status" aria-live="polite">{{ locatorNotice }}</text>
        <text class="locator-caption">仅在已加载图谱内定位，不生成回答。支持“找到某实体”“A和B有什么联系”“回到上一步”“看原文”。</text>
      </view>
      <view v-if="contextGoal || selected || navigationHistory.length" id="graph-context" class="graph-context">
        <view class="context-heading"><text class="context-eyebrow">这次思路</text><button class="quiet-button" :disabled="navigationBlocked || !navigationHistory.length" role="button" tabindex="0" @tap="goBack" @keydown="navigationKeydown($event, 'back')">← 回到上一步</button></view>
        <text class="context-question">{{ contextGoal || '从一个知识点开始探索' }}</text>
        <view class="context-current"><text class="context-dot" aria-hidden="true">●</text><text>现在看：{{ currentLocation }}</text></view>
        <text v-if="historyLabels.length" class="context-history">刚才看过：{{ historyLabels.join(' / ') }}（浏览记录，不代表关系链）</text>
        <text class="context-boundary">本页保留最近 12 步；切会话或离开页面后清空，不写入长期记忆。</text>
        <button v-if="selected" class="quiet-button context-source" :disabled="voiceBusy" role="button" tabindex="0" @tap="showSelectedEvidence" @keydown="navigationKeydown($event, 'source')">核对当前知识点的原文 ↓</button>
      </view>
      <view v-if="locatorResult" class="locator-result" role="status" aria-live="polite">
        <text class="locator-result-note">{{ locatorResult.message }}</text>
        <view v-if="locatorResult.kind === 'clarify' && locatorQuery === locatorDraft" class="locator-candidates"><button v-for="candidate in locatorResult.candidates" :key="candidate.id" class="locator-candidate" :disabled="navigationBlocked || Boolean(pendingVoiceDraft)" @tap="chooseCandidate(candidate.id)"><text>{{ candidate.label }}</text><text class="candidate-detail">{{ candidate.type || '实体' }} · {{ candidate.id }}</text></button></view>
        <view v-if="locatorResult.kind === 'path'" class="connection-route">
          <view v-for="(node,index) in routeNodes" :key="node.id" class="route-stop">
            <button class="route-node" :class="{'route-node-current':selected?.id === node.id}" :disabled="voiceBusy" role="button" tabindex="0" @tap="selectNode(node)" @keydown="selectNodeKeydown($event, node)"><text class="route-number">{{ index + 1 }}</text><view><text class="route-label">{{ node.label }}</text><text class="route-node-note">{{ node.is_discovered ? '补充实体 · 待核验' : '点开核对这个实体的原文' }}</text></view><text aria-hidden="true">↗</text></button>
            <view v-if="locatorResult.path.edges[index]" class="route-edge" :class="{'route-edge-inferred':locatorResult.path.edges[index].is_discovered}"><text class="route-edge-label">{{ locatorResult.path.edges[index].is_discovered ? '补充关联 · 待核验' : '现有关联 · 非因果证明' }}</text><text>{{ locatorResult.path.edges[index].description || '未提供关系说明，需核对原文。' }}</text><text v-if="locatorResult.path.edges[index].descriptionTruncated" class="route-edge-label">关系说明已缩短，完整内容请核对原文。</text></view>
          </view>
          <text class="locator-caption">只展示本次找到的一条 3 跳以内的联系。实体原文不一定支持每条边；未找到不代表不存在。</text>
        </view>
      </view>
      <view class="graph-search"><text class="search-symbol" aria-hidden="true">⌕</text><input v-model="search" class="graph-search-input" placeholder="搜索实体名称或类型" :maxlength="120" /><button v-if="search" class="clear-search" aria-label="清空搜索" @tap="search = ''">×</button></view>
      <view class="section-heading"><text class="section-title">探索一个知识点</text><text class="muted-text">{{ filteredNodes.length }} 个</text></view>
      <view v-if="!filteredNodes.length" class="status-box">没有匹配实体，试试更短的名称。</view>
      <scroll-view v-else class="node-list" scroll-y>
        <button v-for="node in displayedNodes" :key="node.id || node.key" class="node-row" :class="{ 'node-selected': selected?.id === node.id }" :disabled="!node.id || voiceBusy" @tap="selectNode(node)">
          <view class="node-monogram" aria-hidden="true">{{ Array.from(node.label)[0] }}</view><view class="node-row-main"><text class="node-name">{{ node.label }}</text><view class="node-tags"><text class="node-kind">{{ node.type || '实体' }}</text><text v-if="node.is_discovered" class="node-inferred">补充 · 待核验</text></view></view><text class="node-arrow" aria-hidden="true">{{ selected?.id === node.id ? '✓' : '↗' }}</text>
        </button>
      </scroll-view>
      <button v-if="filteredNodes.length > visibleCount" class="quiet-button full-width" @tap="visibleCount += 60">再显示 60 个实体（还有 {{ filteredNodes.length - visibleCount }} 个）</button>
      <button v-if="selected" class="selected-evidence-link" role="button" @tap="showSelectedEvidence"><text class="selected-evidence-name">已展开 · {{ selected.label }}</text><text class="selected-evidence-action">查看关系与原文 ↓</text></button>
      <view v-if="selected" id="knowledge-evidence" class="evidence-section">
        <view class="section-heading evidence-heading"><view class="evidence-title-block"><text class="evidence-eyebrow">KNOWLEDGE DETAIL</text><text class="section-title">{{ selected.label }}</text></view><button class="quiet-button" :disabled="voiceBusy" @tap="collapseEvidence">收起</button></view>
        <view class="evidence-context"><text>{{ contextGoal ? '围绕：' + contextGoal : '当前：' + selected.label }}</text><button class="quiet-button" :disabled="voiceBusy" role="button" tabindex="0" @tap="showContext" @keydown="navigationKeydown($event, 'context')">回看问题与联系 ↑</button></view>
        <!-- #ifdef MP-WEIXIN -->
        <text v-if="selected.description" class="node-description" user-select>{{ selected.description }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text v-if="selected.description" class="node-description" selectable>{{ selected.description }}</text>
        <!-- #endif -->
        <view class="relations-section"><view class="relation-heading"><text class="relation-title">它与哪些知识相连</text><text class="relation-count">{{ selectedRelations.length }} 条</text></view><text class="relation-note">线条表示接口返回的邻接关系，不表示因果方向；补充关联需核验。</text>
          <view v-if="!selectedRelations.length" class="relation-empty">当前摘要没有返回它的邻接关系。</view>
          <view v-for="relation in displayedRelations" :key="relation.key" class="relation-card" :class="{ 'relation-inferred': relation.is_discovered }"><view class="relation-endpoints"><text class="relation-origin">{{ selected.label }}</text><text class="relation-line" aria-hidden="true">—</text><button class="relation-target" :disabled="!relation.node" @tap="selectNeighbor(relation)">{{ relation.label }}<text v-if="relation.node" aria-hidden="true"> ↗</text></button></view><text class="relation-kind">{{ relation.is_discovered ? '补充关联 · 待核验' : '图谱关联 · 请核原文' }}</text>
            <!-- #ifdef MP-WEIXIN -->
            <text v-if="relation.description" class="relation-description" user-select>{{ relation.description }}</text>
            <!-- #endif -->
            <!-- #ifndef MP-WEIXIN -->
            <text v-if="relation.description" class="relation-description" selectable>{{ relation.description }}</text>
            <!-- #endif -->
          </view>
          <button v-if="selectedRelations.length > relationLimit" class="quiet-button full-width" @tap="relationLimit += 8">再看 8 条关联（还有 {{ selectedRelations.length - relationLimit }} 条）</button>
        </view>
        <view class="source-heading"><text class="source-title">回到原文</text><text class="source-badge">EVIDENCE</text></view><text class="evidence-caption">实体描述、关系标签都不能代替原文。以下是该实体关联的片段，不一定支持每一条关系。</text>
        <view v-if="evidenceLoading" class="status-box">正在查找这个实体对应的原文…</view>
        <view v-else-if="evidenceError" class="error-box"><text>{{ evidenceError }}</text><button class="quiet-button" @tap="selectNode(selected)">重新读取</button></view>
        <view v-else-if="!chunks.length" class="status-box">暂未返回可查看的原文。这个节点本身不足以证明结论，请到桌面端继续核查。</view>
        <view v-for="(chunk, index) in chunks" :key="chunk.id || chunk.chunk_id || index" class="evidence-item"><text class="evidence-number">{{ String(index + 1).padStart(2, '0') }} / 关联原文{{ chunk.file_name ? ' · ' + chunk.file_name : '' }}</text>
          <!-- #ifdef MP-WEIXIN -->
          <text class="evidence-content" user-select>{{ chunk.content || chunk.text || '该条原文没有可显示的内容。' }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="evidence-content" selectable>{{ chunk.content || chunk.text || '该条原文没有可显示的内容。' }}</text>
          <!-- #endif -->
        </view>
      </view>
      <view v-else class="graph-explanation"><view class="exploration-mark" aria-hidden="true">↗</view><view><text class="exploration-title">沿着一个知识点，继续探索</text><text>点选实体，查看邻接关系与原文。当前是轻量证据视图，不展示完整推理链。</text></view></view>
    </view>
    </template>
  </view>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue';
import { onHide, onShow, onUnload } from '@dcloudio/uni-app';
import { api, getPreferences, getActiveSession, setActiveSession } from '../../services/api.js';
import ModeBanner from '../../components/ModeBanner.vue';
import WorkspaceBar from '../../components/WorkspaceBar.vue';
import BrandHeader from '../../components/BrandHeader.vue';
import CapabilityBoundary from '../../components/CapabilityBoundary.vue';
import GraphVoiceInput from '../../components/GraphVoiceInput.vue';
import { planGraphNavigation, GRAPH_NAVIGATION_LIMITS } from '../../services/graph-navigation.js';

const preferences = ref({ mode: 'demo' }), sessions = ref([]), sessionId = ref('');
const nodes = ref([]), edges = ref([]), search = ref(''), selected = ref(null), chunks = ref([]), visibleCount = ref(60), relationLimit = ref(8);
const edgeCount = computed(() => edges.value.length);
const loading = ref(false), evidenceLoading = ref(false), error = ref(''), evidenceError = ref('');
const pageVisible = ref(false), voiceBusy = ref(false), navigationRevision = ref(0), locatorInputRevision = ref(0);
const locatorDraft = ref(''), pendingVoiceDraft = ref(''), locatorNotice = ref(''), locatorResult = ref(null), locatorQuery = ref(''), choiceResolutions = ref({});
const contextGoal = ref(''), navigationHistory = ref([]);
const locatorTooLong = computed(() => locatorDraft.value.length > GRAPH_NAVIGATION_LIMITS.queryChars);
const navigationBlocked = computed(() => !pageVisible.value || loading.value || voiceBusy.value || !sessionId.value || !nodes.value.length || preferences.value.connectionKind === 'invite');
const routeNodes = computed(() => (locatorResult.value?.path?.nodeIds || []).map(id => nodes.value.find(node => node.id === id)).filter(Boolean));
const currentLocation = computed(() => selected.value?.label || (locatorResult.value?.kind === 'path' ? routeNodes.value.map(node => node.label).join(' — ') : locatorResult.value?.kind === 'clarify' ? '等待你确认实体' : '还未选定知识点'));
const historyLabels = computed(() => navigationHistory.value.slice(-4).map(item => item.label));
let generation = 0, evidenceGeneration = 0, visible = false;
let scrollGeneration = 0;
const filteredNodes = computed(() => { const query = search.value.trim().toLocaleLowerCase(); return nodes.value.filter(node => !query || `${node.label} ${node.type}`.toLocaleLowerCase().includes(query)); });
const displayedNodes = computed(() => filteredNodes.value.slice(0, visibleCount.value));
const selectedRelations = computed(() => {
  if (!selected.value) return [];
  const id = selected.value.id, byId = new Map(nodes.value.map(node => [node.id, node]));
  return edges.value.filter(edge => edge.source === id || edge.target === id).map(edge => {
    const neighborId = edge.source === id ? edge.target : edge.source, node = byId.get(neighborId);
    return { ...edge, node, label: node?.label || neighborId };
  });
});
const displayedRelations = computed(() => selectedRelations.value.slice(0, relationLimit.value));
const discovered = value => value === true || value === 1 || value === 'true';
function endpointId(value) { return String(value && typeof value === 'object' ? value.id ?? value.entity_id ?? '' : value ?? ''); }
function selectNeighbor(relation) { if (relation.node) return selectNode(relation.node); }
watch(search, () => { visibleCount.value = 60; });
function clearSelection() { evidenceGeneration += 1; selected.value = null; chunks.value = []; relationLimit.value = 8; evidenceError.value = ''; evidenceLoading.value = false; }
async function revealEvidence(request, pageGeneration, session, nodeId) {
  const scroll = ++scrollGeneration;
  await nextTick();
  if (!visible || scroll !== scrollGeneration || pageGeneration !== generation || request !== evidenceGeneration || session !== sessionId.value || selected.value?.id !== nodeId) return;
  if (typeof uni.pageScrollTo === 'function') {
    try { uni.pageScrollTo({selector:'#knowledge-evidence',duration:180}); }
    catch { /* An unavailable scroll API must not interfere with evidence reads. */ }
  }
}
function showSelectedEvidence() {
  if (!visible || voiceBusy.value || !selected.value?.id || preferences.value.connectionKind === 'invite') return;
  void revealEvidence(evidenceGeneration, generation, sessionId.value, selected.value.id);
}
async function showContext() {
  if (navigationBlocked.value || (!contextGoal.value && !selected.value && !navigationHistory.value.length)) return;
  const scroll = ++scrollGeneration, page = generation;
  await nextTick();
  if (!visible || scroll !== scrollGeneration || page !== generation) return;
  try { uni.pageScrollTo?.({selector:'#graph-context',duration:180}); } catch { /* The context card remains accessible without scrolling. */ }
}
function resetNavigation() {
  navigationRevision.value += 1; scrollGeneration += 1; voiceBusy.value = false;
  locatorDraft.value = ''; pendingVoiceDraft.value = ''; locatorNotice.value = ''; locatorResult.value = null; locatorQuery.value = ''; choiceResolutions.value = {};
  contextGoal.value = ''; navigationHistory.value = [];
}
function rememberLocation() {
  if (!selected.value && !locatorResult.value && !contextGoal.value) return;
  const snapshot = {nodeId:selected.value?.id || '',result:locatorResult.value,query:locatorQuery.value,goal:contextGoal.value,label:currentLocation.value};
  const last = navigationHistory.value.at(-1);
  if (last?.nodeId === snapshot.nodeId && last?.result === snapshot.result && last?.goal === snapshot.goal) return;
  navigationHistory.value = [...navigationHistory.value.slice(-11),snapshot];
}
function sameLocation(plan, query) {
  const current = locatorResult.value;
  if (!current || contextGoal.value !== query.trim() || current.kind !== plan.kind) return false;
  if ((selected.value?.id || '') !== (plan.kind === 'select' ? plan.nodeId : '')) return false;
  return current.nodeId === plan.nodeId && current.slot === plan.slot && current.reason === plan.reason
    && JSON.stringify(current.path?.nodeIds || []) === JSON.stringify(plan.path?.nodeIds || [])
    && JSON.stringify(current.candidates || []) === JSON.stringify(plan.candidates || []);
}
async function runLocator(resolutions = {}, remember = true) {
  if (navigationBlocked.value || pendingVoiceDraft.value) return;
  if (locatorTooLong.value) { locatorNotice.value = `文字已保留，请缩短到 ${GRAPH_NAVIGATION_LIMITS.queryChars} 字符以内再定位。`; return; }
  const query = locatorDraft.value;
  const plan = planGraphNavigation({query,nodes:nodes.value,edges:edges.value,selectedId:selected.value?.id || '',resolutions});
  if (plan.kind === 'back') return goBack();
  if (plan.kind === 'source') { locatorNotice.value = '已定位到当前实体的原文区域；实体片段不等于每条关系的证据。'; return showSelectedEvidence(); }
  if (sameLocation(plan,query)) { locatorNotice.value = '当前已经是这个位置；可以点节点核对原文。'; void showContext(); return; }
  if (remember) rememberLocation();
  contextGoal.value = query.trim(); locatorQuery.value = query; choiceResolutions.value = {...resolutions}; locatorResult.value = plan; locatorNotice.value = '';
  clearSelection();
  if (plan.kind === 'select') {
    const node = nodes.value.find(item => item.id === plan.nodeId);
    if (node) await selectNode(node, false);
  } else { void showContext(); }
}
function chooseCandidate(id) {
  const plan = locatorResult.value;
  if (navigationBlocked.value || plan?.kind !== 'clarify' || locatorQuery.value !== locatorDraft.value || !plan.candidates.some(item => item.id === id)) return;
  return runLocator({...choiceResolutions.value,[plan.slot]:id},false);
}
function receiveLocatorTranscript(text) {
  if (!visible || loading.value || preferences.value.connectionKind === 'invite' || typeof text !== 'string' || !text.trim()) return;
  if (locatorDraft.value.trim() && locatorDraft.value !== text) pendingVoiceDraft.value = text;
  else locatorDraft.value = text;
  locatorNotice.value = '转写已完成，请检查文字，再点“定位 / 看联系”；尚未查询或发送问题。';
}
function resolveVoiceDraft(replace) {
  if (!visible || (replace && navigationBlocked.value)) return;
  if (replace) locatorDraft.value = pendingVoiceDraft.value;
  pendingVoiceDraft.value = ''; locatorNotice.value = replace ? '已使用转写文字，检查后再定位。' : '已保留原定位文字。';
}
async function goBack() {
  if (navigationBlocked.value || !navigationHistory.value.length) { if (visible) locatorNotice.value = '本页还没有可返回的位置。'; return; }
  const previous = navigationHistory.value.at(-1); navigationHistory.value = navigationHistory.value.slice(0,-1);
  clearSelection(); locatorResult.value = previous.result; locatorQuery.value = previous.query; locatorDraft.value = previous.query; contextGoal.value = previous.goal;
  // uni-app H5 can skip restoring a focused input to its last user-entered
  // value after a programmatic voice draft; remount only on explicit Back.
  locatorInputRevision.value += 1;
  pendingVoiceDraft.value = ''; locatorNotice.value = ''; choiceResolutions.value = {};
  const node = nodes.value.find(item => item.id === previous.nodeId);
  if (node) await selectNode(node,false); else void showContext();
}
function collapseEvidence() { if (voiceBusy.value) return; rememberLocation(); clearSelection(); void showContext(); }
function keyboardAction(event) {
  if (event?.defaultPrevented || event?.isComposing || event?.keyCode === 229 || event?.repeat || event?.ctrlKey || event?.metaKey || event?.altKey || event?.shiftKey) return false;
  if (!['Enter',' ','Space','Spacebar'].includes(event?.key)) return false;
  event.preventDefault?.(); return true;
}
function navigationKeydown(event, action) {
  if (!keyboardAction(event)) return;
  const run = {locate:()=>runLocator(),back:goBack,source:showSelectedEvidence,context:showContext}[action]; run?.();
}
function selectNodeKeydown(event,node) { if (keyboardAction(event)) void selectNode(node); }
function locatorKeydown(event) {
  if (!event?.defaultPrevented && !event?.isComposing && event?.keyCode !== 229 && !event?.repeat && !event?.altKey && !event?.shiftKey && event?.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault?.(); void runLocator(); }
}
function refreshGraph() { if (visible && !loading.value && !voiceBusy.value) return reload(); }
async function reload() {
  resetNavigation();
  const request = ++generation; preferences.value = getPreferences(); loading.value = true; error.value = ''; nodes.value = []; edges.value = []; clearSelection();
  sessions.value = []; sessionId.value = '';
  if (preferences.value.connectionKind === 'invite') { loading.value = false; return; }
  try {
    const available = await api.sessions();
    if (!visible || request !== generation) return;
    sessions.value = Array.isArray(available) ? available : [];
    const stored = String(getActiveSession() || '');
    sessionId.value = sessions.value.some(item => String(item.id) === stored) ? stored : String(sessions.value[0]?.id || '');
    setActiveSession(sessionId.value);
    if (!sessionId.value) return;
    const graph = await api.graph(sessionId.value);
    if (!visible || request !== generation) return;
    nodes.value = (Array.isArray(graph?.nodes) ? graph.nodes : []).map((node, index) => ({
      id: String(node.id ?? node.entity_id ?? ''), key: index,
      label: String(node.label || node.name || node.id || node.entity_id || '未命名实体'),
      type: String(node.type || node.entity_type || ''), description: String(node.description || ''), is_discovered: discovered(node.is_discovered),
    }));
    const relations = Array.isArray(graph?.edges) ? graph.edges : Array.isArray(graph?.links) ? graph.links : [];
    edges.value = relations.map((edge, index) => ({ key: index, source: endpointId(edge.source), target: endpointId(edge.target), description: String(edge.description || edge.label || edge.keywords || ''), is_discovered: discovered(edge.is_discovered) })).filter(edge => edge.source && edge.target);
  } catch (issue) { if (visible && request === generation) error.value = issue?.message || '图谱读取失败，请稍后刷新。'; }
  finally { if (request === generation) loading.value = false; }
}
async function selectNode(node, recordHistory = true) {
  if (!visible || voiceBusy.value || preferences.value.connectionKind === 'invite' || !node?.id || !sessionId.value) return;
  const canonical = nodes.value.find(item => item.id === node.id);
  if (!canonical) return;
  node = canonical;
  if (recordHistory && selected.value?.id !== node.id) rememberLocation();
  const request = ++evidenceGeneration, session = sessionId.value;
  selected.value = node; chunks.value = []; relationLimit.value = 8; evidenceError.value = ''; evidenceLoading.value = true;
  void revealEvidence(request, generation, session, node.id);
  try {
    const result = await api.chunks(session, node.id);
    if (visible && request === evidenceGeneration && session === sessionId.value) chunks.value = Array.isArray(result) ? result : [];
  } catch (issue) { if (visible && request === evidenceGeneration) evidenceError.value = issue?.message || '原文读取失败，请重试。'; }
  finally { if (request === evidenceGeneration) evidenceLoading.value = false; }
}
async function changeSession(id) { if (!visible || voiceBusy.value) return; search.value = ''; setActiveSession(id); await reload(); }
function openDocuments() { uni.switchTab({ url: '/pages/documents/index' }); }
function openChat() { uni.switchTab({ url: '/pages/chat/index' }); }
function hide() { visible = false; pageVisible.value = false; generation += 1; resetNavigation(); clearSelection(); loading.value = false; }
onShow(() => { visible = true; pageVisible.value = true; void reload(); }); onHide(hide); onUnload(hide);
</script>

<style>
@import '../../components/mobile.css';
.graph-controls { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: -12px; margin-bottom: 8px; color: #7d8b72; font-size: 11px; }
.graph-controls .quiet-button { min-height: 44px; padding-right: 0; color: #637e50; font-size: 12px; }
.graph-summary { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; margin-bottom:18px; color:#5f6d61; font-size:12px; }
.graph-locator { padding:22px; margin-bottom:18px; border:1px solid #cdd5be; border-radius:22px; background:#fffefa; }
.locator-eyebrow { display:block; font-size:11px; color:#657a52; letter-spacing:1px; }
.locator-title { display:block; margin-top:4px; color:#233d2c; font-size:23px; font-weight:650; letter-spacing:-.5px; }
.locator-copy,.locator-caption { display:block; font-size:12px; color:#5f6d60; line-height:1.85; margin-top:10px; overflow-wrap:anywhere; }
.locator-caption { font-size:11px; }
.locator-input { display:block; width:100%; min-height:82px; max-height:240px; margin:18px 0 12px; padding:14px; border:1px solid #dce0d3; border-radius:13px; background:#f5f6f0; color:#263b2d; font-size:16px; line-height:1.8; }
.locator-actions { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.locator-actions > view { flex:1; min-width:160px; }
.locator-submit { flex-shrink:0; font-size:13px; min-height:46px; padding:12px 16px; }
.locator-warning { display:block; margin-top:10px; color:#8d4f31; font-size:12px; line-height:1.8; }
.voice-draft-choice { margin:12px 0; padding:14px; background:#eff2e7; border:1px solid #dce5cd; border-radius:12px; color:#3f5337; font-size:12px; line-height:1.8; }
.voice-draft-preview { display:block; white-space:pre-wrap; overflow-wrap:anywhere; max-height:200px; overflow-y:auto; padding:8px 0; }
.graph-context { padding:18px 22px; margin:18px 0; border-radius:20px; background:#203d35; color:#f4f8ea; }
.context-heading { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.context-eyebrow { color:#c9dfad; font-size:11px; letter-spacing:2px; }
.context-heading .quiet-button,.graph-context .context-source { color:#e2efca; font-size:12px; }
.context-question { display:block; font-size:18px; font-weight:600; line-height:1.8; overflow-wrap:anywhere; margin:3px 0 10px; }
.context-current { display:flex; align-items:baseline; gap:8px; padding-top:12px; border-top:1px solid #4f6655; font-size:13px; color:#edf3e7; overflow-wrap:anywhere; }
.context-current > text:last-child { min-width:0; }
.context-dot { flex-shrink:0; font-size:9px; color:#d6fa75; }
.context-history,.context-boundary { display:block; margin-top:8px; font-size:11px; color:#c4d2c4; line-height:1.8; overflow-wrap:anywhere; }
.context-boundary { color:#b1c5b8; }
.locator-result { margin:18px 0; padding:18px; border:1px solid #dce3d1; border-radius:20px; background:#fffefa; }
.locator-result-note { display:block; color:#465d3d; font-size:13px; line-height:1.85; overflow-wrap:anywhere; }
.locator-candidates { display:grid; gap:8px; margin-top:12px; }
.locator-candidate { margin:0; padding:12px 14px; min-height:48px; border:1px solid #dce2d2; border-radius:12px; background:#f1f4ea; text-align:left; color:#2b4430; font-size:14px; line-height:1.7; overflow-wrap:anywhere; }
.candidate-detail { display:block; font-size:11px; color:#687660; }
.connection-route { margin-top:18px; }
.route-node { display:flex; gap:12px; align-items:center; width:100%; min-height:66px; padding:12px; margin:0; border:1px solid #dfe5d7; border-radius:14px; background:#f4f6ef; color:#244b32; text-align:left; line-height:1.6; }
.route-node > view { flex:1; min-width:0; }
.route-node-current { background:#e5efcf; border-color:#9db977; }
.route-number { display:flex; align-items:center; justify-content:center; flex-shrink:0; width:28px; height:28px; border-radius:50%; color:#405e34; background:#e2e9d8; font-size:11px; }
.route-label { display:block; font-size:15px; font-weight:600; overflow-wrap:anywhere; }
.route-node-note { display:block; color:#627453; font-size:11px; margin-top:3px; }
.route-edge { margin-left:26px; border-left:2px solid #b4c795; padding:12px 12px 16px 22px; font-size:12px; line-height:1.8; color:#53634a; overflow-wrap:anywhere; white-space:pre-wrap; }
.route-edge-label { display:block; font-size:10px; color:#657851; margin-bottom:3px; }
.route-edge-inferred { border-left-style:dashed; border-left-color:#b49a66; color:#806340; }
.route-edge-inferred .route-edge-label { color:#806340; }
.evidence-context { margin:14px 0; padding:12px 14px; border-left:3px solid #b2ca8e; background:#f1f4e9; font-size:12px; color:#4b6243; line-height:1.8; overflow-wrap:anywhere; }
.evidence-context .quiet-button { justify-content:flex-start; padding-left:0; font-size:12px; }
.graph-overview { padding: 20px 24px; margin-bottom: 22px; border: 1px solid #203d35; border-radius: 24px; color: #fff; background: #203d35; }
.graph-overview-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 20px; }
.graph-eyebrow { color: #d6e1ca; font-size: 12px; letter-spacing: 1px; }
.graph-overview-label { color: #9eb794; font-size: 8px; letter-spacing: 1px; }
.graph-metrics { display: flex; align-items: center; justify-content: space-between; padding: 6px 10px 16px; }
.graph-metric { min-width: 64px; }
.graph-metric-number { display: block; color: #e0f3b3; font-size: 44px; line-height: 1.2; font-weight: 500; letter-spacing: -2px; }
.graph-metric-label { display: block; color: #c2d1c0; font-size: 12px; margin-top: 7px; }
.metric-connector { display: flex; align-items: center; gap: 7px; color: #80997b; }
.metric-connector view { width: 12px; height: 12px; border: 1px solid #b1c99a; border-radius: 50%; }
.graph-overview-note { display: block; border-top: 1px solid #40584b; padding-top: 13px; color: #a7bfaa; font-size: 11px; }
.graph-search { display: flex; align-items: center; min-height: 50px; padding: 0 16px; border: 1px solid #e1e5d9; border-radius: 15px; background: #fff; }
.search-symbol { flex-shrink: 0; margin-right: 10px; color: #688163; font-size: 26px; }
.graph-search-input { flex: 1; width: 0; min-width: 0; height: 48px; color: #25372f; font-size: 14px; }
.clear-search { flex-shrink: 0; width: 44px; min-height: 44px; margin: 0 -10px 0 0; padding: 0; background: transparent; color: #6c8067; font-size: 24px; line-height: 44px; }
.node-list { height: 260px; max-height: 42vh; border: 1px solid #e4e7dd; border-radius: 20px; background: #fff; overflow: hidden; }
.node-row { display: flex; align-items: center; gap: 12px; width: 100%; min-height: 76px; margin: 0; padding: 15px 17px; border-radius: 0; border-bottom: 1px solid #f0f2ea; background: #fff; text-align: left; line-height: 1.5; }
.node-row.node-selected { background: #eaf1dc; }
.node-monogram { display: flex; align-items: center; justify-content: center; flex-shrink: 0; width: 36px; height: 36px; border-radius: 11px; background: #eff3e7; color: #5e7954; font-size: 15px; }
.node-selected .node-monogram { background: #d7e6bd; color: #203d35; }
.node-row-main { flex: 1; min-width: 0; }
.node-name { display: block; color: #25372f; font-size: 14px; font-weight: 550; word-break: break-word; }
.node-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.node-kind { color: #697960; font-size: 11px; }
.node-inferred { color: #8e743e; font-size: 10px; }
.node-arrow { flex-shrink: 0; color: #6f8860; font-size: 18px; }
.selected-evidence-link { display:flex; align-items:center; justify-content:space-between; gap:10px; width:100%; min-height:48px; margin:12px 0 0; padding:11px 14px; border:1px solid #dce5ce; border-radius:13px; background:#eaf1dc; color:#375535; text-align:left; line-height:1.7; }
.selected-evidence-name { min-width:0; font-size:12px; font-weight:600; overflow-wrap:anywhere; }
.selected-evidence-action { flex-shrink:0; font-size:11px; }
.evidence-section { margin-top: 24px; padding: 20px; border: 1px solid #e3e7da; border-radius: 24px; background: #fff; }
.evidence-heading { margin-top: 0; }
.evidence-title-block { min-width: 0; word-break: break-word; }
.evidence-eyebrow { display: block; margin-bottom: 7px; color: #90a17e; font-size: 9px; letter-spacing: 1.4px; }
.node-description { display: block; color: #66745f; font-size: 13px; line-height: 1.9; white-space: pre-wrap; word-break: break-word; }
.relations-section { margin: 24px 0; padding-top: 20px; border-top: 1px solid #edf0e6; }
.relation-heading, .source-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.relation-title, .source-title { color: #304632; font-size: 15px; font-weight: 600; }
.relation-count { color: #88987b; font-size: 11px; }
.relation-note, .evidence-caption { display: block; margin: 9px 0 14px; color: #68765f; font-size: 12px; line-height: 1.8; }
.relation-empty { padding: 16px; border-radius: 12px; background: #f6f7f1; color: #829174; font-size: 12px; }
.relation-card { padding: 14px; margin-top: 10px; border-radius: 14px; background: #f5f7f0; border: 1px solid #e9eddf; }
.relation-card.relation-inferred { background: #fbf7ed; border-style: dashed; border-color: #e8dfca; }
.relation-endpoints { display: flex; align-items: center; gap: 8px; }
.relation-origin { flex: 1; min-width: 0; color: #697b5b; font-size: 12px; word-break: break-word; }
.relation-line { flex-shrink: 0; color: #a1b393; }
.relation-target { flex: 1; min-width: 0; min-height: 44px; margin: -5px 0; padding: 8px; border-radius: 8px; color: #375535; background: #e9efdf; font-size: 12px; text-align: left; line-height: 1.5; word-break: break-word; }
.relation-target[disabled] { opacity: 1; color: #6d7865; background: #eef0e7; }
.relation-kind { display: block; margin-top: 10px; color: #667b54; font-size: 11px; }
.relation-inferred .relation-kind { color: #9a7b43; }
.relation-description { display: block; margin-top: 6px; color: #6a7960; font-size: 12px; line-height: 1.8; word-break: break-word; white-space: pre-wrap; }
.source-badge { color: #617b4f; font-size: 9px; letter-spacing: 1px; }
.evidence-item { border-left: 3px solid #c3da9d; padding: 2px 0 2px 14px; margin: 20px 0; }
.evidence-number { display: block; margin-bottom: 10px; color: #8c9d7b; font-size: 10px; letter-spacing: .5px; }
.evidence-content { display: block; color: #43523b; font-size: 13px; line-height: 1.9; white-space: pre-wrap; word-break: break-word; }
.graph-explanation { display: flex; align-items: flex-start; gap: 14px; margin-top: 22px; padding: 20px; border-radius: 20px; background: #e9eedf; color: #617354; font-size: 12px; line-height: 1.8; }
.exploration-mark { flex-shrink: 0; color: #7d9963; font-size: 26px; line-height: 1.3; }
.exploration-title { display: block; margin-bottom: 4px; color: #526c45; font-size: 14px; font-weight: 600; }
</style>
