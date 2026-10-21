<template>
  <view class="mobile-page memory-page">
    <BrandHeader eyebrow="MEMORY" title="让思考有记忆" subtitle="看看这段对话，留下了哪些线索。" />
    <ModeBanner :mode="preferences.mode" :connection-kind="preferences.connectionKind" :invite-pending="preferences.invitePending" :research-enabled="researchEnabled" />
    <view v-if="error && preferences.connectionKind === 'invite' && !researchEnabled" class="error-box">{{ error }}</view>
    <view v-if="loading && !capabilityReady" class="status-box">正在读取记忆能力…</view>
    <CapabilityBoundary v-else-if="preferences.connectionKind === 'invite' && !researchEnabled" feature-id="memory" :invite-pending="preferences.invitePending" />
    <template v-else>
    <WorkspaceBar :sessions="sessions" :session-id="sessionId" :disabled="loading || loadingMore || saving" @change="changeSession" />
    <MemoryObservatory v-if="researchEnabled || preferences.mode === 'demo' || (preferences.mode === 'connected' && preferences.connectionKind !== 'invite')" :session-id="sessionId" :active="capabilityReady" :enabled="researchEnabled || (preferences.mode === 'connected' && preferences.connectionKind !== 'invite')" :mode="researchEnabled ? 'research' : 'legacy'" :demo="preferences.mode === 'demo'" :disabled="loading || saving" :refresh-key="observationRevision" @source="readConfirmedSource" />
    <template v-if="researchEnabled">
      <view class="confirmed-intro"><text class="confirmed-eyebrow">YOU DECIDE WHAT STAYS</text><text class="confirmed-heading">记住你确认的，不替你下结论</text><text class="confirmed-copy">资料与回答不会自动变成记忆。你亲自保存或修改的内容，只在当前研究会话中使用；“已确认”代表由你确认保存，不代表已经证实。</text><button class="primary-button full-width" :disabled="loading || saving || !sessionId || !memoryWriteEnabled" @tap="startMemory">保存一条我确认的内容 <text aria-hidden="true">＋</text></button></view>
      <view v-if="error" class="error-box">{{ error }}</view><view v-if="notice" class="status-box">{{ notice }}</view>
      <view v-if="editorOpen" class="confirmed-editor"><view class="section-heading"><text class="section-title">{{ editingId ? '修正这条确认记忆' : '留下一个可靠的起点' }}</text><button class="quiet-button" :disabled="saving" @tap="cancelMemory">取消</button></view><text class="confirmed-copy">先写清楚你愿意在后续问题中保留的内容。不要把尚未验证的猜想写成事实。</text><text v-if="editorOrigin === 'answer_sources'" class="confirmed-copy">已带入该回答的来源编号，但没有自动复制或保存回答。请你写出值得确认的结论，再核对这些原文。</text><view class="confirmed-types"><button v-for="kind in ['finding','preference','decision']" :key="kind" class="confirmed-type" :class="{ 'confirmed-type-active': editorKind === kind }" :disabled="saving" @tap="editorKind = kind">{{ confirmedKind(kind) }}</button></view><textarea v-model="editorText" class="confirmed-textarea" :maxlength="-1" :disabled="saving" auto-height placeholder="例如：这项研究目前只观察到相关性，下一步需要验证因果方向。" /><text class="confirmed-counter">{{ Array.from(editorText).length }} / 2000 字符 · {{ editorSources.length ? `${editorSources.length} 个关联来源` : '没有原文关联，仅作为你确认的内容' }}</text><button v-for="id in editorSources" :key="id" class="quiet-button" :disabled="saving" @tap="readConfirmedSource(id)">核对关联原文</button><button v-if="editorSources.length" class="quiet-button" :disabled="saving" @tap="editorSources = []">去掉来源关联（保留内容草稿）</button><button class="primary-button full-width" :disabled="saving || !canSaveMemory" @tap="saveMemory">{{ saving ? '保存中…' : editingId ? '确认修正并保存' : '确认保存到本会话' }}</button></view>
      <view v-if="editorOpen" class="memory-validity"><text class="section-title">适用时间</text><text class="confirmed-copy">填写的是内容何时适用。每次修正的保存时间会单独记录，不会覆盖过去版本。</text><view class="memory-date-controls"><picker mode="date" :value="editorFrom" :disabled="saving" @change="editorFrom = $event.detail.value"><view class="memory-date-button">{{ editorFrom ? `开始：${editorFrom}` : '开始：保存时' }}</view></picker><picker mode="date" :value="editorUntil" :disabled="saving" @change="editorUntil = $event.detail.value"><view class="memory-date-button">{{ editorUntil ? `结束：${editorUntil} 当日结束` : '结束：未指定' }}</view></picker><button v-if="editorFrom || editorUntil" class="quiet-button" :disabled="saving" @tap="editorFrom = ''; editorUntil = ''">清空日期</button></view></view>
      <view class="section-heading"><text class="section-title">确认记忆 · {{ confirmedTotal }} 条</text><button class="quiet-button" :disabled="loading || saving || loadingMore" @tap="reload">刷新</button></view>
      <view class="memory-search"><text aria-hidden="true">⌕</text><input v-model="confirmedQuery" class="memory-search-input" :maxlength="500" placeholder="搜索整个项目的记忆" :disabled="loading || saving || loadingMore" @confirm="searchConfirmed" /><button class="quiet-button" :disabled="loading || saving || loadingMore" @tap="searchConfirmed">查找</button></view>
      <view class="memory-date-controls"><picker mode="date" :value="confirmedDate" :disabled="loading || saving || loadingMore" @change="selectConfirmedDate"><view class="memory-date-button">{{ confirmedDate ? `回看 ${confirmedDate} 当天` : '时间：现在' }}</view></picker><button v-if="confirmedDate" class="quiet-button" :disabled="loading || saving || loadingMore" @tap="clearConfirmedDate">回到现在</button><button class="quiet-button" :aria-pressed="showArchived" :disabled="loading || saving || loadingMore" @tap="toggleArchived">{{ showArchived ? '含已归档' : '显示已归档' }}</button></view>
      <text v-if="confirmedDate" class="memory-search-note">按设备时区回看到当天结束；已撤销来源不会因回看而恢复。</text>
      <view v-if="loading" class="status-box">正在读取当前会话的确认记忆…</view>
      <view v-else-if="!sessionId" class="empty-state memory-empty"><text class="empty-title">先选择研究会话</text><text class="empty-copy">确认内容只留在所属会话，不会混入其他人的研究。</text><button class="secondary-button" @tap="openChat">去助手页</button></view>
      <view v-else-if="!confirmedItems.length && !error" class="empty-state memory-empty"><view class="memory-empty-mark" aria-hidden="true">✳</view><text class="empty-title">把值得保留的，交给未来的你</text><text class="empty-copy">这里不会自动堆积模型推测。保存研究边界、偏好或已确认的进展，下次提问可以继续。</text></view>
      <view v-for="item in confirmedItems" :key="item.id" class="memory-card">
        <view class="memory-card-top"><text class="memory-kind">{{ confirmedKind(item.kind) }}</text><text class="memory-review confirmed-review">{{ item.status === 'invalidated' ? '来源已撤销 · 需重新确认' : '你已确认' }} · 版本 {{ item.revision }}</text></view>
        <!-- #ifdef MP-WEIXIN -->
        <text class="confirmed-memory-content" user-select>{{ item.redacted ? '关联来源已撤销，正文已隐藏。' : item.text }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="confirmed-memory-content" selectable>{{ item.redacted ? '关联来源已撤销，正文已隐藏。' : item.text }}</text>
        <!-- #endif -->
        <text v-if="item.status === 'invalidated'" class="confirmed-copy">{{ item.redacted ? '这条内容已停止使用。可以查看版本记录，或重新写一条经过核对的内容；回看不会恢复已撤销来源。' : '关联资料已移除，这条内容不再用于新问题。若仍要保留，可复制成无来源的新草稿，核对后明确确认；不会恢复旧来源。' }}</text>
        <text class="confirmed-copy">{{ item.source_ids.length ? `${item.source_ids.length} 个原文关联；相关性和结论仍需你核对。` : '无原文关联，不应当作资料已证实的事实。' }}</text>
        <text class="confirmed-copy">{{ item.archived ? '已归档 · 不用于新问题' : '' }}{{ validityLabel(item) }}</text>
        <view class="confirmed-item-actions"><button v-for="(id, index) in item.source_ids" :key="id" class="quiet-button" :disabled="saving" @tap="readConfirmedSource(id)">查看关联原文 {{ index + 1 }}</button><button class="quiet-button" :disabled="saving" @tap="openMemoryHistory(item)">版本与来历</button><button class="quiet-button" :disabled="saving || !memoryWriteEnabled || Boolean(confirmedDate)" @tap="editMemory(item)">{{ item.status === 'invalidated' ? '重新整理为草稿' : '修正' }}</button><button class="quiet-button" :disabled="saving || !memoryWriteEnabled || Boolean(confirmedDate)" @tap="archiveMemory(item)">{{ item.archived ? '恢复使用' : '归档' }}</button><button class="quiet-button confirmed-remove" :disabled="saving || !memoryWriteEnabled || Boolean(confirmedDate)" @tap="removeMemory(item)">删除</button></view>
      </view>
      <button v-if="confirmedCursor && !loading" class="secondary-button full-width memory-more" :disabled="loadingMore || saving" @tap="loadMoreConfirmed">{{ loadingMore ? '读取中…' : '继续读取记忆' }}</button>
      <view v-if="loadMoreError" class="error-box">{{ loadMoreError }}</view>
      <view v-if="memoryHistoryId" class="memory-history"><view class="section-heading"><text class="section-title">版本与来历</text><button class="quiet-button" :disabled="saving" @tap="closeMemoryHistory">收起</button></view><text class="confirmed-copy">保存时间记录你何时做了修正，适用时间记录内容何时有效。这里的旧版本仅供回看。</text><view v-for="version in memoryVersions" :key="version.revision" class="memory-history-version"><text class="memory-kind">版本 {{ version.revision }}</text><text class="confirmed-copy">{{ formatDate(version.recorded_at).replace('整理','保存') }}{{ validityLabel(version) }}</text><text class="confirmed-memory-content">{{ version.redacted ? '来源已撤销或内容已删除，不再展示该版本正文。' : version.text }}</text></view><button v-if="historyCursor" class="quiet-button" :disabled="saving" @tap="loadMoreHistory">继续读取旧版本</button></view>
      <view v-if="confirmedSource" class="memory-question">
        <view class="section-heading"><text class="section-title">关联原文</text><button class="quiet-button" @tap="confirmedSource = null">收起</button></view><text class="confirmed-copy">{{ confirmedSource.filename }}{{ confirmedSource.page ? ` · 第 ${confirmedSource.page} 页` : '' }}</text>
        <!-- #ifdef MP-WEIXIN -->
        <text class="confirmed-source-content" user-select>{{ confirmedSource.content }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="confirmed-source-content" selectable>{{ confirmedSource.content }}</text>
        <!-- #endif -->
        <text class="confirmed-copy">这是提取的原文片段，不是模型总结。来源编号：{{ confirmedSource.id }}</text>
      </view>
      <text class="memory-footer">保存、修正、删除和查看原文均不调用模型。删除后不再用于新问题，但历史回答中的旧文字不会自动改写；这不是整项研究的彻底数据擦除。</text>
    </template>
    <template v-else>
    <view class="memory-intro">
      <view class="memory-intro-heading"><text class="memory-intro-mark" aria-hidden="true">✳</text><text class="memory-intro-title">当前会话记忆</text><text class="memory-readonly">只读</text></view>
      <text class="memory-intro-copy">这里是助手从对话中整理的摘要，不是用户原话，也不是已经核实的事实。仅展示当前会话，不包含项目或用户共享记忆。</text>
    </view>
    <view class="section-heading memory-list-heading"><view><text class="section-title">已整理的线索</text><text class="memory-loaded">本次已加载 {{ items.length }} 条，不代表全部记忆</text></view><button class="quiet-button" :disabled="loading || loadingMore" @tap="reload">{{ loading ? '读取中' : '刷新' }}</button></view>
    <view v-if="error" class="error-box">{{ error }}</view>
    <view v-if="loading" class="status-box">正在读取当前会话的记忆…</view>
    <view v-else-if="!sessionId" class="empty-state memory-empty"><text class="empty-title">先选择一个会话</text><text class="empty-copy">在助手页新建会话，或从上方选择已有会话，再查看它的记忆。</text><button class="secondary-button" @tap="openChat">去助手页</button></view>
    <view v-else-if="!items.length && !error" class="empty-state memory-empty"><view class="memory-empty-mark" aria-hidden="true">✳</view><text class="empty-title">还没有整理好的记忆</text><text class="empty-copy">不是每次对话都会形成长期记忆。后台整理可能尚未完成，或当前会话还没有可展示的摘要；可稍后刷新。</text></view>
    <view v-if="!loading && items.length">
      <view class="memory-search"><text aria-hidden="true">⌕</text><input v-model="search" class="memory-search-input" :maxlength="120" placeholder="搜索已加载的摘要与提问" /><button v-if="search" class="memory-clear" aria-label="清空搜索" @tap="search = ''">×</button></view>
      <text v-if="search" class="memory-search-note">仅搜索已加载内容 · {{ filteredItems.length }} 条匹配</text>
      <view v-if="!filteredItems.length" class="status-box">没有匹配的已加载记忆，试试其他关键词。</view>
      <view v-for="item in filteredItems" :key="item.id" class="memory-card">
        <view class="memory-card-top"><text class="memory-kind">{{ kindLabel(item.kind) }}</text><text class="memory-review">自动整理 · 待核对</text></view>
        <text class="memory-content-label">助手整理摘要</text>
        <!-- #ifdef MP-WEIXIN -->
        <text class="memory-summary" user-select>{{ item.summary || '这条记忆没有可显示的摘要。' }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="memory-summary" selectable>{{ item.summary || '这条记忆没有可显示的摘要。' }}</text>
        <!-- #endif -->
        <text v-if="item.truncated_fields.includes('summary')" class="memory-truncated">摘要仅展示前 1500 个字符，并非完整内容。</text>
        <view class="memory-card-bottom"><text class="memory-date">{{ formatDate(item.created_at) }}</text><button v-if="item.question" class="quiet-button memory-source-toggle" :aria-expanded="expandedIds.includes(item.id)" @tap="toggleQuestion(item.id)">{{ expandedIds.includes(item.id) ? '收起提问 −' : '查看对应提问 ＋' }}</button></view>
        <view v-if="item.question && expandedIds.includes(item.id)" class="memory-question">
          <text class="memory-question-label">用户当时的提问 · 对话记录</text>
          <!-- #ifdef MP-WEIXIN -->
          <text class="memory-question-content" user-select>{{ item.question }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="memory-question-content" selectable>{{ item.question }}</text>
          <!-- #endif -->
          <text v-if="item.truncated_fields.includes('question')" class="memory-truncated">仅展示提问前 800 个字符，不能作为完整原话引用。</text><text class="memory-question-note">这是摘要对应的提问，不代表用户确认过上方摘要。</text>
        </view>
      </view>
    </view>
    <view v-if="loadMoreError" class="error-box">{{ loadMoreError }}</view>
    <button v-if="!loading && nextOffset !== null" class="secondary-button full-width memory-more" :disabled="loadingMore" @tap="loadMore">{{ loadingMore ? '正在读取…' : '再加载 20 条记忆' }}</button>
    <text v-if="windowLimited" class="memory-limit-note">已到移动端浏览范围上限。这里不是完整的记忆导出。</text>
    <text class="memory-footer">此页只查看，不修改、不删除，也不增加记忆使用次数。关闭记忆不等于删除已有记录；完整撤销与删除暂未在移动端提供。</text>
    </template>
    </template>
  </view>
</template>

<script setup>
import { computed, ref } from 'vue';
import { onHide, onShow, onUnload } from '@dcloudio/uni-app';
import { api, getPreferences, getActiveSession, setActiveSession } from '../../services/api.js';
import { research } from '../../services/research.js';
import BrandHeader from '../../components/BrandHeader.vue';
import ModeBanner from '../../components/ModeBanner.vue';
import WorkspaceBar from '../../components/WorkspaceBar.vue';
import CapabilityBoundary from '../../components/CapabilityBoundary.vue';
import MemoryObservatory from '../../components/MemoryObservatory.vue';

const preferences = ref({ mode: 'demo' }), sessions = ref([]), sessionId = ref('');
const items = ref([]), search = ref(''), expandedIds = ref([]), nextOffset = ref(null), windowLimited = ref(false);
const loading = ref(false), loadingMore = ref(false), error = ref(''), loadMoreError = ref('');
const researchEnabled = ref(false), memoryWriteEnabled = ref(false), confirmedItems = ref([]), confirmedSource = ref(null), notice = ref('');
const capabilityReady = ref(false), confirmedQuery = ref(''), confirmedDate = ref(''), confirmedCursor = ref(null), confirmedTotal = ref(0), showArchived = ref(false);
const observationRevision = ref(0);
const memoryHistoryId = ref(''), memoryVersions = ref([]), historyCursor = ref(null), editorFrom = ref(''), editorUntil = ref(''), editorOriginalDates = ref(null);
let confirmedFilters = {query:'',as_of:null,include_archived:false};
const editorOpen = ref(false), editorText = ref(''), editorKind = ref('finding'), editorSources = ref([]), editorOrigin = ref(''), editingId = ref(''), editingRevision = ref(null), saving = ref(false);
const canSaveMemory = computed(() => editorText.value.trim() && Array.from(editorText.value).length <= 2000);
let generation = 0, visible = false;
const pageSize = 20;
const filteredItems = computed(() => { const query = search.value.trim().toLocaleLowerCase(); return items.value.filter(item => !query || `${item.summary} ${item.question}`.toLocaleLowerCase().includes(query)); });
function kindLabel(kind) { return ({ user_preference: '偏好线索', feedback_rule: '反馈整理', project_state: '进展摘要', reference_pointer: '参考线索', reasoning_trace: '思考摘要', insight: '对话摘要' })[kind] || '对话整理'; }
function formatDate(value) { if (value === null || value === undefined || !Number.isFinite(Number(value))) return '整理时间未提供'; const date = new Date(Number(value) * 1000); return Number.isNaN(date.getTime()) ? '整理时间未提供' : `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}.${String(date.getDate()).padStart(2, '0')} 整理`; }
function toggleQuestion(id) { expandedIds.value = expandedIds.value.includes(id) ? expandedIds.value.filter(value => value !== id) : [...expandedIds.value, id]; }
function confirmedKind(kind) { return ({finding:'研究结论',preference:'回答偏好',decision:'研究决策'})[kind] || '确认内容'; }
function dateValue(value,end = false) { if (!value) return null; if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error('日期格式不符合要求'); const date = new Date(value+(end ? 'T23:59:59' : 'T00:00:00')); if (!Number.isFinite(date.getTime()) || date.getTime() < 0) throw new Error('日期不符合要求'); return date.getTime()/1000; }
function inputDate(epoch) { if (epoch === null || epoch === undefined || !Number.isFinite(epoch)) return ''; const date = new Date(epoch*1000); return `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`; }
function validityLabel(item) { return `${item.valid_from !== null && item.valid_from !== undefined ? ` · ${inputDate(item.valid_from)} 起适用` : ''}${item.valid_until !== null && item.valid_until !== undefined ? `，至 ${inputDate(item.valid_until)}` : ''}`; }
function closeMemoryHistory() { memoryHistoryId.value = ''; memoryVersions.value = []; historyCursor.value = null; }
function cancelMemory() { if (saving.value) return; editorOpen.value = false; editorText.value = ''; editorKind.value = 'finding'; editorSources.value = []; editorOrigin.value = ''; editingId.value = ''; editingRevision.value = null; editorFrom.value = ''; editorUntil.value = ''; editorOriginalDates.value=null; }
function startMemory() { if (!visible || !researchEnabled.value || !memoryWriteEnabled.value || !sessionId.value || loading.value || saving.value) return; cancelMemory(); editorOpen.value = true; notice.value = ''; }
function editMemory(item) { if (!visible || !researchEnabled.value || !memoryWriteEnabled.value || saving.value || loading.value || loadingMore.value || confirmedDate.value) return; editorOpen.value = true; editorText.value = item.text; editorKind.value = item.kind; editorSources.value = item.status === 'invalidated' ? [] : [...item.source_ids]; editorOrigin.value = ''; editingId.value = item.status === 'invalidated' ? '' : item.id; editingRevision.value = item.status === 'invalidated' ? null : item.revision; editorFrom.value = ''; editorUntil.value = inputDate(item.valid_until); editorOriginalDates.value={from:null,until:item.valid_until ?? null}; notice.value = item.status === 'invalidated' ? '已复制为没有来源关联的新草稿。请核对内容后再保存；旧记忆仍然失效，没有恢复原文来源。' : '本次修正默认从保存时生效，旧版本仍可回看。若需要追溯生效，请明确选择开始日期。'; }
async function confirmedAction(action) {
  if (!visible || !researchEnabled.value || !sessionId.value || saving.value || loading.value || loadingMore.value) return;
  const request = generation, sid = sessionId.value; saving.value = true; error.value = '';
  try { await action(sid,() => visible && request === generation && sid === sessionId.value); }
  catch (issue) { if (visible && request === generation) error.value = issue?.status === 409 ? '这条内容已有更新。先刷新并核对新版本，再决定是否修改；没有覆盖别人或另一页面的更新。' : issue?.message || '操作未完成，请刷新确认；没有自动重试。'; }
  finally { if (request === generation) saving.value = false; }
}
function confirmOperation(options) { return new Promise((resolve,reject) => { const timer = setTimeout(()=>reject(new Error('确认等待超时，没有继续操作')),60000); try { uni.showModal({...options,success:value=>{clearTimeout(timer);resolve(value);},fail:()=>{clearTimeout(timer);reject(new Error('未能确认操作，没有继续保存或删除'));}}); } catch { clearTimeout(timer); reject(new Error('当前平台无法确认操作，没有继续')); } }); }
async function saveMemory() {
  if (!memoryWriteEnabled.value || !editorOpen.value || !canSaveMemory.value) return;
  let validFrom,validUntil;
  try { validFrom = editorOriginalDates.value && editorFrom.value === inputDate(editorOriginalDates.value.from) ? editorOriginalDates.value.from : dateValue(editorFrom.value); validUntil = editorOriginalDates.value && editorUntil.value === inputDate(editorOriginalDates.value.until) ? editorOriginalDates.value.until : dateValue(editorUntil.value,true); if (validFrom !== null && validUntil !== null && validUntil <= validFrom) throw new Error('结束日期需要晚于开始时间'); }
  catch (issue) { error.value = issue.message; return; }
  const value = {text:editorText.value,kind:editorKind.value,source_ids:[...editorSources.value],...(editingId.value ? {memory_id:editingId.value,expected_revision:editingRevision.value} : {}),...(validFrom !== null ? {valid_from:validFrom} : {}),valid_until:validUntil};
  await confirmedAction(async (sid,current) => { const confirm = await confirmOperation({title:value.memory_id ? '确认修正记忆？' : '确认保留这条内容？',content:'这条内容将保存在当前研究会话，后续提问可能发送给模型。仅代表你确认保存，并不代表系统核验其真实性。',confirmText:'确认保存',cancelText:'再检查'}); if (!confirm.confirm || !current()) return;
    const result = await research.saveMemory(sid,value); if (!current()) return;
    const index = confirmedItems.value.findIndex(item=>item.id === result.id); if (index >= 0) confirmedItems.value[index] = result; else if (!confirmedFilters.query && confirmedFilters.as_of === null) { confirmedItems.value.unshift(result); confirmedTotal.value += 1; }
    observationRevision.value += 1; closeMemoryHistory(); editorOpen.value = false; editorText.value = ''; editingId.value = ''; editingRevision.value = null; editorSources.value = []; editorFrom.value = ''; editorUntil.value = ''; notice.value = '已保存你确认的内容。没有调用模型，没有自动生成新记忆。';
  });
}
async function removeMemory(item) { if (!memoryWriteEnabled.value || confirmedDate.value) return; await confirmedAction(async (sid,current) => {
  const confirm = await confirmOperation({title:'删除这条确认记忆？',content:'删除后不再用于新问题。历史回答中的旧内容不会自动改写；此操作不能撤销。',confirmText:'删除',cancelText:'保留'}); if (!confirm.confirm || !current()) return;
  await research.deleteMemory(sid,item.id,item.revision); if (!current()) return;
  confirmedItems.value = confirmedItems.value.filter(value=>value.id !== item.id); confirmedSource.value = null; confirmedTotal.value = Math.max(0,confirmedTotal.value-1); closeMemoryHistory();
  if (editingId.value === item.id) { editorOpen.value = false; editorText.value = ''; editingId.value = ''; editingRevision.value = null; editorSources.value = []; }
  observationRevision.value += 1; notice.value = '确认记忆已删除，不再用于新问题。';
}); }
async function readConfirmedSource(id) { await confirmedAction(async (sid,current) => { const result = await research.source(sid,id); if (current()) confirmedSource.value = result; }); }
async function openMemoryHistory(item) { await confirmedAction(async (sid,current)=>{ const result = await research.memoryHistory(sid,item.id); if (!current()) return; memoryHistoryId.value = item.id; memoryVersions.value = result.items; historyCursor.value = result.next_cursor; }); }
async function loadMoreHistory() { const id = memoryHistoryId.value, cursor = historyCursor.value; if (!id || !cursor) return; await confirmedAction(async(sid,current)=>{ const result = await research.memoryHistory(sid,id,{cursor}); if (!current() || memoryHistoryId.value !== id) return; const seen = new Set(memoryVersions.value.map(item=>item.revision)); memoryVersions.value.push(...result.items.filter(item=>!seen.has(item.revision))); historyCursor.value = result.next_cursor === cursor ? null : result.next_cursor; }); }
async function archiveMemory(item) { if (!memoryWriteEnabled.value || confirmedDate.value) return; await confirmedAction(async(sid,current)=>{ const confirm = await confirmOperation({title:item.archived ? '恢复这条记忆的使用？' : '将这条记忆归档？',content:item.archived ? '恢复后，仅在内容有效且来源仍可用时参与相关问题。' : '归档后不再用于新问题，内容和版本仍可搜索、查看，也可恢复。',confirmText:item.archived ? '恢复使用' : '归档',cancelText:'取消'}); if (!confirm.confirm || !current()) return; const result = await research.archiveMemory(sid,item.id,item.revision,!item.archived); if (!current()) return; const index = confirmedItems.value.findIndex(value=>value.id === item.id); if (index >= 0) { if (result.archived && !showArchived.value) { confirmedItems.value.splice(index,1); confirmedTotal.value = Math.max(0,confirmedTotal.value-1); } else confirmedItems.value[index] = result; } observationRevision.value += 1; closeMemoryHistory(); notice.value = result.archived ? '已归档，不再用于新问题；内容仍可回看。' : '已恢复使用，来源状态和适用时间仍会参与检查。'; }); }
async function searchConfirmed() {
  if (!visible || !researchEnabled.value || !sessionId.value || saving.value || loading.value || loadingMore.value) return;
  let filters; try { filters = {query:confirmedQuery.value.trim(),as_of:dateValue(confirmedDate.value,true),include_archived:showArchived.value}; } catch(issue) { error.value=issue.message; return; }
  const request = ++generation, sid = sessionId.value; loading.value=true; error.value=''; loadMoreError.value=''; confirmedSource.value=null; confirmedItems.value=[];confirmedCursor.value=null;confirmedTotal.value=0;closeMemoryHistory();
  try { const result = await research.memoryPage(sid,filters); if (!visible || request !== generation || sid !== sessionId.value) return; confirmedFilters=filters; confirmedItems.value=result.items; confirmedCursor.value=result.next_cursor; confirmedTotal.value=result.total; }
  catch(issue) { if (visible && request === generation) error.value=issue?.message || '记忆搜索失败，请重试。'; }
  finally { if (request === generation) loading.value=false; }
}
async function loadMoreConfirmed() {
  if (!visible || !researchEnabled.value || loading.value || loadingMore.value || saving.value || !confirmedCursor.value) return;
  const request=generation,sid=sessionId.value,cursor=confirmedCursor.value; loadingMore.value=true;loadMoreError.value='';
  try { const result=await research.memoryPage(sid,{...confirmedFilters,cursor}); if (!visible || request!==generation || sid!==sessionId.value) return; const seen=new Set(confirmedItems.value.map(item=>item.id));confirmedItems.value.push(...result.items.filter(item=>!seen.has(item.id)));confirmedCursor.value=result.next_cursor===cursor ? null : result.next_cursor;confirmedTotal.value=result.total; }
  catch(issue) { if (visible && request===generation) loadMoreError.value=issue?.message || '更多记忆读取失败，可重试。'; }
  finally { if(request===generation) loadingMore.value=false; }
}
async function selectConfirmedDate(event) { if (saving.value || loading.value || loadingMore.value || !/^\d{4}-\d{2}-\d{2}$/.test(event?.detail?.value)) return; confirmedDate.value=event.detail.value; await searchConfirmed(); }
async function clearConfirmedDate() { if (saving.value || loading.value || loadingMore.value) return; confirmedDate.value='';await searchConfirmed(); }
async function toggleArchived() { if (saving.value || loading.value || loadingMore.value) return; showArchived.value=!showArchived.value;await searchConfirmed(); }
function readPage(result, offset) {
  if (!result || result.scope !== 'session' || !Array.isArray(result.items) || result.items.length > pageSize) throw new Error('记忆范围或返回格式不符合要求，请刷新后重试。');
  const page = result.items.map(item => {
    if (!item || typeof item.id !== 'string' || !item.id || item.id.length > 128 || item.source_type !== 'assistant_summary' || item.question_source !== 'user_message') throw new Error('记忆来源标记不完整，暂不展示。');
    return { id: item.id, kind: String(item.kind || ''), summary: String(item.summary || ''), question: String(item.question || ''), created_at: item.created_at, truncated_fields: Array.isArray(item.truncated_fields) ? item.truncated_fields : [] };
  });
  const next = result.next_offset;
  const canContinue = result.has_more === true && Number.isInteger(next) && next > offset && next <= 1000;
  return { page, next: canContinue ? next : null, limited: result.has_more === true && !canContinue };
}
async function reload() {
  const request = ++generation; preferences.value = getPreferences(); loading.value = true; loadingMore.value = false; capabilityReady.value = false;
  error.value = ''; loadMoreError.value = ''; items.value = []; expandedIds.value = []; nextOffset.value = null; windowLimited.value = false; search.value = '';
  confirmedItems.value = []; confirmedSource.value = null; confirmedCursor.value = null; confirmedTotal.value = 0; closeMemoryHistory();
  sessions.value = []; sessionId.value = '';
  try {
    const capabilities = await research.capabilities({refresh:true});
    if (!visible || request !== generation) return;
    researchEnabled.value = capabilities.enabled === true; memoryWriteEnabled.value = capabilities.memory_write_enabled === true; capabilityReady.value = true;
    if (preferences.value.connectionKind === 'invite' && !researchEnabled.value) return;
    const available = await api.sessions();
    if (!visible || request !== generation) return;
    sessions.value = Array.isArray(available) ? available : [];
    const stored = String(getActiveSession() || '');
    // A deleted/invalid explicit selection must not silently show another session.
    sessionId.value = stored ? (sessions.value.some(item => String(item.id) === stored) ? stored : '') : String(sessions.value[0]?.id || '');
    setActiveSession(sessionId.value);
    if (!sessionId.value) return;
    if (researchEnabled.value) {
      confirmedFilters = {query:confirmedQuery.value.trim(),as_of:dateValue(confirmedDate.value,true),include_archived:showArchived.value};
      const result = await research.memoryPage(sessionId.value,confirmedFilters); if (!visible || request !== generation) return; confirmedItems.value = result.items; confirmedCursor.value = result.next_cursor; confirmedTotal.value = result.total;
      const draft = memoryWriteEnabled.value ? research.consumeMemoryDraft(sessionId.value) : null;
      if (draft) { editorOpen.value = true; editorText.value = draft.text; editorKind.value = draft.kind; editorSources.value = draft.source_ids; editorOrigin.value = draft.origin; editingId.value = ''; editingRevision.value = null; }
      return;
    }
    const result = await api.memory(sessionId.value, { limit: pageSize, offset: 0 });
    if (!visible || request !== generation) return;
    const page = readPage(result, 0); items.value = page.page; nextOffset.value = page.next; windowLimited.value = page.limited;
  } catch (issue) { if (visible && request === generation) error.value = issue?.message || '记忆读取失败，请稍后刷新。'; }
  finally { if (request === generation) loading.value = false; }
}
async function loadMore() {
  if (!visible || !sessionId.value || loading.value || loadingMore.value || nextOffset.value === null) return;
  const request = generation, sid = sessionId.value, offset = nextOffset.value;
  loadingMore.value = true; loadMoreError.value = '';
  try {
    const result = await api.memory(sid, { limit: pageSize, offset });
    if (!visible || request !== generation || sid !== sessionId.value) return;
    const page = readPage(result, offset), seen = new Set(items.value.map(item => item.id));
    items.value = [...items.value, ...page.page.filter(item => !seen.has(item.id))]; nextOffset.value = page.next; windowLimited.value = page.limited;
  } catch (issue) { if (visible && request === generation) loadMoreError.value = issue?.message || '更多记忆读取失败，可重试。'; }
  finally { if (request === generation) loadingMore.value = false; }
}
async function changeSession(id) { if (saving.value) return; cancelMemory(); confirmedQuery.value = ''; confirmedDate.value = ''; showArchived.value = false; confirmedItems.value = []; confirmedSource.value = null; notice.value = ''; setActiveSession(id); await reload(); }
function openChat() { uni.switchTab({ url: '/pages/chat/index' }); }
function hide() { visible = false; generation += 1; items.value = []; confirmedItems.value = []; confirmedSource.value = null; researchEnabled.value = false; memoryWriteEnabled.value = false; capabilityReady.value=false; saving.value = false; cancelMemory(); closeMemoryHistory(); confirmedCursor.value=null; confirmedTotal.value=0; confirmedQuery.value='';confirmedDate.value='';showArchived.value=false; notice.value = ''; sessions.value = []; sessionId.value = ''; expandedIds.value = []; loading.value = false; loadingMore.value = false; nextOffset.value = null; }
onShow(() => { visible = true; void reload(); }); onHide(hide); onUnload(hide);
</script>

<style>
@import '../../components/mobile.css';
.memory-intro { padding: 21px; border: 1px solid #dce4d1; border-radius: 22px; background: #e6ecdd; }
.memory-intro-heading { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.memory-intro-mark { color: #6d8851; font-size: 24px; }
.memory-intro-title { flex: 1; color: #203d35; font-size: 17px; font-weight: 600; }
.memory-readonly { padding: 3px 8px; border: 1px solid #cad6bd; border-radius: 7px; color: #657957; font-size: 10px; }
.memory-intro-copy { display: block; color: #5d7054; font-size: 12px; line-height: 1.85; }
.memory-loaded { display: block; margin-top: 5px; color: #6d7b62; font-size: 11px; }
.memory-empty { border: 1px dashed #d6deca; border-radius: 22px; padding: 36px 18px; }
.memory-empty .empty-title { font-size: 19px; }
.memory-empty-mark { margin-bottom: 18px; color: #94aa7e; font-size: 40px; }
.memory-search { display: flex; align-items: center; gap: 10px; min-height: 48px; padding: 0 15px; margin-bottom: 16px; border: 1px solid #dfe5d5; border-radius: 14px; background: #fff; color: #6d855b; font-size: 24px; }
.memory-search-input { flex: 1; min-width: 0; width: 0; height: 46px; font-size: 13px; color: #25372f; }
.memory-clear { flex-shrink: 0; width: 44px; height: 44px; margin: 0 -10px 0 0; padding: 0; line-height: 44px; background: transparent; color: #6b7b5e; font-size: 23px; }
.memory-search-note { display: block; margin: -4px 3px 16px; color: #6c7c5e; font-size: 11px; }
.memory-card { padding: 21px; margin-bottom: 14px; border: 1px solid #e2e7d9; border-radius: 22px; background: #fff; }
.memory-card-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 18px; }
.memory-kind { padding: 4px 9px; border-radius: 6px; background: #edf3e3; color: #587445; font-size: 11px; }
.memory-review { color: #8a7546; font-size: 10px; }
.memory-content-label { display: block; margin-bottom: 6px; color: #708160; font-size: 11px; }
.memory-summary, .confirmed-memory-content { display: block; color: #2c4533; font-size: 14px; line-height: 1.95; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
.memory-card-bottom { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 14px; padding-top: 8px; border-top: 1px solid #eff2e8; }
.memory-date { color: #728265; font-size: 10px; }
.memory-source-toggle { padding-right: 0; font-size: 11px; min-height: 44px; }
.memory-question { margin-top: 8px; padding: 15px; background: #f6f7f1; border-left: 3px solid #cfddb9; border-radius: 4px 12px 12px 4px; }
.memory-question-label { display: block; margin-bottom: 8px; color: #6b7d59; font-size: 11px; }
.memory-question-content, .confirmed-source-content { display: block; color: #506346; font-size: 13px; line-height: 1.85; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
.memory-question-note { display: block; margin-top: 10px; color: #6d7c60; font-size: 11px; line-height: 1.7; }
.memory-truncated { display: block; margin-top: 10px; color: #947343; font-size: 11px; line-height: 1.8; }
.memory-more { margin: 18px 0; }
.memory-limit-note, .memory-footer { display: block; margin: 20px 4px; color: #6c7b5f; font-size: 11px; line-height: 1.9; }
.confirmed-intro { padding:24px; margin-bottom:24px; border-radius:24px; background:#203d35; }
.confirmed-eyebrow { display:block; color:#d4ed96; letter-spacing:1.5px; font-size:10px; }
.confirmed-heading { display:block; margin:12px 0; color:#fff; font-size:22px; line-height:1.5; }
.confirmed-copy { display:block; margin:9px 0; color:#65765b; font-size:12px; line-height:1.9; overflow-wrap:anywhere; }
.confirmed-intro .confirmed-copy { margin-bottom:20px; color:#d2ded0; }
.confirmed-intro .primary-button { background:#d4ed96; color:#203d35; }
.confirmed-editor { padding:20px; margin:20px 0; border:1px solid #d8e4c9; border-radius:22px; background:#fff; }
.confirmed-types { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0; }
.confirmed-type { flex:1 1 40%; min-height:44px; margin:0; padding:10px 8px; border-radius:10px; font-size:12px; color:#577148; background:#f2f5ec; }
.confirmed-type-active { background:#dcebbf; color:#203d35; }
.confirmed-textarea { box-sizing:border-box; width:100%; min-height:120px; max-height:420px; padding:14px; border:1px solid #dfe6d3; border-radius:12px; background:#f9faf5; color:#2c4533; font-size:14px; line-height:1.9; }
.confirmed-counter { display:block; margin:10px 0 16px; color:#687b59; font-size:11px; line-height:1.8; }
.confirmed-review { color:#587745; }
.confirmed-item-actions { display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }
.confirmed-remove { color:#995a4d; }
.memory-validity,.memory-history { padding:20px; border:1px solid #e0e6d5; border-radius:20px; background:#f7f9f1; margin:16px 0; }
.memory-date-controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:8px 0 18px; }
.memory-date-button { padding:10px 12px; border:1px solid #dce4d0; border-radius:10px; background:#f6f8ef; color:#546a48; font-size:12px; line-height:1.8; }
.memory-history-version { padding:18px 0; border-top:1px solid #e2e8d9; }
@media (max-width: 360px) { .memory-intro, .memory-card { padding: 17px; } .memory-card-top { gap: 5px; } .memory-review { font-size: 9px; } }
</style>
