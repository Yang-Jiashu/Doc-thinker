<template>
  <view class="mobile-page documents-page" :class="{ 'has-research-library': researchEnabled }">
    <BrandHeader eyebrow="LIBRARY" title="你的资料库" subtitle="把零散的资料，变成思考的起点。" />
    <ModeBanner compact :mode="preferences.mode" :connection-kind="preferences.connectionKind" :invite-pending="preferences.invitePending" :research-enabled="researchEnabled" />
    <view v-if="loading && !capabilityReady" class="status-box" role="status">正在连接你的资料库…</view>
    <view v-else-if="!capabilityReady && error" class="empty-state"><text class="empty-title">暂时没有连上资料库</text><text class="empty-copy">{{ error }}</text><button class="secondary-button" @tap="reload">重新连接</button></view>
    <CapabilityBoundary v-else-if="preferences.connectionKind === 'invite' && !researchEnabled" feature-id="documents" :invite-pending="preferences.invitePending" />
    <template v-else>
    <WorkspaceBar :sessions="sessions" :session-id="sessionId" :disabled="loading || uploading || busyDocument || copyingExport || refreshingStatuses" @change="changeSession" />
    <view v-if="error" class="error-box" role="alert">{{ error }}</view>
    <view v-if="notice" class="status-box" role="status">{{ notice }}</view>
    <view class="library-workspace" :class="{ 'has-research-reader': researchEnabled }">
    <view id="library-catalog" class="library-catalog" aria-label="当前会话资料列表">
    <view class="library-upload-card">
      <view class="upload-card-heading"><view class="upload-mark" aria-hidden="true">＋</view><view><text class="upload-card-title">给思考加一点依据</text><text class="upload-card-subtitle">{{ researchEnabled ? '论文、研究笔记，或一段原始资料。' : '文档、笔记，或一张有信息的图片。' }}</text></view></view>
      <button class="primary-button full-width upload-action" :disabled="!sessionId || loading || uploading || busyDocument || copyingExport || refreshingStatuses || (researchEnabled && !uploadsAllowed)" @tap="upload()">{{ uploading ? '等待选择或上传，请稍候…' : researchEnabled && !uploadsAllowed ? '当前服务未开放上传' : preferences.mode === 'demo' ? '添加示例资料' : '选择资料上传' }}<text v-if="!uploading" aria-hidden="true"> ↗</text></button>
      <!-- #ifdef MP-WEIXIN -->
      <button v-if="preferences.mode !== 'demo' && !researchEnabled" class="secondary-button full-width image-upload-action" :disabled="!sessionId || loading || uploading" @tap="upload('image')">从相册选择图片</button>
      <!-- #endif -->
      <view class="upload-note">
      <text v-if="preferences.mode === 'demo'">仅添加演示记录，不读取或上传你的文件。</text>
      <text v-else>{{ researchEnabled ? '只加入当前会话。解析不调用模型；你提问时，相关片段才可能发送给模型。' : '仅加入当前会话，发送至当前服务；提交后还需等待解析。' }}</text>
      <!-- #ifdef MP-WEIXIN -->
      <text v-if="preferences.mode !== 'demo'" class="platform-note">文档从微信聊天中选择。选好后再次确认上传，不会自动读取聊天记录。</text>
      <!-- #endif -->
      <!-- #ifdef APP-PLUS -->
      <text class="platform-note">{{ researchEnabled ? 'App 本版尚未接入系统文档选择，请先用浏览器上传。' : 'App 本版从相册选择图片；系统 PDF / Word 文档选择尚未接入。' }}</text>
      <!-- #endif -->
      <!-- #ifndef APP-PLUS -->
      <text v-if="preferences.mode !== 'demo'" class="platform-note">{{ researchEnabled ? 'PDF / TXT / MD · 单份不超过 10 MB · 扫描 PDF 暂不支持 OCR' : 'PDF / TXT / MD / Word / 图片 · 单份不超过 20 MB' }}</text>
      <!-- #endif -->
      </view>
    </view>
    <view class="section-heading library-heading"><view><text class="section-title">当前会话</text><text class="library-count">{{ files.length }} 份资料 · {{ readyCount }} 份可提问</text></view><button class="quiet-button refresh-action" :disabled="loading || refreshingStatuses || uploading || busyDocument || copyingExport" @tap="reload">{{ loading || refreshingStatuses ? '更新中' : '刷新列表' }}</button></view>
    <view v-if="researchEnabled && pendingCount" class="library-processing" role="status"><view><text class="processing-title">{{ pendingCount }} 份资料正在准备</text><text class="processing-copy">{{ trackingState === 'watching' ? '本页会自动更新解析状态，你可以先查看已就绪的资料。' : trackingState === 'error' ? '状态暂时未能更新，资料可能仍在后台解析。请手动刷新确认。' : '暂时还没完成。已暂停自动查看，稍后可手动刷新。' }}</text></view><text v-if="trackingState === 'watching'" class="processing-indicator" aria-hidden="true">···</text></view>
    <view v-if="loading" class="status-box">正在读取资料列表…</view>
    <view v-else-if="!sessionId" class="empty-state"><text class="empty-title">先创建一个会话</text><text class="empty-copy">回到问答页新建会话，再把需要一起分析的资料放进来。</text><button class="secondary-button" @tap="openChat">去问答页</button></view>
    <view v-else-if="!files.length" class="empty-state library-empty"><view class="empty-paper" aria-hidden="true">▤</view><text class="empty-title">这里，留给你的第一份资料</text><text class="empty-copy">从一份文档开始。我们会在这里显示它的解析进度，完成后就能围绕它提问。</text></view>
    <view v-else>
      <view v-if="readyCount" class="library-next-action"><button class="secondary-button full-width" role="button" :disabled="loading || uploading || !sessionId" @tap="openReadyChat">围绕当前会话资料提问 <text aria-hidden="true">↗</text></button><text class="library-next-note">只回到助手，仍由你编辑并发送问题；不限定单份文件，也不会自动调用模型。</text></view>
      <view v-if="files.length > 1" class="library-tools"><view class="library-search"><input v-model="fileQuery" class="library-search-input" placeholder="查找资料名称" aria-label="查找资料名称" maxlength="120" /><button v-if="fileQuery" class="quiet-button" aria-label="清空搜索" @tap="fileQuery = ''">清空</button></view><view class="library-filters"><button v-for="filter in fileFilters" :key="filter.id" class="library-filter" :class="{ 'is-selected': fileFilter === filter.id }" :aria-pressed="fileFilter === filter.id" @tap="fileFilter = filter.id">{{ filter.label }} {{ filter.count }}</button></view></view>
      <view v-if="!visibleFiles.length" class="empty-state library-empty"><text class="empty-title">没有符合条件的资料</text><text class="empty-copy">试试其他名称，或查看全部资料。</text><button class="secondary-button" @tap="clearFileFilters">查看全部</button></view>
      <view v-for="(file, index) in visibleFiles" :id="'library-document-' + index" :key="file.id || file.file_name || file.filename || index" class="document-card" :class="{ 'is-reading': isReading(file) }" tabindex="-1">
        <view class="document-row"><view class="document-symbol" aria-hidden="true"><text class="document-glyph">▤</text><text class="document-extension">{{ fileExtension(file) }}</text></view>
        <view class="document-info"><text class="row-title">{{ file.file_name || file.filename || file.name || '未命名文件' }}</text><view class="document-meta"><text class="file-status" :class="statusClass(file.status || file.processing_status)">{{ statusLabel(file.status || file.processing_status) }}</text><text v-if="fileSize(file)" class="file-size">{{ fileSize(file) }}</text></view></view></view>
        <text class="document-description">{{ statusCopy(file.status || file.processing_status) }}</text><text v-if="file.error || file.message" class="file-detail">{{ file.error || file.message }}</text>
        <view v-if="researchEnabled" class="research-document-actions"><text class="research-chunk-count">{{ file.chunk_count }} 个片段</text><button class="quiet-button" role="button" :tabindex="busyDocument || loading || uploading || refreshingStatuses ? -1 : 0" :aria-label="'查看 ' + (file.filename || file.file_name || file.name || '未命名文件')" :aria-expanded="isReading(file)" aria-controls="library-reader" :disabled="busyDocument || loading || uploading || refreshingStatuses" @tap="inspectDocument(file)" @keydown="readerKeydown($event, 'inspect', file)">{{ busyDocument && activeDocument === file.id ? '正在读取…' : isReading(file) ? '正在查看' : file.status === 'ready' ? '查看原文片段' : '查看处理状态' }}</button><button v-if="file.status === 'failed' && uploadsAllowed" class="quiet-button" :disabled="busyDocument || loading || uploading || refreshingStatuses" @tap="retryDocument(file)">重新解析</button><button class="quiet-button destructive-button" :disabled="busyDocument || loading || uploading || refreshingStatuses" @tap="removeDocument(file)">移除资料</button></view>
        <text v-if="researchEnabled && file.error_code" class="file-detail">{{ parseError(file.error_code) }}</text>
      </view>
      <text class="list-footnote">{{ researchEnabled ? '查看状态和原文不调用模型。资料不会自动加入记忆。移除操作仅作用于当前会话，详情会在确认前说明。' : '解析中时可稍后刷新。删除、重试解析等管理操作请使用桌面端。' }}</text>
    </view>
    </view>
    <view v-if="researchEnabled" id="library-reader" class="library-reader-column" role="region" aria-label="所选资料原文阅读区">
    <view v-if="!documentDetail && !sourceDetail" class="reader-empty" aria-label="尚未选择资料">
      <view class="reader-empty-icon" aria-hidden="true">▤</view><text class="reader-empty-title">留一半屏幕，给原文</text><text class="reader-empty-copy">在左侧选择一份资料，原文片段会在这里展开。随时对照，不必来回切页。</text><text class="reader-empty-note">只读取你打开的资料，不调用模型。</text>
    </view>
    <view v-if="documentDetail && !sourceDetail" id="document-reader" class="research-reader" tabindex="-1">
      <view class="section-heading reader-heading"><view><text class="reader-eyebrow">原文预览</text><text class="section-title">{{ documentDetail.filename }}</text></view><button class="quiet-button reader-close" role="button" :tabindex="busyDocument ? -1 : 0" :disabled="busyDocument" aria-label="关闭原文并返回选中的资料" @tap="dismissReader" @keydown="readerKeydown($event, 'dismiss')">关闭</button></view>
      <text class="research-copy">{{ statusLabel(documentDetail.status) }} · {{ documentDetail.chunk_count }} 个片段。只展示下方提供的来源，不代表已浏览全文。</text>
      <text v-if="documentDetail.preview_limited" class="research-copy">此页只预览前 20 个片段，不是全文导出。</text>
      <view v-for="chunk in documentDetail.chunks" :key="chunk.id" class="research-source">
        <text class="source-location">{{ sourceLabel(chunk) }}</text>
        <!-- #ifdef MP-WEIXIN -->
        <text v-if="chunk.content" class="source-excerpt" user-select>{{ chunk.content }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text v-if="chunk.content" class="source-excerpt" selectable>{{ chunk.content }}</text>
        <!-- #endif -->
        <text v-if="chunk.excerpt_limited" class="research-copy">这里只预览片段前 1200 字符，打开原文可查看完整片段。</text>
        <button class="quiet-button" role="button" :tabindex="busyDocument ? -1 : 0" :disabled="busyDocument" @tap="readSource(chunk.id)" @keydown="readerKeydown($event, 'source', chunk.id)">打开原文片段</button>
      </view>
      <button v-for="id in documentDetail.source_ids.filter(id => !documentDetail.chunks.some(chunk => chunk.id === id))" :key="id" class="secondary-button full-width" role="button" :tabindex="busyDocument ? -1 : 0" :disabled="busyDocument" @tap="readSource(id)" @keydown="readerKeydown($event, 'source', id)">打开来源 {{ id }}</button>
      <text v-if="!documentDetail.chunks.length && !documentDetail.source_ids.length" class="research-copy">{{ documentDetail.status === 'ready' ? '尚未找到可打开的原文，请刷新确认。' : '解析尚未完成，列表会在本页短暂自动更新；没有自动重试解析。' }}</text>
    </view>
    <view v-if="sourceDetail" id="source-reader" class="research-reader" tabindex="-1">
      <view class="reader-navigation"><button class="quiet-button" role="button" :tabindex="busyDocument ? -1 : 0" :disabled="busyDocument" @tap="closeSource" @keydown="readerKeydown($event, 'back')">← 返回资料预览</button><button class="quiet-button reader-close" role="button" :tabindex="busyDocument ? -1 : 0" :disabled="busyDocument" aria-label="关闭原文并返回选中的资料" @tap="dismissReader" @keydown="readerKeydown($event, 'dismiss')">关闭</button></view>
      <view class="section-heading reader-heading"><text class="section-title">原文片段</text></view>
      <text class="source-location">{{ sourceDetail.filename }} · {{ sourceLabel(sourceDetail) }}</text>
      <!-- #ifdef MP-WEIXIN -->
      <text class="source-excerpt" user-select>{{ sourceDetail.content }}</text>
      <!-- #endif -->
      <!-- #ifndef MP-WEIXIN -->
      <text class="source-excerpt" selectable>{{ sourceDetail.content }}</text>
      <!-- #endif -->
      <text class="research-copy">片段编号：{{ sourceDetail.id }}。这是提取的原文，不是模型总结；PDF 提取顺序可能与版面不同。</text>
    </view>
    </view>
    </view>
    <view v-if="researchEnabled && sessionId" class="research-export">
      <button class="secondary-button full-width" :disabled="loading || uploading || busyDocument" @tap="previewExport">预览当前研究会话导出</button>
      <text class="research-copy">导出当前资料目录与确认记忆，不含论文全文或聊天记录。不创建公开链接，不自动写入剪贴板。</text>
      <view v-if="exportPreview">
        <!-- #ifdef MP-WEIXIN -->
        <text class="source-excerpt" user-select>{{ exportPreview }}</text>
        <!-- #endif -->
        <!-- #ifndef MP-WEIXIN -->
        <text class="source-excerpt" selectable>{{ exportPreview }}</text>
        <!-- #endif -->
        <text v-if="exportMarkdown.length > exportPreview.length" class="research-copy">上方仅预览开头；完整内容仍保留，可下载或按顺序分段复制，不会截断导出。</text>
        <!-- #ifdef H5 -->
        <button class="secondary-button" :disabled="copyingExport" @tap="downloadExport">下载完整 Markdown</button>
        <!-- #endif -->
        <button v-if="exportParts.length === 1" class="secondary-button" :disabled="copyingExport" @tap="copyExport()">{{ copyingExport ? '操作中…' : '复制完整 Markdown' }}</button>
        <button class="quiet-button" :disabled="copyingExport" :aria-expanded="exportPartsOpen" @tap="exportPartsOpen = !exportPartsOpen">{{ exportPartsOpen ? '收起复制段' : `分段查看与复制（共 ${exportParts.length} 段）` }}</button>
        <view v-if="exportPartsOpen && exportParts.length">
          <text class="research-copy">第 {{ selectedExportPart + 1 }} / {{ exportParts.length }} 段。请依次粘贴到同一文件，不额外插入分隔符，即可还原全文；复制不会自动跳到下一段。</text>
          <view class="export-part-controls"><button class="quiet-button" :disabled="copyingExport || selectedExportPart === 0" @tap="exportPartIndex = selectedExportPart - 1">上一段</button><button class="secondary-button" :disabled="copyingExport" @tap="copyExport(selectedExportPart)">复制第 {{ selectedExportPart + 1 }} 段</button><button class="quiet-button" :disabled="copyingExport || selectedExportPart + 1 >= exportParts.length" @tap="exportPartIndex = selectedExportPart + 1">下一段</button></view>
          <!-- #ifdef MP-WEIXIN -->
          <text class="source-excerpt" user-select>{{ exportParts[selectedExportPart] }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="source-excerpt" selectable>{{ exportParts[selectedExportPart] }}</text>
          <!-- #endif -->
        </view>
        <button class="quiet-button" :disabled="copyingExport" @tap="exportPreview = ''; exportMarkdown = ''">收起导出</button><text class="research-copy">文件和剪贴板可能含个人内容；请检查后再分享。不会自动上传，也不会生成公开链接。</text>
      </view>
    </view>
    </template>
  </view>
</template>

<script setup>
import { computed, ref } from 'vue';
import { onHide, onShow, onUnload } from '@dcloudio/uni-app';
import { api, getPreferences, getActiveSession, setActiveSession, researchConnectionKey } from '../../services/api.js';
import { research } from '../../services/research.js';
import { researchExportParts, downloadResearchMarkdown } from '../../services/research-export.js';
import ModeBanner from '../../components/ModeBanner.vue';
import WorkspaceBar from '../../components/WorkspaceBar.vue';
import BrandHeader from '../../components/BrandHeader.vue';
import CapabilityBoundary from '../../components/CapabilityBoundary.vue';

const preferences = ref({ mode: 'demo' }), sessions = ref([]), sessionId = ref(''), files = ref([]);
const loading = ref(false), uploading = ref(false), error = ref(''), notice = ref('');
const researchEnabled = ref(false), uploadsAllowed = ref(false), busyDocument = ref(false), documentDetail = ref(null), sourceDetail = ref(null);
const exportPreview = ref(''), exportMarkdown = ref(''), copyingExport = ref(false);
const exportPartIndex = ref(0), exportPartsOpen = ref(false);
const exportParts = computed(() => researchExportParts(exportMarkdown.value));
const selectedExportPart = computed(() => Math.max(0, Math.min(exportPartIndex.value, exportParts.value.length - 1)));
const capabilityReady = ref(false), refreshingStatuses = ref(false), trackingState = ref(''), activeDocument = ref('');
const fileQuery = ref(''), fileFilter = ref('all');
const readyCount = computed(() => files.value.filter(file => statusClass(file.status || file.processing_status) === 'file-ready').length);
const pendingCount = computed(() => files.value.filter(file => statusClass(file.status || file.processing_status) === 'file-processing').length);
const fileFilters = computed(() => [{id:'all',label:'全部',count:files.value.length},{id:'file-ready',label:'可提问',count:readyCount.value},{id:'file-processing',label:'处理中',count:pendingCount.value},{id:'attention',label:'需处理',count:files.value.filter(file=>!['file-ready','file-processing'].includes(statusClass(file.status || file.processing_status))).length}]);
const visibleFiles = computed(() => files.value.filter(file => {
  const status = statusClass(file.status || file.processing_status), name = String(file.file_name || file.filename || file.name || '').toLowerCase();
  return name.includes(fileQuery.value.trim().toLowerCase()) && (fileFilter.value === 'all' || (fileFilter.value === 'attention' ? !['file-ready','file-processing'].includes(status) : status === fileFilter.value));
}));
let generation = 0, visible = false, uploadGeneration = 0, trackingGeneration = 0, trackingTimer = null;
let exportScope = null;
function currentExport() { return exportScope !== null && pageCurrent(exportScope.request,exportScope.sid,exportScope.key); }
const statusDelays = [3000,5000,8000,13000,20000,30000];
function clearFileFilters() { fileQuery.value = ''; fileFilter.value = 'all'; }
const statusValue = value => String(value || '').toLowerCase();
function statusClass(value) {
  const status = statusValue(value);
  if (['processed', 'completed', 'ready', 'success', 'indexed', 'done'].includes(status)) return 'file-ready';
  if (['error', 'failed', 'failure'].includes(status)) return 'file-failed';
  if (['processing', 'parsing', 'pending', 'queued', 'uploading', 'uploaded'].includes(status)) return 'file-processing';
  return '';
}
function statusLabel(value) { return ({ 'file-ready': '可提问', 'file-failed': '解析失败', 'file-processing': '处理中' })[statusClass(value)] || '待确认'; }
function statusCopy(value) { return ({ 'file-ready': '已准备好，可以围绕这份资料提问。', 'file-failed': researchEnabled.value ? '暂时没能读出资料，请按下方提示处理后再试。' : '暂未完成解析，请查看提示或在桌面端检查。', 'file-processing': researchEnabled.value ? '正在等待或执行解析，完成后会显示“可提问”。' : '正在等待或执行解析，稍后刷新查看。' })[statusClass(value)] || '暂时无法确认处理结果，请刷新查看。'; }
function fileExtension(file) { const name = String(file.file_name || file.filename || file.name || ''); const suffix = name.includes('.') ? name.split('.').pop() : ''; return /^[a-z0-9]{1,5}$/i.test(suffix) ? suffix.toUpperCase() : 'FILE'; }
function fileSize(file) { const size = Number(file.size_bytes ?? file.file_size ?? file.size); if (!Number.isFinite(size) || size < 0) return ''; return size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(1)} MB` : size >= 1024 ? `${Math.round(size / 1024)} KB` : `${size} B`; }
function parseError(code) { return ({no_extractable_text:'未提取到可用文字；扫描 PDF 暂不支持 OCR。',invalid_pdf:'PDF 结构无法解析，请尝试文本版 PDF。',utf8_required:'文本编码无法读取，请转成 UTF-8 文件。',encrypted_pdf:'PDF 已加密，请上传没有密码保护的合法副本。',page_limit:'超过 300 页解析范围，请拆分文件。',text_limit:'提取的文字过多，请拆分文件。',chunk_limit:'资料片段过多，请拆分文件。',extraction_resource_limit:'解析超出本机资源或时间限制，没有自动重试。',page_stream_limit:'PDF 页面资源超过限制，请换用文本版或拆分文件。',parser_capacity_unavailable:'当前解析资源不足，请稍后手动重试或联系运营者；没有自动重试。'})[code] || '解析未完成，请核对格式或联系运营者；没有自动重试。'; }
function sourceLabel(chunk) { return chunk.page ? `PDF 第 ${chunk.page} 页` : Number.isInteger(chunk.ordinal) ? `原文片段 ${chunk.ordinal + 1}` : '原文来源'; }
function isReading(file) { return Boolean(documentDetail.value && file.id && documentDetail.value.id === file.id); }
function readerKeydown(event, action, target) {
  // uni-button is not a native keyboard button on H5. Keep the four read-only
  // actions explicit; repeats, IME input and browser shortcuts must not activate.
  if (!visible || busyDocument.value || !event || !['Enter', ' '].includes(event.key) || event.repeat || event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.defaultPrevented || typeof event.preventDefault !== 'function') return;
  const run = action === 'inspect' ? () => inspectDocument(target) : action === 'source' ? () => readSource(target) : action === 'dismiss' ? dismissReader : action === 'back' ? closeSource : null;
  if (!run) return;
  event.preventDefault(); event.stopPropagation?.();
  return run();
}
function closeReader() { documentDetail.value = null; sourceDetail.value = null; }
function dismissReader() {
  if (!visible || busyDocument.value) return;
  const index = visibleFiles.value.findIndex(file => file.id === documentDetail.value?.id);
  closeReader(); revealReader(index >= 0 ? `#library-document-${index}` : '#library-catalog');
}
function closeSource() {
  if (!visible || busyDocument.value) return;
  sourceDetail.value = null;
  revealReader(documentDetail.value ? '#document-reader' : '#library-catalog');
}
function pageCurrent(request,sid,key) {
  if (!visible || request !== generation || (sid !== null && sid !== sessionId.value)) return false;
  if (key === researchConnectionKey()) return true;
  // An identity change may occur without a page navigation (for example expiry).
  // Clear the old private view before allowing the user to reconnect.
  hide(); visible = true; preferences.value = getPreferences(); error.value = '身份或连接已改变，请重新连接资料库。';
  return false;
}
function stopStatusTracking(state = '') { trackingGeneration += 1; if (trackingTimer !== null) clearTimeout(trackingTimer); trackingTimer = null; refreshingStatuses.value = false; trackingState.value = state; }
function trackProcessing() {
  stopStatusTracking();
  if (!visible || !researchEnabled.value || !sessionId.value || !pendingCount.value) return;
  const request = generation, sid = sessionId.value, key = researchConnectionKey(), cycle = trackingGeneration, deadline = Date.now() + 120000;
  let checks = 0;
  trackingState.value = 'watching';
  const current = () => cycle === trackingGeneration && pageCurrent(request,sid,key);
  const schedule = () => {
    if (!current()) return;
    if (!pendingCount.value) { stopStatusTracking(); return; }
    if (checks >= statusDelays.length || Date.now() >= deadline) { stopStatusTracking('paused'); return; }
    trackingTimer = setTimeout(check, statusDelays[checks]);
  };
  const check = async () => {
    trackingTimer = null;
    if (!current()) return;
    if (Date.now() >= deadline) { stopStatusTracking('paused'); return; }
    if (loading.value || uploading.value || busyDocument.value || copyingExport.value) { schedule(); return; }
    refreshingStatuses.value = true; checks += 1;
    try {
      // Only read status. Never retry a parse, re-upload, or call a model here.
      const result = await research.documents(sid,{background:true});
      if (!current()) return;
      const previousReady = readyCount.value;
      files.value = Array.isArray(result) ? result : [];
      if (documentDetail.value) {
        const updated = files.value.find(file=>file.id === documentDetail.value.id);
        if (!updated || updated.status !== documentDetail.value.status) closeReader();
      }
      exportPreview.value = ''; exportMarkdown.value = '';
      if (readyCount.value > previousReady) notice.value = `${readyCount.value - previousReady} 份资料已准备好，可以开始提问。`;
      if (!pendingCount.value) {
        if (files.value.some(file=>file.status === 'failed')) notice.value = '处理已结束。部分资料未能解析，请查看对应提示；没有自动重试。';
        stopStatusTracking(); return;
      }
    } catch { if (current()) { stopStatusTracking('error'); return; } }
    finally { if (current()) refreshingStatuses.value = false; }
    schedule();
  };
  schedule();
}
function revealReader(selector) {
  const request = generation, sid = sessionId.value, key = researchConnectionKey();
  setTimeout(() => {
    if (!pageCurrent(request,sid,key)) return;
    try {
      // #ifdef H5
      if (typeof window !== 'undefined' && window.matchMedia?.('(min-width: 1100px)').matches && typeof document !== 'undefined') {
        const target = document.querySelector(selector);
        if (selector === '#document-reader' || selector === '#source-reader') {
          const panel = document.querySelector('#library-reader');
          if (panel) panel.scrollTop = 0;
        } else target?.scrollIntoView?.({block:'nearest',behavior:'auto'});
        target?.focus?.({preventScroll:true});
        return;
      }
      // #endif
      uni.pageScrollTo?.({selector,duration:0});
    } catch { /* Reading remains available if this platform cannot scroll to an anchor. */ }
  },0);
}
async function reload() {
  if (uploading.value || busyDocument.value || copyingExport.value || refreshingStatuses.value) return;
  stopStatusTracking();
  const request = ++generation, key = researchConnectionKey(); preferences.value = getPreferences(); loading.value = true; busyDocument.value = false; error.value = ''; files.value = [];
  capabilityReady.value = false; researchEnabled.value = false; uploadsAllowed.value = false;
  closeReader(); exportPreview.value = ''; exportMarkdown.value = ''; copyingExport.value = false;
  sessions.value = []; sessionId.value = '';
  try {
    const capabilities = await research.capabilities({refresh:true});
    if (!pageCurrent(request,null,key)) return;
    researchEnabled.value = capabilities.enabled === true; uploadsAllowed.value = capabilities.uploads_allowed === true; capabilityReady.value = true;
    if (preferences.value.connectionKind === 'invite' && !researchEnabled.value) return;
    const available = await api.sessions();
    if (!pageCurrent(request,null,key)) return;
    sessions.value = Array.isArray(available) ? available : [];
    const stored = String(getActiveSession() || '');
    sessionId.value = stored ? (sessions.value.some(item => String(item.id) === stored) ? stored : '') : String(sessions.value[0]?.id || '');
    setActiveSession(sessionId.value);
    if (sessionId.value) {
      const result = researchEnabled.value ? await research.documents(sessionId.value) : await api.files(sessionId.value);
      if (pageCurrent(request,null,key)) { files.value = Array.isArray(result) ? result : []; trackProcessing(); }
    }
  } catch (issue) { if (pageCurrent(request,null,key)) error.value = issue?.message || '资料列表读取失败，请稍后刷新。'; }
  finally { if (request === generation) loading.value = false; }
}
async function changeSession(id) { if (loading.value || uploading.value || busyDocument.value || copyingExport.value || refreshingStatuses.value) return; notice.value = ''; clearFileFilters(); closeReader(); exportPreview.value = ''; exportMarkdown.value = ''; setActiveSession(id); await reload(); }
async function upload(kind = 'document') {
  if (!visible || !sessionId.value || uploading.value || loading.value || busyDocument.value || copyingExport.value || refreshingStatuses.value || (researchEnabled.value && !uploadsAllowed.value)) return;
  stopStatusTracking();
  const session = sessionId.value, request = generation, operation = ++uploadGeneration, key = researchConnectionKey();
  uploading.value = true; error.value = ''; notice.value = '';
  try {
    const options = {kind,isCurrent:() => pageCurrent(request,session,key)};
    const result = researchEnabled.value ? await research.upload(session,options) : await api.upload(session,options);
    if (!pageCurrent(request,session,key)) return;
    if (result?.cancelled) { notice.value = '已取消，没有上传文件。'; return; }
    clearFileFilters();
    notice.value = preferences.value.mode === 'demo' ? '已添加演示资料，没有上传真实文件。' : '资料已提交。解析可能需要一些时间，请查看下方状态。';
    if (researchEnabled.value && result?.document) {
      const index = files.value.findIndex(file=>file.id === result.document.id);
      if (index < 0) files.value = [result.document,...files.value]; else files.value[index] = result.document;
      closeReader(); exportPreview.value = ''; exportMarkdown.value = '';
      if (result.document.status === 'ready') notice.value = '资料已准备好，可以开始提问。';
      if (result.document.status === 'failed') notice.value = '文件已收到，但暂时没能解析。请查看下方提示，没有自动重试。';
    } else { uploading.value = false; await reload(); }
  } catch (issue) { if (pageCurrent(request,session,key)) error.value = issue?.message || '上传未完成，请重试。'; }
  finally { if (operation === uploadGeneration) { uploading.value = false; if (visible && request === generation) trackProcessing(); } }
}
function openChat() { uni.switchTab({ url: '/pages/chat/index' }); }
function openReadyChat() {
  if (!visible || (preferences.value.connectionKind === 'invite' && !researchEnabled.value) || loading.value || uploading.value || !sessionId.value || !readyCount.value) return;
  setActiveSession(sessionId.value);
  openChat();
}
async function documentAction(action) {
  if (!visible || !researchEnabled.value || !sessionId.value || busyDocument.value || loading.value || uploading.value || refreshingStatuses.value || copyingExport.value) return;
  const request = generation, sid = sessionId.value, key = researchConnectionKey(); busyDocument.value = true; error.value = '';
  try { await action(sid,() => pageCurrent(request,sid,key)); }
  catch (issue) { if (pageCurrent(request,sid,key)) error.value = issue?.message || '资料操作未完成，请刷新确认；没有自动重试。'; }
  finally { if (request === generation) { busyDocument.value = false; activeDocument.value = ''; } }
}
async function inspectDocument(file) {
  if (!visible || !researchEnabled.value || !sessionId.value || busyDocument.value || loading.value || uploading.value || refreshingStatuses.value || copyingExport.value) return;
  if (isReading(file)) {
    sourceDetail.value = null; revealReader('#document-reader'); return;
  }
  await documentAction(async (sid,current) => { activeDocument.value = file.id; const result = await research.document(sid,file.id); if (!current()) return; documentDetail.value = result; sourceDetail.value = null; const index = files.value.findIndex(item=>item.id === result.id); if (index >= 0) files.value[index] = result; revealReader('#document-reader'); });
}
async function readSource(id) { await documentAction(async (sid,current) => { const result = await research.source(sid,id); if (current()) { sourceDetail.value = result; revealReader('#source-reader'); } }); }
function confirmRemoval(options) { return new Promise((resolve,reject) => { const timer = setTimeout(()=>reject(new Error('确认等待超时，没有移除资料')),60000); try { uni.showModal({...options,success:value=>{clearTimeout(timer);resolve(value);},fail:()=>{clearTimeout(timer);reject(new Error('没有确认移除，资料未变更'));}}); } catch { clearTimeout(timer); reject(new Error('当前平台无法确认移除，没有继续')); } }); }
async function removeDocument(file) { await documentAction(async (sid,current) => {
  const confirmation = await confirmRemoval({title:'移除当前会话资料？',content:`${file.filename}\n删除当前服务的原文件和片段，相关来源标记会被撤销；历史回答、确认内容、导出与备份副本不自动删除。此操作不能撤销。`,confirmText:'移除',cancelText:'保留'});
  if (!confirmation.confirm || !current()) return; await research.deleteDocument(sid,file.id); if (!current()) return;
  files.value = files.value.filter(item=>item.id !== file.id); closeReader(); exportPreview.value = ''; exportMarkdown.value = ''; notice.value = '资料已移除，不再参与新问题检索。历史回答与确认内容仍保留。';
}); }
async function retryDocument(file) { if (!uploadsAllowed.value || file.status !== 'failed') return; await documentAction(async (sid,current) => {
  const confirmation = await confirmRemoval({title:'重新解析这份资料？',content:'只在当前服务重新解析原文件，不调用模型。文件格式、编码或资源限制未改变时仍可能失败；不会自动循环重试。',confirmText:'重新解析',cancelText:'取消'});
  if (!confirmation.confirm || !current()) return; const result = await research.retryDocument(sid,file.id); if (!current()) return;
  const index = files.value.findIndex(item=>item.id === file.id); if (index >= 0) files.value[index] = result; closeReader(); notice.value = '已提交重新解析。本页会短暂自动更新状态，没有调用模型。'; trackProcessing();
}); }
async function previewExport() { await documentAction(async (sid,current) => { const result = await research.export(sid); if (current()) { exportScope = {request:generation,sid,key:researchConnectionKey()}; exportMarkdown.value = result; exportPreview.value = result.slice(0,8000); exportPartIndex.value = 0; exportPartsOpen.value = false; } }); }
async function copyExport(index = 0) {
  if (!visible || copyingExport.value || !researchEnabled.value || !currentExport() || !Number.isInteger(index) || !exportParts.value[index]) return;
  const request = generation, sid = sessionId.value, key = researchConnectionKey(), content = exportParts.value[index], count = exportParts.value.length;
  copyingExport.value = true; error.value = ''; notice.value = '';
  try { await new Promise((resolve,reject) => {
    const timer = setTimeout(()=>reject(new Error('复制等待超时，请展开对应复制段并手动选择文字。')),8000);
    try { uni.setClipboardData({data:content,showToast:false,success:()=>{clearTimeout(timer);resolve();},fail:()=>{clearTimeout(timer);reject(new Error('复制未完成，请展开对应复制段并手动选择文字。'));}}); }
    catch { clearTimeout(timer); reject(new Error('复制不可用，请展开对应复制段并手动选择文字。')); }
  }); if (pageCurrent(request,sid,key)) notice.value = count === 1 ? '已复制完整 Markdown，请检查隐私内容后再分享。' : `已复制第 ${index + 1} / ${count} 段，请按顺序粘贴并继续下一段。`; }
  catch (issue) { if (pageCurrent(request,sid,key)) { exportPartsOpen.value = true; error.value = issue?.message || '复制不可用，完整内容仍在对应复制段中。'; } }
  finally { if (request === generation) copyingExport.value = false; }
}
async function downloadExport() {
  if (!visible || copyingExport.value || !researchEnabled.value || !currentExport() || !exportMarkdown.value) return;
  const request = generation, sid = sessionId.value, key = researchConnectionKey(), content = exportMarkdown.value;
  const current = () => pageCurrent(request,sid,key) && exportMarkdown.value === content;
  copyingExport.value = true; error.value = ''; notice.value = '';
  try {
    const confirmed = await confirmRemoval({title:'下载完整研究记录？',content:'将资料目录和确认记忆保存为本地 Markdown 文件，可能含个人内容。不会自动上传或公开分享；本地副本不会随服务端删除而自动撤销。',confirmText:'下载',cancelText:'取消'});
    if (!confirmed.confirm || !current()) return;
    downloadResearchMarkdown(content,sid,{isCurrent:current});
    if (current()) notice.value = '已请求浏览器下载完整 Markdown，请查看下载列表；若未保存，可使用分段查看与复制。';
  } catch { if (current()) { exportPartsOpen.value = true; error.value = '下载未完成，可用分段查看与复制，完整内容已保留。'; } }
  finally { if (request === generation) copyingExport.value = false; }
}
function hide() { visible = false; generation += 1; uploadGeneration += 1; stopStatusTracking(); clearFileFilters(); loading.value = false; uploading.value = false; busyDocument.value = false; copyingExport.value = false; capabilityReady.value = false; researchEnabled.value = false; activeDocument.value = ''; files.value = []; sessions.value = []; sessionId.value = ''; closeReader(); exportPreview.value = ''; exportMarkdown.value = ''; }
onShow(() => { visible = true; notice.value = ''; void reload(); }); onHide(hide); onUnload(hide);
</script>

<style>
@import '../../components/mobile.css';
.library-workspace,.library-catalog,.library-reader-column { min-width:0; }
.export-part-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
.reader-empty { display:none; }
.reader-heading { align-items:flex-start; margin:0 0 16px; }
.reader-heading .section-title { display:block; overflow-wrap:anywhere; }
.reader-eyebrow { display:block; margin-bottom:6px; color:#75826f; font-size:11px; letter-spacing:1px; }
.reader-close { flex-shrink:0; }
.reader-navigation { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:-8px -8px 12px; }
.library-upload-card { padding: 24px; border: 1px solid #dce3d5; border-radius: 24px; background: #e6ecdd; }
.upload-card-heading { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
.upload-mark { display: flex; align-items: center; justify-content: center; flex-shrink: 0; width: 44px; height: 48px; border: 1px solid #cbd6c3; border-radius: 12px; background: #f6f8f0; color: #203d35; font-size: 28px; }
.upload-card-title { display: block; color: #203d35; font-size: 18px; font-weight: 650; }
.upload-card-subtitle { display: block; margin-top: 4px; color: #657566; font-size: 12px; line-height: 1.6; }
.upload-action { justify-content: space-between; min-height: 50px; padding: 14px 20px; border-radius: 15px; }
.image-upload-action { margin-top:10px; min-height:48px; }
.upload-note { display: block; margin: 13px 0 0; color: #627360; font-size: 11px; line-height: 1.8; }
.platform-note { display: block; margin-top: 8rpx; }
.library-heading { margin-top: 28px; margin-bottom: 14px; }
.library-count { display: block; margin-top: 4px; color: #697660; font-size: 12px; }
.refresh-action { min-height: 44px; padding: 8px 0 8px 12px; color: #496250; font-size: 12px; }
.library-next-action { margin:0 0 18px; }
.library-next-action .secondary-button { justify-content:space-between; min-height:48px; background:#edf3e4; border-color:#d7e2ca; }
.library-next-note { display:block; margin:8px 3px 0; color:#5f6f56; font-size:11px; line-height:1.8; }
.library-processing { display:flex; align-items:center; gap:14px; padding:14px 16px; margin:0 0 18px; border:1px solid #ece2c6; border-radius:16px; background:#faf6ea; }
.processing-title { display:block; color:#6e5d34; font-size:13px; font-weight:600; }
.processing-copy { display:block; margin-top:4px; color:#7c7259; font-size:12px; line-height:1.7; }
.processing-indicator { margin-left:auto; color:#8a703b; font-size:24px; }
.library-tools { margin-bottom:16px; }
.library-search { display:flex; align-items:center; gap:8px; padding:0 12px; border:1px solid #dfe5d7; border-radius:14px; background:#fff; }
.library-search-input { flex:1; min-width:0; height:44px; color:#263b30; font-size:13px; }
.library-filters { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
.library-filter { margin:0; padding:7px 11px; min-height:40px; color:#66725f; background:#f0f2e9; border:1px solid transparent; border-radius:11px; font-size:11px; line-height:1.6; }
.library-filter.is-selected { color:#244333; background:#e4edd9; border-color:#c9dabc; font-weight:600; }
.document-card { padding: 19px; margin-bottom: 12px; border: 1px solid #e6e8df; border-radius: 20px; background: #fff; }
.document-card.is-reading { border-color:#7c9873; background:#f8fbf3; box-shadow:inset 3px 0 #79996c; }
.document-row { display: flex; align-items: flex-start; gap: 14px; }
.document-symbol { flex-shrink: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; width: 46px; height: 58px; border-radius: 12px; color: #526953; background: #f0f3e9; }
.document-glyph { font-size: 25px; line-height: 1.1; }
.document-extension { margin-top: 4px; font-size: 9px; font-weight: 700; letter-spacing: .7px; }
.document-info { min-width: 0; flex: 1; }
.document-info .row-title { color: #25372f; font-size: 15px; line-height: 1.6; font-weight: 600; }
.document-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 8px; }
.file-status { padding: 3px 8px; border-radius: 6px; font-size: 10px; color: #72796e; background: #f0f1ec; }
.file-ready { color: #3f6342; background: #eaf2df; }
.file-failed { color: #954f40; background: #fcf0e9; }
.file-processing { color: #8a703b; background: #f8f2df; }
.file-size { color: #697660; font-size: 11px; }
.document-description { display: block; padding-top: 14px; margin-top: 14px; border-top: 1px solid #f1f2ec; color: #65745e; font-size: 12px; line-height: 1.7; }
.file-detail { display: block; margin-top: 10px; color: #8b5d5d; font-size: 12px; line-height: 1.7; word-break: break-word; }
.list-footnote { display: block; margin: 20px 4px; color: #6b7861; font-size: 12px; line-height: 1.8; }
.library-empty { padding: 38px 12px; border: 1px dashed #d9dfce; border-radius: 22px; }
.empty-paper { width: 52px; height: 60px; margin: 0 auto 20px; border-radius: 12px; background: #e6ecdd; color: #73816b; font-size: 34px; line-height: 60px; }
.library-empty .empty-title { font-size: 17px; }
.research-copy { display:block; margin:9px 0; color:#65745e; font-size:12px; line-height:1.8; overflow-wrap:anywhere; }
.research-document-actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px; }
.research-chunk-count { flex:1; font-size:11px; color:#65745e; }
.destructive-button { color:#995a4d; }
.research-reader,.research-export { padding:20px; margin:18px 0; border:1px solid #dae4cd; border-radius:22px; background:#fff; }
.research-source { padding:14px 0; border-top:1px solid #e8eddf; }
.source-location { display:block; font-size:11px; color:#557a49; overflow-wrap:anywhere; }
.source-excerpt { display:block; padding:15px; margin:10px 0; border-radius:12px; background:#f5f7ee; color:#2b4535; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; font-size:13px; line-height:1.9; }
/* #ifdef H5 */
@media (min-width:1100px) {
  .documents-page.mobile-page.has-research-library { max-width:1360px; }
  .library-workspace.has-research-reader { display:grid; grid-template-columns:minmax(290px, .9fr) minmax(0, 1.2fr); gap:24px; align-items:start; }
  .library-reader-column { position:sticky; top:24px; max-height:calc(100vh - 48px); max-height:calc(100dvh - 48px); overflow-y:auto; overscroll-behavior:contain; scrollbar-gutter:stable; border:1px solid #dae4cd; border-radius:22px; background:#fff; }
  .library-reader-column .research-reader { margin:0; padding:24px; border:0; border-radius:0; }
  .reader-empty { display:flex; min-height:460px; padding:40px 28px; flex-direction:column; align-items:center; justify-content:center; text-align:center; }
  .reader-empty-icon { display:flex; align-items:center; justify-content:center; width:62px; height:74px; margin-bottom:22px; border:1px solid #dce5d3; border-radius:16px; background:#f3f6ed; color:#84987b; font-size:36px; }
  .reader-empty-title { color:#354b3b; font-size:21px; font-weight:600; }
  .reader-empty-copy { display:block; max-width:290px; margin-top:12px; color:#72816c; font-size:13px; line-height:1.9; }
  .reader-empty-note { display:block; margin-top:28px; color:#87917f; font-size:11px; }
  .library-catalog .library-upload-card { padding:20px; }
  .library-catalog .upload-card-heading { gap:12px; }
  .library-catalog .upload-card-title { font-size:17px; }
  .library-catalog .document-card { padding:16px; }
  .library-catalog .research-document-actions { gap:3px; }
  .library-catalog .research-chunk-count { flex-basis:100%; margin-top:3px; }
  .library-catalog .research-document-actions .quiet-button { font-size:12px; padding-left:8px; padding-right:8px; }
  .research-export { margin-top:28px; }
  .research-reader:focus,.document-card:focus { outline:2px solid #859e77; outline-offset:3px; }
}
/* #endif */
@media (max-width: 360px) { .library-upload-card { padding: 18px; } .upload-card-heading { gap: 10px; } .upload-card-title { font-size: 16px; } }
</style>
