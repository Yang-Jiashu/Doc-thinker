<template>
  <view class="mobile-page chat-page">
    <BrandHeader />
    <view v-if="!messages.length && !loading" class="first-value-entry">
      <view class="first-value-intro"><text class="first-value-eyebrow">你的判断 · 你的资料 · 看得见的依据</text><text class="first-value-title">你的判断，<text class="first-value-highlight">原文支持吗？</text></text><text class="first-value-subtitle">带上论文、报告或笔记，一起核对出处、条件和证据缺口。</text></view>
      <view class="first-value-route" aria-label="加入自己的资料、提出自己的问题、核对原文"><text>01 加入资料</text><text aria-hidden="true">→</text><text>02 问你的问题</text><text aria-hidden="true">→</text><text>03 点开原文</text></view>
    </view>
    <ModeBanner compact :mode="preferences.mode" :connection-kind="preferences.connectionKind" :invite-pending="preferences.invitePending" :research-enabled="researchEnabled" />
    <WorkspaceBar :sessions="sessions" :session-id="sessionId" :disabled="switching || networkBusy || loading || preparingRequest || asking || creating || checkingRequest || Boolean(pendingQuestion) || recording || audioReady || voiceBusy" allow-create @change="changeSession" @create="createSession" />
    <EvidenceIntake v-show="!messages.length && !loading" :session-id="intakeSessionId" :identity="workbenchIdentity" :enabled="researchEnabled" :uploads-allowed="researchUploadsAllowed" :mode="preferences.mode" :connection-kind="preferences.connectionKind" :invite-pending="preferences.invitePending" :active="pageVisible" :disabled="intakeBlocked" @busy="setEvidenceBusy" @dirty="intakeDirty = $event" @ready="evidenceReadyCount = $event" @navigate="navigateEvidence" @ask="focusComposer" />
    <ProjectWorkbench compact :session-id="sessionId" :identity="workbenchIdentity" :demo="preferences.mode === 'demo'" :answers="messages" :disabled="switching || taskBusy || feedbackBusy || evidenceBusy || loading || preparingRequest || asking || creating || checkingRequest || Boolean(pendingQuestion) || recording || audioReady || voiceBusy" @draft="fillDraft" @busy="setWorkbenchBusy" @dirty="workbenchDirty = $event" />
    <ProjectTasks v-if="researchEnabled" :session-id="sessionId" :identity="workbenchIdentity" :active="pageVisible" :demo="preferences.mode === 'demo'" :disabled="switching || workbenchBusy || feedbackBusy || evidenceBusy || loading || preparingRequest || asking || creating || checkingRequest || Boolean(pendingQuestion) || recording || audioReady || voiceBusy" @busy="setTaskBusy" @dirty="taskDirty = $event" />
    <view v-if="receipt" class="execution-receipt" role="status" aria-live="polite"><view class="receipt-dot" /><view><text class="receipt-title">{{ receipt.title }}</text><text class="receipt-copy">{{ receipt.detail }}</text><text v-if="pendingQuestion?.requestId" class="receipt-number">请求编号：{{ pendingQuestion.requestId }}</text></view></view>
    <view v-if="researchEnabled && sessionId && messages.length" class="research-chat-bar"><view><text class="research-chat-kicker">当前项目 · 依据优先</text><text class="research-chat-copy">原文、推测与确认记忆分开使用。</text></view><view class="research-chat-actions"><button class="quiet-button" :disabled="firstValueBlocked" @tap="openResearchLibrary">加入资料 ↗</button><button class="quiet-button" :aria-pressed="useProjectMemory" :disabled="preparingRequest || asking || loading || Boolean(pendingQuestion)" @tap="useProjectMemory = !useProjectMemory">确认记忆：{{ useProjectMemory ? '开' : '关' }}</button></view></view>
    <view v-if="error" class="error-box"><text>{{ error }}</text><button v-if="!asking" class="quiet-button" @tap="reload">刷新会话与历史</button></view>
    <view v-if="pendingQuestion && !asking" class="request-state-panel">
      <text class="request-state-kicker">本次提问 · {{ pendingQuestion.status === 'failed' ? '已结束，未生成完整回答' : pendingQuestion.status === 'running' ? '处理中' : pendingQuestion.status === 'not_found' ? '未找到执行记录' : '结果尚未确认' }}</text>
      <text class="request-state-copy">{{ pendingQuestion.status === 'failed' ? pendingQuestion.failure?.message : pendingQuestion.status === 'running' ? '后台仍在处理。下面的按钮只读取状态，不会再请求模型。' : pendingQuestion.status === 'not_found' ? '暂未找到这个编号的执行记录。可以用同一编号继续原问题，也可查看历史后再决定。' : '连接中断或后台重启后，无法确认模型是否已经执行。请先查询状态或查看历史，系统不会自动重发。' }}</text>
      <text v-if="pendingQuestion.status === 'failed'" class="request-state-copy">这次调用可能已消耗额度；可先缩小问题或联系服务管理员。再次点击发送属于新的提问，会再次使用模型额度。</text>
      <button v-if="pendingQuestion.status !== 'failed'" class="request-state-button" role="button" :disabled="checkingRequest || loading" @tap="checkPendingQuestion">{{ checkingRequest ? '正在读取状态…' : '查看这次提问的状态' }}</button>
      <button v-if="pendingQuestion.status === 'not_found' && pendingQuestion.question" class="request-state-secondary" role="button" :disabled="checkingRequest || loading" @tap="continuePendingQuestion">用同一编号继续原问题</button>
      <button v-if="pendingQuestion.status !== 'running'" class="request-state-secondary" role="button" :disabled="checkingRequest || loading" @tap="releasePendingQuestion">{{ pendingQuestion.status === 'failed' ? '返回草稿，暂不发送' : '查看历史后，准备新的提问' }}</button>
    </view>
    <view v-if="requestNotice" class="request-state-notice" role="status" aria-live="polite">{{ requestNotice }}</view>
    <view v-if="loading" class="status-box">正在读取会话与历史…</view>
    <view v-else-if="!messages.length && !sessionId" class="empty-state chat-welcome">
      <text class="empty-title">从你在意的事开始。</text><text class="empty-copy">{{ preferences.connectionKind === 'invite' ? '聊一个想法，拆解一个问题，或者一起比较不同观点。' : '聊一个想法，拆解一个问题，或者带上资料一起研究。' }}</text>
      <button v-if="!sessionId" class="secondary-button" :disabled="creating" @tap="createSession">{{ creating ? '正在新建…' : '新建一个会话' }}</button>
    </view>
    <view v-else-if="messages.length" class="messages">
      <view v-for="(message, index) in messages" :id="'message-' + index" :key="message.key || index" class="message" :class="message.role === 'user' ? 'message-user' : 'message-assistant'">
        <view class="message-byline"><image v-if="message.role !== 'user'" src="/static/brand/logo-32.png" class="message-mark" /><text class="message-role">{{ message.role === 'user' ? '你' : 'DocThinker' }}{{ preferences.mode === 'demo' && message.role !== 'user' ? ' · 模拟回答' : '' }}</text></view>
        <AnswerBody v-if="message.role === 'assistant'" :text="message.content" />
        <template v-else>
          <!-- #ifdef MP-WEIXIN -->
          <text class="message-content" user-select>{{ message.content }}</text>
          <!-- #endif -->
          <!-- #ifndef MP-WEIXIN -->
          <text class="message-content" selectable>{{ message.content }}</text>
          <!-- #endif -->
        </template>
        <AnswerEvidence v-if="message.role === 'assistant'" :sources="message.sources" :demo="preferences.mode === 'demo'" />
        <ReasoningEvidence v-if="message.role === 'assistant' && message.research_core" :reasoning="message.reasoning" :claim-audit="message.claim_audit" :sources="message.sources" :citation-validation="message.citation_validation" />
        <view v-if="message.role === 'assistant' && message.research_core" class="research-answer-status"><text>{{ researchAnswerNote(message) }}</text><text v-if="message.used_memory_count">参考了 {{ message.used_memory_count }} 条你确认的项目记忆。</text><text v-if="Number.isFinite(message.memory_as_of)">记忆按 {{ formatMemoryAsOf(message.memory_as_of) }} 的适用时间回看；代表目前所知的历史背景。</text><button v-if="researchEnabled && message.sources?.length" class="quiet-button" :disabled="asking || loading || Boolean(pendingQuestion)" @tap="prepareMemory(message)">留下一条已核对的结论 ↗</button></view>
        <KnowledgeSkillEvidence v-if="message.role === 'assistant' && message.research_core" :value="message.knowledge_skill" />
        <AnswerDeliverable v-if="message.role === 'assistant'" :answer="message.content" :question="index > 0 && messages[index - 1].role === 'user' ? messages[index - 1].content : ''" :sources="message.sources" :demo="preferences.mode === 'demo'" :disabled="asking || loading" :research-core="message.research_core === true" :citation-validation="message.citation_validation" :answer-mode="message.answer_mode" :retrieval-gaps="message.retrieval_gaps" :reasoning="message.reasoning" :claim-audit="message.claim_audit" :memory-as-of="message.memory_as_of" :knowledge-skill="message.knowledge_skill" />
        <AnswerFeedback v-if="message.role === 'assistant' && isFeedbackAnswerId(message.id, preferences.mode === 'demo')" :session-id="sessionId" :answer-id="message.id" :initial="message.feedback || null" :demo="preferences.mode === 'demo'" :disabled="networkBusy || asking || preparingRequest || loading || checkingRequest || Boolean(pendingQuestion) || voiceBusy" @busy="setFeedbackBusy" />
        <button v-if="preferences.connectionKind !== 'invite' && message.role === 'assistant' && message.content" class="quiet-button speech-button" :disabled="speakingIndex !== index && (voiceActionBlocked || recording || audioReady)" @tap="readAnswer(message.content, index)">{{ speakingIndex === index ? '■ 停止 AI 朗读' : '▷ AI 朗读' }}</button>
      </view>
      <view v-if="asking" id="message-pending" class="message message-assistant"><text class="message-role">DocThinker</text><text class="muted-text">{{ preferences.mode === 'demo' ? '正在组织模拟回答…' : preferences.connectionKind === 'invite' ? '正在回答…' : '正在查找依据并回答…' }}</text></view>
    </view>
    <view id="chat-composer" class="composer" :class="{'composer-focused':composerFocused}">
      <view class="composer-heading"><text class="composer-label">{{ messages.length ? '接着这个思路，继续聊。' : '想核对哪句话？也可以直接提问。' }}</text><text class="composer-tag">{{ preferences.mode === 'demo' ? '演示草稿' : '由你发送' }}</text></view>
      <button v-if="researchEnabled && !messages.length" class="quiet-button initial-memory-control" :aria-pressed="useProjectMemory" :disabled="preparingRequest || asking || loading || Boolean(pendingQuestion)" @tap="useProjectMemory = !useProjectMemory">参考确认记忆：{{ useProjectMemory ? '开' : '关' }}</button>
      <view v-if="knowledgeSkillsEnabled" class="knowledge-selection">
        <button class="knowledge-selection-button" :aria-pressed="knowledgeSkillMode === 'auto'" :disabled="firstValueBlocked" @tap="toggleKnowledgeSkills">知识辅助：{{ knowledgeSkillMode === 'auto' ? '自动' : '关闭' }}</button>
        <text v-if="knowledgeSkillMode === 'auto'">按任务查询可能相关的原创方法单元，加入上下文时展示依据和边界；历史原文另行标注，不自动写入记忆。</text>
      </view>
      <view v-if="researchEnabled && useProjectMemory" class="memory-time-controls">
        <picker mode="date" :value="memoryDate" :disabled="preparingRequest || asking || loading || Boolean(pendingQuestion)" @change="setMemoryDate"><view class="memory-time-button">{{ memoryDate ? `回看 ${memoryDate} 当天的记忆` : '记忆时间：现在' }}</view></picker><button v-if="memoryDate" class="quiet-button" :disabled="preparingRequest || asking || loading || Boolean(pendingQuestion)" @tap="memoryDate = ''">回到现在</button>
      </view>
      <textarea :key="composerRevision" v-model="draft" class="draft-input" aria-label="提问草稿" :focus="composerFocused" :maxlength="-1" :disabled="switching || creating || preparingRequest || asking || voiceBusy || recording" :auto-height="true" placeholder="写下一个问题，或一段还没想清楚的想法…" :show-confirm-bar="false" :adjust-position="true" :cursor-spacing="24" @focus="composerFocused = true" @blur="composerFocused = false" @keydown="composerKeydown" />
      <view v-if="draftSuggestion" class="draft-choice" role="status" aria-live="polite"><text class="draft-choice-title">保留你的思路，再决定怎么开始。</text><text class="draft-choice-copy">草稿已有内容。快捷问题不会直接覆盖，也不会自动发送。</text><text class="draft-choice-preview">{{ draftSuggestion }}</text><view class="draft-choice-actions"><button role="button" @tap="resolveDraftSuggestion('append')">追加到草稿</button><button role="button" @tap="resolveDraftSuggestion('replace')">替换草稿</button><button role="button" @tap="resolveDraftSuggestion('cancel')">保留原草稿</button></view></view>
      <view v-if="researchEnabled" class="evidence-check-action"><button class="quiet-button" role="button" :disabled="firstValueBlocked || !draft.trim()" @tap="prepareEvidenceCheck" @keydown="firstValueKeydown($event, 'check')">加上核对要求</button><text>整理为“依据 · 条件 · 缺口”的问题，检查草稿后再发送。</text></view>
      <view v-if="draftLength" class="draft-length" :class="{ 'draft-length-over': draftOverLimit }"><text>{{ draftLength }} / {{ MAX_DRAFT_LENGTH }} 字符{{ draftOverLimit ? ' · 已超限，完整文字已保留，请编辑后发送' : '' }}</text></view>
      <text v-if="(preferences.connectionKind === 'invite' || researchEnabled) && draftLength" class="draft-byte-hint" :class="{'draft-length-over':inviteQuestionOverLimit}">{{ Number.isFinite(draftBytes) ? draftBytes : '无效 Unicode' }} / {{ INVITE_MAX_QUESTION_BYTES }} UTF-8 字节{{ inviteQuestionOverLimit ? ' · 输入预算已超限，请编辑缩短，完整草稿仍保留' : '' }}</text>
      <text v-if="requestIdUnavailable" class="draft-byte-hint" role="status">本版尚未接入当前环境的安全请求编号，暂不能在这里发起新的研究提问。请用现代浏览器打开 HTTPS 网站；草稿仍可编辑，没有请求模型。</text>
      <view v-if="voiceNotice" class="voice-notice" :class="{'voice-is-recording':recording}" role="status" aria-live="polite">
        <view v-if="voiceStage" class="voice-stage-heading"><view class="voice-stage-dot" aria-hidden="true" /><text>{{ voiceStageTitle }}</text><text v-if="recording && preferences.mode !== 'demo'" class="voice-stage-limit">最长 60 秒</text></view>
        <text>{{ voiceNotice }}</text>
        <view v-if="voiceStage" class="voice-flow"><text :class="{'voice-flow-current':recording || voiceStage === 'preparing' || audioReady}">1 说出来</text><text aria-hidden="true">→</text><text :class="{'voice-flow-current':voiceStage === 'transcribing' || voiceStage === 'review'}">2 修改草稿</text><text aria-hidden="true">→</text><text>3 由你发送</text></view>
        <button v-if="canUndoVoiceDraft" class="quiet-button voice-undo" role="button" tabindex="0" @tap="undoVoiceDraft" @keydown="voiceKeydown($event, 'undo')">撤回这次转写</button>
      </view>
      <view class="composer-actions">
        <view class="inline-actions">
          <button v-if="preferences.connectionKind !== 'invite'" class="quiet-button voice-button" :class="{ 'is-recording': recording }" role="button" :tabindex="voiceActionBlocked ? -1 : 0" :disabled="voiceActionBlocked" @tap="toggleRecording" @keydown="voiceKeydown($event, 'record')">{{ voiceBusy ? voiceStage === 'transcribing' ? '转写中…' : '准备录音…' : recording ? '■ 结束并转文字' : audioReady ? '转成文字' : '◉ 说话转文字' }}</button>
          <button v-if="recording || audioReady || voiceBusy" class="quiet-button" role="button" tabindex="0" @tap="cancelVoice" @keydown="voiceKeydown($event, 'cancel')">取消</button>
        </view>
        <button class="primary-button send-button" role="button" tabindex="0" :disabled="!canSend" @tap="send" @keydown="sendFromKey">{{ preparingRequest ? '正在准备…' : asking ? '回答中' : '发送 ↑' }}</button>
      </view>
      <text v-if="sendBlockReason" class="send-block-reason" role="status">{{ sendBlockReason }}</text>
      <!-- #ifdef H5 -->
      <text class="keyboard-hint">Enter 换行 · Ctrl / ⌘ + Enter 发送</text>
      <!-- #endif -->
      <text class="composer-hint">{{ researchEnabled ? '直接提问、讨论或写作，后台按需查找资料和整理参考线索。原文与推测分开，重要结论请核对；只使用你确认的记忆，不自动保存回答。' : preferences.connectionKind === 'invite' ? '文字内测只携带最近最多 6 条、累计 8192 字节的对话历史；不是完整长期记忆。回答可能有误。' : '语音先转成草稿，再由你发送。记忆和回答都可以有误，请核对重要内容。' }}</text>
    </view>
    <view v-if="!messages.length && !sessionId" class="starter-section">
      <view class="starter-heading"><text>给思路一个起点</text><text class="starter-kicker">只起草，不自动发送</text></view>
      <view class="starter-grid">
        <button class="suggestion" :disabled="asking || recording || audioReady || voiceBusy || loading || creating || Boolean(pendingQuestion)" @tap="fillDraft('我想一起梳理一件事。请先问我目标、目前的进展和主要困难，再一起确定下一步。')"><view class="suggestion-top"><text class="suggestion-symbol">↗</text><text class="suggestion-arrow">＋</text></view><text class="suggestion-title">理清下一步</text><text class="suggestion-detail">从你的目标和困难开始</text></button>
        <button class="suggestion suggestion-connect" :disabled="asking || recording || audioReady || voiceBusy || loading || creating || Boolean(pendingQuestion)" @tap="fillDraft(preferences.connectionKind === 'invite' ? '我想比较几种观点。请先问我讨论的主题和不同主张，再一起梳理各自的前提、优点与局限。' : '请比较当前会话资料中的主要观点、证据和分歧，并提出可验证的研究问题；明确区分原文与假设。')"><view class="suggestion-top"><text class="suggestion-symbol">⌘</text><text class="suggestion-arrow">＋</text></view><text class="suggestion-title">{{ preferences.connectionKind === 'invite' ? '比较不同观点' : '一起做研究' }}</text><text class="suggestion-detail">{{ preferences.connectionKind === 'invite' ? '把前提、优点与局限说清楚' : '连接观点，也寻找反证' }}</text></button>
      </view>
      <text class="grounding-note">有依据地回答 · 给推断留出边界</text>
    </view>
    <!-- #ifdef H5 -->
    <ComposerJump :active="pageVisible && messages.length > 0 && !loading" :disabled="switching || creating || preparingRequest || asking || voiceBusy || recording" @jump="focusComposer" />
    <!-- #endif -->
  </view>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue';
import { onHide, onShow, onUnload } from '@dcloudio/uni-app';
import { api, getPreferences, getActiveSession, setActiveSession, isFeedbackAnswerId, createQueryRequestIdAsync, queryRequestIdSupported, questionByteLength, INVITE_MAX_QUESTION_BYTES, researchAnswerMetadata, researchConnectionKey } from '../../services/api.js';
import { research } from '../../services/research.js';
import { boundedAnswerSources } from '../../services/research-source-contract.js';
import { startRecording, stopRecording, cancelRecording, speak, stopSpeaking } from '../../services/voice.js';
import ModeBanner from '../../components/ModeBanner.vue';
import WorkspaceBar from '../../components/WorkspaceBar.vue';
import BrandHeader from '../../components/BrandHeader.vue';
import AnswerEvidence from '../../components/AnswerEvidence.vue';
import AnswerBody from '../../components/AnswerBody.vue';
import ReasoningEvidence from '../../components/ReasoningEvidence.vue';
import AnswerDeliverable from '../../components/AnswerDeliverable.vue';
import KnowledgeSkillEvidence from '../../components/KnowledgeSkillEvidence.vue';
import AnswerFeedback from '../../components/AnswerFeedback.vue';
import ProjectWorkbench from '../../components/ProjectWorkbench.vue';
import ProjectTasks from '../../components/ProjectTasks.vue';
import EvidenceIntake from '../../components/EvidenceIntake.vue';
import { buildEvidenceCheckQuestion } from '../../services/evidence-check.js';
// #ifdef H5
import ComposerJump from '../../components/ComposerJump.vue';
// #endif
import {requestReceipt} from '../../services/project-workbench.js';

const preferences = ref({ mode: 'demo' });
const sessions = ref([]), sessionId = ref(''), messages = ref([]), draft = ref('');
const loading = ref(false), asking = ref(false), creating = ref(false), switching = ref(false), error = ref('');
const workbenchDirty = ref(false);
const taskBusy = ref(false), taskDirty = ref(false);
const pendingQuestion = ref(null), checkingRequest = ref(false), requestNotice = ref('');
const preparingRequest = ref(false), requestIdsSupported = ref(queryRequestIdSupported());
const workbenchBusy = ref(false), workbenchIdentity = ref(researchConnectionKey()), evidenceBusy = ref(false), evidenceReadyCount = ref(null), intakeDirty = ref(false), intakeSessionId = ref('');
const feedbackRequests = ref([]),feedbackBusy = computed(()=>feedbackRequests.value.length > 0),networkBusy=computed(()=>workbenchBusy.value || taskBusy.value || feedbackBusy.value || evidenceBusy.value);
let reloadAfterWorkbench = false;
const receipt = computed(()=>requestReceipt(pendingQuestion.value?.status,{asking:asking.value,preparing:preparingRequest.value,demo:preferences.value.mode === 'demo'}));
function finishNetworkTask() { if(!networkBusy.value && reloadAfterWorkbench && visible) {reloadAfterWorkbench=false;void reload();} }
function setWorkbenchBusy(value) { workbenchBusy.value = value === true;finishNetworkTask(); }
function setTaskBusy(value) { taskBusy.value = value === true;finishNetworkTask(); }
function setEvidenceBusy(value) { evidenceBusy.value = value === true;finishNetworkTask(); }
function setFeedbackBusy(value) { if(!value || typeof value.answerId!=='string')return;feedbackRequests.value=value.active===true?[...new Set([...feedbackRequests.value,value.answerId])]:feedbackRequests.value.filter(id=>id!==value.answerId);finishNetworkTask(); }
const draftSuggestion = ref(''), composerFocused = ref(false);
const composerRevision = ref(0);
const pageVisible = ref(false);
const researchEnabled = ref(false), researchUploadsAllowed = ref(false), researchMemoryWrite = ref(false), useProjectMemory = ref(true);
const memoryDate = ref('');
const knowledgeSkillsEnabled = ref(false), knowledgeSkillMode = ref('auto');
let knowledgeSkillScope = '', knowledgeSkillIdentity = '';
function toggleKnowledgeSkills() { if (!firstValueBlocked.value && knowledgeSkillsEnabled.value) knowledgeSkillMode.value = knowledgeSkillMode.value === 'auto' ? 'off' : 'auto'; }
function setMemoryDate(event) { if (!preparingRequest.value && !asking.value && !loading.value && !pendingQuestion.value && /^\d{4}-\d{2}-\d{2}$/.test(event?.detail?.value)) memoryDate.value = event.detail.value; }
function selectedMemoryTime() { if (!memoryDate.value) return null; const value = new Date(memoryDate.value+'T23:59:59'); return Number.isFinite(value.getTime()) ? value.getTime()/1000 : null; }
function formatMemoryAsOf(value) { const date = new Date(value*1000);return `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`; }
const recording = ref(false), audioReady = ref(false), voiceBusy = ref(false), voiceNotice = ref(''), speakingIndex = ref(-1);
const voiceStage = ref(''), lastVoiceDraft = ref(null);
const voiceStageTitle = computed(() => ({preparing:'准备语音输入',recording:preferences.value.mode === 'demo' ? '演示录音 · 未开启麦克风' : '正在录音',ready:'录音已停止 · 尚未转写',transcribing:'正在转成文字',review:'已转为草稿 · 等你检查'}[voiceStage.value] || ''));
const voiceActionBlocked = computed(() => !pageVisible.value || networkBusy.value || preparingRequest.value || asking.value || voiceBusy.value || loading.value || (!recording.value && (switching.value || creating.value || checkingRequest.value || Boolean(pendingQuestion.value) || Boolean(draftSuggestion.value))));
const canUndoVoiceDraft = computed(() => pageVisible.value && !voiceActionBlocked.value && !recording.value && !audioReady.value && Boolean(lastVoiceDraft.value) && draft.value === lastVoiceDraft.value?.after);
const intakeBlocked = computed(() => !pageVisible.value || switching.value || workbenchBusy.value || taskBusy.value || feedbackBusy.value || loading.value || preparingRequest.value || asking.value || creating.value || checkingRequest.value || Boolean(pendingQuestion.value) || recording.value || audioReady.value || voiceBusy.value || speakingIndex.value >= 0 || workbenchDirty.value || Boolean(draftSuggestion.value));
const firstValueBlocked = computed(() => intakeBlocked.value || evidenceBusy.value || intakeDirty.value);
const MAX_DRAFT_LENGTH = 6000;
const draftLength = computed(() => Array.from(draft.value).length);
const draftOverLimit = computed(() => draftLength.value > MAX_DRAFT_LENGTH);
const draftBytes = computed(() => questionByteLength(draft.value));
const inviteQuestionOverLimit = computed(() => (preferences.value.connectionKind === 'invite' || researchEnabled.value) && draftBytes.value > INVITE_MAX_QUESTION_BYTES);
const requestIdUnavailable = computed(() => preferences.value.mode === 'connected' && (preferences.value.connectionKind === 'invite' || researchEnabled.value) && !requestIdsSupported.value);
let generation = 0, voiceGeneration = 0, speechGeneration = 0, visible = false;
let draftService = '', draftSession = '', memoryPreferenceKey = '';
const canSend = computed(() => Boolean(draft.value.trim() && !switching.value && !networkBusy.value && !intakeDirty.value && !draftSuggestion.value && !draftOverLimit.value && !inviteQuestionOverLimit.value && !requestIdUnavailable.value && !preparingRequest.value && sessionId.value && !pendingQuestion.value && !checkingRequest.value && !asking.value && !loading.value && !creating.value && !voiceBusy.value && !recording.value && !audioReady.value));
const sendBlockReason = computed(() => {
  if (!draft.value.trim() || asking.value || preparingRequest.value || loading.value) return '';
  if (!sessionId.value) return '先点「新建」建立会话，草稿会保留，不会自动发送。';
  if (intakeDirty.value) return '上方还有未提交的原文。请先确认加入资料，或清空原文草稿，再发送问题。';
  if (draftSuggestion.value) return '先选择追加、替换或保留原草稿，再发送。';
  if (pendingQuestion.value) return '上一条提问还未确认，请先查看上方的请求状态。';
  if (audioReady.value) return '先把录音转成文字，或取消录音，再发送。';
  if (networkBusy.value) return '正在保存或读取项目内容，完成后即可发送。';
  return '';
});
const errorText = (issue, fallback) => typeof issue?.message === 'string' ? issue.message : fallback;
function starterAvailable() { return visible && !switching.value && !networkBusy.value && !preparingRequest.value && !asking.value && !recording.value && !audioReady.value && !voiceBusy.value && !loading.value && !creating.value && !pendingQuestion.value; }
async function focusComposer() { const request = generation; await nextTick(); if (!visible || request !== generation) return; composerFocused.value = true; if (typeof uni.pageScrollTo === 'function') { try { uni.pageScrollTo({selector:'#chat-composer',duration:180}); } catch { /* Draft selection must work even when platform scrolling is unavailable. */ } } }
function fillDraft(text) {
  if (!starterAvailable() || typeof text !== 'string' || !text.trim()) return;
  if (draft.value.trim() && draft.value !== text) draftSuggestion.value = text;
  else { draft.value = text; draftSuggestion.value = ''; }
  void focusComposer();
}
function resolveDraftSuggestion(action) {
  if (action === 'cancel') { draftSuggestion.value = ''; return; }
  if (!starterAvailable() || !draftSuggestion.value || !['append','replace'].includes(action)) return;
  draft.value = action === 'append' && draft.value ? `${draft.value}\n\n${draftSuggestion.value}` : draftSuggestion.value;
  draftSuggestion.value = ''; void focusComposer();
}
function sendFromKey(event) { if (!visible || !event || event.repeat || event.ctrlKey || event.altKey || event.metaKey || !['Enter',' ','Spacebar'].includes(event.key)) return; event.preventDefault?.(); if (canSend.value) void send(); }
function composerKeydown(event) {
  if (!visible || !event || event.key !== 'Enter' || (!event.ctrlKey && !event.metaKey) || event.shiftKey || event.altKey || event.isComposing || event.keyCode === 229 || event.repeat) return;
  event.preventDefault?.();
  if (canSend.value) void send();
}
function prepareEvidenceCheck() { if (firstValueBlocked.value || !researchEnabled.value || !draft.value.trim()) return; draft.value=buildEvidenceCheckQuestion(draft.value); composerRevision.value++; void focusComposer(); }
function navigateEvidence(page) { if (firstValueBlocked.value) return; if (page === 'chat') { if (!sessionId.value) void createSession(); else void focusComposer(); } else if (page === 'documents') openResearchLibrary(); else if (page === 'settings') uni.switchTab({url:'/pages/settings/index'}); else if (page === 'invite') uni.navigateTo({url:'/pages/invite/index'}); }
function firstValueKeydown(event, action) {
  if (!event || event.repeat || event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || !['Enter',' ','Spacebar'].includes(event.key)) return;
  event.preventDefault?.();
  if (action === 'check') prepareEvidenceCheck();
  else if (action === 'materials') openResearchLibrary();
}
function openResearchLibrary() { if (!firstValueBlocked.value && researchEnabled.value && sessionId.value) uni.switchTab({url:'/pages/documents/index'}); }
function researchAnswerNote(message) {
  const check = message.citation_validation;
  if (check?.warnings?.some(item=>item.code === 'source_revoked')) return '部分来源已被删除，相关结论无法继续核对。';
  if (check?.warnings?.some(item=>item.code === 'provenance_unavailable')) return '依据记录不可用，请勿将回答当作已核验的资料结论。';
  if (check?.unknown_labels?.length || check?.missing_citations) return '引用不完整或标签不在本次来源内，请核对，不建议直接记为事实。';
  if (message.answer_mode === 'insufficient_evidence') return '资料不足，本次没有请求模型生成资料结论。';
  if (message.retrieval_gaps?.some(item=>['graph_capacity_reached','graph_path_window_incomplete'].includes(item.code))) return '部分关联原文未能完整进入本次上下文，不能据此声称推理链完整。';
  if (message.retrieval_gaps?.some(item=>['graph_counterevidence_in_path','graph_path_conditions_differ'].includes(item.code))) return '关联材料存在否定或条件差异，需要核对，不能直接串成因果链。';
  if (message.retrieval_gaps?.some(item=>['evidence_facets_uncovered','evidence_anchors_uncovered'].includes(item.code))) return '部分问题要点还缺少具体原文，不能用背景介绍代替，也不能据此补出结论。';
  if (message.retrieval_gaps?.some(item=>item.code === 'complete_statement_exceeds_budget')) return '部分完整原句或限定条件未能放入本次上下文，请回到原文核对。';
  if (message.retrieval_gaps?.some(item=>item.code === 'source_antecedent_unverified')) return '部分节选的“这个问题”等回指缺少可见对象；请回到原文核对，不能从下一实验猜前提。';
  if (message.retrieval_gaps?.some(item=>item.code === 'source_boundary_unverified')) return '部分节选没有可确认的句尾，可能遗漏条件；标点完整也不等于前提完整。';
  if (message.retrieval_gaps?.some(item=>item.code === 'literal_coverage_scan_limited')) return '词面选证达到扫描上限，没有查完全部资料；未找到不代表没有相关内容。';
  if (message.retrieval_gaps?.some(item=>item.code === 'semantic_retrieval_unavailable')) return '语义检索暂不可用，本次仅按词面查找，可能漏掉同义或跨语言的证据。';
  if (message.retrieval_gaps?.some(item=>item.code === 'semantic_retrieval_windowed')) return '语义检索达到扫描上限，不代表已查完全部资料。';
  if (message.retrieval_gaps?.some(item=>item.code === 'bounded_overview')) return '这里只抽取各资料的开头节选，不是完整全文总结。';
  if (message.retrieval_gaps?.some(item=>item.code === 'uncovered_query_terms')) return '部分问题用词未在本次节选匹配；这不是概念缺失或全文结论，查英文资料可带上原文术语。';
  if (message.retrieval_gaps?.some(item=>['context_budget_omitted','confirmed_memory_budget_omitted'].includes(item.code))) return '受输入预算限制，部分历史或确认记忆未加入本次回答；没有改写你的记忆。';
  if (message.retrieval_gaps?.some(item=>item.code === 'evidence_coverage_proxy')) return '找到相关片段不代表已找齐关键证据或验证结论。';
  if (!message.sources?.length && message.knowledge_skill?.status === 'ready') return '本次为回答提供了方法知识参考，未附你的资料原文；方法不证明对你情况的事实判断，也不代表模型已正确遵循。';
  return message.sources?.length ? '引用只标原文出处，不代表已经验证结论正确。' : '这是普通交流或确认记忆背景，不是原文证据。';
}
function prepareMemory(message) {
  if (!visible || networkBusy.value || !researchEnabled.value || preparingRequest.value || asking.value || loading.value || pendingQuestion.value || !sessionId.value) return;
  if (!researchMemoryWrite.value) { error.value = '当前服务没有开放确认记忆保存，没有提交内容。'; return; }
  const ids = (message.sources || []).map(source=>source.id).filter(id=>typeof id === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id));
  if (!ids.length) return;
  try {
    research.stageMemoryDraft({session_id:sessionId.value,text:'',kind:'finding',source_ids:ids});
    uni.switchTab({url:'/pages/memory/index'});
  } catch { error.value = '来源或连接已改变，请刷新依据后再留下结论；没有保存记忆。'; }
}
async function scrollToLast() {
  await nextTick();
  if (visible && typeof uni.pageScrollTo === 'function') uni.pageScrollTo({selector:asking.value ? '#message-pending' : `#message-${messages.value.length - 1}`,duration:180});
}
function stopAudio() {
  voiceGeneration += 1; speechGeneration += 1;
  cancelRecording(); stopSpeaking(); recording.value = false; audioReady.value = false; voiceBusy.value = false; speakingIndex.value = -1;
  voiceStage.value = ''; lastVoiceDraft.value = null;
}
async function reload() {
  if(networkBusy.value) {reloadAfterWorkbench=true;return;}
  const request = ++generation;
  preparingRequest.value = false; requestIdsSupported.value = queryRequestIdSupported();
  draftSuggestion.value = ''; composerFocused.value = false;
  preferences.value = getPreferences(); workbenchIdentity.value = researchConnectionKey(); loading.value = true; error.value = ''; messages.value = []; researchEnabled.value = false; researchMemoryWrite.value = false; researchUploadsAllowed.value = false; evidenceReadyCount.value = null;
  knowledgeSkillsEnabled.value = false;
  if (knowledgeSkillIdentity && knowledgeSkillIdentity !== researchConnectionKey()) knowledgeSkillMode.value = 'auto';
  knowledgeSkillIdentity = researchConnectionKey();
  const service = JSON.stringify([preferences.value.mode, preferences.value.baseUrl || '', preferences.value.connectionKind || '', preferences.value.inviteUser?.id || '']);
  const savedMemory = JSON.stringify([service,preferences.value.useMemory !== false]);
  if (savedMemory !== memoryPreferenceKey) { useProjectMemory.value = preferences.value.useMemory !== false; memoryPreferenceKey = savedMemory; }
  if (draftService && service !== draftService) { draft.value = ''; voiceNotice.value = ''; draftSession = ''; pendingQuestion.value = null; requestNotice.value = ''; checkingRequest.value = false; memoryDate.value = '';stopAudio(); }
  draftService = service; sessions.value = []; sessionId.value = '';
  try {
    // The gateway admits one active request per user. Probe, list and history
    // must be sequential; two harmless simultaneous reads still receive 429.
    const capability = preferences.value.mode === 'connected' ? await research.capabilities({refresh:true}) : {enabled:false};
    if (!visible || request !== generation) return;
    researchEnabled.value = capability.enabled === true;
    researchUploadsAllowed.value = capability.enabled === true && capability.uploads_allowed === true;
    researchMemoryWrite.value = capability.enabled === true && capability.memory_write_enabled === true;
    knowledgeSkillsEnabled.value = capability.enabled === true && capability.knowledge_skills === true;
    const available = await api.sessions();
    if (!visible || request !== generation) return;
    sessions.value = Array.isArray(available) ? available : [];
    const stored = String(getActiveSession() || '');
    sessionId.value = sessions.value.some(item => String(item.id) === stored) ? stored : String(sessions.value[0]?.id || '');
    const nextKnowledgeScope = `${researchConnectionKey()}|${sessionId.value}`;
    if (knowledgeSkillScope && nextKnowledgeScope !== knowledgeSkillScope) knowledgeSkillMode.value = 'auto';
    knowledgeSkillScope = nextKnowledgeScope;
    if (draftSession && draftSession !== sessionId.value) { draft.value = ''; voiceNotice.value = ''; pendingQuestion.value = null; requestNotice.value = ''; memoryDate.value = '';stopAudio(); }
    draftSession = sessionId.value;
    setActiveSession(sessionId.value);
    if (sessionId.value) {
      const history = await api.history(sessionId.value);
      if (!visible || request !== generation) return;
      messages.value = (Array.isArray(history) ? history : []).filter(item => item.role === 'user' || item.role === 'assistant').map(item => ({ ...item, content: String(item.content || ''),...researchAnswerMetadata(item),...(item.research_core === true ? {sources:boundedAnswerSources(item.sources,true)} : {}) }));
      if ((preferences.value.connectionKind === 'invite' || researchEnabled.value) && !pendingQuestion.value) {
        const previous = typeof api.pendingQueries === 'function' ? api.pendingQueries(sessionId.value)[0] : null;
        if (previous) pendingQuestion.value = {requestId:previous.request_id,status:previous.request_status,session:sessionId.value,question:'',contextEnabled:preferences.value.useContext};
      }
      await scrollToLast();
    }
  } catch (issue) { if (visible && request === generation) error.value = errorText(issue, '会话读取失败，请稍后重试。'); }
  finally { if (request === generation) { intakeSessionId.value = sessionId.value; loading.value = false; } }
}
async function confirmLeavingDraft() {
  if (!draft.value.trim() && !workbenchDirty.value && !taskDirty.value && !intakeDirty.value) return true;
  try {
    const result = await new Promise(resolve => uni.showModal({title:'离开当前会话？',content:'当前有未发送的问题、未提交的原文或未保存的项目简报、任务计划。切换后不会带到另一个会话，请先发送、保存或复制需要保留的内容。',confirmText:'放弃并继续',cancelText:'留在这里',success:resolve,fail:()=>resolve({confirm:false})}));
    return result?.confirm === true;
  } catch { return false; }
}
async function changeSession(id) {
  if (!visible || switching.value || networkBusy.value || preparingRequest.value || asking.value || creating.value || recording.value || audioReady.value || voiceBusy.value || pendingQuestion.value || checkingRequest.value || id === sessionId.value || !sessions.value.some(item=>String(item.id) === id)) return;
  const request = generation, identity = researchConnectionKey(); switching.value = true;
  try {
    if (!(await confirmLeavingDraft()) || !visible || request !== generation || identity !== researchConnectionKey()) return;
    stopAudio(); voiceNotice.value = ''; draft.value = ''; memoryDate.value = ''; setActiveSession(id); await reload();
  } finally { switching.value = false; }
}
async function createSession() {
  if (!starterAvailable() || checkingRequest.value) return;
  creating.value = true; error.value = ''; stopAudio();
  const request = generation, identity = researchConnectionKey();
  try {
    // A first draft belongs to the new conversation; an existing project's
    // unsaved work must be explicitly discarded before creating another one.
    if (sessionId.value && !(await confirmLeavingDraft())) return;
    if (!visible || request !== generation || identity !== researchConnectionKey()) return;
    let title = '新对话';
    if (researchEnabled.value) {
      const name = await new Promise(resolve => uni.showModal({title:'新建研究项目',editable:true,
        placeholderText:'例如：能源转型文献研究',confirmText:'建立项目',success:resolve,fail:()=>resolve({confirm:false})}));
      if (!visible || request !== generation || identity !== researchConnectionKey() || !name.confirm) return;
      title = String(name.content || '').trim();
      if (!title || Array.from(title).length > 80) { error.value = '项目名称需要 1–80 个字符，没有建立项目。'; return; }
    }
    const session = await api.createSession(title);
    if (!visible || request !== generation || identity !== researchConnectionKey()) return;
    setActiveSession(String(session.id)); await reload();
  } catch (issue) { if (visible && request === generation) error.value = errorText(issue, '新建失败，请重试。'); }
  finally { creating.value = false; }
}
async function send() {
  if (!visible || !canSend.value) return;
  const request = generation, session = sessionId.value, identity = researchConnectionKey(), question = draft.value.trim();
  const context = preferences.value.useContext, memory = researchEnabled.value ? useProjectMemory.value : preferences.value.useMemory;
  const cache = preferences.value.useCache, evolution = preferences.value.useEvolution;
  const asOf = researchEnabled.value && memory ? selectedMemoryTime() : undefined;
  const selectedKnowledgeMode = researchEnabled.value && knowledgeSkillsEnabled.value ? knowledgeSkillMode.value : undefined;
  const current = () => visible && request === generation && session === sessionId.value && identity === researchConnectionKey() && question === draft.value.trim();
  let requestId = null;
  if (preferences.value.connectionKind === 'invite' || researchEnabled.value) {
    preparingRequest.value = true;
    try { requestId = await createQueryRequestIdAsync({isCurrent:current}); }
    catch { if (current()) error.value = '安全请求编号未能生成，请稍后手动重试或使用现代浏览器打开 HTTPS 网站。草稿已保留，没有发送问题。'; return; }
    finally { if (request === generation) preparingRequest.value = false; }
  }
  if (!current()) return;
  // New chat and writing requests let the server organise evidence in the
  // background. Keep selectedMode below only for an already accepted legacy
  // continuation: changing its fingerprint could otherwise repeat model work.
  await submitQuestion(question,requestId,context,memory,cache,evolution,undefined,asOf,selectedKnowledgeMode);
}
async function submitQuestion(question, requestId, contextEnabled, memoryEnabled = researchEnabled.value ? useProjectMemory.value : preferences.value.useMemory,
  cacheEnabled = preferences.value.useCache, evolutionEnabled = preferences.value.useEvolution, selectedMode, memoryAsOf, selectedKnowledgeMode) {
  stopAudio(); error.value = ''; voiceNotice.value = '';
  requestNotice.value = '';
  const session = sessionId.value, request = generation;
  if (requestId) pendingQuestion.value = {requestId,question,session,status:'running',contextEnabled,memoryEnabled,cacheEnabled,evolutionEnabled,selectedMode,memoryAsOf,selectedKnowledgeMode};
  const pending = { role: 'user', content: question, key: `pending-${Date.now()}` };
  messages.value.push(pending); asking.value = true; await scrollToLast();
  try {
    const result = requestId ? await api.ask(session, question, {requestId,useConversationContext:contextEnabled,useProjectMemory:memoryEnabled,
      useLLMCache:cacheEnabled,useSelfEvolution:evolutionEnabled,...(selectedMode !== undefined ? {reasoningMode:selectedMode} : {}),...(memoryAsOf !== undefined ? {memoryAsOf} : {}),...(selectedKnowledgeMode !== undefined ? {knowledgeSkillMode:selectedKnowledgeMode} : {})}) : researchEnabled.value ? await api.ask(session, question, {useProjectMemory:memoryEnabled,...(selectedMode !== undefined ? {reasoningMode:selectedMode} : {}),...(memoryAsOf !== undefined ? {memoryAsOf} : {}),...(selectedKnowledgeMode !== undefined ? {knowledgeSkillMode:selectedKnowledgeMode} : {})}) : await api.ask(session, question);
    if (!visible || request !== generation || session !== sessionId.value) return;
    if (requestId && result?.request_id !== requestId) throw new Error('请求回执无法确认，请查询状态或查看历史；没有自动重发。');
    if (!requestId && result?.request_status === 'failed') {
      messages.value = messages.value.filter(item => item.key !== pending.key);
      requestNotice.value = `${result.request_error.message} 这次调用可能已消耗额度；再次点击发送会发起新请求。`;
      return;
    }
    if (requestId && ['running','unknown','failed'].includes(result?.request_status)) {
      pendingQuestion.value = {...pendingQuestion.value,status:result.request_status,...(result.request_status === 'failed' ? {failure:result.request_error} : {})};
      messages.value = messages.value.filter(item => item.key !== pending.key);
      return;
    }
    if (requestId && (result?.request_id !== requestId || result?.request_status !== 'completed')) throw new Error('请求回执无法确认，请查询状态或查看历史；没有自动重发。');
    const answer = String(result?.answer || '');
    if (!answer) throw new Error('服务未返回答案，草稿已保留，请稍后重试。');
    messages.value.push({ role: 'assistant', content: answer,
      id:isFeedbackAnswerId(result?.answer_id,preferences.value.mode === 'demo') ? result.answer_id : '',
      sources:boundedAnswerSources(result?.sources,result.research_core === true), ...researchAnswerMetadata(result) });
    pendingQuestion.value = null; draft.value = '';
  } catch (issue) {
    if (visible && request === generation) {
      messages.value = messages.value.filter(item => item.key !== pending.key);
      if (requestId && (!Number.isInteger(issue?.status) || issue.status >= 500 || [408,409,429].includes(issue.status))) {
        pendingQuestion.value = {...pendingQuestion.value,status:'unconfirmed',...(issue.status === 409 ? {question:''} : {})};
      }
      else pendingQuestion.value = null;
      error.value = `${errorText(issue, '发送失败。')} 草稿已保留；若连接中断，请先刷新历史确认是否已有回答，再重试。`;
    }
  } finally {
    if (request === generation) { asking.value = false; if (visible) await scrollToLast(); }
  }
}
async function checkPendingQuestion() {
  const current = pendingQuestion.value;
  if (!visible || networkBusy.value || !current || current.session !== sessionId.value || asking.value || checkingRequest.value || loading.value) return;
  const request = generation;
  const active = () => visible && request === generation && pendingQuestion.value?.requestId === current.requestId && sessionId.value === current.session;
  checkingRequest.value = true; error.value = ''; requestNotice.value = '';
  try {
    const result = await api.queryStatus(current.session,current.requestId);
    if (!active()) return;
    if (result.request_status !== 'completed') {
      pendingQuestion.value = {...current,status:result.request_status,...(result.request_status === 'failed' ? {failure:result.request_error} : {})};
      return;
    }
    const id = isFeedbackAnswerId(result.answer_id,false) ? result.answer_id : '';
    if (!id || !messages.value.some(item=>item.role === 'assistant' && item.id === id)) {
      if (current.question && !(messages.value.at(-1)?.role === 'user' && messages.value.at(-1)?.content === current.question)) messages.value.push({role:'user',content:current.question});
      messages.value.push({role:'assistant',id,content:result.answer,sources:boundedAnswerSources(result.sources,result.research_core === true), ...researchAnswerMetadata(result)});
    }
    if (current.question && draft.value.trim() === current.question) draft.value = '';
    pendingQuestion.value = null; requestNotice.value = '已取回之前的回答，没有再次请求模型。'; await scrollToLast();
  } catch { if (active()) error.value = '未能读取本次提问状态，请稍后手动查询或查看历史。没有发送新问题或自动重试。'; }
  finally { if (request === generation) checkingRequest.value = false; }
}
async function continuePendingQuestion() {
  const current = pendingQuestion.value;
  if (!visible || networkBusy.value || !current?.question || current.status !== 'not_found' || current.session !== sessionId.value || asking.value || checkingRequest.value || loading.value) return;
  await submitQuestion(current.question,current.requestId,current.contextEnabled,current.memoryEnabled,current.cacheEnabled,current.evolutionEnabled,current.selectedMode,current.memoryAsOf,current.selectedKnowledgeMode);
}
async function releasePendingQuestion() {
  const current = pendingQuestion.value;
  if (!visible || !current || current.status === 'running' || asking.value || checkingRequest.value || loading.value) return;
  if (current.status === 'failed') {
    pendingQuestion.value = null; error.value = '';
    requestNotice.value = '草稿已保留，没有重新发送。再次点击发送会创建新请求并使用模型额度。';
    return;
  }
  const request = generation;
  try {
    const confirmation = await new Promise((resolve,reject) => {
      if (typeof uni.showModal !== 'function') return reject(new Error('Confirmation unavailable'));
      uni.showModal({title:'准备新的提问？',content:'请先核对会话历史。旧请求可能已经使用模型额度或稍后完成；新的提问可能重复使用额度。这不会取消旧请求。',confirmText:'已核对，继续',cancelText:'先不继续',success:resolve,fail:reject});
    });
    if (confirmation?.confirm !== true || !visible || request !== generation || pendingQuestion.value?.requestId !== current.requestId) return;
    pendingQuestion.value = null; error.value = ''; requestNotice.value = '已结束本页追踪，没有取消或重发旧请求。下次发送会作为新的提问。';
  } catch { if (visible && request === generation) error.value = '无法确认此操作，请先查看历史。没有发送新问题。'; }
}
async function toggleRecording() {
  if (!visible) return;
  if (preferences.value.connectionKind === 'invite') { voiceNotice.value = '邀请内测暂未开放语音，请直接输入文字'; return; }
  if (voiceActionBlocked.value) return;
  if (!recording.value && draftLength.value >= MAX_DRAFT_LENGTH) {
    voiceNotice.value = audioReady.value
      ? `草稿已达到 ${MAX_DRAFT_LENGTH} 字符，请先编辑缩短，再点击转文字；录音仍保留，也可以取消。`
      : `草稿已达到 ${MAX_DRAFT_LENGTH} 字符，请先编辑缩短再录音，避免产生无法发送的转写。`;
    return;
  }
  if (!recording.value && researchEnabled.value && (!Number.isFinite(draftBytes.value) || draftBytes.value >= INVITE_MAX_QUESTION_BYTES)) {
    voiceNotice.value = audioReady.value
      ? '草稿已达到 UTF-8 字节预算或包含无效字符，请先编辑缩短再转文字；录音仍保留，也可以取消。'
      : '草稿已达到 UTF-8 字节预算或包含无效字符，请先编辑缩短再录音，避免额外转写费用。';
    return;
  }
  stopSpeaking(); speakingIndex.value = -1; error.value = '';
  const request = ++voiceGeneration; voiceBusy.value = true;
  lastVoiceDraft.value = null;
  if (recording.value || audioReady.value) {
    recording.value = false; audioReady.value = false; voiceStage.value = 'transcribing'; voiceNotice.value = '只转文字，不自动提问；已有草稿会保留。';
    try {
      const text = await stopRecording();
      if (!visible || request !== voiceGeneration) return;
      const transcript = String(text || '');
      if (transcript.trim()) {
        const before = draft.value;
        draft.value = draft.value ? `${draft.value}\n${transcript}` : transcript;
        lastVoiceDraft.value = {before,after:draft.value}; voiceStage.value = 'review';
        voiceNotice.value = draftOverLimit.value
          ? `转写已完整保留，合计 ${draftLength.value} 字符，超过 ${MAX_DRAFT_LENGTH} 字符上限。请编辑缩短后发送，没有自动提交。`
          : inviteQuestionOverLimit.value ? '转写已完整保留，但超过 UTF-8 字节预算或包含无效字符。请编辑缩短后发送，没有自动提交。'
          : '文字已完整放入草稿，和打字使用同一套问答流程。请检查识别结果，再点击发送。';
        void focusComposer();
      }
      else { voiceStage.value = ''; voiceNotice.value = '没有识别到文字，可重新录音或直接输入。'; }
    } catch (issue) { if (visible && request === voiceGeneration) { voiceStage.value = ''; voiceNotice.value = errorText(issue, '语音识别失败，可继续打字。'); } }
    finally { if (request === voiceGeneration) voiceBusy.value = false; }
  } else {
    voiceStage.value = 'preparing'; voiceNotice.value = preferences.value.mode === 'demo' ? '准备演示语音，不会使用麦克风。' : '正在检查语音服务与麦克风权限，可以随时取消。';
    let terminalState = false;
    try {
      await startRecording({ onState: event => {
        if (!visible || request !== voiceGeneration || terminalState) return;
        if (event?.state === 'recording') { recording.value = true; audioReady.value = false; voiceStage.value = 'recording'; }
        else if (event?.state === 'stopped') {
          terminalState = true; recording.value = false; audioReady.value = true; voiceBusy.value = false; voiceStage.value = 'ready';
          voiceNotice.value = event.reason === 'limit'
            ? '已到录音时长上限，录音已停止。点击「转成文字」生成草稿，也可以取消。'
            : '录音已停止。点击「转成文字」生成草稿，也可以取消。';
        } else if (event?.state === 'error') {
          terminalState = true; recording.value = false; audioReady.value = false; voiceBusy.value = false; voiceStage.value = '';
          voiceNotice.value = event.message || '录音已中断，可重新录音或继续打字。';
        }
      } });
      if (!visible || request !== voiceGeneration) return;
      if (terminalState) return;
      recording.value = true; voiceStage.value = 'recording';
      voiceNotice.value = preferences.value.mode === 'demo' ? '演示录音中，未使用麦克风。再次点击可生成示例草稿。' : '正在录音。说完点击「结束并转文字」；也可以取消。';
    } catch (issue) {
      if (visible && request === voiceGeneration) {
        recording.value = false; audioReady.value = false; voiceStage.value = '';
        if (!terminalState) voiceNotice.value = errorText(issue, '无法开始录音，请检查权限或继续打字。');
      }
    }
    finally { if (request === voiceGeneration) voiceBusy.value = false; }
  }
}
function cancelVoice() {
  if (!visible) return;
  const wasTranscribing = voiceStage.value === 'transcribing';
  voiceGeneration += 1; cancelRecording(); recording.value = false; audioReady.value = false; voiceBusy.value = false; voiceStage.value = ''; lastVoiceDraft.value = null;
  voiceNotice.value = wasTranscribing ? '已取消接收转写，没有发送提问。若转写已提交服务端，取消不能撤回已开始的处理。' : '已取消录音，原草稿保留，没有发送任何问题。';
}
function undoVoiceDraft() {
  if (!canUndoVoiceDraft.value) return;
  draft.value = lastVoiceDraft.value.before; lastVoiceDraft.value = null; voiceStage.value = '';
  // uni-app's focused H5 field skips a value equal to its last USER input.
  // Recreate only on this explicit undo so the visible text matches the draft.
  composerFocused.value = false; composerRevision.value += 1;
  voiceNotice.value = '已撤回这次转写，原草稿保留。'; void focusComposer();
}
function voiceKeydown(event, action) {
  if (!visible || !event || event.defaultPrevented || event.repeat || event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || !['Enter',' ','Spacebar'].includes(event.key)) return;
  const run = action === 'record' ? toggleRecording : action === 'cancel' ? cancelVoice : action === 'undo' ? undoVoiceDraft : null;
  if (!run) return; event.preventDefault?.(); void run();
}
async function readAnswer(text, index) {
  if (!visible) return;
  if (preferences.value.connectionKind === 'invite') return;
  const wasSpeaking = speakingIndex.value === index;
  if (!wasSpeaking && (voiceActionBlocked.value || recording.value || audioReady.value)) return;
  const request = ++speechGeneration; stopSpeaking(); speakingIndex.value = -1;
  if (wasSpeaking) return;
  speakingIndex.value = index;
  try { await speak(text); }
  catch (issue) { if (visible && request === speechGeneration) error.value = errorText(issue, '当前设备无法朗读，可直接阅读文字。'); }
  finally { if (request === speechGeneration) speakingIndex.value = -1; }
}
function hide() { visible = false; pageVisible.value = false; generation += 1; preparingRequest.value = false; draftSuggestion.value = ''; composerFocused.value = false; stopAudio(); if (pendingQuestion.value?.status === 'running') pendingQuestion.value = {...pendingQuestion.value,status:'unconfirmed'}; asking.value = false; checkingRequest.value = false; loading.value = false; voiceNotice.value = ''; }
onShow(() => { visible = true; pageVisible.value = true; void reload(); });
onHide(hide); onUnload(hide);
</script>

<style scoped>
@import '../../components/mobile.css';
.chat-page :deep(.brand-header){margin-bottom:8px}.chat-page :deep(.brand-topline){padding-bottom:8px;border-bottom:0}.chat-page .composer{padding:18px 20px 14px;border-radius:19px}.chat-page .composer-heading{margin-bottom:10px}.chat-page .draft-input{min-height:78px}.send-block-reason{display:block;margin-top:9px;font-size:12px;color:#85602a;line-height:1.7}.keyboard-hint{display:block;font-size:10px;color:#74806b;margin-top:9px}.chat-page .composer-hint{font-size:11px;padding-top:8px}.chat-page :deep(.workbench-compact){padding:0 14px;border-radius:13px}.chat-page :deep(.workspace-bar){margin-bottom:14px}.chat-page .research-chat-bar{margin:0 auto 14px;max-width:820px;padding:10px 14px;border-radius:13px}.chat-page .research-chat-copy{display:none}
@media(max-width:380px){.chat-page .composer{padding:15px}.chat-page :deep(.disclosure-hint){font-size:10px}.chat-page .composer-label{font-size:14px}}
.first-value-entry { max-width:820px; margin:8px auto 22px; }
.chat-page :deep(.evidence-intake) { max-width:820px; margin:0 auto 16px; }
.first-value-intro { margin-bottom:19px; }
.first-value-eyebrow { display:block; color:#68745d; font-size:11px; letter-spacing:1.2px; }
.first-value-title { display:block; margin-top:8px; color:#203d35; font-size:36px; font-weight:700; letter-spacing:-1.2px; line-height:1.35; }
.first-value-highlight { box-shadow:inset 0 -10px 0 #d6fa7580; }
.first-value-subtitle { display:block; margin-top:8px; color:#67725e; font-size:13px; line-height:1.7; }
.first-value-route { display:flex; flex-wrap:wrap; align-items:center; gap:11px; margin-top:14px; color:#4c6649; font-size:11px; line-height:1.8; }
.first-value-route text:nth-child(odd) { padding:5px 9px; background:#edf1e5; border:1px solid #dfe5d3; border-radius:8px; }
.evidence-check-action { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-top:10px; color:#69775f; font-size:10px; line-height:1.7; }
.evidence-check-action .quiet-button { margin:0; padding:8px 11px; background:#edf3df; border:1px solid #d8e2c5; border-radius:10px; font-size:12px; }
.initial-memory-control { margin:0 0 5px; padding:6px 0; justify-content:flex-start; font-size:11px; }
@media (max-width:600px) {
  .first-value-entry { margin-top:4px; margin-bottom:18px; }
  .first-value-intro { margin-bottom:16px; }
  .first-value-title { font-size:29px; letter-spacing:-1px; }
  .first-value-subtitle { font-size:12px; }
  .first-value-route { gap:8px; margin-top:12px; font-size:10px; }
}
@media (max-width:360px) { .first-value-title { font-size:26px; } .first-value-route { gap:4px; } .first-value-route text:nth-child(odd) { padding:5px 6px; } }
.research-chat-bar { display:flex; justify-content:space-between; gap:16px; align-items:center; margin:14px 0; padding:15px 18px; border:1px solid #d8dfc9; border-radius:18px; background:#edf2e2; }
.memory-time-controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:5px 0 12px; }
.memory-time-button { padding:8px 0; color:#50694a; font-size:11px; line-height:1.8; }
.research-chat-kicker { display:block; color:#32483d; font-size:13px; font-weight:650; }
.research-chat-copy { display:block; margin-top:4px; color:#627164; font-size:12px; line-height:1.6; }
.research-chat-actions { display:flex; flex-wrap:wrap; gap:5px; }
.research-answer-status { display:grid; gap:5px; margin-top:12px; padding-left:10px; border-left:2px solid #c9dcb7; color:#66705f; font-size:12px; line-height:1.7; }
.research-answer-status .quiet-button { margin:5px 0 0; color:#42563c; justify-self:start; }
@media (max-width:600px) { .research-chat-bar { align-items:flex-start; flex-direction:column; gap:8px; } }
.request-state-panel { margin:14px 0; padding:17px; border:1px solid #dedfc9; border-radius:18px; background:#f4f4e7; }
.request-state-kicker { display:block; color:#51623f; font-size:12px; font-weight:600; }
.request-state-copy { display:block; margin-top:7px; color:#73745d; font-size:12px; line-height:1.8; }
.request-state-button, .request-state-secondary { display:block; min-height:44px; margin:12px 0 0; padding:9px 12px; border:0; border-radius:11px; background:#203d35; color:#fff; font-size:12px; line-height:1.8; }
.request-state-secondary { margin-top:6px; background:transparent; color:#68735c; }
.request-state-button::after, .request-state-secondary::after { border:0; }
.request-state-notice { margin:12px 0; padding:12px 15px; border-radius:13px; background:#edf2e4; color:#596f4d; font-size:12px; line-height:1.8; }
.draft-byte-hint { display:block; margin:4px 0 9px; color:#78826e; font-size:10px; line-height:1.8; }
.chat-welcome { padding:20px 12px; }
.starter-section { margin-top:23px; }
.starter-heading { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:0 0 12px; color:#57694e; font-size:12px; }
.starter-kicker { font-size:8px; letter-spacing:1px; color:#7d8b72; }
.starter-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.suggestion { display:block; width:100%; margin:0; padding:14px 16px 17px; min-height:115px; border:1px solid #e0e4d5; border-radius:20px; background:#edeee4; color:#3c5137; line-height:1.5; text-align:left; }
.suggestion-connect { background:#e6ecdd; border-color:#dce4d1; }
.suggestion-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.suggestion-symbol { font-size:20px; line-height:24px; }
.suggestion-arrow { color:#809070; font-size:15px; }
.suggestion-title { display:block; font-size:14px; font-weight:650; }
.suggestion-detail { display:block; margin-top:4px; color:#6c7a61; font-size:10px; }
.grounding-note { display:block; margin:23px 0 4px; color:#7c8971; text-align:center; font-size:10px; letter-spacing:.5px; }
.messages { margin-bottom:16px; }
.message { margin:6px 0 20px; padding:12px 2px; }
.message-user { margin-left:25px; padding:15px 18px; border-radius:20px 20px 5px 20px; background:#e7eddd; }
.message-byline { display:flex; align-items:center; gap:6px; margin-bottom:9px; }
.message-mark { width:19px; height:19px; }
.message-role { display:block; color:#65775c; font-size:11px; font-weight:600; }
.message-content { display:block; font-size:14px; line-height:1.9; white-space:pre-wrap; overflow-wrap:anywhere; }
.speech-button { justify-content:flex-start; min-height:44px; padding-left:0; margin-top:7px; }
.composer { border:1px solid #d9e0ce; border-radius:22px; padding:17px 17px 12px; background:#fffefa; box-shadow:0 5px 20px #203d3505; }
.composer-label { display:block; font-size:13px; font-weight:600; color:#435b3b; margin-bottom:8px; }
.draft-input { display:block; width:100%; min-height:64px; max-height:180px; color:#25372f; font-size:15px; line-height:1.75; }
.draft-length { margin-top:6px; color:#748269; font-size:11px; line-height:1.7; }
.draft-length-over { color: #963e3e; }
.composer-actions { display:flex; justify-content:space-between; align-items:center; gap:4px; margin-top:10px; }
.composer-actions .inline-actions { gap:2px; }
.send-button { flex-shrink:0; min-width:66px; padding-left:12px; padding-right:12px; border-radius:14px; }
.voice-button { padding-left:0; font-size:12px; }
.voice-button.is-recording { color: #a23131; }
.voice-notice { padding:12px 14px; border:1px solid #dde5d6; border-radius:13px; background:#f4f7ef; color:#57694f; font-size:12px; line-height:1.8; margin-top:8px; }
.voice-stage-heading { display:flex; align-items:center; gap:8px; margin-bottom:7px; color:#304c36; font-size:13px; font-weight:600; }
.voice-stage-dot { width:7px; height:7px; flex-shrink:0; border-radius:50%; background:#799554; }
.voice-stage-limit { margin-left:auto; color:#6b7665; font-size:11px; font-weight:400; }
.voice-is-recording { border-color:#e4c9bd; background:#fff6ef; }
.voice-is-recording .voice-stage-dot { background:#b34f3a; }
.voice-flow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:9px; color:#72806b; font-size:11px; }
.voice-flow-current { color:#324e36; font-weight:600; }
.voice-undo { margin:6px 0 -4px; padding-left:0; color:#405c3c; font-size:12px; }
.composer-hint { display:block; color:#73816a; font-size:11px; line-height:1.7; margin-top:10px; border-top:1px solid #edf0e6; padding-top:10px; }
@media (max-width:360px) { .starter-kicker { display:none; } }

/* Product workbench: one readable task surface and a distinctive context motif. */
.chat-page { max-width:1080px; padding-bottom:48px; color:var(--dt-ink,#1c2723); }
.messages,.composer,.starter-section,.chat-welcome,.request-state-panel,.request-state-notice,.error-box,.status-box { max-width:820px; margin-left:auto; margin-right:auto; }
.chat-page :deep(.mode-banner),.chat-page :deep(.workspace-bar) { max-width:820px; margin-left:auto; margin-right:auto; }
.chat-page :deep(.mode-banner) { background:#f0f1e9; border-color:#dfe3d7; }
.chat-page :deep(.mode-label) { font-size:11px; color:#374438; }
.chat-page :deep(.mode-description) { color:#59665a; font-size:12px; }
.chat-page :deep(.mode-content) { display:flex; align-items:center; gap:12px; }
.chat-page :deep(.mode-description) { flex:1; min-width:0; }
.chat-page :deep(.mode-guide) { flex-shrink:0; width:auto; min-height:44px; margin:0; padding:7px 2px; border:0; background:transparent; color:#354a2c; }
.chat-page :deep(.workspace-bar) { border-radius:15px; background:#fffefa; border-color:#dfe3d9; }
.chat-page :deep(.workspace-create) { background:#edf4df; color:#263725; }
.chat-welcome { padding:14px 12px 20px; }
.empty-title { font-size:21px; letter-spacing:-.5px; }
.composer { padding:23px 24px 17px; border:1px solid #cfd6c8; border-radius:24px; background:#fffefa; box-shadow:0 8px 34px #1c272308; transition:border-color .18s,box-shadow .18s; }
.composer-focused { border-color:#70884b; box-shadow:0 0 0 3px #d6fa7542,0 8px 34px #1c272308; }
.composer-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:15px; }
.composer-label { margin:0; color:#26352b; font-size:16px; letter-spacing:-.3px; }
.composer-tag { flex-shrink:0; color:#61705c; font-size:11px; border:1px solid #e1e6d7; padding:3px 8px; border-radius:7px; background:#f5f7ef; }
.draft-input { min-height:84px; font-size:16px; color:#1c2723; line-height:1.8; }
.composer-actions { margin-top:12px; }
.composer-hint { color:#667263; font-size:12px; line-height:1.8; padding-top:13px; }
.send-button { min-height:46px; min-width:92px; border-radius:13px; background:#1c2723; border-color:#1c2723; color:#d6fa75; }
.voice-button { color:#52604e; }
.draft-byte-hint { color:#627059; font-size:11px; }
.draft-choice { margin:12px 0; padding:15px; border:1px solid #d8e4bf; border-radius:14px; background:#f4f8e9; }
.draft-choice-title { display:block; font-size:13px; color:#293d24; font-weight:650; }
.draft-choice-copy { display:block; margin-top:5px; color:#5c6a50; font-size:12px; line-height:1.8; }
.draft-choice-preview { display:block; margin-top:10px; font-size:12px; line-height:1.9; color:#4c5f3e; white-space:pre-wrap; overflow-wrap:anywhere; }
.draft-choice-actions { display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }
.draft-choice-actions button { min-height:44px; margin:0; padding:9px 12px; border:1px solid #d9e3ca; border-radius:10px; background:#fffefa; font-size:12px; color:#2d4128; line-height:24px; }
.draft-choice-actions button:first-child { background:#253b2a; color:#fff; border-color:#253b2a; }
.starter-section { margin-top:25px; }
.starter-heading { color:#455140; font-size:13px; margin-bottom:13px; }
.starter-kicker { font-size:11px; letter-spacing:0; color:#66735f; }
.starter-grid { gap:14px; }
.suggestion { padding:18px 20px; min-height:130px; background:#f1f2ea; border-color:#dce1d3; border-radius:18px; transition:transform .18s,border-color .18s; }
.suggestion-connect { background:#eaf0df; border-color:#d7e2c5; }
.suggestion-symbol { color:#273829; font-size:24px; }
.suggestion-arrow { color:#5b6e4c; }
.suggestion-title { color:#28382b; font-size:16px; }
.suggestion-detail { color:#5e6c54; font-size:12px; line-height:1.7; }
.grounding-note { color:#6c7866; font-size:11px; margin:18px 0 22px; }
.message { padding:18px 0; margin-bottom:22px; }
.message-user { margin-left:48px; padding:20px 22px; background:#eaf0e0; border:1px solid #e0e7d4; border-radius:22px 22px 5px 22px; }
.message-content { font-size:16px; color:#253229; line-height:1.95; }
.message-role { color:#5c6a56; font-size:12px; }
.message-mark { width:22px; height:22px; }
.request-state-panel { background:#f3f5e8; padding:20px; border-color:#d9e0c8; }
.request-state-kicker { color:#394b2a; font-size:13px; }
.request-state-copy { color:#5e6b50; font-size:13px; }
.request-state-button { background:#1c2723; color:#d6fa75; font-size:13px; }
.request-state-notice { font-size:13px; }
/* #ifdef H5 */
@media (hover:hover) { .suggestion:hover:not([disabled]) { transform:translateY(-3px); border-color:#819564; } }
/* #endif */
@media (min-width:900px) {
  .chat-page :deep(.brand-header) { margin-bottom:14px; }
  .composer { padding:20px 22px 14px; }
  .composer-heading { margin-bottom:12px; }
  .draft-input { min-height:48px; }
  .composer-actions { margin-top:8px; }
}
@media (max-width:480px) {
  .composer { padding:18px 16px 14px; border-radius:20px; }
  .composer-label { font-size:14px; }
  .composer-tag { font-size:10px; }
  .draft-input { font-size:16px; min-height:80px; }
  .suggestion { padding:16px 14px; min-height:122px; }
  .suggestion-title { font-size:15px; }
  .suggestion-detail { font-size:11px; }
  .starter-heading { font-size:12px; }
  .starter-kicker { font-size:10px; }
  .message-user { margin-left:18px; padding:17px; }
  .message-content { font-size:15px; }
}
@media (max-width:360px) { .starter-kicker { display:none; } }
@media (prefers-reduced-motion:reduce) { .composer,.suggestion { transition:none; } }
.execution-receipt{display:flex;gap:11px;max-width:820px;margin:0 auto 18px;padding:15px 17px;border:1px solid #d9e2cc;border-radius:14px;background:#edf3e3}.receipt-dot{flex-shrink:0;width:7px;height:7px;margin-top:8px;border-radius:50%;background:#658346}.receipt-title{display:block;color:#36532e;font-size:13px;font-weight:600}.receipt-copy,.receipt-number{display:block;font-size:11px;line-height:1.85;color:#607453}.receipt-number{overflow-wrap:anywhere;font-variant-numeric:tabular-nums;margin-top:5px}
@media(max-width:759px){.keyboard-hint{display:none}}
.knowledge-selection{margin:8px 0 10px}.knowledge-selection-button{display:inline-block;min-height:44px;margin:0;box-sizing:border-box;padding:10px 12px;border:1px solid #e4ddc8;border-radius:12px;background:#f8f5ec;color:#77603d;font-size:12px;line-height:22px}.knowledge-selection-button::after{border:0}.knowledge-selection-button[disabled]{opacity:.5}.knowledge-selection>text{display:block;margin:5px 2px;color:#84755b;font-size:11px;line-height:1.8}
</style>
