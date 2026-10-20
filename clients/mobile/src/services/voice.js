import { connection, getPreferences, request, uploadFile } from './api.js';
import { ensureWeChatPrivacy, ensureWeChatRecordingPermission } from './wechat-privacy.js';

let epoch = 0;
let recording = null;
let consentDestination = '';
let speechEpoch = 0;
let player = null;
let cleanupAudio = null;
let settlePlayback = null;
let nativeRecorder = null;
let nativeBusy = false;
let cancelSpeechDownload = null;

function notifyRecording(op, state, reason, message) {
  if (recording !== op || op.cancelled || op.epoch !== epoch) return;
  try { op.onState?.({state,reason,message}); } catch { /* UI observers cannot break resource cleanup. */ }
}

function cancelled() { return new Error('录音已取消'); }
function matchesDestination(destination) {
  try {
    const preferences = getPreferences();
    if (preferences.mode !== 'connected' || preferences.connectionKind === 'invite') return false;
    const active = connection();
    return typeof destination?.url === 'string' && destination.url.length > 0
      && typeof destination?.header?.Authorization === 'string' && destination.header.Authorization.length > 0
      && active?.url === destination.url && active?.header?.Authorization === destination.header.Authorization;
  } catch { return false; }
}
async function consent() {
  const destination = connection().url;
  if (consentDestination === destination) return;
  const result = await new Promise(resolve => uni.showModal({
    title:'语音隐私提示',
    content:`录音将发送到 ${getPreferences().baseUrl}，并由该服务配置的语音提供商处理。仅用于转文字，识别后由你确认发送；请勿录入敏感资料。`,
    confirmText:'同意录音',cancelText:'取消',success:resolve,fail:() => resolve({confirm:false}),
  }));
  if (!result.confirm) throw cancelled();
  consentDestination = destination;
}

function nativeManager() {
  if (nativeRecorder) return nativeRecorder;
  if (typeof uni.getRecorderManager !== 'function') throw new Error('当前客户端不支持录音，仍可输入文字');
  nativeRecorder = uni.getRecorderManager();
  nativeRecorder.onStart(() => {
    const op = recording;
    if (!op || op.cancelled) { nativeRecorder.stop(); return; }
    op.phase = 'recording'; op.started?.(); notifyRecording(op,'recording');
  });
  nativeRecorder.onStop(result => {
    nativeBusy = false;
    const op = recording;
    if (!op) { removeTemporary(result.tempFilePath); return; }
    op.result = result; op.phase = 'ready'; op.finished?.(result);
    if (op.cancelled) { removeTemporary(result.tempFilePath); if (recording === op) recording = null; }
    else notifyRecording(op,'stopped',op.stopReason || 'device');
  });
  nativeRecorder.onError(() => {
    nativeBusy = false;
    const op = recording;
    if (op) notifyRecording(op,'error','device','录音失败，请检查麦克风权限后重试');
    recording = null;
    op?.failed?.(new Error('无法录音，请检查麦克风权限')); op?.finishFailed?.(new Error('录音失败，请重试'));
  });
  // Supported by some mini-program runtimes. A system interruption is not
  // consent to resume recording when the phone call / other audio ends.
  if (typeof nativeRecorder.onInterruptionBegin === 'function') {
    nativeRecorder.onInterruptionBegin(() => {
      const op = recording;
      if (!op || op.cancelled) return;
      notifyRecording(op,'error','interrupted','录音被系统中断，请重新开始录音');
      cancelRecording();
    });
  }
  return nativeRecorder;
}

export async function startRecording({onState} = {}) {
  if (getPreferences().connectionKind === 'invite') throw new Error('邀请内测暂未开放语音，请直接输入文字');
  stopSpeaking();
  if (recording || nativeBusy) throw new Error('上一段录音尚未结束');
  const current = ++epoch;
  if (getPreferences().mode === 'demo') { recording = {kind:'demo',epoch:current,onState}; notifyRecording(recording,'recording'); return; }
  const destination = connection();
  const isCurrent = () => current === epoch && matchesDestination(destination);
  await consent();
  if (current !== epoch) throw cancelled();
  const confirmed = connection();
  if (confirmed.url !== destination.url || confirmed.header.Authorization !== destination.header.Authorization) throw new Error('服务配置已变化，请重新开始录音');
  // #ifdef MP-WEIXIN
  await ensureWeChatPrivacy({mode:getPreferences().mode,action:'录音转文字',isCurrent});
  if (!isCurrent()) throw cancelled();
  // #endif
  const cap = await request('/voice/capabilities');
  if (!cap.asr?.available) throw new Error('私人服务尚未配置语音识别模型，仍可输入文字');
  if (current !== epoch) throw cancelled();
  // #ifdef MP-WEIXIN
  await ensureWeChatRecordingPermission({mode:getPreferences().mode,isCurrent});
  // #endif
  if (!isCurrent()) throw cancelled();

  // #ifdef H5
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') throw new Error('当前浏览器不支持录音，请使用 HTTPS 或原生客户端');
  const stream = await navigator.mediaDevices.getUserMedia({audio:true,video:false});
  if (!isCurrent()) { stream.getTracks().forEach(track => track.stop()); throw cancelled(); }
  const type = ['audio/webm;codecs=opus','audio/mp4'].find(t => MediaRecorder.isTypeSupported(t));
  if (!type) { stream.getTracks().forEach(track => track.stop()); throw new Error('当前浏览器没有受支持的音频编码'); }
  let recorder;
  try { recorder = new MediaRecorder(stream,{mimeType:type}); }
  catch { stream.getTracks().forEach(track => track.stop()); throw new Error('无法创建录音器'); }
  const op = {kind:'web',epoch:current,destination,stream,recorder,chunks:[],type,phase:'recording',onState};
  recording = op;
  op.done = new Promise((resolve,reject) => {
    recorder.ondataavailable = event => { if (event.data.size) op.chunks.push(event.data); };
    recorder.onstop = () => {
      stream.getTracks().forEach(track => track.stop()); clearTimeout(op.timer);
      if (op.phase === 'error') return;
      op.phase = 'ready'; resolve(new Blob(op.chunks,{type}));
      notifyRecording(op,'stopped',op.stopReason || 'device');
    };
    recorder.onerror = () => {
      stream.getTracks().forEach(track => track.stop()); clearTimeout(op.timer);
      op.phase = 'error'; reject(new Error('录音失败'));
      notifyRecording(op,'error','device','录音失败，请重新开始录音');
      if (recording === op) recording = null;
    };
  });
  op.done.catch(() => {});
  try { recorder.start(); }
  catch { stream.getTracks().forEach(track => track.stop()); recording = null; throw new Error('录音启动失败'); }
  notifyRecording(op,'recording');
  op.timer = setTimeout(() => { if (recorder.state !== 'inactive') { op.stopReason = 'limit'; recorder.stop(); } },60000);
  return;
  // #endif
  // #ifndef H5
  const manager = nativeManager();
  const operation = {kind:'native',epoch:current,destination,phase:'starting',cancelled:false,onState};
  recording = operation; nativeBusy = true;
  await new Promise((resolve,reject) => {
    operation.started = resolve; operation.failed = reject;
    try { manager.start({duration:60000,sampleRate:16000,numberOfChannels:1,encodeBitRate:48000,format:'mp3'}); }
    catch { nativeBusy = false; recording = null; reject(new Error('无法启动麦克风')); }
  });
  if (current !== epoch) throw cancelled();
  // #endif
}

export function stopRecording() {
  const op = recording;
  if (!op || op.cancelled) return Promise.reject(new Error('尚未开始录音或录音已取消'));
  if (!op.stopPromise) op.stopPromise = finishRecording(op);
  return op.stopPromise;
}

async function finishRecording(op) {
  if (op.kind === 'demo') { recording = null; return '【演示转写，未使用麦克风】可再生能源投资如何影响能源安全？'; }
  let path, result;
  const checkDestination = () => {
    if (!matchesDestination(op.destination)) throw new Error('服务配置已变化，录音不会发送，请重新录制');
  };
  // A background document read can delay the upload. Consent must still be
  // current when that queue is released, not only when the user taps Stop.
  const isCurrent = () => recording === op && !op.cancelled && op.epoch === epoch && matchesDestination(op.destination);
  try {
    if (op.phase === 'starting') { cancelRecording(); throw cancelled(); }
    if (op.phase !== 'ready') op.stopReason = 'user';
    // #ifdef H5
    clearTimeout(op.timer);
    if (op.recorder.state !== 'inactive') op.recorder.stop();
    const blob = await op.done;
    if (op.epoch !== epoch) throw cancelled();
    if (!blob.size || blob.size > 10 * 1024 * 1024) throw new Error('录音为空或超过 10 MB，请缩短录音');
    const file = new File([blob],op.type.includes('mp4') ? 'recording.m4a' : 'recording.webm',{type:blob.type});
    path = URL.createObjectURL(file);
    checkDestination();
    result = await uploadFile('/voice/transcribe',path,{},file,{isCurrent});
    // #endif
    // #ifndef H5
    const audio = op.phase === 'ready' ? op.result : await new Promise((resolve,reject) => { op.finished = resolve; op.finishFailed = reject; nativeManager().stop(); });
    path = audio.tempFilePath;
    if (op.epoch !== epoch) throw cancelled();
    if (audio.fileSize > 10 * 1024 * 1024) throw new Error('录音超过 10 MB');
    checkDestination();
    result = await uploadFile('/voice/transcribe',path,{},undefined,{isCurrent});
    // #endif
    if (!isCurrent()) throw cancelled();
    if (!result.text?.trim()) throw new Error('没有识别到文字，请重试或手动输入');
    return result.text;
  } finally {
    if (recording === op) recording = null;
    if (path) removeTemporary(path);
  }
}

function removeTemporary(path) {
  if (!path) return;
  // #ifdef H5
  if (path.startsWith('blob:')) URL.revokeObjectURL(path);
  // #endif
  // #ifdef MP-WEIXIN
  try { uni.getFileSystemManager().unlink({filePath:path,fail:() => {}}); } catch { /* platform temp file lifecycle */ }
  // #endif
  // #ifdef APP-PLUS
  try { plus.io.resolveLocalFileSystemURL(path,entry => { if (entry.isFile) entry.remove(() => {},() => {}); },() => {}); } catch { /* app shutting down or file already removed */ }
  // #endif
}

export function cancelRecording() {
  epoch += 1;
  const op = recording;
  if (!op) return;
  op.cancelled = true;
  if (op.kind === 'native') {
    op.failed?.(cancelled()); op.finishFailed?.(cancelled());
    if (op.phase === 'ready') { removeTemporary(op.result?.tempFilePath); recording = null; }
    else { try { nativeManager().stop(); } catch { nativeBusy = false; recording = null; } }
  } else {
    recording = null;
    clearTimeout(op.timer);
    if (op.recorder?.state === 'recording') op.recorder.stop();
    op.stream?.getTracks().forEach(track => track.stop());
  }
}

export function stopSpeaking() {
  speechEpoch += 1;
  cancelSpeechDownload?.(); cancelSpeechDownload = null;
  settlePlayback?.(); settlePlayback = null;
  if (player) { try { player.stop(); player.destroy(); } catch {} player = null; }
  cleanupAudio?.(); cleanupAudio = null;
}

// #ifdef APP-PLUS
function downloadAppSpeech(text, current) {
  if (typeof plus === 'undefined' || typeof plus.downloader?.createDownload !== 'function'
      || typeof plus.io?.resolveLocalFileSystemURL !== 'function'
      || typeof uni.createInnerAudioContext !== 'function') {
    throw new Error('此 App 基座不支持语音下载和清理，请更新 HBuilderX 基座');
  }
  const destination = connection();
  const filename = `_doc/docthinker-speech-${Date.now()}-${current}.mp3`;
  const limit = 10 * 1024 * 1024;
  const deadline = Date.now() + 45000;
  return new Promise((resolve,reject) => {
    let task, timer, settled = false, wasCancelled = false, successful = false;
    const removeFiles = download => {
      removeTemporary(filename);
      if (download?.filename && download.filename !== filename) removeTemporary(download.filename);
    };
    const finish = (error, path = '') => {
      if (settled) return;
      settled = true; successful = Boolean(path); clearTimeout(timer);
      if (cancelSpeechDownload === cancel) cancelSpeechDownload = null;
      // abort removes THIS task's persisted request metadata. When already
      // complete, it keeps the finished file for playback; we delete it later.
      try { task?.abort(); } catch { /* task may already be removed */ }
      if (!path) removeFiles(task);
      if (error) reject(error); else resolve(path);
    };
    const cancel = () => { wasCancelled = true; finish(null); };
    const completed = (download,status) => {
      if (settled) { if (!successful || wasCancelled || current !== speechEpoch) removeFiles(download); return; }
      if (current !== speechEpoch) { cancel(); removeFiles(download); return; }
      try {
        if (Date.now() >= deadline) throw new Error('语音下载超时，请稍后重试');
        if (status !== 200) throw new Error('语音服务暂不可用，请检查配置后重试');
        const mime = String(download.getResponseHeader('Content-Type') || '').split(';',1)[0].trim().toLowerCase();
        if (!['audio/mpeg','audio/mp3'].includes(mime)) throw new Error('语音服务返回了不支持的音频格式');
        if (!Number.isFinite(download.downloadedSize) || download.downloadedSize <= 0
            || download.downloadedSize > limit || download.totalSize > limit) throw new Error('语音音频为空或超过 10 MB');
        if (!download.filename) throw new Error('无法保存临时语音文件');
        finish(null,download.filename);
      } catch (error) { finish(error); }
    };
    try {
      task = plus.downloader.createDownload(destination.url + '/voice/speak',{
        method:'POST',data:JSON.stringify({text}),filename,timeout:45,retry:0,
      },completed);
      if (!task || ['start','abort','setRequestHeader','getResponseHeader','addEventListener'].some(name => typeof task[name] !== 'function')) {
        throw new Error('此 App 基座不支持安全语音下载，请更新基座');
      }
      task.setRequestHeader('Authorization',destination.header.Authorization);
      task.setRequestHeader('Content-Type','application/json');
      task.addEventListener('statechanged',download => {
        if (settled) return;
        if (current !== speechEpoch) { cancel(); return; }
        if (download.downloadedSize > limit || download.totalSize > limit) finish(new Error('语音音频超过 10 MB'));
      });
      cancelSpeechDownload = cancel;
      // Native timeout is time-to-response, not total download time. This
      // independent timer bounds the active operation while JS is running.
      timer = setTimeout(() => finish(new Error('语音下载超时，请稍后重试')),45000);
      task.start();
    } catch { finish(new Error('无法启动语音下载，请检查 App 基座与服务配置')); }
  });
}
// #endif

export async function speak(text) {
  if (getPreferences().connectionKind === 'invite') throw new Error('邀请内测暂未开放 AI 朗读');
  stopSpeaking(); cancelRecording();
  if (getPreferences().mode === 'demo') { uni.showToast({title:'演示模式不调用语音模型；连接服务后可朗读',icon:'none'}); return; }
  if (!text?.trim()) throw new Error('没有可朗读的内容');
  if (text.length > 1500) throw new Error('本版单次朗读最多 1500 字，请使用较短回答');
  const current = speechEpoch;
  const destination = connection();
  const isCurrent = () => current === speechEpoch && matchesDestination(destination);
  // #ifdef MP-WEIXIN
  await ensureWeChatPrivacy({mode:getPreferences().mode,action:'AI 朗读',isCurrent});
  if (!isCurrent()) return;
  // #endif
  let path;
  // #ifdef APP-PLUS
  path = await downloadAppSpeech(text,current);
  if (!path) return;
  // #endif
  // #ifndef APP-PLUS
  const data = await request('/voice/speak',{text},'POST','arraybuffer',{isCurrent});
  if (!isCurrent()) return;
  // #endif
  // #ifdef H5
  path = URL.createObjectURL(new Blob([data],{type:'audio/mpeg'}));
  // #endif
  // #ifdef MP-WEIXIN
  let fs;
  try { fs = uni.getFileSystemManager(); } catch { /* unsupported runtime */ }
  if (!fs?.writeFile) throw new Error('此运行基座不支持音频文件写入，请更新 HBuilderX/微信');
  path = `${wx.env.USER_DATA_PATH}/docthinker-speech-${Date.now()}.mp3`;
  await new Promise((resolve,reject) => fs.writeFile({filePath:path,data,success:resolve,fail:() => reject(new Error('无法创建临时音频'))}));
  // #endif
  if (!isCurrent()) { removeTemporary(path); return; }
  cleanupAudio = () => removeTemporary(path);
  let activePlayer;
  try {
    player = uni.createInnerAudioContext(); activePlayer = player; player.src = path;
    await new Promise((resolve,reject) => {
      settlePlayback = resolve;
      activePlayer.onEnded(() => { if (player === activePlayer) stopSpeaking(); resolve(); });
      activePlayer.onError(() => { reject(new Error('音频播放失败，请检查设备音量与格式')); if (player === activePlayer) stopSpeaking(); });
      activePlayer.onStop(resolve);
      activePlayer.play();
    });
  } catch (error) {
    if (!activePlayer || player === activePlayer) stopSpeaking();
    throw error;
  }
}
