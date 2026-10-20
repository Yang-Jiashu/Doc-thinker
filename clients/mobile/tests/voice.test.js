import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const raw = fs.readFileSync(new URL('../src/services/voice.js', import.meta.url), 'utf8');
const apiRaw = fs.readFileSync(new URL('../src/services/api.js', import.meta.url), 'utf8');
const privacyRaw = fs.readFileSync(new URL('../src/services/wechat-privacy.js', import.meta.url), 'utf8');
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const flush = async () => { for (let i = 0; i < 32; i++) await Promise.resolve(); };

function preprocess(source, flags) {
  const stack = [true]; const output = [];
  for (const line of source.split('\n')) {
    const conditional = line.match(/^\s*\/\/\s*#(ifdef|ifndef)\s+(\S+)/);
    if (conditional) {
      const enabled = flags.has(conditional[2]);
      stack.push(stack.at(-1) && (conditional[1] === 'ifdef' ? enabled : !enabled));
    } else if (/^\s*\/\/\s*#endif/.test(line)) stack.pop();
    else if (stack.at(-1)) output.push(line);
  }
  assert.equal(stack.length, 1, 'unbalanced platform directives');
  return output.join('\n').replace(/^import .*;\s*$/gm, '').replaceAll('export ', '');
}

function harness(platform = 'H5') {
  const calls = {mic: 0, upload: [], requests: [], privacy: [], revoked: [], unlinked: [], players: [], recorders: [], tracks: [], written: [], downloads: [], removed: []};
  const prefs = {mode: 'connected', baseUrl: 'https://private.example', accessToken: 'private-mobile-token'};
  const handlers = {
    request: async path => path.endsWith('/capabilities') ? {asr: {available: true}} : new Uint8Array([1, 2, 3]).buffer,
    upload: async () => ({text: '识别结果'}),
    mic: async () => stream(),
    modal: options => options.success({confirm: true}),
    privacySetting: options => options.success({needAuthorization:false}),
    privacyAuthorize: options => options.success({}),
    recordingSetting: options => options.success({authSetting:{'scope.record':true}}),
    recordingAuthorize: options => options.success({}),
  };
  function stream() {
    const track = {stopped: 0, stop() { this.stopped += 1; }};
    calls.tracks.push(track);
    return {getTracks: () => [track]};
  }
  class Recorder {
    static isTypeSupported(type) { return type.startsWith('audio/webm'); }
    constructor(stream, options) { this.stream = stream; this.options = options; this.state = 'inactive'; calls.recorders.push(this); }
    start() { this.state = 'recording'; }
    stop() {
      if (this.state === 'inactive') return;
      this.state = 'inactive';
      queueMicrotask(() => { this.ondataavailable?.({data: new Blob(['fake recording'])}); this.onstop?.(); });
    }
  }
  const native = {
    callbacks: {}, stops: 0, starts: 0,
    onStart(fn) { this.callbacks.start = fn; }, onStop(fn) { this.callbacks.stop = fn; }, onError(fn) { this.callbacks.error = fn; },
    onInterruptionBegin(fn) { this.callbacks.interruption = fn; },
    start() { this.starts += 1; }, stop() { this.stops += 1; },
  };
  const timers = new Map(); let timerId = 0, urlId = 0;
  const context = {
    Blob, File, console, queueMicrotask,
    setTimeout: fn => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: id => timers.delete(id),
    URL: {createObjectURL: () => 'blob:test-' + ++urlId, revokeObjectURL: url => calls.revoked.push(url)},
    navigator: {mediaDevices: {getUserMedia: async () => { calls.mic += 1; return handlers.mic(); }}},
    MediaRecorder: Recorder,
    getPreferences: () => prefs,
    connection: () => ({url: prefs.baseUrl + '/api/v1/mobile', header: {Authorization: 'Bearer ' + prefs.accessToken}}),
    request: (...args) => { calls.requests.push(args); return handlers.request(...args); },
    uploadFile: (...args) => { calls.upload.push(args); return handlers.upload(...args); },
    wx: {
      env: {USER_DATA_PATH: '/fake-user-data'},
      getPrivacySetting: options => { calls.privacy.push('getPrivacySetting'); handlers.privacySetting(options); },
      requirePrivacyAuthorize: options => { calls.privacy.push('requirePrivacyAuthorize'); handlers.privacyAuthorize(options); },
      getSetting: options => { calls.privacy.push('getSetting'); handlers.recordingSetting(options); },
      authorize: options => { calls.privacy.push(options.scope); handlers.recordingAuthorize(options); },
    },
    plus: {
      downloader: {createDownload: (url,options,complete) => {
        const task = {url,options,filename:options.filename,headers:{},listeners:{},started:0,aborted:0,
          downloadedSize:3,totalSize:3,mime:'audio/mpeg',
          start() { this.started += 1; }, abort() { this.aborted += 1; },
          setRequestHeader(name,value) { this.headers[name] = value; },
          getResponseHeader() { return this.mime; },
          addEventListener(name,listener) { this.listeners[name] = listener; },
          complete(status = 200) { complete(this,status); },
        };
        calls.downloads.push(task); return task;
      }},
      io: {resolveLocalFileSystemURL: (path,success) => success({isFile:true,remove: done => { calls.removed.push(path); done(); }})},
    },
    uni: {
      showModal: options => handlers.modal(options), showToast() {},
      getRecorderManager: () => native,
      getFileSystemManager: () => ({
        unlink: options => { calls.unlinked.push(options.filePath); options.success?.(); },
        writeFile: options => { calls.written.push(options.filePath); options.success(); },
      }),
      createInnerAudioContext: () => {
        const audio = {callbacks: {}, played: 0, stopped: 0, destroyed: 0,
          onEnded(fn) { this.callbacks.ended = fn; }, onError(fn) { this.callbacks.error = fn; }, onStop(fn) { this.callbacks.stop = fn; },
          play() { this.played += 1; }, stop() { this.stopped += 1; this.callbacks.stop?.(); }, destroy() { this.destroyed += 1; }};
        calls.players.push(audio); return audio;
      },
    },
  };
  if (platform === 'APP-PLUS') delete context.uni.getFileSystemManager;
  vm.runInNewContext(preprocess(privacyRaw + '\n' + raw, new Set([platform])) + '\nglobalThis.voice = {startRecording, stopRecording, cancelRecording, speak, stopSpeaking};', context);
  return {voice: context.voice, context, calls, prefs, handlers, stream, native, timers};
}

// Exercise the real API background-read barrier. Only the final platform I/O
// is fake; no microphone, server, transcription, or synthesis is contacted.
function queuedTransportHarness(platform) {
  const h = harness(platform), storage = new Map(), requests = [], uploads = [];
  Object.assign(h.context.uni, {
    getStorageSync: key => storage.get(key),
    setStorageSync: (key, value) => storage.set(key, value),
    request: options => {
      requests.push(options);
      if (options.url.endsWith('/voice/capabilities')) queueMicrotask(() => options.success({statusCode:200,data:{asr:{available:true}}}));
      return {abort() { options.fail({errMsg:'test cancellation'}); }};
    },
    uploadFile: options => {
      uploads.push(options);
      queueMicrotask(() => options.success({statusCode:200,data:{text:'Queued transcript'}}));
      return {abort() { options.fail({errMsg:'test cancellation'}); }};
    },
  });
  vm.runInNewContext(preprocess(apiRaw,new Set([platform])) + '\nglobalThis.transport = {researchStatusRequest,savePreferences,getPreferences};',h.context);
  const transport = h.context.transport;
  transport.savePreferences({mode:'connected',baseUrl:'https://private.example',accessToken:'offline-test-token'});
  return {...h,transport,requests,uploads};
}

async function preparedRecording(h, platform) {
  const starting = h.voice.startRecording();
  await flush();
  if (platform !== 'H5') h.native.callbacks.start();
  await starting;
}

function backgroundRead(h) {
  const result = h.transport.researchStatusRequest('/research/documents?session_id=%2300001');
  result.catch(() => {});
  const request = h.requests.at(-1);
  return () => request.success({statusCode:200,data:{documents:[]}});
}

test('invitation chat-only mode blocks recording and speech before any privacy, capture or network work',async () => {
  const h=harness();h.prefs.connectionKind='invite';
  await assert.rejects(h.voice.startRecording(),/暂未开放语音/);await assert.rejects(h.voice.speak('hello'),/暂未开放 AI 朗读/);
  assert.equal(h.calls.mic,0);assert.equal(h.calls.requests.length,0);assert.equal(h.calls.upload.length,0);assert.equal(h.calls.privacy.length,0);assert.equal(h.calls.players.length,0);
});

test('demo never requests a microphone, upload, or audio provider', async () => {
  const h = harness(); h.prefs.mode = 'demo';
  await h.voice.startRecording();
  assert.match(await h.voice.stopRecording(), /演示转写/);
  await h.voice.speak('demo');
  assert.equal(h.calls.mic, 0); assert.equal(h.calls.upload.length, 0); assert.equal(h.calls.requests.length, 0);
});

test('late microphone permission result is stopped after cancellation', async () => {
  const h = harness(); const permission = deferred(); h.handlers.mic = () => permission.promise;
  const start = h.voice.startRecording(); const rejected = assert.rejects(start, /取消/);
  await flush(); assert.equal(h.calls.mic, 1);
  h.voice.cancelRecording(); permission.resolve(h.stream()); await rejected;
  assert.equal(h.calls.tracks[0].stopped, 1); assert.equal(h.calls.recorders.length, 0); assert.equal(h.calls.upload.length, 0);
});

for (const change of ['baseUrl','accessToken','mode','unreadable','invalid']) {
  test(`web permission grant cannot capture after ${change} invalidates its destination`,async () => {
    const h = harness(), permission = deferred(); h.handlers.mic = () => permission.promise;
    const starting = h.voice.startRecording(), rejected = assert.rejects(starting,/取消/);
    await flush(); assert.equal(h.calls.mic,1);
    if (change === 'unreadable') h.context.connection = () => { throw new Error('configuration inaccessible'); };
    else if (change === 'invalid') h.context.connection = () => null;
    else h.prefs[change] = change === 'mode' ? 'demo' : h.prefs[change] + '-changed';
    permission.resolve(h.stream()); await rejected;
    assert.equal(h.calls.tracks[0].stopped,1); assert.equal(h.calls.recorders.length,0); assert.equal(h.calls.upload.length,0);
  });
}

test('cancel during capability request never asks for microphone permission', async () => {
  const h = harness(); const caps = deferred(); h.handlers.request = () => caps.promise;
  const start = h.voice.startRecording(); const rejected = assert.rejects(start, /取消/);
  await flush(); h.voice.cancelRecording(); caps.resolve({asr: {available: true}}); await rejected;
  assert.equal(h.calls.mic, 0);
});

test('cancelled web recording stops tracks and does not upload', async () => {
  const h = harness(); await h.voice.startRecording(); h.voice.cancelRecording(); await flush();
  await assert.rejects(h.voice.stopRecording(), /取消|尚未开始/);
  assert.ok(h.calls.tracks.every(track => track.stopped >= 1)); assert.equal(h.calls.upload.length, 0); assert.equal(h.timers.size, 0);
});

test('successful web recording uploads once and revokes the temporary URL', async () => {
  const h = harness(); await h.voice.startRecording();
  assert.equal(await h.voice.stopRecording(), '识别结果');
  assert.equal(h.calls.upload.length, 1); assert.equal(h.calls.upload[0][0], '/voice/transcribe');
  assert.equal(h.calls.upload[0][3].name, 'recording.webm');
  assert.equal(h.calls.revoked.length, 1); assert.ok(h.calls.tracks[0].stopped >= 1);
});

for (const field of ['baseUrl', 'accessToken']) {
  test(`changing ${field} during consent cannot request a microphone or send audio`, async () => {
    const h = harness(); let consent;
    h.handlers.modal = options => { consent = options; };
    const start = h.voice.startRecording(); const rejected = assert.rejects(start, /服务配置已变化/);
    await flush(); h.prefs[field] += '-changed'; consent.success({confirm: true}); await rejected;
    assert.equal(h.calls.mic, 0); assert.equal(h.calls.upload.length, 0); assert.equal(h.calls.requests.length, 0);
  });

  for (const platform of ['H5', 'MP-WEIXIN']) {
    test(`${platform} recording is not uploaded after ${field} changes`, async () => {
      const h = harness(platform); const start = h.voice.startRecording();
      await flush(); if (platform !== 'H5') h.native.callbacks.start(); await start;
      h.prefs[field] += '-changed';
      const stop = h.voice.stopRecording(); const rejected = assert.rejects(stop, /服务配置已变化/);
      if (platform !== 'H5') h.native.callbacks.stop({tempFilePath: '/temp/destination-changed.mp3', fileSize: 15});
      await rejected;
      assert.equal(h.calls.upload.length, 0);
      if (platform === 'H5') {
        assert.equal(h.calls.revoked.length, 1); assert.ok(h.calls.tracks[0].stopped >= 1);
      } else assert.ok(h.calls.unlinked.includes('/temp/destination-changed.mp3'));
    });
  }
}

test('cancel after upload starts discards its late text without a second upload', async () => {
  const h = harness(); const upload = deferred(); h.handlers.upload = () => upload.promise;
  await h.voice.startRecording(); const stopping = h.voice.stopRecording(); const rejected = assert.rejects(stopping, /取消/);
  await flush(); assert.equal(h.calls.upload.length, 1);
  h.voice.cancelRecording(); upload.resolve({text: 'late text'}); await rejected;
  assert.equal(h.calls.upload.length, 1); assert.equal(h.calls.revoked.length, 1);
});

for (const platform of ['H5','MP-WEIXIN']) {
  for (const change of ['cancel-or-page-hide','baseUrl','accessToken']) {
    test(`${platform} queued ASR does not upload after ${change}`,async () => {
      const h = queuedTransportHarness(platform); await preparedRecording(h,platform);
      const releaseRead = backgroundRead(h), stopping = h.voice.stopRecording();
      const rejected = assert.rejects(stopping,/取消|改变/);
      if (platform !== 'H5') h.native.callbacks.stop({tempFilePath:'/temp/queued.mp3',fileSize:15});
      await flush(); assert.equal(h.uploads.length,0);
      if (change === 'cancel-or-page-hide') h.voice.cancelRecording();
      else {
        const preferences = h.transport.getPreferences();
        h.transport.savePreferences({...preferences,[change]:preferences[change] + '-changed'});
      }
      releaseRead(); await rejected;
      assert.equal(h.uploads.length,0);
      if (platform === 'H5') assert.equal(h.calls.revoked.length,1);
      else assert.ok(h.calls.unlinked.includes('/temp/queued.mp3'));
    });
  }

  test(`${platform} valid queued ASR uploads exactly once after the read settles`,async () => {
    const h = queuedTransportHarness(platform); await preparedRecording(h,platform);
    const releaseRead = backgroundRead(h), first = h.voice.stopRecording(), second = h.voice.stopRecording();
    if (platform !== 'H5') h.native.callbacks.stop({tempFilePath:'/temp/queued-valid.mp3',fileSize:15});
    await flush(); assert.equal(h.uploads.length,0);
    releaseRead(); assert.deepEqual(await Promise.all([first,second]),['Queued transcript','Queued transcript']);
    assert.equal(h.uploads.length,1); assert.ok(h.uploads[0].url.endsWith('/voice/transcribe'));
  });

  for (const change of ['cancel-or-page-hide','baseUrl','accessToken']) {
    test(`${platform} queued TTS does not synthesize after ${change}`,async () => {
      const h = queuedTransportHarness(platform), releaseRead = backgroundRead(h);
      const speaking = h.voice.speak('Only synthesize while this intent is current');
      const rejected = assert.rejects(speaking,/取消|改变/);
      await flush(); assert.equal(h.requests.filter(item => item.url.endsWith('/voice/speak')).length,0);
      if (change === 'cancel-or-page-hide') h.voice.stopSpeaking();
      else {
        const preferences = h.transport.getPreferences();
        h.transport.savePreferences({...preferences,[change]:preferences[change] + '-changed'});
      }
      releaseRead(); await rejected;
      assert.equal(h.requests.filter(item => item.url.endsWith('/voice/speak')).length,0);
      assert.equal(h.calls.players.length,0); assert.equal(h.calls.written.length,0);
    });
  }

  test(`${platform} valid queued TTS synthesizes once after the read settles`,async () => {
    const h = queuedTransportHarness(platform), releaseRead = backgroundRead(h);
    const speaking = h.voice.speak('Read this once'); await flush();
    assert.equal(h.requests.length,1); releaseRead(); await flush();
    const synthesis = h.requests.filter(item => item.url.endsWith('/voice/speak'));
    assert.equal(synthesis.length,1); assert.equal(synthesis[0].method,'POST');
    synthesis[0].success({statusCode:200,data:new Uint8Array([1,2,3]).buffer}); await flush();
    assert.equal(h.calls.players.length,1); assert.equal(h.calls.players[0].played,1);
    h.voice.stopSpeaking(); await speaking;
  });

  for (const change of ['unreadable','invalid']) {
    test(`${platform} queued upload callback fails closed for ${change} destination`,async () => {
      const h = harness(platform), upload = deferred(); h.handlers.upload = () => upload.promise;
      await preparedRecording(h,platform);
      const stopping = h.voice.stopRecording(), rejected = assert.rejects(stopping,/取消/);
      if (platform !== 'H5') h.native.callbacks.stop({tempFilePath:'/temp/invalid-destination.mp3',fileSize:15});
      await flush(); const isCurrent = h.calls.upload[0][4].isCurrent;
      assert.equal(isCurrent(),true);
      h.context.connection = change === 'invalid' ? () => null : () => {throw new Error('not available');};
      assert.equal(isCurrent(),false);
      upload.resolve({text:'Must not reach the draft'}); await rejected;
      assert.equal(h.calls.upload.length,1);
    });
  }

  test(`${platform} changed destination after ASR request discards its late transcript`,async () => {
    const h = harness(platform), upload = deferred(); h.handlers.upload = () => upload.promise;
    await preparedRecording(h,platform);
    const stopping = h.voice.stopRecording(), rejected = assert.rejects(stopping,/取消/);
    if (platform !== 'H5') h.native.callbacks.stop({tempFilePath:'/temp/late-result.mp3',fileSize:15});
    await flush(); h.prefs.accessToken += '-changed'; upload.resolve({text:'Old identity transcript'}); await rejected;
    assert.equal(h.calls.upload.length,1);
  });

  test(`${platform} changed destination after TTS request never plays its late audio`,async () => {
    const h = harness(platform), audio = deferred(); h.handlers.request = () => audio.promise;
    const speaking = h.voice.speak('Old identity answer'); await flush(); h.prefs.accessToken += '-changed';
    audio.resolve(new Uint8Array([1,2,3]).buffer); await speaking;
    assert.equal(h.calls.requests.length,1); assert.equal(h.calls.players.length,0); assert.equal(h.calls.written.length,0);
  });
}

test('concurrent stop requests cannot charge for duplicate transcriptions', async () => {
  const h = harness(); const upload = deferred(); h.handlers.upload = () => upload.promise;
  await h.voice.startRecording();
  const stops = [h.voice.stopRecording(), h.voice.stopRecording()];
  const completed = Promise.allSettled(stops);
  await flush(); upload.resolve({text: 'result'}); await completed;
  assert.equal(h.calls.upload.length, 1);
});

test('cancelled synthesis response never creates or plays an audio object', async () => {
  const h = harness(); const speech = deferred(); h.handlers.request = () => speech.promise;
  const speaking = h.voice.speak('hello'); h.voice.stopSpeaking(); speech.resolve(new ArrayBuffer(3)); await speaking;
  assert.equal(h.calls.players.length, 0); assert.equal(h.calls.revoked.length, 0);
});

test('manual playback stop settles the promise and releases audio resources', async () => {
  const h = harness(); const speaking = h.voice.speak('hello'); await flush();
  assert.equal(h.calls.players[0].played, 1);
  h.voice.stopSpeaking(); await speaking;
  assert.equal(h.calls.players[0].destroyed, 1); assert.equal(h.calls.revoked.length, 1);
});

test('playback failure rejects instead of being resolved by stop cleanup', async () => {
  const h = harness(); const speaking = h.voice.speak('hello');
  const rejected = assert.rejects(speaking, /播放失败/);
  await flush(); h.calls.players[0].callbacks.error(); await rejected;
  assert.equal(h.calls.players[0].destroyed, 1); assert.equal(h.calls.revoked.length, 1);
});

test('late native onStart after cancellation is stopped and never uploads', async () => {
  const h = harness('MP-WEIXIN'); const starting = h.voice.startRecording(); const rejected = assert.rejects(starting, /取消/);
  await flush(); h.voice.cancelRecording(); h.native.callbacks.start(); await rejected;
  h.native.callbacks.stop({tempFilePath: '/temp/cancelled.mp3', fileSize: 15});
  assert.ok(h.native.stops >= 1); assert.equal(h.calls.upload.length, 0);
  assert.ok(h.calls.unlinked.includes('/temp/cancelled.mp3'));
});

test('native recording cancelled while awaiting onStop cleans the late temporary file', async () => {
  const h = harness('MP-WEIXIN'); const starting = h.voice.startRecording(); await flush(); h.native.callbacks.start(); await starting;
  const stopping = h.voice.stopRecording(); const rejected = assert.rejects(stopping, /取消/);
  h.voice.cancelRecording(); await rejected;
  h.native.callbacks.stop({tempFilePath: '/temp/late-stop.mp3', fileSize: 15});
  assert.equal(h.calls.upload.length, 0); assert.ok(h.calls.unlinked.includes('/temp/late-stop.mp3'));
});

test('App TTS uses exactly one authenticated POST download and removes its task record', async () => {
  const h = harness('APP-PLUS'); const speaking = h.voice.speak('hello');
  assert.equal(h.calls.requests.length, 0); assert.equal(h.calls.downloads.length, 1);
  const task = h.calls.downloads[0];
  assert.equal(task.url, 'https://private.example/api/v1/mobile/voice/speak');
  assert.equal(task.options.method, 'POST'); assert.equal(task.options.data, JSON.stringify({text:'hello'}));
  assert.equal(task.options.retry, 0); assert.equal(task.options.timeout, 45);
  assert.equal(task.headers.Authorization, 'Bearer private-mobile-token'); assert.equal(task.headers['Content-Type'], 'application/json');
  assert.match(task.filename, /^_doc\/docthinker-speech-.*\.mp3$/); assert.equal(task.started, 1);
  task.complete(); await flush();
  assert.equal(task.aborted, 1, 'completed native task metadata is removed');
  assert.equal(h.calls.players[0].played, 1); assert.equal(h.calls.removed.length, 0);
  task.complete(); assert.equal(h.calls.removed.length, 0, 'duplicate success cannot delete the playing file');
  h.voice.stopSpeaking(); await speaking;
  assert.ok(h.calls.removed.includes(task.filename)); assert.equal(h.timers.size, 0); assert.equal(h.calls.written.length, 0);
});

for (const missing of ['plus','download','cleanup','player','taskMethod']) {
  test(`App missing ${missing} capability fails before any network request`, async () => {
    const h = harness('APP-PLUS');
    if (missing === 'plus') delete h.context.plus;
    if (missing === 'download') delete h.context.plus.downloader.createDownload;
    if (missing === 'cleanup') delete h.context.plus.io.resolveLocalFileSystemURL;
    if (missing === 'player') delete h.context.uni.createInnerAudioContext;
    if (missing === 'taskMethod') {
      const create = h.context.plus.downloader.createDownload;
      h.context.plus.downloader.createDownload = (...args) => { const task = create(...args); delete task.getResponseHeader; return task; };
    }
    await assert.rejects(h.voice.speak('hello'), /基座/);
    assert.equal(h.calls.requests.length, 0); assert.ok(h.calls.downloads.every(task => task.started === 0));
  });
}

test('App cancellation aborts and discards a late completed download without playback', async () => {
  const h = harness('APP-PLUS'); const speaking = h.voice.speak('hello'); const task = h.calls.downloads[0];
  h.voice.stopSpeaking(); await speaking;
  assert.equal(task.aborted, 1); assert.ok(h.calls.removed.includes(task.filename));
  task.filename += '.late.mp3'; task.complete(); await flush();
  assert.ok(h.calls.removed.includes(task.filename)); assert.equal(h.calls.players.length, 0); assert.equal(h.timers.size, 0);
});

test('App cancellation immediately after download completion cleans the unused file', async () => {
  const h = harness('APP-PLUS'); const speaking = h.voice.speak('hello'); const task = h.calls.downloads[0];
  task.complete(); h.voice.stopSpeaking(); await speaking;
  assert.ok(h.calls.removed.includes(task.filename)); assert.equal(h.calls.players.length, 0);
});

for (const scenario of ['status','mime','empty','nan','large','announcedLarge']) {
  test(`App rejects ${scenario} audio response without playback or retry`, async () => {
    const h = harness('APP-PLUS'); const speaking = h.voice.speak('hello');
    const rejected = assert.rejects(speaking, /语音/); const task = h.calls.downloads[0];
    if (scenario === 'mime') task.mime = 'application/json';
    if (scenario === 'empty') task.downloadedSize = 0;
    if (scenario === 'nan') task.downloadedSize = NaN;
    if (scenario === 'large') task.downloadedSize = 10 * 1024 * 1024 + 1;
    if (scenario === 'announcedLarge') task.totalSize = 10 * 1024 * 1024 + 1;
    task.complete(scenario === 'status' ? 503 : 200); await rejected;
    assert.equal(task.aborted, 1); assert.ok(h.calls.removed.includes(task.filename));
    assert.equal(h.calls.players.length, 0); assert.equal(h.calls.downloads.length, 1); assert.equal(h.timers.size, 0);
  });
}

test('App stops oversized downloads on progress instead of waiting for completion', async () => {
  const h = harness('APP-PLUS'); const speaking = h.voice.speak('hello'); const rejected = assert.rejects(speaking, /超过/);
  const task = h.calls.downloads[0]; task.downloadedSize = 10 * 1024 * 1024 + 1;
  task.listeners.statechanged(task); await rejected;
  assert.equal(task.aborted, 1); assert.equal(h.calls.players.length, 0); assert.ok(h.calls.removed.includes(task.filename));
});

test('App total-operation timer aborts stalled downloads and ignores late success', async () => {
  const h = harness('APP-PLUS'); const speaking = h.voice.speak('hello'); const rejected = assert.rejects(speaking, /超时/);
  const task = h.calls.downloads[0]; [...h.timers.values()][0](); await rejected;
  task.filename += '.late.mp3'; task.complete(); await flush();
  assert.ok(h.calls.removed.includes(task.filename), 'late file after failure must also be deleted');
  assert.equal(task.aborted, 1); assert.equal(h.calls.players.length, 0); assert.equal(h.timers.size, 0);
});

for (const stage of ['create','src','play']) {
  test(`App synchronous ${stage} playback failure cleans the downloaded file`, async () => {
    const h = harness('APP-PLUS'); const create = h.context.uni.createInnerAudioContext;
    h.context.uni.createInnerAudioContext = () => {
      if (stage === 'create') throw new Error('create failed');
      const player = create();
      if (stage === 'src') Object.defineProperty(player, 'src', {set() { throw new Error('src failed'); }});
      if (stage === 'play') player.play = () => { throw new Error('play failed'); };
      return player;
    };
    const speaking = h.voice.speak('hello'); const rejected = assert.rejects(speaking, /failed/);
    const task = h.calls.downloads[0]; task.complete(); await rejected;
    assert.ok(h.calls.removed.includes(task.filename)); assert.equal(h.timers.size, 0);
    h.voice.stopSpeaking();
  });
}

test('App cancelled native recordings use plus.io cleanup, never the mini-program filesystem', async () => {
  const h = harness('APP-PLUS'); const start = h.voice.startRecording(); await flush(); h.native.callbacks.start(); await start;
  h.voice.cancelRecording(); h.native.callbacks.stop({tempFilePath:'_doc/cancelled.mp3'});
  assert.ok(h.calls.removed.includes('_doc/cancelled.mp3')); assert.equal(h.calls.upload.length, 0);
});

test('web 60-second stop notifies ready state but waits for explicit transcription', async () => {
  const h = harness(); const states = []; await h.voice.startRecording({onState: state => states.push(state)});
  [...h.timers.values()][0](); await flush();
  assert.deepEqual(states.map(item => item.state), ['recording','stopped']); assert.equal(states[1].reason, 'limit');
  assert.equal(h.calls.upload.length, 0); assert.equal(h.timers.size, 0);
  assert.equal(await h.voice.stopRecording(), '识别结果'); assert.equal(h.calls.upload.length, 1);
});

test('web recorder failure notifies once, releases the microphone and ignores its late stop', async () => {
  const h = harness(); const states = []; await h.voice.startRecording({onState: state => states.push(state)});
  const recorder = h.calls.recorders[0]; recorder.onerror(); recorder.onstop();
  assert.deepEqual(states.map(item => item.state), ['recording','error']);
  assert.ok(h.calls.tracks[0].stopped >= 1); assert.equal(h.timers.size, 0);
  await assert.rejects(h.voice.stopRecording(), /尚未开始/); assert.equal(h.calls.upload.length, 0);
});

test('cancelled web callbacks cannot publish ready state into a new recording', async () => {
  const h = harness(); const oldStates = [], newStates = [];
  await h.voice.startRecording({onState: state => oldStates.push(state)}); const old = h.calls.recorders[0];
  h.voice.cancelRecording(); await h.voice.startRecording({onState: state => newStates.push(state)});
  old.onstop(); old.onerror();
  assert.deepEqual(oldStates.map(item => item.state), ['recording']); assert.deepEqual(newStates.map(item => item.state), ['recording']);
  h.voice.cancelRecording();
});

test('native automatic stop notifies ready state and error is visible without an upload', async () => {
  const h = harness('MP-WEIXIN'); const states = [];
  const start = h.voice.startRecording({onState: state => states.push(state)}); await flush(); h.native.callbacks.start(); await start;
  h.native.callbacks.stop({tempFilePath:'/temp/auto.mp3',fileSize:15});
  assert.deepEqual(states.map(item => item.state), ['recording','stopped']); assert.equal(h.calls.upload.length, 0);
  h.voice.cancelRecording();
  const next = h.voice.startRecording({onState: state => states.push(state)}); await flush(); h.native.callbacks.start(); await next;
  h.native.callbacks.error(); assert.equal(states.at(-1).state, 'error'); assert.equal(h.calls.upload.length, 0);
});

test('native system interruption cancels without automatically resuming or uploading', async () => {
  const h = harness('MP-WEIXIN'); const states = [];
  const start = h.voice.startRecording({onState: state => states.push(state)}); await flush(); h.native.callbacks.start(); await start;
  h.native.callbacks.interruption(); h.native.callbacks.stop({tempFilePath:'/temp/interrupted.mp3'});
  assert.deepEqual(states.map(item => item.state), ['recording','error']); assert.equal(states[1].reason, 'interrupted');
  assert.equal(h.native.starts, 1); assert.ok(h.native.stops >= 1); assert.equal(h.calls.upload.length, 0);
  assert.ok(h.calls.unlinked.includes('/temp/interrupted.mp3'));
});

test('WeChat demo makes no privacy, recorder, microphone, upload, or provider calls', async () => {
  const h = harness('MP-WEIXIN'); h.prefs.mode = 'demo';
  await h.voice.startRecording(); await h.voice.stopRecording(); await h.voice.speak('演示回答');
  assert.equal(h.calls.privacy.length,0); assert.equal(h.native.starts,0);
  assert.equal(h.calls.requests.length,0); assert.equal(h.calls.upload.length,0); assert.equal(h.calls.players.length,0);
});

test('declining the destination notice never opens WeChat authorization', async () => {
  const h = harness('MP-WEIXIN'); h.handlers.modal = options => options.success({confirm:false});
  await assert.rejects(h.voice.startRecording(),/取消/);
  assert.equal(h.calls.privacy.length,0); assert.equal(h.calls.requests.length,0); assert.equal(h.native.starts,0);
});

test('WeChat privacy rejection blocks capability checks, audio capture, and uploads', async () => {
  const h = harness('MP-WEIXIN');
  h.handlers.privacySetting = options => options.success({needAuthorization:true});
  h.handlers.privacyAuthorize = options => options.fail({});
  await assert.rejects(h.voice.startRecording(),/隐私授权/);
  assert.equal(h.calls.requests.length,0); assert.equal(h.native.starts,0); assert.equal(h.calls.upload.length,0);
});

test('missing WeChat privacy capability never falls back to an unguarded recorder', async () => {
  const h = harness('MP-WEIXIN'); delete h.context.wx.requirePrivacyAuthorize;
  await assert.rejects(h.voice.startRecording(),/升级微信/);
  assert.equal(h.calls.requests.length,0); assert.equal(h.native.starts,0);
});

test('WeChat microphone denial does not start the recorder or automatically retry', async () => {
  const h = harness('MP-WEIXIN');
  h.handlers.recordingSetting = options => options.success({authSetting:{'scope.record':false}});
  await assert.rejects(h.voice.startRecording(),/麦克风权限已关闭/);
  assert.equal(h.native.starts,0); assert.equal(h.calls.upload.length,0);
  assert.equal(h.calls.privacy.includes('scope.record'),false);
});

test('first WeChat recording waits for privacy and microphone permission before native start', async () => {
  const h = harness('MP-WEIXIN'); let privacy, microphone;
  h.handlers.privacySetting = options => options.success({needAuthorization:true});
  h.handlers.privacyAuthorize = options => { privacy = options; };
  h.handlers.recordingSetting = options => options.success({authSetting:{}});
  h.handlers.recordingAuthorize = options => { microphone = options; };
  const start = h.voice.startRecording(); await flush();
  assert.equal(h.native.starts,0); assert.equal(h.calls.requests.length,0);
  privacy.success({}); await flush();
  assert.equal(h.native.starts,0); assert.equal(h.calls.requests.length,1);
  microphone.success({}); await flush();
  assert.equal(h.native.starts,1);
  h.native.callbacks.start(); await start;
  h.voice.cancelRecording(); h.native.callbacks.stop({tempFilePath:'/temp/authorized.mp3'});
  assert.equal(h.calls.upload.length,0);
});

test('background cancellation during WeChat privacy consent cannot start recording later', async () => {
  const h = harness('MP-WEIXIN'); let privacy;
  h.handlers.privacySetting = options => options.success({needAuthorization:true});
  h.handlers.privacyAuthorize = options => { privacy = options; };
  const start = h.voice.startRecording(); const rejected = assert.rejects(start,/取消/);
  await flush(); h.voice.cancelRecording(); privacy.success({}); await rejected;
  assert.equal(h.native.starts,0); assert.equal(h.calls.requests.length,0); assert.equal(h.calls.upload.length,0);
});

test('changing private service while waiting for WeChat microphone permission prevents late capture', async () => {
  const h = harness('MP-WEIXIN'); let microphone;
  h.handlers.recordingSetting = options => options.success({authSetting:{}});
  h.handlers.recordingAuthorize = options => { microphone = options; };
  const start = h.voice.startRecording(); const rejected = assert.rejects(start,/配置已变化/);
  await flush(); h.prefs.baseUrl = 'https://different.example'; microphone.success({}); await rejected;
  assert.equal(h.native.starts,0); assert.equal(h.calls.upload.length,0);
});

test('WeChat AI speech privacy rejection sends no answer text and writes no temporary audio', async () => {
  const h = harness('MP-WEIXIN');
  h.handlers.privacySetting = options => options.success({needAuthorization:true});
  h.handlers.privacyAuthorize = options => options.fail({});
  await assert.rejects(h.voice.speak('private answer'),/AI 朗读需要微信隐私授权/);
  assert.equal(h.calls.requests.length,0); assert.equal(h.calls.written.length,0); assert.equal(h.calls.players.length,0);
});

test('cancelled WeChat AI speech consent cannot send answer text after a late grant', async () => {
  const h = harness('MP-WEIXIN'); let privacy;
  h.handlers.privacySetting = options => options.success({needAuthorization:true});
  h.handlers.privacyAuthorize = options => { privacy = options; };
  const speech = h.voice.speak('private answer'); const rejected = assert.rejects(speech,/取消/);
  await flush(); h.voice.stopSpeaking(); privacy.success({}); await rejected;
  assert.equal(h.calls.requests.length,0); assert.equal(h.calls.written.length,0);
});
