import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const apiSource = fs.readFileSync(new URL('../src/services/api.js', import.meta.url), 'utf8');
const privacySource = fs.readFileSync(new URL('../src/services/wechat-privacy.js', import.meta.url), 'utf8');
const flush = async () => { for (let i = 0; i < 24; i++) await Promise.resolve(); };

function preprocess(source) {
  const flags = new Set(['MP-WEIXIN']); const stack = [true], output = [];
  for (const line of source.split('\n')) {
    const directive = line.match(/^\s*\/\/\s*#(ifdef|ifndef)\s+(\S+)/);
    if (directive) {
      const enabled = flags.has(directive[2]);
      stack.push(stack.at(-1) && (directive[1] === 'ifdef' ? enabled : !enabled));
    } else if (/^\s*\/\/\s*#endif/.test(line)) stack.pop();
    else if (stack.at(-1)) output.push(line);
  }
  assert.equal(stack.length,1,'unbalanced platform directives');
  return output.join('\n').replace(/^import .*;\s*$/gm,'').replaceAll('export ','');
}

function harness({demo = false} = {}) {
  const storage = new Map(), timers = new Map(); let timerId = 0;
  const calls = {order:[],privacy:[],files:[],images:[],modals:[],uploads:[],requests:[]};
  const handlers = {
    privacySetting: options => options.success({needAuthorization:true}),
    privacyAuthorize: options => options.success({}),
    chooseFile: options => options.success({tempFiles:[{name:'研究笔记.PDF',path:'wxfile://tmp/random.pdf',size:1024}]}),
    chooseImage: options => options.success({tempFiles:[{path:'wxfile://tmp/album.jpg',size:2048}],tempFilePaths:['wxfile://tmp/album.jpg']}),
    confirm: options => options.success({confirm:true}),
    upload: options => options.success({statusCode:200,data:JSON.stringify({status:'processing',file_name:'研究笔记.PDF'})}),
  };
  const context = {
    setTimeout: fn => { timers.set(++timerId,fn); return timerId; },
    clearTimeout: id => timers.delete(id),
    wx: {
      getPrivacySetting: options => { calls.order.push('privacy-setting'); calls.privacy.push(options); handlers.privacySetting(options); },
      requirePrivacyAuthorize: options => { calls.order.push('privacy-authorize'); calls.privacy.push(options); handlers.privacyAuthorize(options); },
      chooseMessageFile: options => { calls.order.push('choose-document'); calls.files.push(options); handlers.chooseFile(options); },
    },
    uni: {
      getStorageSync: key => storage.get(key),
      setStorageSync: (key,value) => storage.set(key,value),
      chooseFile: () => { throw new Error('H5 picker must not run in a WeChat build'); },
      chooseImage: options => { calls.order.push('choose-image'); calls.images.push(options); handlers.chooseImage(options); },
      showModal: options => { calls.order.push('confirm'); calls.modals.push(options); handlers.confirm(options); },
      uploadFile: options => { calls.order.push('upload'); calls.uploads.push(options); handlers.upload(options); },
      request: options => { calls.requests.push(options); throw new Error('Upload flow must not make auxiliary network requests'); },
    },
  };
  vm.runInNewContext(preprocess(privacySource + '\n' + apiSource)
    + '\nglobalThis.service = {api,savePreferences,getPreferences};',context);
  if (!demo) context.service.savePreferences({mode:'connected',baseUrl:'https://private.example',accessToken:'private-test-code'});
  return {context,...context.service,calls,handlers,storage,timers};
}

test('WeChat demo upload never touches privacy, picker, confirmation, microphone, or network', async () => {
  const h = harness({demo:true}); delete h.context.wx;
  const result = await h.api.upload('#demo1');
  assert.equal(result.status,'demo'); assert.match(result.message,/没有读取或上传真实文件/);
  assert.equal(h.calls.order.length,0); assert.equal(h.calls.requests.length,0);
  assert.equal((await h.api.files('#demo1')).length,2);
});

test('upload waits for privacy → choose → confirm → one upload in that order', async () => {
  const h = harness(); let privacy, picker, confirmation;
  h.handlers.privacyAuthorize = options => { privacy = options; };
  h.handlers.chooseFile = options => { picker = options; };
  h.handlers.confirm = options => { confirmation = options; };
  const uploading = h.api.upload('#00001'); await flush();
  assert.deepEqual(h.calls.order,['privacy-setting','privacy-authorize']);
  privacy.success({}); await flush();
  assert.equal(h.calls.modals.length,0); assert.equal(h.calls.uploads.length,0);
  picker.success({tempFiles:[{name:'结论.pdf',path:'wxfile://tmp/document',size:50}]}); await flush();
  assert.equal(h.calls.modals.length,1); assert.equal(h.calls.uploads.length,0);
  assert.match(confirmation.content,/结论.pdf/); assert.match(confirmation.content,/https:\/\/private.example/);
  assert.equal(confirmation.content.includes('private-test-code'),false);
  confirmation.success({confirm:true}); await uploading;
  assert.deepEqual(h.calls.order,['privacy-setting','privacy-authorize','choose-document','confirm','upload']);
  assert.equal(h.calls.uploads.length,1); assert.equal(h.calls.requests.length,0);
});

test('privacy rejection prevents reading a selected file or sending a request', async () => {
  const h = harness(); h.handlers.privacyAuthorize = options => options.fail({errMsg:'denied'});
  await assert.rejects(h.api.upload('#00001'),/隐私授权/);
  assert.equal(h.calls.files.length,0); assert.equal(h.calls.modals.length,0); assert.equal(h.calls.uploads.length,0);
});

test('missing platform privacy API fails closed without invoking a picker', async () => {
  const h = harness(); delete h.context.wx.requirePrivacyAuthorize;
  await assert.rejects(h.api.upload('#00001'),/升级微信/);
  assert.equal(h.calls.files.length,0); assert.equal(h.calls.uploads.length,0);
});

test('already authorized WeChat users are still asked to confirm the file and destination', async () => {
  const h = harness(); h.handlers.privacySetting = options => options.success({needAuthorization:false});
  await h.api.upload('#00001');
  assert.deepEqual(h.calls.order,['privacy-setting','choose-document','confirm','upload']);
});

test('cancelling file selection does not show upload confirmation or retry selection', async () => {
  const h = harness(); h.handlers.chooseFile = options => options.fail({errMsg:'cancel'});
  await assert.rejects(h.api.upload('#00001'),/取消选择/);
  assert.equal(h.calls.files.length,1); assert.equal(h.calls.modals.length,0); assert.equal(h.calls.uploads.length,0);
});

test('declining the upload confirmation returns cancelled without sending a file', async () => {
  const h = harness(); h.handlers.confirm = options => options.success({confirm:false});
  const result = await h.api.upload('#00001');
  assert.equal(result.cancelled,true); assert.equal(h.calls.modals.length,1); assert.equal(h.calls.uploads.length,0);
});

test('an unavailable confirmation dialog does not upload', async () => {
  const h = harness(); h.handlers.confirm = options => options.fail({});
  await assert.rejects(h.api.upload('#00001'),/无法确认上传/);
  assert.equal(h.calls.uploads.length,0);
});

for (const stage of ['privacy','picker','confirmation']) {
  test(`page hide/cancel during ${stage} prevents a late successful callback from uploading`, async () => {
    const h = harness(); let callback, active = true;
    h.handlers[stage === 'privacy' ? 'privacyAuthorize' : stage === 'picker' ? 'chooseFile' : 'confirm'] = options => { callback = options; };
    const uploading = h.api.upload('#00001',{isCurrent:()=>active});
    const rejected = assert.rejects(uploading,/取消|配置已变化/);
    await flush(); active = false;
    callback.success(stage === 'picker' ? {tempFiles:[{name:'private.pdf',path:'wxfile://tmp/private',size:10}]} : {confirm:true});
    await rejected;
    assert.equal(h.calls.uploads.length,0);
    if (stage !== 'confirmation') assert.equal(h.calls.modals.length,0);
  });
}

for (const field of ['baseUrl','accessToken','mode']) {
  for (const stage of ['picker','confirmation']) {
    test(`changing ${field} during ${stage} cannot upload the old file to either service`, async () => {
      const h = harness(); let callback;
      h.handlers[stage === 'picker' ? 'chooseFile' : 'confirm'] = options => { callback = options; };
      const uploading = h.api.upload('#00001'); const rejected = assert.rejects(uploading,/配置已变化/);
      await flush();
      h.savePreferences({...h.getPreferences(),[field]:field === 'baseUrl' ? 'https://other.example' : field === 'mode' ? 'demo' : 'changed-access-code'});
      callback.success(stage === 'picker' ? {tempFiles:[{name:'private.pdf',path:'wxfile://tmp/private',size:10}]} : {confirm:true});
      await rejected;
      assert.equal(h.calls.uploads.length,0);
    });
  }
}

test('an already inactive page cannot even open the privacy dialog', async () => {
  const h = harness();
  await assert.rejects(h.api.upload('#00001',{isCurrent:()=>false}),/取消/);
  assert.equal(h.calls.order.length,0);
});

test('the file picker filters supported document types and sends the original basename as form data', async () => {
  const h = harness();
  h.handlers.chooseFile = options => options.success({tempFiles:[{name:'folder\\研究 报告.DOCX',path:'wxfile://tmp/renamed',size:1200}]});
  await h.api.upload('#00001');
  const picker = h.calls.files[0], upload = h.calls.uploads[0];
  assert.equal(picker.count,1); assert.equal(picker.type,'file');
  assert.deepEqual(Array.from(picker.extension),['pdf','txt','md','docx','png','jpg','jpeg']);
  assert.equal(upload.formData.original_filename,'研究 报告.DOCX');
  assert.equal(upload.formData.session_id,'#00001'); assert.equal(upload.filePath,'wxfile://tmp/renamed');
  assert.equal(upload.url,'https://private.example/api/v1/mobile/upload');
  assert.equal(upload.header.Authorization,'Bearer private-test-code'); assert.equal(upload.name,'file');
  assert.equal(upload.url.includes('private-test-code'),false); assert.equal(upload.timeout,120000);
});

test('image upload selects the album only, never the camera or document picker', async () => {
  const h = harness(); await h.api.upload('#00001',{kind:'image'});
  assert.equal(h.calls.files.length,0); assert.equal(h.calls.images.length,1);
  assert.deepEqual(Array.from(h.calls.images[0].sourceType),['album']);
  assert.deepEqual(Array.from(h.calls.images[0].sizeType),['original']);
  assert.equal(h.calls.uploads[0].filePath,'wxfile://tmp/album.jpg');
  assert.equal('original_filename' in h.calls.uploads[0].formData,false);
  assert.equal(h.calls.modals.length,1);
});

test('files larger than 20 MiB fail before confirmation and upload; exact boundary is allowed', async () => {
  const h = harness();
  h.handlers.chooseFile = options => options.success({tempFiles:[{name:'large.pdf',path:'wxfile://tmp/large',size:20 * 1024 * 1024 + 1}]});
  await assert.rejects(h.api.upload('#00001'),/20 MB/);
  assert.equal(h.calls.modals.length,0); assert.equal(h.calls.uploads.length,0);
  h.handlers.chooseFile = options => options.success({tempFiles:[{name:'boundary.pdf',path:'wxfile://tmp/boundary',size:20 * 1024 * 1024}]});
  await h.api.upload('#00001'); assert.equal(h.calls.uploads.length,1);
});

for (const name of ['archive.zip','macro.docm','payload.pdf.exe','legacy.doc','extensionless']) {
  test(`unsupported filename ${name} is rejected even if returned by the platform picker`, async () => {
    const h = harness(); h.handlers.chooseFile = options => options.success({tempFiles:[{name,path:'wxfile://tmp/document',size:1}]});
    await assert.rejects(h.api.upload('#00001'),/请选择 PDF/);
    assert.equal(h.calls.modals.length,0); assert.equal(h.calls.uploads.length,0);
  });
}

test('missing selected file path is rejected before asking for upload confirmation', async () => {
  const h = harness(); h.handlers.chooseFile = options => options.success({tempFiles:[{name:'notes.pdf',size:10}]});
  await assert.rejects(h.api.upload('#00001'),/无法读取/);
  assert.equal(h.calls.modals.length,0); assert.equal(h.calls.uploads.length,0);
});

for (const failure of ['network','status','invalid-json']) {
  test(`${failure} upload failure is not retried and does not expose raw server details`, async () => {
    const h = harness(); h.handlers.upload = options => failure === 'network'
      ? options.fail({errMsg:'secret local path'})
      : options.success({statusCode:failure === 'status' ? 500 : 200,data:'private server detail'});
    await assert.rejects(h.api.upload('#00001'),error => !/private server|secret local/.test(error.message));
    assert.equal(h.calls.uploads.length,1); assert.equal(h.calls.files.length,1); assert.equal(h.calls.modals.length,1);
  });
}

test('duplicate success callbacks cannot submit the same file twice', async () => {
  const h = harness();
  h.handlers.chooseFile = options => { const result = {tempFiles:[{name:'one.pdf',path:'wxfile://tmp/one',size:10}]}; options.success(result); options.success(result); };
  h.handlers.confirm = options => { options.success({confirm:true}); options.success({confirm:true}); };
  await h.api.upload('#00001');
  assert.equal(h.calls.modals.length,1); assert.equal(h.calls.uploads.length,1);
});
