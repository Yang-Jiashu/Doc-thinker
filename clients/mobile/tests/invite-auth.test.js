import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import {webcrypto} from 'node:crypto';
import {sanitizeReasoningReceipts} from '../src/services/reasoning-evidence.js';
import {knowledgeSkillMetadata} from '../src/services/knowledge-skills.js';

const require = createRequire(import.meta.url);
const {preprocess} = require('@dcloudio/uni-cli-shared/lib/preprocess');
const read = path => readFileSync(new URL('../src/'+path,import.meta.url),'utf8');
const source = ['services/research-source-contract.js','services/api.js','services/invite-auth.js'].map(read).join('\n').replace(/^import[^;]+;\s*$/gm,'').replaceAll('export ','');
const flush = async () => {for (let i = 0; i < 32; i++) await Promise.resolve();};
const deferred = () => {let resolve;const promise = new Promise(done => {resolve = done;});return {promise,resolve};};
const auth = {access_token:'private_user_token_not_for_storage',expires_in:604800,user:{id:'user_a',username:'invited_user'}};
const fields = {username:'invited_user',password:'correct password 123',invite_code:'i'.repeat(40)};
const ok = data => ({ok:true,status:200,json:async () => data});
function harness({origin = 'https://invite.example',storage = new Map(),platform = 'H5'} = {}) {
  const calls = [], timers = new Map(), navigation = []; let clock = 1000;
  const handlers = {fetch:async (url,options) => url.endsWith('/status') ? ok({service:'docthinker-invite',registration:'invitation-only'}) : url.endsWith('/login') || url.endsWith('/register') ? ok(auth)
    : url.endsWith('/query') ? ok({answer:'answer',sources:[],request_id:JSON.parse(options.body).request_id,request_status:'completed'}) : ok({answer:'answer',sessions:[]})};
  const context = {
    AbortController,TextEncoder,Blob,File,FormData,crypto:webcrypto,Date:{now:()=>clock},sanitizeReasoningReceipts,knowledgeSkillMetadata,
    window:{location:{origin,pathname:'/index.html',replace:value=>navigation.push(value),reload:()=>navigation.push('reload')}},
    uni:{getStorageSync:key=>storage.get(key),setStorageSync:(key,value)=>storage.set(key,value),request:()=>{throw new Error('Invitation must not use a redirect-following native transport');},chooseFile:()=>{throw new Error('No picker');}},
    ensureWeChatPrivacy:()=>{throw new Error('No WeChat privacy or file selection');},
    fetch:(url,options)=>{calls.push({url,options});return handlers.fetch(url,options);},
    setTimeout:(fn,ms)=>{const id = Symbol();timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id),
  };
  const code = preprocess(source,{H5:platform==='H5'},{type:'js'});
  vm.runInNewContext(code+'\nglobalThis.service = {api,getPreferences,savePreferences,getActiveSession,setActiveSession,connection,request,uploadFile,checkConnection,clearInviteSession,selectInviteMode,activateInviteSession,inviteOrigin,getInviteIdentity,enterInvitation,checkInvitationService,authenticateInvitation,validateInvitationForm,cancelInviteRequest,logoutInvitation,reloadInvitation};',context);
  return {...context.service,calls,storage,handlers,context,timers,navigation,advance:ms=>{clock+=ms;}};
}

test('invitation terminal failure displays known code, drops raw data and never starts a retry',async()=>{
  const h=harness();h.activateInviteSession(auth);
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  h.handlers.fetch=async()=>ok({request_id:id,request_status:'failed',answer:'secret partial response',request_error:{code:'model_response_truncated',message:'secret provider error',automatic_retry:false}});
  const result=await h.api.ask('#00001','Question',{requestId:id});assert.equal(result.request_status,'failed');assert.match(result.request_error.message,/输出上限/);assert.doesNotMatch(JSON.stringify(result),/secret/);
  assert.equal(h.calls.length,1);assert.equal(h.timers.size,0);
  const status=await h.api.queryStatus('#00001',id);assert.deepEqual(JSON.parse(JSON.stringify(status)),JSON.parse(JSON.stringify(result)));assert.equal(h.calls.length,2);assert.equal(h.calls[1].options.method,'GET');
});
test('invitation answer and status keep the two added research sources but reject S11',async()=>{
  const h=harness();h.activateInviteSession(auth);
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  const sources=Array.from({length:11},(_,index)=>({id:`fb27e5ae-5cb9-4650-9dfd-${String(index+1).padStart(12,'0')}`,label:`S${index+1}`,content:`Original ${index+1}`}));
  h.handlers.fetch=async()=>ok({request_id:id,request_status:'completed',answer:'Evidence [S9] [S10]',research_core:true,sources});
  for(const result of [await h.api.ask('#00001','Question',{requestId:id}),await h.api.queryStatus('#00001',id)]) {
    assert.equal(result.sources.length,10);assert.equal(result.sources[8].label,'S9');assert.equal(result.sources[9].label,'S10');
    assert.equal(result.citation_validation.entailment_verified,false);
  }
  assert.equal(h.calls.length,2);assert.equal(h.calls[1].options.method,'GET');
});

test('default demo is unchanged and opening invitation only checks the same-origin status',async () => {
  const h = harness();assert.equal(h.getPreferences().mode,'demo');assert.equal(h.calls.length,0);
  h.enterInvitation();await h.checkInvitationService();assert.equal(h.calls.length,1);
  assert.equal(h.calls[0].url,'https://invite.example/api/v1/invite/status');
  assert.equal(h.calls[0].options.method,'GET');assert.equal(h.calls[0].options.body,undefined);
  assert.equal(h.getPreferences().invitePending,true);assert.equal(h.getInviteIdentity(),null);
  await assert.rejects(h.api.sessions(),/尚未登录/);assert.equal(h.calls.length,1);
});
test('registration sends credentials only in an explicit same-origin JSON POST with no redirects or cookies',async () => {
  const h = harness();await h.authenticateInvitation(fields,{register:true});
  assert.equal(h.calls.length,1);const {url,options} = h.calls[0];
  assert.equal(url,'https://invite.example/api/v1/invite/register');assert.equal(options.method,'POST');
  assert.equal(options.redirect,'error');assert.equal(options.credentials,'omit');assert.equal(options.cache,'no-store');
  assert.deepEqual(JSON.parse(options.body),fields);
  for (const secret of [auth.access_token,fields.password,fields.invite_code,auth.user.username,auth.user.id]) assert.equal(JSON.stringify([...h.storage]).includes(secret),false);
  assert.equal(h.getInviteIdentity().username,'invited_user');assert.equal(h.getPreferences().invitePending,false);
});
test('invitation mobile calls are same-origin user-token requests and force unopened features off',async () => {
  const h = harness();await h.authenticateInvitation(fields);h.savePreferences({...h.getPreferences(),useContext:false,useCache:true,useMemory:true,useEvolution:true});
  await h.api.ask('#00001','hello');const {url,options} = h.calls[1];
  assert.equal(url,'https://invite.example/api/v1/mobile/query');assert.equal(options.headers.Authorization,'Bearer '+auth.access_token);
  assert.equal(options.redirect,'error');assert.equal(options.credentials,'omit');assert.equal(options.cache,'no-store');
  const data = JSON.parse(options.body);for(const key of ['use_memory','use_self_evolution','use_llm_cache','use_conversation_context']) assert.equal(data[key],false);
  assert.match(data.request_id,/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.throws(()=>h.savePreferences({...h.getPreferences(),baseUrl:'https://other.example'}),/当前网站/);
  assert.throws(()=>h.savePreferences({...h.getPreferences(),mode:'demo'}),/退出/);
  h.context.window.location.origin = 'https://different.example';assert.throws(()=>h.connection(),/尚未登录/);
});
test('explicit continuation retains the same request id and original context option without automatic retries',async()=>{
  const h=harness();await h.authenticateInvitation(fields);
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  h.handlers.fetch=async()=>({ok:true,status:202,json:async()=>({request_id:id,request_status:'running'})});
  const result=await h.api.ask('#00001','Original question',{requestId:id,useConversationContext:false});
  assert.equal(result.request_status,'running');assert.equal(h.calls.length,2);
  h.savePreferences({...h.getPreferences(),useContext:true});
  await h.api.ask('#00001','Original question',{requestId:id,useConversationContext:false});
  assert.deepEqual(JSON.parse(h.calls[1].options.body),JSON.parse(h.calls[2].options.body));
  assert.equal(h.calls.length,3);assert.ok(!JSON.stringify([...h.storage]).includes('Original question'));
});
test('202 never masquerades as a completed or legacy answer and 200 must carry a completed receipt',async()=>{
  const h=harness();await h.authenticateInvitation(fields);const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  for(const [status,result] of [[202,{answer:'legacy answer'}],[202,{request_id:id,request_status:'completed',answer:'Incorrect completion'}],[200,{request_id:id,request_status:'unknown'}],[200,{answer:'Legacy answer'}]]) {
    h.handlers.fetch=async()=>({ok:true,status,json:async()=>result});
    const before=h.calls.length;await assert.rejects(h.api.ask('#00001','Question',{requestId:id}),/未完成/);assert.equal(h.calls.length,before+1);
  }
});
test('invitation UTF8 budget and malformed ids reject before sending and retain unmodified caller text',async()=>{
  const h=harness();await h.authenticateInvitation(fields);const before=h.calls.length;
  for(const text of ['文'.repeat(2731),'🧠'.repeat(2049),'\ud800']) await assert.rejects(h.api.ask('#00001',text),/字节/);
  await assert.rejects(h.api.ask('#00001','Valid',{requestId:'../other'}),/编号/);
  await assert.rejects(h.api.ask('#00001','Valid',{useConversationContext:'true'}),/设置/);
  assert.equal(h.calls.length,before);
  await h.api.ask('#00001','🧠'.repeat(2048));assert.equal(h.calls.length,before+1);
});
test('status reads are one authenticated GET with encoded scope, not a new query or hidden polling',async()=>{
  const h=harness();await h.authenticateInvitation(fields);const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  h.handlers.fetch=async()=>ok({request_id:id,request_status:'unknown',detail:'secret-key'});
  const result=await h.api.queryStatus('#00001',id);assert.deepEqual({...result},{request_id:id,request_status:'unknown'});
  const request=h.calls.at(-1),url=new URL(request.url);
  assert.equal(url.pathname,'/api/v1/mobile/query/status');assert.equal(url.searchParams.get('session_id'),'#00001');assert.equal(url.searchParams.get('request_id'),id);assert.equal(url.hash,'');
  assert.equal(request.options.method,'GET');assert.equal(request.options.body,undefined);assert.equal(h.calls.length,2);
  await assert.rejects(h.api.queryStatus('#00002','../foreign'),/有效/);assert.equal(h.calls.length,2);
});
test('wrong, malformed or incomplete status never becomes a successful answer or exposes response prose',async()=>{
  const h=harness();await h.authenticateInvitation(fields);const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  for(const result of [null,{request_id:'other',request_status:'completed',answer:'secret-key'},{request_id:id,request_status:'successful',answer:'secret-key'},{request_id:id,request_status:'completed',answer:'  '},{answer:'legacy response'}]) {
    h.handlers.fetch=async()=>ok(result);
    await assert.rejects(h.api.queryStatus('#00001',id),error=>/不符合协议/.test(error.message)&&!error.message.includes('secret-key'));
  }
});
test('history restores at most five scoped request references without storing bodies or another network read',async()=>{
  const h=harness();await h.authenticateInvitation(fields);const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  h.handlers.fetch=async()=>ok({history:[{role:'user',content:'History question'}],query_requests:[{request_id:id,request_status:'unknown',question:'must not copy hidden field'},{request_id:'invalid',request_status:'running'}]});
  await h.api.history('#00001');assert.equal(h.calls.length,2);
  const refs=h.api.pendingQueries('#00001');assert.deepEqual(JSON.parse(JSON.stringify(refs)),[{request_id:id,request_status:'unknown'}]);
  assert.equal(h.api.pendingQueries('#00002').length,0);assert.ok(!JSON.stringify([...h.storage]).includes(id));
  refs[0].request_status='completed';assert.equal(h.api.pendingQueries('#00001')[0].request_status,'unknown');
  h.clearInviteSession();assert.equal(h.api.pendingQueries('#00001').length,0);
});
test('invitation business failures preserve fixed status guidance without exposing worker errors or retrying',async () => {
  const h=harness();await h.authenticateInvitation(fields);
  for(const [status,pattern] of [[401,/登录已过期/],[403,/没有访问权限/],[429,/额度已达上限/],[502,/查看历史/],[504,/检查历史/]]) {
    const before=h.calls.length;
    h.handlers.fetch=async()=>({ok:false,status,json:async()=>({detail:'PRIVATE KEY /internal/path'})});
    await assert.rejects(h.api.ask('#00001','hello'),error=>error.status===status && pattern.test(error.message) && !error.message.includes('PRIVATE'));
    assert.equal(h.calls.length,before+1);
  }
  h.handlers.fetch=async()=>{throw new Error('PRIVATE KEY');};
  await assert.rejects(h.api.ask('#00001','hello'),error=>/不会自动重复提交/.test(error.message) && !error.message.includes('PRIVATE'));
});
test('feedback travels only to current authenticated origin without sending answer text, API keys or AI requests',async()=>{
  const h=harness();await h.authenticateInvitation(fields);
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  await h.api.submitFeedback('#00001',id,{rating:'inaccurate',note:'漏了关键条件'});
  const {url,options}=h.calls.at(-1);assert.equal(url,'https://invite.example/api/v1/mobile/feedback');
  assert.equal(options.method,'POST');assert.equal(options.headers.Authorization,'Bearer '+auth.access_token);
  assert.equal(options.redirect,'error');assert.equal(options.credentials,'omit');assert.equal(options.cache,'no-store');
  assert.deepEqual(JSON.parse(options.body),{session_id:'#00001',answer_id:id,rating:'inaccurate',note:'漏了关键条件'});
  assert.equal(h.calls.filter(call=>call.url.endsWith('/query')).length,0);
  const before=h.calls.length;await assert.rejects(h.api.submitFeedback('#00001','../other',{rating:'helpful'}));
  await assert.rejects(h.api.submitFeedback('#00001',id,{rating:'other'}));
  await assert.rejects(h.api.submitFeedback('#00001',id,{rating:'helpful',note:'🧠'.repeat(501)}));assert.equal(h.calls.length,before);
});
test('refresh keeps nonsecret invitation mode, never old private token, session, demo or server',async () => {
  const h = harness();h.savePreferences({mode:'connected',baseUrl:'https://old-private.example',accessToken:'old_private_token'});h.setActiveSession('private-session');
  await h.authenticateInvitation(fields);h.setActiveSession('invited-session');
  const refreshed = harness({storage:h.storage});
  assert.equal(refreshed.getPreferences().connectionKind,'invite');assert.equal(refreshed.getPreferences().mode,'connected');
  assert.equal(refreshed.getPreferences().accessToken,'');assert.equal(refreshed.getPreferences().baseUrl,'https://invite.example');
  assert.equal(refreshed.getActiveSession(),'');assert.equal(refreshed.getInviteIdentity(),null);
  await assert.rejects(refreshed.api.ask('invited-session','draft'),/尚未登录/);assert.equal(refreshed.calls.length,0);
});
test('expired invitation is blocked locally without silently using demo or making a request',async () => {
  const h = harness();await h.authenticateInvitation(fields);h.advance(604800001);
  assert.equal(h.getPreferences().invitePending,true);assert.equal(h.getInviteIdentity(),null);
  await assert.rejects(h.api.ask('a','hi'),/登录已过期/);assert.equal(h.calls.length,1);
});
test('logout immediately cancels pending user responses and clears user/session before revocation returns',async () => {
  const h = harness();await h.authenticateInvitation(fields);h.setActiveSession('#00001');
  const question = deferred(), logout = deferred();
  h.handlers.fetch = url => url.endsWith('/logout') ? logout.promise : question.promise;
  const pending = h.api.ask('#00001','private question');const rejected = assert.rejects(pending,/身份或连接/);
  const exiting = h.logoutInvitation();
  assert.equal(h.getInviteIdentity(),null);assert.equal(h.getPreferences().accessToken,'');assert.equal(h.getActiveSession(),'');
  await flush();assert.equal(h.calls[1].options.signal.aborted,true);
  question.resolve(ok({answer:'old user answer'}));logout.resolve(ok({ok:true}));await rejected;await exiting;
  assert.equal(h.getPreferences().invitePending,true);assert.equal(h.calls.at(-1).url,'https://invite.example/api/v1/invite/logout');
  assert.equal(h.calls.at(-1).options.headers.Authorization,'Bearer '+auth.access_token);
  h.reloadInvitation();assert.deepEqual(h.navigation,['https://invite.example/index.html#/pages/invite/index','reload']);
});
test('logout transport failure never restores credentials; only explicit leave returns to offline demo',async () => {
  const h = harness();await h.authenticateInvitation(fields);
  h.handlers.fetch = async () => {throw new Error('SECRET backend error');};await h.logoutInvitation();
  assert.equal(h.getPreferences().invitePending,true);assert.equal(h.getActiveSession(),'');
  await h.logoutInvitation({leave:true});assert.equal(h.getPreferences().mode,'demo');assert.equal(h.getPreferences().baseUrl,'');
  const count = h.calls.length;await h.api.sessions();assert.equal(h.calls.length,count);
});
test('cancelling a login prevents a late response from restoring identity, including before transport starts',async () => {
  for(const beforeFetch of [true,false]) {
    const h = harness(), wait = deferred();h.handlers.fetch = () => wait.promise;
    const pending = h.authenticateInvitation(fields);const rejected = assert.rejects(pending,/取消/);
    if (!beforeFetch) await flush();h.cancelInviteRequest();await rejected;
    wait.resolve(ok(auth));await flush();assert.equal(h.getInviteIdentity(),null);
    assert.equal(h.calls.length,beforeFetch?0:1);
  }
});
test('late parsing after logout cannot write a previous identity or resolve previous user content',async () => {
  const h = harness(), parsed = deferred();h.handlers.fetch = async () => ({ok:true,json:()=>parsed.promise});
  const pending = h.authenticateInvitation(fields);const rejected = assert.rejects(pending,/取消/);await flush();h.cancelInviteRequest();parsed.resolve(auth);await rejected;await flush();
  assert.equal(h.getInviteIdentity(),null);
});
test('singleflight and deadline do not retry or echo transport/server content',async () => {
  const h = harness(), wait = deferred();h.handlers.fetch = ()=>wait.promise;
  const first = h.checkInvitationService();await assert.rejects(h.checkInvitationService(),/正在处理/);await flush();assert.equal(h.calls.length,1);
  const timeout = assert.rejects(first,/连接超时/);[...h.timers.values()][0].fn();await timeout;
  assert.equal(h.calls[0].options.signal.aborted,true);wait.resolve(ok({service:'docthinker-invite',registration:'invitation-only'}));await flush();assert.equal(h.calls.length,1);
  for(const response of [{ok:false,status:500,json:async()=>({detail:'PRIVATE KEY'})},{ok:true,json:async()=>{throw new SyntaxError('PRIVATE KEY');}}]) {
    h.handlers.fetch = async()=>response;await assert.rejects(h.checkInvitationService(),error=>!error.message.includes('PRIVATE KEY'));
  }
});
test('status and malformed authentication response fail closed',async () => {
  for(const data of [null,{},[],{service:'other',registration:'invitation-only'}]) {
    const h = harness();h.handlers.fetch = async()=>ok(data);await assert.rejects(h.checkInvitationService(),/不是兼容/);
  }
  for(const data of [{...auth,expires_in:604801},{...auth,access_token:'bad token\n'},{...auth,user:{username:'invalid user',id:'x'}},null]) {
    const h = harness();h.handlers.fetch = async()=>ok(data);await assert.rejects(h.authenticateInvitation(fields));assert.equal(h.getInviteIdentity(),null);
  }
});
test('unopened invitation features reject before file selection, privacy or network',async () => {
  const h = harness();h.enterInvitation();
  for(const method of ['files','upload','memory','graph','chunks']) await assert.rejects(h.api[method]('a'),/尚未开放/);
  assert.equal(h.calls.length,0);
});
test('private connection probe is unavailable to invitation tokens and never uses redirect-following transport',async () => {
  const h=harness();await h.authenticateInvitation(fields);const count=h.calls.length;
  const result=await h.checkConnection();assert.equal(result.connected,false);assert.equal(result.reason,'invitation-only');assert.equal(h.calls.length,count);
});
test('direct upload export cannot bypass the invitation chat-only boundary, including paid ASR uploads',async () => {
  const h=harness();await h.authenticateInvitation(fields);const count=h.calls.length;
  for(const path of ['/upload','/voice/transcribe']) assert.throws(()=>h.uploadFile(path,'fake-path',{},new Blob(['fake'])),/尚未开放/);
  assert.equal(h.calls.length,count);
});
test('unsafe origins and native builds cannot send invitation credentials',async () => {
  for(const origin of ['http://192.168.1.2','file://','null','https://user:pass@example.com','https://example.com/path']) {
    const h = harness({origin});await assert.rejects(h.authenticateInvitation(fields));assert.equal(h.calls.length,0);
  }
  for(const platform of ['MP-WEIXIN','APP-PLUS']) {
    const h = harness({platform});await assert.rejects(h.authenticateInvitation(fields),/仅支持/);assert.equal(h.calls.length,0);
  }
  for(const origin of ['http://localhost:5174','http://127.0.0.1:5174']) { const h = harness({origin});await h.checkInvitationService();assert.equal(h.calls.length,1); }
});
test('invalid form fields never leave the browser and password spaces are not silently trimmed',async () => {
  for(const changes of [{username:' bad'},{username:'xy'},{username:'a@example.com'},{password:'short'},{password:' '.repeat(12)},{password:'x'.repeat(129)},{invite_code:'https://invite.example/#secret'}]) {
    const h = harness();await assert.rejects(h.authenticateInvitation({...fields,...changes},{register:true}));assert.equal(h.calls.length,0);
  }
  const h = harness();await h.authenticateInvitation({...fields,password:'  valid password  '});assert.equal(JSON.parse(h.calls[0].options.body).password,'  valid password  ');
  await assert.rejects(h.authenticateInvitation({...fields,username:'second_user'}),/先退出/);assert.equal(h.calls.length,1);
});
