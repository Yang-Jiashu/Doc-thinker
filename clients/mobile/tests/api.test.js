import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
let api, storage, requests, moduleIndex = 0;
beforeEach(async () => {
  storage = new Map(); requests = [];
  globalThis.uni = {getStorageSync:key=>storage.get(key),setStorageSync:(key,value)=>storage.set(key,value),
    request:options=>{requests.push(options); options.success({statusCode:200,data:{answer:'test',sessions:[]}});},
    uploadFile:()=>{throw new Error('No real uploads in tests');}};
  api = await import(`../src/services/api.js?case=${moduleIndex++}`);
});
test('demo flows never request network, microphone or real files',async () => {
  assert.equal(api.getPreferences().mode,'demo');
  const sessions = await api.api.sessions();
  const sid = sessions[0].id;
  assert.match((await api.api.ask(sid,'测试')).answer,/演示回答/);
  assert.equal((await api.api.history(sid)).length,2);
  assert.equal((await api.api.graph(sid)).nodes.length,6);
  assert.match((await api.api.chunks(sid,'气候目标'))[0].content,/模拟/);
  const memory = await api.api.memory(sid);
  assert.equal(memory.scope,'session'); assert.equal(memory.items.length,2);
  assert.ok(memory.items.every(item => item.summary.includes('示例') && !('confidence' in item)));
  await api.api.upload(sid);
  assert.equal((await api.api.files(sid)).length,2);
  assert.equal(requests.length,0);
});

test('known terminal model failures keep only safe local messages and status reads never resend',async()=>{
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  for(const code of ['model_response_truncated','model_response_empty','model_response_invalid']) {
    const data={request_id:id,request_status:'failed',answer:'secret partial response',request_error:{code,message:'secret upstream text',automatic_retry:false,token:'secret-key'}};
    uni.request=options=>{requests.push(options);options.success({statusCode:200,data});};
    const failed=await api.api.ask('#00001','Original question',{requestId:id});
    assert.equal(failed.request_status,'failed');assert.equal(failed.request_error.code,code);
    assert.equal(failed.request_error.automatic_retry,false);assert.match(failed.request_error.message,/没有自动重试/);
    assert.doesNotMatch(JSON.stringify(failed),/secret/);assert.ok(!('answer' in failed));
    const before=requests.length;assert.deepEqual(await api.api.queryStatus('#00001',id),failed);
    assert.equal(requests.length,before+1);assert.equal(requests.at(-1).method,'GET');
  }
});

test('unknown or retry-enabled failure codes are protocol errors rather than instructions',async()=>{
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  for(const request_error of [null,{}, {code:'secret-provider-error',automatic_retry:false}, {code:'model_response_empty',automatic_retry:true}]) {
    uni.request=options=>{requests.push(options);options.success({statusCode:200,data:{request_id:id,request_status:'failed',request_error}});};
    const before=requests.length;await assert.rejects(api.api.ask('#00001','Question',{requestId:id}),/失败回执不符合协议/);
    assert.equal(requests.length,before+1);
  }
});

test('legacy untracked known failure is sanitized without fabricating request identity',async()=>{
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  uni.request=options=>{requests.push(options);options.success({statusCode:200,data:{request_status:'failed',answer:'secret',request_error:{code:'model_response_empty',message:'secret',automatic_retry:false}}});};
  const result=await api.api.ask('#00001','Question');assert.equal(result.request_status,'failed');assert.ok(!('request_id' in result));assert.doesNotMatch(JSON.stringify(result),/secret/);assert.equal(requests.length,1);
});
test('research metadata keeps evidence limitations as bounded codes without trusting remote messages',()=>{
  const codes=['evidence_coverage_proxy','evidence_facets_uncovered','evidence_anchors_uncovered',
    'complete_statement_exceeds_budget','semantic_retrieval_unavailable','semantic_retrieval_windowed',
    'literal_coverage_scan_limited','source_boundary_unverified','source_antecedent_unverified'];
  const input=[...codes,...codes,'upstream-secret-code'].map(code=>({code,message:'upstream-secret-message',count:999999}));
  const result=api.researchAnswerMetadata({research_core:true,retrieval_gaps:input});
  assert.deepEqual(result.retrieval_gaps,codes.map(code=>({code})));
  assert.doesNotMatch(JSON.stringify(result),/upstream-secret|999999/);
  assert.equal(result.citation_validation.entailment_verified,false);
  assert.deepEqual(api.researchAnswerMetadata({retrieval_gaps:input}),{});
  assert.deepEqual(api.researchAnswerMetadata({research_core:true,retrieval_gaps:Array(100).fill(null)}).retrieval_gaps,[]);
});
test('citation-shaped control IDs survive as warnings and never become source references',async()=>{
  const unknown=['C1','L2','E3','P1','Q1','A1','G1','c0','S99','K99'];
  const value=api.researchAnswerMetadata({research_core:true,sources:[],citation_validation:{
    valid:true,entailment_verified:true,unknown_labels:[...unknown,'X1','private-message',{},'C'+'1'.repeat(17)],
    warnings:[{code:'unknown_citation_labels',message:'private-upstream-text'}]}});
  assert.deepEqual(value.citation_validation.unknown_labels,unknown);
  assert.equal(value.citation_validation.entailment_verified,false);
  assert.ok(!('sources' in value));
  assert.doesNotMatch(JSON.stringify(value),/private|X1/);
  const {answerDeliverable}=await import('../src/services/answer-deliverable.js');
  const exported=answerDeliverable({answer:'Scope limit [C1].',researchCore:true,
    citationValidation:value.citation_validation});
  assert.equal(exported.available,true);
  assert.match(exported.text,/引用缺失或标签不在本次来源内/);
  assert.match(exported.text,/Scope limit \[C1\]/);
});
test('knowledge helper field is omitted for old callers and only accepts automatic routing or off',async()=>{
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  await api.api.ask('#00001','普通提问');assert.equal('knowledge_skill_mode' in requests.at(-1).data,false);
  await api.api.ask('#00001','职场提问',{knowledgeSkillMode:'auto'});assert.equal(requests.at(-1).data.knowledge_skill_mode,'auto');
  await api.api.ask('#00001','关闭',{knowledgeSkillMode:'off'});assert.equal(requests.at(-1).data.knowledge_skill_mode,'off');
  const count=requests.length;for(const value of ['remote-executable',{},true,null]) await assert.rejects(api.api.ask('#00001','提问',{knowledgeSkillMode:value}),/知识辅助/);assert.equal(requests.length,count);
});
test('tracked recovery and history retain distinct bounded K knowledge references without adding them to S sources',async()=>{
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  const knowledge={id:'mao-workplace',name:'职场方法 · 毛选启发',version:'2',status:'ready',sources:[{id:'investigate',label:'K1',title:'核实事实',source_title:'历史启发篇目',source_url:'https://www.12371.cn/example.shtml',excerpt:'短摘',commentary:'原创方法解读',boundaries:['需核对情境']}]};
  const result={research_core:true,answer:'整理事实 [K1]',answer_id:id,request_id:id,request_status:'completed',sources:[],knowledge_skill:knowledge};
  globalThis.uni.request=options=>{requests.push(options);options.success({statusCode:200,data:options.url.includes('/history?')?{history:[{role:'assistant',content:result.answer,...result}]}:result});};
  const recovered=await api.api.queryStatus('#00001',id);assert.equal(recovered.knowledge_skill.sources[0].label,'K1');assert.deepEqual(recovered.sources,[]);assert.equal(recovered.knowledge_skill.validation_status,'design_unvalidated');
  const history=await api.api.history('#00001');assert.equal(api.researchAnswerMetadata(history[0]).knowledge_skill.sources[0].id,'investigate');assert.equal(requests.length,2);
});
test('original knowledge packs keep their attribution and constraints through receipt recovery and history',async()=>{
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  for(const skillId of ['research-ideation','literature-synthesis','evidence-writing','decision-review']) {
    const knowledge={id:skillId,name:skillId,version:'1',status:'ready',sources:[{id:'method',label:'K1',title:'保留待验证条件',reference_kind:'project_original',source_title:'DocThinker · 方法规则',source_url:'',published:'2026-10-11',excerpt:'',commentary:'项目方法知识，不是研究事实',boundaries:['原始证据要单独核对'],required_facts:['原文证据'],decision_rule:'证据不足不推断事实',stop_conditions:['关键前提无法核验']}]};
    const result={research_core:true,answer:'先核对事实 [K1]',answer_id:id,request_id:id,request_status:'completed',sources:[],knowledge_skill:knowledge};
    globalThis.uni.request=options=>{requests.push(options);options.success({statusCode:200,data:options.url.includes('/history?')?{history:[{role:'assistant',content:result.answer,...result}]}:result});};
    const recovered=await api.api.queryStatus('#00001',id), history=await api.api.history('#00001');
    assert.deepEqual(recovered.sources,[]);assert.deepEqual(history[0].sources,[]);
    for(const response of [recovered,api.researchAnswerMetadata(history[0])]) {assert.equal(response.knowledge_skill.id,skillId);assert.equal(response.knowledge_skill.sources[0].reference_kind,'project_original');assert.equal(response.knowledge_skill.sources[0].source_url,'');assert.deepEqual(response.knowledge_skill.sources[0].stop_conditions,['关键前提无法核验']);}
  }
  assert.equal(requests.length,8);
});
test('demo new session does not inherit graph or history',async () => {
  const sid = (await api.api.createSession('独立会话')).id;
  assert.deepEqual(await api.api.history(sid),[]);
  assert.deepEqual(await api.api.graph(sid),{nodes:[],edges:[]});
  assert.deepEqual((await api.api.memory(sid)).items,[]);
  await assert.rejects(api.api.ask('#00001','test'),/当前模式/);
});
test('demo answer feedback is local, replaceable, belongs to its saved session, and never calls a model',async()=>{
  const first=(await api.api.sessions())[0].id;
  const answer=await api.api.ask(first,'测试问题');assert.ok(api.isFeedbackAnswerId(answer.answer_id,true));
  await api.api.submitFeedback(first,answer.answer_id,{rating:'helpful'});
  await api.api.submitFeedback(first,answer.answer_id,{rating:'inaccurate',note:'缺少条件'});
  const records=(await api.api.feedback(first)).items;assert.equal(records.length,1);assert.equal(records[0].rating,'inaccurate');
  assert.deepEqual((await api.api.history(first)).at(-1).feedback,records[0]);
  const second=(await api.api.createSession('另一个项目')).id;
  assert.deepEqual((await api.api.feedback(second)).items,[]);
  await assert.rejects(api.api.submitFeedback(second,answer.answer_id,{rating:'helpful'}),/当前会话/);
  assert.equal(requests.length,0);
});
test('only persisted v4 answer IDs or explicit demo IDs enable feedback',()=>{
  assert.equal(api.isFeedbackAnswerId('ed0b7a48-cb27-430d-95f1-02b2757a69cc'),true);
  for(const value of [null,{},'demo-answer-2','ed0b7a48-cb27-130d-95f1-02b2757a69cc','ed0b7a48-cb27-430d-05f1-02b2757a69cc','../answer']) assert.equal(api.isFeedbackAnswerId(value),false);
  assert.equal(api.isFeedbackAnswerId('demo-answer-2',true),true);
});
test('memory pagination is scoped, encoded, bounded and never sent as a write',async () => {
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  await api.api.memory('#00001',{limit:10,offset:20});
  const url = new URL(requests[0].url);
  assert.equal(url.searchParams.get('session_id'),'#00001');
  assert.equal(url.searchParams.get('limit'),'10'); assert.equal(url.searchParams.get('offset'),'20');
  assert.equal(requests[0].method,'GET'); assert.equal(url.hash,'');
  for (const options of [{limit:21},{offset:-1},{offset:1001},{limit:0},{offset:1.5}]) await assert.rejects(api.api.memory('#00001',options),/分页/);
  assert.equal(requests.length,1);
});
test('origin validation rejects credential/query/path and nonlocal plaintext destinations',() => {
  for (const value of ['https://user:pass@host','https://host/path','https://host?key=a','https://host/#a','http://192.168.1.2:8001','javascript:alert(1)','https://host:99999','https://host%2f.other']) assert.throws(()=>api.normalizeBaseUrl(value));
  assert.equal(api.normalizeBaseUrl(' HTTPS://Example.COM:8443/ '),'https://example.com:8443');
  assert.equal(api.normalizeBaseUrl('http://127.0.0.1:8001'),'http://127.0.0.1:8001');
});
test('access token is memory only and not inherited when destination changes',() => {
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'private-test-token'});
  assert.equal(api.getPreferences().accessToken,'private-test-token');
  assert.equal(JSON.stringify([...storage.values()]).includes('private-test-token'),false);
  api.setActiveSession('#00001');
  api.savePreferences({...api.getPreferences(),baseUrl:'https://two.test'});
  assert.equal(api.getPreferences().accessToken,'');
  assert.equal(api.getActiveSession(),'');
  assert.throws(()=>api.connection(),/访问码/);
});
test('demo clears token and session cannot leak across modes',() => {
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  api.setActiveSession('#00001');
  api.savePreferences({...api.getPreferences(),mode:'demo'});
  assert.equal(api.getPreferences().accessToken,''); assert.equal(api.getActiveSession(),'');
});
test('connected question passes the four controls without auto-retry',async () => {
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test',useMemory:false,useContext:false,useCache:false,useEvolution:false});
  await api.api.ask('#00001','hello');
  assert.equal(requests.length,1);
  assert.equal(requests[0].url,'https://one.test/api/v1/mobile/query');
  assert.equal(requests[0].header.Authorization,'Bearer test');
  for(const field of ['use_memory','use_conversation_context','use_llm_cache','use_self_evolution']) assert.equal(requests[0].data[field],false);
  assert.equal('request_id' in requests[0].data,false,'legacy private profile retains its original protocol');
});
test('private research opts into tracked receipts and preserves all explicitly captured controls',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test',useMemory:true,useContext:true,useCache:true,useEvolution:true});
  uni.request=options=>{requests.push(options);options.success({statusCode:202,data:{request_id:id,request_status:'running',answer:'not an answer'}});};
  const options={requestId:id,useProjectMemory:false,useConversationContext:false,useLLMCache:false,useSelfEvolution:false};
  assert.deepEqual(await api.api.ask('#00001','Original research question',options),{request_id:id,request_status:'running'});
  assert.equal(requests.length,1);assert.equal(requests[0].data.request_id,id);
  for(const field of ['use_memory','use_conversation_context','use_llm_cache','use_self_evolution']) assert.equal(requests[0].data[field],false);
  assert.equal(JSON.stringify([...storage.values()]).includes(id),false);
});
test('legacy reasoning mode and memory time travel remain bounded protocol fields',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  uni.request=options=>{requests.push(options);options.success({statusCode:202,data:{request_id:id,request_status:'running'}});};
  await api.api.ask('#00001','Compare old decisions',{requestId:id,reasoningMode:'deduction',memoryAsOf:0});assert.equal(requests[0].data.reasoning_mode,'deduction');assert.equal(requests[0].data.memory_as_of,0);assert.equal(requests.length,1);
  for(const options of [{reasoningMode:'magic'},{memoryAsOf:'2026-10-10'},{memoryAsOf:Infinity},{memoryAsOf:-1}]) await assert.rejects(api.api.ask('#00001','No invalid request',{requestId:id,...options}));assert.equal(requests.length,1);
});
test('normal research chat omits reasoning mode and sends exactly one model request',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  uni.request=options=>{requests.push(options);options.success({statusCode:202,data:{request_id:id,request_status:'running'}});};
  await api.api.ask('#00001','帮我参考资料写一段讨论',{requestId:id,memoryAsOf:0});assert.equal(requests.length,1);assert.equal('reasoning_mode' in requests[0].data,false);assert.equal(requests[0].data.memory_as_of,0);
});
test('private tracked query rejects malformed receipt status and never falls back to an untracked paid call',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  for(const [status,data] of [[202,{answer:'Legacy answer'}],[202,{request_id:id,request_status:'completed',answer:'Premature answer'}],
    [200,{request_id:id,request_status:'running'}],[200,{request_id:'other',request_status:'completed',answer:'Wrong answer'}],
    [200,{request_id:id,request_status:'completed',answer:' '}],[undefined,{request_id:id,request_status:'completed',answer:'Wrong status'}]]) {
    uni.request=options=>{requests.push(options);options.success({statusCode:status,data});};
    const before=requests.length;await assert.rejects(api.api.ask('#00001','Question',{requestId:id}));assert.equal(requests.length,before+1);
  }
});
test('private query status is an explicit scoped GET and recovers all ten research sources',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  const sources=Array.from({length:11},(_,index)=>({id:String(index),label:`S${index+1}`,content:`Original ${index}`}));
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  uni.request=options=>{requests.push(options);options.success({statusCode:200,data:{request_id:id,request_status:'completed',answer:'Recovered [S1]',research_core:true,sources,citation_validation:{warnings:[{code:'source_revoked',message:'never-copy'}],entailment_verified:true}}});};
  const result=await api.api.queryStatus('#00001',id);assert.equal(result.sources.length,10);assert.equal(result.sources[9].label,'S10');
  assert.deepEqual(result.citation_validation.warnings,[{code:'source_revoked'}]);assert.equal(result.citation_validation.entailment_verified,false);
  const target=new URL(requests[0].url);assert.equal(requests[0].method,'GET');assert.equal(target.searchParams.get('session_id'),'#00001');assert.equal(target.searchParams.get('request_id'),id);assert.equal(target.hash,'');assert.equal(requests[0].data,undefined);
  for(const [sid,rid] of [['../project',id],['#00002','bad']]) await assert.rejects(api.api.queryStatus(sid,rid));assert.equal(requests.length,1);
});
test('private history retains only bounded opaque unresolved request references in memory',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  uni.request=options=>{requests.push(options);options.success({statusCode:200,data:{history:[],query_requests:[{request_id:id,request_status:'unknown',question:'Hidden quote',token:'must-not-copy'},{request_id:'invalid',request_status:'running'}]}});};
  await api.api.history('#00001');assert.deepEqual(api.api.pendingQueries('#00001'),[{request_id:id,request_status:'unknown'}]);assert.deepEqual(api.api.pendingQueries('#00002'),[]);
  const reference=api.api.pendingQueries('#00001')[0];reference.request_status='changed';assert.equal(api.api.pendingQueries('#00001')[0].request_status,'unknown');
  assert.doesNotMatch(JSON.stringify([...storage.values()]),/Hidden quote|must-not-copy|ed0b7a48/);
  api.savePreferences({...api.getPreferences(),baseUrl:'https://two.test'});assert.deepEqual(api.api.pendingQueries('#00001'),[]);
});
test('invalid private tracked inputs are rejected before network without truncation or weakening IDs',async()=>{
  const id='ed0b7a48-cb27-430d-95f1-02b2757a69cc';
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  for(const options of [{requestId:null},{requestId:'../id'},{requestId:id,useConversationContext:'off'},{requestId:id,useLLMCache:0},{requestId:id,useSelfEvolution:1}]) await assert.rejects(api.api.ask('#00001','Valid',options));
  await assert.rejects(api.api.ask('a','Valid',{requestId:id}));
  for(const question of ['文'.repeat(3000),'\ud800']) await assert.rejects(api.api.ask('#00001',question,{requestId:id}));
  assert.equal(requests.length,0);
});
test('session and entity identifiers are encoded, never fragments',async () => {
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  await api.api.chunks('#00001','A/#?& +');
  const url = new URL(requests[0].url);
  assert.equal(url.hash,''); assert.equal(url.searchParams.get('session_id'),'#00001');
  assert.equal(url.searchParams.get('entity_id'),'A/#?& +');
});
test('provider errors and network failures neither expose raw body nor retry',async () => {
  api.savePreferences({mode:'connected',baseUrl:'https://one.test',accessToken:'test'});
  let count = 0;
  uni.request=options=>{count++; options.success({statusCode:500,data:{detail:'secret-key /private/file'}});};
  await assert.rejects(api.api.ask('#00001','hello'),error=>!error.message.includes('secret-key'));
  assert.equal(count,1);
  uni.request=options=>{count++; options.fail({errMsg:'token'});};
  await assert.rejects(api.api.ask('#00001','hello'),/不会自动重复提交/);
  assert.equal(count,2);
});
