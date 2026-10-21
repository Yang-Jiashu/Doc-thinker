import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { computed, effectScope, nextTick, ref, watch } from 'vue';
import {planGraphNavigation, GRAPH_NAVIGATION_LIMITS} from '../src/services/graph-navigation.js';

const root = new URL('../src/', import.meta.url);
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return {promise, resolve}; }

// Exercise the real page setup with all navigation and service effects captured.
function mount(page, options = {}) {
  const filename = `pages/${page}/index.vue`, source = readFileSync(new URL(filename, root), 'utf8');
  const {descriptor, errors} = parse(source, {filename});
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, {id:page});
  assert.deepEqual(compileTemplate({source:descriptor.template.content, filename, id:page, compilerOptions:{bindingMetadata:script.bindings}}).errors, []);
  const calls = {sessions:0, graph:0, files:0, chunks:[], ask:0, scroll:[], navigation:[], active:[]}, hooks = {};
  let active = 'a';
  const preferences = {mode:'demo', ...options.preferences};
  const api = {
    sessions:async()=>{ calls.sessions++; return [{id:'a',title:'A'},{id:'b',title:'B'}]; },
    graph:async()=>{ calls.graph++; return {nodes:[{id:'one',label:'节点一'},{id:'two',label:'节点二'}],edges:[]}; },
    files:async()=>{ calls.files++; return [{filename:'example.pdf',status:'completed'}]; },
    chunks:async(...args)=>{ calls.chunks.push(args); return [{content:'服务端原文'}]; },
    ask:async()=>{ calls.ask++; throw new Error('Navigation must not call a model'); },
    ...options.api,
  };
  const scope = {
    computed, ref, watch, nextTick:options.nextTick || nextTick, api, research:{capabilities:async()=>({enabled:false})},
    createMemoryObservationClient:()=>({invalidate(){},load:async()=>{throw new Error('Demo navigation must not request connected memory observations');}}),GraphCanvas:{},
    researchConnectionKey:()=>JSON.stringify(preferences),getPreferences:()=>({...preferences}), getActiveSession:()=>active,
    setActiveSession:value=>{ active=value; calls.active.push(value); },
    onShow:callback=>{ hooks.show=callback; }, onHide:callback=>{ hooks.hide=callback; }, onUnload:callback=>{ hooks.unload=callback; },
    uni:{switchTab:options=>calls.navigation.push(options), ...(options.noScroll ? {} : {pageScrollTo:options=>calls.scroll.push(options)})},
    planGraphNavigation, GRAPH_NAVIGATION_LIMITS, GraphVoiceInput:{}, ModeBanner:{}, WorkspaceBar:{}, BrandHeader:{}, CapabilityBoundary:{},
  };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default','return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const effects = effectScope(), ui = effects.run(()=>component.setup({}, {expose(){}}));
  return {ui, hooks, calls, template:descriptor.template.content, dispose:()=>{ hooks.hide?.(); effects.stop(); }};
}

test('an explicit graph selection reveals its loading detail immediately with only its existing evidence read', async t => {
  const response = deferred();
  const f = mount('graph', {api:{chunks:(...args)=>{ f.calls.chunks.push(args); return response.promise; }}});
  t.after(f.dispose); f.hooks.show(); await settle();
  const request = f.ui.selectNode(f.ui.nodes.value[0]);
  await nextTick();
  assert.equal(f.ui.evidenceLoading.value,true);
  assert.deepEqual(f.calls.chunks,[['a','one']]);
  assert.deepEqual(f.calls.scroll,[{selector:'#knowledge-evidence',duration:180}]);
  assert.match(f.template,/id="knowledge-evidence"/);
  f.ui.showSelectedEvidence(); await nextTick();
  assert.equal(f.calls.scroll.length,2);
  assert.equal(f.calls.chunks.length,1);
  assert.equal(f.calls.ask,0);
  response.resolve([{content:'节点一的片段'}]); await request;
  assert.deepEqual(f.ui.chunks.value,[{content:'节点一的片段'}]);
  assert.equal(f.calls.scroll.length,2,'a late read must not move the viewport again');
});

test('rapidly changing selection only scrolls to the current node after rendering', async t => {
  const tick = deferred(), f = mount('graph',{nextTick:()=>tick.promise});
  t.after(f.dispose); f.hooks.show(); await settle();
  const first=f.ui.selectNode(f.ui.nodes.value[0]), second=f.ui.selectNode(f.ui.nodes.value[1]);
  assert.equal(f.calls.scroll.length,0);
  tick.resolve(); await Promise.all([first,second]); await settle();
  assert.equal(f.ui.selected.value.id,'two');
  assert.equal(f.calls.scroll.length,1);
  assert.deepEqual(f.calls.chunks,[['a','one'],['a','two']]);
});

test('queued graph scrolling cannot cross hide, selection clear or session reload', async t => {
  for (const boundary of ['hide','clear','session']) {
    const tick=deferred(), f=mount('graph',{nextTick:()=>tick.promise});
    t.after(f.dispose); f.hooks.show(); await settle();
    const request=f.ui.selectNode(f.ui.nodes.value[0]);
    if (boundary === 'hide') f.hooks.hide();
    else if (boundary === 'clear') f.ui.clearSelection();
    else await f.ui.changeSession('b');
    tick.resolve(); await request; await settle();
    assert.deepEqual(f.calls.scroll,[],boundary);
    assert.equal(f.ui.selected.value,null,boundary);
  }
});

test('a platform without pageScrollTo still shows selected evidence', async t => {
  const f=mount('graph',{noScroll:true}); t.after(f.dispose);
  f.hooks.show(); await settle(); await f.ui.selectNode(f.ui.nodes.value[0]); await nextTick();
  assert.equal(f.ui.selected.value.id,'one');
  assert.deepEqual(f.ui.chunks.value,[{content:'服务端原文'}]);
  assert.equal(f.ui.evidenceError.value,'');
});

test('ready documents offer an explicit current-session return without asking or uploading', async t => {
  const f=mount('documents'); t.after(f.dispose);
  f.hooks.show(); await settle();
  assert.equal(f.ui.readyCount.value,1);
  assert.deepEqual(f.calls.navigation,[]);
  f.ui.openReadyChat();
  assert.deepEqual(f.calls.navigation,[{url:'/pages/chat/index'}]);
  assert.equal(f.calls.active.at(-1),'a');
  assert.equal(f.calls.ask,0); assert.equal(f.calls.files,1);
  assert.match(f.template,/只回到助手，仍由你编辑并发送问题/);
  assert.match(f.template,/不限定单份文件/);
});

test('document continuation stays unavailable for loading, uploads, absent or unready sessions and hidden pages', async t => {
  const f=mount('documents'); t.after(f.dispose);
  f.hooks.show(); await settle();
  f.ui.loading.value=true; f.ui.openReadyChat(); f.ui.loading.value=false;
  f.ui.uploading.value=true; f.ui.openReadyChat(); f.ui.uploading.value=false;
  f.ui.sessionId.value=''; f.ui.openReadyChat(); f.ui.sessionId.value='a';
  f.ui.files.value=[{filename:'unready.pdf',status:'processing'}]; f.ui.openReadyChat();
  f.ui.files.value=[{filename:'ready.pdf',status:'ready'}]; f.hooks.hide(); f.ui.openReadyChat();
  assert.deepEqual(f.calls.navigation,[]);
  assert.equal(f.calls.ask,0);
});

test('invitation navigation cannot activate unopened graph or document capability', async t => {
  for (const page of ['graph','documents']) {
    const f=mount(page,{preferences:{mode:'connected',connectionKind:'invite'}}); t.after(f.dispose);
    f.hooks.show(); await settle();
    if (page === 'graph') { await f.ui.selectNode({id:'one',label:'One'}); f.ui.showSelectedEvidence(); }
    else { f.ui.sessionId.value='a'; f.ui.files.value=[{status:'ready'}]; f.ui.openReadyChat(); }
    assert.equal(f.calls.sessions,0); assert.equal(f.calls.graph,0); assert.equal(f.calls.files,0);
    assert.deepEqual(f.calls.chunks,[]); assert.deepEqual(f.calls.scroll,[]); assert.deepEqual(f.calls.navigation,[]);
    assert.equal(f.calls.ask,0);
  }
});

const connectedGraph = {
  nodes:[{id:'a',label:'气候目标'},{id:'b',label:'政策工具'},{id:'c',label:'可再生能源投资'},{id:'d',label:'国内供给增加'}],
  edges:[{source:'a',target:'b',description:'关联一'},{source:'b',target:'c',description:'关联二'},{source:'c',target:'d',description:'关联三'}],
};
function graphFixture(options = {}) {
  const f = mount('graph',{...options,api:{...options.api,graph:async(...args)=>{f.calls.graph++;return options.api?.graph ? options.api.graph(...args) : connectedGraph;}}});
  return f;
}

test('a relation request shows a bounded local route and goal without asking or prefetching evidence',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();
  f.ui.locatorDraft.value='气候目标和国内供给增加有什么联系？';await f.ui.runLocator();await nextTick();
  assert.equal(f.ui.locatorResult.value.kind,'path');assert.deepEqual(f.ui.routeNodes.value.map(n=>n.id),['a','b','c','d']);
  assert.equal(f.ui.contextGoal.value,f.ui.locatorDraft.value);assert.match(f.ui.currentLocation.value,/气候目标 — 政策工具/);
  assert.equal(f.ui.selected.value,null);assert.equal(f.calls.ask,0);assert.deepEqual(f.calls.chunks,[]);
  assert.equal(f.calls.graph,1);assert.equal(f.calls.scroll.at(-1).selector,'#graph-context');
  assert.match(f.ui.locatorResult.value.message,/不代表因果证明/);
});
test('opening route nodes reads only each explicit node and back restores the preceding visual context',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();
  f.ui.locatorDraft.value='气候目标和国内供给增加有什么联系';await f.ui.runLocator();
  await f.ui.selectNode(f.ui.routeNodes.value[1]);await f.ui.selectNode(f.ui.routeNodes.value[2]);
  assert.deepEqual(f.calls.chunks,[['a','b'],['a','c']]);assert.equal(f.ui.navigationHistory.value.length,2);
  assert.equal(f.ui.contextGoal.value,'气候目标和国内供给增加有什么联系');
  await f.ui.goBack();assert.equal(f.ui.selected.value.id,'b');assert.equal(f.ui.navigationHistory.value.length,1);assert.equal(f.ui.locatorInputRevision.value,1);
  await f.ui.goBack();assert.equal(f.ui.selected.value,null);assert.equal(f.ui.locatorResult.value.kind,'path');
  assert.deepEqual(f.calls.chunks,[['a','b'],['a','c'],['a','b']]);assert.equal(f.calls.ask,0);
});
test('repeating a local locate neither fills history with identical positions nor rereads source',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();
  f.ui.locatorDraft.value='气候目标和国内供给增加有什么联系';await f.ui.runLocator();await f.ui.runLocator();
  assert.equal(f.ui.navigationHistory.value.length,0);assert.deepEqual(f.calls.chunks,[]);
  f.ui.locatorDraft.value='找到政策工具';await f.ui.runLocator();const history=f.ui.navigationHistory.value.length;
  await f.ui.runLocator();assert.equal(f.ui.navigationHistory.value.length,history);assert.deepEqual(f.calls.chunks,[['a','b']]);
});
test('voice fills a locator draft only and never overwrites occupied text without an explicit choice',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();
  const transcript='找到政策工具';f.ui.receiveLocatorTranscript(transcript);
  assert.equal(f.ui.locatorDraft.value,transcript);assert.equal(f.ui.locatorResult.value,null);assert.deepEqual(f.calls.chunks,[]);
  f.ui.receiveLocatorTranscript('找到气候目标');assert.equal(f.ui.locatorDraft.value,transcript);
  assert.equal(f.ui.pendingVoiceDraft.value,'找到气候目标');await f.ui.runLocator();assert.equal(f.ui.selected.value,null);
  f.ui.resolveVoiceDraft(false);assert.equal(f.ui.locatorDraft.value,transcript);
  f.ui.receiveLocatorTranscript('找到气候目标');f.ui.resolveVoiceDraft(true);
  assert.equal(f.ui.locatorDraft.value,'找到气候目标');assert.equal(f.ui.selected.value,null);
  await f.ui.runLocator();assert.equal(f.ui.selected.value.id,'a');assert.equal(f.calls.ask,0);
});
test('full over-limit transcription remains editable and never triggers graph search or truncation',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();const text='字'.repeat(257);
  f.ui.receiveLocatorTranscript(text);assert.equal(f.ui.locatorDraft.value,text);assert.equal(f.ui.locatorTooLong.value,true);
  await f.ui.runLocator();assert.equal(f.ui.locatorResult.value,null);assert.equal(f.ui.selected.value,null);assert.deepEqual(f.calls.chunks,[]);
  f.ui.locatorDraft.value='找到政策工具';await f.ui.runLocator();assert.equal(f.ui.selected.value.id,'b');
});
test('ambiguous endpoints require explicit sequential selection and edited queries cannot reuse stale candidates',async t=>{
  const f=graphFixture({api:{graph:async()=>({nodes:[{id:'x1',label:'政策'},{id:'x2',label:'政策'},{id:'y1',label:'供给'},{id:'y2',label:'供给'}],edges:[{source:'x2',target:'y1'}]})}});
  t.after(f.dispose);f.hooks.show();await settle();f.ui.locatorDraft.value='政策和供给有什么联系';await f.ui.runLocator();
  assert.equal(f.ui.locatorResult.value.kind,'clarify');assert.equal(f.ui.locatorResult.value.slot,'source');
  await f.ui.chooseCandidate('foreign');assert.equal(f.ui.locatorResult.value.slot,'source');
  await f.ui.chooseCandidate('x2');assert.equal(f.ui.locatorResult.value.slot,'target');assert.equal(f.ui.choiceResolutions.value.source,'x2');
  f.ui.locatorDraft.value='已编辑';await f.ui.chooseCandidate('y1');assert.equal(f.ui.locatorResult.value.kind,'clarify');
  f.ui.locatorDraft.value='政策和供给有什么联系';await f.ui.chooseCandidate('y1');
  assert.equal(f.ui.locatorResult.value.kind,'path');assert.deepEqual(f.ui.routeNodes.value.map(n=>n.id),['x2','y1']);assert.deepEqual(f.calls.chunks,[]);
});
test('source and back utterances operate on current context instead of becoming new model questions',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();
  f.ui.locatorDraft.value='找到气候目标';await f.ui.runLocator();await f.ui.selectNeighbor(f.ui.selectedRelations.value[0]);
  const reads=f.calls.chunks.length;f.ui.locatorDraft.value='看原文';await f.ui.runLocator();await nextTick();
  assert.equal(f.ui.contextGoal.value,'找到气候目标');assert.equal(f.calls.chunks.length,reads);assert.equal(f.calls.scroll.at(-1).selector,'#knowledge-evidence');
  f.ui.locatorDraft.value='回到上一步';await f.ui.runLocator();assert.equal(f.ui.selected.value.id,'a');assert.equal(f.ui.contextGoal.value,'找到气候目标');assert.equal(f.calls.ask,0);
});
test('navigation trail is bounded and neither session changes nor page hides retain visual context',async t=>{
  const nodes=Array.from({length:18},(_,i)=>({id:String(i),label:`节点${i}`}));
  const f=graphFixture({api:{graph:async()=>({nodes,edges:[]})}});t.after(f.dispose);f.hooks.show();await settle();
  for(const node of f.ui.nodes.value) await f.ui.selectNode(node);
  assert.equal(f.ui.navigationHistory.value.length,12);f.ui.locatorDraft.value='保留当前';f.ui.receiveLocatorTranscript('新语音');
  await f.ui.changeSession('b');assert.equal(f.ui.navigationHistory.value.length,0);assert.equal(f.ui.contextGoal.value,'');assert.equal(f.ui.pendingVoiceDraft.value,'');
  f.ui.locatorDraft.value='找到节点0';await f.ui.runLocator();f.hooks.hide();
  assert.equal(f.ui.pageVisible.value,false);assert.equal(f.ui.locatorDraft.value,'');assert.equal(f.ui.locatorResult.value,null);assert.equal(f.ui.selected.value,null);
  const reads=f.calls.chunks.length;f.ui.receiveLocatorTranscript('迟到的转写');await f.ui.runLocator();await f.ui.goBack();
  assert.equal(f.ui.locatorDraft.value,'');assert.equal(f.calls.chunks.length,reads);
});
test('recording blocks graph mutations while a pending old evidence response cannot overwrite a new route',async t=>{
  const pending=deferred();const f=graphFixture({api:{chunks:()=>pending.promise}});t.after(f.dispose);f.hooks.show();await settle();
  f.ui.voiceBusy.value=true;f.ui.locatorDraft.value='找到政策工具';await f.ui.runLocator();await f.ui.selectNode(f.ui.nodes.value[0]);await f.ui.changeSession('b');await f.ui.refreshGraph();
  assert.equal(f.ui.selected.value,null);assert.equal(f.ui.sessionId.value,'a');assert.equal(f.calls.graph,1);
  f.ui.voiceBusy.value=false;const old=f.ui.selectNode(f.ui.nodes.value[0]);
  f.ui.locatorDraft.value='政策工具和国内供给增加有什么联系';await f.ui.runLocator();
  pending.resolve([{content:'OLD'}]);await old;assert.equal(f.ui.selected.value,null);assert.deepEqual(f.ui.chunks.value,[]);assert.equal(f.ui.locatorResult.value.kind,'path');
});
test('navigation keyboard actions reject IME/repeats and plain Enter in a text field does not search',async t=>{
  const f=graphFixture();t.after(f.dispose);f.hooks.show();await settle();f.ui.locatorDraft.value='找到政策工具';
  let prevented=0;const event={key:'Enter',preventDefault:()=>prevented++};
  for(const flags of [{},{ctrlKey:true,isComposing:true},{metaKey:true,keyCode:229},{ctrlKey:true,repeat:true}]) f.ui.locatorKeydown({...event,...flags});
  for(const flags of [{repeat:true},{isComposing:true},{ctrlKey:true},{defaultPrevented:true}]) f.ui.navigationKeydown({...event,...flags},'locate');
  assert.equal(f.ui.locatorResult.value,null);assert.equal(prevented,0);
  f.ui.locatorKeydown({...event,ctrlKey:true});await settle();assert.equal(f.ui.selected.value.id,'b');assert.equal(prevented,1);assert.equal(f.calls.ask,0);
});
test('missing paths and external node objects cannot invent nodes, edges or evidence',async t=>{
  const f=graphFixture({api:{graph:async()=>({...connectedGraph,edges:[]})}});t.after(f.dispose);f.hooks.show();await settle();
  f.ui.locatorDraft.value='气候目标和国内供给增加有什么联系';await f.ui.runLocator();assert.equal(f.ui.locatorResult.value.kind,'none');assert.deepEqual(f.ui.routeNodes.value,[]);
  await f.ui.selectNode({id:'outside',label:'注入实体'});assert.equal(f.ui.selected.value,null);assert.deepEqual(f.calls.chunks,[]);
  assert.match(f.template,/实体原文不一定支持每条边/);assert.doesNotMatch(f.template,/v-html/);
});
