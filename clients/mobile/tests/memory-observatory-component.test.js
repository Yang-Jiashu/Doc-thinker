import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
import {ref,computed,watch,reactive,effectScope,nextTick} from 'vue';
import {sampleMemoryObservation} from '../src/services/memory-observability.js';

const filename=new URL('../src/components/MemoryObservatory.vue',import.meta.url),source=readFileSync(filename,'utf8');
const settle=async()=>{await nextTick();await new Promise(resolve=>setImmediate(resolve));await nextTick();};
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
function mount(overrides={}){
  const {descriptor,errors}=parse(source,{filename:String(filename)});assert.deepEqual(errors,[]);
  const script=compileScript(descriptor,{id:'observatory'}),template=compileTemplate({source:descriptor.template.content,filename:String(filename),id:'observatory',compilerOptions:{bindingMetadata:script.bindings}});assert.deepEqual(template.errors,[]);
  let key='identity-a';const hooks={},emissions=[],calls=[];
  const props=reactive({sessionId:'#00001',identity:'identity-a',active:true,enabled:true,mode:'research',demo:false,disabled:false,refreshKey:0,...overrides.props});
  const handlers={load:async sid=>sampleMemoryObservation(sid),...overrides.handlers};
  const client={load:async(...args)=>{calls.push(args);return handlers.load(...args);},invalidate:()=>{}};
  const scope={ref,computed,watch,onBeforeUnmount:fn=>{hooks.unmount=fn;},onHide:fn=>{hooks.hide=fn;},onUnload:fn=>{hooks.unload=fn;},researchConnectionKey:()=>key,createMemoryObservationClient:()=>client,sampleMemoryObservation};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope)),effects=effectScope();
  const ui=effects.run(()=>component.setup(props,{expose(){},emit:(...args)=>emissions.push(args)}));
  return {ui,props,calls,handlers,hooks,emissions,key:value=>{key=value;},dispose(){hooks.unmount();effects.stop();}};
}

test('observatory compiles responsive tabs, refresh and honest unknown display without model or polling code',async()=>{
  const f=mount();await settle();assert.equal(f.calls.length,1);assert.equal(f.ui.data.value.session_id,'#00001');assert.equal(f.ui.counter(null),'未提供');assert.equal(f.ui.counter(0),'0');
  assert.deepEqual(f.ui.tabs.map(tab=>tab.id),['layers','sources','events']);assert.match(source,/role="tablist"/);assert.match(source,/@media\(max-width:540px\)/);assert.match(source,/prefers-reduced-motion/);
  assert.doesNotMatch(source,/setInterval|setTimeout|\.ask\(|\.query\(|v-html|confidence/);assert.match(source,/没有连续历史时，不编造增长曲线/);f.dispose();
});
test('disabled feature and default invite boundary cannot request or manufacture observations',async()=>{
  const f=mount({props:{enabled:false}});await settle();assert.equal(f.calls.length,0);assert.equal(f.ui.data.value,null);
  f.props.enabled=true;await settle();assert.equal(f.calls.length,1);f.props.active=false;await settle();assert.equal(f.ui.data.value,null);assert.equal(f.ui.busy.value,false);f.dispose();
});
test('offline sample is explicit, local only, not a connected-service failure fallback',async()=>{
  const sample=mount({props:{demo:true,enabled:false,identity:''}});await settle();assert.equal(sample.calls.length,0);assert.equal(sample.ui.data.value.memories.length,1);assert.match(source,/离线交互样例 · 非真实增长/);assert.equal(sample.emissions.at(-1)[1].sample,true);sample.dispose();
  const down=mount({handlers:{load:async()=>{throw new Error('Temporary unavailable');}}});await settle();assert.equal(down.ui.data.value,null);assert.equal(down.ui.error.value,'Temporary unavailable');assert.equal(down.calls.length,1);down.dispose();
});
test('late session A data never overwrites B and hide/unmount invalidate in-flight results',async()=>{
  const a=deferred(),f=mount({handlers:{load:sid=>sid==='#00001'?a.promise:Promise.resolve(sampleMemoryObservation(sid))}});await settle();
  f.props.sessionId='#00002';await settle();assert.equal(f.ui.data.value.session_id,'#00002');a.resolve(sampleMemoryObservation('#00001'));await settle();assert.equal(f.ui.data.value.session_id,'#00002');
  const late=deferred();f.handlers.load=()=>late.promise;const read=f.ui.refresh();f.hooks.hide();late.resolve(sampleMemoryObservation('#00002'));await read;assert.equal(f.ui.data.value,null);assert.equal(f.emissions.at(-1)[1],null);f.dispose();
});
test('connection changes discard pending data, refresh is singleflight, and explicit mutation refresh re-reads',async()=>{
  const pending=deferred(),f=mount({handlers:{load:()=>pending.promise}});await settle();await f.ui.refresh();assert.equal(f.calls.length,1);
  f.key('identity-b');pending.resolve(sampleMemoryObservation());await settle();assert.equal(f.ui.data.value,null);
  f.props.identity='identity-b';f.handlers.load=async sid=>sampleMemoryObservation(sid);await settle();assert.equal(f.ui.data.value.session_id,'#00001');
  const before=f.calls.length;f.props.refreshKey++;await settle();assert.equal(f.calls.length,before+1);f.dispose();
});
test('live provenance source clicks emit only known non-redacted source IDs and never model requests',async()=>{
  const value=sampleMemoryObservation();value.graph.nodes[1].source_ids=['source-safe'];const f=mount({handlers:{load:async()=>value}});await settle();
  f.ui.openSource(value.graph.nodes[1]);assert.deepEqual(f.emissions.at(-1),['source','source-safe']);
  const before=f.emissions.length;f.ui.openSource({...value.graph.nodes[1],redacted:true});assert.equal(f.emissions.length,before);
  f.props.demo=true;await settle();f.ui.openSource(value.graph.nodes[1]);assert.notEqual(f.emissions.at(-1)[0],'source');f.dispose();
});
test('stored status and checked-window state are shown separately; absent event names stay out-of-window',async()=>{
  const value=sampleMemoryObservation();value.counts.memories.by_status={active:1,invalidated:null};value.window.states={memories:{invalidated:1}};
  const f=mount({handlers:{load:async()=>value}});await settle();assert.equal(f.ui.metrics.value[3].value,null);assert.equal(f.ui.validityStates.value[0].value,1);assert.equal(f.ui.eventName('not-returned'),'窗口外事件');assert.equal(f.ui.eventDate(null),'时间未提供');f.dispose();
});
test('keyboard activation rejects repeat/composition/disabled targets and supports tabs without mouse',async()=>{
  const f=mount();await settle();let clicks=0,prevented=0;const target={closest:()=>({hasAttribute:()=>false,getAttribute:()=>null,click:()=>{clicks++;}})};
  f.ui.activateButton({key:'Enter',target,preventDefault:()=>{prevented++;}});assert.equal(clicks,1);assert.equal(prevented,1);
  f.ui.activateButton({key:'Enter',target,repeat:true});f.ui.activateButton({key:'Enter',target,isComposing:true});assert.equal(clicks,1);f.dispose();
});
test('legacy connected snapshots are read in legacy mode and never labelled as confirmed memory',async()=>{
  const value=sampleMemoryObservation();value.profile='legacy_session_memory';value.counts.documents.total=null;value.counts.memories.by_status=null;value.counts.memories.versions=null;value.counts.memories.source_links=null;
  const f=mount({props:{mode:'legacy'},handlers:{load:async()=>value}});await settle();assert.equal(f.calls[0][1].mode,'legacy');assert.equal(f.ui.metrics.value[1].label,'会话摘要');assert.equal(f.ui.layers.value[2].label,'会话摘要');assert.equal(f.ui.typeLabel('memory'),'会话摘要');assert.equal(f.ui.metrics.value[0].value,null);assert.equal(f.ui.memoryStates.value.length,0);f.dispose();
});
