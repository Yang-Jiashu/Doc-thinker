import test from 'node:test';
import assert from 'node:assert/strict';
import {createMemoryObservationClient,normalizeMemoryObservation,sampleMemoryObservation} from '../src/services/memory-observability.js';

const clone=value=>structuredClone(value);
const fixture=()=>sampleMemoryObservation('#00001');
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
function harness(){
  let key='user-a',prefs={mode:'connected',connectionKind:'invite'},caps={enabled:true};
  const calls=[],handlers={capabilities:async()=>caps,request:async()=>fixture()};
  const client=createMemoryObservationClient({identity:()=>key,preferences:()=>prefs,capabilities:async()=>{calls.push('capabilities');return handlers.capabilities();},request:async(...args)=>{calls.push(args);return handlers.request(...args);}});
  return {client,calls,handlers,key:value=>{key=value;},prefs:value=>{prefs={...prefs,...value};},caps:value=>{caps=value;}};
}

test('observation GET is explicitly research enabled, encoded and contains no generation request',async()=>{
  const h=harness(),out=await h.client.load('#00001');
  assert.equal(out.session_id,'#00001');assert.equal(out.read_only,true);assert.equal(h.calls.length,2);
  assert.equal(h.calls[1][0],'/research/observability?session_id=%2300001&limit=20');assert.equal(h.calls[1][2],'GET');assert.equal(h.calls[1][1],undefined);
  assert.ok(h.calls[1][4].isCurrent());assert.equal(out.graph.causality_verified,false);
});
test('demo, pending invitation and disabled research cannot dispatch observability or fallback to legacy',async()=>{
  for(const prefs of [{mode:'demo'},{invitePending:true}]){
    const h=harness();h.prefs(prefs);await assert.rejects(h.client.load('#00001'),/未开放/);assert.equal(h.calls.length,0);
  }
  const h=harness();h.caps({enabled:false});await assert.rejects(h.client.load('#00001'),/未开放/);assert.deepEqual(h.calls,['capabilities']);
  const invited=harness();await assert.rejects(invited.client.load('#00001',{mode:'legacy'}),/未开放/);assert.equal(invited.calls.length,0);
});
test('private legacy adapter is an explicit separate path and has unknown counts, never invented zero',async()=>{
  const h=harness();h.prefs({connectionKind:'private'});
  const value=fixture();value.profile='legacy_session_memory';value.graph.kind='legacy_memory_projection';value.window.counts_scope='stored_active_session_summaries';value.availability.confirmed_memories=false;value.memories[0].revision=null;value.memories[0].confirmed_by_user=false;
  value.counts.memories={total:null,by_status:null,versions:null,source_links:null};h.handlers.request=async()=>value;
  const out=await h.client.load('#00001',{mode:'legacy',limit:40});assert.equal(out.counts.memories.total,null);assert.equal(out.counts.memories.by_status,null);
  assert.equal(h.calls.length,1);assert.equal(h.calls[0][0],'/memory/dashboard?session_id=%2300001&limit=40');
});
test('unknown, zero and bounded totals retain their distinct meaning without global validity inference',()=>{
  const value=fixture();value.counts.memories.by_status={active:1,invalidated:0,archived:null};value.window.limited.memories=true;
  value.window.states={memories:{active:0,invalidated:1}};value.graph.window_limited=true;
  const out=normalizeMemoryObservation(value,'#00001');assert.equal(out.counts.memories.by_status.invalidated,0);assert.equal(out.counts.memories.by_status.archived,null);
  assert.equal(out.window.states.memories.invalidated,1);assert.equal(out.window.counts_scope,'stored_session_rows');assert.equal(out.window.validity_scope,'returned_window');assert.equal(out.graph.window_limited,true);
});
test('late capability and data reads cannot cross identity, session, hidden-page or invalidation boundaries',async()=>{
  const h=harness(),wait=deferred();h.handlers.capabilities=()=>wait.promise;const pending=h.client.load('#00001');h.key('user-b');wait.resolve({enabled:true});await assert.rejects(pending,/已改变/);assert.equal(h.calls.length,1);
  for(const invalidate of ['identity','session','hide','invalidate']){
    const next=harness(),data=deferred();let active=true;next.handlers.request=()=>data.promise;
    const read=next.client.load('#00001',{isCurrent:()=>active});await new Promise(resolve=>setImmediate(resolve));
    if(invalidate==='identity')next.key('user-b');else if(invalidate==='invalidate')next.client.invalidate();else active=false;
    data.resolve(fixture());await assert.rejects(read,/已改变/);
  }
});
test('new session supersedes an older pending read even if the identity remains the same',async()=>{
  const h=harness(),wait=deferred();h.handlers.request=path=>path.includes('%2300001')?wait.promise:Promise.resolve({...fixture(),session_id:'#00002'});
  const a=h.client.load('#00001');await new Promise(resolve=>setImmediate(resolve));const b=await h.client.load('#00002');assert.equal(b.session_id,'#00002');wait.resolve(fixture());await assert.rejects(a,/已改变/);
});
test('malformed scopes, counts, graph proof claims and dangling graph edges fail closed',()=>{
  const cases=[value=>{value.session_id='#00002';},value=>{value.read_only=false;},value=>{value.counts.memories.total=-1;},value=>{value.counts.events.total='1';},value=>{value.graph.causality_verified=true;},value=>{value.graph.edges[0].target='foreign';},value=>{value.graph.edges[0].causality_verified=true;},value=>{value.graph.nodes.push(clone(value.graph.nodes[0]));},value=>{value.memories[0].session_id='#00002';},value=>{value.window.states='all valid';}];
  for(const mutate of cases){const value=fixture();mutate(value);assert.throws(()=>normalizeMemoryObservation(value,'#00001'),/格式/);}
});
test('projection strips redacted text and provenance; source IDs survive only on live nodes',()=>{
  const value=fixture();value.memories[0].redacted=true;value.memories[0].text='revoked secret';value.events[0].redacted=true;value.events[0].action='revoked event';
  value.graph.nodes[2].redacted=true;value.graph.nodes[2].label='revoked label';value.graph.nodes[2].description='revoked secret';value.graph.nodes[2].source_ids=['revoked-source'];value.graph.nodes[1].source_ids=['live-source'];
  const out=normalizeMemoryObservation(value,'#00001');assert.equal(out.memories[0].text,'');assert.deepEqual(out.memories[0].source_ids,[]);assert.equal(out.events[0].title,'');
  assert.equal(out.graph.nodes[2].label,'内容已撤销');assert.equal(out.graph.nodes[2].description,'');assert.deepEqual(out.graph.nodes[2].source_ids,[]);assert.deepEqual(out.graph.nodes[1].source_ids,['live-source']);
});
test('historical event occurrence dates are preserved without changing snapshot clock semantics',()=>{
  const value=fixture();value.events[0].occurred_from=-1000;value.events[0].action='A historic event';
  assert.equal(normalizeMemoryObservation(value,'#00001').events[0].occurred_at,-1000);value.sampled_at=-1000;assert.throws(()=>normalizeMemoryObservation(value,'#00001'),/格式/);
});
test('withdrawn relations do not require or reveal revoked endpoints and descriptions',()=>{
  const value=fixture();value.relations=[{id:'relation-withdrawn',status:'stale_endpoint',redacted:true}];
  const out=normalizeMemoryObservation(value,'#00001');assert.equal(out.relations[0].source,null);assert.equal(out.relations[0].target,null);assert.equal(out.relations[0].relation,'');assert.equal(out.relations[0].redacted,true);
});
test('graph keeps explicit conditions, source pointers and projection truncation flags without proof claims',()=>{
  const value=fixture();value.graph.nodes[0].label_truncated=true;value.graph.nodes[0].sources_limited=true;
  Object.assign(value.graph.edges[0],{conditions:['Only under this condition'],source_ids:['source-pointer'],sources_limited:true,source_scope:'endpoint_and_explicit_origins_not_entailment',description:'Source reference, not semantic proof'});
  const out=normalizeMemoryObservation(value,'#00001');assert.equal(out.graph.nodes[0].label_truncated,true);assert.equal(out.graph.nodes[0].sources_limited,true);
  assert.deepEqual(out.graph.edges[0].conditions,['Only under this condition']);assert.deepEqual(out.graph.edges[0].source_ids,['source-pointer']);assert.equal(out.graph.edges[0].sources_limited,true);assert.equal(out.graph.edges[0].causality_verified,false);
});
test('failed service reads have no retry, no sample substitution and no alternate endpoint probing',async()=>{
  const h=harness();h.handlers.request=async()=>{throw Object.assign(new Error('Unavailable'),{status:503});};
  await assert.rejects(h.client.load('#00001'),/Unavailable/);assert.equal(h.calls.length,2);assert.equal(h.calls.filter(Array.isArray).length,1);
});
test('range validation runs before network and fixtures are isolated fresh objects',async()=>{
  for(const options of [{limit:0},{limit:101},{limit:2.5},{mode:'anything'}]){const h=harness();await assert.rejects(h.client.load('#00001',options));assert.equal(h.calls.length,0);}
  const h=harness();await assert.rejects(h.client.load('#demo1'));assert.equal(h.calls.length,0);
  const a=fixture(),b=fixture();a.graph.nodes[0].label='changed';assert.notEqual(b.graph.nodes[0].label,'changed');assert.equal(b.counts.relations.total,0);
});
