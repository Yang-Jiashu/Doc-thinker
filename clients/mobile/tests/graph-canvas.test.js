// Execute the real SFC setup and registered handlers with deterministic DOM /
// RAF doubles. These are interaction contracts, not phone or camera acceptance.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as physics from '../src/services/graph-scene.js';
import {createGraphPainter,GRAPH_COLORS} from '../src/services/graph-paint.js';

const component = readFileSync(new URL('../src/components/GraphCanvas.vue',import.meta.url),'utf8');
const setup = component.split('<script setup>')[1].split('</script>')[0].replace(/^import .*;\s*$/gm,'');
const almost = (a,b) => assert.ok(Math.abs(a-b)<1e-7, `${a} differs from ${b}`);

class Target {
  constructor(){this.listeners=new Map();}
  addEventListener(name,callback){if(!this.listeners.has(name))this.listeners.set(name,new Set());this.listeners.get(name).add(callback);}
  removeEventListener(name,callback){this.listeners.get(name)?.delete(callback);}
  dispatch(name,values={}){
    const event={type:name,button:0,pointerId:1,clientX:0,clientY:0,defaultPrevented:false,
      preventDefault(){this.defaultPrevented=true;},...values};
    for(const callback of [...(this.listeners.get(name)||[])])callback(event);
    return event;
  }
  listenerCount(){return [...this.listeners.values()].reduce((total,value)=>total+value.size,0);}
}

function fixture({reduced=false,resizeObserver=true,canvas=true,nextTick=null,props:overrides={}}={}){
  const props={nodes:[{id:'a',label:'个人安排',type:'event'},{id:'b',label:'原文依据',type:'source'},
    {id:'c',label:'确认偏好',type:'memory'}],edges:[{source:'a',target:'b'},{source:'c',target:'b'}],
    selectedId:'',highlightIds:[],active:true,resetKey:'#00001',caption:'离线测试',windowLimited:false,...overrides};
  const emissions=[],mounted=[],unmount=[],watchers=[],frames=new Map(),cancelled=[];
  const media=new Target(),window=new Target(),surface=new Target(),children=[];
  let frameId=0,draws=0,observer=null,removed=false,created=0;
  media.matches=reduced; window.devicePixelRatio=3; window.matchMedia=()=>media;
  const context2d=Object.fromEntries(['setTransform','clearRect','save','translate','scale','setLineDash',
    'beginPath','moveTo','lineTo','quadraticCurveTo','closePath','rect','clip','stroke','arc','fill','restore','strokeText','fillText'].map(name=>[name,()=>{draws++;}]));
  context2d.measureText=text=>({width:Array.from(text).length*12});
  const rect={left:11,top:17,width:800,height:430};
  const captures=new Set();
  Object.assign(surface,{style:{},attributes:{},focusCalls:0,getBoundingClientRect:()=>({...rect}),
    setAttribute(name,value){this.attributes[name]=value;},focus(){this.focusCalls++;},
    getContext:()=>canvas?context2d:null,setPointerCapture:id=>captures.add(id),
    hasPointerCapture:id=>captures.has(id),releasePointerCapture:id=>captures.delete(id),
    remove(){removed=true;}});
  const stage={getBoundingClientRect:()=>({...rect}),appendChild(node){children.push(node);}};
  const copy=value=>Array.isArray(value)?Array.from(value):value;
  const same=(a,b)=>Array.isArray(a)&&Array.isArray(b)?a.length===b.length&&a.every((v,i)=>Object.is(v,b[i])):Object.is(a,b);
  const context={...physics,createGraphPainter,GRAPH_COLORS,console,Map,Set,Math,Array,Number,String,window,
    document:{createElement(name){assert.equal(name,'canvas');created++;return surface;}},
    computed:fn=>({get value(){return fn();}}),ref:value=>({value}),
    nextTick:nextTick||(()=>Promise.resolve()),
    defineProps:()=>props,defineEmits:()=> (...values)=>emissions.push(values),
    onMounted:fn=>mounted.push(fn),onBeforeUnmount:fn=>unmount.push(fn),
    watch(getter,callback){watchers.push({getter,callback,value:copy(getter())});},
    requestAnimationFrame(callback){const id=++frameId;frames.set(id,callback);return id;},
    cancelAnimationFrame(id){cancelled.push(id);frames.delete(id);},
  };
  if(resizeObserver)context.ResizeObserver=class {
    constructor(callback){this.callback=callback;this.disconnected=false;observer=this;}
    observe(value){assert.equal(value,stage);}
    disconnect(){this.disconnected=true;}
  };
  vm.createContext(context);
  vm.runInContext(setup+`;globalThis.canvasContract={stage,zoom,fit,focusSelected,toggleLayout,rebuild,isControlDisabled,activateControl,
    state:()=>({scene,view:{...view},pointers:pointers.size,paused:paused.value,
      renderError:renderError.value,reducedMotion:reducedMotion.value,disposed}),};`,context,
    {filename:'GraphCanvas.real-setup.js'});
  const api=context.canvasContract;
  api.stage.value=stage;
  function update(values){
    Object.assign(props,values);
    for(const watcher of [...watchers]){
      const next=copy(watcher.getter());
      if(!same(next,watcher.value)){const old=watcher.value;watcher.value=next;watcher.callback(next,old);}
    }
  }
  function frame(){const next=frames.entries().next().value;if(!next)return false;frames.delete(next[0]);next[1](0);return true;}
  function settle(limit=200){let count=0;while(frames.size&&count<limit){frame();count++;}assert.equal(frames.size,0,'animation exceeded its finite RAF budget');return count;}
  function nodePoint(id='a'){
    const state=api.state(),node=state.scene.byId.get(id);
    return{clientX:rect.left+node.x*state.view.k+state.view.x,
      clientY:rect.top+node.y*state.view.k+state.view.y};
  }
  return{api,props,emissions,media,window,surface,captures,children,frames,cancelled,update,frame,settle,nodePoint,
    mount:async()=>{for(const callback of mounted)await callback();},
    destroy(){for(const callback of unmount)callback();watchers.length=0;},
    get observer(){return observer;},get removed(){return removed;},get created(){return created;},get draws(){return draws;}};
}

test('SFC mounts a real canvas target, caps pixel ratio and installs explicit interaction handlers',async()=>{
  const f=fixture();await f.mount();
  assert.equal(f.created,1);assert.equal(f.children[0],f.surface);
  assert.equal(f.surface.width,1600);assert.equal(f.surface.height,860);
  assert.equal(f.surface.style.touchAction,'none');assert.equal(f.surface.attributes.tabindex,'0');
  for(const name of ['pointerdown','pointermove','pointerup','pointercancel','lostpointercapture','wheel','keydown'])assert.equal(f.surface.listeners.get(name)?.size,1,name);
  assert.equal(f.frames.size,1);f.frame();assert.ok(f.draws>0);f.destroy();
});

test('a click and sub-threshold movement select exactly once without pinning a layout node',async()=>{
  const f=fixture();await f.mount();f.settle();const p=f.nodePoint(),node=f.api.state().scene.byId.get('a');
  f.surface.dispatch('pointerdown',{pointerId:1,...p});
  assert.equal(node.pinned,false);
  f.surface.dispatch('pointermove',{pointerId:1,clientX:p.clientX+2,clientY:p.clientY+1});
  assert.equal(node.pinned,false);
  f.surface.dispatch('pointerup',{pointerId:1,clientX:p.clientX+2,clientY:p.clientY+1});
  assert.equal(node.pinned,false);assert.deepEqual(f.emissions,[['select','a']]);
  assert.equal(f.captures.size,0);f.destroy();
});

test('continuous node drag updates position, coalesces RAF and retains only an intentional pin',async()=>{
  const f=fixture();await f.mount();f.settle();const p=f.nodePoint(),node=f.api.state().scene.byId.get('a');
  const original={x:node.x,y:node.y},edges=f.api.state().scene.edges.length;
  f.surface.dispatch('pointerdown',{pointerId:3,...p});
  for(let i=1;i<=50;i++)f.surface.dispatch('pointermove',{pointerId:3,clientX:p.clientX+i,clientY:p.clientY+20});
  assert.equal(node.pinned,true);assert.ok(node.x!==original.x||node.y!==original.y);
  assert.equal(f.frames.size,1,'many pointer events must not queue many animation frames');
  f.surface.dispatch('pointerup',{pointerId:3,clientX:p.clientX+50,clientY:p.clientY+20});
  assert.equal(node.pinned,true);assert.equal(f.api.state().scene.edges.length,edges);
  assert.deepEqual(f.emissions,[['select','a']]);f.destroy();
});

test('continuous background dragging pans without changing graph topology or selecting nodes',async()=>{
  const f=fixture();await f.mount();f.settle();const before=f.api.state().view;
  f.surface.dispatch('pointerdown',{pointerId:1,clientX:12,clientY:18});
  for(let i=1;i<=30;i++)f.surface.dispatch('pointermove',{pointerId:1,clientX:12+i*2,clientY:18+i});
  f.surface.dispatch('pointerup',{pointerId:1,clientX:72,clientY:48});
  almost(f.api.state().view.x,before.x+60);almost(f.api.state().view.y,before.y+30);
  assert.equal(f.api.state().scene.edges.length,2);assert.deepEqual(f.emissions,[]);f.destroy();
});

test('second-finger pinch restores a temporary node pin, ignores a third contact and keeps panning continuous',async()=>{
  const f=fixture();await f.mount();f.settle();const p=f.nodePoint(),node=f.api.state().scene.byId.get('a');
  f.surface.dispatch('pointerdown',{pointerId:1,...p});
  f.surface.dispatch('pointermove',{pointerId:1,clientX:p.clientX+12,clientY:p.clientY});
  assert.equal(node.pinned,true);
  f.surface.dispatch('pointerdown',{pointerId:2,clientX:p.clientX+150,clientY:p.clientY});
  assert.equal(node.pinned,false,'pinch must not permanently pin the first touched node');
  const before=f.api.state().view;
  f.surface.dispatch('pointerdown',{pointerId:3,clientX:100,clientY:100});
  assert.equal(f.api.state().pointers,2);assert.equal(f.captures.size,2);
  for(let i=1;i<=20;i++)f.surface.dispatch('pointermove',{pointerId:2,clientX:p.clientX+150+i*4,clientY:p.clientY+i});
  assert.ok(f.api.state().view.k>before.k);assert.equal(f.frames.size,1);
  f.surface.dispatch('pointerup',{pointerId:2});
  const oneFinger=f.api.state().view;
  f.surface.dispatch('pointermove',{pointerId:1,clientX:p.clientX+20,clientY:p.clientY+5});
  assert.ok(f.api.state().view.x!==oneFinger.x||f.api.state().view.y!==oneFinger.y);
  f.surface.dispatch('pointerup',{pointerId:1});
  assert.equal(node.pinned,false);assert.equal(f.captures.size,0);assert.deepEqual(f.emissions,[]);f.destroy();
});

for(const cancellation of ['pointercancel','lostpointercapture']){
  for(const initiallyPinned of [false,true]){
    test(`${cancellation} restores the prior pin and does not accidentally select (${initiallyPinned})`,async()=>{
      const f=fixture();await f.mount();f.settle();const p=f.nodePoint(),node=f.api.state().scene.byId.get('a');node.pinned=initiallyPinned;
      f.surface.dispatch('pointerdown',{pointerId:8,...p});
      f.surface.dispatch('pointermove',{pointerId:8,clientX:p.clientX+25,clientY:p.clientY+15});
      f.surface.dispatch(cancellation,{pointerId:8});
      assert.equal(node.pinned,initiallyPinned);assert.equal(f.api.state().pointers,0);
      assert.equal(f.captures.size,0);assert.deepEqual(f.emissions,[]);f.destroy();
    });
  }
}

test('inactive page cancels captures and RAF, rejects late input, and resumes only after activation',async()=>{
  const f=fixture();await f.mount();f.settle();const p=f.nodePoint(),node=f.api.state().scene.byId.get('a');
  f.surface.dispatch('pointerdown',{pointerId:1,...p});
  f.surface.dispatch('pointermove',{pointerId:1,clientX:p.clientX+30,clientY:p.clientY});
  assert.equal(node.pinned,true);assert.equal(f.frames.size,1);
  f.update({active:false});
  assert.equal(f.captures.size,0);assert.equal(f.api.state().pointers,0);assert.equal(node.pinned,false);
  assert.equal(f.frames.size,0);assert.ok(f.cancelled.length>0);const view=f.api.state().view;
  f.surface.dispatch('pointermove',{pointerId:1,clientX:2000,clientY:2000});
  f.surface.dispatch('wheel',{deltaY:-120,clientX:200,clientY:200});
  f.surface.dispatch('keydown',{key:'ArrowRight'});
  f.surface.dispatch('pointerup',{pointerId:1});
  assert.deepEqual(f.api.state().view,view);assert.deepEqual(f.emissions,[]);
  f.update({active:true});assert.equal(f.frames.size,1);f.destroy();
});

test('unmount cancels animation, captures, listeners and observer without late selection',async()=>{
  const f=fixture();await f.mount();const p=f.nodePoint();
  f.surface.dispatch('pointerdown',{pointerId:1,...p});f.surface.dispatch('pointermove',{pointerId:1,clientX:p.clientX+15,clientY:p.clientY});
  const previous=f.api.state().scene.byId.get('a');f.destroy();
  assert.equal(previous.pinned,false);assert.equal(f.frames.size,0);assert.equal(f.captures.size,0);
  assert.equal(f.surface.listenerCount(),0);assert.equal(f.media.listenerCount(),0);
  assert.equal(f.observer.disconnected,true);assert.equal(f.removed,true);assert.equal(f.api.state().disposed,true);
  assert.equal(f.api.state().scene.nodes.length,0);
  f.surface.dispatch('pointerup',{pointerId:1});assert.deepEqual(f.emissions,[]);
});

test('unmount before the mount tick does not create a stale canvas',async()=>{
  let release;const tick=new Promise(resolve=>{release=resolve;});
  const f=fixture({nextTick:()=>tick}),mounting=f.mount();f.destroy();release();await mounting;
  assert.equal(f.created,0);assert.equal(f.children.length,0);assert.equal(f.frames.size,0);
});

test('fallback resize listener is removed when ResizeObserver is unavailable',async()=>{
  const f=fixture({resizeObserver:false});await f.mount();
  assert.equal(f.window.listeners.get('resize')?.size,1);f.destroy();assert.equal(f.window.listenerCount(),0);
});

test('wheel and keyboard transform the view without mutating nodes or overflowing zoom',async()=>{
  const f=fixture();await f.mount();f.settle();const before=f.api.state().view;
  const event=f.surface.dispatch('wheel',{deltaY:-120,clientX:411,clientY:232});
  assert.equal(event.defaultPrevented,true);assert.ok(f.api.state().view.k>before.k);
  for(let i=0;i<80;i++)f.surface.dispatch('wheel',{deltaY:-120,clientX:411,clientY:232});
  assert.equal(f.api.state().view.k,physics.SCENE_LIMITS.scaleMax);assert.equal(f.frames.size,1);
  const moved=f.api.state().view;f.surface.dispatch('keydown',{key:'ArrowRight'});
  almost(f.api.state().view.x,moved.x-40);
  const minus=f.surface.dispatch('keydown',{key:'-'});assert.equal(minus.defaultPrevented,true);
  assert.ok(f.api.state().view.k<physics.SCENE_LIMITS.scaleMax);
  f.surface.dispatch('keydown',{key:'Home'});
  const fit=physics.fitScene(f.api.state().scene,800,430);
  almost(f.api.state().view.x,fit.x);almost(f.api.state().view.y,fit.y);almost(f.api.state().view.k,fit.k);
  assert.equal(f.api.state().scene.nodes.length,3);assert.equal(f.api.state().scene.edges.length,2);f.destroy();
});

test('pause and reduced motion stop animation scheduling; layout ticks remain bounded',async()=>{
  const f=fixture();await f.mount();f.api.toggleLayout();
  assert.equal(f.api.state().paused,true);f.frame();assert.equal(f.frames.size,0);
  f.api.toggleLayout();assert.equal(f.frames.size,1);f.settle();
  assert.ok(f.api.state().scene.ticks<=physics.SCENE_LIMITS.ticks);f.destroy();
  const reduced=fixture({reduced:true});await reduced.mount();
  const ticks=reduced.api.state().scene.ticks;reduced.settle();assert.equal(reduced.api.state().scene.ticks,ticks);
  assert.equal(reduced.api.state().reducedMotion,true);reduced.destroy();
});

test('selected IDs redraw without rebuilding stable geometry; source reset cancels an active drag',async()=>{
  const f=fixture();await f.mount();f.settle();const original=f.api.state().scene;
  f.update({selectedId:'a'});assert.equal(f.api.state().scene,original);assert.equal(f.frames.size,1);
  const p=f.nodePoint();f.surface.dispatch('pointerdown',{pointerId:1,...p});
  f.surface.dispatch('pointermove',{pointerId:1,clientX:p.clientX+20,clientY:p.clientY});
  f.update({nodes:[{id:'new',label:'当前会话新记录'}],edges:[],resetKey:'#00002'});
  assert.equal(f.api.state().pointers,0);assert.equal(f.captures.size,0);
  assert.equal(f.api.state().scene.nodes.length,1);assert.equal(f.api.state().scene.nodes[0].id,'new');
  f.surface.dispatch('pointerup',{pointerId:1});assert.deepEqual(f.emissions,[]);f.destroy();
});

test('unsupported canvas reports the explicit list fallback rather than installing partial input handlers',async()=>{
  const f=fixture({canvas:false});await f.mount();
  assert.equal(f.api.state().renderError,true);assert.equal(f.surface.listenerCount(),0);
  assert.equal(f.frames.size,0);f.destroy();assert.equal(f.removed,true);
});

const controls=['zoomOut','zoomIn','fit','focus','layout','rebuild','inspect'];
const activation=(key='Enter',overrides={})=>({type:'keydown',key,defaultPrevented:false,
  preventDefault(){this.defaultPrevented=true;},...overrides});

test('all seven uni-button controls expose matching role, tab order, disabled state and activation handlers',()=>{
  const tags=[...component.split('</template>')[0].matchAll(/<button\b[^>]*>/g)].map(match=>match[0]);
  assert.equal(tags.length,7);
  for(const action of controls){
    const tag=tags.find(value=>value.includes(`@click="activateControl($event,'${action}')"`));
    assert.ok(tag,`${action} uses the guarded click path`);
    assert.ok(tag.includes('role="button"'));
    assert.ok(tag.includes(`:disabled="isControlDisabled('${action}')"`));
    assert.ok(tag.includes(`:aria-disabled="isControlDisabled('${action}')"`));
    assert.ok(tag.includes(`:tabindex="isControlDisabled('${action}') ? -1 : 0"`));
    assert.ok(tag.includes(`@keydown="activateControl($event,'${action}')"`));
  }
  assert.doesNotMatch(component,/font:\s*12px\/1\.5\s+inherit/);
});

for(const key of ['Enter',' ','Spacebar']){
  test(`${JSON.stringify(key)} activates the actual control handler once and prevents native duplicate activation`,async()=>{
    const f=fixture({props:{selectedId:'a'}});await f.mount();f.settle();
    let previous=f.api.state().view.k;const event=activation(key);
    f.api.activateControl(event,'zoomIn');assert.equal(event.defaultPrevented,true);
    almost(f.api.state().view.k,previous*1.25);
    previous=f.api.state().view.k;f.api.activateControl(activation(key),'zoomOut');almost(f.api.state().view.k,previous*.8);
    f.api.activateControl(activation(key),'focus');const focused=f.api.state();
    almost(focused.scene.byId.get('a').x*focused.view.k+focused.view.x,400);
    almost(focused.scene.byId.get('a').y*focused.view.k+focused.view.y,215);
    f.api.activateControl(activation(key),'fit');const fit=physics.fitScene(f.api.state().scene,800,430);
    almost(f.api.state().view.k,fit.k);
    f.api.activateControl(activation(key),'layout');assert.equal(f.api.state().paused,true);
    f.api.activateControl(activation(key,{repeat:true}),'layout');assert.equal(f.api.state().paused,true);
    const old=f.api.state().scene;f.api.activateControl(activation(key),'rebuild');assert.notEqual(f.api.state().scene,old);
    f.api.activateControl(activation(key),'inspect');assert.deepEqual(f.emissions,[['inspect']]);f.destroy();
  });
}

test('disabled, modified, composing and unrelated control activations cannot mutate view or emit events',async()=>{
  const f=fixture();await f.mount();f.settle();assert.equal(f.api.isControlDisabled('focus'),true);
  const before=f.api.state().view,old=f.api.state().scene;
  f.api.activateControl(activation(),'focus');assert.deepEqual(f.api.state().view,before);
  for(const event of [activation('Escape'),activation('Enter',{ctrlKey:true}),activation('Enter',{metaKey:true}),
    activation('Enter',{altKey:true}),activation('Enter',{isComposing:true})]){
    f.api.activateControl(event,'zoomIn');assert.equal(event.defaultPrevented,false);
  }
  f.update({active:false});
  for(const action of controls){
    assert.equal(f.api.isControlDisabled(action),true);
    f.api.activateControl(activation(),action);f.api.activateControl({type:'click'},action);
  }
  assert.deepEqual(f.api.state().view,before);assert.equal(f.api.state().scene,old);
  assert.equal(f.api.state().paused,false);assert.deepEqual(f.emissions,[]);
  f.update({active:true});f.api.activateControl({type:'click'},'zoomIn');assert.ok(f.api.state().view.k>before.k);
  f.destroy();f.api.activateControl({type:'click'},'inspect');assert.deepEqual(f.emissions,[]);
  const reduced=fixture({reduced:true});await reduced.mount();assert.equal(reduced.api.isControlDisabled('layout'),true);
  reduced.api.activateControl(activation(),'layout');assert.equal(reduced.api.state().paused,false);reduced.destroy();
});
