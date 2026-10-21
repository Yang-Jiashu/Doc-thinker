<template>
  <section class="graph-canvas" aria-label="交互图谱">
    <header class="canvas-heading"><div><span class="canvas-eyebrow">CONNECTION ATLAS</span><h2>看见联系，沿着依据走。</h2></div><span class="canvas-inventory">{{ sceneCount }} 个节点 · {{ edgeCount }} 条关系</span></header>
    <p class="canvas-boundary">{{ caption }} 连线表示已记录的联系，不自动证明因果；布局变化不会新增关系。</p>
    <div ref="stage" class="graph-canvas-stage" data-testid="graph-stage"></div>
    <div class="canvas-tools" aria-label="画布控制">
      <div class="canvas-tool-group"><button role="button" aria-label="缩小图谱" :disabled="isControlDisabled('zoomOut')" :aria-disabled="isControlDisabled('zoomOut')" :tabindex="isControlDisabled('zoomOut') ? -1 : 0" @click="activateControl($event,'zoomOut')" @keydown="activateControl($event,'zoomOut')">−</button><output aria-label="图谱缩放比例">{{ scaleLabel }}</output><button role="button" aria-label="放大图谱" :disabled="isControlDisabled('zoomIn')" :aria-disabled="isControlDisabled('zoomIn')" :tabindex="isControlDisabled('zoomIn') ? -1 : 0" @click="activateControl($event,'zoomIn')" @keydown="activateControl($event,'zoomIn')">＋</button></div>
      <button role="button" :disabled="isControlDisabled('fit')" :aria-disabled="isControlDisabled('fit')" :tabindex="isControlDisabled('fit') ? -1 : 0" @click="activateControl($event,'fit')" @keydown="activateControl($event,'fit')">适应画布</button><button role="button" :disabled="isControlDisabled('focus')" :aria-disabled="isControlDisabled('focus')" :tabindex="isControlDisabled('focus') ? -1 : 0" @click="activateControl($event,'focus')" @keydown="activateControl($event,'focus')">聚焦当前</button><button role="button" :disabled="isControlDisabled('layout')" :aria-disabled="isControlDisabled('layout')" :tabindex="isControlDisabled('layout') ? -1 : 0" :aria-pressed="paused" @click="activateControl($event,'layout')" @keydown="activateControl($event,'layout')">{{ paused ? '继续布局' : '暂停布局' }}</button><button role="button" :disabled="isControlDisabled('rebuild')" :aria-disabled="isControlDisabled('rebuild')" :tabindex="isControlDisabled('rebuild') ? -1 : 0" @click="activateControl($event,'rebuild')" @keydown="activateControl($event,'rebuild')">重新排布</button>
    </div>
    <div class="canvas-legend"><span v-for="(kind,index) in kinds.slice(0,8)" :key="kind"><i :style="{background:colors[index % colors.length]}" aria-hidden="true"></i>{{ typeLabel(kind) }}</span><span v-if="kinds.length>8">另 {{ kinds.length-8 }} 类</span></div>
    <div v-if="selectedNode" class="canvas-selected" role="status" aria-live="polite"><span><strong>{{ selectedNode.label }}</strong><small>{{ typeLabel(selectedNode.type) }} · {{ selectedNode.degree }} 个相邻节点</small></span><button role="button" :disabled="isControlDisabled('inspect')" :aria-disabled="isControlDisabled('inspect')" :tabindex="isControlDisabled('inspect') ? -1 : 0" @click="activateControl($event,'inspect')" @keydown="activateControl($event,'inspect')">核对详情与原文 ↗</button></div>
    <p class="canvas-hint">拖动画布平移，拖动节点后固定位置，“重新排布”恢复；滚轮或双指缩放。键盘方向键平移，＋ / − 缩放，Home 复位。也可使用下方实体列表。</p>
    <p v-if="limited" class="canvas-window" role="status">当前显示有界窗口，不是全库导出。未显示的节点不代表不存在；可在列表或来源页继续核查。</p>
    <p v-if="renderError" class="canvas-window" role="status">此设备未能打开画布，请使用下方实体列表查看同一份关系和原文。</p>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { createScene, fitScene, hitNode, indexScene, stepScene, worldPoint, zoomAt, clamp } from '../services/graph-scene.js';

const props=defineProps({nodes:{type:Array,default:()=>[]},edges:{type:Array,default:()=>[]},selectedId:{type:String,default:''},highlightIds:{type:Array,default:()=>[]},active:{type:Boolean,default:true},resetKey:{type:String,default:''},caption:{type:String,default:'当前会话的实体与关系。'},windowLimited:{type:Boolean,default:false}});
const emit=defineEmits(['select','inspect']);
const stage=ref(null),scaleLabel=ref('100%'),paused=ref(false),reducedMotion=ref(false),renderError=ref(false),sceneCount=ref(0),edgeCount=ref(0),sceneRevision=ref(0);
const colors=['#35665a','#5c789d','#a17b48','#907294','#6d8451','#9a6860'];
const typeLabel=value=>({document:'资料',source:'原文',memory:'确认记忆',event:'事件'}[value] || value || '知识点');
let scene=createScene(),surface=null,ctx=null,observer=null,raf=0,width=800,height=430,view={x:400,y:215,k:1},running=false,disposed=false,interacted=false,media=null,labelBudget=new Set(),gridDirty=false;
const pointers=new Map();let drag=null,pinch=null;const listeners=[];
const kinds=computed(()=>[...new Set(props.nodes.slice(0,2000).map(n=>String(n.type||'知识点')))].sort());
const selectedNode=computed(()=>{sceneRevision.value;return scene.byId.get(props.selectedId)||null;});
const limited=computed(()=>props.windowLimited||props.nodes.length>2000||props.edges.length>8000);
function paintSoon(){if(!raf&&!disposed&&surface&&props.active)raf=requestAnimationFrame(frame);}
function frame(){raf=0;if(disposed||!props.active)return;if(running&&!paused.value&&!reducedMotion.value){running=stepScene(scene);gridDirty=false;if(!running&&!interacted)view=fitScene(scene,width,height);}else if(gridDirty){indexScene(scene);gridDirty=false;}draw();if(running&&!paused.value&&!reducedMotion.value)paintSoon();}
function draw(){
  if(!ctx)return;const ratio=Math.min(2,window.devicePixelRatio||1);ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  ctx.save();ctx.translate(view.x,view.y);ctx.scale(view.k,view.k);
  const route=new Set(props.highlightIds),emphasis=new Set(route);if(props.selectedId){emphasis.add(props.selectedId);for(const id of scene.adjacency.get(props.selectedId)||[])emphasis.add(id);}
  const labels=new Set();if(props.selectedId)labels.add(props.selectedId);for(const id of route){if(labels.size>=45)break;labels.add(id);}for(const id of scene.adjacency.get(props.selectedId)||[]){if(labels.size>=12)break;labels.add(id);}for(const id of labelBudget){if(labels.size>=45)break;labels.add(id);}
  const visible=n=>{const x=n.x*view.k+view.x,y=n.y*view.k+view.y;return x>=-60&&y>=-60&&x<=width+60&&y<=height+60;};
  for(const e of scene.edges){
    const focused=e.source.id===props.selectedId||e.target.id===props.selectedId||(route.has(e.source.id)&&route.has(e.target.id));
    ctx.globalAlpha=emphasis.size && !focused ? .22 : 1;ctx.strokeStyle=focused?'#54745a':'#c6cec3';ctx.lineWidth=(focused?1.6:.8)/view.k;ctx.setLineDash(e.is_discovered?[5/view.k,5/view.k]:[]);
    ctx.beginPath();ctx.moveTo(e.source.x,e.source.y);ctx.lineTo(e.target.x,e.target.y);ctx.stroke();
    if(e.directed&&focused){const a=Math.atan2(e.target.y-e.source.y,e.target.x-e.source.x),r=11/view.k,x=e.target.x-Math.cos(a)*r,y=e.target.y-Math.sin(a)*r;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-Math.cos(a-.45)*7/view.k,y-Math.sin(a-.45)*7/view.k);ctx.moveTo(x,y);ctx.lineTo(x-Math.cos(a+.45)*7/view.k,y-Math.sin(a+.45)*7/view.k);ctx.stroke();}
  }
  ctx.setLineDash([]);
  const typeRanks=new Map(kinds.value.map((k,i)=>[k,i]));
  for(const n of scene.nodes){
    if(!visible(n))continue;const selected=n.id===props.selectedId,highlight=emphasis.has(n.id),radius=clamp(4+Math.sqrt(n.degree)*1.1,4,10)/Math.sqrt(view.k);
    ctx.globalAlpha=emphasis.size && !highlight ? .26 : 1;ctx.fillStyle=colors[(typeRanks.get(n.type||'知识点')||0)%colors.length];
    if(selected){ctx.beginPath();ctx.arc(n.x,n.y,radius+8/view.k,0,Math.PI*2);ctx.fillStyle='#e1edc9';ctx.fill();ctx.fillStyle='#254d3f';}
    ctx.beginPath();ctx.arc(n.x,n.y,radius,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fffefa';ctx.lineWidth=1.7/view.k;ctx.stroke();
    if(labels.has(n.id)){
      const text=Array.from(n.label).slice(0,selected?48:18).join('')+(Array.from(n.label).length>(selected?48:18)?'…':'');
      ctx.font=`${selected?'600':'400'} ${selected?13:11}px -apple-system, BlinkMacSystemFont, sans-serif`;ctx.textAlign='center';ctx.textBaseline='top';
      // Labels stay in screen pixels; restore world transform after each label.
      ctx.save();ctx.translate(n.x,n.y+radius+5/view.k);ctx.scale(1/view.k,1/view.k);ctx.strokeStyle='#fbfcf8';ctx.lineWidth=4;ctx.strokeText(text,0,0);ctx.fillStyle=selected?'#203d35':'#4b6053';ctx.fillText(text,0,0);ctx.restore();
    }
  }
  ctx.restore();ctx.globalAlpha=1;scaleLabel.value=`${Math.round(view.k*100)}%`;
}
function resize(){
  if(!surface||!stage.value)return;const rect=stage.value.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);
  const ratio=Math.min(2,window.devicePixelRatio||1);surface.width=Math.round(width*ratio);surface.height=Math.round(height*ratio);surface.style.width=`${width}px`;surface.style.height=`${height}px`;
  if(!interacted)view=fitScene(scene,width,height);paintSoon();
}
function rebuild(){
  clearPointers();scene=createScene(props.nodes,props.edges);sceneCount.value=scene.nodes.length;edgeCount.value=scene.edges.length;sceneRevision.value++;
  labelBudget=new Set([...scene.nodes].sort((a,b)=>b.degree-a.degree).slice(0,scene.nodes.length<35?35:45).map(n=>n.id));
  indexScene(scene);gridDirty=false;view=fitScene(scene,width,height);interacted=false;running=scene.nodes.length>0;paintSoon();
}
function fit(){interacted=true;view=fitScene(scene,width,height);paintSoon();}
function zoom(factor){interacted=true;view=zoomAt(view,{x:width/2,y:height/2},factor);paintSoon();}
function focusSelected(){const n=scene.byId.get(props.selectedId);if(!n)return;interacted=true;view={k:Math.max(.8,view.k),x:0,y:0};view.x=width/2-n.x*view.k;view.y=height/2-n.y*view.k;paintSoon();}
function toggleLayout(){paused.value=!paused.value;if(!paused.value&&scene.ticks>=160)scene.ticks=12;running=!paused.value;paintSoon();}
// uni-button is a custom element on H5: supply native-equivalent semantics and
// one guarded activation path rather than relying on browser button defaults.
function isControlDisabled(action){return disposed||!props.active||(action==='focus'&&!props.selectedId)||(action==='layout'&&reducedMotion.value);}
function activateControl(event,action){
  if(event.type==='keydown'){
    if(event.ctrlKey||event.metaKey||event.altKey||event.isComposing||!['Enter',' ','Spacebar'].includes(event.key))return;
    event.preventDefault();if(event.repeat)return;
  }
  if(isControlDisabled(action))return;
  if(action==='zoomOut')zoom(.8);else if(action==='zoomIn')zoom(1.25);else if(action==='fit')fit();
  else if(action==='focus')focusSelected();else if(action==='layout')toggleLayout();else if(action==='rebuild')rebuild();else if(action==='inspect')emit('inspect');
}
function point(event){const rect=surface.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top};}
function pinchState(){const [a,b]=[...pointers.values()];return{x:(a.x+b.x)/2,y:(a.y+b.y)/2,d:Math.max(1,Math.hypot(a.x-b.x,a.y-b.y))};}
function pointerDown(event){
  if(!props.active||event.button>0||pointers.size>=2)return;event.preventDefault();surface.focus({preventScroll:true});interacted=true;
  const p=point(event);pointers.set(event.pointerId,p);try{surface.setPointerCapture(event.pointerId);}catch{/* A missing capture still permits explicit controls. */}
  if(pointers.size===2){if(drag?.node)drag.node.pinned=drag.wasPinned;drag=null;pinch=pinchState();return;}
  if(gridDirty){indexScene(scene);gridDirty=false;}const node=hitNode(scene,p,view);drag={id:event.pointerId,node,last:p,start:p,moved:false,wasPinned:node?.pinned===true};
  surface.style.cursor=node?'grabbing':'grabbing';
}
function pointerMove(event){
  if(!props.active||!pointers.has(event.pointerId))return;const p=point(event);pointers.set(event.pointerId,p);
  if(pointers.size===2){const next=pinchState();if(pinch){view=zoomAt(view,pinch,next.d/pinch.d);view.x+=next.x-pinch.x;view.y+=next.y-pinch.y;}pinch=next;paintSoon();return;}
  if(!drag)return;drag.moved ||= Math.hypot(p.x-drag.start.x,p.y-drag.start.y)>5;
  if(drag.node&&drag.moved){const world=worldPoint(p,view);drag.node.pinned=true;drag.node.x=world.x;drag.node.y=world.y;drag.node.vx=0;drag.node.vy=0;gridDirty=true;}else if(!drag.node){view.x+=p.x-drag.last.x;view.y+=p.y-drag.last.y;}
  drag.last=p;paintSoon();
}
function pointerUp(event){
  if(!pointers.has(event.pointerId))return;const selected=drag?.id===event.pointerId?drag.node:null;
  if(selected)selected.pinned=drag.moved&&event.type==='pointerup' ? true : drag.wasPinned;
  pointers.delete(event.pointerId);drag=null;pinch=null;
  try{if(surface.hasPointerCapture(event.pointerId))surface.releasePointerCapture(event.pointerId);}catch{/* Surface may have unmounted. */}
  if(pointers.size===1){const [id,p]=[...pointers][0];drag={id,node:null,last:p,start:p,moved:true};}
  surface.style.cursor='grab';if(selected&&event.type!=='pointercancel'&&event.type!=='lostpointercapture')emit('select',selected.id);paintSoon();
}
function clearPointers(){if(drag?.node)drag.node.pinned=drag.wasPinned;if(surface)for(const id of pointers.keys()){try{if(surface.hasPointerCapture(id))surface.releasePointerCapture(id);}catch{/* No capture after page teardown. */}}pointers.clear();drag=null;pinch=null;}
function wheel(event){if(!props.active)return;event.preventDefault();interacted=true;const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?height:1);view=zoomAt(view,point(event),Math.exp(-clamp(delta,-120,120)*.002));paintSoon();}
function keydown(event){if(!props.active||event.ctrlKey||event.metaKey||event.altKey||event.isComposing||event.keyCode===229)return;const key=event.key;if(['+','=','-','_','Home','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(key))event.preventDefault();else return;
  if(key==='Home')fit();else if(key==='+'||key==='=')zoom(1.2);else if(key==='-'||key==='_')zoom(1/1.2);else{interacted=true;view.x+=key==='ArrowLeft'?40:key==='ArrowRight'?-40:0;view.y+=key==='ArrowUp'?40:key==='ArrowDown'?-40:0;paintSoon();}}
function listen(target,name,callback,options){target.addEventListener(name,callback,options);listeners.push(()=>target.removeEventListener(name,callback,options));}
onMounted(async()=>{
  try{
    await nextTick();if(disposed)return;
    surface=document.createElement('canvas');surface.setAttribute('aria-label','交互图谱画布；可拖动平移、双指缩放，或使用下方实体列表');surface.setAttribute('role','img');surface.setAttribute('tabindex','0');surface.setAttribute('data-testid','graph-surface');surface.style.touchAction='none';surface.style.cursor='grab';surface.style.display='block';
    stage.value.appendChild(surface);ctx=surface.getContext('2d');if(!ctx)throw new Error('canvas unavailable');
    media=window.matchMedia?.('(prefers-reduced-motion: reduce)');reducedMotion.value=media?.matches===true;
    if(media){const update=()=>{reducedMotion.value=media.matches;if(reducedMotion.value)running=false;paintSoon();};media.addEventListener?.('change',update);listeners.push(()=>media?.removeEventListener?.('change',update));}
    for(const [name,fn] of [['pointerdown',pointerDown],['pointermove',pointerMove],['pointerup',pointerUp],['pointercancel',pointerUp],['lostpointercapture',pointerUp],['keydown',keydown]])listen(surface,name,fn);
    listen(surface,'wheel',wheel,{passive:false});
    if(typeof ResizeObserver!=='undefined'){observer=new ResizeObserver(resize);observer.observe(stage.value);}else listen(window,'resize',resize);
    rebuild();resize();
  }catch{renderError.value=true;}
});
watch(()=>[props.nodes,props.edges,props.resetKey],rebuild);
watch(()=>[props.selectedId,props.highlightIds],()=>{sceneRevision.value++;paintSoon();});
watch(()=>props.active,value=>{if(!value){clearPointers();if(raf)cancelAnimationFrame(raf);raf=0;}else paintSoon();});
onBeforeUnmount(()=>{disposed=true;clearPointers();if(raf)cancelAnimationFrame(raf);observer?.disconnect();for(const stop of listeners)stop();surface?.remove();surface=null;ctx=null;scene=createScene();});
</script>

<style scoped>
.graph-canvas{overflow:hidden;margin:14px 0 26px;border:1px solid #dfe4d8;border-radius:24px;background:#fffefa;color:#253f32}.canvas-heading{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:23px 24px 0}.canvas-eyebrow{font:10px/1.5 ui-monospace,monospace;letter-spacing:2px;color:#789071}.canvas-heading h2{margin:5px 0 0;font-size:23px;font-weight:600;letter-spacing:-.6px}.canvas-inventory{font-size:11px;color:#74836c;white-space:nowrap}.canvas-boundary,.canvas-hint,.canvas-window{margin:10px 24px 0;font-size:11px;line-height:1.8;color:#72816d}.graph-canvas-stage{height:430px;width:100%;overflow:hidden;background:radial-gradient(ellipse at 55% 52%,#eff3e580 0%,#fffefa00 70%)}.graph-canvas-stage :deep(canvas:focus-visible){outline:2px solid #819c57;outline-offset:-4px}.canvas-tools{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 22px;padding:14px 0;border-top:1px solid #e9ecdf}.canvas-tools button,.canvas-selected button{border:1px solid #e0e5d6;border-radius:10px;background:#f6f8f0;color:#435c3b;min-height:38px;padding:8px 12px;font-family:inherit;font-size:12px;line-height:1.5;cursor:pointer}.canvas-tools button:disabled,.canvas-selected button:disabled,.canvas-tools button[aria-disabled="true"],.canvas-selected button[aria-disabled="true"]{opacity:.45;cursor:default}.canvas-tools button:focus-visible,.canvas-selected button:focus-visible{outline:2px solid #819c57;outline-offset:2px}.canvas-tool-group{display:flex;align-items:center;gap:5px}.canvas-tool-group output{min-width:42px;text-align:center;font-size:11px;color:#697861}.canvas-tool-group button{font-size:17px;min-width:38px;padding:3px 8px}.canvas-legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin:0 24px;color:#77866c;font-size:10px}.canvas-legend span{display:flex;align-items:center;gap:6px}.canvas-legend i{width:7px;height:7px;border-radius:50%}.canvas-selected{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:15px 23px 0;padding:12px 14px;background:#edf3e4;border-radius:13px}.canvas-selected span{min-width:0}.canvas-selected strong{display:block;font-size:14px;overflow-wrap:anywhere}.canvas-selected small{display:block;color:#748566;font-size:10px;margin-top:4px}.canvas-selected button{flex-shrink:0;background:#fffefa}.canvas-hint{margin-bottom:19px;font-size:10px}.canvas-window{margin-top:0;margin-bottom:20px;color:#92743c}@media(max-width:600px){.canvas-heading{padding:18px 16px 0;align-items:flex-start}.canvas-heading h2{font-size:20px}.canvas-inventory{white-space:normal;font-size:10px;text-align:right}.canvas-boundary,.canvas-hint,.canvas-window{margin-left:16px;margin-right:16px}.graph-canvas-stage{height:360px}.canvas-tools{margin:0 14px;gap:6px}.canvas-tools button{font-size:11px;min-height:44px;padding:8px 10px}.canvas-legend{margin:0 16px}.canvas-selected{margin-left:14px;margin-right:14px}.canvas-selected button{font-size:11px;min-height:44px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
