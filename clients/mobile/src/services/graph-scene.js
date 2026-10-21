// Bounded, deterministic display physics. Positions never create knowledge edges.
// One grid aggregation + one edge pass per tick; no all-pairs repulsion or model.
export const SCENE_LIMITS = Object.freeze({nodes:2000,edges:8000,cell:100,scaleMin:0.05,scaleMax:5,ticks:160});
const finite = (n, fallback=0) => typeof n === 'number' && Number.isFinite(n) ? n : fallback;
export const clamp = (n,lo,hi) => Math.max(lo,Math.min(hi,finite(n,lo)));
const label = (s,max=160) => typeof s === 'string' ? s.slice(0,max).replace(/[\u0000-\u001f\u007f\u202a-\u202e\u2066-\u2069]/g,'') : '';
const identifier = s => typeof s === 'string' && s && s.length <= 256 && label(s,256) === s ? s : '';
const endpoint = s => identifier(s && typeof s === 'object' ? s.id : s);
function hash(s) { let h=2166136261; for (const c of s) {h^=c.codePointAt(0);h=Math.imul(h,16777619);} return h>>>0; }
export const worldPoint = (point,view) => ({x:(finite(point.x)-finite(view.x))/clamp(view.k,SCENE_LIMITS.scaleMin,SCENE_LIMITS.scaleMax),y:(finite(point.y)-finite(view.y))/clamp(view.k,SCENE_LIMITS.scaleMin,SCENE_LIMITS.scaleMax)});
export function zoomAt(view,point,factor) {
  const k=clamp(view.k,SCENE_LIMITS.scaleMin,SCENE_LIMITS.scaleMax), next=clamp(k*finite(factor,1),SCENE_LIMITS.scaleMin,SCENE_LIMITS.scaleMax);
  const world=worldPoint(point,{...view,k});
  return {k:next,x:finite(point.x)-world.x*next,y:finite(point.y)-world.y*next};
}
export function fitScene(scene,width,height) {
  width=Math.max(1,finite(width,800));height=Math.max(1,finite(height,420));
  if (!scene.nodes.length) return {x:width/2,y:height/2,k:1};
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const n of scene.nodes){minX=Math.min(minX,n.x-25);maxX=Math.max(maxX,n.x+25);minY=Math.min(minY,n.y-25);maxY=Math.max(maxY,n.y+25);}
  const k=clamp(Math.min((width-72)/Math.max(100,maxX-minX),(height-72)/Math.max(100,maxY-minY),1.8),SCENE_LIMITS.scaleMin,SCENE_LIMITS.scaleMax);
  return {k,x:width/2-(minX+maxX)/2*k,y:height/2-(minY+maxY)/2*k};
}
export function createScene(rawNodes=[],rawEdges=[]) {
  const byId=new Map(),links=[],adjacency=new Map();
  const input=Array.isArray(rawNodes)?rawNodes:[],edges=Array.isArray(rawEdges)?rawEdges:[];
  for(const item of input.slice(0,SCENE_LIMITS.nodes)){
    if (!item || typeof item!=='object' || !identifier(item.id) || byId.has(item.id)) continue;
    const n={id:item.id,label:label(item.label || item.name || item.id),type:label(item.type,80),
      status:label(item.status,40),is_discovered:item.is_discovered===true,x:0,y:0,vx:0,vy:0,degree:0,pinned:false};
    byId.set(n.id,n);adjacency.set(n.id,new Set());
  }
  // Sorted identities make layout stable across response ordering. Input stays untouched.
  const nodes=[...byId.values()].sort((a,b)=>a.id<b.id?-1:a.id>b.id?1:0),seen=new Set();
  for(const item of edges.slice(0,SCENE_LIMITS.edges)){
    if(!item||typeof item!=='object')continue;
    const source=endpoint(item.source),target=endpoint(item.target);
    if(!source||!target||source===target||!byId.has(source)||!byId.has(target))continue;
    const kind=label(item.relation || item.type || item.description,100),key=JSON.stringify([source,target,kind]);
    if(seen.has(key))continue;seen.add(key);
    links.push({source:byId.get(source),target:byId.get(target),description:label(item.description || item.relation || item.type,240),is_discovered:item.is_discovered===true,directed:item.directed===true});
    adjacency.get(source).add(target);adjacency.get(target).add(source);
  }
  links.sort((a,b)=>a.source.id.localeCompare(b.source.id)||a.target.id.localeCompare(b.target.id)||a.description.localeCompare(b.description));
  const types=[...new Set(nodes.map(n=>n.type || '知识点'))].sort();
  const typeRanks=new Map(types.map((t,i)=>[t,i])),counts=new Map();
  const orbit=types.length>1?Math.min(360,Math.sqrt(nodes.length)*22):0;
  for(const n of nodes){
    n.degree=adjacency.get(n.id).size;
    const type=n.type||'知识点',i=counts.get(type)||0;counts.set(type,i+1);
    const seed=hash(n.id),theta=i*2.399963+seed%100/1000,ring=24+Math.sqrt(i)*26;
    const group=typeRanks.get(type)/Math.max(1,types.length)*Math.PI*2;
    n.x=Math.cos(group)*orbit+Math.cos(theta)*ring;n.y=Math.sin(group)*orbit+Math.sin(theta)*ring;
  }
  return {nodes,edges:links,byId,adjacency,ticks:0,grid:new Map(),limited:input.length>SCENE_LIMITS.nodes||edges.length>SCENE_LIMITS.edges};
}
const cellKey = (x,y) => `${x},${y}`;
export function indexScene(scene) {
  const cells=new Map();
  for(const n of scene.nodes){
    const x=Math.floor(n.x/SCENE_LIMITS.cell),y=Math.floor(n.y/SCENE_LIMITS.cell),key=cellKey(x,y);
    let cell=cells.get(key);if(!cell){cell={x,y,sx:0,sy:0,n:0,nodes:[]};cells.set(key,cell);}
    cell.sx+=n.x;cell.sy+=n.y;cell.n++;cell.nodes.push(n);
  }
  scene.grid=cells;return cells;
}
export function stepScene(scene) {
  if(scene.ticks>=SCENE_LIMITS.ticks)return false;
  const cells=indexScene(scene),size=SCENE_LIMITS.cell;
  for(const n of scene.nodes){
    const cx=Math.floor(n.x/size),cy=Math.floor(n.y/size);
    let fx=-n.x*.0015,fy=-n.y*.0015;
    for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){
      const cell=cells.get(cellKey(cx+dx,cy+dy));if(!cell)continue;
      const self=dx===0&&dy===0,count=cell.n-(self?1:0);if(!count)continue;
      const mx=(cell.sx-(self?n.x:0))/count,my=(cell.sy-(self?n.y:0))/count;
      let vx=n.x-mx,vy=n.y-my;
      if(Math.abs(vx)+Math.abs(vy)<.01){const a=hash(n.id)%628/100;vx=Math.cos(a);vy=Math.sin(a);}
      const distance=Math.max(8,Math.hypot(vx,vy)),force=Math.min(4,300*count/(distance*distance));
      fx+=vx/distance*force;fy+=vy/distance*force;
    }
    n.vx=(n.vx+fx)*.76;n.vy=(n.vy+fy)*.76;
  }
  for(const edge of scene.edges){
    const a=edge.source,b=edge.target,dx=b.x-a.x,dy=b.y-a.y,dist=Math.max(1,Math.hypot(dx,dy));
    const force=clamp((dist-96)*.007,-1.8,1.8),fx=dx/dist*force,fy=dy/dist*force;
    a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;
  }
  let energy=0;
  for(const n of scene.nodes){
    if(n.pinned){n.vx=0;n.vy=0;continue;}
    n.x=clamp(n.x+clamp(n.vx,-7,7),-100000,100000);n.y=clamp(n.y+clamp(n.vy,-7,7),-100000,100000);
    energy+=Math.abs(n.vx)+Math.abs(n.vy);
  }
  scene.ticks++;indexScene(scene);
  return scene.ticks<SCENE_LIMITS.ticks&&(scene.ticks<12||energy>scene.nodes.length*.06);
}
export function hitNode(scene,point,view,radius=20) {
  const world=worldPoint(point,view),k=clamp(view.k,SCENE_LIMITS.scaleMin,SCENE_LIMITS.scaleMax),r=Math.max(5,finite(radius,20))/k;
  const cellRadius=Math.ceil(r/SCENE_LIMITS.cell),cx=Math.floor(world.x/SCENE_LIMITS.cell),cy=Math.floor(world.y/SCENE_LIMITS.cell);
  let best=null,bestDistance=r;
  // Low zoom hits can span cells. Bound by the already bounded node inventory.
  const check=n=>{const d=Math.hypot(world.x-n.x,world.y-n.y);if(d<bestDistance){best=n;bestDistance=d;}};
  if(cellRadius>4){for(const n of scene.nodes)check(n);}
  else for(let dx=-cellRadius;dx<=cellRadius;dx++)for(let dy=-cellRadius;dy<=cellRadius;dy++)for(const n of scene.grid.get(cellKey(cx+dx,cy+dy))?.nodes||[])check(n);
  return best;
}
