import test from 'node:test';
import assert from 'node:assert/strict';
import {createScene,stepScene,indexScene,hitNode,worldPoint,zoomAt,fitScene,SCENE_LIMITS as limits} from '../src/services/graph-scene.js';

const nodes=[{id:'a',label:'目标',type:'event'},{id:'b',label:'依据',type:'source'},{id:'c',label:'偏好',type:'memory'}];
const edges=[{source:'a',target:'b',description:'原文依据'},{source:'c',target:'b',description:'确认的来源'}];
const positions=s=>s.nodes.map(n=>({id:n.id,x:n.x,y:n.y}));
test('scene is a copy, only actual valid relationships exist, and inputs are never mutated',()=>{
  const data=structuredClone({nodes,edges}),before=structuredClone(data),s=createScene(data.nodes,data.edges);
  for(let i=0;i<25;i++)stepScene(s);
  assert.deepEqual(data,before);assert.equal(s.nodes.length,3);assert.equal(s.edges.length,2);
  assert.equal(s.adjacency.get('b').size,2);assert.equal(s.adjacency.get('a').has('c'),false);
});
test('initial and simulated layout is deterministic across identity and edge order',()=>{
  const a=createScene(nodes,edges),b=createScene([...nodes].reverse(),[...edges].reverse());
  for(let i=0;i<40;i++){stepScene(a);stepScene(b);}
  assert.deepEqual(positions(a),positions(b));
});
test('duplicate identities, invalid endpoints, self-loops and prototype-like IDs remain safe',()=>{
  const s=createScene([...nodes,{id:'a',label:'重复'},{id:'__proto__',label:'<img onerror=alert(1)>'},{id:' bad ',label:'空白'},null],
    [...edges,...edges,{source:'a',target:'missing'},{source:'a',target:'a'},null,{source:'__proto__',target:'a'}]);
  assert.equal(s.byId.get('a').label,'目标');assert.equal(s.byId.get('__proto__').label,'<img onerror=alert(1)>');
  assert.equal(s.edges.length,3);assert.equal(Object.getPrototypeOf({}),Object.prototype);
});
test('render budgets cap inventories and explicitly mark the projection as limited',()=>{
  const many=Array.from({length:limits.nodes+100},(_,i)=>({id:`n${i}`,label:`Node ${i}`}));
  const s=createScene(many,Array.from({length:limits.edges+10},()=>({source:'n0',target:'n1'})));
  assert.equal(s.nodes.length,limits.nodes);assert.equal(s.limited,true);assert.equal(s.edges.length,1);
});
test('grid contains every displayed node once and picks nearest point in world coordinates',()=>{
  const s=createScene(nodes,edges);indexScene(s);const n=s.byId.get('a'),view={x:17,y:-21,k:.8};
  assert.equal([...s.grid.values()].reduce((sum,c)=>sum+c.n,0),3);
  assert.equal(hitNode(s,{x:n.x*view.k+view.x,y:n.y*view.k+view.y},view)?.id,'a');
  assert.equal(hitNode(s,{x:100000,y:100000},view),null);
});
test('low-zoom hit testing stays bounded and works beyond immediate grid cells',()=>{
  const s=createScene([{id:'a',label:'A'}]);indexScene(s);
  const n=s.nodes[0],v={x:0,y:0,k:.05};
  assert.equal(hitNode(s,{x:n.x*v.k+3,y:n.y*v.k+3},v)?.id,'a');
});
test('zoom preserves cursor anchor, clamps extremes and never emits NaN',()=>{
  const v={x:73,y:92,k:.8},p={x:120,y:200},before=worldPoint(p,v);
  assert.deepEqual(worldPoint(p,zoomAt(v,p,2)),before);
  assert.equal(zoomAt(v,p,1e9).k,limits.scaleMax);assert.equal(zoomAt(v,p,-3).k,limits.scaleMin);
  for(const value of Object.values(zoomAt({x:NaN,y:Infinity,k:0},{x:NaN,y:0},NaN)))assert.equal(Number.isFinite(value),true);
});
test('fit includes node bounds, is valid for empty or zero-sized screens',()=>{
  const s=createScene(nodes,edges),v=fitScene(s,800,430);
  for(const n of s.nodes){assert.ok(n.x*v.k+v.x>0&&n.x*v.k+v.x<800);assert.ok(n.y*v.k+v.y>0&&n.y*v.k+v.y<430);}
  assert.deepEqual(fitScene(createScene(),800,430),{x:400,y:215,k:1});
  assert.ok(Number.isFinite(fitScene(s,0,0).k));
});
test('pinned drag position is retained and layout always reaches its finite tick bound',()=>{
  const s=createScene(nodes,edges);s.nodes[0].pinned=true;s.nodes[0].x=100;s.nodes[0].y=80;
  for(let i=0;i<limits.ticks+10;i++)stepScene(s);
  assert.equal(s.nodes[0].x,100);assert.equal(s.nodes[0].y,80);assert.equal(s.ticks,limits.ticks);assert.equal(stepScene(s),false);
  for(const n of s.nodes)for(const key of ['x','y','vx','vy'])assert.ok(Number.isFinite(n[key]));
});
test('dense large view uses one bounded grid and edge pass without growing graph or retained history',()=>{
  const data=Array.from({length:1000},(_,i)=>({id:`n${i}`,label:`Node ${i}`,type:'node'}));
  const links=data.slice(1).map((n,i)=>({source:`n${i}`,target:n.id}));
  const s=createScene(data,links);for(let i=0;i<12;i++)stepScene(s);
  assert.equal(s.nodes.length,1000);assert.equal(s.edges.length,999);assert.ok(s.grid.size<=1000);assert.equal(s.ticks,12);
});
