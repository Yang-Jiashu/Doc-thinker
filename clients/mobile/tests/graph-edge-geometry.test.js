import test from 'node:test';
import assert from 'node:assert/strict';
import {buildEdgeRoutes} from '../src/services/graph-edge-geometry.js';

const point=(x,y,radius=6)=>({x,y,radius});
const inventory=(a=point(0,0),b=point(240,0))=>new Map([['a',a],['b',b]]);
const edge=(source='a',target='b',description='supports',directed=true)=>({source,target,description,directed});
const near=(a,b,epsilon=.0001)=>assert.ok(Math.abs(a-b)<epsilon,`${a} ≉ ${b}`);
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const routeShape=route=>({source:typeof route.edge.source==='object'?route.edge.source.id:route.edge.source,
  target:typeof route.edge.target==='object'?route.edge.target.id:route.edge.target,
  description:route.edge.description,start:route.start,end:route.end,control:route.control,arrow:route.arrow,mid:route.mid});

test('ordinary edges stay exactly straight and stop outside each projected node',()=>{
  const points=inventory(point(10,20,8),point(310,20,12)),original=edge();
  const [route]=buildEdgeRoutes([original],points);
  assert.equal(route.edge,original);
  assert.equal(route.control,null);
  assert.deepEqual(route.start,{x:20,y:20});
  assert.deepEqual(route.end,{x:296,y:20});
  assert.deepEqual(route.mid,{x:158,y:20});
  assert.deepEqual(route.arrow,[{x:296,y:20},{x:289,y:22},{x:289,y:18}]);
});

test('circle padding and arrow size are CSS pixels, not world/zoom units',()=>{
  for(const zoom of [.05,.25,1,5]) {
    const a=point(30,40,7),b=point(30+600*zoom,40+240*zoom,5);
    const [route]=buildEdgeRoutes([edge()],inventory(a,b));
    near(distance(route.start,a),9);near(distance(route.end,b),7);
    if(route.arrow) {
      const base={x:(route.arrow[1].x+route.arrow[2].x)/2,y:(route.arrow[1].y+route.arrow[2].y)/2};
      near(distance(route.arrow[0],base),7);near(distance(route.arrow[1],route.arrow[2]),4);
    }
  }
});

test('parallel links alone receive stable fanned lanes with exact curved circle trims',()=>{
  const points=inventory(point(0,0,13),point(300,100,19));
  const routes=buildEdgeRoutes([edge('a','b','beta'),edge('a','b','alpha')],points);
  assert.equal(routes.length,2);
  assert.notDeepEqual(routes[0].control,routes[1].control);
  for(const route of routes) {
    assert.ok(route.control);
    near(distance(route.start,points.get('a')),15);
    near(distance(route.end,points.get('b')),21);
    const tip=route.arrow[0],base={x:(route.arrow[1].x+route.arrow[2].x)/2,y:(route.arrow[1].y+route.arrow[2].y)/2};
    const tangent={x:route.end.x-route.control.x,y:route.end.y-route.control.y};
    near((tip.x-base.x)*tangent.y-(tip.y-base.y)*tangent.x,0,.001);
    assert.ok((tip.x-base.x)*tangent.x+(tip.y-base.y)*tangent.y>0);
  }
});

test('reciprocal links occupy different physical sides and arrows follow each direction',()=>{
  const routes=buildEdgeRoutes([edge('a','b','same'),edge('b','a','same')],inventory());
  assert.equal(routes.length,2);
  assert.ok(routes[0].mid.y*routes[1].mid.y<0);
  assert.ok(routes[0].start.x<routes[0].end.x);
  assert.ok(routes[1].start.x>routes[1].end.x);
  assert.ok(routes[0].arrow[0].x>routes[0].arrow[1].x);
  assert.ok(routes[1].arrow[0].x<routes[1].arrow[1].x);
});

test('reversing an ordinary directed edge reverses its arrow, never its label',()=>{
  const description='A does not imply B',original=edge('b','a',description);
  const [route]=buildEdgeRoutes([original],inventory());
  assert.equal(route.edge.description,description);
  assert.deepEqual(route.start,{x:232,y:0});
  assert.deepEqual(route.end,{x:8,y:0});
  assert.deepEqual(route.arrow,[{x:8,y:0},{x:15,y:-2},{x:15,y:2}]);
});

test('lane geometry is deterministic across edge and point insertion ordering',()=>{
  const edges=[edge('a','b','second'),edge('b','a','reverse'),edge('a','b','first'),edge('a','b','third')];
  const first=buildEdgeRoutes(edges,inventory()).map(routeShape);
  const second=buildEdgeRoutes([...edges].reverse(),new Map([...inventory()].reverse())).map(routeShape);
  assert.deepEqual(second,first);
});

test('fanned inventories never pile reciprocal lanes on the same centre line',()=>{
  const edges=Array.from({length:20},(_,i)=>edge(i%2?'b':'a',i%2?'a':'b',`${i}`));
  const routes=buildEdgeRoutes(edges,inventory(point(0,0),point(600,0)));
  const centers=routes.map(route=>Math.round(route.mid.y*10000)/10000);
  assert.equal(new Set(centers).size,20);
  assert.ok(centers.every(value=>Math.abs(value)<=36.001));
});

test('topology plans cache no coordinates and never modify caller data',()=>{
  const a=Object.freeze({...point(0,0),id:'a'}),b={...point(240,0),id:'b'};
  const original=Object.freeze({...edge(Object.freeze({id:'a'}),Object.freeze({id:'b'})),metadata:Object.freeze({meaning:'unchanged'})});
  const edges=Object.freeze([original]),points=new Map([['a',a],['b',b]]);
  const snapshot=JSON.stringify({a,b,edges});
  const first=buildEdgeRoutes(edges,points);
  assert.equal(JSON.stringify({a,b,edges}),snapshot);
  b.x=340;
  const next=buildEdgeRoutes(edges,points);
  assert.notDeepEqual(next[0].end,first[0].end);
  assert.equal(next[0].edge,original);
});

test('undirected and too-short visible edges do not acquire arrowheads',()=>{
  assert.equal(buildEdgeRoutes([edge('a','b','related',false)],inventory())[0].arrow,null);
  const [tiny]=buildEdgeRoutes([edge()],inventory(point(0,0,6),point(22,0,6)));
  assert.ok(tiny);assert.equal(tiny.arrow,null);
});

test('missing, overlapping, coincident, invalid and self-loop geometry is omitted',()=>{
  assert.deepEqual(buildEdgeRoutes(null,inventory()),[]);
  assert.deepEqual(buildEdgeRoutes([edge()],{}),[]);
  for(const points of [inventory(point(0,0),point(0,0)),inventory(point(0,0,20),point(30,0,20)),
    inventory(point(NaN,0),point(20,0)),inventory(point(0,0),point(Infinity,0)),
    inventory(point(0,0,-1),point(240,0)),inventory(point(0,0),{x:240,y:0}),new Map([['a',point(0,0)]])]) {
    assert.deepEqual(buildEdgeRoutes([edge()],points),[]);
  }
  assert.deepEqual(buildEdgeRoutes([edge('a','a'),null,{},edge('a','missing')],inventory()),[]);
});

test('all return geometry is finite even for malformed very large coordinates',()=>{
  const cases=[inventory(point(1e308,1e308),point(-1e308,-1e308)),
    inventory(point(1e307,1e307,6),point(1.1e307,1.1e307,6)),
    inventory(point(0,0),point(17,0))];
  for(const points of cases) for(const route of buildEdgeRoutes([edge('a','b','1'),edge('a','b','2')],points)) {
    for(const vertex of [route.start,route.end,route.control,route.mid,...(route.arrow||[])].filter(Boolean)) {
      assert.ok(Number.isFinite(vertex.x)&&Number.isFinite(vertex.y));
    }
  }
});

test('work and output stay bounded at 8,000 edges and 2,000 projected nodes',()=>{
  const points=new Map(Array.from({length:2200},(_,i)=>[`n${i}`,point((i%100)*100,Math.floor(i/100)*100,5)]));
  const edges=Array.from({length:10000},(_,i)=>edge(`n${i%1999}`,`n${i%1999+1}`,`edge${i}`,false));
  const routes=buildEdgeRoutes(edges,points);
  assert.equal(routes.length,8000);
  assert.ok(routes.every(route=>route.edge===edges[Number(route.edge.description.slice(4))]));
  assert.deepEqual(buildEdgeRoutes([edge('n0','n2100')],points),[]);
});
