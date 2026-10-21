import test from 'node:test';
import assert from 'node:assert/strict';
import {createGraphPainter,GRAPH_FONT} from '../src/services/graph-paint.js';
import {createScene,SCENE_LIMITS} from '../src/services/graph-scene.js';

// A Canvas2D draw-call double, not a DOM/style snapshot. Widths vary by real
// glyph class and font weight, so layout and painting must agree in CSS pixels.
function context() {
  const calls=[],stack=[];
  const ctx={calls,font:'10px sans-serif',globalAlpha:1,fillStyle:'#000',strokeStyle:'#000',lineWidth:1,
    lineCap:'butt',lineJoin:'miter',textAlign:'start',textBaseline:'alphabetic',fontKerning:'auto',dash:[],clipped:false,path:[]};
  const state=()=>({font:ctx.font,alpha:ctx.globalAlpha,fill:ctx.fillStyle,stroke:ctx.strokeStyle,
    width:ctx.lineWidth,lineCap:ctx.lineCap,lineJoin:ctx.lineJoin,dash:[...ctx.dash],clipped:ctx.clipped});
  const record=(op,args=[])=>{
    for(const value of args)if(typeof value==='number')assert.ok(Number.isFinite(value),`${op} received non-finite coordinate`);
    calls.push({op,args,state:state(),path:ctx.path.map(item=>[...item])});
  };
  ctx.save=()=>{stack.push({font:ctx.font,globalAlpha:ctx.globalAlpha,fillStyle:ctx.fillStyle,strokeStyle:ctx.strokeStyle,
    lineWidth:ctx.lineWidth,lineCap:ctx.lineCap,lineJoin:ctx.lineJoin,textAlign:ctx.textAlign,textBaseline:ctx.textBaseline,
    fontKerning:ctx.fontKerning,dash:[...ctx.dash],clipped:ctx.clipped});record('save');};
  ctx.restore=()=>{Object.assign(ctx,stack.pop());record('restore');};
  ctx.beginPath=()=>{ctx.path=[];record('beginPath');};
  for(const op of ['moveTo','lineTo','quadraticCurveTo','arc','rect'])ctx[op]=(...args)=>{record(op,args);ctx.path.push([op,...args]);};
  ctx.closePath=()=>{ctx.path.push(['closePath']);record('closePath');};
  ctx.clip=(rule)=>{ctx.clipped=true;record('clip',[rule]);};
  ctx.setLineDash=dash=>{assert.ok(dash.every(Number.isFinite));ctx.dash=[...dash];record('setLineDash',dash);};
  for(const op of ['stroke','fill','fillRect','strokeRect','roundRect','strokeText','fillText'])ctx[op]=(...args)=>record(op,args);
  ctx.measureText=text=>{
    record('measureText',[text]);
    const size=Number(ctx.font.match(/([\d.]+)px/)?.[1]||10),weight=ctx.font.startsWith('600 ')?1.035:1;
    return {width:Array.from(text).reduce((sum,glyph)=>sum+(/^[\x20-\x7e]$/u.test(glyph)?size*.53:size),0)*weight};
  };
  return ctx;
}

function sceneAt(nodes,edges=[]) {
  const scene=createScene(nodes.map((node,index)=>({id:`n${index}`,type:'memory',label:`项目资料 ${index}`,...node})),edges);
  for(let index=0;index<nodes.length;index++) {
    const original=nodes[index],node=scene.byId.get(original.id||`n${index}`);
    if(node){node.x=original.x??80+(index%6)*125;node.y=original.y??60+Math.floor(index/6)*80;}
  }
  return scene;
}
const view={x:0,y:0,k:1};
const paint=(painter,ctx,scene,options={})=>painter.paint(ctx,{scene,view,width:900,height:600,kinds:['memory','event'],...options});
const edgeStrokes=ctx=>ctx.calls.filter(call=>call.op==='stroke'&&[.85,1.55].includes(call.state.width));
const nodeFills=ctx=>ctx.calls.filter(call=>call.op==='fill'&&call.path.some(item=>item[0]==='arc'));
const overlap=(a,b)=>a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y;
function assertLegible(labels,scene,viewport,width,height) {
  for(const label of labels) {
    assert.ok(label.x>=8&&label.y>=8,`${label.id} outside leading inset`);
    assert.ok(label.x+label.width<=width-8&&label.y+label.height<=height-8,`${label.id} outside trailing inset`);
    for(const other of labels)if(other!==label)assert.equal(overlap(label,other),false,`${label.id} overlaps ${other.id}`);
    for(const node of scene.nodes) {
      const x=node.x*viewport.k+viewport.x,y=node.y*viewport.k+viewport.y;
      const radius=Math.max(4.5,Math.min(8.5,4.5+Math.sqrt(node.degree)*.9))*Math.max(.85,Math.min(1.4,Math.sqrt(viewport.k)));
      if(x+radius<0||x-radius>width||y+radius<0||y-radius>height)continue;
      const nx=Math.max(label.x,Math.min(label.x+label.width,x)),ny=Math.max(label.y,Math.min(label.y+label.height,y));
      assert.ok((nx-x)**2+(ny-y)**2>=(radius+3)**2-1e-7,`${label.id} overlaps node ${node.id}`);
    }
  }
}

test('screen-pixel type stays legible across zoom with the CJK system font stack',()=>{
  const scene=sceneAt([{id:'a',label:'长期记忆 👩🏽‍💻',x:100,y:90},{id:'b',label:'Evidence 原文.pdf',x:200,y:140}]);
  for(const k of [.25,1,5]) {
    const ctx=context(),painter=createGraphPainter();
    const result=paint(painter,ctx,scene,{view:{x:140,y:90,k},selectedId:'a'});
    const text=ctx.calls.filter(call=>call.op==='fillText');
    assert.ok(text.length>0);
    assert.equal(text.length,result.labels.length);
    for(const call of text) {
      assert.match(call.state.font,/^(400|600) 12\.5px /);
      assert.ok(call.state.font.endsWith(GRAPH_FONT));
      assert.equal(call.state.alpha,1);
    }
    for(const call of ctx.calls.filter(call=>call.op==='measureText'))assert.equal(call.state.font,`600 12.5px ${GRAPH_FONT}`);
    assert.ok(text.some(call=>call.args[0].startsWith('长期')&&call.state.font.startsWith('600 ')));
  }
});

test('unchanged paints reuse measured text, reset clears it, and viewport font sizes have separate cache keys',()=>{
  const scene=sceneAt([{id:'a',label:'精确来源记录与审阅.pdf',x:100,y:90},{id:'b',label:'项目偏好',x:240,y:140}]);
  const ctx=context(),painter=createGraphPainter();
  paint(painter,ctx,scene);
  const first=ctx.calls.filter(call=>call.op==='measureText').length;
  assert.ok(first>0);
  paint(painter,ctx,scene);
  assert.equal(ctx.calls.filter(call=>call.op==='measureText').length,first);
  paint(painter,ctx,scene,{width:390,height:360});
  const mobile=ctx.calls.filter(call=>call.op==='measureText').length;
  assert.ok(mobile>first);
  assert.ok(ctx.calls.some(call=>call.op==='fillText'&&call.state.font.includes(' 12px ')));
  painter.reset();paint(painter,ctx,scene);
  assert.ok(ctx.calls.filter(call=>call.op==='measureText').length>mobile);
});

test('the measurement cache evicts old strings after its bounded 1024-entry window',()=>{
  const ctx=context(),painter=createGraphPainter();
  for(let index=0;index<1100;index++)paint(painter,ctx,sceneAt([{label:`unique${index}`,x:100,y:100}]));
  const before=ctx.calls.filter(call=>call.op==='measureText').length;
  assert.equal(before,1100);
  paint(painter,ctx,sceneAt([{label:'unique1099',x:100,y:100}]));
  assert.equal(ctx.calls.filter(call=>call.op==='measureText').length,before);
  paint(painter,ctx,sceneAt([{label:'unique0',x:100,y:100}]));
  assert.equal(ctx.calls.filter(call=>call.op==='measureText').length,before+1);
});

test('background edges paint first and focused edges last with rounded constant-pixel strokes',()=>{
  const scene=sceneAt([{id:'a',x:80,y:100},{id:'b',x:260,y:120},{id:'c',x:500,y:240},{id:'d',x:680,y:400}],
    [{source:'a',target:'b',directed:true},{source:'b',target:'c'},{source:'c',target:'d',is_discovered:true}]);
  const ctx=context();paint(createGraphPainter(),ctx,scene,{selectedId:'a'});
  const strokes=edgeStrokes(ctx);
  assert.equal(strokes.length,3);
  assert.deepEqual(strokes.map(call=>call.state.width),[.85,.85,1.55]);
  assert.deepEqual(strokes.map(call=>call.state.alpha),[.34,.34,.95]);
  for(const call of strokes){assert.equal(call.state.lineCap,'round');assert.equal(call.state.lineJoin,'round');assert.equal(call.state.clipped,true);}
  assert.ok(strokes.some(call=>call.state.dash.join(',')==='3,5'));
  assert.equal(ctx.calls.filter(call=>call.op==='fill'&&call.path.some(item=>item[0]==='closePath')).length,1);
});

test('ordinary relationships remain straight while only real parallel links curve',()=>{
  for(const parallel of [false,true]) {
    const edges=[{source:'a',target:'b',description:'依据'}];
    if(parallel)edges.push({source:'a',target:'b',description:'依赖'});
    const scene=sceneAt([{id:'a',x:80,y:100},{id:'b',x:460,y:280}],edges);
    const ctx=context();paint(createGraphPainter(),ctx,scene);
    assert.equal(ctx.calls.filter(call=>call.op==='quadraticCurveTo').length,parallel?2:0);
    assert.equal(edgeStrokes(ctx).length,parallel?2:1);
  }
});

test('labels clear every visible projected node and each other with eight-pixel viewport margins',()=>{
  const nodes=Array.from({length:150},(_,index)=>({id:`n${String(index).padStart(3,'0')}`,
    label:`项目需求与证据记录 ${index}.pdf`,x:9+(index%15)*61,y:9+Math.floor(index/15)*58}));
  const scene=sceneAt(nodes);
  for(const [width,height,viewport] of [[900,600,view],[390,360,{x:0,y:0,k:.47}],[900,600,{x:60,y:30,k:.8}]]) {
    const result=paint(createGraphPainter(),context(),scene,{width,height,view:viewport,selectedId:'n000',highlightIds:['n001','n002']});
    assert.ok(result.labels.length>0);
    // No label may be forced through a completely occupied small viewport;
    // when there is a safe selected placement it must win the priority order.
    if(result.labels.some(label=>label.id==='n000'))assert.equal(result.labels[0].id,'n000');
    assertLegible(result.labels,scene,viewport,width,height);
  }
});

test('captions reserve even-odd line whitespace without opaque text boxes or thick strokeText halos',()=>{
  const scene=sceneAt([{id:'a',label:'上游依据',x:80,y:100},{id:'b',label:'下游事件',x:480,y:260}],
    [{source:'a',target:'b'}]);
  const ctx=context(),result=paint(createGraphPainter(),ctx,scene);
  const clips=ctx.calls.filter(call=>call.op==='clip');
  assert.equal(clips.length,1);assert.deepEqual(clips[0].args,['evenodd']);
  assert.equal(clips[0].path.filter(item=>item[0]==='rect').length,result.labels.length+1);
  assert.deepEqual(clips[0].path[0],['rect',0,0,900,600]);
  for(const label of result.labels)assert.ok(clips[0].path.some(item=>JSON.stringify(item)===JSON.stringify(['rect',label.x,label.y,label.width,label.height])));
  assert.equal(ctx.calls.some(call=>['fillRect','strokeRect','roundRect','strokeText'].includes(call.op)),false);
  for(const call of ctx.calls.filter(call=>call.op==='fill'))assert.ok(call.path.some(item=>item[0]==='arc')||call.path.some(item=>item[0]==='closePath'));
  assert.ok(ctx.calls.findIndex(call=>call.op==='clip')<ctx.calls.findIndex(call=>call.op==='stroke'));
  for(const call of ctx.calls.filter(call=>call.op==='fillText'))assert.equal(call.state.clipped,false);
});

test('selected emphasis retains readable background nodes instead of washing them out',()=>{
  const scene=sceneAt([{id:'a',x:80,y:100},{id:'b',x:260,y:120},{id:'c',x:500,y:300}],
    [{source:'a',target:'b'}]);
  const ctx=context();paint(createGraphPainter(),ctx,scene,{selectedId:'a'});
  const fills=nodeFills(ctx);
  assert.equal(fills.length,4); // Selected halo plus three actual node disks.
  assert.ok(fills.some(call=>call.state.alpha===.6));
  assert.ok(fills.every(call=>call.state.alpha>=.6));
  assert.ok(fills.some(call=>call.state.fill==='#e5efda'));
  assert.ok(fills.some(call=>call.state.fill==='#294f43'));
  assert.ok(ctx.calls.filter(call=>call.op==='fillText').every(call=>call.state.alpha===1));
});

test('painting never mutates caller scenes, views, raw graph records or highlight arrays',()=>{
  const rawNodes=[{id:'a',label:'原始来源',type:'memory',x:100,y:100,source_ids:['s1']},{id:'b',label:'确认事件',type:'event',x:400,y:280}];
  const rawEdges=[{source:'a',target:'b',description:'已记录的联系',directed:true,conditions:['待核查']}];
  const scene=sceneAt(rawNodes,rawEdges),viewport={x:40,y:20,k:1.2},highlightIds=['a','b'];
  const snapshot=JSON.stringify({scene,viewport,rawNodes,rawEdges,highlightIds});
  const painter=createGraphPainter();
  paint(painter,context(),scene,{view:viewport,selectedId:'a',highlightIds});
  paint(painter,context(),scene,{view:viewport,selectedId:'b',highlightIds});
  assert.equal(JSON.stringify({scene,viewport,rawNodes,rawEdges,highlightIds}),snapshot);
  assert.deepEqual([...scene.adjacency.get('a')],['b']);
});

test('desktop labels cap at 32 and mobile at 16 without scaling typography down',()=>{
  const scene=sceneAt(Array.from({length:80},(_,index)=>({id:`n${index}`,label:`项${index}`,x:35+(index%10)*150,y:30+Math.floor(index/10)*100})));
  for(const [width,height,cap,size,k] of [[1800,1100,32,12.5,1],[390,1600,16,12,.21]]) {
    const ctx=context(),result=paint(createGraphPainter(),ctx,scene,{width,height,view:{x:0,y:0,k}});
    assert.ok(result.labels.length>0&&result.labels.length<=cap);
    if(width>500)assert.equal(result.labels.length,cap);
    assert.ok(ctx.calls.filter(call=>call.op==='fillText').every(call=>call.state.font.includes(` ${size}px `)));
    assertLegible(result.labels,scene,{x:0,y:0,k},width,height);
  }
});

test('normal zoom extremes keep draw geometry finite and node sizes bounded in screen pixels',()=>{
  const scene=sceneAt([{id:'a',x:20,y:30},{id:'b',x:90,y:50}],
    [{source:'a',target:'b',description:'first',directed:true},{source:'b',target:'a',description:'second',directed:true}]);
  for(const k of [SCENE_LIMITS.scaleMin,.5,1,SCENE_LIMITS.scaleMax]) {
    const ctx=context();paint(createGraphPainter(),ctx,scene,{view:{x:40,y:50,k},selectedId:'a'});
    assert.ok(ctx.calls.length>0);
    for(const call of ctx.calls.filter(call=>call.op==='arc'))assert.ok(call.args[2]>0&&call.args[2]<=17);
    for(const call of ctx.calls.filter(call=>['stroke','fill'].includes(call.op)))assert.ok(Number.isFinite(call.state.width)&&call.state.width>0&&call.state.alpha>=0&&call.state.alpha<=1);
  }
});

test('large real scenes respect the existing 2000-node and 8000-edge bounds',()=>{
  const nodes=Array.from({length:2100},(_,index)=>({id:`n${index}`,label:`材料${index}`,x:(index%100)*100,y:Math.floor(index/100)*100}));
  const edges=Array.from({length:9000},(_,index)=>({source:`n${index%1999}`,target:`n${index%1999+1}`,description:`关系${index}`}));
  const scene=sceneAt(nodes,edges),ctx=context();
  assert.equal(scene.nodes.length,2000);assert.equal(scene.edges.length,8000);
  const result=paint(createGraphPainter(),ctx,scene,{width:1000,height:600,view:{x:10,y:10,k:.05}});
  assert.equal(result.nodes,2000);assert.ok(result.edges<=8000);assert.ok(result.labels.length<=32);
  assert.ok(ctx.calls.filter(call=>call.op==='quadraticCurveTo').length<=8000);
  assert.ok(ctx.calls.filter(call=>call.op==='measureText').length<1000);
});

test('empty inventories draw no synthetic nodes, labels or relationships',()=>{
  const ctx=context(),result=paint(createGraphPainter(),ctx,createScene());
  assert.deepEqual(result,{labels:[],nodes:0,edges:0});
  assert.equal(ctx.calls.some(call=>['stroke','fill','fillText','measureText'].includes(call.op)),false);
});
