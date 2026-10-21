import test from 'node:test';
import assert from 'node:assert/strict';
import {fitGraphLabel,layoutGraphLabels,GRAPH_LABEL_LIMITS as limits} from '../src/services/graph-label-layout.js';

const graphemes=text=>Array.from(new Intl.Segmenter(undefined,{granularity:'grapheme'}).segment(text),item=>item.segment);
const measureText=text=>graphemes(text).reduce((sum,part)=>sum+(/^[\x20-\x7E]+$/u.test(part)?6.3:12),0);
const node=(id,x,y,extra={})=>({id,label:`知识 ${id}`,x,y,radius:6,...extra});
const layout=(nodes,options={})=>layoutGraphLabels({nodes,width:900,height:600,measureText,...options});
function assertClear(labels,nodes,width=900,height=600) {
  for (const label of labels) {
    assert.ok(label.x >= 8 && label.y >= 8,`${label.id} violates leading inset`);
    assert.ok(label.x+label.width <= width-8 && label.y+label.height <= height-8,`${label.id} clips at trailing inset`);
    assert.ok(label.width >= 14 && label.height > 8);
    for (const disk of nodes) {
      if (!Number.isFinite(disk.x) || !Number.isFinite(disk.y)) continue;
      const x=Math.max(label.x,Math.min(label.x+label.width,disk.x));
      const y=Math.max(label.y,Math.min(label.y+label.height,disk.y));
      assert.ok((x-disk.x)**2+(y-disk.y)**2 >= (disk.radius+3)**2-1e-8,`${label.id} overlaps node ${disk.id}`);
    }
    for (const other of labels) {
      if (label === other) continue;
      assert.ok(label.x+label.width+3 <= other.x || other.x+other.width+3 <= label.x || label.y+label.height+3 <= other.y || other.y+other.height+3 <= label.y,
        `${label.id} overlaps label ${other.id}`);
    }
  }
}

test('CJK, emoji sequences and mixed titles fit by measured pixels, not character count',()=>{
  for (const title of ['使用长期记忆定位项目的关键证据','👩🏽‍💻科研计划 🇨🇳 研究资料','Evidence 证据连接 👨‍👩‍👧‍👦 2026']) {
    const result=fitGraphLabel(title,88,measureText);
    assert.ok(result.width <= 88);
    assert.equal(result.fullText,false);
    assert.ok(result.text.endsWith('…'));
    const prefix=result.text.slice(0,-1);
    assert.ok(graphemes(title).some((_,index)=>graphemes(title).slice(0,index+1).join('') === prefix) || prefix === '');
    assert.equal(result.text.includes('\uFFFD'),false);
  }
});
test('short complete labels stay complete and filename ellipsis retains a useful extension',()=>{
  assert.deepEqual(fitGraphLabel('原文',120,measureText),{text:'原文',width:24,fullText:true});
  const result=fitGraphLabel('一份有很长标题的项目需求评审记录.pdf',110,measureText);
  assert.ok(result.text.endsWith('….pdf'));
  assert.ok(result.width <= 110);
  assert.equal(result.fullText,false);
  assert.equal(fitGraphLabel('测试',2,measureText).text,'');
});
test('oversized labels remain bounded and never claim the clipped source was displayed in full',()=>{
  const result=fitGraphLabel(`${'a'.repeat(511)}👨‍👩‍👧‍👦${'b'.repeat(5000)}`,100000,measureText);
  assert.equal(result.fullText,false);
  assert.ok(result.text.endsWith('…'));
  assert.equal(result.text.includes('\uFFFD'),false);
  assert.equal(graphemes(result.text).length,513);
  assert.ok(result.text.endsWith('👨‍👩‍👧‍👦…'));
});
test('older clients keep emoji and combining sequences intact without Intl.Segmenter',()=>{
  const original=Intl.Segmenter;
  try {
    Intl.Segmenter=undefined;
    const measure=text=>Array.from(text).length*8;
    const result=fitGraphLabel('👩🏽‍💻é🇨🇳后续资料',76,measure);
    assert.ok(result.width <= 76);
    assert.ok(['👩🏽‍💻…','👩🏽‍💻é…','👩🏽‍💻é🇨🇳…'].includes(result.text));
  } finally {Intl.Segmenter=original;}
});
test('all dense graph pills avoid other pills and every projected node disk',()=>{
  const nodes=Array.from({length:240},(_,i)=>node(`n${String(i).padStart(3,'0')}`,30+(i%20)*44,30+Math.floor(i/20)*48));
  const labels=layout(nodes,{maxLabels:45});
  assert.ok(labels.length > 0 && labels.length <= 45);
  assertClear(labels,nodes);
});
test('selected nodes at each viewport edge are readable, wholly inset and not crowded by ordinary labels',()=>{
  const width=390,height=360;
  for (const [x,y] of [[7,7],[width-7,7],[7,height-7],[width-7,height-7],[width/2,7],[width-7,height/2]]) {
    const nodes=[...Array.from({length:100},(_,i)=>node(`filler${i}`,35+(i%10)*34,60+Math.floor(i/10)*22)),
      node('selected',x,y,{label:'这是一个很长的当前选中知识节点标题'} )];
    const labels=layout(nodes,{width,height,selectedId:'selected',maxLabels:45});
    assert.equal(labels[0]?.id,'selected');
    assertClear(labels,nodes,width,height);
  }
});
test('the selected node, route and neighbor priority precede degree and stable IDs',()=>{
  const nodes=[node('a',50,80,{degree:99}),node('neighbor',50,180,{priority:1,degree:0}),node('route',50,280),node('selected',50,380)];
  const labels=layout(nodes,{selectedId:'selected',routeIds:['route'],maxLabels:4});
  assert.deepEqual(labels.map(item=>item.id),['selected','route','neighbor','a']);
  assert.deepEqual(layout(nodes,{selectedId:'selected',maxLabels:1}).map(item=>item.id),['selected']);
});
test('reordered equivalent input returns exactly the same deterministic layout',()=>{
  const nodes=[node('c',300,180),node('a',120,180),node('b',500,180)];
  assert.deepEqual(layout(nodes),layout([...nodes].reverse()));
  assert.deepEqual(layout(nodes),layout(nodes));
});
test('previous anchor preference persists across tiny movement and only changes when blocked',()=>{
  const nodes=[node('a',300,250)],first=layout(nodes);
  const previous=new Map(first.map(item=>[item.id,{...item,anchor:'w'}]));
  const moved=[{...nodes[0],x:300.5,y:250.5}],next=layout(moved,{previous});
  assert.equal(next[0].anchor,'w');
  assert.equal(previous.get('a').anchor,'w');
  const blocked=layout([...moved,node('obstacle',next[0].x+next[0].width/2,next[0].y+next[0].height/2,{label:''})],{previous});
  assert.notEqual(blocked[0].anchor,'w');
  assertClear(blocked,[...moved,node('obstacle',next[0].x+next[0].width/2,next[0].y+next[0].height/2,{label:''})]);
});
test('zoom is handled entirely in projected pixels and text stays screen-sized',()=>{
  const small=[node('a',120,120,{radius:3})],large=[node('a',240,240,{radius:18})];
  const [a]=layout(small),[b]=layout(large);
  assert.equal(a.width,b.width);assert.equal(a.height,b.height);
  assertClear([a],small);assertClear([b],large);
  assert.equal(a.x-small[0].x,small[0].radius+7);
  assert.equal(b.x-large[0].x,large[0].radius+7);
});
test('mobile ordinary label widths are smaller than desktop while selected titles have more space',()=>{
  const nodes=[node('a',20,150,{label:'用于检索和来源定位的长标题项目文档'})];
  const [mobile]=layout(nodes,{width:390}),[desktop]=layout(nodes),[selected]=layout(nodes,{width:390,selectedId:'a'});
  assert.ok(mobile.width <= 134);assert.ok(desktop.width <= 174);assert.ok(selected.width <= 274);
  assert.ok(mobile.width < desktop.width && desktop.width < selected.width);
});
test('positions, graph relations and previous geometry are never mutated',()=>{
  const nodes=[node('a',100,100,{source_ids:['private'],relations:[{target:'b'}]}),node('b',400,200)];
  const previous=new Map([['a',{anchor:'n',x:100,y:40}]]),input=structuredClone(nodes),before=structuredClone(previous);
  const labels=layout(nodes,{previous});
  assert.deepEqual(nodes,input);assert.deepEqual(previous,before);
  assert.equal('relations' in labels[0],false);assert.equal('source_ids' in labels[0],false);
});
test('invalid view extents, measurement or coordinates produce no nonfinite geometry',()=>{
  for (const value of [0,-1,NaN,Infinity,1e308]) {
    assert.deepEqual(layout([node('a',100,100)],{width:value}),[]);
    assert.deepEqual(layout([node('a',100,100)],{height:value}),[]);
  }
  for (const fontSize of [0,-3,NaN,Infinity]) assert.deepEqual(layout([node('a',100,100)],{fontSize}),[]);
  assert.deepEqual(layout([node('a',NaN,30),node('b',10,Infinity),node('c',10,10,{radius:0})]),[]);
  assert.deepEqual(layout([node('a',100,100)],{measureText:()=>NaN}),[]);
  assert.deepEqual(layout([node('a',100,100)],{measureText:()=>{throw new Error('unavailable');}}),[]);
});
test('inventories, candidate measurements and returned labels are hard bounded',()=>{
  const nodes=Array.from({length:5000},(_,i)=>node(`n${i}`,100+(i%25)*100,100+Math.floor(i/25)*50));
  const seen=new Set();
  const labels=layout(nodes,{width:3000,height:5000,maxLabels:999,measureText:text=>{seen.add(text);return measureText(text);}});
  assert.ok(labels.length <= limits.labels);
  assert.ok(labels.every(item=>Number(item.id.slice(1)) < limits.nodes));
  assert.ok(seen.size <= limits.candidates*12);
  assert.deepEqual(layout(nodes,{maxLabels:0}),[]);
});
test('unlabeled nodes remain obstacles and extreme radii do not allocate a huge grid',()=>{
  const covered=layout([node('a',300,200),node('blank',300,200,{label:'',radius:1e300})]);
  assert.deepEqual(covered,[]);
  const nodes=[node('a',300,200),node('blank',345,200,{label:''})],labels=layout(nodes);
  assert.equal(labels[0]?.id,'a');assert.notEqual(labels[0]?.anchor,'e');
  assertClear(labels,nodes);
});
test('duplicate identities are safe and control characters are not painted',()=>{
  const labels=layout([node('a',100,100,{label:'标题\u202E 资料\n\t 2'}),node('a',400,400),node('__proto__',500,100)]);
  assert.equal(labels.length,2);
  assert.equal(labels[0].text.includes('\u202E'),false);
  assert.equal(labels[0].text.includes('\n'),false);
  assert.equal(Object.getPrototypeOf({}),Object.prototype);
});
