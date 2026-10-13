import test from 'node:test';
import assert from 'node:assert/strict';
import {validResearchExport,researchExportParts,downloadResearchMarkdown,MAX_RESEARCH_EXPORT_BYTES,EXPORT_COPY_PART_UNITS} from '../src/services/research-export.js';

test('export bounds match UTF-8 server bytes, not an incompatible character cap',()=>{
  assert.equal(validResearchExport('a'.repeat(MAX_RESEARCH_EXPORT_BYTES)),true);
  assert.equal(validResearchExport('a'.repeat(MAX_RESEARCH_EXPORT_BYTES+1)),false);
  assert.equal(validResearchExport('🧠'.repeat(200000)),true);
  assert.equal(validResearchExport('🧠'.repeat(200001)),false);
  for(const value of [null,{},'', '  ', '\ud800','x\udfff']) assert.equal(validResearchExport(value),false);
});

test('copy pieces reconstruct every original character including surrogate boundaries without separators',()=>{
  const text='a'.repeat(39999)+'🧠'+'\r\n中文\n'.repeat(15000)+'last';
  const parts=researchExportParts(text);
  assert.ok(parts.length > 2); assert.equal(parts.join(''),text);
  assert.equal(parts[0].length,39999); assert.equal(parts[1].slice(0,2),'🧠');
  for(const part of parts) { assert.ok(part.length <= EXPORT_COPY_PART_UNITS); assert.equal(validResearchExport(part),true); }
  assert.deepEqual(researchExportParts(''),[]);
  assert.throws(()=>researchExportParts('x'.repeat(800001)),/没有截断/);
});

function browser(overrides={}) {
  const calls={blobs:[],urls:[],revoked:[],clicked:0,removed:0,timers:[]};
  const link={download:'',click(){calls.clicked++;},remove(){calls.removed++;}};
  const document={createElement:tag=>{assert.equal(tag,'a');return link;},body:{appendChild:value=>{assert.equal(value,link);}}};
  const URL={createObjectURL:blob=>{calls.blobs.push(blob);calls.urls.push('blob:local-only');return 'blob:local-only';},revokeObjectURL:value=>calls.revoked.push(value)};
  return {calls,link,options:{document,URL,Blob,defer:(fn,ms)=>calls.timers.push({fn,ms}),...overrides}};
}

test('download is a complete literal local blob with fixed safe filename, no HTML interpretation',async()=>{
  const text='# 私密\n<script>alert(1)</script>\n'+'full'.repeat(50000),b=browser();
  const result=downloadResearchMarkdown(text,'#00001',b.options);
  assert.deepEqual(result,{requested:true,filename:'DocThinker-research-00001.md'});
  assert.equal(await b.calls.blobs[0].text(),text);assert.equal(b.calls.blobs[0].type,'text/markdown;charset=utf-8');
  assert.equal(b.link.href,'blob:local-only');assert.equal(b.link.download,result.filename);
  assert.equal(b.calls.clicked,1);assert.equal(b.calls.removed,1);assert.equal(b.calls.revoked.length,0);
  assert.equal(b.calls.timers[0].ms,15000);b.calls.timers[0].fn();assert.deepEqual(b.calls.revoked,['blob:local-only']);
});

test('invalid session, missing native support and stale scope cause no download',()=>{
  const b=browser();assert.throws(()=>downloadResearchMarkdown('Private','#00001/../../other',b.options));
  assert.throws(()=>downloadResearchMarkdown('Private','#00001',{...b.options,document:null}),/分段查看/);
  assert.throws(()=>downloadResearchMarkdown('Private','#00001',{...b.options,isCurrent:()=>false}));
  assert.equal(b.calls.clicked,0);assert.equal(b.calls.blobs.length,0);
});

test('scope change after preparing a blob revokes it and cannot click an old export',()=>{
  let checks=0;const b=browser({isCurrent:()=>++checks === 1});
  assert.throws(()=>downloadResearchMarkdown('Private','#00001',b.options),/内容已保留/);
  assert.equal(b.calls.clicked,0);assert.equal(b.calls.removed,1);assert.deepEqual(b.calls.revoked,['blob:local-only']);
});

test('browser rejection cleans up and never leaks native errors or claims a saved file',()=>{
  const b=browser();b.link.click=()=>{throw new Error('SENSITIVE native detail');};
  assert.throws(()=>downloadResearchMarkdown('Private','#00001',b.options),error=>/内容已保留/.test(error.message)&&!/SENSITIVE/.test(error.message));
  assert.equal(b.calls.removed,1);assert.deepEqual(b.calls.revoked,['blob:local-only']);assert.equal(b.calls.timers.length,0);
});
