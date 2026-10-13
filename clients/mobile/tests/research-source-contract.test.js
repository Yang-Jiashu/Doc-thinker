import test from 'node:test';
import assert from 'node:assert/strict';
import {MAX_RESEARCH_SOURCES,isResearchSourceLabel,boundedAnswerSources} from '../src/services/research-source-contract.js';

test('final research packet accepts exactly S1 through S10, bounded before filtering',()=>{
  assert.equal(MAX_RESEARCH_SOURCES,10);
  for(let index=1;index<=10;index++) assert.equal(isResearchSourceLabel(`S${index}`),true);
  for(const value of ['S0','S01','S11','S99','S100','s10',' S10','S10\n','K1','S1x',null,{},10]) assert.equal(isResearchSourceLabel(value),false);
  const sources=Array.from({length:11},(_,index)=>({label:`S${index+1}`,content:`source ${index+1}`}));
  assert.deepEqual(boundedAnswerSources(sources,true),sources.slice(0,10));
  assert.deepEqual(boundedAnswerSources([{label:'S11',content:'invalid'},...sources],true),sources.slice(0,9));
  assert.deepEqual(boundedAnswerSources(Array(10).fill(null).concat(sources),true),[]);
  assert.equal(sources.length,11);
});

test('legacy fragments remain bounded at five without research label requirements',()=>{
  const sources=Array.from({length:10},(_,index)=>({content:`legacy ${index}`}));
  assert.deepEqual(boundedAnswerSources(sources),sources.slice(0,5));
  assert.deepEqual(boundedAnswerSources(sources,true),[]);
  for(const value of [null,undefined,{},'sources']) assert.deepEqual(boundedAnswerSources(value,true),[]);
});
