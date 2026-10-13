import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript,compileTemplate} from '@vue/compiler-sfc';
const require=createRequire(import.meta.url);
const {preprocess}=require('@dcloudio/uni-cli-shared/lib/preprocess');
const read=path=>readFileSync(new URL('../src/'+path,import.meta.url),'utf8');
const elements=node=>[...(node.type===1?[node]:[]),...(node.children||[]).flatMap(elements)];
const attr=(node,name)=>node.props.find(prop=>prop.type===6&&prop.name===name)?.value?.content;
test('invitation inputs have explicit accessible names and every action has a button role',()=>{
  const nodes=elements(parse(read('pages/invite/index.vue')).descriptor.template.ast);
  const inputs=nodes.filter(node=>node.tag==='input');assert.equal(inputs.length,3);
  const names=inputs.map(node=>{
    const fixed=attr(node,'aria-label');if(fixed)return fixed;
    const binding=node.props.find(prop=>prop.type===7&&prop.name==='bind'&&prop.arg?.content==='aria-label');
    assert.ok(binding);const evaluate=new Function('register',`return (${binding.exp.content});`);
    assert.equal(evaluate(false),'密码');assert.equal(evaluate(true),'设置密码');return '密码';
  });
  assert.deepEqual(names,['受邀用户名','密码','邀请码']);
  const buttons=nodes.filter(node=>node.tag==='button');assert.equal(buttons.length,7);
  for(const button of buttons)assert.equal(attr(button,'role'),'button');
});
test('invitation empty-state and starter discuss supplied ideas, while demo/private retain document prompts',()=>{
  const nodes=elements(parse(read('pages/chat/index.vue')).descriptor.template.ast);
  const empty=nodes.find(node=>attr(node,'class')==='empty-copy');
  const starter=nodes.find(node=>attr(node,'class')==='suggestion suggestion-connect');
  const title=elements(starter).find(node=>attr(node,'class')==='suggestion-title');
  const expression=node=>node.children.find(child=>child.type===5).content.content;
  const tap=starter.props.find(prop=>prop.type===7&&prop.name==='on'&&prop.arg?.content==='tap').exp.content;
  for(const preferences of [{mode:'connected',connectionKind:'invite'},{mode:'demo'},{mode:'connected'}]) {
    const copy=new Function('preferences',`return (${expression(empty)});`)(preferences);
    const label=new Function('preferences',`return (${expression(title)});`)(preferences);
    let draft='';new Function('preferences','fillDraft',tap)(preferences,value=>{draft=value;});
    if(preferences.connectionKind==='invite') {assert.match(copy,/比较不同观点/);assert.equal(label,'比较不同观点');assert.match(draft,/比较几种观点/);assert.doesNotMatch(copy+draft,/资料|文档|原文/);}
    else {assert.match(copy,/带上资料一起研究/);assert.equal(label,'一起做研究');assert.match(draft,/当前会话资料/);}
  }
});
test('invitation route is independent and native pages have no invitation auth runtime or forms',()=>{
  const pages=JSON.parse(read('pages.json'));
  assert.equal(pages.pages[0].path,'pages/chat/index');assert.ok(pages.pages.some(page=>page.path==='pages/invite/index'));
  assert.ok(!pages.tabBar.list.some(page=>page.pagePath==='pages/invite/index'));
  for(const platform of ['H5','MP_WEIXIN','APP_PLUS']) {
    const context={H5:false,MP_WEIXIN:false,APP_PLUS:false,[platform]:true};
    const {descriptor}=parse(read('pages/invite/index.vue'));
    const template=preprocess(descriptor.template.content,context,{type:'html'});
    const script=preprocess(descriptor.scriptSetup.content,context,{type:'js'});
    const component=parse(`<template>${template}</template><script setup>${script}</script>`).descriptor;
    const compiled=compileScript(component,{id:platform});
    assert.deepEqual(compileTemplate({source:template,id:platform,compilerOptions:{bindingMetadata:compiled.bindings}}).errors,[]);
    if(platform!=='H5') {assert.doesNotMatch(script,/invite-auth|authenticateInvitation|enterInvitation/);assert.doesNotMatch(template,/v-model|@tap="submit"/);assert.match(template,/目前仅支持网页/);}
    else {assert.match(script,/authenticateInvitation/);assert.match(template,/尚未开放/);}
    const appScript=preprocess(parse(read('App.vue')).descriptor.script.content,context,{type:'js'});
    if(platform!=='H5') assert.doesNotMatch(appScript,/window|invite-auth|clearInviteSession/);
  }
});
test('web startup locks pending invitation to login; pagehide clears identity and BFCache cannot restore old private views',()=>{
  const script=preprocess(parse(read('App.vue')).descriptor.script.content,{H5:true},{type:'js'}).replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const events={},calls=[];let preferences={mode:'demo'};
  const scope={getPreferences:()=>preferences,clearInviteSession:()=>calls.push('clear'),cancelInviteRequest:()=>calls.push('cancel-auth'),cancelRecording:()=>calls.push('stop-recording'),stopSpeaking:()=>calls.push('stop-speaking'),
    uni:{reLaunch:options=>calls.push(options.url)},window:{addEventListener:(name,fn)=>{events[name]=fn;},location:{reload:()=>calls.push('reload')}}};
  const app=new Function(...Object.keys(scope),script)(...Object.values(scope));
  app.onLaunch();assert.deepEqual(calls,[]);events.pagehide();events.pageshow({persisted:true});assert.deepEqual(calls,[]);
  preferences={mode:'connected',connectionKind:'invite',invitePending:true};app.onLaunch();assert.deepEqual(calls,['/pages/invite/index']);
  events.pagehide();assert.deepEqual(calls.slice(1),['cancel-auth','clear','stop-recording','stop-speaking']);events.pageshow({persisted:true});assert.equal(calls.at(-1),'reload');
});
