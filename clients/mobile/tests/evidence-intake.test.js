import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { ref, computed, watch, reactive, effectScope } from 'vue';

const source=readFileSync(new URL('../src/components/EvidenceIntake.vue',import.meta.url),'utf8');
const ready={id:'ready-1',filename:'My original.pdf',status:'ready',chunk_count:3};
const pending={id:'queued-1',filename:'New paper.pdf',status:'queued',chunk_count:0};
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return{promise,resolve,reject};};
function mount(t,options={}) {
  const {descriptor,errors}=parse(source);assert.deepEqual(errors,[]);
  const script=compileScript(descriptor,{id:'evidence-intake'});
  assert.deepEqual(compileTemplate({source:descriptor.template.content,id:'evidence-intake',compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);
  const props=reactive({sessionId:'#00001',identity:'account-a',enabled:true,uploadsAllowed:true,disabled:false,active:true,mode:'connected',connectionKind:'private',invitePending:false,...options.props});
  const events=[],calls={documents:[],upload:[],text:[]},hooks={};
  const scope={ref,computed,watch,onBeforeUnmount:fn=>{hooks.unmount=fn;},onHide:fn=>{hooks.hide=fn;},research:{
    documents:async(...args)=>{calls.documents.push(args);return options.documents ? options.documents(...args) : [ready];},
    upload:async(...args)=>{calls.upload.push(args);return options.upload ? options.upload(...args) : {document:pending};},
    uploadText:async(...args)=>{calls.text.push(args);return options.uploadText ? options.uploadText(...args) : {document:pending};},
  }};
  const executable=script.content.replace(/^import[^;]+;\s*$/gm,'').replace('export default','return');
  const component=new Function(...Object.keys(scope),executable)(...Object.values(scope));
  const effects=effectScope(),ui=effects.run(()=>component.setup(props,{expose(){},emit:(...event)=>events.push(event)}));
  let disposed=false;const dispose=()=>{if(!disposed){disposed=true;hooks.unmount();effects.stop();}};t.after(dispose);
  return{props,events,calls,hooks,ui,dispose};
}
const emitted=(f,name)=>f.events.filter(([key])=>key===name).map(([,value])=>value);

test('explicit clearing releases an unsent original without deleting uploaded documents or calling APIs',t=>{
  const f=mount(t);f.ui.title.value='Original';f.ui.text.value='Keep until explicit';f.ui.documents.value=[ready];
  f.props.disabled=true;f.ui.clearPaste();assert.equal(f.ui.text.value,'Keep until explicit');
  f.props.disabled=false;f.ui.clearPaste();assert.equal(f.ui.title.value,'');assert.equal(f.ui.text.value,'');assert.equal(emitted(f,'dirty').at(-1),false);
  assert.equal(f.ui.documents.value[0].id,ready.id);assert.deepEqual(f.calls,{documents:[],upload:[],text:[]});
});

test('opening, editing and unfolding the real-material panel never call an API or change connection settings',t=>{
  const f=mount(t);f.ui.togglePaste();f.ui.title.value='My paper';f.ui.text.value='My own original text';
  assert.equal(f.ui.pasteOpen.value,true);assert.deepEqual(f.calls,{documents:[],upload:[],text:[]});assert.deepEqual(f.events,[['dirty',true]]);
  assert.match(source,/maxlength="-1"/);assert.match(source,/<!-- #ifdef H5 -->/);
  assert.doesNotMatch(source,/getPreferences|savePreferences|api\.ask|setStorage|localStorage|setInterval/);
});

test('explicit checking reports real ready, processing and failed metadata, not a simulated success',async t=>{
  const f=mount(t,{documents:async()=>[ready,pending,{id:'failed',filename:'Bad.pdf',status:'failed',chunk_count:0},{id:'empty',filename:'Empty.pdf',status:'ready',chunk_count:0},{id:'deleted',filename:'Gone.pdf',status:'deleted',chunk_count:4}]});
  assert.equal(f.ui.checked.value,false);await f.ui.checkDocuments();
  assert.deepEqual(f.calls.documents,[['#00001']]);assert.equal(f.ui.readyCount.value,1);assert.equal(f.ui.pendingCount.value,1);assert.equal(f.ui.failedCount.value,1);assert.equal(f.ui.documents.value.length,4);
  assert.match(f.ui.statusCopy.value,/1 份可提问.*1 份准备中.*1 份处理失败/);assert.equal(f.ui.documentStatus({status:'ready',chunk_count:0}),'暂无可用片段');
  assert.deepEqual(emitted(f,'busy'),[true,false]);assert.deepEqual(emitted(f,'ready'),[0,1]);assert.deepEqual(emitted(f,'ask'),[]);
  f.ui.ask();assert.deepEqual(emitted(f,'ask'),[undefined]);assert.equal(f.calls.documents.length,1);
});

test('upload receipt keeps its actual queued status and requires explicit status checking before asking',async t=>{
  const f=mount(t);await f.ui.uploadFile();
  assert.equal(f.calls.upload.length,1);assert.equal(f.calls.upload[0][0],'#00001');assert.equal(f.calls.upload[0][1].isCurrent(),true);
  assert.equal(f.ui.documents.value[0].status,'queued');assert.equal(f.ui.checked.value,false);assert.match(f.ui.notice.value,/排队中/);assert.doesNotMatch(f.ui.notice.value,/已完成|可提问/);
  assert.equal(f.calls.documents.length,0);f.ui.ask();assert.equal(emitted(f,'ask').length,0);
  await f.ui.checkDocuments();f.ui.ask();assert.equal(emitted(f,'ask').length,1);
});

test('paste confirmation uploads the complete user original only after deliberate confirmation',async t=>{
  const f=mount(t);f.ui.togglePaste();f.ui.title.value='Own paper';f.ui.text.value='  Exact source\nwith linebreaks  ';
  assert.equal(f.calls.text.length,0);await f.ui.uploadPastedText();
  assert.equal(f.calls.text.length,1);assert.equal(f.calls.text[0][0],'#00001');assert.equal(f.calls.text[0][1].title,'Own paper');assert.equal(f.calls.text[0][1].text,'  Exact source\nwith linebreaks  ');
  assert.equal(typeof f.calls.text[0][1].isCurrent,'function');assert.equal(f.ui.title.value,'');assert.equal(f.ui.text.value,'');assert.equal(f.ui.pasteOpen.value,false);assert.equal(f.calls.documents.length,0);
});

test('empty, over-limit and invalid text is never silently truncated or submitted',async t=>{
  const f=mount(t);const cases=[['','Original'],['Paper','   '],['文'.repeat(121),'Original'],['Paper','🧠'.repeat(12001)],['Paper','bad\ud800'],['bad\0','Original']];
  for(const [title,text] of cases){f.ui.title.value=title;f.ui.text.value=text;assert.equal(f.ui.canUploadText.value,false);await f.ui.uploadPastedText();assert.equal(f.ui.title.value,title);assert.equal(f.ui.text.value,text);}
  assert.equal(f.calls.text.length,0);
  f.ui.title.value='文'.repeat(116);f.ui.text.value='🧠'.repeat(12000);assert.equal(f.ui.canUploadText.value,true);assert.equal(f.ui.textLength.value,12000);
});

test('source-name validation matches final TXT filenames and never repairs names without consent',async t=>{
  const f=mount(t);f.ui.text.value='Source';
  for(const name of [' Leading','Trailing ','a/b','a\\b','a\nb','a\tb','a\u007fb','.','..','文'.repeat(117),'🧠'.repeat(117),'x'.repeat(117)+'.txt']){
    f.ui.title.value=name;assert.equal(f.ui.validTitle.value,false,name);assert.equal(f.ui.canUploadText.value,false,name);await f.ui.uploadPastedText();assert.equal(f.ui.title.value,name);assert.match(f.ui.pasteError.value,/来源名称/);
  }
  assert.equal(f.calls.text.length,0);
  for(const name of ['Valid paper','文'.repeat(116),'🧠'.repeat(116),'x'.repeat(116)+'.txt','x'.repeat(116)+'.TXT']){f.ui.title.value=name;assert.equal(f.ui.validTitle.value,true,name);assert.equal(f.ui.canUploadText.value,true,name);assert.ok(f.ui.filenameLength.value<=120);}
});

test('dirty reports unsent raw text through edits, collapse and hide, then clears only when draft is empty',t=>{
  const f=mount(t);f.ui.title.value=' ';f.ui.text.value='Original';f.ui.title.value='';f.ui.togglePaste();f.ui.togglePaste();f.hooks.hide();
  assert.deepEqual(emitted(f,'dirty'),[true]);assert.equal(f.ui.text.value,'Original');f.ui.text.value='';assert.deepEqual(emitted(f,'dirty'),[true,false]);
  f.ui.text.value='Another';f.props.identity='new identity';assert.deepEqual(emitted(f,'dirty'),[true,false,true,false]);
});

test('successful upload clears dirty only after confirmation while failure retains it',async t=>{
  const f=mount(t);f.ui.title.value='Source';f.ui.text.value='Original';await f.ui.uploadPastedText();assert.deepEqual(emitted(f,'dirty'),[true,false]);
  const failed=mount(t,{uploadText:async()=>{throw new Error('Unknown outcome');}});failed.ui.title.value='Source';failed.ui.text.value='Original';await failed.ui.uploadPastedText();assert.deepEqual(emitted(failed,'dirty'),[true]);
});

test('temporary empty session preserves raw draft, but a different subsequent real session clears it',t=>{
  const f=mount(t);f.ui.title.value='Private source';f.ui.text.value='Private original';f.props.sessionId='';assert.equal(f.ui.text.value,'Private original');
  f.props.sessionId='#00001';assert.equal(f.ui.text.value,'Private original');f.props.sessionId='';f.props.sessionId='#00002';assert.equal(f.ui.text.value,'');assert.deepEqual(emitted(f,'dirty'),[true,false]);
});

test('cancelled and ambiguous upload results preserve the raw original and never claim readiness',async t=>{
  for(const result of [{cancelled:true},{}]){
    const f=mount(t,{uploadText:async()=>result});f.ui.title.value='My source';f.ui.text.value='Original';await f.ui.uploadPastedText();
    assert.equal(f.ui.title.value,'My source');assert.equal(f.ui.text.value,'Original');assert.equal(f.ui.checked.value,false);assert.equal(f.ui.readyCount.value,0);
    assert.equal(f.calls.documents.length,0);
  }
});

for(const boundary of ['hide','active','session','identity','mode','invite','disabled','uploads','unmount']) {
  test(`${boundary} invalidates an in-flight text upload and ignores its late receipt`,async t=>{
    const wait=deferred(),f=mount(t,{uploadText:()=>wait.promise});f.ui.pasteOpen.value=true;f.ui.title.value='Private title';f.ui.text.value='Private original';const operation=f.ui.uploadPastedText();
    assert.equal(f.ui.busy.value,true);const isCurrent=f.calls.text[0][1].isCurrent;
    if(boundary==='hide')f.hooks.hide();else if(boundary==='active')f.props.active=false;else if(boundary==='session')f.props.sessionId='#00002';else if(boundary==='identity')f.props.identity='account-b';else if(boundary==='mode')f.props.mode='demo';else if(boundary==='invite')f.props.invitePending=true;else if(boundary==='disabled')f.props.disabled=true;else if(boundary==='uploads')f.props.uploadsAllowed=false;else f.dispose();
    assert.equal(isCurrent(),false);assert.equal(f.ui.busy.value,false);wait.resolve({document:ready});await operation;
    assert.equal(f.ui.documents.value.length,0);assert.equal(f.ui.checked.value,false);assert.equal(f.ui.readyCount.value,0);assert.ok(!emitted(f,'ready').includes(1));
    if(['hide','active','disabled','uploads'].includes(boundary)){assert.equal(f.ui.text.value,'Private original');assert.match(f.ui.notice.value,/可能仍会完成/);}else assert.equal(f.ui.text.value,'');
    assert.deepEqual(emitted(f,'busy'),[true,false]);
  });
}

test('ordinary hide keeps a local unsubmitted paste draft but identity changes clear it',t=>{
  const f=mount(t);f.ui.togglePaste();f.ui.title.value='My title';f.ui.text.value='Never submitted';f.hooks.hide();f.props.active=false;
  assert.equal(f.ui.text.value,'Never submitted');assert.equal(f.calls.text.length,0);f.props.active=true;assert.equal(f.ui.text.value,'Never submitted');
  f.props.identity='someone-else';assert.equal(f.ui.text.value,'');assert.equal(f.ui.title.value,'');assert.equal(f.ui.pasteOpen.value,false);
});

test('late library reads cannot overwrite the new project and failed checks remove stale ready claims',async t=>{
  const wait=deferred();let number=0;const f=mount(t,{documents:async()=>{number++;if(number===1)return wait.promise;if(number===3)throw new Error('private-upstream');return[ready];}});
  const first=f.ui.checkDocuments();f.props.sessionId='#00002';await f.ui.checkDocuments();assert.equal(f.ui.readyCount.value,1);
  wait.resolve([pending]);await first;assert.equal(f.ui.documents.value[0].id,ready.id);
  await f.ui.checkDocuments();assert.equal(f.ui.checked.value,false);f.ui.ask();assert.deepEqual(emitted(f,'ask'),[]);assert.doesNotMatch(f.ui.error.value,/private-upstream/);assert.equal(emitted(f,'ready').at(-1),0);
});

test('busy lock prevents double submissions and overlap across upload, checks and paste actions',async t=>{
  const wait=deferred(),f=mount(t,{upload:()=>wait.promise});f.ui.title.value='T';f.ui.text.value='Source';const first=f.ui.uploadFile();
  await f.ui.uploadFile();await f.ui.uploadPastedText();await f.ui.checkDocuments();f.ui.togglePaste();f.ui.ask();f.ui.navigateLibrary();
  assert.equal(f.calls.upload.length,1);assert.equal(f.calls.text.length,0);assert.equal(f.calls.documents.length,0);assert.equal(f.ui.pasteOpen.value,false);assert.equal(emitted(f,'navigate').length,0);
  wait.resolve({cancelled:true});await first;assert.equal(f.ui.busy.value,false);
});

test('demo, missing research, inactive, invalid project, unauthenticated and disabled states cannot submit anything',async t=>{
  for(const props of [{mode:'demo'},{enabled:false},{active:false},{sessionId:''},{sessionId:'bad'},{invitePending:true},{disabled:true}]){
    const f=mount(t,{props});f.ui.title.value='Source';f.ui.text.value='Original';await f.ui.checkDocuments();await f.ui.uploadFile();await f.ui.uploadPastedText();f.ui.ask();
    assert.deepEqual(f.calls,{documents:[],upload:[],text:[]});assert.equal(emitted(f,'ask').length,0);
  }
  const f=mount(t,{props:{uploadsAllowed:false}});await f.ui.uploadFile();await f.ui.checkDocuments();assert.equal(f.calls.upload.length,0);assert.equal(f.calls.documents.length,1);
});

test('connection navigation is explicit and emits destinations without changing preferences',t=>{
  for(const [props,target] of [[{mode:'demo'},'settings'],[{invitePending:true,connectionKind:'invite'},'invite'],[{sessionId:''},'chat'],[{enabled:false,connectionKind:'invite'},'invite']]){
    const f=mount(t,{props});assert.deepEqual(emitted(f,'navigate'),[]);f.ui.navigateSetup();assert.deepEqual(emitted(f,'navigate'),[target]);
  }
});

test('errors preserve source drafts, expose only a fixed safe message and never auto retry',async t=>{
  const f=mount(t,{uploadText:async()=>{throw new Error('https://private.invalid?token=secret');}});f.ui.title.value='Original source';f.ui.text.value='Source still editable';await f.ui.uploadPastedText();
  assert.equal(f.ui.text.value,'Source still editable');assert.match(f.ui.error.value,/草稿已保留/);assert.doesNotMatch(f.ui.error.value,/private|secret|https/);assert.equal(f.calls.text.length,1);assert.equal(f.calls.documents.length,0);assert.equal(f.ui.busy.value,false);
});

test('keyboard actions reject IME, shortcuts and repeats and honor Enter and Space',async t=>{
  const f=mount(t);let prevented=0;
  for(const extra of [{isComposing:true},{keyCode:229},{ctrlKey:true},{metaKey:true},{altKey:true},{shiftKey:true},{repeat:true},{defaultPrevented:true}])f.ui.onKeydown({key:'Enter',...extra,preventDefault:()=>prevented++},'paste');
  assert.equal(f.ui.pasteOpen.value,false);assert.equal(prevented,0);
  f.ui.onKeydown({key:'Enter',preventDefault:()=>prevented++},'paste');assert.equal(f.ui.pasteOpen.value,true);
  f.ui.onKeydown({key:' ',preventDefault:()=>prevented++},'paste');assert.equal(f.ui.pasteOpen.value,false);assert.equal(prevented,2);assert.equal(f.calls.text.length,0);
});
