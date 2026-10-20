import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileStyle} from '@vue/compiler-sfc';

const read=path=>readFileSync(new URL('../src/'+path,import.meta.url),'utf8');
const app=read('App.vue');
const {descriptor}=parse(app);

test('adaptive shell CSS compiles and uses the same breakpoint and width as its navigation',()=>{
  const css=compileStyle({source:descriptor.styles[0].content,filename:'App.vue',id:'adaptive-shell'});
  assert.deepEqual(css.errors,[]);
  const rail=read('components/DesktopNavigation.vue');
  for(const source of [app,rail]) {
    assert.match(source,/@supports selector\(:has\(\*\)\)/);
    assert.match(source,/@media\s*\(min-width:1100px\)/);
  }
  assert.match(app,/uni-page-wrapper:has\(\.desktop-navigation\)\s*\{[^}]*padding-left:208px/);
  assert.match(rail,/width:208px/);
  assert.match(app,/uni-page:has\(\.desktop-navigation\)\s*\{[^}]*--window-bottom:0px/);
  assert.match(app,/uni-app:has\(\.desktop-navigation\) uni-tabbar\s*\{[^}]*display:none/);
});

test('phone shell retains original tabs and readable input with no breakpoint JS or network work',()=>{
  const pages=JSON.parse(read('pages.json'));
  assert.equal(pages.tabBar.list.length,5);
  assert.deepEqual(pages.tabBar.list.map(item=>item.text),['助手','资料','记忆','图谱','设置']);
  assert.match(app,/@media\s*\(max-width:759px\)\s*\{\s*uni-page-body \.mobile-page input, uni-page-body \.mobile-page textarea\s*\{ font-size:16px/);
  assert.doesNotMatch(descriptor.script.content,/resize|matchMedia|fetch|setInterval/);
});
