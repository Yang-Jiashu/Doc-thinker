import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const file = path => readFileSync(new URL('../' + path,import.meta.url));
const manifest = JSON.parse(file('src/manifest.json'));
const pages = JSON.parse(file('src/pages.json'));

test('native installation icons are wired, correctly sized opaque PNGs',() => {
  const check = (path,size) => {
    const png = file('src/' + path);
    assert.equal(png.toString('hex',0,8),'89504e470d0a1a0a');
    assert.equal(png.readUInt32BE(16),size); assert.equal(png.readUInt32BE(20),size);
    assert.equal(png[25],2,'native icons must use opaque RGB');
  };
  const icons = manifest['app-plus'].distribute.icons;
  for (const [key,size] of Object.entries({mdpi:48,hdpi:72,xhdpi:96,xxhdpi:144,xxxhdpi:192})) check(icons.android[key],size);
  check(icons.ios.appstore,1024);
  for (const [key,size] of Object.entries({'app@2x':120,'app@3x':180,'spotlight@2x':80,'spotlight@3x':120,'settings@2x':58,'settings@3x':87,'notification@2x':40,'notification@3x':60})) check(icons.ios.iphone[key],size);
  for (const [key,size] of Object.entries({app:76,'app@2x':152,'proapp@2x':167,spotlight:40,'spotlight@2x':80,settings:29,'settings@2x':58,notification:20,'notification@2x':40})) check(icons.ios.ipad[key],size);
});

test('all five navigation tabs have distinct source and active icons',() => {
  assert.equal(pages.tabBar.list.length,5);
  assert.deepEqual(pages.tabBar.list.map(tab => tab.text),['助手','资料','记忆','图谱','设置']);
  for (const tab of pages.tabBar.list) {
    assert.notEqual(tab.iconPath,tab.selectedIconPath);
    for (const path of [tab.iconPath,tab.selectedIconPath]) {
      const png = file('src/' + path); assert.equal(png.readUInt32BE(16),81); assert.equal(png.readUInt32BE(20),81);
    }
  }
});

test('brand comparison includes six editable SVG directions without external resources',() => {
  const catalog = JSON.parse(file('src/static/brand/catalog.json'));
  assert.equal(catalog.variants.length,6);
  for (const variant of catalog.variants) assert.match(file('src/static/brand/' + variant.id + '.svg').toString(),/viewBox="0 0 100 100"/);
  const showcase = file('public/brand/index.html').toString();
  assert.ok(showcase.includes('DocThinker'));
  assert.ok(!/(?:src|href)=["']https?:\/\//.test(showcase),'showcase must not load third-party resources');
  assert.ok(!showcase.includes('aria-pressed="undefined"'));
});
