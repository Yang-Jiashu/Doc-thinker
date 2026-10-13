import test from 'node:test';
import assert from 'node:assert/strict';
import { wechatNavigationLayout } from '../src/services/navigation-layout.js';

test('custom WeChat navigation aligns to safe area and keeps capsule clear', () => {
  const layout = wechatNavigationLayout({windowWidth:390,statusBarHeight:47}, {left:293,top:55,height:32});
  assert.deepEqual(layout, {header:{paddingTop:'47px'},row:{minHeight:'48px',paddingRight:'87px'}});
});

test('missing and invalid capsule geometry falls back without negative sizes', () => {
  assert.deepEqual(wechatNavigationLayout(), {header:{paddingTop:'24px'},row:{minHeight:'48px',paddingRight:'96px'}});
  const layout = wechatNavigationLayout({windowWidth:320,statusBarHeight:20}, {left:0,top:0,height:0});
  assert.equal(layout.header.paddingTop,'20px'); assert.equal(layout.row.paddingRight,'96px');
});
