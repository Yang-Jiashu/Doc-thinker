import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import * as Vue from 'vue';
import {
  DESKTOP_NAVIGATION_WIDTH, DESKTOP_NAVIGATION_MIN_WIDTH,
  desktopNavigationItems, desktopNavigationRoute, desktopNavigationTarget, switchDesktopPage,
} from '../src/services/adaptive-navigation.js';

const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');
const read = path => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8');
const source = read('components/DesktopNavigation.vue');
const processFor = (value, platform, type = 'html') => preprocess(value,
  { H5: false, MP_WEIXIN: false, APP_PLUS: false, [platform]: true }, { type });
const nodes = node => [node, ...(Array.isArray(node.children) ? node.children.flatMap(nodes) : [])];

function mount(hash = '#/pages/chat/index') {
  const { descriptor, errors } = parse(source, { filename: 'DesktopNavigation.vue' });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: 'desktop-navigation-test' });
  const template = compileTemplate({
    source: processFor(descriptor.template.content, 'H5'), filename: 'DesktopNavigation.vue', id: 'desktop-navigation-test',
    compilerOptions: { bindingMetadata: script.bindings, isCustomElement: tag => ['nav', 'div', 'span', 'p', 'img'].includes(tag) },
  });
  assert.deepEqual(template.errors, []);
  const hooks = {}, listeners = new Map(), calls = [];
  const window = {
    location: { hash },
    addEventListener: (event, fn) => listeners.set(event, fn),
    removeEventListener: (event, fn) => { if (listeners.get(event) === fn) listeners.delete(event); },
  };
  const scope = {
    h: Vue.h, ref: Vue.ref, onMounted: fn => { hooks.mount = fn; }, onUnmounted: fn => { hooks.unmount = fn; },
    onShow: fn => { hooks.show = fn; }, onHide: fn => { hooks.hide = fn; },
    desktopNavigationItems, desktopNavigationRoute, switchDesktopPage, window,
    getCurrentPages: () => [{ route: 'pages/chat/index' }],
    uni: { switchTab: options => calls.push(options) },
  };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const ui = component.setup({}, { expose() {} });
  const executableTemplate = template.code.replace(/import \{([^}]+)\} from "vue"/, (_, names) => `const {${names.replace(/\bas\b/g, ':')}} = Vue;`)
    .replace('export function render', 'return function render');
  const render = new Function('Vue', executableTemplate)(Vue), setup = Vue.proxyRefs(ui), cache = [];
  const renderNodes = () => nodes(render({}, cache, {}, setup));
  return {
    ui, hooks, listeners, calls, window, renderNodes,
    nativeButtons: () => renderNodes().filter(node => node.type === ui.NativeButton)
      .map(node => ui.NativeButton({}, { attrs: node.props, slots: node.children })),
  };
}

test('desktop destinations match exactly the existing five tab pages and bundled icon assets', () => {
  const config = JSON.parse(read('pages.json'));
  assert.deepEqual(desktopNavigationItems.map(item => item.path), config.tabBar.list.map(item => '/' + item.pagePath));
  assert.ok(Object.isFrozen(desktopNavigationItems));
  assert.ok(desktopNavigationItems.every(item => Object.isFrozen(item)));
  for (const item of desktopNavigationItems) {
    assert.equal(desktopNavigationTarget(item.id), item.path);
    assert.ok(readFileSync(new URL(`../src${item.icon}`, import.meta.url)).length > 0);
    assert.ok(readFileSync(new URL(`../src${item.activeIcon}`, import.meta.url)).length > 0);
  }
});

test('route normalization accepts only known internal paths and never unknown pages or external URLs', () => {
  for (const route of ['/pages/chat/index', 'pages/chat/index', '#/pages/chat/index', '#/pages/chat/index?entry=docs', '#/', '/']) {
    assert.equal(desktopNavigationRoute(route), '/pages/chat/index');
  }
  for (const value of ['', null, {}, undefined, '__proto__', 'constructor', 'toString', '/pages/invite/index', '/pages/studio/index',
    'https://evil.example/pages/chat/index', '//evil.example/pages/chat/index', 'javascript:alert(1)', '#/pages/chat/index#other',
    '#/pages/chat/index/../settings/index', '/pages%2Fchat/index', '/pages\\chat/index', '/pages/chat/index\n', 'x'.repeat(2049)]) {
    assert.equal(desktopNavigationRoute(value), '', String(value));
    assert.equal(desktopNavigationTarget(value), '', String(value));
  }
});

test('fixed navigation is explicit, no-op on current route, and never accepts an arbitrary URL', () => {
  const calls = [], switchTab = value => calls.push(value), success = () => {};
  assert.equal(switchDesktopPage('chat', '/pages/chat/index', switchTab), false);
  for (const invalid of ['__proto__', 'constructor', '/pages/documents/index', 'https://evil.example', null, {}]) {
    assert.equal(switchDesktopPage(invalid, '/pages/chat/index', switchTab), false);
  }
  for (const unknown of ['', '/pages/invite/index', '/pages/studio/index', '/unrecognized']) {
    assert.equal(switchDesktopPage('chat', unknown, switchTab), false);
  }
  assert.equal(calls.length, 0);
  assert.equal(switchDesktopPage('documents', '/pages/chat/index', switchTab, { success }), true);
  assert.deepEqual(calls, [{ url: '/pages/documents/index', success, fail: undefined, complete: undefined }]);
});

test('actual compiled component renders native keyboard buttons and exactly one aria-current', () => {
  const f = mount();
  assert.equal(f.nativeButtons().length, 0);
  f.hooks.mount();
  assert.equal(f.calls.length, 0);
  const buttons = f.nativeButtons();
  assert.equal(buttons.length, 5);
  assert.ok(buttons.every(button => button.type === 'button' && button.props.type === 'button'));
  assert.equal(buttons.filter(button => button.props['aria-current'] === 'page').length, 1);
  assert.equal(buttons[0].props['aria-current'], 'page');
  assert.ok(buttons.every(button => typeof button.props.onClick === 'function' && !button.props.onKeydown));
  buttons[0].props.onClick(); assert.equal(f.calls.length, 0);
  buttons[1].props.onClick(); assert.equal(f.calls[0].url, '/pages/documents/index');
  assert.equal(f.ui.navigating.value, true);
  f.nativeButtons()[2].props.onClick(); assert.equal(f.calls.length, 1);
  f.window.location.hash = '#/pages/documents/index';
  f.calls[0].success(); f.calls[0].complete();
  assert.equal(f.ui.navigating.value, false);
  assert.equal(f.nativeButtons()[1].props['aria-current'], 'page');
  f.hooks.unmount(); assert.equal(f.listeners.size, 0);
});

test('browser back and unknown/login routes update current state without triggering navigation', () => {
  const f = mount(); f.hooks.mount();
  f.window.location.hash = '#/pages/memory/index'; f.listeners.get('hashchange')();
  assert.equal(f.nativeButtons()[2].props['aria-current'], 'page');
  f.window.location.hash = '#/pages/settings/index'; f.listeners.get('popstate')();
  assert.equal(f.nativeButtons()[4].props['aria-current'], 'page');
  for (const hash of ['#/pages/invite/index', '#/pages/studio/index', '#/unknown']) {
    f.window.location.hash = hash; f.listeners.get('hashchange')();
    assert.equal(f.nativeButtons().length, 0);
    f.ui.navigate('chat'); assert.equal(f.calls.length, 0);
  }
  f.hooks.unmount();
});

test('failed navigation is recoverable and late callbacks cannot update an unmounted component', () => {
  const f = mount(); f.hooks.mount();
  f.ui.navigate('documents'); f.calls[0].fail(); f.calls[0].complete();
  assert.match(f.ui.navigationError.value, /没有切换成功/);
  assert.equal(f.ui.navigating.value, false);
  f.ui.navigate('settings'); assert.equal(f.calls.length, 2);
  f.hooks.unmount(); f.calls[1].fail(); f.calls[1].complete(); f.calls[1].success();
  assert.equal(f.ui.navigationError.value, '');
  f.ui.navigate('documents'); assert.equal(f.calls.length, 2);
});

test('cached hidden pages remove the rail from DOM and do not hide another page bottom navigation', () => {
  const f = mount(); f.hooks.mount();
  assert.equal(f.nativeButtons().length, 5);
  f.hooks.hide();
  assert.equal(f.nativeButtons().length, 0);
  f.window.location.hash = '#/pages/settings/index'; f.listeners.get('hashchange')();
  assert.equal(f.nativeButtons().length, 0);
  f.ui.navigate('documents'); assert.equal(f.calls.length, 0);
  f.hooks.show(); assert.equal(f.nativeButtons().length, 5);
  assert.equal(f.nativeButtons()[4].props['aria-current'], 'page');
  f.hooks.unmount();
});

test('late navigation failure on a hidden page stays silent and releases the old navigation lock', () => {
  const f = mount(''); f.hooks.mount();
  assert.equal(f.ui.currentRoute.value, '/pages/chat/index');
  f.nativeButtons()[1].props.onClick();
  f.hooks.hide(); f.calls[0].fail(); f.calls[0].complete();
  assert.equal(f.ui.navigationError.value, ''); assert.equal(f.ui.navigating.value, false);
  f.calls[0].success(); assert.equal(f.nativeButtons().length, 0); assert.equal(f.calls.length, 1);
  f.hooks.show(); f.nativeButtons()[4].props.onClick(); assert.equal(f.calls.length, 2);
  f.hooks.unmount();
});

test('WeChat and App compilation remove desktop markup, imports and DOM hooks entirely', () => {
  for (const platform of ['MP_WEIXIN', 'APP_PLUS']) {
    const { descriptor } = parse(source);
    assert.equal(processFor(descriptor.template.content, platform).trim(), '');
    assert.equal(processFor(descriptor.scriptSetup.content, platform, 'js').trim(), '');
    const brand = parse(read('components/BrandHeader.vue')).descriptor;
    const template = processFor(brand.template.content, platform);
    const script = processFor(brand.scriptSetup.content, platform, 'js');
    assert.doesNotMatch(template + script, /DesktopNavigation|adaptive-navigation|hashchange|window\./);
    assert.deepEqual(compileTemplate({ source: template, filename: 'BrandHeader.vue', id: platform }).errors, []);
  }
});

test('mobile and unsupported desktop browsers retain their original bottom navigation with no focusable rail', () => {
  assert.equal(DESKTOP_NAVIGATION_WIDTH, 208); assert.equal(DESKTOP_NAVIGATION_MIN_WIDTH, 1100);
  const css = parse(source).descriptor.styles[0].content;
  assert.match(css, /\.desktop-navigation\s*\{\s*display:none;\s*\}/);
  assert.match(css, /@supports selector\(:has\(\*\)\)/);
  assert.match(css, /@media \(min-width:1100px\)/);
  assert.match(css, /width:208px/);
  assert.match(css, /:focus-visible/); assert.match(css, /prefers-reduced-motion/);
  for (const value of [source, read('services/adaptive-navigation.js')]) {
    assert.doesNotMatch(value, /fetch\s*\(|uni\.request|api\.|getStorage|localStorage|setStorage|setPreferences|ask\(/);
  }
});
