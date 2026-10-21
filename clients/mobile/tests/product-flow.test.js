import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import * as Vue from 'vue';
import { productExperience, featureBoundary, openProductPage } from '../src/services/product-capabilities.js';

const root = new URL('../src/', import.meta.url);
const read = path => readFileSync(new URL(path, root), 'utf8');
const nodes = node => [node, ...(Array.isArray(node.children) ? node.children.flatMap(nodes) : [])];
const text = rendered => nodes(rendered).filter(node => typeof node.children === 'string').map(node => node.children).join('\n');
const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');

function mount(name, initialProps) {
  const filename = `components/${name}.vue`, source = read(filename);
  const { descriptor, errors } = parse(source, { filename });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: name });
  const template = compileTemplate({ source: descriptor.template.content, filename, id: name, compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(template.errors, []);
  const destinations = [];
  const scope = {
    ref: Vue.ref, computed: Vue.computed, watch: Vue.watch,
    productExperience, featureBoundary, openProductPage: page => destinations.push(page),
    uni: { reLaunch: options => destinations.push(options.url), switchTab: options => destinations.push(options.url) },
  };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const props = Vue.reactive(initialProps), effects = Vue.effectScope();
  const ui = effects.run(() => component.setup(props, { expose() {} }));
  const templateCode = template.code.replace(/import \{([^}]+)\} from "vue"/, (_, names) => `const {${names.replace(/\bas\b/g, ':')}} = Vue;`)
    .replace('export function render', 'return function render');
  const render = new Function('Vue', templateCode)(Vue), setup = Vue.proxyRefs(ui), cache = [];
  return { props, ui, destinations, render: () => render({}, cache, props, setup), dispose: () => effects.stop() };
}

test('each mode tells the truth about its scope and a connection does not assert readiness', () => {
  const demo = productExperience({ mode: 'demo' });
  assert.equal(demo.kind, 'demo'); assert.match(demo.summary, /本地示例/); assert.match(demo.boundary, /不调用模型/);
  assert.deepEqual(demo.primary, { label: '连接真实服务', page: 'settings' });
  assert.deepEqual(demo.secondary, { label: '查看功能入口', page: 'chat' });
  assert.match(demo.summary, /不能核对你的资料/);
  assert.match(demo.boundary, /不会自动连接或切换模式/);
  const privateService = productExperience({ mode: 'connected', baseUrl: 'https://private.example', accessToken: 'secret-never-display', useMemory: true });
  assert.equal(privateService.kind, 'private'); assert.match(privateService.status, /需后端确认/);
  assert.match(privateService.summary, /不等于后端功能已经就绪/);
  assert.ok(privateService.available.every(value => value.endsWith('入口')));
  assert.match(privateService.boundary, /不会验证模型、资料解析、记忆或语音/);
  assert.doesNotMatch(JSON.stringify(privateService), /secret-never-display|private\.example/);
  assert.equal(productExperience(null).kind, 'demo');
});

test('unconnected first use points to real setup, not a fixed sample or automatic mode change', () => {
  const preferences = Object.freeze({ mode: 'demo', connectionKind: 'private', baseUrl: 'https://private.example' });
  const experience = productExperience(preferences);
  assert.equal(experience.primary.page, 'settings');
  assert.equal(experience.secondary.page, 'chat');
  assert.ok(![experience.primary, experience.secondary].some(action => action.page === 'studio'));
  assert.match(experience.steps.join(' '), /研究服务开放后/);
  assert.match(experience.steps.join(' '), /自己的资料/);
  assert.doesNotMatch(experience.steps.join(' '), /自动|洞见体验|30 秒/);
  assert.equal(preferences.mode, 'demo');
  assert.equal(preferences.baseUrl, 'https://private.example');
});

test('invited users get only the supported scope and pending login overrides a connected setting', () => {
  const loggedIn = productExperience({ mode: 'connected', connectionKind: 'invite', invitePending: false });
  assert.equal(loggedIn.kind, 'invite'); assert.deepEqual(loggedIn.available, ['文字对话', '会话历史', '对话上下文开关']);
  assert.equal(loggedIn.primary.page, 'chat'); assert.equal(loggedIn.secondary.page, 'invite');
  assert.match(loggedIn.boundary, /聊天历史不是长期记忆/);
  assert.match(loggedIn.steps.join(' '), /非敏感内容/);
  const pending = productExperience({ mode: 'connected', connectionKind: 'invite', invitePending: true });
  assert.equal(pending.status, '等待登录'); assert.equal(pending.primary.page, 'invite'); assert.equal(pending.secondary, null);
});

test('unknown feature names cannot inherit capabilities and returned feature copy is isolated', () => {
  for (const feature of ['', 'invalid', '__proto__', 'constructor', undefined]) {
    assert.equal(featureBoundary(feature).name, '此功能');
  }
  for (const feature of ['documents', 'memory', 'graph']) {
    const copy = featureBoundary(feature); copy.title = 'caller mutation';
    assert.notEqual(featureBoundary(feature).title, 'caller mutation');
    assert.match(featureBoundary(feature).description, /内测版本|当前邀请内测/);
  }
});

test('product destinations are fixed internal routes, never arbitrary URLs', () => {
  const oldUni = globalThis.uni, calls = [];
  globalThis.uni = {
    reLaunch: args => calls.push(['reLaunch', args]),
    navigateTo: args => calls.push(['navigateTo', args]),
    switchTab: args => calls.push(['switchTab', args]),
  };
  try {
    for (const page of ['invite', 'studio', 'chat', 'documents', 'settings']) openProductPage(page);
    assert.deepEqual(calls, [
      ['reLaunch', { url: '/pages/invite/index' }], ['navigateTo', { url: '/pages/studio/index' }],
      ['switchTab', { url: '/pages/chat/index' }], ['switchTab', { url: '/pages/documents/index' }],
      ['switchTab', { url: '/pages/settings/index' }],
    ]);
    for (const page of ['https://external.example', '/pages/chat/index', '__proto__', undefined]) assert.equal(openProductPage(page), false);
    assert.equal(calls.length, 5);
  } finally { if (oldUni === undefined) delete globalThis.uni; else globalThis.uni = oldUni; }
});

test('guide starts collapsed; explicit buttons navigate and mode changes fold stale details', async () => {
  const f = mount('ProductGuide', { preferences: { mode: 'demo' } });
  assert.equal(f.ui.expanded.value, false); assert.deepEqual(f.destinations, []);
  let rendered = nodes(f.render());
  assert.ok(rendered.some(node => node.children === '连接真实服务'));
  assert.doesNotMatch(text(f.render()), /不读取你的文件/); // Progressive disclosure.
  rendered.find(node => node.type === 'button' && node.children === '连接真实服务').props.onTap();
  assert.deepEqual(f.destinations, ['settings']);
  rendered.find(node => node.type === 'button' && node.props.class.includes('guide-toggle')).props.onTap();
  assert.equal(f.ui.expanded.value, true); assert.match(text(f.render()), /不读取你的文件/);
  f.props.preferences = { mode: 'connected', connectionKind: 'invite', invitePending: true }; await Vue.nextTick();
  assert.equal(f.ui.expanded.value, false); assert.match(text(f.render()), /等待登录/);
  rendered = nodes(f.render());
  rendered.find(node => node.type === 'button' && node.children === '前往登录').props.onTap();
  assert.deepEqual(f.destinations, ['settings', 'invite']);
  f.ui.expanded.value = true; f.props.preferences.invitePending = false; await Vue.nextTick();
  assert.equal(f.ui.expanded.value, false); assert.match(text(f.render()), /邀请内测工作台/);
  f.dispose();
});

test('unopened feature panel returns pending users to login and active users to supported chat', () => {
  const pending = mount('CapabilityBoundary', { featureId: 'documents', invitePending: true });
  assert.match(text(pending.render()), /不会读取或上传你的文件/);
  assert.match(text(pending.render()), /不是你的资料丢失或上传失败/);
  let buttons = nodes(pending.render()).filter(node => node.type === 'button');
  assert.equal(buttons.length, 1); buttons[0].props.onTap(); assert.deepEqual(pending.destinations, ['invite']);
  pending.dispose();
  const active = mount('CapabilityBoundary', { featureId: 'memory', invitePending: false });
  assert.match(text(active.render()), /历史还在，长期记忆尚未开放/);
  buttons = nodes(active.render()).filter(node => node.type === 'button');
  assert.equal(buttons.length, 2); buttons[0].props.onTap(); buttons[1].props.onTap();
  assert.deepEqual(active.destinations, ['chat', 'settings']); active.dispose();
});

test('feature boundaries do not replace personal/demo flows or trigger automatic network/model work', () => {
  for (const page of ['documents', 'memory', 'graph']) {
    const source = read(`pages/${page}/index.vue`);
    assert.match(source, /<CapabilityBoundary v-(?:else-)?if="preferences\.connectionKind === 'invite'(?: && !researchEnabled)?(?: && !loading)?"/);
    {
      assert.match(source, /await research\.capabilities\(\{refresh:true\}\)/);
      assert.match(source, /if \(preferences\.value\.connectionKind === 'invite' && !researchEnabled\.value\) return;/);
    }
    assert.match(source, /<template v-else>/);
  }
  for (const path of ['services/product-capabilities.js', 'components/ProductGuide.vue', 'components/CapabilityBoundary.vue', 'components/ModeBanner.vue']) {
    assert.doesNotMatch(read(path), /\bfetch\s*\(|uni\.request|api\.|getStorage|localStorage|setStorage|requestPermissions/);
  }
});

test('mode banner exposes explicit guide/login navigation and settings can suppress the redundant guide', () => {
  const demo = mount('ModeBanner', { mode: 'demo', connectionKind: '', invitePending: false, showGuide: true });
  let buttons = nodes(demo.render()).filter(node => node.type === 'button');
  assert.equal(buttons.length, 1); buttons[0].props.onTap(); assert.deepEqual(demo.destinations, ['/pages/settings/index']);
  demo.props.showGuide = false; assert.equal(nodes(demo.render()).filter(node => node.type === 'button').length, 0);
  demo.dispose();
  const pending = mount('ModeBanner', { mode: 'connected', connectionKind: 'invite', invitePending: true, showGuide: false });
  buttons = nodes(pending.render()).filter(node => node.type === 'button');
  assert.equal(buttons.length, 1); assert.equal(buttons[0].children, '前往邀请登录');
  buttons[0].props.onTap(); assert.deepEqual(pending.destinations, ['/pages/invite/index']); pending.dispose();
});

test('guide, feature boundary and mode banner compile on H5, WeChat and native App with accessible actions', () => {
  for (const component of ['ProductGuide', 'CapabilityBoundary', 'ModeBanner']) {
    const source = read(`components/${component}.vue`);
    const descriptor = parse(source).descriptor;
    for (const platform of ['H5', 'MP_WEIXIN', 'APP_PLUS']) {
      const context = { H5: false, MP_WEIXIN: false, APP_PLUS: false, [platform]: true };
      const template = preprocess(descriptor.template.content, context, { type: 'html' });
      const script = preprocess(descriptor.scriptSetup.content, context, { type: 'js' });
      const parsed = parse(`<template>${template}</template><script setup>${script}</script>`);
      assert.deepEqual(parsed.errors, []);
      const compiled = compileScript(parsed.descriptor, { id: `${component}-${platform}` });
      assert.deepEqual(compileTemplate({ source: template, id: `${component}-${platform}`, compilerOptions: { bindingMetadata: compiled.bindings } }).errors, []);
      for (const button of nodes(parsed.descriptor.template.ast).filter(node => node.tag === 'button')) {
        assert.ok(button.props.some(prop => prop.type === 6 && prop.name === 'role' && prop.value?.content === 'button'));
      }
    }
    assert.match(source, /min-height:44px/);
  }
});
