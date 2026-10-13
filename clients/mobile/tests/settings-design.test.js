import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileScript, compileTemplate, compileStyle } from '@vue/compiler-sfc';
import * as Vue from 'vue';

const filename = 'pages/settings/index.vue';
const source = readFileSync(new URL('../src/pages/settings/index.vue', import.meta.url), 'utf8');
const nodes = node => [node, ...(Array.isArray(node.children) ? node.children.flatMap(nodes) : [])];
const text = node => nodes(node).filter(value => value.type !== Vue.Comment && typeof value.children === 'string').map(value => value.children).join('\n');
const button = (rendered, label) => nodes(rendered).find(value => value.type === 'button' && text(value).trim() === label);
const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');

/** Render the actual settings page offline, retaining its watchers and lifecycle hooks. */
function mount(initial = {}) {
  const { descriptor, errors } = parse(source, { filename });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: 'settings-design' });
  const template = compileTemplate({ source: descriptor.template.content, filename, id: 'settings-design', compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(template.errors, []);
  let preferences = { mode:'demo', baseUrl:'', accessToken:'', useMemory:true, useContext:true, useCache:true, useEvolution:false, ...initial };
  const hooks = {}, calls = { save:0, connection:0, navigation:0 };
  const scope = {
    ref:Vue.ref, computed:Vue.computed, watch:Vue.watch,
    onShow:callback => { hooks.show = callback; }, onHide:callback => { hooks.hide = callback; }, onUnload:callback => { hooks.unload = callback; },
    api:{checkConnection:async () => { calls.connection++; return {connected:true}; }}, cancelConnectionCheck:() => {},
    getPreferences:() => ({...preferences}), savePreferences:async value => { calls.save++; preferences = {...value}; }, setActiveSession:() => {},
    uni:{reLaunch:() => { calls.navigation++; }},
    BrandHeader:{name:'BrandHeader'}, ModeBanner:{name:'ModeBanner'}, ProductGuide:{name:'ProductGuide'}, InstallGuide:{name:'InstallGuide'},
  };
  const executable = script.content.replace(/^import[^;]+;\s*$/gm, '').replace('export default', 'return');
  const component = new Function(...Object.keys(scope), executable)(...Object.values(scope));
  const effects = Vue.effectScope();
  const ui = effects.run(() => component.setup({}, {expose(){}}));
  const templateCode = template.code.replace(/import \{([^}]+)\} from "vue"/, (_, names) => `const {${names.replace(/\bas\b/g, ':')}} = Vue;`).replace('export function render', 'return function render');
  const render = new Function('Vue', templateCode)(Vue), setup = Vue.proxyRefs(ui), cache = [];
  hooks.show();
  return {ui, calls, render:() => render({}, cache, {}, setup), dispose:() => {hooks.hide(); effects.stop();}};
}

test('settings starts with the experience entry rather than all configuration panels', () => {
  const f = mount(), rendered = f.render();
  assert.equal(f.ui.activeSection.value, 'start');
  assert.ok(nodes(rendered).some(value => value.type?.name === 'ProductGuide'));
  assert.match(text(rendered), /少一点配置，多一点开始/);
  assert.doesNotMatch(text(rendered), /API 服务地址|回答与对比测试|先说清楚，才放心使用|添加到手机桌面/);
  assert.equal(f.calls.save, 0); assert.equal(f.calls.connection, 0); assert.equal(f.calls.navigation, 0);
  f.dispose();
});

test('four manual sections expose only the relevant panel without saving, navigation or connection calls', () => {
  const f = mount();
  button(f.render(), '连接').props.onTap();
  assert.equal(f.ui.activeSection.value, 'connection'); assert.match(text(f.render()), /选择使用方式/);
  assert.doesNotMatch(text(f.render()), /回答与对比测试|先说清楚，才放心使用/);
  button(f.render(), '回答').props.onTap();
  assert.match(text(f.render()), /回答与对比测试/); assert.doesNotMatch(text(f.render()), /选择使用方式/);
  button(f.render(), '隐私').props.onTap();
  assert.match(text(f.render()), /先说清楚，才放心使用/); assert.doesNotMatch(text(f.render()), /回答与对比测试/);
  const privacy = nodes(f.render()).find(value => value.type === 'button' && value.props?.class.includes('privacy-toggle'));
  assert.equal(privacy.props['aria-expanded'], false); privacy.props.onTap();
  assert.match(text(f.render()), /关闭记忆不等于无痕/); assert.match(text(f.render()), /删除会话不等于彻底删除/);
  button(f.render(), '开始').props.onTap(); assert.equal(f.ui.activeSection.value, 'start');
  assert.deepEqual(f.calls, {save:0,connection:0,navigation:0}); f.dispose();
});

test('a manual section switch preserves unsaved form values and the explicit save action', async () => {
  const f = mount();
  button(f.render(), '回答').props.onTap();
  f.ui.form.value.useContext = false;
  button(f.render(), '隐私').props.onTap();
  assert.equal(f.ui.dirty.value, true); assert.equal(f.ui.form.value.useContext, false);
  assert.match(text(f.render()), /切换分区不会丢失/);
  await button(f.render(), '保存我的设置').props.onTap(); await Vue.nextTick();
  assert.equal(f.calls.save, 1); assert.equal(f.calls.connection, 0); assert.equal(f.ui.dirty.value, false);
  button(f.render(), '回答').props.onTap(); assert.equal(f.ui.form.value.useContext, false);
  assert.equal(button(f.render(), '设置已保存').props.disabled, true); f.dispose();
});

test('invite account and unsupported experiments remain separated from private service credentials', () => {
  const f = mount({mode:'connected',connectionKind:'invite',inviteUser:{username:'Synthetic preview user'},useMemory:false,useCache:false,useEvolution:false});
  button(f.render(), '账号').props.onTap();
  assert.match(text(f.render()), /Synthetic preview user/);
  assert.doesNotMatch(text(f.render()), /API 服务地址|私人测试访问码|选择使用方式/);
  button(f.render(), '回答').props.onTap();
  const switches = nodes(f.render()).filter(value => value.type === 'switch');
  assert.equal(switches.length, 4);
  assert.deepEqual(switches.map(value => value.props.disabled === true), [true,false,true,true]);
  assert.match(text(f.render()), /尚未开放长期记忆|尚未开放自进化检索/);
  assert.deepEqual(f.calls, {save:0,connection:0,navigation:0}); f.dispose();
});

test('section buttons provide pressed state and local styles preserve touch targets, motion preference and ink/lime identity', () => {
  const f = mount();
  const navigation = nodes(f.render()).filter(value => value.type === 'button' && value.props?.class.includes('settings-nav-item'));
  assert.equal(navigation.length, 4); assert.deepEqual(navigation.map(value => value.props['aria-pressed']), [true,false,false,false]);
  assert.ok(navigation.every(value => value.props.role === 'button' && value.props.tabindex === '0'));
  const { descriptor } = parse(source, {filename});
  assert.equal(descriptor.styles.at(-1).scoped, true);
  const css = descriptor.styles.at(-1).content;
  assert.match(css, /#f6f5f0/); assert.match(css, /#1c2723/); assert.match(css, /#d6fa75/);
  assert.match(css, /\.settings-nav-item[^}]+min-height:44px/); assert.match(css, /prefers-reduced-motion:reduce/);
  assert.doesNotMatch(css, /\.mobile-page\s*\{|\.primary-button\s*\{/);
  f.dispose();
});

test('Enter and Space activate the rendered section buttons without automatic save, navigation or service checks', () => {
  const f = mount(); let prevented = 0;
  const event = key => ({key, preventDefault:() => {prevented++;}});
  const connection = button(f.render(), '连接');
  connection.props.onKeydown(event('ArrowRight'));
  connection.props.onKeydown({...event('Enter'),ctrlKey:true});
  connection.props.onKeydown({...event(' '),altKey:true});
  connection.props.onKeydown({...event('Enter'),metaKey:true});
  connection.props.onKeydown({...event('Enter'),repeat:true});
  assert.equal(f.ui.activeSection.value, 'start'); assert.equal(prevented, 0);
  connection.props.onKeydown(event('Enter'));
  assert.equal(f.ui.activeSection.value, 'connection'); assert.equal(prevented, 1);
  button(f.render(), '回答').props.onKeydown(event(' '));
  assert.equal(f.ui.activeSection.value, 'thinking'); assert.equal(prevented, 2);
  const navigation = nodes(f.render()).filter(value => value.type === 'button' && value.props?.class.includes('settings-nav-item'));
  assert.deepEqual(navigation.map(value => value.props['aria-pressed']), [false,false,true,false]);
  f.ui.activateSection(event('Enter'),'not-a-section'); assert.equal(f.ui.activeSection.value, 'thinking');
  button(f.render(), '开始').props.onKeydown(event('Enter'));
  const adjustment = nodes(f.render()).find(value => value.type === 'button' && text(value).includes('调整回答方式'));
  assert.equal(adjustment.props.role, 'button'); assert.equal(adjustment.props.tabindex, '0');
  adjustment.props.onKeydown(event(' ')); assert.equal(f.ui.activeSection.value, 'thinking');
  assert.deepEqual(f.calls, {save:0,connection:0,navigation:0}); f.dispose();
});

test('the settings sections compile on H5, WeChat and App without platform-only navigation dependencies', () => {
  for (const platform of ['H5','MP-WEIXIN','APP-PLUS']) {
    const built = preprocess(source, {[platform]:true}, {type:'html'});
    const {descriptor, errors} = parse(built, {filename}); assert.deepEqual(errors, []);
    const script = compileScript(descriptor, {id:platform});
    const template = compileTemplate({source:descriptor.template.content,filename,id:platform,compilerOptions:{bindingMetadata:script.bindings}});
    assert.deepEqual(template.errors, []);
    const localStyle = descriptor.styles.at(-1);
    const style = compileStyle({source:localStyle.content,filename,id:platform,scoped:localStyle.scoped});
    assert.deepEqual(style.errors, []);
  }
});
