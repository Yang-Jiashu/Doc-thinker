import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { parse, compileTemplate } from '@vue/compiler-sfc';
import postcss from 'postcss';
import selectorParser from 'postcss-selector-parser';

const require = createRequire(import.meta.url);
const { preprocess } = require('@dcloudio/uni-cli-shared/lib/preprocess');
const read = path => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8');

function elements(node) {
  return [ ...(node.type === 1 ? [node] : []), ...(node.children || []).flatMap(elements) ];
}
function attribute(node, name) { return node.props.find(prop => prop.type === 6 && prop.name === name); }
function binding(node, name) { return node.props.find(prop => prop.type === 7 && prop.name === 'bind' && prop.arg?.content === name); }

test('WeChat omits unsupported app permission but preserves runtime and native microphone authorization', () => {
  const manifest = JSON.parse(read('manifest.json'));
  assert.equal(manifest['mp-weixin'].permission?.['scope.record'], undefined);
  assert.equal(manifest['mp-weixin'].setting.urlCheck, true);
  const native = manifest['app-plus'].distribute;
  assert.ok(native.android.permissions.some(value => value.includes('android.permission.RECORD_AUDIO')));
  assert.ok(native.ios.privacyDescription.NSMicrophoneUsageDescription);
  // Runtime behavior (including rejected permissions) is exercised in wechat-privacy.test.js.
  const privacy = read('services/wechat-privacy.js');
  assert.match(privacy, /'authorize'[\s\S]*?scope:\s*'scope\.record'/);
  assert.match(privacy, /requirePrivacyAuthorize/);
  assert.match(read('services/voice.js'), /ensureWeChatRecordingPermission/);
});

test('WorkspaceBar retains disabled behavior without forbidden component CSS selectors', () => {
  const { descriptor, errors } = parse(read('components/WorkspaceBar.vue'));
  assert.deepEqual(errors, []);
  const button = elements(descriptor.template.ast).find(node => node.tag === 'button');
  assert.equal(binding(button, 'disabled').exp.content, 'disabled');
  const classes = new Function('disabled', `return (${binding(button, 'class').exp.content});`);
  assert.deepEqual(classes(false), { 'workspace-create-disabled': false });
  assert.deepEqual(classes(true), { 'workspace-create-disabled': true });
  let disabledRule = false;
  for (const style of descriptor.styles) postcss.parse(style.content).walkRules(rule => {
    selectorParser(selectors => selectors.walk(node => {
      assert.ok(!['tag', 'id', 'attribute'].includes(node.type), `Unsupported component selector: ${rule.selector}`);
    })).processSync(rule.selector);
    if (rule.selector === '.workspace-create-disabled') {
      disabledRule = true;
      assert.ok(rule.nodes.some(node => node.prop === 'opacity' && Number(node.value) === 0.45));
    }
  });
  assert.equal(disabledRule, true);
});

test('all selectable content uses WeChat user-select while H5 and App retain their original text behavior', () => {
  const pages = {
    chat: ['message-content'],
    graph: ['node-description', 'relation-description', 'evidence-content'],
    memory: ['memory-summary', 'memory-question-content'],
  };
  for (const platform of ['MP_WEIXIN', 'H5', 'APP_PLUS']) {
    const context = { MP_WEIXIN: false, H5: false, APP_PLUS: false, [platform]: true };
    for (const [page, classes] of Object.entries(pages)) {
      const filename = `pages/${page}/index.vue`;
      // Use the same DCloud HTML preprocessor as the build, without creating dist output.
      const raw = parse(read(filename)).descriptor.template.content;
      const template = preprocess(raw, context, { type: 'html' });
      assert.doesNotMatch(template, /#ifn?def|#endif/);
      const { descriptor, errors } = parse(`<template>${template}</template>`);
      assert.deepEqual(errors, []);
      const compiled = compileTemplate({ source: template, filename, id: 'compat' });
      assert.deepEqual(compiled.errors, []);
      const texts = elements(descriptor.template.ast).filter(node => node.tag === 'text');
      const expected = platform === 'MP_WEIXIN' ? 'user-select' : 'selectable';
      const forbidden = platform === 'MP_WEIXIN' ? 'selectable' : 'user-select';
      assert.equal(texts.filter(node => attribute(node, forbidden)).length, 0, `${platform}: ${page}`);
      for (const className of classes) {
        const matches = texts.filter(node => attribute(node, 'class')?.value?.content === className);
        assert.equal(matches.length, 1, `${platform}: ${className} renders once`);
        assert.ok(attribute(matches[0], expected), `${platform}: ${className} remains selectable`);
      }
    }
  }
});
