import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildWeixin, finalizeWeixinBuild, GUEST_APP_ID, MAX_MAIN_PACKAGE_BYTES, resolveWeixinAppId } from '../scripts/build-weixin.mjs';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

async function fixture(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'docthinker-weixin-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const projectRoot = path.join(directory, 'clients/mobile');
  const output = path.join(projectRoot, 'dist/build/mp-weixin');
  await mkdir(path.join(projectRoot, 'src'), { recursive: true });
  await mkdir(path.join(directory, 'docs'), { recursive: true });
  await mkdir(path.join(output, 'pages/chat'), { recursive: true });
  await mkdir(path.join(output, 'static/brand'), { recursive: true });
  for (const name of ['project.config.json', 'sitemap.json']) {
    await copyFile(path.join(project, 'src', name), path.join(projectRoot, 'src', name));
  }
  await writeFile(path.join(directory, 'docs/WECHAT_MINIPROGRAM.md'), '# Test import guide\nGuest simulator only.');
  await writeFile(path.join(projectRoot, 'src/manifest.json'), '{"mp-weixin":{"appid":""}}');
  await writeFile(path.join(output, 'app.js'), 'App({});');
  await writeFile(path.join(output, 'app.json'), JSON.stringify({
    pages: ['pages/chat/index'],
    tabBar: { list: [{ pagePath: 'pages/chat/index', iconPath: 'static/brand/tab-chat.png', selectedIconPath: 'static/brand/tab-chat-active.png' }] },
  }));
  await writeFile(path.join(output, 'project.config.json'), JSON.stringify({
    appid: 'old-appid', setting: { urlCheck: false, uploadWithSourceMap: true, es6: true }, libVersion: '',
  }));
  await writeFile(path.join(output, 'pages/chat/index.js'), 'const logo = "/static/brand/logo-64.png";');
  await writeFile(path.join(output, 'pages/chat/index.json'), '{}');
  await writeFile(path.join(output, 'pages/chat/index.wxml'), '<image src="/static/brand/logo-32.png"/>');
  for (const name of ['tab-chat.png', 'tab-chat-active.png', 'logo-64.png', 'logo-32.png', 'icon-1024.png', 'logo.svg', '06-memory-lattice.svg', 'catalog.json']) {
    await writeFile(path.join(output, 'static/brand', name), 'fixture ' + name);
  }
  return { directory, projectRoot, output };
}

test('guest identity is explicit and real AppID validation accepts only exact lowercase hex format', () => {
  assert.equal(resolveWeixinAppId(), GUEST_APP_ID);
  assert.equal(resolveWeixinAppId(''), 'touristappid');
  assert.equal(resolveWeixinAppId('wx1234567890abcdef'), 'wx1234567890abcdef');
  for (const invalid of ['touristappid', ' ', ' wx1234567890abcdef', 'wx1234567890ABCDEF', 'wx1234567890abcde', 'wx1234567890abcdef0', 'wx1234567890abcdeg', '__UNI__abcd', null, 123]) {
    assert.throws(() => resolveWeixinAppId(invalid), /DOCTHINKER_WEIXIN_APPID/);
  }
});

test('importable guest build enforces domain validation, preserves pages, installs sitemap and guide', async t => {
  const f = await fixture(t);
  const result = await finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot });
  const config = JSON.parse(await readFile(path.join(f.output, 'project.config.json'), 'utf8'));
  assert.equal(config.appid, 'touristappid'); assert.equal(config.compileType, 'miniprogram');
  assert.equal(config.miniprogramRoot, './'); assert.equal(config.setting.urlCheck, true);
  assert.equal(config.setting.uploadWithSourceMap, false); assert.equal(config.setting.es6, true);
  assert.equal(result.mode, 'guest'); assert.equal(result.measurement, 'uncompressed-all-output-files');
  const app = JSON.parse(await readFile(path.join(f.output, 'app.json'), 'utf8'));
  assert.deepEqual(app.pages, ['pages/chat/index']); assert.equal(app.sitemapLocation, 'sitemap.json');
  const sitemap = JSON.parse(await readFile(path.join(f.output, app.sitemapLocation), 'utf8'));
  assert.deepEqual(sitemap.rules, [{ action: 'disallow', page: '*' }]);
  assert.match(await readFile(path.join(f.output, 'README.md'), 'utf8'), /Guest simulator/);
  assert.deepEqual(config.packOptions.ignore, [{ type: 'file', value: 'README.md' }]);
});

test('unused brand variants are removed from output only and all compiled image references survive', async t => {
  const f = await fixture(t);
  await mkdir(path.join(f.projectRoot, 'src/static/brand'), { recursive: true });
  await writeFile(path.join(f.projectRoot, 'src/static/brand/icon-1024.png'), 'source asset retained');
  const result = await finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot });
  assert.deepEqual(result.brand.kept, ['static/brand/logo-32.png', 'static/brand/logo-64.png', 'static/brand/tab-chat-active.png', 'static/brand/tab-chat.png']);
  assert.equal(result.brand.removed.length, 4);
  assert.deepEqual((await readdir(path.join(f.output, 'static/brand'))).sort(), result.brand.kept.map(name => path.basename(name)));
  assert.equal(await readFile(path.join(f.projectRoot, 'src/static/brand/icon-1024.png'), 'utf8'), 'source asset retained');
});

test('missing or dynamically constructed brand paths fail before unused assets are deleted', async t => {
  const f = await fixture(t);
  await writeFile(path.join(f.output, 'pages/chat/index.js'), 'const logo = "/static/brand/missing.png";');
  await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot }), /Referenced brand asset is missing/);
  assert.ok((await stat(path.join(f.output, 'static/brand/icon-1024.png'))).isFile());
  await writeFile(path.join(f.output, 'pages/chat/index.js'), 'const logo = `/static/brand/logo-${size}.png`;');
  await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot }), /Dynamic or unsupported brand asset/);
});

test('configured identity only changes ignored output, with no source identity or secret written', async t => {
  const f = await fixture(t);
  const originalProject = await readFile(path.join(f.projectRoot, 'src/project.config.json'));
  const originalManifest = await readFile(path.join(f.projectRoot, 'src/manifest.json'));
  const result = await finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot, appId: 'wx1234567890abcdef' });
  assert.equal(result.mode, 'configured-appid'); assert.equal(JSON.stringify(result).includes('wx1234567890abcdef'), false);
  const config = JSON.parse(await readFile(path.join(f.output, 'project.config.json'), 'utf8'));
  assert.equal(config.appid, 'wx1234567890abcdef'); assert.equal(config.setting.urlCheck, true);
  assert.deepEqual(await readFile(path.join(f.projectRoot, 'src/project.config.json')), originalProject);
  assert.deepEqual(await readFile(path.join(f.projectRoot, 'src/manifest.json')), originalManifest);
  assert.equal('AppSecret' in config, false); assert.equal('appsecret' in config, false);
});

test('size gate counts all remaining files exactly and fails at the first byte over the bound', async t => {
  const f = await fixture(t);
  const first = await finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot });
  assert.ok(first.bytes > 0); assert.equal(first.limitBytes, MAX_MAIN_PACKAGE_BYTES);
  const exact = await finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot, maxBytes: first.bytes });
  assert.equal(exact.bytes, first.bytes);
  await writeFile(path.join(f.output, 'not-a-runtime-file.bin'), Buffer.from([1]));
  await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot, maxBytes: first.bytes }), /above the .* uncompressed project budget/);
  for (const maxBytes of [0, -1, 1.5, NaN, Infinity, MAX_MAIN_PACKAGE_BYTES + 1]) {
    await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot, maxBytes }), /Package budget/);
  }
});

test('full main-package ceiling cannot be bypassed by a new subpackage declaration', async t => {
  const f = await fixture(t);
  await writeFile(path.join(f.output, 'app.json'), JSON.stringify({ pages: ['pages/chat/index'], subPackages: [{ root: 'heavy', pages: ['index'] }] }));
  await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot }), /per-package size accounting/);
});

test('incomplete pages and output symlinks are rejected without modifying the target file', async t => {
  const f = await fixture(t);
  await rm(path.join(f.output, 'pages/chat/index.wxml'));
  await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot }), /Incomplete page/);
  const outside = path.join(f.directory, 'outside.txt');
  await writeFile(outside, 'unchanged');
  await symlink(outside, path.join(f.output, 'outside.txt'));
  await assert.rejects(finalizeWeixinBuild(f.output, { projectRoot: f.projectRoot }), /Symlink is not allowed/);
  assert.equal(await readFile(outside, 'utf8'), 'unchanged');
});

test('invalid environment identity invokes no compiler, and compiler failure produces no success report', async t => {
  const f = await fixture(t); let calls = 0;
  await assert.rejects(buildWeixin({ projectRoot: f.projectRoot, env: { DOCTHINKER_WEIXIN_APPID: 'invalid' }, runCompiler: async () => { calls++; } }), /DOCTHINKER_WEIXIN_APPID/);
  assert.equal(calls, 0);
  const report = path.join(f.projectRoot, 'dist/build/mp-weixin-size-report.json');
  await writeFile(report, 'stale previous success');
  await assert.rejects(buildWeixin({ projectRoot: f.projectRoot, env: {}, runCompiler: async () => { calls++; throw new Error('compiler failed'); } }), /compiler failed/);
  assert.equal(calls, 1); await assert.rejects(readFile(report), { code: 'ENOENT' });
});

test('build runs only the injected compiler then reports outside the import directory without identity', async t => {
  const f = await fixture(t); const calls = [];
  const result = await buildWeixin({ projectRoot: f.projectRoot, env: { DOCTHINKER_WEIXIN_APPID: 'wx1234567890abcdef' }, runCompiler: async value => { calls.push(value); } });
  assert.equal(calls.length, 1); assert.equal(calls[0].output, f.output);
  const report = await readFile(path.join(f.projectRoot, 'dist/build/mp-weixin-size-report.json'), 'utf8');
  assert.deepEqual(JSON.parse(report), result); assert.equal(report.includes('wx1234567890abcdef'), false);
  await assert.rejects(readFile(path.join(f.output, 'mp-weixin-size-report.json')), { code: 'ENOENT' });
  const packageJson = JSON.parse(await readFile(path.join(project, 'package.json'), 'utf8'));
  assert.equal(packageJson.scripts['build:mp-weixin'], 'node scripts/build-weixin.mjs');
  const builder = await readFile(path.join(project, 'scripts/build-weixin.mjs'), 'utf8');
  assert.match(builder, /\[cli, 'build', '-p', 'mp-weixin'\]/); assert.match(builder, /shell: false/);
});
