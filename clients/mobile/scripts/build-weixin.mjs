import { spawn } from 'node:child_process';
import { copyFile, lstat, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// Project policy, not WeChat's compressed/uploaded size measurement. Using
// decimal 2 MB and counting all output files leaves a conservative bound.
export const MAX_MAIN_PACKAGE_BYTES = 2_000_000;
export const GUEST_APP_ID = 'touristappid';

export function resolveWeixinAppId(value) {
  if (value === undefined || value === '') return GUEST_APP_ID;
  if (typeof value !== 'string' || !/^wx[a-f0-9]{16}$/.test(value)) {
    throw new Error('DOCTHINKER_WEIXIN_APPID must be wx followed by 16 lowercase hexadecimal characters; unset it for guest mode.');
  }
  return value;
}

async function inventory(directory, prefix = '') {
  if (!(await lstat(directory)).isDirectory()) throw new Error('Build output must be a real directory, not a symlink.');
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = prefix + entry.name;
    if (entry.isSymbolicLink()) throw new Error(`Symlink is not allowed in the mini-program build: ${filename}`);
    if (entry.isDirectory()) files.push(...await inventory(path.join(directory, entry.name), filename + '/'));
    else if (entry.isFile()) files.push(filename);
    else throw new Error(`Unsupported build entry: ${filename}`);
  }
  return files.sort();
}

async function pruneUnusedBrandFiles(output, files) {
  const required = new Set();
  const directory = /^static\/(?:brand|original-brand)\//;
  const literal = /static\/(brand|original-brand)\/([A-Za-z0-9][A-Za-z0-9_.-]*\.(?:png|jpe?g|webp|svg))/g;
  for (const filename of files) {
    if (directory.test(filename) || !/\.(?:js|json|wxml|wxss)$/.test(filename)) continue;
    const source = await readFile(path.join(output, filename), 'utf8');
    for (const match of source.matchAll(literal)) required.add('static/' + match[1] + '/' + match[2]);
    // Fail instead of silently removing an image constructed at runtime. Use
    // literal asset paths in this small app so its packaged resources are known.
    if (/static\/(?:brand|original-brand)\//.test(source.replace(literal, ''))) {
      throw new Error(`Dynamic or unsupported brand asset path in ${filename}; use a complete literal file path.`);
    }
  }
  for (const filename of required) {
    if (!files.includes(filename)) throw new Error(`Referenced brand asset is missing: ${filename}`);
  }
  const removed = files.filter(filename => directory.test(filename) && !required.has(filename));
  for (const filename of removed) await rm(path.join(output, filename));
  return { kept: [...required].sort(), removed };
}

export async function finalizeWeixinBuild(output, {
  appId,
  projectRoot = root,
  maxBytes = MAX_MAIN_PACKAGE_BYTES,
} = {}) {
  const appid = resolveWeixinAppId(appId);
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_MAIN_PACKAGE_BYTES) {
    throw new Error('Package budget must be a positive integer no larger than 2,000,000 bytes.');
  }
  output = path.resolve(output);
  const files = await inventory(output);
  for (const name of ['app.js', 'app.json', 'project.config.json']) {
    if (!files.includes(name)) throw new Error(`Incomplete WeChat build: ${name} is missing.`);
  }
  const app = JSON.parse(await readFile(path.join(output, 'app.json'), 'utf8'));
  if (!Array.isArray(app.pages) || !app.pages.length) throw new Error('WeChat build has no pages.');
  // This project currently has one main package. A future split must implement
  // per-package accounting instead of accidentally relaxing this total limit.
  if (app.subPackages?.length || app.subpackages?.length) {
    throw new Error('Subpackages require explicit per-package size accounting before this build can be released.');
  }
  for (const page of app.pages) {
    if (typeof page !== 'string' || !/^[A-Za-z0-9_/-]+$/.test(page) || page.includes('..')) throw new Error('Invalid page path.');
    for (const extension of ['js', 'json', 'wxml']) {
      if (!files.includes(`${page}.${extension}`)) throw new Error(`Incomplete page: ${page}.${extension}`);
    }
  }
  const generated = JSON.parse(await readFile(path.join(output, 'project.config.json'), 'utf8'));
  const template = JSON.parse(await readFile(path.join(projectRoot, 'src/project.config.json'), 'utf8'));
  const project = {
    ...generated, ...template, appid, compileType: 'miniprogram', miniprogramRoot: './',
    setting: { ...generated.setting, ...template.setting, urlCheck: true, uploadWithSourceMap: false },
  };
  await writeFile(path.join(output, 'project.config.json'), JSON.stringify(project, null, 2) + '\n');
  app.sitemapLocation = 'sitemap.json';
  await writeFile(path.join(output, 'app.json'), JSON.stringify(app, null, 2) + '\n');
  await copyFile(path.join(projectRoot, 'src/sitemap.json'), path.join(output, 'sitemap.json'));
  await copyFile(path.resolve(projectRoot, '../../docs/WECHAT_MINIPROGRAM.md'), path.join(output, 'README.md'));
  const brand = await pruneUnusedBrandFiles(output, await inventory(output));
  const sizes = await Promise.all((await inventory(output)).map(async filename => ({
    path: filename, bytes: (await stat(path.join(output, filename))).size,
  })));
  const bytes = sizes.reduce((total, file) => total + file.bytes, 0);
  if (bytes > maxBytes) {
    throw new Error(`WeChat main package is ${bytes} bytes, above the ${maxBytes}-byte uncompressed project budget. Remove unused assets or design explicit subpackages.`);
  }
  return {
    mode: appid === GUEST_APP_ID ? 'guest' : 'configured-appid',
    files: sizes.length, bytes, limitBytes: maxBytes, measurement: 'uncompressed-all-output-files',
    brand, largestFiles: sizes.sort((a, b) => b.bytes - a.bytes || a.path.localeCompare(b.path)).slice(0, 8),
  };
}

async function compileWeixin({ projectRoot, output, env }) {
  const cli = path.join(projectRoot, 'node_modules/@dcloudio/vite-plugin-uni/bin/uni.js');
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, 'build', '-p', 'mp-weixin'], {
      cwd: projectRoot, stdio: 'inherit', shell: false,
      env: { ...env, UNI_INPUT_DIR: path.join(projectRoot, 'src'), UNI_OUTPUT_DIR: output, UNI_PLATFORM: 'mp-weixin' },
    });
    child.once('error', reject);
    child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`WeChat compiler failed (${signal || code}).`)));
  });
}

export async function buildWeixin({ projectRoot = root, env = process.env, runCompiler = compileWeixin } = {}) {
  const appId = env.DOCTHINKER_WEIXIN_APPID;
  resolveWeixinAppId(appId); // Invalid identity must fail before invoking the compiler.
  const output = path.join(projectRoot, 'dist/build/mp-weixin');
  await rm(path.join(projectRoot, 'dist/build/mp-weixin-size-report.json'), { force: true });
  await runCompiler({ projectRoot, output, env });
  const result = await finalizeWeixinBuild(output, { projectRoot, appId });
  // This report is outside the import directory and contains no AppID or keys.
  await writeFile(path.join(projectRoot, 'dist/build/mp-weixin-size-report.json'), JSON.stringify(result, null, 2) + '\n');
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const result = await buildWeixin();
    console.log(`WeChat ${result.mode}: ${result.files} files, ${result.bytes}/${result.limitBytes} uncompressed bytes; ${result.brand.removed.length} unused brand files removed.`);
    console.log('Import dist/build/mp-weixin in WeChat DevTools. An AppID only configures project identity; account type, device access, upload and trial-release permissions remain unverified.');
  } catch (error) {
    console.error(error.message); process.exitCode = 1;
  }
}
