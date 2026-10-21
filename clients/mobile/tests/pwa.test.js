import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { buildPwa } from '../scripts/build-pwa.mjs';

const template = await readFile(new URL('../scripts/service-worker-template.js',import.meta.url),'utf8');
const registrationSource = await readFile(new URL('../public/pwa-register.js',import.meta.url),'utf8');
const manifest = JSON.parse(await readFile(new URL('../public/manifest.webmanifest',import.meta.url),'utf8'));
const files = ['index.html','assets/app.hash.js','static/brand/icon-192.png'].map(path => ({path,integrity:'sha256-ZmFrZQ=='}));

function workerHarness() {
  const scope = 'https://example.test/mobile/';
  const prefix = 'docthinker-mobile-shell-v1:' + encodeURIComponent('/mobile/') + ':';
  const current = prefix + 'test-version';
  const handlers = {}, stores = new Map(), calls = {fetch:[],put:[],deleted:[],skip:0,claim:0};
  const api = {fetch: async request => new Response('static:' + request.url)};
  const caches = {
    async open(name) {
      if (!stores.has(name)) stores.set(name,new Map());
      const store = stores.get(name);
      return {put:async (url,response) => { calls.put.push(url); store.set(url,response); },match:async url => store.get(url)?.clone()};
    },
    keys:async () => [...stores.keys()],
    delete:async name => { calls.deleted.push(name); return stores.delete(name); },
  };
  vm.runInNewContext(template.replace('__DT_PWA_VERSION__','test-version').replace('__DT_PWA_FILES__',JSON.stringify(files)),{
    URL,Request,Response,Set,caches,
    fetch: request => { calls.fetch.push(request); return api.fetch(request); },
    self:{registration:{scope},addEventListener:(name,handler) => { handlers[name] = handler; },
      skipWaiting:() => { calls.skip += 1; },clients:{claim:() => { calls.claim += 1; }}},
  });
  async function lifecycle(name) { let promise; handlers[name]({waitUntil:value => { promise = value; }}); return promise; }
  function fetchEvent(url,options = {}) {
    let response;
    handlers.fetch({request:new Request(new URL(url,scope),options),respondWith:value => { response = value; }});
    return response;
  }
  return {scope,prefix,current,stores,calls,api,lifecycle,fetchEvent};
}

test('manifest supplies standalone Android/Apple metadata and separate maskable icons', async () => {
  assert.equal(manifest.display,'standalone'); assert.equal(manifest.scope,'./');
  assert.equal(manifest.prefer_related_applications,false); assert.match(manifest.start_url,/#\/pages\/chat\/index$/);
  assert.ok(manifest.icons.some(icon => icon.sizes === '192x192' && icon.purpose === 'any'));
  assert.ok(manifest.icons.some(icon => icon.sizes === '512x512' && icon.purpose === 'any'));
  assert.ok(manifest.icons.some(icon => icon.purpose === 'maskable' && icon.src.includes('icon-maskable-512')));
  const html = await readFile(new URL('../index.html',import.meta.url),'utf8');
  assert.match(html,/apple-touch-icon/); assert.match(html,/icon-180\.png/); assert.match(html,/manifest\.webmanifest/);
  assert.match(html,/data-mode="%MODE%"/);
});

test('shell install requests only exact static bytes without credentials and never forces takeover', async () => {
  const h = workerHarness(); await h.lifecycle('install');
  assert.deepEqual(h.calls.fetch.map(request => request.url),files.map(file => h.scope + file.path));
  for (const request of h.calls.fetch) {
    assert.equal(request.credentials,'omit'); assert.equal(request.redirect,'error');
    assert.equal(request.cache,'reload'); assert.equal(request.integrity,'sha256-ZmFrZQ==');
    assert.equal(request.headers.has('Authorization'),false);
  }
  assert.equal(h.calls.skip,0); assert.equal(h.calls.claim,0);
});

test('offline hash-router root and exact static files come from shell, not a fake API answer', async () => {
  const h = workerHarness(); await h.lifecycle('install'); h.api.fetch = async () => { throw new Error('offline'); };
  const count = h.calls.fetch.length;
  assert.match(await (await h.fetchEvent(h.scope)).text(),/index\.html/);
  assert.match(await (await h.fetchEvent('assets/app.hash.js')).text(),/app\.hash\.js/);
  assert.equal(h.calls.fetch.length,count);
});

for (const [url,options] of [
  ['api/v1/mobile/sessions',{}],['api/v1/mobile/query',{method:'POST',body:'private question'}],
  ['api/v1/mobile/voice/speak',{method:'POST',body:'private answer'}],['uploads/private.png',{}],
  ['auth/login',{}],['assets/not-in-build.js',{}],['index.html?token=private',{}],
  ['assets/app.hash.js',{headers:{Authorization:'Bearer private'}}],
  ['assets/app.hash.js',{headers:{Range:'bytes=0-10'}}],['https://private.example/api/v1/mobile/graph',{}],
  ['https://cdn.example/assets/app.hash.js',{}],
]) {
  test(`private or unknown request passes through untouched: ${url}`, async () => {
    const h = workerHarness(); await h.lifecycle('install'); const putCount = h.calls.put.length;
    assert.equal(h.fetchEvent(url,options),undefined);
    assert.equal(h.calls.put.length,putCount);
  });
}

test('a missing static cache entry falls through without runtime caching its response', async () => {
  const h = workerHarness(); await h.lifecycle('install'); h.stores.get(h.current).delete(h.scope + 'assets/app.hash.js');
  const putCount = h.calls.put.length; await h.fetchEvent('assets/app.hash.js');
  assert.equal(h.calls.put.length,putCount); assert.equal(h.calls.fetch.length,files.length + 1);
  h.api.fetch = async () => { throw new Error('offline'); };
  await assert.rejects(h.fetchEvent('assets/app.hash.js'),/offline/);
});

test('partial failed install removes only its new cache and preserves the working version', async () => {
  const h = workerHarness(); h.stores.set(h.prefix + 'old',new Map());
  h.api.fetch = async request => request.url.endsWith('.js') ? new Response('failure',{status:503}) : new Response('shell');
  await assert.rejects(h.lifecycle('install'),/unavailable/);
  assert.equal(h.stores.has(h.current),false); assert.equal(h.stores.has(h.prefix + 'old'),true);
});

test('activation deletes only this app and scope old caches without touching active pages or other data', async () => {
  const h = workerHarness();
  for (const name of [h.current,h.prefix + 'old','another-app-cache','docthinker-mobile-shell-v1:%2Fother%2F:old']) h.stores.set(name,new Map());
  await h.lifecycle('activate');
  assert.deepEqual(h.calls.deleted,[h.prefix + 'old']); assert.equal(h.calls.claim,0); assert.equal(h.calls.skip,0);
});

test('build integration produces deterministic allowlist and version changes with static bytes', async () => {
  const output = await mkdtemp(path.join(os.tmpdir(),'docthinker-pwa-test-'));
  try {
    await mkdir(path.join(output,'assets')); await mkdir(path.join(output,'static/brand'),{recursive:true});
    await mkdir(path.join(output,'static/original-brand'),{recursive:true});
    await mkdir(path.join(output,'uploads')); await mkdir(path.join(output,'api'));
    await writeFile(path.join(output,'index.html'),'<!doctype html>shell');
    await writeFile(path.join(output,'assets/app.hash.js'),'export const demo = true');
    for (const name of ['logo-32','logo-64','icon-180','icon-192','icon-512','icon-maskable-512']) await writeFile(path.join(output,`static/original-brand/${name}.png`),'fake test image');
    await writeFile(path.join(output,'uploads/private.png'),'private upload'); await writeFile(path.join(output,'api/private.json'),'private response');
    const first = await buildPwa(output), second = await buildPwa(output);
    assert.equal(first.version,second.version);
    assert.ok(first.entries.every(entry => /^(index\.html|manifest\.webmanifest|pwa-register\.js|assets\/|static\/(?:brand|original-brand)\/)/.test(entry.path)));
    assert.ok(first.entries.some(entry => entry.path === 'static/original-brand/logo-64.png'));
    assert.ok(!first.entries.some(entry => entry.path === 'static/original-brand/logo.png'),'do not precache the high-resolution source');
    assert.ok(first.entries.every(entry => entry.integrity.startsWith('sha256-')));
    const worker = await readFile(path.join(output,'service-worker.js'),'utf8');
    assert.ok(!worker.includes('private.png')); assert.ok(!worker.includes('private.json')); assert.ok(!worker.includes('__DT_PWA_'));
    await writeFile(path.join(output,'assets/app.hash.js'),'export const demo = false');
    assert.notEqual((await buildPwa(output)).version,first.version);
  } finally { await rm(output,{recursive:true,force:true}); }
});

test('H5 build publishes the logo showcase and all six proposals without precaching them', async () => {
  const output = await mkdtemp(path.join(os.tmpdir(),'docthinker-brand-build-test-'));
  try {
    await mkdir(path.join(output,'assets')); await mkdir(path.join(output,'static/brand'),{recursive:true});
    await mkdir(path.join(output,'static/original-brand'),{recursive:true});
    await writeFile(path.join(output,'index.html'),'<!doctype html>shell');
    for (const name of ['logo-32','logo-64','icon-180','icon-192','icon-512','icon-maskable-512']) await writeFile(path.join(output,`static/original-brand/${name}.png`),'fake test image');
    const result = await buildPwa(output);
    const showcaseFiles = ['index.html','01-evidence-window.svg','02-source-fold.svg','03-reasoning-trail.svg',
      '04-insight-aperture.svg','05-knowledge-seed.svg','06-memory-lattice.svg'];
    for (const filename of showcaseFiles) {
      const source = await readFile(new URL('../public/brand/' + filename,import.meta.url));
      assert.deepEqual(await readFile(path.join(output,'brand',filename)),source);
    }
    assert.deepEqual(await readFile(path.join(output,'original-brand/logo.png')),
      await readFile(new URL('../public/original-brand/logo.png',import.meta.url)));
    assert.ok(result.entries.every(entry => !entry.path.startsWith('original-brand/')));
    assert.ok(result.entries.every(entry => !entry.path.startsWith('brand/')));
    const h = workerHarness();
    assert.equal(h.fetchEvent('brand/'),undefined);
    assert.equal(h.fetchEvent('brand/01-evidence-window.svg'),undefined);
  } finally { await rm(output,{recursive:true,force:true}); }
});

function installerHarness({production = true,secure = true,supported = true,fail = false,waiting = false} = {}) {
  const listeners = {}, registrationListeners = {}, calls = {register:[],reload:0};
  const registration = {waiting,addEventListener:(event,fn) => { registrationListeners[event] = fn; }};
  const context = {
    URL,Set,document:{currentScript:{src:'https://example.test/mobile/pwa-register.js',dataset:{mode:production ? 'production' : 'development'}},readyState:'complete'},
    location:{href:'https://example.test/mobile/'},navigator:{},
    window:{isSecureContext:secure,matchMedia:() => ({matches:false,addEventListener(){}}),addEventListener:(event,fn) => { listeners[event] = fn; },location:{reload:() => { calls.reload += 1; }}},
  };
  if (supported) context.navigator.serviceWorker = {controller:{},ready:Promise.resolve(registration),register:async (...args) => {
    calls.register.push(args); if (fail) throw new Error('private deployment details'); return registration;
  }};
  vm.runInNewContext(registrationSource,context);
  return {api:context.window.DocThinkerInstall,listeners,calls,registration,registrationListeners};
}
const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };

for (const options of [{production:false},{secure:false},{supported:false}]) {
  test(`unsupported or development page never registers a worker: ${JSON.stringify(options)}`, async () => {
    const h = installerHarness(options); await flush(); assert.equal(h.calls.register.length,0);
    assert.equal(h.api.getState().canPrompt,false); assert.equal(h.api.getState().offlineReady,false);
  });
}

test('installer registers scoped worker and reports waiting update without reload', async () => {
  const h = installerHarness({waiting:true}); await flush();
  assert.equal(h.calls.register[0][0],'https://example.test/mobile/service-worker.js');
  assert.equal(h.calls.register[0][1].scope,'https://example.test/mobile/');
  assert.equal(h.calls.register[0][1].updateViaCache,'none');
  assert.equal(h.api.getState().updateAvailable,true); assert.equal(h.api.getState().offlineReady,true); assert.equal(h.calls.reload,0);
});

test('browser install prompt requires explicit click and accepted choice is not faked as installed', async () => {
  const h = installerHarness(); let prompts = 0, prevented = 0;
  h.listeners.beforeinstallprompt({preventDefault:() => { prevented += 1; },prompt:async () => { prompts += 1; },userChoice:Promise.resolve({outcome:'accepted'})});
  assert.equal(prompts,0); assert.equal(prevented,1); assert.equal(h.api.getState().canPrompt,true);
  assert.equal(await h.api.prompt(),'accepted'); assert.equal(prompts,1); assert.equal(h.api.getState().installed,false);
  assert.equal(await h.api.prompt(),'unavailable'); assert.equal(prompts,1);
  h.listeners.appinstalled(); assert.equal(h.api.getState().installed,true);
});

test('installation failure is safe and does not claim offline readiness', async () => {
  const h = installerHarness({fail:true}); await flush();
  assert.equal(h.api.getState().offlineReady,false); assert.match(h.api.getState().error,/尚未准备/);
  assert.ok(!h.api.getState().error.includes('private')); assert.equal(h.calls.reload,0);
});

test('install guide is valid Vue and production command generates PWA without affecting native build', async () => {
  const {descriptor,errors} = parse(await readFile(new URL('../src/components/InstallGuide.vue',import.meta.url),'utf8'));
  assert.deepEqual(errors,[]); const script = compileScript(descriptor,{id:'install'});
  const compiled = compileTemplate({source:descriptor.template.content,id:'install',compilerOptions:{bindingMetadata:script.bindings}});
  assert.deepEqual(compiled.errors,[]);
  const pkg = JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
  assert.match(pkg.scripts['build:h5'],/scripts\/build-pwa\.mjs/); assert.equal(pkg.scripts['build:app'],'uni build -p app');
  assert.match(pkg.scripts['preview:h5'],/5175/);
});
