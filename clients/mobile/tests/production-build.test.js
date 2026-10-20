import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));

async function filesUnder(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await filesUnder(filename));
    else if (entry.isFile()) files.push(filename);
    else assert.fail(`Do not publish symlinks or special files: ${path.relative(root, filename)}`);
  }
  return files;
}

test('container H5 builds use a synchronized dependency lock and a static production command', async () => {
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
  const lock = JSON.parse(await readFile(path.join(root, 'package-lock.json'), 'utf8'));
  assert.equal(lock.lockfileVersion, 3);
  for (const group of ['dependencies', 'devDependencies']) {
    assert.deepEqual(lock.packages[''][group], pkg[group]);
    for (const [name, version] of Object.entries(pkg[group])) {
      assert.match(version, /^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/, `${name} must have an exact version`);
    }
  }
  assert.equal(pkg.scripts['build:h5'], 'uni build && node scripts/build-pwa.mjs');
  const vite = await readFile(path.join(root, 'vite.config.js'), 'utf8');
  assert.doesNotMatch(vite, /sourcemap\s*:\s*(?:true|['"]inline['"]|['"]hidden['"])/);
});

test('public build inputs contain only reviewed static asset formats, never credential or source-map files', async () => {
  for (const filename of await filesUnder(path.join(root, 'public'))) {
    const relative = path.relative(path.join(root, 'public'), filename);
    assert.ok(relative.split(path.sep).every(part => !part.startsWith('.')), relative);
    assert.match(relative, /\.(?:html|json|webmanifest|js|svg|png)$/i);
    assert.doesNotMatch(relative, /(?:^|[/\\])(?:secrets?|credentials?|private)(?:[./\\]|$)/i);
    assert.doesNotMatch(relative, /(?:\.map|\.pem|\.key|\.env)$/i);
  }
});

test('mobile browser code does not receive provider credentials through bundler environment variables', async () => {
  const credentialEnv = /(?:import\.meta\.env|process\.env)\s*(?:\[\s*['"]|\.)(?:VITE_)?(?:OPENAI_API_KEY|LLM_BINDING_API_KEY|STEPFUN_API_KEY|STEP_API_KEY|ANTHROPIC_API_KEY|MOBILE_ACCESS_TOKEN|INVITE_SIGNING_KEY)/i;
  for (const filename of await filesUnder(path.join(root, 'src'))) {
    if (!/\.(?:vue|js|json|css)$/.test(filename)) continue;
    const source = await readFile(filename, 'utf8');
    assert.doesNotMatch(source, credentialEnv, path.relative(root, filename));
  }
  const manifest = JSON.parse(await readFile(path.join(root, 'src/manifest.json'), 'utf8'));
  assert.equal(manifest.h5.router.mode, 'hash');
});
