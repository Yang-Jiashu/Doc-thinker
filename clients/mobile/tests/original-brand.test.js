import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const file = name => readFileSync(new URL('../' + name, import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const originalHash = 'c3b36dd3de1e8ebcb222fcdfc724139ed7b82a16f218763a71bba8e1832b90df';

test('both active brand directories keep an exact copy of the frozen original dog', () => {
  const original = readFileSync(new URL('../../../logo.png', import.meta.url));
  assert.equal(digest(original), originalHash);
  for (const directory of ['src/static/original-brand', 'public/original-brand']) {
    assert.deepEqual(file(directory + '/logo.png'), original);
    const catalog = JSON.parse(file(directory + '/catalog.json'));
    assert.equal(catalog.identity, 'original-dog');
    assert.equal(catalog.source_sha256, originalHash);
    assert.equal(catalog.maskable_safe_square_fraction, 0.56);
    for (const [name, hash] of Object.entries(catalog.files)) {
      assert.equal(digest(file(directory + '/' + name)), hash);
      assert.deepEqual(file(directory + '/' + name), file('src/static/original-brand/' + name));
    }
  }
});

test('current headers, conversation avatar and installation metadata use the original dog', () => {
  for (const name of ['src/components/BrandHeader.vue', 'src/components/DesktopNavigation.vue', 'src/pages/chat/index.vue']) {
    const source = file(name).toString();
    assert.match(source, /\/static\/original-brand\/logo-(?:32|64)\.png/);
    assert.doesNotMatch(source, /\/static\/brand\/logo-/);
  }
  for (const name of ['src/manifest.json', 'index.html', 'public/manifest.webmanifest']) {
    const source = file(name).toString();
    assert.match(source, /static\/original-brand\/(?:icon|logo)-/);
    assert.doesNotMatch(source, /static\/brand\/(?:icon|logo)-/);
  }
});

test('small active logos and maskable export keep opaque square PNG dimensions', () => {
  for (const [name, size] of [['logo-16.png', 16], ['logo-32.png', 32], ['logo-64.png', 64],
    ['logo-256.png', 256], ['icon-maskable-512.png', 512]]) {
    const png = file('src/static/original-brand/' + name);
    assert.equal(png.toString('hex', 0, 8), '89504e470d0a1a0a');
    assert.equal(png.readUInt32BE(16), size);
    assert.equal(png.readUInt32BE(20), size);
    assert.equal(png[25], 2);
  }
});

test('historical brand exploration cannot overwrite either committed identity', () => {
  const generator = file('scripts/generate-brand-assets.mjs').toString();
  assert.match(generator, /const production = path\.resolve\(root, '\.\.\/\.\.\/output\/brand-exploration\/production'\)/);
  assert.match(generator, /const showcase = path\.resolve\(root, '\.\.\/\.\.\/output\/brand-exploration\/showcase'\)/);
  assert.doesNotMatch(file('scripts/build-pwa.mjs').toString(), /generate-brand-assets|generate_original_brand_assets/);
});
