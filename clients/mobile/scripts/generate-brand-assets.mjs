#!/usr/bin/env node
/** Historical SVG exploration export. Never overwrites the active dog identity
 * or the frozen legacy files; output goes to the ignored output/ directory.
 * Reproducible local SVG → PNG export. No network, API keys, or image model.
 * Run: SHARP_MODULE=/absolute/path/to/sharp node scripts/generate-brand-assets.mjs
 * Or install sharp in a separate tooling environment and set SHARP_MODULE there.
 * SVG design sources below remain editable; raster files are generated outputs.
 */
import { createRequire } from 'node:module';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const sharp = require(process.env.SHARP_MODULE || 'sharp');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const production = path.resolve(root, '../../output/brand-exploration/production');
const showcase = path.resolve(root, '../../output/brand-exploration/showcase');
const COLORS = { paper: '#F6F5F0', ink: '#203D35', sage: '#E6ECDD', lime: '#D7EF99', muted: '#89928B' };
const APP_ICON_SIZES = [20, 29, 40, 48, 58, 60, 72, 76, 80, 87, 96, 120, 144, 152, 167, 180, 192, 512, 1024];

const variants = [
  {
    id: '01-evidence-window', name: '证据之窗', english: 'Evidence Window', pattern: '字母几何 × 节点连接', primary: true,
    rationale: '把 Doc 的 D 变成一扇有边界的文档窗口，短线只连接窗口内的一个来源节点。先有依据，再生长洞见；不会把无边界的补边当作事实。实心轮廓适合 App 与小尺寸。',
    geometry: `<g fill="currentColor"><path fill-rule="evenodd" d="M30 20H48C68 20 82 33 82 50S68 80 48 80H30A8 8 0 0 1 22 72V28A8 8 0 0 1 30 20ZM36 34V66H48C60 66 68 59 68 50S60 34 48 34H36Z"/><circle cx="55" cy="50" r="6"/></g><path d="M33 50H50" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`,
  },
  {
    id: '02-source-fold', name: '引文折页', english: 'Source Fold', pattern: '几何叠层 × 留白',
    rationale: '两张错位折页保留资料的来源层次。右上角的独立圆点代表从文档中提取的想法，文档主体与推断之间仍留有清晰空隙。偏轻盈、阅读工具感。',
    geometry: `<g fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"><path d="M61 24H33A7 7 0 0 0 26 31V68A7 7 0 0 0 33 75H62A7 7 0 0 0 69 68V44"/><path d="M38 37H53M38 49H56M38 61H51"/></g><path d="M60 24V35A5 5 0 0 0 65 40H76" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><circle cx="75" cy="23" r="6" fill="currentColor"/>`,
  },
  {
    id: '03-reasoning-trail', name: '推理小径', english: 'Reasoning Trail', pattern: '非对称节点网络',
    rationale: '三个层次分明的节点由一条有方向的曲线串联，把散点变成可逐步检查的推理路径。末端不封闭成万能答案，适合强调关系探索的方向。',
    geometry: `<path d="M28 72C28 52 51 69 51 49S73 49 73 28" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><g fill="currentColor"><circle cx="28" cy="72" r="6"/><circle cx="51" cy="49" r="8"/><circle cx="73" cy="28" r="6"/></g>`,
  },
  {
    id: '04-insight-aperture', name: '洞见光阑', english: 'Insight Aperture', pattern: '密集线系统 × 负空间',
    rationale: '六条短弧围出安静的中心，像把多份资料逐步对焦。洞见来自中心留白而非额外装饰；这是更抽象、更偏研究工具的方案。',
    geometry: `<g fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"><path d="M46 21C53 21 60 25 64 31"/><path d="M46 21C53 21 60 25 64 31" transform="rotate(60 50 50)"/><path d="M46 21C53 21 60 25 64 31" transform="rotate(120 50 50)"/><path d="M46 21C53 21 60 25 64 31" transform="rotate(180 50 50)"/><path d="M46 21C53 21 60 25 64 31" transform="rotate(240 50 50)"/><path d="M46 21C53 21 60 25 64 31" transform="rotate(300 50 50)"/></g><circle cx="50" cy="50" r="5" fill="currentColor"/>`,
  },
  {
    id: '05-knowledge-seed', name: '知识种子', english: 'Knowledge Seed', pattern: '曲面相交 × 单焦点',
    rationale: '两片翻开的纸页组成种子的外形，交汇处出现一个新的核心。让“持续整理与学习”显得温和、有人文感；更适合知识伙伴而非数据库的定位。',
    geometry: `<g fill="none" stroke="currentColor" stroke-width="4" stroke-linejoin="round"><path d="M50 75C29 70 19 50 25 26C46 31 56 51 50 75Z"/><path d="M50 75C44 51 54 31 75 26C81 50 71 70 50 75Z"/></g><circle cx="50" cy="63" r="6" fill="currentColor"/>`,
  },
  {
    id: '06-memory-lattice', name: '记忆格', english: 'Memory Lattice', pattern: '模块点阵 × 选择性缺口',
    rationale: '五个圆角模块从左下向右上重组，最后一格由圆点替代。规则部分是可验证的知识骨架，变化部分是新想法；适合强调可控记忆与自改进的方向。',
    geometry: `<g fill="currentColor"><rect x="23" y="57" width="16" height="16" rx="5"/><rect x="23" y="35" width="16" height="16" rx="5"/><rect x="45" y="57" width="16" height="16" rx="5"/><rect x="45" y="35" width="16" height="16" rx="5"/><rect x="67" y="57" width="12" height="16" rx="5"/><circle cx="73" cy="24" r="6"/></g>`,
  },
];

const tabs = {
  chat: `<path d="M29 24H71A9 9 0 0 1 80 33V59A9 9 0 0 1 71 68H48L31 79V68H29A9 9 0 0 1 20 59V33A9 9 0 0 1 29 24Z"/><path d="M34 43H66M34 54H55"/>`,
  documents: `<path d="M36 20H62L77 35V67A7 7 0 0 1 70 74H36A7 7 0 0 1 29 67V27A7 7 0 0 1 36 20Z"/><path d="M61 21V36H75M43 49H63M43 60H57M20 34V73A9 9 0 0 0 29 82H59"/>`,
  memory: `<rect x="22" y="22" width="24" height="24" rx="6"/><rect x="56" y="22" width="24" height="24" rx="6"/><path d="M24 59H52M24 72H42M69 57V79M58 68H80"/>`,
  graph: `<path d="M30 65L46 35M54 34L72 62M34 73L68 71"/><circle cx="50" cy="27" r="8"/><circle cx="26" cy="73" r="8"/><circle cx="76" cy="70" r="8"/>`,
  settings: `<path d="M22 29H32M46 29H78M22 50H57M71 50H78M22 71H33M47 71H78"/><circle cx="39" cy="29" r="7"/><circle cx="64" cy="50" r="7"/><circle cx="40" cy="71" r="7"/>`,
};

function svg(geometry, { title = 'DocThinker', color = COLORS.ink, background, scale = 1 } = {}) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" color="${color}" role="img" aria-label="${title}"><title>${title}</title>${background ? `<rect width="100" height="100" fill="${background}"/>` : ''}<g transform="translate(${50 * (1 - scale)} ${50 * (1 - scale)}) scale(${scale})">${geometry}</g></svg>\n`;
}
async function saveText(name, source) {
  await Promise.all([production, showcase].map(dir => writeFile(path.join(dir, name), source)));
}
async function png(name, source, size, opaque = false) {
  let pipeline = sharp(Buffer.from(source), { density: 384 }).resize(size, size);
  if (opaque) pipeline = pipeline.flatten({ background: COLORS.paper });
  const buffer = await pipeline.png({ compressionLevel: 9 }).toBuffer();
  await Promise.all([production, showcase].map(dir => writeFile(path.join(dir, name), buffer)));
  return buffer;
}

await Promise.all([production, showcase].map(dir => mkdir(dir, { recursive: true })));
for (const variant of variants) await saveText(`${variant.id}.svg`, svg(variant.geometry, { title: `DocThinker · ${variant.name}` }));
const primary = variants.find(item => item.primary);
const primarySvg = svg(primary.geometry, { title: 'DocThinker · 证据之窗' });
await saveText('logo.svg', primarySvg);
for (const size of [16, 32, 64, 256]) await png(`logo-${size}.png`, primarySvg, size);
await png('logo-light-64.png', svg(primary.geometry, { color: COLORS.paper }), 64);
const appSvg = svg(primary.geometry, { background: COLORS.paper, scale: 1 });
await saveText('icon.svg', appSvg);
for (const size of APP_ICON_SIZES) await png(`icon-${size}.png`, appSvg, size, true);
// Separate maskable export. The scaled symbol fits entirely inside the center
// 80%-diameter safe circle, including antialiasing and all foreground pixels.
const maskableSvg = svg(primary.geometry, { background: COLORS.paper, scale: 0.9 });
await saveText('icon-maskable.svg', maskableSvg);
await png('icon-maskable-512.png', maskableSvg, 512, true);
for (const [name, geometry] of Object.entries(tabs)) {
  for (const active of [false, true]) {
    const stem = `tab-${name}${active ? '-active' : ''}`;
    const source = svg(`<g fill="none" stroke="currentColor" stroke-width="${active ? 5 : 4}" stroke-linecap="round" stroke-linejoin="round">${geometry}</g>`, { title: `DocThinker ${name}`, color: active ? COLORS.ink : COLORS.muted });
    await saveText(`${stem}.svg`, source);
    await png(`${stem}.png`, source, 81);
  }
}

const catalog = {
  brand: 'DocThinker', primary: primary.id, provisional: true,
  note: '当前为可替换的默认设计，待用户选择确认。仅矢量本地渲染，未生成付费场景图。',
  palette: COLORS,
  variants: variants.map(({ geometry, ...metadata }) => metadata),
  appIcons: APP_ICON_SIZES.map(size => ({ file: `icon-${size}.png`, size, opaque: true, preRounded: false })),
  maskable: 'icon-maskable-512.png',
};
await saveText('catalog.json', `${JSON.stringify(catalog, null, 2)}\n`);

// Small-size visual proof sheet: actual PNG exports at native size plus all six
// candidates. SVG text uses system fonts only; the logo itself contains no text.
const pieces = [];
for (let index = 0; index < variants.length; index += 1) {
  const left = 32 + (index % 3) * 288, top = 32 + Math.floor(index / 3) * 240;
  pieces.push({ input: await sharp(Buffer.from(svg(variants[index].geometry))).resize(144, 144).png().toBuffer(), left: left + 68, top: top + 24 });
  const label = `<svg xmlns="http://www.w3.org/2000/svg" width="272" height="50"><text x="136" y="23" font-family="Arial,sans-serif" font-size="16" fill="${COLORS.ink}" text-anchor="middle">${variants[index].english}</text><text x="136" y="44" font-family="Arial,sans-serif" font-size="12" fill="#717970" text-anchor="middle">${index === 0 ? '01 · DEFAULT / EDITABLE' : String(index + 1).padStart(2, '0') + ' · ALTERNATIVE'}</text></svg>`;
  pieces.push({ input: Buffer.from(label), left, top: top + 174 });
}
for (const [index, size] of [16, 32, 64, 128].entries()) {
  const left = 78 + index * 150;
  pieces.push({ input: await sharp(Buffer.from(primarySvg)).resize(size, size).png().toBuffer(), left, top: 590 - Math.floor(size / 2) });
}
for (const [index, name] of Object.keys(tabs).entries()) {
  pieces.push({ input: await sharp(path.join(production, `tab-${name}-active.png`)).resize(27, 27).png().toBuffer(), left: 88 + index * 108, top: 686 });
}
const sheet = await sharp({ create: { width: 928, height: 760, channels: 3, background: COLORS.paper } }).composite(pieces).png().toBuffer();
await writeFile(path.join(showcase, 'small-size-proof.png'), sheet);

const template = await readFile(path.join(root, 'public/brand/showcase.template.html'), 'utf8');
const cards = variants.map((variant, index) => `<article class="logo-card${variant.primary ? ' selected' : ''}" data-id="${variant.id}"><button class="pick-logo" type="button" aria-pressed="${Boolean(variant.primary)}" aria-label="对比 ${variant.name}"><span class="card-index">${String(index + 1).padStart(2, '0')} / ${variant.pattern}</span><span class="logo-canvas">${svg(variant.geometry, { title: variant.name })}</span><span class="card-heading">${variant.name}<span class="selection-label">${variant.primary ? '默认方案' : '加入对比'}</span></span><span class="card-subtitle">${variant.english}</span></button><p>${variant.rationale}</p><a class="download-link" href="${variant.id}.svg" download>下载可编辑 SVG ↗</a></article>`).join('\n');
await writeFile(path.join(showcase, 'index.html'), template.replace('<!-- LOGO_GRID -->', cards));

// Validate exports, transparency and safe area without relying on screenshots.
for (const size of APP_ICON_SIZES) {
  const info = await sharp(path.join(production, `icon-${size}.png`)).metadata();
  if (info.width !== size || info.height !== size || info.hasAlpha) throw new Error(`Invalid opaque icon ${size}`);
}
const transparent = await sharp(path.join(production, 'logo-32.png')).metadata();
if (!transparent.hasAlpha) throw new Error('Transparent logo export lost alpha');
const { data, info } = await sharp(path.join(production, 'icon-maskable-512.png')).raw().toBuffer({ resolveWithObject: true });
const paper = [246, 245, 240];
let maxRadius = 0;
for (let y = 0; y < info.height; y += 1) for (let x = 0; x < info.width; x += 1) {
  const start = (y * info.width + x) * info.channels;
  if (paper.some((value, channel) => Math.abs(data[start + channel] - value) > 4)) maxRadius = Math.max(maxRadius, Math.hypot(x + 0.5 - 256, y + 0.5 - 256));
}
if (maxRadius > 204.8) throw new Error(`Maskable foreground outside safe circle: ${maxRadius}`);
console.log(`Generated ${variants.length} SVG concepts, app/PWA icons and ${Object.keys(tabs).length * 2} tab PNGs. Maskable radius ${maxRadius.toFixed(1)} / 204.8 px.`);
console.log(`Preview: ${path.join(showcase, 'index.html')}`);
