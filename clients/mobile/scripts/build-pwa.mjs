import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const staticExtension = /\.(?:js|css|woff2?|ttf|svg|png|webp|jpe?g)$/;

async function collect(directory, prefix) {
  const files = [];
  for (const entry of await readdir(directory,{withFileTypes:true})) {
    // No symlinks or files outside the build directory.
    if (entry.isDirectory()) files.push(...await collect(path.join(directory,entry.name),prefix + entry.name + '/'));
    else if (entry.isFile() && staticExtension.test(entry.name)) files.push(prefix + entry.name);
  }
  return files;
}

async function copyPublicDirectory(source, destination) {
  await mkdir(destination,{recursive:true});
  for (const entry of await readdir(source,{withFileTypes:true})) {
    const from = path.join(source,entry.name), to = path.join(destination,entry.name);
    if (entry.isDirectory()) await copyPublicDirectory(from,to);
    else if (entry.isFile()) await copyFile(from,to);
    // Do not publish symlinks to files outside this generated public folder.
  }
}

export async function buildPwa(output = path.join(root,'dist/build/h5')) {
  for (const filename of ['manifest.webmanifest','pwa-register.js']) {
    await copyFile(path.join(root,'public',filename),path.join(output,filename));
  }
  // uni's H5 build does not reliably copy public/. Keep the design showcase
  // available online, but deliberately outside the offline shell allowlist.
  await copyPublicDirectory(path.join(root,'public/brand'),path.join(output,'brand'));
  // The original dog is the active identity. The older six-direction gallery
  // remains available as history, never as the source of generated app icons.
  await copyPublicDirectory(path.join(root,'public/original-brand'),path.join(output,'original-brand'));
  const required = ['index.html','manifest.webmanifest','pwa-register.js',
    'static/original-brand/logo-32.png','static/original-brand/logo-64.png',
    'static/original-brand/icon-180.png','static/original-brand/icon-192.png',
    'static/original-brand/icon-512.png','static/original-brand/icon-maskable-512.png'];
  const files = [...new Set([...required,...await collect(path.join(output,'assets'),'assets/'),
    ...await collect(path.join(output,'static/brand'),'static/brand/')])].sort();
  if (files.length > 250) throw new Error('PWA static shell exceeds 250 files; review its allowlist');
  let bytes = 0;
  const entries = [];
  for (const filename of files) {
    const body = await readFile(path.join(output,filename)); bytes += body.length;
    entries.push({path:filename,integrity:'sha256-' + createHash('sha256').update(body).digest('base64')});
  }
  if (bytes > 20 * 1024 * 1024) throw new Error('PWA static shell exceeds 20 MiB; review its allowlist');
  const template = await readFile(path.join(root,'scripts/service-worker-template.js'),'utf8');
  const version = createHash('sha256').update(JSON.stringify(entries)).update(template).digest('hex').slice(0,20);
  const worker = template.replace('__DT_PWA_VERSION__',version).replace('__DT_PWA_FILES__',JSON.stringify(entries));
  await writeFile(path.join(output,'service-worker.js'),worker);
  return {version,entries,bytes};
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const result = await buildPwa();
  console.log(`PWA shell: ${result.entries.length} static files, ${result.bytes} bytes, version ${result.version}`);
}
