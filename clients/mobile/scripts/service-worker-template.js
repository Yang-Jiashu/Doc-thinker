/* Build-time exact static allowlist. Never runtime-cache API/private responses. */
const VERSION = '__DT_PWA_VERSION__';
const FILES = __DT_PWA_FILES__;
const SCOPE = new URL(self.registration.scope);
const PREFIX = 'docthinker-mobile-shell-v1:' + encodeURIComponent(SCOPE.pathname) + ':';
const CACHE = PREFIX + VERSION;
const ENTRIES = FILES.map(file => ({...file,url:new URL(file.path,SCOPE).href}));
const ALLOWED = new Set(ENTRIES.map(file => file.url));
const SHELL = new URL('index.html',SCOPE).href;

self.addEventListener('install',event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    try {
      // Fetch exactly the bytes built for this version, without cookies or
      // auth, and fail the install if a gateway injects HTML or redirects.
      for (const file of ENTRIES) {
        const response = await fetch(new Request(file.url,{
          credentials:'omit',cache:'reload',redirect:'error',integrity:file.integrity,
        }));
        if (!response.ok || response.redirected || response.type === 'opaque') throw new Error('Static shell unavailable');
        await cache.put(file.url,response);
      }
    } catch (error) { await caches.delete(CACHE); throw error; }
    // No skipWaiting: an update must not replace an active draft's runtime.
  })());
});

self.addEventListener('activate',event => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith(PREFIX) && name !== CACHE) await caches.delete(name);
    }
    // No clients.claim or reload. Existing pages keep their current runtime.
  })());
});

self.addEventListener('fetch',event => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== SCOPE.origin || url.search
      || request.headers.has('authorization') || request.headers.has('range')) return;
  // Hash routing needs only these two document URLs. Never send the shell
  // for /api, uploads, voice, auth, arbitrary paths or failed private calls.
  const key = url.href === SCOPE.href ? SHELL : url.href;
  if (!ALLOWED.has(key)) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    return (await cache.match(key)) || fetch(request);
  })());
});
