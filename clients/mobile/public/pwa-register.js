/* No private content, credentials, analytics, storage mutation or automatic reload. */
(() => {
  const script = document.currentScript;
  const production = script?.dataset.mode === 'production';
  const scope = new URL('./',script?.src || location.href);
  const listeners = new Set();
  const display = window.matchMedia?.('(display-mode: standalone)');
  let pendingPrompt = null;
  const state = {
    production, secure: window.isSecureContext === true,
    installed: Boolean(navigator.standalone || display?.matches),
    canPrompt:false, offlineReady:false, updateAvailable:false, error:'',
  };
  const publish = patch => {
    Object.assign(state,patch);
    for (const listener of listeners) { try { listener({...state}); } catch { /* observers are isolated */ } }
  };
  window.DocThinkerInstall = {
    getState: () => ({...state}),
    subscribe(listener) { listeners.add(listener); listener({...state}); return () => listeners.delete(listener); },
    async prompt() {
      const prompt = pendingPrompt;
      if (!prompt) return 'unavailable';
      pendingPrompt = null; publish({canPrompt:false});
      try { await prompt.prompt(); return (await prompt.userChoice)?.outcome || 'dismissed'; }
      catch { publish({error:'无法打开系统安装提示，请使用浏览器菜单添加到主屏幕。'}); return 'unavailable'; }
    },
  };
  display?.addEventListener?.('change',event => publish({installed:event.matches}));
  window.addEventListener('appinstalled',() => { pendingPrompt = null; publish({installed:true,canPrompt:false}); });
  if (!production || !state.secure || !('serviceWorker' in navigator)) return;
  window.addEventListener('beforeinstallprompt',event => {
    event.preventDefault(); pendingPrompt = event; publish({canPrompt:true});
  });
  const register = async () => {
    try {
      const registration = await navigator.serviceWorker.register(new URL('service-worker.js',scope).href,{
        scope:scope.href,updateViaCache:'none',
      });
      if (registration.waiting) publish({updateAvailable:true});
      registration.addEventListener('updatefound',() => {
        const worker = registration.installing;
        worker?.addEventListener('statechange',() => {
          if (worker.state === 'installed' && navigator.serviceWorker.controller) publish({updateAvailable:true});
        });
      });
      await navigator.serviceWorker.ready;
      publish({offlineReady:true});
    } catch { publish({error:'离线界面尚未准备完成；当前页面仍可在线使用。'}); }
  };
  if (document.readyState === 'complete') register();
  else window.addEventListener('load',register,{once:true});
})();
