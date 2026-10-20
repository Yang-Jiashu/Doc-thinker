<script>
import { cancelRecording, stopSpeaking } from './services/voice.js';
// #ifdef H5
import { clearInviteSession, getPreferences } from './services/api.js';
import { cancelInviteRequest } from './services/invite-auth.js';
// #endif
export default {
  onLaunch() {
    // #ifdef H5
    if (getPreferences().invitePending) uni.reLaunch({url:'/pages/invite/index'});
    window.addEventListener('pagehide',() => {
      if (getPreferences().connectionKind === 'invite') { cancelInviteRequest(); clearInviteSession(); cancelRecording(); stopSpeaking(); }
    });
    window.addEventListener('pageshow',event => {
      if (event.persisted && getPreferences().connectionKind === 'invite') window.location.reload();
    });
    // #endif
  },
  onHide() { cancelRecording(); stopSpeaking(); },
};
</script>
<style>
page { --dt-bg:#f6f5f0; --dt-ink:#1c2723; --dt-accent:#d6fa75; --dt-muted:#626c65; --dt-border:#dde0d7; --dt-surface:#fffefa; --dt-soft:#eeefe8; --dt-radius:22px; background:var(--dt-bg); color:var(--dt-ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif; font-size:15px; line-height:1.65; }
view, button, input, textarea { box-sizing: border-box; }
button { font:inherit; min-height:44px; border-radius:14px; }
button::after { border: none; }
button[disabled] { opacity: .45; }
input, textarea { font: inherit; }
/* #ifdef H5 */
html, body, #app { --dt-bg:#f6f5f0; --dt-ink:#1c2723; --dt-accent:#d6fa75; --dt-muted:#626c65; --dt-border:#dde0d7; --dt-surface:#fffefa; --dt-soft:#eeefe8; --dt-radius:22px; background:var(--dt-bg); color:var(--dt-ink); }
html { -webkit-text-size-adjust:100%; }
uni-tabbar .uni-tabbar { border-top:1px solid #dde0d7!important; box-shadow:0 -8px 32px rgba(28,39,35,.045); }
uni-tabbar .uni-tabbar__item { min-height:64px; }
uni-tabbar .uni-tabbar__label { font-size:12px!important; font-weight:600; letter-spacing:.1px; }
uni-tabbar .uni-tabbar__icon { margin-top:6px; }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline:3px solid #637f25; outline-offset:3px; }
@media (min-width:760px) { uni-tabbar.uni-tabbar-bottom .uni-tabbar { width:calc(100% - 64px); max-width:720px; left:50%; right:auto; transform:translateX(-50%); border:1px solid #dde0d7; border-radius:22px 22px 0 0; box-shadow:0 -8px 36px rgba(28,39,35,.06); } }
/* The rail and its reserved space share one breakpoint. Older browsers keep
   the existing bottom navigation; login and standalone pages keep their shell. */
@supports selector(:has(*)) {
  @media (min-width:1100px) {
    uni-app:has(.desktop-navigation) uni-tabbar { display:none!important; }
    uni-page:has(.desktop-navigation) { --window-bottom:0px!important; }
    uni-page-wrapper:has(.desktop-navigation) { padding-left:208px; box-sizing:border-box; }
  }
}
/* Keep phone input text readable without iOS zooming the page on focus. */
@media (max-width:759px) {
  uni-page-body .mobile-page input, uni-page-body .mobile-page textarea { font-size:16px; }
}
@media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation:none!important; transition:none!important; scroll-behavior:auto!important; } }
/* #endif */
</style>
