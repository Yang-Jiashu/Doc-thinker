<script>
import { defineComponent, h, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';

export default defineComponent({
  name: 'ComposerJump',
  props: {
    active: { type: Boolean, default: false },
    disabled: { type: Boolean, default: false },
  },
  emits: ['jump'],
  setup(props, { emit }) {
    const offscreen = ref(false);
    let mounted = false, generation = 0, observer = null;

    function stop() {
      generation += 1;
      observer?.disconnect();
      observer = null;
      offscreen.value = false;
    }

    async function observe() {
      stop();
      if (!mounted || !props.active) return;
      const current = generation;
      await nextTick();
      if (!mounted || !props.active || current !== generation) return;
      if (typeof window === 'undefined' || typeof document === 'undefined' ||
          typeof window.IntersectionObserver !== 'function') return;
      const target = document.getElementById('chat-composer');
      if (!target) return;
      try {
        observer = new window.IntersectionObserver(entries => {
          if (!mounted || !props.active || current !== generation) return;
          const entry = entries.find(item => item.target === target);
          if (entry) offscreen.value = !entry.isIntersecting;
        // Reserve the bottom navigation area rather than count covered content as visible.
        }, { threshold: 0, rootMargin: '0px 0px -86px 0px' });
        observer.observe(target);
      } catch {
        // Browsers without a usable observer keep the ordinary page flow.
        stop();
      }
    }

    function jump() {
      if (mounted && props.active && !props.disabled && offscreen.value) emit('jump');
    }

    watch(() => props.active, observe, { flush: 'sync' });
    onMounted(() => { mounted = true; observe(); });
    onBeforeUnmount(() => { mounted = false; stop(); });

    // A real HTML button keeps Enter/Space activation and focus semantics on H5.
    return () => props.active && !props.disabled && offscreen.value
      ? h('button', { type: 'button', class: 'composer-jump', onClick: jump,
        'aria-label': '回到输入框', 'aria-controls': 'chat-composer' }, '↓ 回到输入框')
      : null;
  },
});
</script>

<style scoped>
.composer-jump {
  position:fixed; right:18px; bottom:calc(86px + env(safe-area-inset-bottom, 0px)); z-index:30;
  min-height:44px; padding:10px 16px; border:1px solid #d6dfd0; border-radius:24px;
  background:#fffef9; color:#37533b; box-shadow:0 4px 20px #2942261a;
  font:inherit; font-size:13px; font-weight:600; line-height:1.5; cursor:pointer;
}
.composer-jump:focus-visible { outline:3px solid #6a855e; outline-offset:3px; }
@media (min-width:1100px) {
  @supports selector(:has(*)) { .composer-jump { right:24px; bottom:24px; } }
}
@media (prefers-reduced-motion:reduce) { .composer-jump { scroll-behavior:auto; } }
</style>
