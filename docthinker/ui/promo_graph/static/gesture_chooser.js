/* Standard graph adapter for the same tested gesture runtime as immersive view. */
(function () {
    'use strict';
    const script = document.currentScript;
    const runtimeUrl = script?.dataset.runtimeUrl || '/promo-graph-assets/dist/gesture-runtime.js';
    const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = script?.dataset.stylesheetUrl || '/promo-graph-assets/gesture_chooser.css'; document.head.append(css);
    const storageKey = 'docthinker.gesture.preferences.v1';
    let runtime, controller, runtimePromise, frame, dialog, opener;
    let manuallyPaused = false;
    let attractedNode = null;
    const stage = () => document.getElementById('graph-container');
    const item = index => graphData.nodes[index];
    const noop = () => {};

    function updateStatus() {
        const state = controller?.getState() || {};
        const text = state.error || (state.starting ? '摄像头启动中，可随时关闭' : state.enabled ? state.suspended ? '手势已暂停 · 在设置中恢复' : '手势已开启 · 鼠标仍可用' : '摄像头已关闭');
        document.getElementById('shared-gesture-state').textContent = text;
        document.getElementById('gesture-mode').textContent = text;
        if (dialog) {
            dialog.querySelector('#classic-gesture-status').textContent = text;
            dialog.querySelector('#classic-gesture-start').disabled = Boolean(state.starting || state.enabled);
            dialog.querySelector('#classic-gesture-pause').disabled = !state.enabled;
            dialog.querySelector('#classic-gesture-pause').textContent = state.suspended ? '恢复手势' : '暂停手势';
            dialog.querySelector('#classic-gesture-stop').disabled = !(state.enabled || state.starting);
        }
        const layer = document.getElementById('gesture-layer');
        layer.classList.toggle('hidden', !state.enabled);
    }
    function tick(now) {
        controller?.tick(now);
        if (controller?.getState().enabled) frame = requestAnimationFrame(tick);
    }
    function pause() {
        manuallyPaused = true; controller?.setSuspended(true); updateStatus();
    }
    function takeManualControl() {
        const state = controller?.getState();
        if (state?.suspended || !(state?.enabled || state?.starting)) return;
        pause();
    }
    function stop() {
        cancelAnimationFrame(frame); controller?.stop(); manuallyPaused = false; updateStatus();
    }
    function panBy(dx, dy) {
        svg.interrupt().call(graphZoom.transform, currentTransform.translate(dx / currentTransform.k, dy / currentTransform.k));
    }
    function rendererAdapter() {
        return {
            pick(x, y) {
                let best = -1, distance = 27;
                node?.each(function (d) {
                    const point = currentTransform.apply([d.x || 0, d.y || 0]);
                    const delta = Math.hypot(x - point[0], y - point[1]);
                    if (delta < distance) { best = graphData.nodes.indexOf(d); distance = delta; }
                });
                return best;
            },
            setHovered(index) { node?.classed('gesture-hover', d => d === item(index)); },
            fitToGraph() { svg.interrupt().call(graphZoom.transform, d3.zoomIdentity); },
            panBy,
            orbitBy(delta) { panBy(delta * 160, 0); },
            zoomAt(x, y, factor) { svg.interrupt().call(graphZoom.scaleBy, factor, [x, y]); },
            getNodeWorldPosition(index) { const d = item(index); return {x: d?.x || 0, y: d?.y || 0, z: 0}; },
            screenToWorld(x, y) { const point = currentTransform.invert([x, y]); return {x: point[0], y: point[1], z: 0}; },
            setNodeTransientPosition(index, x, y) {
                const d = item(index); if (!d) return;
                d.x = d.fx = x; d.y = d.fy = y; updateGraphPositions();
            },
        };
    }
    async function ensureRuntime() {
        if (!runtimePromise) runtimePromise = import(runtimeUrl).catch(error => { runtimePromise = null; throw error; });
        runtime = await runtimePromise;
        if (!controller) {
            controller = new runtime.GestureController(stage(), document.getElementById('gesture-layer'), document.getElementById('gesture-video'), document.getElementById('gesture-cursor'), rendererAdapter(), {
                onSelect(index) { if (item(index)) selectNode(item(index)); },
                onClearSelection: () => resetHighlight(),
                onHover: noop, onNodeMoved: noop, onGestureActiveChange: noop, onNodeDraggingChange: noop,
                onNodeAttractionStart(index) {
                    const d = item(index);
                    if (d) { attractedNode = {node: d, fx: d.fx, fy: d.fy}; simulation?.stop(); d.fx = d.x; d.fy = d.y; }
                },
                onNodeAttractionEnd() {
                    if (!attractedNode) return;
                    const previous = attractedNode; attractedNode = null;
                    previous.node.fx = previous.fx; previous.node.fy = previous.fy;
                    resumeRenderedGraphForceLayout();
                },
                onStateChange: updateStatus,
            });
        }
        let value = {}; try { value = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) {}
        controller.setPreferences(runtime.normalizeGesturePreferences(value));
        return controller;
    }
    function updateGuide() {
        const presentation = dialog.querySelector('#classic-gesture-mode').value === 'presentation';
        dialog.querySelector('#classic-gesture-guide').textContent = presentation
            ? '演示模式：张掌横移平移图谱，张掌停稳放大，单手捏合缩小。松开后再换动作。握拳长按复位。沉浸视图中横移对应旋转。'
            : '精准模式：食指指向并停留查看节点；捏住节点拖动，捏住空白平移；双手捏合后拉开或靠近缩放。握拳长按复位。';
    }
    function showPreferences() {
        const prefs = controller.getPreferences();
        for (const [field, id] of [['mode','mode'],['dominantHand','hand'],['dwellMs','dwell'],['panGain','gain'],['smoothingMs','smoothing']]) {
            const select = dialog.querySelector('#classic-gesture-' + id);
            if (![...select.options].some(option => option.value === String(prefs[field]))) {
                const option = document.createElement('option'); option.value = String(prefs[field]); option.textContent = String(prefs[field]) + ' · 自定义'; select.append(option);
            }
            select.value = prefs[field];
        }
        updateGuide();
    }
    function applyPreferences() {
        if (!controller) return;
        const gain = Number(dialog.querySelector('#classic-gesture-gain').value);
        const prefs = runtime.normalizeGesturePreferences({...controller.getPreferences(),
            mode: dialog.querySelector('#classic-gesture-mode').value,
            dominantHand: dialog.querySelector('#classic-gesture-hand').value,
            dwellMs: Number(dialog.querySelector('#classic-gesture-dwell').value),
            smoothingMs: Number(dialog.querySelector('#classic-gesture-smoothing').value),
            panGain: gain, zoomGain: gain, orbitGain: gain,
        });
        controller.setPreferences(prefs);
        try { localStorage.setItem(storageKey, JSON.stringify(prefs)); } catch (_) {}
        updateGuide();
    }
    function createDialog() {
        dialog = document.createElement('dialog'); dialog.className = 'shared-gesture-dialog';
        dialog.setAttribute('aria-labelledby', 'classic-gesture-title');
        dialog.innerHTML = `
          <header><div><h2 id="classic-gesture-title">手势设置</h2><p>标准与沉浸视图共用操作偏好</p></div><button type="button" data-close aria-label="关闭手势设置">×</button></header>
          <p class="shared-gesture-privacy">画面仅在本机用于手部识别，不上传给模型。只有点击“开启摄像头”才申请权限；切页会关闭摄像头。</p>
          <div class="shared-gesture-grid">
            <label>操作方式<select id="classic-gesture-mode"><option value="inspect">精准操作（默认）</option><option value="presentation">演示模式</option></select></label>
            <label>惯用手<select id="classic-gesture-hand"><option value="right">右手</option><option value="left">左手</option><option value="auto">自动</option></select></label>
            <label>灵敏度<select id="classic-gesture-gain"><option value="0.6">慢速 · 易控制</option><option value="1">标准</option><option value="1.5">快速 · 大范围</option></select></label>
            <label>停留选择<select id="classic-gesture-dwell"><option value="450">快速 · 450 毫秒</option><option value="650">标准 · 650 毫秒</option><option value="1000">稳妥 · 1 秒</option></select></label>
            <label>防抖平滑<select id="classic-gesture-smoothing"><option value="35">轻 · 更跟手</option><option value="70">标准</option><option value="120">强 · 更稳定</option></select></label>
          </div>
          <p id="classic-gesture-guide" class="shared-gesture-guide"></p>
          <p id="classic-gesture-status" role="status">正在读取手势配置…</p>
          <div class="shared-gesture-actions"><button type="button" id="classic-gesture-start">开启摄像头</button><button type="button" id="classic-gesture-pause" disabled>暂停手势</button><button type="button" id="classic-gesture-stop" disabled>关闭摄像头</button></div>
          <p class="shared-gesture-footnote">鼠标、触控、图谱滚轮和工具按钮优先接管并暂停手势；需要时点击“恢复手势”。切到其他窗口也会暂停。关闭会释放摄像头。手势只操作视图，不删除或修改知识。</p>`;
        document.body.append(dialog);
        dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
        dialog.addEventListener('close', () => opener?.focus());
        dialog.querySelectorAll('select').forEach(select => select.addEventListener('change', applyPreferences));
        dialog.querySelector('#classic-gesture-start').addEventListener('click', async () => {
            applyPreferences(); manuallyPaused = false; controller.setSuspended(false);
            const startPromise = controller.start(); updateStatus();
            if (await startPromise) { cancelAnimationFrame(frame); frame = requestAnimationFrame(tick); dialog.close(); }
            updateStatus();
        });
        dialog.querySelector('#classic-gesture-pause').addEventListener('click', () => {
            manuallyPaused = !controller.getState().suspended; controller.setSuspended(manuallyPaused); updateStatus();
            if (!manuallyPaused) dialog.close();
        });
        dialog.querySelector('#classic-gesture-stop').addEventListener('click', stop);
        return dialog;
    }
    window.openGestureExperienceChooser = async function () {
        if (dialog?.open) return;
        opener = document.activeElement;
        if (controller?.getState().enabled) pause();
        (dialog || createDialog()).showModal();
        dialog.querySelector('#classic-gesture-start').disabled = true;
        try { await ensureRuntime(); showPreferences(); updateStatus(); }
        catch (_) { dialog.querySelector('#classic-gesture-status').textContent = '手势模块未能加载。可以继续使用鼠标，关闭后重试。'; }
    };
    window.KGSharedGesture = { pause, stop };
    window.addEventListener('pagehide', () => { saveStandardGraphState(); stop(); });
    window.addEventListener('blur', () => {
        // Permission prompts can blur a starting camera; only pause active capture.
        if (controller?.getState().enabled) takeManualControl();
    });
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
    document.addEventListener('pointerdown', event => {
        if (stage()?.contains(event.target)) takeManualControl();
    }, true);
    document.addEventListener('wheel', event => {
        if (stage()?.contains(event.target)) takeManualControl();
    }, {capture: true, passive: true});
    document.addEventListener('click', event => {
        // Includes keyboard-activated toolbar buttons, but not camera dialog controls.
        if (event.target?.closest?.('.kg-header-actions, .kg-toolbar, #detail-focus-btn')) takeManualControl();
    }, true);
    document.addEventListener('keydown', event => {
        if (event.defaultPrevented || event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey || event.altKey) return;
        if (event.target?.closest?.("input, select, textarea, button, a, [role='textbox'], [contenteditable]:not([contenteditable='false'])")) return;
        if (event.key === 'Escape' && !dialog?.open) takeManualControl();
    });
})();
