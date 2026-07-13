(function () {
    const script = document.currentScript;
    const promoUrl = script?.dataset?.promoUrl || "/promo-graph";
    const stylesheetUrl = script?.dataset?.stylesheetUrl;

    if (stylesheetUrl && !document.querySelector(`link[href="${stylesheetUrl}"]`)) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = stylesheetUrl;
        document.head.appendChild(link);
    }

    function closeChooser() {
        document.querySelector(".promo-chooser-overlay")?.remove();
    }

    function chooseClassic() {
        closeChooser();
        if (typeof window.toggleGestureControl === "function") window.toggleGestureControl();
    }

    window.openGestureExperienceChooser = function () {
        if (document.querySelector(".promo-chooser-overlay")) return;
        const overlay = document.createElement("div");
        overlay.className = "promo-chooser-overlay";
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");
        overlay.setAttribute("aria-label", "选择知识图谱体验");
        overlay.innerHTML = `
            <div class="promo-chooser-dialog">
                <div class="promo-chooser-dialog-head">
                    <h2>选择知识图谱体验</h2>
                    <button class="promo-chooser-close" type="button" aria-label="关闭">&times;</button>
                </div>
                <div class="promo-chooser-grid">
                    <button class="promo-chooser-choice is-classic" type="button">
                        <small>01 / CLASSIC</small>
                        <strong>经典图谱与手势</strong>
                        <span>留在当前页面，使用原有调试图谱和手势控制。</span>
                    </button>
                    <button class="promo-chooser-choice is-promo" type="button">
                        <small>02 / GPU STAR MAP</small>
                        <strong>二维知识星图</strong>
                        <span>进入全量 GPU 星图，使用鼠标或触摸浏览。</span>
                    </button>
                </div>
            </div>
        `;
        overlay.addEventListener("click", event => {
            if (event.target === overlay) closeChooser();
        });
        overlay.querySelector(".promo-chooser-close")?.addEventListener("click", closeChooser);
        overlay.querySelector(".is-classic")?.addEventListener("click", chooseClassic);
        overlay.querySelector(".is-promo")?.addEventListener("click", () => window.location.assign(promoUrl));
        document.body.appendChild(overlay);
        overlay.querySelector(".is-classic")?.focus();
    };

    document.addEventListener("keydown", event => {
        if (event.key === "Escape") closeChooser();
    });
})();
