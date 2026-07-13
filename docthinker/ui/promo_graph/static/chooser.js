(function () {
    const canvas = document.getElementById("chooser-particles");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const palette = ["#f5fbfa", "#13dfcf", "#9bc7ff", "#d8ff46", "#ff9d88"];
    let width = 0;
    let height = 0;
    let nodes = [];
    let links = [];

    function resize() {
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = Math.floor(width * ratio);
        canvas.height = Math.floor(height * ratio);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

        const count = Math.max(28, Math.min(52, Math.floor(width / 28)));
        nodes = Array.from({ length: count }, (_, index) => ({
            x: (index * 83.17) % width,
            y: (index * 47.53 + 80) % height,
            vx: ((index % 5) - 2) * 0.018,
            vy: (((index * 3) % 5) - 2) * 0.014,
            radius: index % 11 === 0 ? 4.8 : index % 5 === 0 ? 2.7 : 1.8,
            color: palette[index % palette.length],
        }));
        links = nodes.slice(1).map((node, index) => ({
            source: node,
            target: nodes[(index * 7) % nodes.length],
            dashed: index % 4 === 0,
        }));
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);
        links.forEach(link => {
            ctx.beginPath();
            ctx.setLineDash(link.dashed ? [3, 8] : []);
            ctx.moveTo(link.source.x, link.source.y);
            ctx.lineTo(link.target.x, link.target.y);
            ctx.strokeStyle = link.dashed ? "rgba(155,199,255,.18)" : "rgba(245,251,250,.13)";
            ctx.lineWidth = 0.7;
            ctx.stroke();
        });
        ctx.setLineDash([]);

        nodes.forEach(node => {
            node.x = (node.x + node.vx + width) % width;
            node.y = (node.y + node.vy + height) % height;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = node.color;
            ctx.shadowBlur = node.radius > 3 ? 15 : 7;
            ctx.shadowColor = node.color;
            ctx.fill();
        });
        ctx.shadowBlur = 0;
        requestAnimationFrame(draw);
    }

    window.addEventListener("resize", resize);
    resize();
    requestAnimationFrame(draw);
})();
