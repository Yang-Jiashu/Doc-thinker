"""Render DocThinker architecture and UI demo assets.

The assets are intentionally deterministic: no external design tools, no
network calls, and fixed coordinates that avoid overlap.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

INK = "#292520"
MUTED = "#766f66"
PAPER = "#fbfaf7"
PANEL = "#fffefa"
HAIRLINE = "#ded8cc"
COPPER = "#b85c38"
SAGE = "#54746b"
LAKE = "#4b6f8f"
OCHRE = "#b7892d"
PLUM = "#8a5f87"
WASH = "#f7f1ea"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F10 = font(10)
F11 = font(11)
F12 = font(12)
F13 = font(13)
F14 = font(14)
F16 = font(16, True)
F18 = font(18, True)
F22 = font(22, True)
F28 = font(28, True)
F34 = font(34, True)


def text_width(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=ft)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_width(draw, trial, ft) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: Sequence[str] = (),
    *,
    fill: str = PANEL,
    outline: str = HAIRLINE,
    accent: str = COPPER,
    title_font: ImageFont.ImageFont = F16,
    body_font: ImageFont.ImageFont = F11,
    radius: int = 14,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)
    draw.rounded_rectangle((x0, y0, x0 + 8, y1), radius=radius, fill=accent)
    draw.text((x0 + 22, y0 + 16), title, font=title_font, fill=INK)
    y = y0 + 46
    for line in body:
        for wrapped in wrap_text(draw, line, body_font, x1 - x0 - 44):
            draw.text((x0 + 22, y), wrapped, font=body_font, fill=MUTED)
            y += 18


def arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    fill: str = "#a89d90",
    width: int = 3,
) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")
    if len(points) < 2:
        return
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 10
    left = (x1 - size * math.cos(angle - math.pi / 6), y1 - size * math.sin(angle - math.pi / 6))
    right = (x1 - size * math.cos(angle + math.pi / 6), y1 - size * math.sin(angle + math.pi / 6))
    draw.polygon([(x1, y1), left, right], fill=fill)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, ft: ImageFont.ImageFont = F11, fill: str = MUTED) -> None:
    draw.rounded_rectangle((xy[0] - 8, xy[1] - 5, xy[0] + text_width(draw, text, ft) + 8, xy[1] + 17), radius=8, fill=PAPER)
    draw.text(xy, text, font=ft, fill=fill)


def render_architecture() -> None:
    img = Image.new("RGB", (1800, 1080), PAPER)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1800, 1080), fill=PAPER)
    draw.text((72, 54), "DocThinker Agentic Memory Architecture", font=F34, fill=INK)
    draw.text((74, 98), "Carrier-grounded memory that can recall, reason, consolidate, and be edited under user control.", font=F16, fill=MUTED)

    columns = [
        (60, 170, 300, 690, "Inputs", COPPER),
        (365, 170, 610, 690, "Session Runtime", LAKE),
        (690, 170, 990, 690, "AgentMemoryCore", INK),
        (1070, 170, 1340, 850, "Pluggable Backends", SAGE),
        (1430, 170, 1720, 850, "User Control + UI", OCHRE),
    ]
    for x0, y0, x1, y1, title, accent in columns:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill="#fdfbf7", outline=HAIRLINE, width=2)
        draw.text((x0 + 22, y0 + 18), title, font=F18, fill=accent)

    input_cards = [
        ("Pure Text", ["Chat, notes, instructions"]),
        ("Image-text Docs", ["PDF pages, figures, tables"]),
        ("Graph Signals", ["Entities + evidence"]),
        ("Edit Commands", ["NL memory edits"]),
    ]
    for i, (title, body) in enumerate(input_cards):
        card(draw, (88, 230 + i * 102, 272, 306 + i * 102), title, body, accent=COPPER, title_font=F14)

    runtime_cards = [
        ("Session Manager", ["Session graph + files"]),
        ("Query Router", ["Quick / Standard / Deep"]),
        ("Memory Policy", ["Scopes, top-k, write controls"]),
        ("Memory Trace", ["Plan, hits, reasoning, writes"]),
    ]
    for i, (title, body) in enumerate(runtime_cards):
        card(draw, (390, 230 + i * 102, 585, 306 + i * 102), title, body, accent=LAKE, title_font=F14)

    card(
        draw,
        (720, 245, 960, 390),
        "Recall Before Generation",
        [
            "Build recall plan",
            "Retrieve durable insights",
            "Attach expanded KG hypotheses",
            "Derive memory-side reasoning",
        ],
        fill="#fff9f2",
        accent=COPPER,
        title_font=F16,
    )
    card(
        draw,
        (720, 470, 960, 615),
        "Consolidate After Response",
        [
            "Write only allowed layers",
            "Promote useful hypotheses",
            "Audit write/skip decision",
            "Update long-horizon memory",
        ],
        fill="#f5f8f5",
        accent=SAGE,
        title_font=F16,
    )
    draw.rounded_rectangle((758, 406, 922, 454), radius=24, fill=INK)
    draw.text((790, 419), "Generation", font=F16, fill="#fffefa")

    backend_cards = [
        ("Conversation", ["Claw working/core/archive"]),
        ("Episodic", ["Neuro memory analogies"]),
        ("Long-horizon", ["Durable insights + reasoning"]),
        ("Expanded KG", ["Candidate hypotheses"]),
        ("Graph Promotion", ["Validated graph writes"]),
        ("Tri-Graph", ["Causal DAG + flow + SEAL"]),
    ]
    for i, (title, body) in enumerate(backend_cards):
        card(draw, (1100, 230 + i * 96, 1310, 300 + i * 96), title, body, accent=[LAKE, PLUM, COPPER, OCHRE, SAGE, INK][i], title_font=F14)

    ui_cards = [
        ("Memory Inspector", ["Recall plans and matches"]),
        ("KG Dashboard", ["Nodes, edges, lifecycle"]),
        ("NL Memory Editor", ["Preview candidates, then confirm"]),
        ("Highlighting", ["Selected nodes and edges glow"]),
        ("MEMORY.md Export", ["Portable audit index"]),
    ]
    for i, (title, body) in enumerate(ui_cards):
        card(draw, (1460, 230 + i * 108, 1692, 306 + i * 108), title, body, accent=[LAKE, SAGE, COPPER, OCHRE, PLUM][i], title_font=F14)

    # Main forward flow.
    arrow(draw, [(300, 430), (365, 430)], fill=COPPER)
    arrow(draw, [(610, 430), (690, 430)], fill=LAKE)
    arrow(draw, [(960, 318), (1070, 318)], fill=COPPER)
    arrow(draw, [(960, 542), (1070, 542)], fill=SAGE)
    arrow(draw, [(1340, 430), (1430, 430)], fill=OCHRE)
    arrow(draw, [(840, 390), (840, 406)], fill=INK)
    arrow(draw, [(840, 454), (840, 470)], fill=INK)

    # Feedback/control loop around the bottom.
    draw.rounded_rectangle((365, 770, 990, 940), radius=22, fill="#fffefa", outline=HAIRLINE, width=2)
    draw.text((395, 800), "Agentic Memory Loop", font=F22, fill=INK)
    loop_items = [
        ("1. Recall", 410, 850, COPPER),
        ("2. Reason", 560, 850, LAKE),
        ("3. Answer", 715, 850, INK),
        ("4. Consolidate", 860, 850, SAGE),
    ]
    for title, x, y, color in loop_items:
        draw.rounded_rectangle((x, y, x + 110, y + 44), radius=20, fill="#f7f1ea", outline=color, width=2)
        draw.text((x + 16, y + 13), title, font=F13, fill=color)
    arrow(draw, [(520, 872), (560, 872)], fill="#a89d90", width=2)
    arrow(draw, [(670, 872), (715, 872)], fill="#a89d90", width=2)
    arrow(draw, [(825, 872), (860, 872)], fill="#a89d90", width=2)
    arrow(draw, [(915, 850), (915, 720), (815, 720), (815, 615)], fill=SAGE, width=2)

    label(draw, (316, 402), "carriers become memory events")
    label(draw, (626, 402), "policy-guided recall")
    label(draw, (990, 286), "read")
    label(draw, (990, 510), "write")
    label(draw, (1360, 402), "inspect + edit")

    draw.text((72, 1000), "No-overlap layout: fixed columns, orthogonal arrows, and separated read/write loops.", font=F13, fill=MUTED)
    img.save(ASSET_DIR / "agentic_memory_architecture.png", quality=95)


def draw_node(draw: ImageDraw.ImageDraw, x: int, y: int, label_text: str, color: str, *, selected: bool = False) -> None:
    r = 32
    if selected:
        draw.ellipse((x - r - 7, y - r - 7, x + r + 7, y + r + 7), fill="#f4d7c8", outline=COPPER, width=4)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="#fffefa", width=3)
    draw.text((x - text_width(draw, label_text, F12) // 2, y - 7), label_text, font=F12, fill="#fffefa")


def draw_ui_frame(step: int = 2) -> Image.Image:
    img = Image.new("RGB", (1600, 960), "#f7f1ea")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1600, 70), fill=PANEL)
    draw.line((0, 70, 1600, 70), fill=HAIRLINE, width=2)
    draw.text((42, 17), "DocThinker Memory Atlas", font=F12, fill=COPPER)
    draw.text((42, 36), "知识图谱", font=F22, fill=INK)
    draw.rounded_rectangle((1230, 18, 1335, 52), radius=8, fill="#f5e6d3", outline="#dfb7a5")
    draw.text((1260, 27), "扩展", font=F13, fill="#5f2e1f")
    draw.rounded_rectangle((1360, 18, 1468, 52), radius=8, fill=COPPER)
    draw.text((1390, 27), "刷新", font=F13, fill="#fffefa")

    # Canvas and toolbar.
    draw.rectangle((0, 70, 1220, 960), fill="#faf8f5")
    draw.rounded_rectangle((26, 94, 296, 315), radius=14, fill="#fffefa", outline=HAIRLINE, width=2)
    draw.rounded_rectangle((45, 114, 278, 150), radius=8, fill=PANEL, outline=HAIRLINE)
    draw.text((62, 123), "搜索节点...", font=F13, fill="#9b9287")
    for i, name in enumerate(["Domain", "Concept", "Instance", "扩展节点"]):
        x = 45 + (i % 2) * 115
        y = 178 + (i // 2) * 46
        draw.rounded_rectangle((x, y, x + 102, y + 32), radius=8, fill="#fffefa", outline="#ebe4d8")
        draw.text((x + 22, y + 9), name, font=F11, fill=MUTED)

    # Graph edges.
    nodes = {
        "Input": (410, 310, COPPER),
        "Core": (620, 255, LAKE),
        "Long": (795, 360, SAGE),
        "KG": (635, 520, OCHRE),
        "UI": (910, 520, PLUM),
    }
    edges = [("Input", "Core"), ("Core", "Long"), ("Core", "KG"), ("Long", "UI"), ("KG", "UI")]
    selected_edge = ("Long", "UI") if step >= 2 else None
    for src, tgt in edges:
        x0, y0, _ = nodes[src]
        x1, y1, _ = nodes[tgt]
        is_selected = selected_edge == (src, tgt)
        draw.line((x0, y0, x1, y1), fill=COPPER if is_selected else "#a89d90", width=6 if is_selected else 2)
        if is_selected:
            mx, my = (x0 + x1) // 2, (y0 + y1) // 2
            draw.rounded_rectangle((mx - 55, my - 18, mx + 55, my + 18), radius=18, fill="#fbede6", outline=COPPER, width=2)
            draw.text((mx - 34, my - 7), "selected", font=F11, fill=COPPER)

    for name, (x, y, color) in nodes.items():
        draw_node(draw, x, y, name, color, selected=(step >= 2 and name in {"Long", "UI"}))

    # Right dashboard.
    draw.rectangle((1220, 70, 1600, 960), fill="#fffefa")
    draw.line((1220, 70, 1220, 960), fill=HAIRLINE, width=2)
    draw.text((1250, 100), "Memory / KG Dashboard", font=F18, fill=INK)
    draw.text((1250, 128), "当前会话的图谱、记忆与扩展节点状态。", font=F11, fill=MUTED)

    def mini_card(x: int, y: int, w: int, h: int, title: str, value: str, color: str = INK) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill="#f7f3ea", outline=HAIRLINE)
        draw.text((x + 12, y + 10), title, font=F11, fill=MUTED)
        draw.text((x + 12, y + 30), value, font=F22, fill=color)

    mini_card(1250, 168, 92, 72, "实体", "42", INK)
    mini_card(1354, 168, 92, 72, "关系", "68", LAKE)
    mini_card(1458, 168, 92, 72, "扩展", "12", OCHRE)

    y = 265
    draw.rounded_rectangle((1250, y, 1550, y + 210), radius=8, fill=PANEL, outline=HAIRLINE)
    draw.text((1266, y + 16), "Natural Language Memory Edit", font=F14, fill=INK)
    draw.rounded_rectangle((1492, y + 13, 1534, y + 36), radius=12, fill="#fbede6", outline="#d7a28b")
    draw.text((1502, y + 18), "edit", font=F10, fill=COPPER)
    draw.rounded_rectangle((1266, y + 50, 1534, y + 100), radius=8, fill="#fffefa", outline=HAIRLINE)
    command = "把 UI 风格的长期记忆改成 Claude-like，并高亮相关节点和边"
    draw.text((1278, y + 61), command, font=F11, fill=INK)
    draw.rounded_rectangle((1266, y + 114, 1394, y + 144), radius=8, fill="#fffefa", outline=HAIRLINE)
    draw.text((1304, y + 123), "预览", font=F12, fill=INK)
    draw.rounded_rectangle((1406, y + 114, 1534, y + 144), radius=8, fill="#fffefa", outline=HAIRLINE)
    draw.text((1444, y + 123), "清除", font=F12, fill=INK)
    if step >= 1:
        draw.rounded_rectangle((1266, y + 160, 1534, y + 194), radius=8, fill="#faf7ef", outline="#ebe4d8")
        draw.text((1278, y + 170), "project_state · score 0.92 · UI style memory", font=F11, fill=INK)

    y2 = 500
    draw.rounded_rectangle((1250, y2, 1550, y2 + 245), radius=8, fill=PANEL, outline=HAIRLINE)
    draw.text((1266, y2 + 16), "Long-horizon Memory", font=F14, fill=INK)
    for i, text in enumerate([
        "DocThinker UI should be Claude-like, artful, restrained, and control-first.",
        "Memory edits require preview, graph highlight, and user confirmation.",
        "Secrets and temporary debug traces are skipped by default.",
    ]):
        yy = y2 + 52 + i * 56
        draw.rounded_rectangle((1266, yy, 1534, yy + 44), radius=8, fill="#faf7ef", outline="#ebe4d8")
        draw.text((1276, yy + 10), text[:54], font=F10, fill=MUTED if i else INK)

    if step >= 3:
        draw.rounded_rectangle((590, 685, 950, 735), radius=12, fill="#f5f8f5", outline=SAGE, width=2)
        draw.text((625, 701), "Memory updated. Selected edge and nodes stay highlighted.", font=F14, fill=SAGE)

    return img


def render_ui_assets() -> None:
    screenshot = draw_ui_frame(step=2)
    screenshot.save(ASSET_DIR / "ui_memory_editor_screenshot.png", quality=95)
    frames = [draw_ui_frame(step=i) for i in (0, 1, 2, 3)]
    frames[0].save(
        ASSET_DIR / "memory_edit_operation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[900, 900, 1100, 1200],
        loop=0,
        optimize=True,
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    render_architecture()
    render_ui_assets()


if __name__ == "__main__":
    main()
