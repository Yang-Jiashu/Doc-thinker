"""Render DocThinker architecture and UI demo assets.

The assets are intentionally deterministic: no external design tools, no
network calls, and fixed coordinates that avoid overlap.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


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


def load_logo(size: int) -> Image.Image | None:
    """Load the project dog mark and crop excess white padding for README demos."""
    for path in (ROOT / "logo.png", ROOT / "docthinker" / "ui" / "static" / "logo.png"):
        if not path.exists():
            continue
        logo = Image.open(path).convert("RGBA")
        # The historical logo is a PNG with a white canvas. Trim that canvas so
        # the dog mark survives inside compact UI mockups.
        white = Image.new("RGBA", logo.size, (255, 255, 255, 255))
        diff = ImageChops.difference(logo, white)
        bbox = diff.getbbox()
        if bbox:
            logo = logo.crop(bbox)
        return ImageOps.contain(logo, (size, size), Image.Resampling.LANCZOS)
    return None


def paste_logo(img: Image.Image, xy: tuple[int, int], size: int) -> None:
    logo = load_logo(size)
    draw = ImageDraw.Draw(img)
    x, y = xy
    draw.rounded_rectangle((x, y, x + size, y + size), radius=14, fill="#fffefa", outline="#ded8cc", width=2)
    if logo is None:
        draw.text((x + 12, y + 18), "DT", font=F18, fill=COPPER)
        return
    px = x + (size - logo.width) // 2
    py = y + (size - logo.height) // 2
    img.alpha_composite(logo, (px, py))


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
    img = Image.new("RGB", (2200, 1300), PAPER)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 2200, 1300), fill=PAPER)

    title = "DocThinker Agentic Memory Architecture"
    subtitle = "text + image-text carriers · recall planning · memory-side reasoning · controlled consolidation · graph evolution"
    draw.text(((2200 - text_width(draw, title, F34)) // 2, 54), title, font=F34, fill=INK)
    draw.text(((2200 - text_width(draw, subtitle, F16)) // 2, 104), subtitle, font=F16, fill=MUTED)

    # Main pipeline row. Cards are separated by generous gutters; arrows stay in
    # those gutters only, so no connector crosses text or card bodies.
    pipeline = [
        ((80, 205, 390, 405), "Input Carriers", ["Pure text", "Image-text interactive docs", "Graph signals", "Edit commands"], COPPER),
        ((470, 205, 780, 405), "Session Runtime", ["Session-scoped KG", "Query API + Web UI", "MemoryPolicy", "MemoryTrace"], LAKE),
        ((860, 185, 1320, 425), "AgentMemoryCore", ["recall(): plan + merge + reason", "generation context assembly", "after_response(): audit + write"], INK),
        ((1400, 205, 1710, 405), "Generation", ["RAG / deep thinking", "sources + memory metadata", "observable answer trace"], PLUM),
        ((1790, 205, 2100, 405), "Answer + Control", ["Generated response", "remember / exclude layers", "selected memory edits"], OCHRE),
    ]
    for xy, title_text, body, accent in pipeline:
        card(draw, xy, title_text, body, fill="#fdfbf7", accent=accent, title_font=F18, body_font=F13, radius=18)

    for left, right, color in [
        (pipeline[0][0], pipeline[1][0], COPPER),
        (pipeline[1][0], pipeline[2][0], LAKE),
        (pipeline[2][0], pipeline[3][0], INK),
        (pipeline[3][0], pipeline[4][0], OCHRE),
    ]:
        arrow(draw, [(left[2] + 14, 305), (right[0] - 14, 305)], fill=color, width=4)

    # Core phases inside the central card; no lines pass through them.
    phase_y = 345
    phases = [
        ("Recall Plan", 895, COPPER),
        ("Layer Merge", 1015, SAGE),
        ("Memory Trace", 1135, LAKE),
        ("Policy Control", 1255, PLUM),
    ]
    for text, x, color in phases:
        draw.rounded_rectangle((x, phase_y, x + 102, phase_y + 38), radius=19, fill=PANEL, outline=color, width=2)
        draw.text((x + 13, phase_y + 11), text, font=F11, fill=color)

    # Backend lane. A single bus below the core fans out vertically; this avoids
    # diagonal connector clutter and keeps every backend card readable.
    draw.rounded_rectangle((210, 545, 1990, 805), radius=22, fill="#fffefa", outline=HAIRLINE, width=2)
    draw.text((250, 575), "Pluggable Memory + Graph Backends", font=F22, fill=SAGE)
    backend_cards = [
        ((250, 655, 500, 755), "Conversation", ["Claw working/core/archive"], LAKE),
        ((545, 655, 795, 755), "Episodic", ["Neuro analogy episodes"], PLUM),
        ((840, 655, 1090, 755), "Long-horizon", ["Durable insights + reasoning"], COPPER),
        ((1135, 655, 1385, 755), "Expanded KG", ["Candidate hypotheses"], OCHRE),
        ((1430, 655, 1680, 755), "GraphCore KG", ["Promoted durable structure"], SAGE),
        ((1725, 655, 1950, 755), "Tri-Graph", ["Causal DAG + flow + SEAL"], INK),
    ]
    bus_y = 620
    draw.line((375, bus_y, 1838, bus_y), fill="#9d9488", width=3)
    arrow(draw, [(1090, 425), (1090, bus_y - 4)], fill=INK, width=3)
    for xy, title_text, body, accent in backend_cards:
        x_mid = (xy[0] + xy[2]) // 2
        draw.line((x_mid, bus_y, x_mid, xy[1]), fill="#9d9488", width=2)
        arrow(draw, [(x_mid, xy[1] - 22), (x_mid, xy[1])], fill="#9d9488", width=2)
        card(draw, xy, title_text, body, fill="#fdfbf7", accent=accent, title_font=F16, body_font=F12, radius=12)

    # Controlled consolidation uses a separate lower channel, never crossing
    # the backend labels.
    draw.rounded_rectangle((620, 850, 1580, 1020), radius=22, fill="#fdfbf7", outline=HAIRLINE, width=2)
    draw.text((660, 882), "Agentic Memory Loop", font=F22, fill=INK)
    loop_steps = [
        ("1 Recall", 675, 935, COPPER),
        ("2 Reason", 850, 935, LAKE),
        ("3 Answer", 1030, 935, INK),
        ("4 Consolidate", 1210, 935, SAGE),
        ("5 Edit / Export", 1420, 935, OCHRE),
    ]
    for step, x, y, color in loop_steps:
        draw.rounded_rectangle((x, y, x + 130, y + 46), radius=23, fill=PANEL, outline=color, width=2)
        draw.text((x + 18, y + 14), step, font=F13, fill=color)
    for i in range(len(loop_steps) - 1):
        x0 = loop_steps[i][1] + 130
        x1 = loop_steps[i + 1][1]
        arrow(draw, [(x0 + 8, 958), (x1 - 8, 958)], fill="#a89d90", width=2)
    arrow(draw, [(1270, 805), (1270, 850)], fill=SAGE, width=3)
    arrow(draw, [(1540, 935), (1540, 825), (1610, 825), (1610, 805)], fill=OCHRE, width=3)

    # UI/control lane. These are outputs of the loop, so connectors use only
    # vertical drops into the lane.
    draw.rounded_rectangle((210, 1080, 1990, 1230), radius=22, fill="#fffefa", outline=HAIRLINE, width=2)
    draw.text((250, 1114), "User-Controlled Operations", font=F22, fill=OCHRE)
    ui_cards = [
        ((545, 1135, 795, 1205), "Memory Inspector", ["recall plans + matches"], LAKE),
        ((840, 1135, 1090, 1205), "KG Dashboard", ["node / edge operations"], SAGE),
        ((1135, 1135, 1385, 1205), "NL Memory Editor", ["preview -> highlight -> confirm"], COPPER),
        ((1430, 1135, 1680, 1205), "Audit Export", ["MEMORY.md + trace"], PLUM),
    ]
    for xy, title_text, body, accent in ui_cards:
        card(draw, xy, title_text, body, fill="#fdfbf7", accent=accent, title_font=F14, body_font=F11, radius=10)
    arrow(draw, [(1540, 1020), (1540, 1080)], fill=OCHRE, width=3)

    draw.text(
        (80, 1260),
        "Layout rule: no diagonal crossings, no arrows through text, and each operation surface has a dedicated lane.",
        font=F13,
        fill=MUTED,
    )
    img.save(ASSET_DIR / "agentic_memory_architecture.png", quality=95)


def draw_node(draw: ImageDraw.ImageDraw, x: int, y: int, label_text: str, color: str, *, selected: bool = False) -> None:
    r = 32
    if selected:
        draw.ellipse((x - r - 7, y - r - 7, x + r + 7, y + r + 7), fill="#f4d7c8", outline=COPPER, width=4)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="#fffefa", width=3)
    draw.text((x - text_width(draw, label_text, F12) // 2, y - 7), label_text, font=F12, fill="#fffefa")


def draw_ui_frame(step: int = 2) -> Image.Image:
    img = Image.new("RGBA", (1600, 960), "#f7f1ea")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1600, 960), fill="#f7f1ea")

    # Persistent app chrome. This mirrors base_modern.html and keeps the dog
    # logo visible in both the screenshot and the README GIF.
    draw.rectangle((0, 0, 248, 960), fill="#fffefa")
    draw.line((248, 0, 248, 960), fill=HAIRLINE, width=2)
    paste_logo(img, (28, 26), 58)
    draw.text((100, 33), "DocThinker", font=F18, fill=INK)
    draw.text((100, 58), "Agentic memory", font=F11, fill=MUTED)
    draw.text((28, 116), "Workspace", font=F10, fill="#9b9287")
    nav_items = [("智能对话", 145, False), ("知识图谱", 190, True), ("设置", 235, False)]
    for label_text, y, active in nav_items:
        fill = "#f5eee4" if active else "#fffefa"
        outline = "#e8ded0" if active else "#fffefa"
        draw.rounded_rectangle((20, y, 228, y + 36), radius=9, fill=fill, outline=outline)
        draw.ellipse((38, y + 12, 50, y + 24), fill=COPPER if active else "#c9c0b5")
        draw.text((66, y + 10), label_text, font=F13, fill=INK if active else MUTED)
    draw.rounded_rectangle((24, 820, 224, 912), radius=14, fill="#fbfaf7", outline=HAIRLINE)
    draw.text((42, 842), "Memory Policy", font=F13, fill=INK)
    draw.text((42, 868), "controllable writes", font=F11, fill=MUTED)

    # Top bar.
    draw.rectangle((248, 0, 1600, 72), fill=PANEL)
    draw.line((248, 72, 1600, 72), fill=HAIRLINE, width=2)
    draw.text((286, 16), "DocThinker Memory Atlas", font=F12, fill=COPPER)
    draw.text((286, 36), "知识图谱", font=F22, fill=INK)
    draw.rounded_rectangle((1278, 18, 1383, 52), radius=8, fill="#fffefa", outline="#ded8cc")
    draw.text((1310, 27), "扩展", font=F13, fill=INK)
    draw.rounded_rectangle((1408, 18, 1516, 52), radius=8, fill=COPPER)
    draw.text((1440, 27), "刷新", font=F13, fill="#fffefa")

    # Canvas and toolbar.
    draw.rectangle((248, 72, 1208, 960), fill="#faf8f5")
    draw.rounded_rectangle((278, 100, 548, 342), radius=14, fill="#fffefa", outline=HAIRLINE, width=2)
    draw.text((302, 123), "Graph Controls", font=F14, fill=INK)
    draw.rounded_rectangle((302, 154, 526, 190), radius=8, fill=PANEL, outline=HAIRLINE)
    draw.text((319, 163), "搜索节点...", font=F13, fill="#9b9287")
    for i, name in enumerate(["Domain", "Concept", "Instance", "扩展节点"]):
        x = 302 + (i % 2) * 110
        y = 216 + (i // 2) * 46
        draw.rounded_rectangle((x, y, x + 96, y + 32), radius=8, fill="#fffefa", outline="#ebe4d8")
        draw.text((x + 17, y + 9), name, font=F11, fill=MUTED)

    # Graph edges.
    nodes = {
        "Input": (620, 315, COPPER),
        "Core": (800, 240, LAKE),
        "Long": (980, 352, SAGE),
        "KG": (755, 548, OCHRE),
        "UI": (1005, 580, PLUM),
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
    draw.rectangle((1208, 72, 1600, 960), fill="#fffefa")
    draw.line((1208, 72, 1208, 960), fill=HAIRLINE, width=2)
    draw.text((1238, 102), "Memory / KG Dashboard", font=F18, fill=INK)
    draw.text((1238, 130), "当前会话的图谱、记忆与扩展节点状态。", font=F11, fill=MUTED)

    def mini_card(x: int, y: int, w: int, h: int, title: str, value: str, color: str = INK) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill="#f7f3ea", outline=HAIRLINE)
        draw.text((x + 12, y + 10), title, font=F11, fill=MUTED)
        draw.text((x + 12, y + 30), value, font=F22, fill=color)

    mini_card(1238, 168, 92, 72, "实体", "42", INK)
    mini_card(1342, 168, 92, 72, "关系", "68", LAKE)
    mini_card(1446, 168, 92, 72, "扩展", "12", OCHRE)

    y = 265
    draw.rounded_rectangle((1238, y, 1562, y + 238), radius=10, fill=PANEL, outline=HAIRLINE)
    draw.text((1256, y + 16), "Natural Language Memory Edit", font=F14, fill=INK)
    draw.rounded_rectangle((1504, y + 13, 1546, y + 36), radius=12, fill="#fbede6", outline="#d7a28b")
    draw.text((1514, y + 18), "edit", font=F10, fill=COPPER)
    draw.rounded_rectangle((1256, y + 52, 1546, y + 112), radius=8, fill="#fffefa", outline=HAIRLINE)
    command = "把 UI 风格的长期记忆改成 Claude-like，并高亮相关节点和边"
    for i, line in enumerate(wrap_text(draw, command, F11, 260)):
        draw.text((1268, y + 63 + i * 18), line, font=F11, fill=INK)
    preview_fill = COPPER if step >= 1 else "#fffefa"
    preview_text = "#fffefa" if step >= 1 else INK
    draw.rounded_rectangle((1256, y + 126, 1396, y + 158), radius=8, fill=preview_fill, outline="#d7a28b")
    draw.text((1304, y + 135), "预览", font=F12, fill=preview_text)
    draw.rounded_rectangle((1406, y + 126, 1546, y + 158), radius=8, fill="#fffefa", outline=HAIRLINE)
    draw.text((1452, y + 135), "清除", font=F12, fill=INK)
    if step >= 1:
        draw.rounded_rectangle((1256, y + 176, 1546, y + 216), radius=8, fill="#faf7ef", outline=COPPER if step >= 2 else "#ebe4d8", width=2 if step >= 2 else 1)
        draw.text((1268, y + 188), "project_state · score 0.92 · UI style memory", font=F11, fill=INK)

    y2 = 500
    draw.rounded_rectangle((1238, y2, 1562, y2 + 272), radius=10, fill=PANEL, outline=HAIRLINE)
    draw.text((1256, y2 + 16), "Long-horizon Memory", font=F14, fill=INK)
    for i, text in enumerate([
        "DocThinker UI should be Claude-like, artful, restrained, and control-first.",
        "Memory edits require preview, graph highlight, and user confirmation.",
        "Secrets and temporary debug traces are skipped by default.",
    ]):
        yy = y2 + 52 + i * 66
        selected = step >= 2 and i == 0
        draw.rounded_rectangle((1256, yy, 1546, yy + 54), radius=8, fill="#fff8f1" if selected else "#faf7ef", outline=COPPER if selected else "#ebe4d8", width=2 if selected else 1)
        for j, line in enumerate(wrap_text(draw, text, F10, 254)[:2]):
            draw.text((1268, yy + 10 + j * 15), line, font=F10, fill=INK if selected else MUTED)

    if step >= 3:
        draw.rounded_rectangle((590, 705, 1015, 760), radius=12, fill="#f5f8f5", outline=SAGE, width=2)
        draw.text((625, 723), "Memory updated. Selected edge and nodes stay highlighted.", font=F14, fill=SAGE)

    return img.convert("RGB")


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
