"""Export installation sizes from the unchanged original DocThinker dog logo.

Run with a tooling environment containing Pillow. This is mechanical resize /
letterboxing only: no AI, tracing, recoloring, crop or change to frozen assets.
Builds consume the committed PNGs; they do not regenerate or replace branding.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "c3b36dd3de1e8ebcb222fcdfc724139ed7b82a16f218763a71bba8e1832b90df"
APP_SIZES = (20, 29, 40, 48, 58, 60, 72, 76, 80, 87, 96, 120, 144,
             152, 167, 180, 192, 512, 1024)
DESTINATIONS = ("clients/mobile/src/static/original-brand",
                "clients/mobile/public/original-brand")


def _png(source: Image.Image, size: int, *, inset: bool = False) -> bytes:
    # A 56%-wide square lies inside the maskable 80%-diameter safe circle.
    extent = int(size * 0.56) if inset else size
    fitted = ImageOps.contain(source, (extent, extent), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), "white")
    canvas.alpha_composite(fitted, ((size - fitted.width) // 2,
                                   (size - fitted.height) // 2))
    buffer = BytesIO()
    canvas.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def generate(root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    source_path = root / "logo.png"
    if source_path.is_symlink():
        raise ValueError("Original logo must not be a symlink")
    original = source_path.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Original logo bytes changed; do not regenerate a replacement")
    with Image.open(BytesIO(original)) as image:
        source = image.convert("RGBA")
    outputs = {"logo.png": original}
    outputs.update({f"logo-{size}.png": _png(source, size)
                    for size in (16, 32, 64, 256)})
    outputs.update({f"icon-{size}.png": _png(source, size) for size in APP_SIZES})
    outputs["icon-maskable-512.png"] = _png(source, 512, inset=True)
    catalog = {"brand": "DocThinker", "identity": "original-dog",
               "source": "logo.png", "source_sha256": SOURCE_SHA256,
               "transformation": "contain-on-white; no crop, redraw or recolor",
               "maskable_safe_square_fraction": 0.56,
               "files": {name: hashlib.sha256(body).hexdigest()
                         for name, body in sorted(outputs.items())}}
    outputs["catalog.json"] = (json.dumps(catalog, indent=2) + "\n").encode()
    for relative in DESTINATIONS:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        if directory.resolve() != directory:
            raise ValueError("Original-brand output must not traverse symlinks")
        for name, body in outputs.items():
            destination = directory / name
            if destination.is_symlink():
                raise ValueError("Original-brand output must not be a symlink")
            destination.write_bytes(body)
    return {"source_sha256": SOURCE_SHA256, "files_per_directory": len(outputs),
            "destinations": list(DESTINATIONS)}


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
