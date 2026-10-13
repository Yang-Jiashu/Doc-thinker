"""Verify the accepted release preserves the original presentation and interfaces.

Requires a Git checkout containing the known baseline; reads tracked source only.
No provider calls, settings, credentials or runtime databases are accessed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

BASELINE = "f48589ec117318a2277b40a72c2b5692560bd018"
ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (
    "docs/assets/", "docthinker/ui/", "scripts/brain_memory_web/",
    "claw/", "neuro_memory/",
)
FIXED = frozenset({
    "logo.png", "LICENSE", "scripts/brain_memory_demo.py", "scripts/memory_demo.py",
    "docs/PROJECT_STRUCTURE.md", "docs/ECLRR_V4_ARCHITECTURE_IMAGE_PROMPT.md",
    "docthinker/memory_core/protocols.py",
})
READMES = ("README.md", "README.zh-CN.md")


def git(*args: str, root: Path = ROOT) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def verify(root: Path = ROOT, baseline: str = BASELINE) -> tuple[int, int]:
    root = root.resolve(strict=True)
    git("rev-parse", "--verify", f"{baseline}^{{commit}}", root=root)
    paths = git("ls-tree", "-r", "--name-only", "-z", baseline, root=root).decode().split("\0")
    protected = [name for name in paths if name and
                 (name in FIXED or name.startswith(PREFIXES))]
    if not protected or not all(name in paths for name in READMES):
        raise ValueError("Incomplete preservation baseline")
    failures = []
    for name in (*protected, *READMES):
        target = root / name
        if target.is_symlink() or not target.is_file() or target.resolve() != target:
            failures.append(name)
            continue
        before = git("show", f"{baseline}:{name}", root=root)
        after = target.read_bytes()
        if not (after.startswith(before) if name in READMES else after == before):
            failures.append(name)
    if failures:
        raise ValueError("Protected content changed: " + ", ".join(failures))
    return len(protected), len(READMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()
    try:
        count, readmes = verify(baseline=args.baseline)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Preservation check failed: {error}", file=sys.stderr)
        return 1
    print(f"Preserved {count} original files byte-for-byte; {readmes} README bodies are full prefixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
