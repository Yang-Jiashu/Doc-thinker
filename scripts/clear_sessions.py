"""清除所有会话数据 + Python 缓存 + 运行时临时文件。

用法:
    python scripts/clear_sessions.py             # 交互确认后清除
    python scripts/clear_sessions.py --force      # 跳过确认直接清除
    python scripts/clear_sessions.py --cache-only # 只清缓存，不删会话
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KEEP_DIRS = {"_system"}
SYSTEM_LEGACY = ["legacy_backup", "legacy_knowledge_bases", "legacy_memory", "knowledge_bases"]
SYSTEM_DB = DATA_DIR / "_system" / "knowledge_base.db"
SYSTEM_SESSIONS_DIR = DATA_DIR / "_system" / "sessions"


def get_session_dirs():
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and p.name not in KEEP_DIRS
    )


def find_legacy_system_dirs():
    """_system 下的历史遗留目录"""
    system_dir = DATA_DIR / "_system"
    targets = []
    for name in SYSTEM_LEGACY:
        p = system_dir / name
        if p.is_dir():
            targets.append(p)
    return targets


def find_cache_dirs():
    """__pycache__, .pyc, output/, neuro_agent_data/, token_stats 等"""
    targets = []
    for d in ROOT.rglob("__pycache__"):
        if ".git" not in d.parts and "openclaw" not in d.parts:
            targets.append(d)
    for name in ["output", "neuro_agent_data", "context_dumps"]:
        p = ROOT / name
        if p.is_dir():
            targets.append(p)
    return targets


def find_cache_files():
    targets = []
    for p in ROOT.glob("token_stats.jsonl"):
        targets.append(p)
    for p in ROOT.rglob("*.pyc"):
        if ".git" not in p.parts and "openclaw" not in p.parts:
            targets.append(p)
    return targets


def main():
    force = "--force" in sys.argv
    cache_only = "--cache-only" in sys.argv

    sessions = [] if cache_only else get_session_dirs()
    legacy_dirs = find_legacy_system_dirs()
    cache_dirs = find_cache_dirs()
    cache_files = find_cache_files()

    total = len(sessions) + len(legacy_dirs) + len(cache_dirs) + len(cache_files)
    if total == 0:
        print("没有找到需要清除的内容。")
        return

    if sessions:
        print(f"会话目录: {len(sessions)} 个")
        for s in sessions[:5]:
            print(f"  {s.name}/")
        if len(sessions) > 5:
            print(f"  ... 还有 {len(sessions) - 5} 个")

    if legacy_dirs:
        print(f"历史遗留: {len(legacy_dirs)} 个")
        for d in legacy_dirs:
            print(f"  _system/{d.name}/")

    if cache_dirs:
        print(f"缓存目录: {len(cache_dirs)} 个")
        for d in cache_dirs[:5]:
            print(f"  {d.relative_to(ROOT)}/")
        if len(cache_dirs) > 5:
            print(f"  ... 还有 {len(cache_dirs) - 5} 个")

    if cache_files:
        print(f"缓存文件: {len(cache_files)} 个")

    if not force:
        answer = input(f"\n确认删除以上 {total} 项？(y/N): ").strip().lower()
        if answer != "y":
            print("已取消。")
            return

    removed = 0
    for s in sessions:
        try:
            shutil.rmtree(s)
            removed += 1
        except Exception as e:
            print(f"删除 {s.name} 失败: {e}")

    if sessions:
        _reset_system_registry()

    for d in legacy_dirs:
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception as e:
            print(f"删除 _system/{d.name} 失败: {e}")

    for d in cache_dirs:
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception as e:
            print(f"删除 {d.relative_to(ROOT)} 失败: {e}")

    for f in cache_files:
        if not f.exists():
            removed += 1
            continue
        try:
            f.unlink()
            removed += 1
        except Exception as e:
            print(f"删除 {f.name} 失败: {e}")

    print(f"\n已清除 {removed}/{total} 项。")


def _reset_system_registry():
    """Delete and recreate the _system registry so session IDs restart from #00001."""
    if SYSTEM_DB.exists():
        try:
            SYSTEM_DB.unlink()
            print("  已重置 _system/knowledge_base.db")
        except Exception as e:
            print(f"  重置 knowledge_base.db 失败: {e}")

    if SYSTEM_SESSIONS_DIR.exists():
        try:
            shutil.rmtree(SYSTEM_SESSIONS_DIR)
            SYSTEM_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            print("  已清空 _system/sessions/")
        except Exception as e:
            print(f"  清空 sessions/ 失败: {e}")

    kb_global = DATA_DIR / "_system" / "knowledge_bases" / "global.json"
    if kb_global.exists():
        try:
            kb_global.unlink()
            print("  已删除 _system/knowledge_bases/global.json")
        except Exception:
            pass


if __name__ == "__main__":
    main()
