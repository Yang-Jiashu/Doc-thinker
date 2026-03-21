from __future__ import annotations

from typing import Any, Optional

from .state import state


def get_session_memory_engine(session_id: Optional[str]) -> Optional[Any]:
    if not session_id:
        return None

    factory = getattr(state, "memory_engine_factory", None)
    if factory is None:
        return None

    cache = getattr(state, "memory_engines", None)
    if cache is None:
        state.memory_engines = {}
        cache = state.memory_engines
    lock = getattr(state, "memory_engine_lock", None)

    if lock:
        with lock:
            engine = cache.get(session_id)
            if engine is not None:
                return engine

    engine = factory(session_id)
    if engine is None:
        return None

    if lock:
        with lock:
            existing = cache.get(session_id)
            if existing is not None:
                return existing
            cache[session_id] = engine
    else:
        cache[session_id] = engine
    return engine


def get_session_claw_manager(session_id: Optional[str]) -> Optional[Any]:
    """Get or create a ClawMemoryManager for a session."""
    if not session_id:
        return None

    factory = getattr(state, "claw_manager_factory", None)
    if factory is None:
        return None

    cache = getattr(state, "claw_managers", None)
    if cache is None:
        state.claw_managers = {}
        cache = state.claw_managers

    mgr = cache.get(session_id)
    if mgr is not None:
        return mgr

    mgr = factory(session_id)
    if mgr is not None:
        cache[session_id] = mgr
    return mgr


def remove_session_memory_engine(session_id: Optional[str], *, save_before_remove: bool = True) -> None:
    if not session_id:
        return
    cache = getattr(state, "memory_engines", None)
    if not isinstance(cache, dict):
        return
    lock = getattr(state, "memory_engine_lock", None)
    engine = None
    if lock:
        with lock:
            engine = cache.pop(session_id, None)
    else:
        engine = cache.pop(session_id, None)
    if save_before_remove and engine is not None:
        try:
            engine.save()
        except Exception:
            pass

    claw_cache = getattr(state, "claw_managers", None)
    if isinstance(claw_cache, dict):
        mgr = claw_cache.pop(session_id, None)
        if save_before_remove and mgr is not None:
            try:
                mgr.save()
            except Exception:
                pass


def save_all_memory_engines() -> None:
    cache = getattr(state, "memory_engines", None)
    if isinstance(cache, dict) and cache:
        lock = getattr(state, "memory_engine_lock", None)
        engines = []
        if lock:
            with lock:
                engines = list(cache.values())
        else:
            engines = list(cache.values())
        for engine in engines:
            try:
                engine.save()
            except Exception:
                continue

    claw_cache = getattr(state, "claw_managers", None)
    if isinstance(claw_cache, dict) and claw_cache:
        for mgr in list(claw_cache.values()):
            try:
                mgr.save()
            except Exception:
                continue
