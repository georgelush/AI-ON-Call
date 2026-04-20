"""
src/routers/router_registry.py - Auto-discovers and registers all routers from src/routers/.

A file is registered if it defines a module-level `router` that is an APIRouter instance.
Skips __init__.py and this file itself.

Usage in server.py:
    from src.routers.router_registry import include_all_routers
    include_all_routers(app)
"""
import importlib
import os
import src.routers as _routers_pkg
from fastapi import FastAPI
from fastapi.routing import APIRouter

_routers_root = os.path.dirname(_routers_pkg.__file__)
_routers_pkg_name = _routers_pkg.__name__  # "src.routers"

_SKIP = {"__init__.py", "router_registry.py"}


def include_all_routers(app: FastAPI) -> None:
    """Scan src/routers/, import every module that exposes a `router` APIRouter, and include it."""
    loaded = []
    for fname in sorted(os.listdir(_routers_root)):
        if not fname.endswith(".py") or fname in _SKIP:
            continue
        module_path = f"{_routers_pkg_name}.{fname.removesuffix('.py')}"
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            print(f"  [!] Could not load router {module_path}: {exc}")
            continue
        router = getattr(mod, "router", None)
        if not isinstance(router, APIRouter):
            continue
        app.include_router(router)
        loaded.append(router.prefix or fname)

    print(f"[RouterRegistry] Loaded {len(loaded)} router(s): {', '.join(loaded)}")
