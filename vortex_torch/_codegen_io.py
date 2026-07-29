"""Safe filesystem helpers for generated Python modules."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Optional


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "vortex"


def _cache_root(configured: Optional[str]) -> Path:
    if configured:
        root = Path(configured).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    env_root = os.environ.get("VORTEX_CODEGEN_CACHE_DIR")
    root = (
        Path(env_root).expanduser()
        if env_root
        else Path.home() / ".cache" / "vortex_torch" / "generated"
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = Path(tempfile.gettempdir()) / "vortex_torch" / "generated"
        root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def write_generated_module(
    *,
    cache_dir: Optional[str],
    flow_name: str,
    namespace: str,
    source: str,
) -> str:
    """Atomically publish a content-addressed generated Python module.

    Separate namespaces prevent indexer and cache modules from overwriting each
    other. The source digest keeps different model/config variants apart, while
    ``os.replace`` ensures concurrent compiler processes only observe complete
    files.
    """

    root = _cache_root(cache_dir)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    filename = (
        f"{_safe_stem(flow_name)}_{_safe_stem(namespace)}_"
        f"{digest}_compiled_func.py"
    )
    destination = root / filename

    fd, temporary = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=str(root)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return str(destination)
