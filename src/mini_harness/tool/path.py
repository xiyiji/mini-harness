"""Path guards. Every file tool goes through one of these two functions.

The model is told "reads stay inside the workspace, writes stay inside
sandbox/", but it is these functions that make it true. The model can only
hand us strings; we resolve them and check.
"""

import fnmatch
from pathlib import Path

from mini_harness.config import CONFIG


def resolve_path(file_path: Path | str, cfg=CONFIG) -> Path:
    """Relative paths are relative to the workspace. Symlinks and `..` are resolved."""
    path = Path(file_path)
    if not path.is_absolute():
        path = cfg.work_space / path
    return path.resolve()


def is_denied(real_path: Path, cfg=CONFIG) -> bool:
    """True for secrets: matches a deny_name pattern or sits under a deny_dir."""
    if any(part in cfg.deny_dir for part in real_path.parts[:-1]):
        return True
    return any(fnmatch.fnmatch(real_path.name, pat) for pat in cfg.deny_name)


def validate_read(file_path: Path | str, cfg=CONFIG) -> Path:
    real = resolve_path(file_path, cfg=cfg)
    if not cfg.guard_read:
        return real
    if not real.is_relative_to(cfg.work_space):
        raise PermissionError(f"[access denied]: reads are limited to {cfg.work_space}, got {real}")
    if is_denied(real, cfg=cfg):
        raise PermissionError(f"[access denied]: {real} looks like a credential file")
    return real


def validate_write(file_path: Path | str, cfg=CONFIG) -> Path:
    real = resolve_path(file_path, cfg=cfg)
    if not cfg.guard_write:
        return real
    if not real.is_relative_to(cfg.sandbox_dir):
        raise PermissionError(f"[access denied]: writes are limited to {cfg.sandbox_dir}, got {real}")
    return real
