"""
config.py
=========
Config-file loader for the pipeline. Merges a JSON file with defaults,
so that a partial config still runs. Cross-platform path handling:

  * Accepts absolute paths (Windows, Posix) or paths relative to either
    the config file's directory or the current working directory.
  * If `xlsx_path` does not exist as given, the loader searches a few
    common locations: alongside the config, in the current working
    directory, and in /home/user/workspace (used in this sandbox).

Keys understood by `run_pipeline`:
  * xlsx_path   — path to the Excel workbook (REQUIRED)
  * out_dir     — base output folder (REQUIRED). A timestamped
                  sub-folder is created on each run.
  * media       — list of sheet names to process
                  (default: AUM / DEX / GLY / HisGly / SUC)
  * fc_thresh, cv_thresh, fc_thresh_return — analysis thresholds
                  (optional overrides)
  * scorer      — "classic" (default) or "enhanced"
"""

from __future__ import annotations
from pathlib import Path
import json
import os

from .constants import MEDIA

DEFAULT_CONFIG = {
    # Will be resolved to a real path by _resolve_xlsx_path() below.
    "xlsx_path":  "Consolidated_CASIC_5media_withClass_final-14.xlsx",
    "out_dir":    "output/CF_results",
    "media":      list(MEDIA),
    "scorer":     "classic",   # or "enhanced"
}


# ── candidate roots to search for xlsx_path / relative out_dir ───────────────
def _candidate_roots(cfg_dir: Path) -> list[Path]:
    roots = []
    # 1. directory the config file lives in
    roots.append(cfg_dir)
    # 2. current working directory
    roots.append(Path.cwd())
    # 3. workspace fallback (sandbox)
    workspace = Path("/home/user/workspace")
    if workspace.exists():
        roots.append(workspace)
    # 4. env override
    env_root = os.environ.get("COWTEAX_ROOT")
    if env_root:
        roots.append(Path(env_root))
    # de-dup while preserving order
    seen = set()
    out = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(r)
    return out


def _resolve_path(raw_path: str, cfg_dir: Path, must_exist: bool = True) -> Path:
    """
    Resolve a possibly-Windows / possibly-relative path. Returns a Path.
    If `must_exist` is True, searches candidate roots; raises
    FileNotFoundError if not found. Otherwise returns the most plausible
    absolute path.
    """
    p = Path(raw_path)

    # if the path exists as given, accept it
    if p.exists():
        return p.resolve()

    # search candidate roots using the *basename* of the path (handles
    # the case where the config carries a Windows path like
    # C:/Users/.../Consolidated_CASIC_...xlsx that doesn't exist here)
    basename = p.name
    for root in _candidate_roots(cfg_dir):
        cand = (root / basename)
        if cand.exists():
            return cand.resolve()

    # also try the path as relative to each root
    for root in _candidate_roots(cfg_dir):
        cand = (root / p)
        if cand.exists():
            return cand.resolve()

    # final fuzzy fallback: match on the basename stem with optional
    # "-NN" / "_NN" suffixes (e.g. "foo.xlsx" vs "foo-14.xlsx")
    stem = p.stem
    suffix = p.suffix
    import re as _re
    for root in _candidate_roots(cfg_dir):
        if not root.exists():
            continue
        for fp in root.iterdir():
            if not fp.is_file() or fp.suffix.lower() != suffix.lower():
                continue
            fp_stem = fp.stem
            if fp_stem == stem:
                return fp.resolve()
            # match prefix with optional [-_]NN tail
            if _re.match(rf"^{_re.escape(stem)}[-_]\w+$", fp_stem):
                return fp.resolve()

    if must_exist:
        searched = "\n".join(f"  {r}" for r in _candidate_roots(cfg_dir))
        raise FileNotFoundError(
            f"Could not resolve path: {raw_path}\nSearched:\n{searched}"
        )

    # for out_dir we don't require existence — strip Windows-style
    # drive letters / absolute-prefix and put it under the first writable root
    import re as _re2
    raw_str = str(p)
    # detect Windows drive (e.g. "C:/...") or UNC ("//...")
    is_windows_abs = bool(_re2.match(r'^[A-Za-z]:[\\/]', raw_str)) \
                     or raw_str.startswith('\\\\')
    if is_windows_abs:
        # Take the trailing two segments as our output folder name
        parts = _re2.split(r'[\\/]', raw_str)
        parts = [pp for pp in parts if pp and not _re2.match(r'^[A-Za-z]:$', pp)]
        tail = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        p = Path(tail)

    for root in _candidate_roots(cfg_dir):
        if os.access(root, os.W_OK):
            return (root / p).resolve()
    return Path(raw_path).resolve()


def load_config(path: str | Path = "config.json") -> dict:
    """
    Load JSON config from `path`, merge on top of DEFAULT_CONFIG, and
    resolve xlsx_path / out_dir to absolute paths that work in this
    environment. Returns a plain dict.
    """
    cfg = DEFAULT_CONFIG.copy()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg.update(user_cfg)

    cfg_dir = cfg_path.parent.resolve()

    # Resolve xlsx_path (required, must exist)
    cfg["xlsx_path"] = str(_resolve_path(cfg["xlsx_path"], cfg_dir, must_exist=True))

    # Resolve out_dir (does not need to exist)
    cfg["out_dir"] = str(_resolve_path(cfg["out_dir"], cfg_dir, must_exist=False))

    return cfg
