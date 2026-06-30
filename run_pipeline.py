"""
run_pipeline.py
===============
Workspace entry-point for the COWTEA-X pipeline.

Run with:
    python run_pipeline.py                 # uses ./config-2.json
    python run_pipeline.py path/to.json

The script makes sure the workspace is on sys.path so the
`crossfeeding_pipeline` package imports work no matter where it is
called from.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossfeeding_pipeline import load_config, run_pipeline


def main(argv: list[str]) -> int:
    cfg_path = Path(argv[1]) if len(argv) > 1 else ROOT / "config-2.json"
    if not cfg_path.exists():
        # Try the normalized name too
        alt = ROOT / "config.json"
        if alt.exists():
            cfg_path = alt
    cfg = load_config(cfg_path)
    print(f"Config: {cfg_path.resolve()}")
    print(f"  xlsx_path: {cfg['xlsx_path']}")
    print(f"  out_dir:   {cfg['out_dir']}")
    print(f"  scorer:    {cfg.get('scorer', 'classic')}")
    out_dir = run_pipeline(cfg)
    print(f"\nResults: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
