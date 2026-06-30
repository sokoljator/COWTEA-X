"""
cli.py
======
Command-line entry-point:  `python -m crossfeeding_pipeline.cli --config X.json`

The previous draft was already correct; it's kept here for parity and
to be explicit about imports.
"""

import argparse
from pathlib import Path

from .config   import load_config
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Cross-feeding GC-MS pipeline"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config file"
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    # Intentional duplication with run_pipeline.py — see note there.
    config = load_config(cfg_path)
    run_pipeline(config)


if __name__ == "__main__":
    main()
