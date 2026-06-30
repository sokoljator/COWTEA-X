"""
crossfeeding_pipeline
=====================
Modular cross-feeding GC-MS metabolomics pipeline (COWTEA-X).

Cleaned/normalized package re-exporting the public API of the original
modules. The original suffixed files (analysis-3.py, scoring-12.py, etc.)
are preserved alongside this package; this package contains importable,
runnable copies plus several improvements:

  * Path handling is cross-platform (Windows/macOS/Linux), driven from
    config or auto-detected relative to the workspace.
  * Visualisation module patched to remove the broken
    `from constants import CHAIN_LABELS` lines (which crashed on
    import outside the source folder).
  * New abundance-bars plotter (`viz_abundance.py`) with correct y-axis
    formatting, CASIC labels, class labels and chain-name brackets.
  * Expanded multi-evidence scoring (`scoring_enhanced.py`).
  * New debug / reporting module (`debug_report.py`).
  * Optional Tkinter GUI helpers for figure selection and methods-writer.

Public API
----------
    from crossfeeding_pipeline import (
        run_pipeline, build_chains, load_config,
        load_sheet,
        normalize_is, compute_stats, raw_sample_means,
        find_candidates,
        score_crossfeeding, score_crossfeeding_pgm,
        score_crossfeeding_enhanced,
        infer_states, compute_cf_probability,
        export_excel,
        log2fc, log2fc_scalar, get_mirror_chain, validate_input,
        fig_abundance_bars,
        DebugReport,
    )
"""

from .pipeline       import run_pipeline, build_chains
from .config         import load_config
from .io             import load_sheet
from .preprocessing  import normalize_is, compute_stats, raw_sample_means
from .analysis       import find_candidates
from .scoring        import score_crossfeeding, score_crossfeeding_pgm
from .scoring_enhanced import score_crossfeeding_enhanced
from .pgm            import infer_states, compute_cf_probability
from .export         import export_excel
from .utils          import log2fc, log2fc_scalar, get_mirror_chain, validate_input
from .viz_abundance  import fig_abundance_bars
from .debug_report   import DebugReport

__all__ = [
    "run_pipeline", "build_chains", "load_config",
    "load_sheet",
    "normalize_is", "compute_stats", "raw_sample_means",
    "find_candidates",
    "score_crossfeeding", "score_crossfeeding_pgm",
    "score_crossfeeding_enhanced",
    "infer_states", "compute_cf_probability",
    "export_excel",
    "log2fc", "log2fc_scalar", "get_mirror_chain", "validate_input",
    "fig_abundance_bars",
    "DebugReport",
]
