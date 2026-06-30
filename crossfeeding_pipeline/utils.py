"""
utils.py
========
Small helpers shared across modules — keep this file dependency-free
(no other pipeline imports) so it can be imported from anywhere.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Vectorised log2 fold-change (pandas Series in, pandas Series out)
# ────────────────────────────────────────────────────────────────────────────
def log2fc(numerator, denominator, eps=1e-9):
    """
    Safe log2 fold-change: zero / negative numerator \u2192 NaN
    (cannot take log of non-positive); zero / negative denominator
    is replaced by eps so the quotient stays finite.
    """
    num = numerator.copy().astype(float)
    den = denominator.copy().astype(float)
    num[num <= 0] = np.nan
    den[den <= 0] = eps
    return np.log2(num / den)


# ────────────────────────────────────────────────────────────────────────────
# Scalar log2 fold-change (used by scorers that iterate met-by-met)
# ────────────────────────────────────────────────────────────────────────────
def log2fc_scalar(num, den, eps=1e-9):
    """Scalar variant. Returns NaN if inputs are unusable."""
    n = float(num) if (num is not None and
                       not (isinstance(num, float) and np.isnan(num))) else np.nan
    d = float(den) if (den is not None and
                       not (isinstance(den, float) and np.isnan(den))) else eps
    if np.isnan(n) or n <= 0:
        return np.nan
    if d <= 0:
        d = eps
    return np.log2(n / d)


# ────────────────────────────────────────────────────────────────────────────
# Validation — fail fast if a chain references samples that don't exist
# ────────────────────────────────────────────────────────────────────────────
def validate_input(mean_df, chains, medium=""):
    """
    Raise ValueError if any chain references a sample missing from mean_df.
    Prints a warning for chains where the mirror is incomplete but allows
    them to continue (mirror is not strictly required for scoring).
    """
    missing = []
    for chain, steps in chains.items():
        for s in steps:
            if s not in mean_df.index:
                missing.append((chain, s))
    if missing:
        msg = "\n".join(f"  {c}: {s}" for c, s in missing)
        raise ValueError(
            f"[{medium}] Missing samples in mean_df "
            f"(expected by build_chains):\n{msg}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Mirror-chain helper (shared by scoring + figures)
# ────────────────────────────────────────────────────────────────────────────
def get_mirror_chain(chain_name):
    """
    'CLN-513p' \u2192 'CLN-513e'   (and vice versa). Returns None if suffix
    is neither 'p' nor 'e'.
    """
    if chain_name.endswith("p"):
        return chain_name[:-1] + "e"
    if chain_name.endswith("e"):
        return chain_name[:-1] + "p"
    return None
