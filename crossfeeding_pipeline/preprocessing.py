"""
preprocessing.py
================
IS normalisation + per-sample mean/CV from technical replicates.

Design decisions (v2.0, see memory 2026-04-09):

1. IS normalisation uses the **geometric mean** of the stable IS, not the
   arithmetic mean. GC-MS peak areas follow a log-normal distribution and
   multiplicative drift is best corrected by dividing each sample by
   exp(mean(log(IS_i))). Refs: Lisec 2006 Plant J; Dunn 2011 Nat Protoc.

2. A stability filter drops any IS whose CV across all samples in the
   sheet exceeds 25 % — an unstable IS would amplify noise instead of
   correcting it.

3. Zero peak areas are treated as missing (NaN), not as true zero
   concentration. GC-MS cannot return a legitimate 0.

4. A single-replicate sample gets CV = sqrt(2) (CV_SENTINEL) — flagged
   so downstream scorers can skip CV-based penalties for it.
"""

import numpy as np
import pandas as pd

from .constants import CV_SENTINEL


# ────────────────────────────────────────────────────────────────────────────
# IS normalisation
# ────────────────────────────────────────────────────────────────────────────
def normalize_is(raw_df, is_feats, bio_feats,
                 sheet_name="", stability_cv_max=0.25, verbose=True):
    """
    Geometric-mean-of-stable-IS normalisation.

    Parameters
    ----------
    raw_df : DataFrame (sample x feature, tech-rep rows kept separately)
    is_feats : list[str]
        All (IS)-prefixed columns found in the sheet.
    bio_feats : list[str]
        Non-IS columns — these are the ones we actually return after division.
    sheet_name : str
        For diagnostics only.
    stability_cv_max : float
        IS is considered stable if its CV across all samples <= this value.
    verbose : bool
        Print which IS were kept/dropped.

    Returns
    -------
    norm_df   : DataFrame (same row index as raw_df, bio_feats columns only)
    stable_is : list[str] — IS columns actually used for normalisation
    """
    is_present = [c for c in is_feats if c in raw_df.columns]

    if not is_present:
        if verbose:
            print(f"  Warning [{sheet_name}]: no IS columns found — "
                  f"data NOT IS-normalised")
        return raw_df[bio_feats].copy(), []

    is_data = raw_df[is_present].replace(0, np.nan)

    # ── Stability filter ────────────────────────────────────────────────────
    mean_is = is_data.mean(axis=0)
    std_is  = is_data.std(axis=0)
    # Guard: mean can legitimately be 0 only if the whole column is NaN already
    is_cv_global = (std_is / mean_is.replace(0, np.nan)).abs()

    stable_is = [c for c in is_present
                 if np.isfinite(is_cv_global.get(c, np.inf))
                 and is_cv_global[c] <= stability_cv_max]

    if not stable_is:
        if verbose:
            print(f"  Warning [{sheet_name}]: all IS failed CV <= "
                  f"{int(stability_cv_max * 100)} % stability filter — "
                  f"using all {len(is_present)} IS as fallback")
        stable_is = is_present
    else:
        dropped = sorted(set(is_present) - set(stable_is))
        if dropped and verbose:
            print(f"  [{sheet_name}] Unstable IS excluded "
                  f"(CV > {int(stability_cv_max * 100)} %): {dropped}")

    # ── Geometric mean per sample: exp(mean(log(IS_i))) ─────────────────────
    log_is     = np.log(is_data[stable_is])
    geomean_is = np.exp(log_is.mean(axis=1))
    geomean_is = geomean_is.replace(0, np.nan)

    norm_df = raw_df[bio_feats].div(geomean_is, axis=0)

    if verbose:
        print(f"  [{sheet_name}] IS normalisation: geometric mean of "
              f"{len(stable_is)}/{len(is_present)} stable IS "
              f"\u2192 {stable_is}")

    return norm_df, stable_is


# ────────────────────────────────────────────────────────────────────────────
# Per-unique-sample mean / CV across technical replicates
# ────────────────────────────────────────────────────────────────────────────
def compute_stats(df):
    """
    Collapse technical replicates into per-sample statistics.

    Parameters
    ----------
    df : DataFrame
        IS-normalised values with replicate rows sharing the same index label.
        Each unique index value may appear 1 or more times (technical reps).

    Returns
    -------
    mean_df : DataFrame
        Per-sample mean across replicates.
    cv_df : DataFrame
        Coefficient of variation per feature. Single-replicate samples receive
        CV_SENTINEL (sqrt(2)) instead of a real CV.
    sd_df : DataFrame
        Standard deviation per feature (0.0 for single-replicate samples).

    .. versionchanged:: 2.1
        Now returns a 3-tuple (mean_df, cv_df, sd_df).
        Previously returned (mean_df, cv_df). Update any caller that
        unpacked only two values, e.g.:
            mean_df, cv_df, sd_df = compute_stats(norm_df)  # correct
            mean_df, cv_df = compute_stats(norm_df)          # will raise ValueError
    """
    unique_samples = list(dict.fromkeys(df.index.tolist()))
    mean_rows, cv_rows, sd_rows = {}, {}, {}

    for samp in unique_samples:
        rows = df.loc[df.index == samp]
        if len(rows) >= 2:
            m  = rows.mean(axis=0)
            s  = rows.std(axis=0, ddof=1)
            cv = (s / m.abs()).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        elif len(rows) == 1:
            m  = rows.iloc[0]
            s  = pd.Series(0.0, index=m.index)   # single replicate → SD = 0
            cv = pd.Series(CV_SENTINEL, index=m.index)
        else:
            continue
        mean_rows[samp] = m
        cv_rows[samp]   = cv
        sd_rows[samp]   = s                      # ← collect SD

    mean_df = pd.DataFrame(mean_rows).T
    cv_df   = pd.DataFrame(cv_rows).T
    sd_df   = pd.DataFrame(sd_rows).T            # ← build SD DataFrame
    return mean_df, cv_df, sd_df                 # ← return it


# ────────────────────────────────────────────────────────────────────────────
# QC helper — raw per-sample means (before IS normalisation) for plots
# ────────────────────────────────────────────────────────────────────────────
def raw_sample_means(raw_df, is_feats, bio_feats):
    """
    Per-unique-sample means of raw (pre-normalisation) values. Used only
    by fig_normalisation_qc to show before/after comparisons.

    Returns (raw_is_meandf, raw_bio_meandf). Either can be an empty
    DataFrame if the corresponding feature list is empty.
    """
    is_present = [c for c in is_feats if c in raw_df.columns]
    unique_samples = list(dict.fromkeys(raw_df.index.tolist()))

    raw_is_rows, raw_bio_rows = {}, {}
    for samp in unique_samples:
        rows = raw_df.loc[raw_df.index == samp]
        if len(rows) < 1:
            continue
        if is_present:
            raw_is_rows[samp] = rows[is_present].replace(0, np.nan).mean(axis=0)
        raw_bio_rows[samp]    = rows[bio_feats].replace(0, np.nan).mean(axis=0)

    raw_is_meandf  = pd.DataFrame(raw_is_rows).T  if raw_is_rows  else pd.DataFrame()
    raw_bio_meandf = pd.DataFrame(raw_bio_rows).T if raw_bio_rows else pd.DataFrame()
    return raw_is_meandf, raw_bio_meandf
