"""
analysis.py
===========
Simple step-by-step |log2FC| candidate detector.

This is the *original* (pattern-free) detection used by:
    * fig_crossfeeding_summary (bubble plot)
    * the "CrossFeeding_Candidates" sheet in the Excel export

It does NOT implement cross-feeding evidence scoring — that lives in
scoring.py. Both detectors run side-by-side in run_pipeline.
"""

import numpy as np
import pandas as pd

from .constants import CV_THRESH, FC_THRESH, STEP_LABELS
from .utils import log2fc


def find_candidates(mean_df, cv_df, chains,
                    fc_thresh=FC_THRESH, cv_thresh=CV_THRESH):
    """
    Flag any metabolite whose |log2FC| between two consecutive steps of a
    chain is >= fc_thresh. Returns a DataFrame with the full schema used
    by downstream figures and by the Excel export.

    Columns
    -------
    chain, step, step_label, from, to, metabolite,
    log2FC_step, log2FC_blank, direction, CV_current, reliable
    """
    records = []
    for chain_name, steps in chains.items():
        blank_name = steps[0] if steps else None
        for step_i in range(1, len(steps)):
            curr_name = steps[step_i]
            prev_name = steps[step_i - 1]

            if curr_name not in mean_df.index or prev_name not in mean_df.index:
                continue

            curr = mean_df.loc[curr_name]
            prev = mean_df.loc[prev_name]
            fc_step = log2fc(curr, prev)

            if blank_name and blank_name in mean_df.index:
                fc_blank = log2fc(curr, mean_df.loc[blank_name])
            else:
                fc_blank = pd.Series(np.nan, index=fc_step.index)

            if cv_df is not None and curr_name in cv_df.index:
                cv_curr = cv_df.loc[curr_name]
            else:
                cv_curr = pd.Series(0.0, index=fc_step.index)

            for met, fc_val in fc_step.items():
                if np.isnan(fc_val) or abs(fc_val) < fc_thresh:
                    continue

                cv_val = float(cv_curr.get(met, 0.0))
                fb_val = fc_blank.get(met, np.nan)
                records.append({
                    "chain":         chain_name,
                    "step":          step_i,
                    "step_label":    STEP_LABELS[step_i] if step_i < len(STEP_LABELS) else str(step_i),
                    "from":          prev_name,
                    "to":            curr_name,
                    "metabolite":    met,
                    "log2FC_step":   round(float(fc_val), 3),
                    "log2FC_blank":  (round(float(fb_val), 3)
                                      if not np.isnan(fb_val) else np.nan),
                    "direction":     "consumed" if fc_val < 0 else "produced/accumulated",
                    "CV_current":    round(cv_val, 3),
                    "reliable":      bool(cv_val <= cv_thresh),
                })

    return pd.DataFrame(records)
