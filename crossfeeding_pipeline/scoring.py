"""
scoring.py
==========
Evidence scoring for cross-feeding. Produces ONE row per (medium, chain,
metabolite) where any evidence for cross-feeding is detected in either
direction.

This module unifies two scorers that were separate in v2.0:

    * Pattern-based 0-4 score     (original score_crossfeeding)
    * Probabilistic case label    (PGM-based assign_cf_case)

They share identical inputs and their outputs travel together in the
result DataFrame. All downstream figures / Excel sheets that previously
consumed the pattern-based columns continue to work, while the new PGM
fields (p_prod_A, p_cons_B, p_prod_B, cf_probability, cf_case,
classification) add richer evidence.

Score rules (both directions)
─────────────────────────────
Direction A  — "produced by solo, consumed by partner"      (max +4)
    +1  fc_solo    >= +FC_THRESH                (producer rises alone)
    +1  fc_partner <= -FC_THRESH                (drops when partner arrives)
    +1  mirror:  fc_mirror_solo does NOT rise   (partner-specific source)
    +1  mirror:  fc_mirror_return >= FC_THRESH_RETURN  (return producer regains X)

Direction B  — "produced by partner, consumed on 20h return" (max +3)
    +1  fc_partner >= +FC_THRESH                (partner produces)
    +1  fc_return  <= -FC_THRESH                (solo consumes it at 20h)
    +1  mirror:  fc_mirror_solo does NOT rise   (same specificity check)

Category is the usual 3-tier banding:
    score >= 4  \u2192  TRUE_CROSSFEED
    score >= 2  \u2192  ORDER_DEPENDENT
    else        \u2192  OPPORTUNISTIC
"""

import numpy as np
import pandas as pd

from .constants import (
    CV_THRESH, FC_THRESH, FC_THRESH_RETURN, CV_SENTINEL,
)
from .utils import log2fc_scalar, get_mirror_chain
from .pgm import infer_states, compute_cf_probability

# PGM posterior probability thresholds for assign_cf_case.
# These are tuned to the FC_THRESH = 1.0 (2-fold) scale:
# a Gaussian centred on MU_PROD=1.0 with sigma~CV gives P>0.70 only
# when the observed FC is clearly above the noise floor.
PGM_HIGH = 0.70   # strong evidence for active state (produced / consumed)
PGM_MED  = 0.50   # moderate evidence
PGM_LOW  = 0.40   # below this = state considered inactive
PGM_CONS = 0.60   # slightly relaxed bar for consumption (noisier signal)

# ────────────────────────────────────────────────────────────────────────────
# Small classification helpers
# ────────────────────────────────────────────────────────────────────────────
def classify_score(cf_prob):
    """Human-readable confidence label from raw CF probability."""
    if cf_prob > 0.7:  return "HIGH_CONFIDENCE"
    if cf_prob > 0.4:  return "MEDIUM_CONFIDENCE"
    return "LOW_CONFIDENCE"


def _cf_case_to_category(case):
    """
    Map CF case string to the 3-tier biological category. Cases that do not
    fit any of the named patterns (UNCLASSIFIED / SHIFT-*) fall into the
    OPPORTUNISTIC bucket for plotting, but keep their true case label in
    the cf_case column.
    """
    if case in ("CF-P\u2194E", "CF-P\u2192E", "CF-E\u2192P"):
        return "TRUE_CROSSFEED"
    if case in ("IND-E-byP", "IND-P-byE", "DEP-byE", "DEP-byP"):
        return "ORDER_DEPENDENT"
    return "OPPORTUNISTIC"


def assign_cf_case(p_prod_A, p_cons_B, p_prod_B):
    """
    Map the three latent probabilities to a biological CF case label.
    Priority order (highest first) matches the v2.0 master classification.
    Thresholds are defined by PGM_HIGH / PGM_MED / PGM_LOW / PGM_CONS above.
    """
    if p_prod_A > PGM_HIGH and p_cons_B > PGM_HIGH and p_prod_B > PGM_HIGH:
        return "CF-P\u2194E"
    if p_prod_A > PGM_HIGH and p_cons_B > PGM_HIGH and p_prod_B < PGM_LOW:
        return "CF-P\u2192E"
    if p_prod_B > PGM_HIGH and p_prod_A < PGM_LOW  and p_cons_B < PGM_LOW:
        return "CF-E\u2192P"
    if p_prod_A < PGM_LOW  and p_cons_B > PGM_CONS and p_prod_B < PGM_LOW:
        return "IND-E-byP"
    if p_prod_A < PGM_LOW  and p_cons_B > PGM_CONS and p_prod_B > PGM_MED:
        return "DEP-byE"
    if p_prod_B > PGM_HIGH and p_prod_A < PGM_LOW  and p_cons_B < 0.30:
        return "IND-P-byE"
    if p_prod_B < PGM_LOW  and p_prod_A < PGM_LOW  and p_cons_B < 0.30:
        return "DEP-byP"
    return "UNCLASSIFIED"


# ────────────────────────────────────────────────────────────────────────────
# Internal safe lookup
# ────────────────────────────────────────────────────────────────────────────
def _val(df, key, met, default=np.nan):
    """Safely pull df.loc[key, met] as a Python float, returning `default`
    if any of key / met / conversion fails."""
    if df is None or key is None:
        return default
    if key not in df.index or met not in df.columns:
        return default
    try:
        return float(df.loc[key, met])
    except (TypeError, ValueError):
        return default


# ────────────────────────────────────────────────────────────────────────────
# Main scorer
# ────────────────────────────────────────────────────────────────────────────
def score_crossfeeding(mean_df, cv_df, chains, medium,
                       fc_thresh=FC_THRESH,
                       fc_thresh_return=FC_THRESH_RETURN,
                       cv_thresh=CV_THRESH):
    """
    Hybrid pattern + PGM scorer. Returns a DataFrame with a single row per
    (chain, metabolite) for which either direction passed its trigger
    conditions. If no chain triggers a direction for a metabolite, that
    metabolite is silently skipped (same behaviour as v2.0).
    """
    rows = []

    for chain_name, steps in chains.items():
        if len(steps) < 4:
            continue
        blank_key, solo_key, partner_key, return_key = steps

        # Mirror chain (may be None if chain isn't a p/e variant)
        mirror = get_mirror_chain(chain_name)
        m_steps = chains.get(mirror, []) if mirror else []
        m_blank_key   = m_steps[0] if len(m_steps) > 0 else None
        m_solo_key    = m_steps[1] if len(m_steps) > 1 else None
        m_partner_key = m_steps[2] if len(m_steps) > 2 else None
        m_return_key  = m_steps[3] if len(m_steps) > 3 else None

        # Common feature set — columns present in BOTH frames
        feats = [c for c in mean_df.columns
                 if (cv_df is None) or (c in cv_df.columns)]

        for met in feats:
            bv  = _val(mean_df, blank_key,   met)
            sv  = _val(mean_df, solo_key,    met)
            pv  = _val(mean_df, partner_key, met)
            rv  = _val(mean_df, return_key,  met)
            mbv = _val(mean_df, m_blank_key, met)
            msv = _val(mean_df, m_solo_key,  met)
            mrv = _val(mean_df, m_return_key, met)
            mpv = _val(mean_df, m_partner_key, met)

            fc_s  = log2fc_scalar(sv, bv)
            fc_p  = log2fc_scalar(pv, sv)
            fc_r  = log2fc_scalar(rv, pv)
            fc_ms = log2fc_scalar(msv, mbv if (not np.isnan(mbv) and mbv > 0) else bv)
            fc_mr = log2fc_scalar(mrv, mpv) if m_return_key else np.nan

            cv_s = _val(cv_df, solo_key,    met, default=0.0)
            cv_p = _val(cv_df, partner_key, met, default=0.0)
            cv_m = _val(cv_df, m_solo_key,  met, default=0.5)

            # Sentinel = single-replicate — flag but don't penalise.
            single_rep = (abs(cv_s - CV_SENTINEL) < 1e-3
                          or abs(cv_p - CV_SENTINEL) < 1e-3)
            reliable = (cv_s <= cv_thresh
                        and cv_p <= cv_thresh
                        and not single_rep)

            # ── PGM posterior (independent of pattern matching) ─────────────
            p_prod_A, p_cons_B, p_prod_B = infer_states(
                fc_s, fc_p, fc_ms, cv_s, cv_p, cv_m
            )
            cf_prob = compute_cf_probability(p_prod_A, p_cons_B, p_prod_B)
            cf_case = assign_cf_case(p_prod_A, p_cons_B, p_prod_B)

            # ── Direction A: produced at solo, consumed at 5h partner ──────
            if (not np.isnan(fc_s) and fc_s >= fc_thresh
                    and not np.isnan(fc_p) and fc_p <= -fc_thresh):
                no_mirror_solo = (np.isnan(fc_ms)
                                  or abs(fc_ms) < fc_thresh
                                  or fc_ms <= 0)
                mirror_rise = (not np.isnan(fc_mr)
                               and fc_mr >= fc_thresh_return)
                score = 2 + int(no_mirror_solo) + int(mirror_rise)
                cat = _cf_case_to_category(cf_case)
                rows.append({
                    "medium":             medium,
                    "chain":              chain_name,
                    "metabolite":         met,
                    "direction":          "consumed_by_partner",
                    "fc_solo":            round(fc_s, 3),
                    "fc_partner":         round(fc_p, 3),
                    "fc_return":          round(fc_r, 3) if not np.isnan(fc_r) else np.nan,
                    "score":              score,
                    "category":           cat,
                    "no_mirror_solo":     bool(no_mirror_solo),
                    "mirror_rise_return": bool(mirror_rise),
                    "cv_solo":            round(cv_s, 3),
                    "cv_partner":         round(cv_p, 3),
                    "reliable":           bool(reliable),
                    "log2FC_step":        round(fc_p, 3),
                    "step_label":         "5h cross-feed",
                    "CV_current":         round(cv_p, 3),
                    "single_replicate":   bool(single_rep),
                    # PGM additions
                    "p_prod_A":           round(p_prod_A, 3),
                    "p_cons_B":           round(p_cons_B, 3),
                    "p_prod_B":           round(p_prod_B, 3),
                    "cf_probability":     round(cf_prob, 3),
                    "cf_case":            cf_case,
                    "classification":     classify_score(cf_prob),
                })

            # ── Direction B: partner produces, solo consumes on 20h return ─
            if (not np.isnan(fc_p) and fc_p >= fc_thresh
                    and not np.isnan(fc_r) and fc_r <= -fc_thresh):
                no_mirror_solo = (np.isnan(fc_ms) or abs(fc_ms) < fc_thresh)
                score = 2 + int(no_mirror_solo)
                cat = _cf_case_to_category(cf_case)
                rows.append({
                    "medium":             medium,
                    "chain":              chain_name,
                    "metabolite":         met,
                    "direction":          "produced_by_partner",
                    "fc_solo":            round(fc_s, 3) if not np.isnan(fc_s) else np.nan,
                    "fc_partner":         round(fc_p, 3),
                    "fc_return":          round(fc_r, 3),
                    "score":              score,
                    "category":           cat,
                    "no_mirror_solo":     bool(no_mirror_solo),
                    "mirror_rise_return": False,
                    "cv_solo":            round(cv_s, 3),
                    "cv_partner":         round(cv_p, 3),
                    "reliable":           bool(reliable),
                    "log2FC_step":        round(fc_r, 3),
                    "step_label":         "20h return",
                    "CV_current":         round(cv_p, 3),
                    "single_replicate":   bool(single_rep),
                    "p_prod_A":           round(p_prod_A, 3),
                    "p_cons_B":           round(p_cons_B, 3),
                    "p_prod_B":           round(p_prod_B, 3),
                    "cf_probability":     round(cf_prob, 3),
                    "cf_case":            cf_case,
                    "classification":     classify_score(cf_prob),
                })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Back-compat alias so existing code that imports score_crossfeeding_pgm
# keeps working.
# ────────────────────────────────────────────────────────────────────────────
score_crossfeeding_pgm = score_crossfeeding
