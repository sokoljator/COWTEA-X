"""
scoring_enhanced.py
===================
Multi-evidence cross-feeding scoring system.

Why a new scorer?
-----------------
The classic v2.0 scorer (`scoring.py :: score_crossfeeding`) is strict
all-or-nothing: it only emits a row when *both* the producer step and
the consumer step pass FC_THRESH in the same chain. This misses several
biologically meaningful patterns:

  * Cases where 3h+5h show the cross-feed but 20h is altered/missing
    because of nutrient scarcity or a metabolic shift.
  * Asymmetric cases where one strain inside a genus (CLN-513 vs
    CLN-614, or CTR-1) supports the pattern and the other partially
    contradicts it.
  * Producer-only or consumer-only evidence with strong CV reliability
    but no triggering 4-bar mirror confirmation.

This module emits an evidence breakdown for *every* (medium, chain,
metabolite) triple so downstream filtering / plotting can decide what
to call cross-feeding for a given audience.

Evidence components (per row, all bounded contributions)
--------------------------------------------------------
The score is the *sum* of bounded sub-scores; the maximum is 12. Each
sub-score is also exported as its own column so the user can plot or
filter on individual evidence types:

    e_paired_strict      (0..4)   Classic 4-bar pattern (producer +
                                  consumer + mirror non-rise +
                                  mirror return rise)
    e_single_chain       (0..2)   3h↑ then 5h↓ (or 5h↑ then 20h↓)
                                  with no mirror requirement
    e_producer           (0..1)   Solo / partner step looks like a
                                  producer regardless of consumer
    e_consumer           (0..1)   Consumer step is clearly negative
    e_genus_partial      (0..1)   At least one other chain in the same
                                  genus pair (513/614/CTR) shows the
                                  same direction
    e_media_consistency  (0..1)   Same metabolite, same direction
                                  detected in ≥2 media (deferred:
                                  computed by the aggregator wrapper)
    e_replicate_conf     (0..1)   CV of all involved samples below
                                  the strict CV threshold
    e_late_shift         (0..1)   3h+5h pattern present *with* a 20h
                                  flip (producer→consumer or vice
                                  versa)
    p_conflict           (0..−2)  Penalty: mirror chain shows the
                                  same direction in the wrong place
                                  (raises false-positive risk)

Final cross-feeding score
-------------------------
    score = clamp(sum(e_*) + p_conflict, 0, 12)
    confidence band:
        score >= 8 → STRONG_EVIDENCE
        score >= 5 → MODERATE_EVIDENCE
        score >= 3 → WEAK_EVIDENCE
        else       → INSUFFICIENT

`pattern_label` field uses a stable short code:
    PE_FULL    — full 4-bar producer+consumer+mirror pattern
    PE_3H5H    — 3h+5h only (20h missing / altered)
    PE_5H20H   — 5h+20h only
    GENUS_HALF — genus partial: one chain supports, sister chain doesn't
    PROD_ONLY  — producer-like behaviour without a consumer
    CONS_ONLY  — consumer-like behaviour without a producer
    NONE       — no consistent direction

The function is deliberately liberal — it emits a row for any chain /
metabolite where ANY sub-component fires. Downstream code is expected
to filter on `score`.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .constants import CV_THRESH, FC_THRESH, FC_THRESH_RETURN, CV_SENTINEL
from .utils import log2fc_scalar, get_mirror_chain


# ── tunable knobs (kept LOCAL to this module; do not mutate constants) ──
W_PAIRED_STRICT     = 4
W_SINGLE_CHAIN      = 2
W_PRODUCER          = 1
W_CONSUMER          = 1
W_GENUS_PARTIAL     = 1
W_REPLICATE_CONF    = 1
W_LATE_SHIFT        = 1
W_CONFLICT_PENALTY  = 2     # subtracted

STRICT_CV = 0.30   # tighter than CV_THRESH for the replicate-confidence bonus

GENUS_PARTNERS = {
    "CLN-513p": "CLN-614p",  "CLN-614p": "CLN-513p",
    "CLN-513e": "CLN-614e",  "CLN-614e": "CLN-513e",
    "CTR-1p":   None,        "CTR-1e":   None,
}


def _val(df, key, met, default=np.nan):
    if df is None or key is None or key not in df.index or met not in df.columns:
        return default
    try:
        v = float(df.loc[key, met])
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _confidence(score: float) -> str:
    if score >= 8: return "STRONG_EVIDENCE"
    if score >= 5: return "MODERATE_EVIDENCE"
    if score >= 3: return "WEAK_EVIDENCE"
    return "INSUFFICIENT"


def score_crossfeeding_enhanced(
        mean_df: pd.DataFrame,
        cv_df:   pd.DataFrame,
        chains:  dict,
        medium:  str,
        fc_thresh:        float = FC_THRESH,
        fc_thresh_return: float = FC_THRESH_RETURN,
        cv_thresh:        float = CV_THRESH,
        emit_all: bool = False,
) -> pd.DataFrame:
    """
    Compute the multi-evidence cross-feeding score.

    Parameters
    ----------
    mean_df, cv_df : DataFrames with sample rows and metabolite columns.
    chains         : dict from build_chains(medium).
    medium         : str, label for the row.
    fc_thresh      : log2FC threshold for primary signal (default 1.0).
    fc_thresh_return : log2FC threshold for the (weaker) 20h return step.
    cv_thresh      : CV threshold for the "reliable" flag.
    emit_all       : if True, emit a row for every (chain, metabolite)
                     pair even when score == 0. Otherwise rows with no
                     evidence are dropped.

    Returns
    -------
    DataFrame with one row per (chain, metabolite) and detailed
    evidence columns. See module docstring for the schema.
    """
    rows = []
    # Pre-compute per-chain direction map for genus-partner cross check.
    # direction_map[(chain, met)] = +1 producer-leaning, -1 consumer-leaning, 0 none
    direction_map: dict[tuple[str, str], int] = {}

    # First pass — primary signals per chain × metabolite
    for chain_name, steps in chains.items():
        if len(steps) < 4:
            continue
        blank_k, solo_k, partner_k, return_k = steps

        for met in mean_df.columns:
            bv = _val(mean_df, blank_k,   met)
            sv = _val(mean_df, solo_k,    met)
            pv = _val(mean_df, partner_k, met)
            rv = _val(mean_df, return_k,  met)

            fc_s = log2fc_scalar(sv, bv)
            fc_p = log2fc_scalar(pv, sv)
            fc_r = log2fc_scalar(rv, pv)

            d = 0
            if not np.isnan(fc_s) and fc_s >= fc_thresh:
                d += 1
            if not np.isnan(fc_p) and fc_p <= -fc_thresh:
                d += 1  # consumer behaviour
            if d > 0:
                direction_map[(chain_name, met)] = d

    # Second pass — score
    for chain_name, steps in chains.items():
        if len(steps) < 4:
            continue
        blank_k, solo_k, partner_k, return_k = steps
        mirror = get_mirror_chain(chain_name)
        m_steps = chains.get(mirror, []) if mirror else []
        m_blank_k   = m_steps[0] if len(m_steps) > 0 else None
        m_solo_k    = m_steps[1] if len(m_steps) > 1 else None
        m_partner_k = m_steps[2] if len(m_steps) > 2 else None
        m_return_k  = m_steps[3] if len(m_steps) > 3 else None

        for met in mean_df.columns:
            bv  = _val(mean_df, blank_k,   met)
            sv  = _val(mean_df, solo_k,    met)
            pv  = _val(mean_df, partner_k, met)
            rv  = _val(mean_df, return_k,  met)
            mbv = _val(mean_df, m_blank_k, met)
            msv = _val(mean_df, m_solo_k,  met)
            mpv = _val(mean_df, m_partner_k, met)
            mrv = _val(mean_df, m_return_k, met)

            fc_s  = log2fc_scalar(sv, bv)
            fc_p  = log2fc_scalar(pv, sv)
            fc_r  = log2fc_scalar(rv, pv)
            fc_ms = log2fc_scalar(msv, mbv if np.isfinite(mbv) and mbv > 0 else bv)
            fc_mp = log2fc_scalar(mpv, msv)
            fc_mr = log2fc_scalar(mrv, mpv)

            cv_s = _val(cv_df, solo_k,    met, default=0.5)
            cv_p = _val(cv_df, partner_k, met, default=0.5)
            cv_r = _val(cv_df, return_k,  met, default=0.5)

            single_rep = (
                abs(cv_s - CV_SENTINEL) < 1e-3
                or abs(cv_p - CV_SENTINEL) < 1e-3
            )

            # ── e_paired_strict ─────────────────────────────────────
            paired = 0
            if not np.isnan(fc_s) and fc_s >= fc_thresh \
                    and not np.isnan(fc_p) and fc_p <= -fc_thresh:
                paired += 2
                no_mirror_solo = (
                    np.isnan(fc_ms) or abs(fc_ms) < fc_thresh or fc_ms <= 0
                )
                mirror_return_rise = (
                    not np.isnan(fc_mr) and fc_mr >= fc_thresh_return
                )
                if no_mirror_solo:
                    paired += 1
                if mirror_return_rise:
                    paired += 1
            e_paired_strict = min(paired, W_PAIRED_STRICT)

            # ── e_single_chain (3h+5h OR 5h+20h) ────────────────────
            single = 0
            if not np.isnan(fc_s) and fc_s >= fc_thresh \
                    and not np.isnan(fc_p) and fc_p <= -fc_thresh:
                single = 2
            elif not np.isnan(fc_p) and fc_p >= fc_thresh \
                    and not np.isnan(fc_r) and fc_r <= -fc_thresh:
                single = 2
            elif not np.isnan(fc_s) and fc_s >= fc_thresh \
                    and not np.isnan(fc_p) and fc_p < 0:   # softer
                single = 1
            e_single_chain = min(single, W_SINGLE_CHAIN)

            # ── e_producer ───────────────────────────────────────────
            e_producer = 0
            if (not np.isnan(fc_s) and fc_s >= fc_thresh) \
                    or (not np.isnan(fc_p) and fc_p >= fc_thresh):
                e_producer = 1

            # ── e_consumer ───────────────────────────────────────────
            e_consumer = 0
            if (not np.isnan(fc_p) and fc_p <= -fc_thresh) \
                    or (not np.isnan(fc_r) and fc_r <= -fc_thresh):
                e_consumer = 1

            # ── e_genus_partial ─────────────────────────────────────
            partner_chain = GENUS_PARTNERS.get(chain_name)
            e_genus = 0
            if partner_chain and direction_map.get((partner_chain, met), 0) >= 1:
                e_genus = 1

            # ── e_replicate_conf ────────────────────────────────────
            e_repconf = 0
            cvs_here = [c for c in (cv_s, cv_p, cv_r) if np.isfinite(c)]
            if cvs_here and max(cvs_here) <= STRICT_CV and not single_rep:
                e_repconf = 1

            # ── e_late_shift (20h flip) ─────────────────────────────
            e_lateshift = 0
            if (not np.isnan(fc_p) and abs(fc_p) >= fc_thresh
                    and not np.isnan(fc_r) and abs(fc_r) >= fc_thresh_return
                    and np.sign(fc_p) != np.sign(fc_r)):
                e_lateshift = 1

            # ── p_conflict (mirror produces in solo step too) ───────
            p_conflict = 0
            if not np.isnan(fc_ms) and fc_ms >= fc_thresh:
                # mirror's "solo" also produces — weakens specificity
                p_conflict = -W_CONFLICT_PENALTY
            elif (not np.isnan(fc_mp) and fc_mp >= fc_thresh
                  and not np.isnan(fc_p) and fc_p >= fc_thresh):
                # both chains "produce" at partner step — no exchange
                p_conflict = -W_CONFLICT_PENALTY

            score = (e_paired_strict + e_single_chain
                     + e_producer + e_consumer + e_genus
                     + e_repconf + e_lateshift + p_conflict)
            score = max(0, min(score, 12))

            # ── pattern_label ───────────────────────────────────────
            if e_paired_strict >= 3:
                pattern = "PE_FULL"
            elif (not np.isnan(fc_s) and fc_s >= fc_thresh
                  and not np.isnan(fc_p) and fc_p <= -fc_thresh):
                pattern = "PE_3H5H"
            elif (not np.isnan(fc_p) and fc_p >= fc_thresh
                  and not np.isnan(fc_r) and fc_r <= -fc_thresh):
                pattern = "PE_5H20H"
            elif e_genus and (e_producer or e_consumer):
                pattern = "GENUS_HALF"
            elif e_producer and not e_consumer:
                pattern = "PROD_ONLY"
            elif e_consumer and not e_producer:
                pattern = "CONS_ONLY"
            else:
                pattern = "NONE"

            if pattern == "NONE" and score == 0 and not emit_all:
                continue

            rows.append({
                "medium":              medium,
                "chain":               chain_name,
                "mirror_chain":        mirror,
                "metabolite":          met,
                "fc_solo":             round(fc_s, 3) if np.isfinite(fc_s) else np.nan,
                "fc_partner":          round(fc_p, 3) if np.isfinite(fc_p) else np.nan,
                "fc_return":           round(fc_r, 3) if np.isfinite(fc_r) else np.nan,
                "fc_mirror_solo":      round(fc_ms, 3) if np.isfinite(fc_ms) else np.nan,
                "fc_mirror_return":    round(fc_mr, 3) if np.isfinite(fc_mr) else np.nan,
                "cv_solo":             round(cv_s, 3) if np.isfinite(cv_s) else np.nan,
                "cv_partner":          round(cv_p, 3) if np.isfinite(cv_p) else np.nan,
                "cv_return":           round(cv_r, 3) if np.isfinite(cv_r) else np.nan,
                "e_paired_strict":     e_paired_strict,
                "e_single_chain":      e_single_chain,
                "e_producer":          e_producer,
                "e_consumer":          e_consumer,
                "e_genus_partial":     e_genus,
                "e_replicate_conf":    e_repconf,
                "e_late_shift":        e_lateshift,
                "p_conflict":          p_conflict,
                "score":               score,
                "confidence":          _confidence(score),
                "pattern_label":       pattern,
                "single_replicate":    bool(single_rep),
                "reliable":            bool(not single_rep
                                            and cv_s <= cv_thresh
                                            and cv_p <= cv_thresh),
            })

    df = pd.DataFrame(rows)

    # ── e_media_consistency post-pass ───────────────────────────────
    # Computed at the cross-media aggregation step; placeholder here.
    df["e_media_consistency"] = 0

    return df


def aggregate_media_consistency(scored_all: pd.DataFrame) -> pd.DataFrame:
    """
    Given the concatenated enhanced scoring across all media, mark
    metabolites that show the same pattern (PE_FULL / PE_3H5H /
    PE_5H20H) in ≥2 media and bump their score by `W_REPLICATE_CONF`
    via the `e_media_consistency` column.

    Returns a new DataFrame (does not mutate input).
    """
    if scored_all is None or scored_all.empty:
        return scored_all

    df = scored_all.copy()
    cf_patterns = {"PE_FULL", "PE_3H5H", "PE_5H20H"}
    is_cf = df["pattern_label"].isin(cf_patterns)
    counts = (df[is_cf]
              .groupby(["metabolite", "chain"])["medium"]
              .nunique()
              .rename("n_media_with_pattern")
              .reset_index())
    df = df.merge(counts, on=["metabolite", "chain"], how="left")
    df["n_media_with_pattern"] = df["n_media_with_pattern"].fillna(0).astype(int)
    df["e_media_consistency"] = (df["n_media_with_pattern"] >= 2).astype(int)
    df["score"] = (df["score"] + df["e_media_consistency"]).clip(upper=12)
    df["confidence"] = df["score"].apply(_confidence)
    return df
