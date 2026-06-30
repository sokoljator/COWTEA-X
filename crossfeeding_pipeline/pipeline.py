"""
pipeline.py
===========
Top-level orchestration. For each medium:

    load → IS-normalise → mean/CV → candidate detection → classic
    scoring → enhanced multi-evidence scoring → figures → Excel export
    → debug-report writing → cross-media aggregation

Highlights vs the original `pipeline-10.py`:

  * Uses `viz_abundance.fig_abundance_bars` (the fixed plotter) instead
    of `visualization.fig_abundance_bars`.
  * Calls `scoring_enhanced.score_crossfeeding_enhanced` in parallel
    with the classic scorer; both outputs are exported.
  * Threads a `DebugReport` through every stage and writes
    `debug_report.{json,md}` at the end of the run.
  * `build_chains` is unchanged — the compound sample-key convention
    matches the workbook headers (`{med}_5H_Tmp05_Tme13`, etc.).
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd

from .constants import MEDIA
from .io             import load_sheet
from .preprocessing  import normalize_is, compute_stats, raw_sample_means
from .analysis       import find_candidates
from .scoring        import score_crossfeeding
from .scoring_enhanced import (score_crossfeeding_enhanced,
                                aggregate_media_consistency)
from .utils          import validate_input, get_mirror_chain
from .export         import export_excel
from .viz_abundance  import fig_abundance_bars
from .debug_report   import DebugReport
from . import visualization as viz

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Chain builder
# ─────────────────────────────────────────────────────────────────────────────
def build_chains(med: str) -> dict[str, list[str]]:
    """
    Ordered dict: chain_name → [blank, 3h, 5h, 20h] sample keys.

    Chain IDs are the user-facing display names (not internal strains):
      CLN-513p = Tmp05 → Tme13 → Tmp05      (Pseudomonas first)
      CLN-513e = Tme13 → Tmp05 → Tme13      (E. coli first)
      CLN-614p = Tmp06 → Tme14 → Tmp06
      CLN-614e = Tme14 → Tmp06 → Tme14
      CTR-1p   = PA01  → K12  → PA01        (lab-strain controls)
      CTR-1e   = K12   → PA01 → K12
    """
    return {
        "CLN-513p": [f"{med}_medium", f"{med}_3H_Tmp05",
                     f"{med}_5H_Tmp05_Tme13", f"{med}_20H_Tmp05_Tme13_Tmp05"],
        "CLN-513e": [f"{med}_medium", f"{med}_3H_Tme13",
                     f"{med}_5H_Tme13_Tmp05", f"{med}_20H_Tme13_Tmp05_Tme13"],
        "CLN-614p": [f"{med}_medium", f"{med}_3H_Tmp06",
                     f"{med}_5H_Tmp06_Tme14", f"{med}_20H_Tmp06_Tme14_Tmp06"],
        "CLN-614e": [f"{med}_medium", f"{med}_3H_Tme14",
                     f"{med}_5H_Tme14_Tmp06", f"{med}_20H_Tme14_Tmp06_Tme14"],
        "CTR-1p":   [f"{med}_medium", f"{med}_3H_PA01",
                     f"{med}_5H_PA01_K12",    f"{med}_20H_PA01_K12_PA01"],
        "CTR-1e":   [f"{med}_medium", f"{med}_3H_K12",
                     f"{med}_5H_K12_PA01",    f"{med}_20H_K12_PA01_K12"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Abundance bar helper — produces argument dicts for the new plotter
# ─────────────────────────────────────────────────────────────────────────────
def _strain_pairs(med: str, mean_df: pd.DataFrame):
    """Extract (p_strain, e_strain) pairs from 3H solo rows."""
    solo_rows = [r for r in mean_df.index
                 if re.match(rf"^{med}_3H_\w+$", r)]
    solo_strains = [r.split("_3H_")[1] for r in solo_rows]
    return [(solo_strains[i], solo_strains[i + 1])
            for i in range(0, len(solo_strains) - 1, 2)]


def _abundance_inputs(med, mean_df, sd_df, cv_df, scored, out_dir,
                      casic_map=None, class_map=None):
    chains = build_chains(med)
    for p_strain, e_strain in _strain_pairs(med, mean_df):
        chain_p_label = next(
            (n for n, steps in chains.items()
             if n.endswith("p") and len(steps) >= 2
             and steps[1] == f"{med}_3H_{p_strain}"), None
        )
        chain_e_label = next(
            (n for n, steps in chains.items()
             if n.endswith("e") and len(steps) >= 2
             and steps[1] == f"{med}_3H_{e_strain}"), None
        )
        row_keys = [
            f"{med}_medium",
            f"{med}_3H_{p_strain}",
            f"{med}_5H_{p_strain}_{e_strain}",
            f"{med}_20H_{p_strain}_{e_strain}_{p_strain}",
            f"{med}_3H_{e_strain}",
            f"{med}_5H_{e_strain}_{p_strain}",
            f"{med}_20H_{e_strain}_{p_strain}_{e_strain}",
        ]
        missing = [r for r in row_keys if r not in mean_df.index]
        if missing:
            print(f"  [AbundanceBars] Skipping {p_strain}/{e_strain}: missing {missing}")
            continue

        for _, row in scored.drop_duplicates(subset="metabolite").iterrows():
            metab = row["metabolite"]
            if metab not in mean_df.columns:
                continue
            m7 = mean_df.loc[row_keys, metab].values.astype(float)
            s7 = (sd_df.loc[row_keys, metab].values.astype(float)
                  if metab in sd_df.columns else np.zeros(7))
            c7 = (cv_df.loc[row_keys, metab].values.astype(float)
                  if (cv_df is not None and metab in cv_df.columns) else None)
            casic_name  = casic_map.get(metab, metab) if casic_map else metab
            metab_class = class_map.get(metab) if class_map else None
            safe_name   = re.sub(r"[^\w]", "_", str(casic_name))[:40]
            out_fname   = Path(out_dir) / (
                f"AbundanceBars_{med}_{chain_p_label}_vs_{chain_e_label}_"
                f"{safe_name}.png"
            )
            yield dict(
                means=m7, sds=s7, cvs=c7,
                chain_p_name=chain_p_label, chain_e_name=chain_e_label,
                metabolite=metab, casic_name=casic_name,
                metab_class=metab_class, condition=med,
                output_path=str(out_fname),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Plot scorer selection
# ─────────────────────────────────────────────────────────────────────────────
def _enhanced_for_legacy_plots(scored_enh: pd.DataFrame) -> pd.DataFrame:
    """
    Make the enhanced scorer output usable by the older score-driven figure
    functions in visualization.py.

    The enhanced scorer already contains a `score` column, but several legacy
    plots also expect `category`, `cf_probability`, `cf_case`, and
    `classification`.  These compatibility fields are derived from the
    enhanced evidence score without changing the exported enhanced columns.
    """
    if scored_enh is None or scored_enh.empty:
        return scored_enh

    df = scored_enh.copy()

    def _cat(score):
        if score >= 8:
            return "TRUE_CROSSFEED"
        if score >= 5:
            return "ORDER_DEPENDENT"
        return "OPPORTUNISTIC"

    if "category" not in df.columns:
        df["category"] = df["score"].apply(_cat)
    if "cf_probability" not in df.columns:
        df["cf_probability"] = (df["score"].astype(float) / 12.0).clip(0, 1)
    if "cf_case" not in df.columns:
        df["cf_case"] = df.get("pattern_label", "ENHANCED")
    if "classification" not in df.columns:
        df["classification"] = df.get("confidence", "ENHANCED")

    return df


def _select_scored_for_plots(scored: pd.DataFrame,
                             scored_enh: pd.DataFrame,
                             scorer: str) -> pd.DataFrame:
    """Return the scoring table that should drive score-based plots."""
    if str(scorer).lower() == "enhanced" and scored_enh is not None and not scored_enh.empty:
        return _enhanced_for_legacy_plots(scored_enh)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Per-medium worker
# ─────────────────────────────────────────────────────────────────────────────
def _run_medium(xlsx_path, med, out_dir, debug: DebugReport,
                scorer: str = "classic"):
    print(f"\n{'=' * 60}\n  {med}\n{'=' * 60}")

    # 1) Load
    raw_df, casic_map, class_map, bio_feats, is_feats = load_sheet(
        xlsx_path, med, media_prefixes=list(MEDIA)
    )

    # 2) IS normalisation
    norm_df, stable_is = normalize_is(
        raw_df, is_feats, bio_feats, sheet_name=med, stability_cv_max=0.25
    )

    # 3) Mean / CV / SD
    mean_df, cv_df, sd_df = compute_stats(norm_df)

    # 4) Build chains + record schema
    chains = build_chains(med)
    missing_chain_keys = [s for steps in chains.values() for s in steps
                          if s not in mean_df.index]
    debug.record_input_schema(
        med,
        n_features_bio=len(bio_feats),
        n_features_is=len(is_feats),
        n_samples_unique=len(mean_df),
        n_rows_raw=len(raw_df),
        missing_chain_keys=missing_chain_keys,
    )
    debug.record_mapping(
        med, casic_map=casic_map, class_map=class_map, bio_feats=bio_feats
    )
    if missing_chain_keys:
        debug.record_warning(f"[{med}] missing samples: {missing_chain_keys}")

    # 5) QC figure
    raw_is_meandf, raw_bio_meandf = raw_sample_means(raw_df, is_feats, bio_feats)
    try:
        viz.fig_normalisation_qc(
            raw_is_meandf, raw_bio_meandf,
            mean_df, cv_df,
            bio_feats, is_feats, casic_map, class_map,
            med, out_dir,
        )
    except Exception as e:
        debug.record_plot_warning(f"[{med}] normalisation QC: {e}")

    # 6) Candidates (classic step-FC)
    candidates = find_candidates(mean_df, cv_df, chains)
    if not candidates.empty:
        candidates = candidates[candidates["reliable"] == True].copy()

    # 7) Classic scoring
    scored = score_crossfeeding(mean_df, cv_df, chains, med)
    debug.record_scoring(med, scored, scorer="classic")

    # 8) Enhanced multi-evidence scoring
    scored_enh = score_crossfeeding_enhanced(mean_df, cv_df, chains, med)
    if scored_enh is not None and not scored_enh.empty:
        scored_enh = scored_enh.copy()
        scored_enh["metabolite_name"] = scored_enh["metabolite"].map(casic_map).fillna(scored_enh["metabolite"])
        scored_enh["metabolite_class"] = scored_enh["metabolite"].map(class_map)
    debug.record_scoring(f"{med}__enhanced", scored_enh, scorer="enhanced")
    scored_for_plots = _select_scored_for_plots(scored, scored_enh, scorer)

    # 9) Per-medium figure suite (best-effort — keep going on per-figure errors)
    print("  Generating figures...")
    for name, fn, args in [
        ("trajectory_pca",       viz.fig_trajectory_pca,
            (mean_df, bio_feats, chains, med, out_dir)),
        ("fc_heatmap",           viz.fig_fc_heatmap,
            (mean_df, cv_df, bio_feats, casic_map, class_map, chains, med, out_dir)),
        ("crossfeeding_summary", viz.fig_crossfeeding_summary,
            (candidates, casic_map, class_map, med, out_dir)),
        ("cv_reliability",       viz.fig_cv_reliability,
            (cv_df, bio_feats, casic_map, med, out_dir)),
        ("mirror_comparison",    viz.fig_mirror_comparison,
            (mean_df, cv_df, bio_feats, casic_map, class_map, chains, med, out_dir)),
    ]:
        try:
            fn(*args)
        except Exception as e:
            debug.record_plot_warning(f"[{med}] {name}: {e}")

    if scored_for_plots is not None and not scored_for_plots.empty:
        print(f"  Score-driven plots use: {str(scorer).lower()} scorer")
        for name, fn, args in [
            ("evidence_heatmap",        viz.fig_evidence_heatmap,
                (scored_for_plots, casic_map, class_map, med, out_dir)),
            ("trajectory_panels",       viz.fig_trajectory_panels,
                (scored_for_plots, mean_df, cv_df, casic_map, class_map, chains, med, out_dir)),
            ("pgm_trajectory_panels",   viz.fig_pgm_trajectory_panels,
                (scored_for_plots, mean_df, cv_df, casic_map, class_map, chains, med, out_dir)),
            ("cf_probability_heatmap",  viz.fig_cf_probability_heatmap,
                (scored_for_plots, casic_map, class_map, med, out_dir)),
            ("cf_case_matrix",          viz.fig_cf_case_matrix,
                (scored_for_plots, casic_map, class_map, med, out_dir)),
            ("candidate_table",         viz.fig_candidate_table,
                (scored_for_plots, casic_map, class_map, med, out_dir)),
        ]:
            try:
                fn(*args)
            except Exception as e:
                debug.record_plot_warning(f"[{med}] {name}: {e}")

        # Abundance bars (uses the FIXED plotter)
        for kw in _abundance_inputs(med, mean_df, sd_df, cv_df, scored_for_plots,
                                    out_dir, casic_map=casic_map,
                                    class_map=class_map):
            try:
                fig_abundance_bars(**kw)
            except Exception as e:
                debug.record_plot_warning(
                    f"[{med}] abundance_bars {kw.get('metabolite')}: {e}")

    # 10) Excel export
    try:
        export_excel(mean_df, cv_df, candidates, scored,
                     casic_map, class_map, bio_feats, med, out_dir)
    except Exception as e:
        debug.record_exception(f"export_excel({med})", e)

    # Also export enhanced scoring as a CSV per-medium for quick inspection
    if not scored_enh.empty:
        enh_path = Path(out_dir) / f"ScoredCandidates_enhanced_{med}.csv"
        scored_enh.to_csv(enh_path, index=False)

    return {
        "candidates":  candidates,
        "scored":      scored,
        "scored_enh":  scored_enh,
        "casic_map":   casic_map,
        "class_map":   class_map,
        "mean_df":     mean_df,
        "cv_df":       cv_df,
        "sd_df":       sd_df,
        "bio_feats":   bio_feats,
        "chains":      chains,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(config: dict) -> Path:
    """
    Run the full cross-feeding pipeline.

    `config` is a dict with keys
        xlsx_path  (str|Path)   — Excel workbook path
        out_dir    (str|Path)   — base output folder (timestamped
                                  sub-folder is created inside)
        media      (list[str])  — sheet names to process
        scorer     (str)        — "classic" (default) or "enhanced"
    """
    debug = DebugReport()
    debug.record_config(config)

    xlsx_path = Path(config["xlsx_path"])
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx_path not found: {xlsx_path.resolve()}")

    media = config.get("media", list(MEDIA))
    scorer = config.get("scorer", "classic")

    base_out = Path(config["out_dir"])
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = base_out.parent / f"{base_out.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir.resolve()}")

    if hasattr(viz, "fig_mirror_explanation"):
        try:
            viz.fig_mirror_explanation(out_dir)
        except Exception as e:
            debug.record_plot_warning(f"fig_mirror_explanation: {e}")

    all_candidates, all_scored, all_enh = [], [], []
    global_casic, global_class = {}, {}

    for med in media:
        try:
            r = _run_medium(xlsx_path, med, out_dir, debug, scorer=scorer)
        except Exception as e:
            print(f"  ERROR in medium {med}: {e}")
            debug.record_exception(f"_run_medium({med})", e)
            logger.exception("Error processing medium %s", med)
            continue

        if r["candidates"] is not None and not r["candidates"].empty:
            all_candidates.append(r["candidates"])
        if r["scored"] is not None and not r["scored"].empty:
            all_scored.append(r["scored"])
        if r["scored_enh"] is not None and not r["scored_enh"].empty:
            all_enh.append(r["scored_enh"])
        global_casic.update(r["casic_map"])
        global_class.update(r["class_map"])

    # ── Cross-media aggregation ─────────────────────────────────────
    if all_candidates:
        combined = pd.concat(all_candidates, ignore_index=True)
        combined.to_csv(out_dir / "All_CrossFeeding_Candidates.csv", index=False)
        print("\n  All_CrossFeeding_Candidates.csv saved.")

    if all_scored:
        combined_scored = pd.concat(all_scored, ignore_index=True)
        combined_scored.to_csv(out_dir / "All_ScoredCandidates.csv", index=False)
        print("  All_ScoredCandidates.csv saved.")
        if str(scorer).lower() != "enhanced":
            try:
                viz.fig_cross_media_summary(
                    combined_scored, global_casic, global_class, out_dir, min_media=1
                )
                viz.fig_pgm_cross_media_summary(
                    combined_scored, global_casic, global_class, out_dir, min_media=1
                )
            except Exception as e:
                debug.record_plot_warning(f"cross_media summary: {e}")

    if all_enh:
        combined_enh = pd.concat(all_enh, ignore_index=True)
        if "metabolite_name" not in combined_enh.columns:
            combined_enh["metabolite_name"] = combined_enh["metabolite"].map(global_casic).fillna(combined_enh["metabolite"])
        if "metabolite_class" not in combined_enh.columns:
            combined_enh["metabolite_class"] = combined_enh["metabolite"].map(global_class)
        combined_enh = aggregate_media_consistency(combined_enh)
        combined_enh.to_csv(out_dir / "All_ScoredCandidates_enhanced.csv",
                            index=False)
        print("  All_ScoredCandidates_enhanced.csv saved.")
        debug.record_scoring("ALL_MEDIA__enhanced", combined_enh,
                             scorer="enhanced")
        if str(scorer).lower() == "enhanced":
            try:
                combined_enh_for_plots = _enhanced_for_legacy_plots(combined_enh)
                print("  Cross-media plots use: enhanced scorer")
                viz.fig_cross_media_summary(
                    combined_enh_for_plots, global_casic, global_class, out_dir, min_media=1
                )
                viz.fig_pgm_cross_media_summary(
                    combined_enh_for_plots, global_casic, global_class, out_dir, min_media=1
                )
            except Exception as e:
                debug.record_plot_warning(f"enhanced cross_media summary: {e}")
    else:
        debug.record_warning("No enhanced-scoring rows produced.")

    # ── Debug report ────────────────────────────────────────────────
    json_path = debug.write_report(out_dir)
    print(f"  Debug report: {json_path.name}")

    print(f"\n{'=' * 60}\n  Done. All outputs in: {out_dir.resolve()}\n"
          f"{'=' * 60}")
    return out_dir
