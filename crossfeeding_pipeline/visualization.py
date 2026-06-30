"""
visualization.py
================
All figure-producing functions for the cross-feeding pipeline.

Ported verbatim from crossfeeding_pipeline_v2.0.py (lines 752-3058).
Only the header was rewritten to:

  * add the imports that the original module-level `import *` relied on
    (numpy, pandas, matplotlib.pyplot, Path, Counter, sklearn)
  * fix the import of PCA  — it lives in sklearn.decomposition,
    NOT sklearn.preprocessing (typo in the user's vizualization.py)
  * pull shared constants from the new `constants` module instead of
    expecting them to be injected by a monolithic script
  * define the private helpers (_cv_hatch, _classify_column,
    _apply_white_style, _make_step_label) that the figure functions use
"""

import os
import collections
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

# PCA lives in sklearn.decomposition (was incorrectly imported from
# sklearn.preprocessing in the user's vizualization.py)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Pull shared constants & helpers
from .constants import (
    MEDIA,
    CLASS_COLORS, CHAIN_COLORS, CHAIN_LABELS,
    CF_CASE_STYLES, CF_CASE_ORDER,
    MIRROR_PAIRS, MIRROR_PAIR_NAMES,
    TRAJ_CLINICAL, _TRAJ_CLINICAL,
    STEP_MARKERS, STEP_LABELS,
    CAT_COLORS, CAT_MARKER_COLORS, SCORE_CELL_COLORS,
    CATEGORY_DISPLAY, CATEGORY_EXPLAIN,
    FC_THRESH, FC_THRESH_RETURN,
    CV_THRESH, CV_THRESH_LIGHT, CV_THRESH_MEDIUM,
    CV_SENTINEL, NODE_SIZES,
)

from .utils import log2fc, log2fc_scalar


# =============================================================================
# PRIVATE HELPERS  — ported from crossfeeding_pipeline_v2.0.py (lines 513-766)
# =============================================================================
# CHAIN_LABELS already imported above from .constants


def infer_chain_id_from_strain(strain_id: str) -> str:
    """
    Map a strain ID (e.g. 'Tmp05', 'Tme14') to its chain ID
    (e.g. 'CLN-513p', 'CLN-614e') using CHAIN_LABELS.

    If not found, returns the original strain_id.
    """
    for chain_id, label in CHAIN_LABELS.items():
        if strain_id in label:
            return chain_id
    return strain_id

def _cv_hatch(cv_val):
    """
    2-level CV hatch selector for mirror-plot cells.
    Returns (grey_alpha, hatch_pattern, level):
      level 0 = clean, 1 = borderline, 2 = unreliable
    """
    if cv_val <= CV_THRESH_LIGHT:
        return 0.0, "",       0
    elif cv_val <= CV_THRESH_MEDIUM:
        return 0.35, "//",    1
    else:
        return 0.70, "////",  2


def _classify_column(d):
    """
    Classify one metabolite column (6-element d vector) into a biological
    case label.

    Row mapping
    -----------
    d[0] = M-R1  → P solo 3h
    d[1] = M-R2  → E in P-spent 5h
    d[2] = M-R3  → P returns 20h
    d[3] = Mi-R1 → E solo 3h
    d[4] = Mi-R2 → P in E-spent 5h
    d[5] = Mi-R3 → E returns 20h

    Values: +1 = produced | 0 = grey / unreliable | -1 = consumed

    .. warning::
        This mapping MUST stay in sync with the step order produced by
        ``build_chains()`` in ``pipeline.py``:
        [blank, 3h-solo, 5h-partner, 20h-return].
        If ``build_chains`` changes, update the row mapping above and
        the index logic below.
    """
    d = [int(x) for x in d]

    cf_pe_main   = (d[0] == +1 and d[1] == -1)
    cf_ep_mirror = (d[3] == +1 and d[4] == -1)
    cf_ep_main   = (d[0] == -1 and d[1] == +1)

    if cf_pe_main and cf_ep_mirror:
        return "CF-P\u2194E"
    if cf_pe_main:
        return "CF-P\u2192E"
    if cf_ep_mirror or cf_ep_main:
        return "CF-E\u2192P"

    if d[1] != 0 and d[0] == 0:
        return "IND-E-byP" if d[1] == +1 else "DEP-byE"
    if d[4] != 0 and d[3] == 0:
        return "IND-P-byE" if d[4] == +1 else "DEP-byP"

    if d[0] != 0 and d[2] != 0 and d[0] != d[2]:
        return "SHIFT-P-20h"
    if d[0] == 0 and d[2] != 0:
        return "SHIFT-P-20h"

    if d[3] != 0 and d[5] != 0 and d[3] != d[5]:
        return "SHIFT-E-20h"
    if d[3] == 0 and d[5] != 0:
        return "SHIFT-E-20h"

    return ""


def _apply_white_style(fig, ax_or_axes):
    """White figure background + uniform light-grey axes with muted ticks."""
    fig.patch.set_facecolor("#FFFFFF")
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        if ax is not None:
            ax.set_facecolor("#F8F9FA")
            for sp in ax.spines.values():
                sp.set_edgecolor("#3A4A5A")
            ax.tick_params(colors="#8899AA")

# ── CF-case colour palette ────────────────────────────────────────────────────
CF_CASE_STYLES = {
    # ── Cross-feeding ─────────────────────────────────────────────────────────
    "CF-P\u2194E":       ("#1565C0", "#BBDEFB", "True cross-feeding: P\u2194E bidirectional"),
    "CF-E\u2194P":       ("#1565C0", "#BBDEFB", "True cross-feeding: E\u2194P bidirectional"),
    "CF-P\u2192E":       ("#0277BD", "#B3E5FC", "Cross-feeding: P donates to E"),
    "CF-E\u2192P":       ("#0277BD", "#B3E5FC", "Cross-feeding: E donates to P"),
    # ── Induced / dependent (new labels from _classify_column) ────────────────
    "IND-E-byP":       ("#00838F", "#E0F7FA", "E induced by P-spent medium"),
    "IND-P-byE":       ("#00838F", "#E0F7FA", "P induced by E-spent medium"),
    "DEP-byE":         ("#558B2F", "#F1F8E9", "P depleted in E-spent medium"),
    "DEP-byP":         ("#558B2F", "#F1F8E9", "E depleted in P-spent medium"),
    # ── Shift at 20 h ────────────────────────────────────────────────────────
    "SHIFT-P-20h":     ("#F57F17", "#FFF9C4", "P-chain: metabolic shift at 20 h"),
    "SHIFT-E-20h":     ("#F57F17", "#FFF9C4", "E-chain: metabolic shift at 20 h"),
    # ── Legacy uppercase labels ───────────────────────────────────────────────
    "TRUE_CROSSFEED":  ("#1565C0", "#BBDEFB", "True cross-feeding: bidirectional exchange"),
    "OPPORTUNISTIC":   ("#E65100", "#FFE0B2", "Opportunistic: one-sided uptake"),
    "ORDER_DEPENDENT": ("#6A1B9A", "#E1BEE7", "Order-dependent: sequential utilisation"),
    "OPP-P":           ("#E65100", "#FFE0B2", "Opportunistic: P-chain uptake"),
    "OPP-E":           ("#E65100", "#FFE0B2", "Opportunistic: E-chain uptake"),
    "OPP":             ("#E65100", "#FFE0B2", "Opportunistic: one-sided uptake"),
    "ORD":             ("#6A1B9A", "#E1BEE7", "Order-dependent: sequential utilisation"),
    "ORD-P":           ("#6A1B9A", "#E1BEE7", "Order-dependent: P-chain"),
    "ORD-E":           ("#6A1B9A", "#E1BEE7", "Order-dependent: E-chain"),
    "UNCLASSIFIED":    ("#607D8B", "#ECEFF1", "Unclassified / low evidence"),
}

def _cf_case_colors(cf_case):
    """Return (border_hex, fill_hex, tooltip_str) for a CF case label."""
    return CF_CASE_STYLES.get(
        cf_case,
        ("#607D8B", "#ECEFF1", "Unclassified / low evidence")
    )

# =============================================================================
# FIGURE FUNCTIONS  — body copied from vizualization.py (line 12 onward)
# =============================================================================

def fig_trajectory_pca(mean_df, bio_feats, chains, medium, out_dir):
    """
    Trajectory PCA v7: solid=Pseudomonas first (p), dashed=E. coli first (e).
    Arrow drawn at every step transition. Legend placed below x-axis labels.
    """
    feats = [f for f in bio_feats if f in mean_df.columns]
    data  = mean_df[feats].copy()
    imp   = SimpleImputer(strategy="median")
    X     = np.log1p(np.maximum(imp.fit_transform(data.values), 0))
    Xs    = StandardScaler().fit_transform(X)
    n_comp = min(5, Xs.shape[0] - 1, Xs.shape[1])
    pca    = PCA(n_components=n_comp)
    scores = pca.fit_transform(Xs)
    ev     = pca.explained_variance_ratio_ * 100
    scores_df = pd.DataFrame(scores, index=data.index,
                              columns=[f"PC{i+1}" for i in range(n_comp)])

    def _draw_panel(ax, pcx, pcy, xi, yi):
        ax.set_facecolor("#F8F9FA")
        ax.spines[:].set_edgecolor("#3A4A5A")
        ax.tick_params(colors="#8899AA", labelsize=9)
        ax.axhline(0, color="#3A4A5A", lw=0.7)
        ax.axvline(0, color="#3A4A5A", lw=0.7)
        ax.set_xlabel(f"{pcx} ({ev[xi]:.1f}%)", color="#333", fontsize=11)
        ax.set_ylabel(
            f"{pcy} ({ev[yi]:.1f}%)" if len(ev) > yi else pcy,
            color="#333", fontsize=11)
        ax.set_title(f"{medium} | PCA | {pcx} vs {pcy}",
                     color="black", fontsize=12, fontweight="bold", pad=9)

        _TRAJ_NODE_SIZES = [120, 220, 320, 440]  # Blank → 3h → 5h → 20h

        for chain_name, steps in chains.items():
            color = CHAIN_COLORS.get(chain_name, "#AAAAAA")
            # e-suffix = E. coli first  → dashed
            # p-suffix = Pseudomonas first → solid
            ls = "--" if chain_name.endswith("e") else "-"

            xs, ys = [], []
            for si, samp in enumerate(steps):
                if samp not in scores_df.index:
                    continue
                if yi >= len(ev):
                    continue
                x = float(scores_df.loc[samp, pcx])
                y = float(scores_df.loc[samp, pcy])
                xs.append(x)
                ys.append(y)

                # Draw marker for each timepoint
                node_color = "#E53935" if si == 0 else color
                ax.scatter(x, y, c=node_color, marker=STEP_MARKERS[si % 4], s=NODE_SIZES[si % 4],
                           zorder=5, edgecolors='black', linewidths=2.2)

            if len(xs) < 2:
                continue

            # ── Lines only — NO annotate/arrowheads ──────────────────────
            # solid for Pseudomonas-first (p), dashed for E. coli-first (e)
            # Draw as one continuous path so dashes are uniform end-to-end
            ax.plot(xs, ys,
                    color=color,
                    lw=2.2,
                    ls=ls,
                    alpha=0.90,
                    zorder=2,
                    solid_capstyle="round",
                    dash_capstyle="round")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#FFFFFF")
    _draw_panel(axes[0], "PC1", "PC2", xi=0, yi=1)
    _draw_panel(axes[1], "PC1", "PC3", xi=0, yi=2)

    # Legend
    handles = (
            [Line2D([0], [0], color=c, lw=2.2,
                    ls="--" if n.endswith("e") else "-",
                    label=CHAIN_LABELS.get(n, n))
             for n, c in CHAIN_COLORS.items()] +
            [Line2D([0], [0], color="w",
                    marker=m, linestyle="",
                    markerfacecolor="grey",
                    markersize=8,
                    label=l,
                    markeredgecolor="black")
             for m, l in zip(STEP_MARKERS, STEP_LABELS)]
    )
    fig.legend(handles=handles,
               loc="lower center",
               ncol=3,
               fontsize=8.5,
               framealpha=0.90,
               labelcolor="black",
               facecolor="white",
               edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, -0.10))

    plt.suptitle(
        f"{medium} \u2014 Sequential Cross-Feeding PCA\n"
        f"Solid lines = Pseudomonas first (p)  "
        f"\u2502  Dashed lines = E. coli first (e)",
        color="black", fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    out = Path(out_dir) / f"PCA_{medium}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")

def fig_fc_heatmap(mean_df, cv_df, bio_feats, casic_map, class_map,
                    chains, medium, out_dir):
    """
    log2 FC vs blank — samples ordered by chain, metabolites by class.
    v6: chain label annotations on left side (fix #6); white background (fix #10).
    """
    blank_key = f"{medium}_medium"
    if blank_key not in mean_df.index:
        print(f"  Warning: blank not found for {medium}")
        return
    blank = mean_df.loc[blank_key]
    feats = [f for f in bio_feats if f in mean_df.columns]

    ordered_samples, seen = [], set()
    for steps in chains.values():
        for s in steps[1:]:
            if s in mean_df.index and s not in seen:
                ordered_samples.append(s); seen.add(s)
    if not ordered_samples:
        return

    fc_data = {s: log2fc(mean_df.loc[s, feats], blank[feats]) for s in ordered_samples}
    fc_df   = pd.DataFrame(fc_data).T

    reliable_mask = cv_df.loc[ordered_samples, feats] <= CV_THRESH
    reliable_fc   = fc_df[reliable_mask].fillna(0)
    valid_mets    = reliable_fc.columns[(reliable_fc.abs() >= FC_THRESH).sum(axis=0) >= 2]
    feats         = [m for m in feats if m in valid_mets]
    if not feats:
        print(f"  Note: No metabolites passed heatmap filter for {medium}.")
        return

    met_cls     = sorted([(m, class_map.get(m, "Unknown")) for m in feats], key=lambda x: x[1])
    sorted_mets = [m for m, _ in met_cls]
    plot_df     = np.clip(fc_df[sorted_mets].values.astype(float), -8, 8)
    col_labels  = [casic_map.get(m, m)[:22] for m in sorted_mets]
    row_labels  = [s.replace(f"{medium}_", "") for s in ordered_samples]

    fw = max(16, len(sorted_mets) * 0.30)
    fh = max(8,  len(ordered_samples) * 0.40)
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")   # FIX #10
    ax.set_facecolor("#0F1923")
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
    im   = ax.imshow(plot_df, aspect="equal", cmap="seismic",
                     norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(sorted_mets)))
    ax.set_xticklabels(col_labels, rotation=50, ha="right", fontsize=7, color="black")
    ax.set_yticks(range(len(ordered_samples)))
    ax.set_yticklabels(row_labels, fontsize=8, color="black")
    ax.tick_params(length=0)
    for xt, met in zip(ax.get_xticklabels(), sorted_mets):
        xt.set_color(CLASS_COLORS.get(class_map.get(met, "Unknown"), "#666666"))
        xt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="grey")])
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.02, pad=0.22, aspect=40)
    cbar.set_label("log2 FC vs. blank medium (blue = consumed, red = produced)",
                   color="black", fontsize=9)
    cbar.ax.tick_params(colors="black", labelsize=8)
    seen_cls, patches = set(), []
    for m in sorted_mets:
        cls = class_map.get(m, "Unknown")
        if cls not in seen_cls:
            patches.append(mpatches.Patch(color=CLASS_COLORS.get(cls, "#ccc"), label=cls))
            seen_cls.add(cls)
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               framealpha=0.15, labelcolor="black", facecolor="white",
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"{medium} | log2 FC vs. Blank Medium (samples ordered by chain)",
                 color="black", fontsize=12, fontweight="bold", pad=10)

    # FIX #6: Chain label annotations on left side
    # Build a list of (chain_name, start_row, end_row) for each chain group
    chain_row_ranges = []
    row_idx = 0
    for chain_name, steps in chains.items():
        chain_steps_present = [s for s in steps[1:] if s in ordered_samples]
        if not chain_steps_present:
            continue
        start_row = ordered_samples.index(chain_steps_present[0])
        end_row   = ordered_samples.index(chain_steps_present[-1])
        chain_row_ranges.append((chain_name, start_row, end_row))

    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for chain_name, start_row, end_row in chain_row_ranges:
        mid_row = (start_row + end_row) / 2.0  # ← defined as mid_row
        color = CHAIN_COLORS.get(chain_name, "#555")
        ax.text(1.01, mid_row, chain_name,  # ← used as mid_row  ✓
                transform=trans, fontsize=7, color=color,
                ha="left", va="center", fontweight="bold", clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Horizontal dashed separators between chains
    y_pos = 0
    for cs in [len(s) - 1 for s in chains.values()][:-1]:
        y_pos += cs
        ax.axhline(y_pos - 0.5, color="#333333", lw=1.0, ls="--", alpha=0.5)
    out = Path(out_dir) / f"Heatmap_{medium}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


def _make_step_label(chain_name, step_idx, medium):
    """Human-readable step label for ranked FC bar subtitles."""
    templates = {
        "CLN-513p": ["\u2014", "Tmp05 solo (3h)",  "Tme13 in Tmp05 medium (5h)", "Tmp05 returns (20h)"],
        "CLN-513e": ["\u2014", "Tme13 solo (3h)",  "Tmp05 in Tme13 medium (5h)", "Tme13 returns (20h)"],
        "CLN-614p": ["\u2014", "Tmp06 solo (3h)",  "Tme14 in Tmp06 medium (5h)", "Tmp06 returns (20h)"],
        "CLN-614e": ["\u2014", "Tme14 solo (3h)",  "Tmp06 in Tme14 medium (5h)", "Tme14 returns (20h)"],
        "CTR-1p":   ["\u2014", "PA01 solo (3h)",   "K12 in PA01 medium (5h)",    "PA01 returns (20h)"],
        "CTR-1e":   ["\u2014", "K12 solo (3h)",    "PA01 in K12 medium (5h)",    "K12 returns (20h)"],
    }
    labels = templates.get(chain_name, ["\u2014"] * 4)
    return labels[step_idx] if 0 <= step_idx < len(labels) else f"Step {step_idx}"


def fig_ranked_fc_bars(mean_df, cv_df, bio_feats, casic_map, class_map,
                        chains, medium, out_dir):
    """
    Ranked FC bar charts per chain, one panel per step.
    v6 fixes:
      #2: FC values capped at [-8, 8] with note "capped at ±8"
      #3: Empty panel shows clean message without dead scatter [0,1] axis
      #5: Panel subtitles show clean step labels
      #10: White background on all axes
    """
    feats      = [f for f in bio_feats if f in mean_df.columns]
    step_names = STEP_LABELS[1:]   # ["3h solo", "5h cross-feed", "20h return"]
    for chain_name, steps in chains.items():
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#FFFFFF")   # FIX #10
        fig.suptitle(f"{medium} | {CHAIN_LABELS.get(chain_name, chain_name)} | Ranked FC per step",
                     color="black", fontsize=13, fontweight="bold", y=1.01)
        for ax_i in range(3):
            ax        = axes[ax_i]
            ax.set_facecolor("#F8F9FA")   # FIX #10
            ax.spines[:].set_edgecolor("#3A4A5A")
            ax.tick_params(colors="#8899AA", labelsize=8)
            curr_name = steps[ax_i + 1] if ax_i + 1 < len(steps) else None
            prev_name = steps[ax_i]
            # FIX #5: clean label
            clean_label = _make_step_label(chain_name, ax_i + 1, medium)

            if curr_name is None or curr_name not in mean_df.index or prev_name not in mean_df.index:
                # FIX #3: Clean empty panel — no dead scatter
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
                ax.text(0.5, 0.5, "Not available", transform=ax.transAxes,
                        ha="center", va="center", color="#333333", fontsize=11)
                ax.set_title(f"Step {ax_i+1}: {step_names[ax_i]}\n{clean_label}",
                             color="black", fontsize=9.5, fontweight="bold", pad=6)
                continue
            fc      = log2fc(mean_df.loc[curr_name, feats], mean_df.loc[prev_name, feats])
            cv_curr = (cv_df.loc[curr_name, feats] if curr_name in cv_df.index
                       else pd.Series(0, index=feats))
            valid_mask = (fc.abs() >= FC_THRESH) & (cv_curr <= CV_THRESH)
            fc_filtered = fc[valid_mask].dropna()
            if fc_filtered.empty:
                # FIX #3: Clean empty panel
                ax.set_xlim(0, 1)
                ax.set_ylim(-FC_THRESH * 2, FC_THRESH * 2)
                ax.axhline(0, color="#333333", lw=0.9)
                ax.axhline( FC_THRESH, color="#E07070", lw=0.7, ls=":", alpha=0.6)
                ax.axhline(-FC_THRESH, color="#6BA3D6", lw=0.7, ls=":", alpha=0.6)
                ax.text(0.5, 0.5, "No metabolites\npassed thresholds",
                        transform=ax.transAxes, ha="center", va="center",
                        color="black", fontsize=10)
                ax.set_title(f"Step {ax_i+1}: {step_names[ax_i]}\n{clean_label}",
                             color="black", fontsize=9.5, fontweight="bold", pad=6)
                ax.set_ylabel("log2 FC vs. previous step", color="#333333", fontsize=9)
                continue

            # FIX #2: Cap FC values at [-8, 8] and note if any were capped
            any_capped = (fc_filtered.abs() > 8).any()
            fc_sorted  = fc_filtered.sort_values()
            fc_capped  = fc_sorted.clip(-8, 8)

            colors_bar = [CLASS_COLORS.get(class_map.get(m, "Unknown"), "#ccc") for m in fc_capped.index]
            hatches    = ["///" if cv_curr.get(m, 0) > CV_THRESH else "" for m in fc_capped.index]
            labels     = [casic_map.get(m, m)[:18] for m in fc_capped.index]
            x_pos      = range(len(fc_capped))
            bars = ax.bar(x_pos, fc_capped.values, color=colors_bar, alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            ax.axhline(0,          color="#333333", lw=0.9)
            ax.axhline( FC_THRESH, color="#E07070", lw=0.7, ls=":", alpha=0.6)
            ax.axhline(-FC_THRESH, color="#6BA3D6", lw=0.7, ls=":", alpha=0.6)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5, color="black")
            ax.set_ylabel("log2 FC vs. previous step", color="#333333", fontsize=9)
            # FIX #5: clean subtitle
            ax.set_title(
                f"Step {ax_i+1}: {step_names[ax_i]}\n{clean_label}",
                color="black", fontsize=9.5, fontweight="bold", pad=6)
            ax.axhspan(-8, 0, alpha=0.04, color="#6BA3D6")
            ax.axhspan( 0, 8, alpha=0.04, color="#E07070")
            ax.set_xlim(-0.7, len(fc_capped) - 0.3)
            ax.set_ylim(-8.5, 8.5)   # FIX #2: shared y-axis scale
            n_unrel = sum(1 for h in hatches if h)
            if n_unrel:
                ax.text(0.01, 0.98, f"! {n_unrel} unreliable (CV>{int(CV_THRESH*100)}%)",
                        transform=ax.transAxes, fontsize=7.5, color="#FBBF24",
                        va="top", style="italic")
            # FIX #2: note if capped
            if any_capped:
                ax.text(0.99, 0.98, "capped at ±8",
                        transform=ax.transAxes, fontsize=7, color="#EF5350",
                        va="top", ha="right", style="italic")
        seen_cls, patches_leg = set(), []
        for m in feats:
            cls = class_map.get(m, "Unknown")
            if cls not in seen_cls:
                patches_leg.append(mpatches.Patch(color=CLASS_COLORS.get(cls, "#ccc"), label=cls))
                seen_cls.add(cls)
        patches_leg.append(mpatches.Patch(facecolor="#888", hatch="///",
                                           label=f"CV>{int(CV_THRESH*100)}% (unreliable)"))
        fig.legend(handles=patches_leg, loc="lower center", ncol=5, fontsize=8,
                   framealpha=0.18, labelcolor="black", facecolor="white",
                   edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        safe = chain_name.replace(" ", "_").replace("->", "_").replace("/", "")
        out  = Path(out_dir) / f"RankedFC_{medium}_{safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out.name}")


def fig_crossfeeding_summary(candidates_df, casic_map, class_map, medium, out_dir):
    """Bubble chart: top 25 metabolites x all chain steps x direction."""
    if candidates_df is None or candidates_df.empty:
        return
    df = candidates_df[candidates_df["reliable"] == True].copy()
    if df.empty:
        df = candidates_df.copy()
        print(f"  Note: all candidates unreliable for {medium}, showing all.")

    chain_counts = df.groupby("metabolite")["chain"].nunique()
    valid_mets   = chain_counts[chain_counts >= 2].index.tolist()
    df           = df[df["metabolite"].isin(valid_mets)]
    if df.empty:
        return

    met_mag  = df.groupby("metabolite")["log2FC_step"].apply(lambda x: np.median(np.abs(x)))
    top_mets = met_mag.sort_values(ascending=False).head(25).index.tolist()
    df       = df[df["metabolite"].isin(top_mets)]

    all_chains = list(CHAIN_COLORS.keys())
    all_steps = ["3h solo", "5h cross-feed", "20h return"]

    STEP_GAP = 0.8
    CHAIN_GAP = 2.0

    x_cats, x_positions = [], []
    x = 0.0
    for c in all_chains:
        for si, s in enumerate(all_steps):
            x_cats.append(f"{s}\n{c}")
            x_positions.append(x)
            x += STEP_GAP if si < len(all_steps) - 1 else CHAIN_GAP

    x_map = {label: pos for label, pos in zip(x_cats, x_positions)}
    df["x_label"] = df.apply(lambda r: f"{r['step_label']}\n{r['chain']}", axis=1)

    y_cats   = top_mets[::-1]
    y_map    = {v: i for i, v in enumerate(y_cats)}
    y_labels = [casic_map.get(m, m)[:25] for m in y_cats]

    fw = max(12, x_positions[-1] * 0.65 + 4.0)
    fh = max(8, len(y_cats) * 0.6)
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")   # FIX #10
    ax.set_facecolor("#F8F9FA")

    for _, row in df.iterrows():
        xi    = x_map.get(row["x_label"])
        yi    = y_map.get(row["metabolite"])
        if xi is None or yi is None:
            continue
        size  = min(abs(row["log2FC_step"]) * 120, 800)
        color = "#EF5350" if row["direction"].startswith("prod") else "#5B9BD5"
        alpha = 0.85 if row["reliable"] else 0.4
        ax.scatter(xi, yi, s=size, c=color, alpha=alpha, zorder=4,
                   edgecolors="black", linewidths=0.5)

    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.set_ylim(-0.5, len(y_cats) - 0.5)
    ax.grid(True, axis="y", color="#CCCCCC", lw=0.8, zorder=0)
    last_chain = None
    prev_xp = None
    for xl, xp in zip(x_cats, x_positions):
        cur = xl.split("\n")[1]
        if last_chain is not None and cur != last_chain:
            ax.axvline((prev_xp + xp) / 2, color="#A0A0A0",
                       lw=1.5, ls="--", zorder=1)
        last_chain = cur
        prev_xp = xp

    handles = [
        mpatches.Patch(color="#EF5350", label="Produced / accumulated"),
        mpatches.Patch(color="#5B9BD5", label="Consumed / depleted"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=11,   label="|log2FC| = 1", markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=15.5, label="|log2FC| = 2", markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=22,   label="|log2FC| = 4", markeredgecolor="black"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9, labelcolor="black", facecolor="white",
              edgecolor="#CCCCCC", borderaxespad=0.)
    ax.set_xticks(x_positions)
    # Step labels at 45° — extract just the step name (before \n)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([xl.split("\n")[0] for xl in x_cats],
                       rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="x", which="both", length=4, color="#555555")

    # Chain name once, centred below its group
    for c in all_chains:
        chain_xpos = [xp for xl, xp in zip(x_cats, x_positions)
                      if xl.split("\n")[1] == c]
        if chain_xpos:
            mid_x = np.mean(chain_xpos)
            ax.annotate(
                c,
                xy=(mid_x, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -42), textcoords="offset points",
                ha="center", va="top",
                fontsize=9, fontweight="bold",
                color=CHAIN_COLORS.get(c, "#333333"),
                annotation_clip=False,
            )

    plt.subplots_adjust(bottom=0.22)

    # Extra bottom margin so chain labels are not clipped
    plt.subplots_adjust(bottom=0.22)
    ax.set_yticks(np.arange(len(y_cats)))
    ax.set_yticklabels(y_labels, fontsize=8)
    for yt, met in zip(ax.get_yticklabels(), y_cats):
        yt.set_color(CLASS_COLORS.get(class_map.get(met, "Unknown"), "#666666"))
        yt.set_fontweight("bold")
        yt.set_path_effects([pe.withStroke(linewidth=0.8, foreground="#DDDDDD")])
    ax.set_title(f"{medium} | Cross-Feeding Candidates (|log2FC| >= 1.5, reliable CV)",
                 color="black", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Chain . Step",  color="black", fontsize=11, labelpad=8)
    ax.set_ylabel("Metabolite",    color="black", fontsize=11, labelpad=8)
    plt.tight_layout()
    out = Path(out_dir) / f"CrossFeeding_Bubble_{medium}.jpg"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_cv_reliability(cv_df, bio_feats, casic_map, medium, out_dir):
    """
    CV heatmap across all samples and metabolites.
    v6 fixes:
      #9: clip colorbar to 0.6 (not 1.0), add "CV threshold = 50%" annotation
      #10: white background
    """
    valid_rows = [s for s in cv_df.index if s.startswith(f"{medium}_")]
    if valid_rows:
        cv_df = cv_df.loc[valid_rows]
    feats = [f for f in bio_feats if f in cv_df.columns]
    col_labels = [casic_map.get(m, m)[:20] for m in feats]
    # FIX #9: clip to 0.6 for better differentiation in low range
    plot_data  = np.clip(cv_df[feats].values.astype(float), 0, 0.6)
    fw = max(14, len(feats) * 0.25)
    fh = max(6,  len(cv_df)  * 0.32)
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")   # FIX #10
    ax.set_facecolor("#0F1923")
    im = ax.imshow(plot_data, aspect="auto", cmap=plt.cm.YlOrRd,
                   vmin=0, vmax=0.6, interpolation="nearest")   # FIX #9: vmax=0.6
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            if plot_data[i, j] >= CV_THRESH:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#EF5350", lw=1.2))
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(col_labels, rotation=50, ha="right", fontsize=6.5, color="black")
    ax.set_yticks(range(len(cv_df)))
    ax.set_yticklabels([s.replace(f"{medium}_", "") for s in cv_df.index],
                        fontsize=7.5, color="black")
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.02, pad=0.22, aspect=40)
    cbar.set_label(f"CV — red outline = CV > {int(CV_THRESH*100)}% (display clipped at 60%)",
                   color="black", fontsize=9)
    cbar.ax.tick_params(colors="black", labelsize=8)
    # FIX #9: Add "CV threshold = 50%" annotation on the colorbar
    cbar.ax.axvline(x=CV_THRESH / 0.6, color="#EF5350", lw=1.5, ls="--")
    cbar.ax.text(CV_THRESH / 0.6, 1.05, "CV threshold = 50%",
                 transform=cbar.ax.transAxes,
                 fontsize=7, color="#EF5350", ha="center", va="bottom",
                 fontweight="bold")
    ax.set_title(f"{medium} | Technical Replicate CV (2 reps per sample)",
                 color="black", fontsize=12, fontweight="bold", pad=10)
    out = Path(out_dir) / f"CV_Reliability_{medium}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


# =============================================================================
# NEW FIGURES — Evidence heatmap, trajectory small-multiples, cross-media dot
# =============================================================================

def fig_evidence_heatmap(scored, casic_map, class_map, medium, out_dir):
    """
    Evidence score heatmap — rows = metabolites, columns = chains.
    v1.6 fix #4: auto-scaling figsize, minimum cell size 0.45.
    v1.6 fix #10: white background.
    """
    if scored is None or scored.empty:
        return

    grp = scored.groupby(["metabolite", "chain"])["score"].max().reset_index()
    if grp.empty:
        return

    top_mets = (grp.groupby("metabolite")["score"].max()
                .sort_values(ascending=False).head(30).index.tolist())
    if not top_mets:
        return

    grp = grp[grp["metabolite"].isin(top_mets)]
    pivot = (grp.pivot(index="metabolite", columns="chain", values="score")
               .reindex(columns=list(CHAIN_COLORS.keys()))
               .fillna(0)
               .loc[top_mets])

    best_cat = (scored.groupby("metabolite")["score"].max()
                .map(lambda s: "TRUE_CROSSFEED" if s >= 4 else
                               "ORDER_DEPENDENT" if s >= 2 else "OPPORTUNISTIC")
                .to_dict())

    chains_cols = list(CHAIN_COLORS.keys())
    n_rows = len(top_mets)
    n_cols = len(chains_cols)
    data   = pivot.values.astype(float)

    # --- Discrete 5-colour colormap (score 0-4) ---------------------------
    score_cmap = ListedColormap([
        "#FFFFFF",  # 0 — white
        "#F8BBD0",  # 1 — light pink
        "#FFF176",  # 2 — yellow
        "#FFB74D",  # 3 — orange
        "#66BB6A",  # 4 — green
    ])
    bounds     = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    score_norm = BoundaryNorm(bounds, score_cmap.N)

    # FIX #4: Auto-scale figsize with minimum cell size 0.45
    CELL    = max(0.45, min(0.8, 12.0 / max(n_rows, n_cols)))
    LEFT_W  = 3.5
    RIGHT_W = 3.0
    TOP_H   = 0.6
    BOT_H   = 1.2

    hm_w = n_cols * CELL
    hm_h = n_rows * CELL
    fw   = LEFT_W + hm_w + RIGHT_W
    fh   = max(4.0, TOP_H + hm_h + BOT_H)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")   # FIX #10
    ax.set_facecolor("#FAFAFA")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")

    # Pin axes to exact pixel-perfect grid
    ax.set_position([LEFT_W / fw,
                     BOT_H  / fh,
                     hm_w   / fw,
                     hm_h   / fh])

    im = ax.imshow(data, aspect="equal", cmap=score_cmap, norm=score_norm,
                   interpolation="nearest")

    # Thin grid lines
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="#CCCCCC", lw=0.5, zorder=2)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="#CCCCCC", lw=0.5, zorder=2)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(chains_cols,
                       rotation=50, ha="right",
                       fontsize=6.5, fontweight="bold")
    for xt, ch in zip(ax.get_xticklabels(), chains_cols):
        xt.set_color(CHAIN_COLORS.get(ch, "#333"))
    ax.tick_params(axis="x", length=0)

    ax.set_yticks([])
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, met in enumerate(top_mets):
        cat       = best_cat.get(met, "OPPORTUNISTIC")
        circle_col = CAT_MARKER_COLORS.get(cat, "#999999")
        class_col  = CLASS_COLORS.get(class_map.get(met, "Unknown"), "#333333")
        label      = casic_map.get(met, met)[:24]

        ax.text(-0.02, i, "●",
                transform=trans, fontsize=6,
                color=circle_col, ha="right", va="center", clip_on=False)
        ax.text(-0.04, i, label,
                transform=trans, fontsize=6,
                color=class_col, ha="right", va="center", clip_on=False,
                path_effects=[pe.withStroke(linewidth=0.3,
                                            foreground="#F0F0F0")])

    # Legends
    from matplotlib.patches import Patch as MPatch
    score_patches = [
        MPatch(color=score_cmap(score_norm(s)),
               edgecolor="#AAAAAA", linewidth=0.5,
               label=f"Score {s}" + (" — True Cross-Feed" if s == 4 else ""))
        for s in [4, 3, 2, 1, 0]
    ]
    cat_patches = []
    for cat_key in ["TRUE_CROSSFEED", "ORDER_DEPENDENT", "OPPORTUNISTIC"]:
        display = CATEGORY_DISPLAY.get(cat_key, cat_key)
        col     = CAT_MARKER_COLORS.get(cat_key, "#999")
        cat_patches.append(MPatch(color=col, label=f"● {display}"))

    leg_x = (LEFT_W + hm_w + 0.15) / fw
    leg_y1 = (BOT_H + hm_h) / fh

    leg1 = fig.add_axes([leg_x, leg_y1 - 0.38, RIGHT_W / fw * 0.95, 0.34])
    leg1.set_axis_off()
    leg1.legend(handles=score_patches,
                title="Evidence Score", title_fontsize=6,
                loc="upper left", fontsize=6,
                facecolor="white", edgecolor="#CCCCCC",
                labelcolor="black", framealpha=0.95,
                handlelength=0.9, handleheight=0.7,
                borderpad=0.35, labelspacing=0.25)

    leg2 = fig.add_axes([leg_x, leg_y1 - 0.72, RIGHT_W / fw * 0.95, 0.30])
    leg2.set_axis_off()
    leg2.legend(handles=cat_patches,
                title="Category (● before name)", title_fontsize=6,
                loc="upper left", fontsize=6,
                facecolor="white", edgecolor="#CCCCCC",
                labelcolor="black", framealpha=0.95,
                handlelength=0.9, handleheight=0.7,
                borderpad=0.35, labelspacing=0.25)

    ax.set_title(
        f"{medium} | Cross-Feeding Evidence Score  ×  Chain",
        color="black", fontsize=8, fontweight="bold", pad=7,
        transform=ax.transAxes,
        x=0.5, y=1 + 0.35 / hm_h
    )

    out = Path(out_dir) / f"CF_EvidenceHeatmap_{medium}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_trajectory_panels(scored, meandf, cvdf, casic_map, classmap,
                          chains, medium, out_dir, ntop=12):
    """
    Small-multiple trajectory panels  (v2.0 -- SD error bands).

    Changes vs v1.7
    -------------
    - SD shaded band around each line via ax.fill_between().
      Recovered from cvdf: sd = cv * mean.
      Single-replicate samples (CV == CV_SENTINEL ~1.414) are excluded.
      Band alpha 0.20 (scored chains) / 0.12 (below-threshold).
    - SD is propagated through the log2(x+1) transform:
        SD_log2 = SD_raw / (ln(2) * (mean + 1))
    - Supra-title and encoding legend updated to mention SD band.
    - All other logic (scoring, norm, layout) unchanged.
    """
    if scored is None or scored.empty or meandf.empty:
        return

    clin_scored = scored[scored["chain"].isin(_TRAJ_CLINICAL)]
    if clin_scored.empty:
        print(f"  Note: no clinical-chain evidence for {medium}, skipping trajectory.")
        return

    top_mets = (clin_scored.groupby("metabolite")["score"].max()
                .sort_values(ascending=False).head(ntop).index.tolist())
    if not top_mets:
        return

    ncols = min(4, len(top_mets))
    nrows = int(np.ceil(len(top_mets) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.4),
                             facecolor="#FFFFFF")
    fig.subplots_adjust(hspace=0.75, wspace=0.38)
    axes = np.array(axes).flatten()
    step_labs = ["Blank", "3h", "5h-", "20h"]

    for idx, met in enumerate(top_mets):
        ax = axes[idx]
        ax.set_facecolor("#F9FAFB")
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")
        ax.axvspan(1.5, 2.5, alpha=0.08, color="#EF5350", zorder=0)

        # shared y-range
        all_log_vals = []
        for ch in _TRAJ_CLINICAL:
            for s in chains.get(ch, []):
                if s in meandf.index and met in meandf.columns:
                    v = meandf.loc[s, met]
                    if not (isinstance(v, float) and np.isnan(v)) and v > 0:
                        all_log_vals.append(np.log2(float(v) + 1))
        if not all_log_vals:
            ax.set_visible(False)
            continue
        ymin = min(all_log_vals)
        ymax = max(all_log_vals)
        yrng = ymax - ymin if ymax > ymin else 1.0

        plotted_any = False
        for chain_name in _TRAJ_CLINICAL:
            steps = chains.get(chain_name, [])
            if not steps:
                continue
            color  = CHAIN_COLORS.get(chain_name, "#888888")
            ls     = "--" if chain_name.endswith("e") else "-"
            lw     = 2.2
            sub    = scored[(scored["metabolite"] == met) & (scored["chain"] == chain_name)]
            passed = not sub.empty
            b_score = int(sub["score"].max())             if passed else 0
            b_cat   = sub.loc[sub["score"].idxmax(), "category"] if passed else ""
            alpha_val = 0.95 if passed else 0.45
            marker    = "o"  if passed else "^"
            m_size    = 6.5  if passed else 7.0
            leg_label = (
                f"{chain_name} [{b_score}] {CATEGORY_DISPLAY.get(b_cat, b_cat)}"
                if passed else f"{chain_name} no evidence"
            )

            ys_norm, sd_norm, xs_valid = [], [], []
            for xi, s in enumerate(steps):
                if s in meandf.index and met in meandf.columns:
                    v = meandf.loc[s, met]
                    if not (isinstance(v, float) and np.isnan(v)) and v > 0:
                        yn = (np.log2(float(v) + 1) - ymin) / yrng
                        ys_norm.append(yn)
                        # recover SD from cvdf
                        cv_val = np.nan
                        if cvdf is not None and s in cvdf.index and met in cvdf.columns:
                            try:
                                cv_val = float(cvdf.loc[s, met])
                            except (TypeError, ValueError):
                                pass
                        is_sentinel = (not np.isnan(cv_val) and abs(cv_val - CV_SENTINEL) < 0.01)
                        is_zero_cv = (not np.isnan(cv_val) and cv_val == 0.0)
                        if (not np.isnan(cv_val) and not is_sentinel
                                and not is_zero_cv and float(v) > 0):
                            sd_raw = cv_val * float(v)
                            sd_log = sd_raw / (np.log(2) * (float(v) + 1))
                            sd_n = sd_log / yrng
                        else:
                            sd_n = np.nan
                            sd_n = np.nan
                        sd_norm.append(sd_n)
                    else:
                        ys_norm.append(np.nan)
                        sd_norm.append(np.nan)
                else:
                    ys_norm.append(np.nan)
                    sd_norm.append(np.nan)
                xs_valid.append(xi)

            if sum(1 for y in ys_norm if not np.isnan(y)) < 2:
                continue

            ax.plot(xs_valid, ys_norm,
                    color=color, ls=ls, lw=lw,
                    marker=marker, markersize=m_size,
                    markeredgecolor="white", markeredgewidth=0.8,
                    alpha=alpha_val, zorder=3, label=leg_label)

            # SD band
            band_alpha = 0.20 if passed else 0.12
            y_arr  = np.array(ys_norm, dtype=float)
            sd_arr = np.array(sd_norm, dtype=float)
            mask   = np.isfinite(y_arr) & np.isfinite(sd_arr)
            if mask.sum() >= 2:
                xs_arr = np.array(xs_valid, dtype=float)
                ax.errorbar(xs_valid, ys_norm, yerr=sd_norm,
                            fmt='none', ecolor=color, elinewidth=1.2,
                            capsize=4, capthick=1.2, alpha=0.7, zorder=4)

            plotted_any = True

        if not plotted_any:
            ax.set_visible(False)
            continue

        bs      = int(clin_scored[clin_scored["metabolite"] == met]["score"].max())
        bcat_idx = clin_scored[clin_scored["metabolite"] == met]["score"].idxmax()
        b_cat   = clin_scored.loc[bcat_idx, "category"]
        title_col = CAT_MARKER_COLORS.get(b_cat, "#333333")
        ax.set_title(
            f"{casic_map.get(met, met)[:22]} [{bs}] {CATEGORY_DISPLAY.get(b_cat, b_cat)}",
            fontsize=7.5, color=title_col, fontweight="bold", pad=4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(step_labs, fontsize=6.5, color="#333")
        ax.set_ylabel("Normalised abundance\n(0-1)", fontsize=6.0, color="#555")
        ax.set_ylim(-0.05, 1.10)
        ax.tick_params(colors="#333", labelsize=6.5, length=2)
        ax.spines["left"].set_edgecolor(
            CLASS_COLORS.get(classmap.get(met, "Unknown"), "#999999"))
        ax.spines["left"].set_linewidth(2.5)
        ax.legend(fontsize=5.2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.32), ncol=1,
                  framealpha=0.90, labelcolor="black",
                  facecolor="white", edgecolor="#CCCCCC",
                  handlelength=2.2, borderpad=0.5, handletextpad=0.4)

    for i in range(len(top_mets), len(axes)):
        axes[i].set_visible(False)

    enc_handles = [
        Line2D([0],[0], color="#D55E00", lw=2.2,
               label="CLN-513 Tmp05 x Tme13"),
        Line2D([0],[0], color="#0072B2", lw=2.2,
               label="CLN-614 Tmp06 x Tme14"),
        Line2D([0],[0], color="#666666", ls="-",  lw=2.0,
               label="Pseudomonas first (p) solid"),
        Line2D([0],[0], color="#666666", ls="--", lw=2.0,
               label="E. coli first (e) dashed"),
        Line2D([0],[0], color="#666666", ls="",
               marker="o", markersize=7,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Evidence >= threshold"),
        Line2D([0],[0], color="#666666", ls="",
               marker="^", markersize=7,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Below threshold (context)"),
        mpatches.Patch(facecolor="#888888", alpha=0.20,
                       label="+/-1 SD (shaded, replicates only)"),
    ]
    # Reserve top and bottom margins first
    fig.tight_layout(rect=[0.04, 0.10, 0.96, 0.92])

    # Put the global legend into the reserved bottom margin
    fig.legend(
        handles=enc_handles,
        loc="lower center",
        ncol=4,
        fontsize=8.0,
        framealpha=0.95,
        labelcolor="black",
        facecolor="white",
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 0.02),
    )

    # Put the title into the reserved top margin
    fig.suptitle(
        f"{medium} -- Clinical Cross-Feeding Trajectories\n"
        f"Top {len(top_mets)} metabolites by evidence score  |  "
        f"Solid = Pseudomonas first  .  Dashed = E. coli first  |  "
        f"Shaded band = +/-1 SD (replicated samples only)",
        fontsize=9.5,
        fontweight="bold",
        color="black",
        y=0.98,
    )

    out = Path(out_dir) / f"CFTrajectory_{medium}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# CROSS-MEDIA SUMMARY  (fix #8: always include all 5 media columns)
# =============================================================================

def fig_cross_media_summary(all_scored, casic_map, class_map, out_dir,
                             min_media=1):
    """
    Global dot-matrix across all media.
    v6 1.fix #8: Always include all 5 media columns regardless of whether they
    have candidates (SUC column was missing when no scored candidates).
    """
    if all_scored is None or all_scored.empty:
        return
    if "medium" not in all_scored.columns:
        return

    # FIX #8: Always use all 5 MEDIA, not just those with scored candidates
    all_media  = MEDIA   # was: sorted(all_scored["medium"].unique().tolist())
    met_media  = all_scored.groupby("metabolite")["medium"].nunique()

    valid_mets = met_media[met_media >= min_media].index.tolist()
    if not valid_mets:
        print("  No metabolites found for cross-media summary.")
        return

    conserved_mets = met_media[met_media >= 2].index.tolist()
    specific_mets  = met_media[met_media == 1].index.tolist()

    df = all_scored[all_scored["metabolite"].isin(valid_mets)].copy()

    def cgroup(c):
        if "CTR" in c:  return "CTR"
        if "513" in c:  return "CLN-513"
        return "CLN-614"

    df["cgroup"] = df["chain"].map(cgroup)
    # FIX #8: x_cats built from MEDIA (all 5) not from data
    x_cats = [f"{m}\n{g}" for m in all_media for g in ["CTRL", "CLIN-1", "CLIN-2"]]
    x_map  = {v: i for i, v in enumerate(x_cats)}

    grp = (df.groupby(["metabolite", "medium", "cgroup"])
             .agg(score=("score", "max"),
                  category=("category",
                             lambda x: x.iloc[
                                 x.map({"TRUE_CROSSFEED": 3,
                                        "ORDER_DEPENDENT": 2,
                                        "OPPORTUNISTIC":  1})
                                  .fillna(0).astype(int).argmax()]))
             .reset_index())
    grp["xl"] = grp.apply(lambda r: f"{r['medium']}\n{r['cgroup']}", axis=1)

    cons_order = (grp[grp["metabolite"].isin(conserved_mets)]
                  .groupby("metabolite")["score"].sum()
                  .sort_values(ascending=False).index.tolist())
    spec_order = (grp[grp["metabolite"].isin(specific_mets)]
                  .groupby("metabolite")["score"].sum()
                  .sort_values(ascending=False).index.tolist())
    met_order  = cons_order + spec_order
    n_cons     = len(cons_order)

    y_map = {m: i for i, m in enumerate(met_order)}

    fw = max(16, len(x_cats) * 0.95)
    fh = max(8,  len(met_order) * 0.48)
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")   # FIX #10
    ax.set_facecolor("#F9FAFB")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")

    for _, row in grp.iterrows():
        xi  = x_map.get(row["xl"])
        yi  = y_map.get(row["metabolite"])
        if xi is None or yi is None:
            continue
        s       = row["score"]
        sz      = (s ** 1.6) * 85
        color   = CAT_COLORS.get(row["category"], "#999")
        is_cons = row["metabolite"] in conserved_mets

        if is_cons:
            ax.scatter(xi, yi, s=sz, c=color, alpha=0.90, zorder=4,
                       edgecolors="white", linewidths=0.8,
                       marker="o")
        else:
            ax.scatter(xi, yi, s=sz, facecolors="none", edgecolors=color,
                       linewidths=1.8, alpha=0.85, zorder=4,
                       marker="D")

    if n_cons > 0 and n_cons < len(met_order):
        ax.axhline(n_cons - 0.5, color="#555555", lw=1.5,
                   ls="--", zorder=5, alpha=0.7)
        ax.text(len(x_cats) - 0.3, n_cons - 0.55,
                "── medium-specific below ──",
                fontsize=7.5, color="#555", ha="right", va="bottom",
                style="italic")

    ax.set_xlim(-0.5, len(x_cats) - 0.5)
    ax.set_ylim(-0.5, len(met_order) - 0.5)
    ax.grid(True, axis="both", color="#E0E0E0", lw=0.6, zorder=0)

    for i, xc in enumerate(x_cats):
        if i > 0 and xc.split("\n")[0] != x_cats[i - 1].split("\n")[0]:
            ax.axvline(i - 0.5, color="#909090", lw=1.5, ls="--", zorder=1)

    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels(x_cats, fontsize=8.5, color="black")
    ax.set_yticks(range(len(met_order)))
    ax.set_yticklabels([casic_map.get(m, m)[:30] for m in met_order], fontsize=7.5)
    for yt, m in zip(ax.get_yticklabels(), met_order):
        yt.set_color(CLASS_COLORS.get(class_map.get(m, "Unknown"), "#444"))
        yt.set_path_effects([pe.withStroke(linewidth=0.4, foreground="#EEE")])
    ax.tick_params(length=0)

    cat_patches = [
        mpatches.Patch(color=CAT_COLORS[c],
                       label=f"{CATEGORY_DISPLAY.get(c, c)}\n"
                             + CATEGORY_EXPLAIN.get(c, "").split("\n")[0])
        for c in ["TRUE_CROSSFEED", "ORDER_DEPENDENT", "OPPORTUNISTIC"]
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#777",
               markersize=ms, label=f"Score {sc}", markeredgecolor="#333")
        for sc, ms in [(1, 8), (2, 12), (3, 16), (4, 20)]
    ]
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#777",
               markersize=11, markeredgecolor="#333",
               label="Conserved (● ≥2 media)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
               markersize=10, markeredgecolor="#777",
               label="Medium-specific (◇ 1 medium)"),
    ]

    leg1 = ax.legend(handles=cat_patches, title="Category",
                     loc="upper left", bbox_to_anchor=(1.02, 1.00),
                     fontsize=7.5, facecolor="white", edgecolor="#CCC",
                     labelcolor="black", framealpha=0.95, title_fontsize=8)
    leg2 = ax.legend(handles=size_handles, title="Evidence Score",
                     loc="upper left", bbox_to_anchor=(1.02, 0.58),
                     fontsize=8, facecolor="white", edgecolor="#CCC",
                     labelcolor="black", framealpha=0.95, title_fontsize=8)
    leg3 = ax.legend(handles=shape_handles, title="Prevalence",
                     loc="upper left", bbox_to_anchor=(1.02, 0.35),
                     fontsize=8, facecolor="white", edgecolor="#CCC",
                     labelcolor="black", framealpha=0.95, title_fontsize=8)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    n_cons_label = len(conserved_mets)
    n_spec_label = len(specific_mets)
    ax.set_title(
        f"Cross-Media Summary — {n_cons_label} conserved (≥2 media) | "
        f"{n_spec_label} medium-specific\n"
        f"(All 5 media shown: {', '.join(MEDIA)})",   # FIX #8: note all 5 shown
        color="black", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    out = Path(out_dir) / "CF_CrossMedia_Summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


# =============================================================================
# NEW FIGURE 11 — Mirror Chain Comparison Heatmap
# =============================================================================

def fig_mirror_comparison(mean_df, cvdf, bio_feats, casic_map, classmap,
                          chains, medium, out_dir):
    """
    Extended 6-row CF Mirror plot (v14).

    Layout changes vs v13:
    - Figure width = nmets * CELL_W_INCH (cell width fixed, not metabolite-label-driven)
    - Row labels (M-R1 … Mi-R3) moved to LEFT y-axis
    - Strain descriptions moved to RIGHT y-axis (secondary)
    - MAIN CHAIN / MIRROR CHAIN shown as coloured rectangles on the LEFT
      (matching the case-label bar style on top), NOT as text over the y-axis
    - Title simplified: "<ChainA> vs <ChainB>  |  <medium>"
    - Cell aspect ratio: cells are square-ish; figure height is fixed 6 rows
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.gridspec import GridSpec
    from pathlib import Path
    from collections import Counter

    # ── Layout constants ───────────────────────────────────────────────
    CELL_W_INCH  = 0.6   # width per metabolite column (inches)
    CELL_H_INCH  = 0.6   # height per row (inches)
    LEFT_MARGIN  = 0.5    # inches for left labels (M-R1 etc + chain badges)
    RIGHT_MARGIN = 1.0    # inches for strain descriptions
    CBAR_W       = 1.1    # inches for colorbar
    TOP_BAR_H    = 0.45   # inches for case-label bar
    LEGEND_H     = 1.8   # inches below heatmap for legends
    N_ROWS       = 6

    feats = [f for f in bio_feats if f in mean_df.columns]

    for chain_a, chain_b in MIRROR_PAIR_NAMES:
        steps_a = chains.get(chain_a, [])
        steps_b = chains.get(chain_b, [])
        if len(steps_a) < 4 or len(steps_b) < 4:
            continue

        blank_a, solo_a, partner_a, return_a = steps_a[:4]
        blank_b, solo_b, partner_b, return_b = steps_b[:4]

        required = [blank_a, solo_a, partner_a, blank_b, solo_b, partner_b]
        missing  = [k for k in required if k not in mean_df.index]
        if missing:
            print(f"  Mirror {chain_a}/{chain_b}: missing {missing}, skipping.")
            continue

        have_ret_a  = return_a in mean_df.index
        have_ret_b  = return_b in mean_df.index
        blank_ref_b = blank_b if blank_b in mean_df.index else blank_a

        fca_solo    = log2fc(mean_df.loc[solo_a,    feats], mean_df.loc[blank_a,     feats])
        fca_partner = log2fc(mean_df.loc[partner_a, feats], mean_df.loc[solo_a,      feats])
        fca_return  = (log2fc(mean_df.loc[return_a, feats], mean_df.loc[partner_a,   feats])
                       if have_ret_a else pd.Series(np.nan, index=feats))
        fcb_solo    = log2fc(mean_df.loc[solo_b,    feats], mean_df.loc[blank_ref_b, feats])
        fcb_partner = log2fc(mean_df.loc[partner_b, feats], mean_df.loc[solo_b,      feats])
        fcb_return  = (log2fc(mean_df.loc[return_b, feats], mean_df.loc[partner_b,   feats])
                       if have_ret_b else pd.Series(np.nan, index=feats))

        all_fc  = [fca_solo, fca_partner, fca_return,
                   fcb_solo, fcb_partner, fcb_return]

        def _cva(samp):
            if cvdf is not None and samp in cvdf.index:
                return cvdf.loc[samp, feats].values.astype(float)
            return np.zeros(len(feats), dtype=float)

        cv_all  = np.array([
            _cva(solo_a), _cva(partner_a),
            _cva(return_a) if have_ret_a else np.zeros(len(feats)),
            _cva(solo_b),  _cva(partner_b),
            _cva(return_b) if have_ret_b else np.zeros(len(feats)),
        ])
        fc_all  = np.array([s.fillna(0).values for s in all_fc], dtype=float)
        clean   = (np.abs(fc_all) >= FC_THRESH) & (cv_all <= CV_THRESH_LIGHT)
        sdir    = np.sign(fc_all) * clean.astype(float)

        # ── Column filter (v11 quality + biology gate) ──────────────────
        rule_c = clean.any(axis=0)
        check_pairs = [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5),(0,3)]
        rule_a = np.zeros(len(feats), dtype=bool)
        for i, k in check_pairs:
            rule_a |= (clean[i] & clean[k] &
                       (sdir[i] != 0) & (sdir[k] != 0) & (sdir[i] != sdir[k]))
        rule_b  = (clean[1] & ~clean[0]) | (clean[4] & ~clean[3])
        keep    = rule_c & (rule_a | rule_b)
        valid_idx = np.where(keep)[0]

        if len(valid_idx) == 0:
            print(f"  Mirror {chain_a}/{chain_b} [{medium}]: no columns pass — skipping.")
            continue

        fa          = np.array(feats)
        vf          = fa[valid_idx].tolist()
        contrast    = (fca_partner[vf].fillna(0) - fcb_partner[vf].fillna(0)).abs()
        sorted_mets = contrast.sort_values(ascending=False).head(60).index.tolist()
        nmets       = len(sorted_mets)

        plot_raw  = np.array([s[sorted_mets].values.astype(float) for s in all_fc])
        plot_data = np.clip(plot_raw, -6, 6)

        def _cvr(samp):
            if cvdf is not None and samp in cvdf.index:
                return cvdf.loc[samp, sorted_mets].values.astype(float)
            return np.zeros(nmets, dtype=float)

        cv_mat = np.array([
            _cvr(solo_a),    _cvr(partner_a),
            _cvr(return_a) if have_ret_a else np.zeros(nmets),
            _cvr(solo_b),    _cvr(partner_b),
            _cvr(return_b) if have_ret_b else np.zeros(nmets),
        ])

        # ── Classify each column ────────────────────────────────────────
        col_cases = []
        for j, met in enumerate(sorted_mets):
            fi = list(feats).index(met)
            col_cases.append(_classify_column(sdir[:, fi]))

        def _strip(s): return s.replace(f"{medium}_", "")

        # Row short labels (LEFT y-axis)
        row_short = ["M-R1", "M-R2", "M-R3", "Mi-R1", "Mi-R2", "Mi-R3"]

        # Strain descriptions (RIGHT y-axis)
        row_desc = [
            f"{_strip(solo_a)} solo 3h",
            f"{_strip(partner_a)} in {_strip(solo_a)} 5h",
            (f"{_strip(return_a)} returns 20h" if have_ret_a
             else f"{_strip(return_a)} 20h [missing]"),
            f"{_strip(solo_b)} solo 3h",
            f"{_strip(partner_b)} in {_strip(solo_b)} 5h",
            (f"{_strip(return_b)} returns 20h" if have_ret_b
             else f"{_strip(return_b)} 20h [missing]"),
        ]

        # ── Figure sizing ───────────────────────────────────────────────
        heatmap_w  = nmets * CELL_W_INCH
        heatmap_h  = N_ROWS * CELL_H_INCH
        fig_w      = LEFT_MARGIN + heatmap_w + RIGHT_MARGIN + CBAR_W
        fig_h      = TOP_BAR_H + heatmap_h + LEGEND_H

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")

        # Compute axes in figure-fraction coordinates
        lm  = LEFT_MARGIN  / fig_w
        rm  = RIGHT_MARGIN / fig_w
        cbw = CBAR_W       / fig_w
        tbh = TOP_BAR_H    / fig_h
        lgh = LEGEND_H     / fig_h
        hmw = heatmap_w    / fig_w
        hmh = heatmap_h    / fig_h

        # y positions (bottom-up)
        leg_y  = 0.0
        heat_y = lgh
        top_y  = lgh + hmh

        # Heatmap axes
        ax = fig.add_axes([lm, heat_y, hmw, hmh])
        # Case-label bar axes (directly above heatmap, same x extent)
        ax_lab = fig.add_axes([lm, top_y, hmw, tbh])
        # Colorbar axes (right of heatmap)
        cbar_x = lm + hmw + 0.01
        ax_cb  = fig.add_axes([cbar_x, heat_y + hmh*0.1, cbw*0.35, hmh*0.8])

        for a in [ax, ax_lab, ax_cb]:
            a.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")

        # ── Case label bar ──────────────────────────────────────────────
        ax_lab.set_xlim(-0.5, nmets - 0.5)
        ax_lab.set_ylim(0, 1)
        ax_lab.axis("off")

        # Group consecutive columns of same case → one wide rectangle
        groups = []
        j = 0
        while j < nmets:
            case = col_cases[j]
            k = j
            while k < nmets and col_cases[k] == case:
                k += 1
            groups.append((j, k-1, case))
            j = k

        for (j0, j1, case) in groups:
            if not case or case not in CF_CASE_STYLES:
                j = j1 + 1
                continue
            tc, bg, _ = CF_CASE_STYLES[case]
            cx   = (j0 + j1) / 2.0
            wid  = (j1 - j0 + 1) - 0.08
            ax_lab.add_patch(plt.Rectangle(
                (j0 - 0.46, 0.08), wid + 0.92, 0.84,
                facecolor=bg, edgecolor=tc, lw=0.9, zorder=3, clip_on=False))
            short = (case.replace("↔", "=").replace("→", ">")
                        .replace("SHIFT-", "S-").replace("-20h", "20h"))
            ax_lab.text(cx, 0.52, short,
                        ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color=tc, zorder=4, clip_on=False)

        # ── Heatmap ─────────────────────────────────────────────────────
        norm = TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
        im   = ax.imshow(plot_data, aspect="auto", cmap="seismic",
                         norm=norm, interpolation="nearest")

        for i in range(N_ROWS):
            for j in range(nmets):
                cv_v  = cv_mat[i, j]
                fc_ok = np.abs(plot_raw[i, j]) >= FC_THRESH
                ag, hp, lv = _cv_hatch(cv_v)
                if not fc_ok:
                    ax.add_patch(plt.Rectangle(
                        (j-0.5, i-0.5), 1, 1,
                        facecolor="#C0C0C0", alpha=0.75,
                        hatch="////", edgecolor="white", lw=0, zorder=3))
                elif lv > 0:
                    ax.add_patch(plt.Rectangle(
                        (j-0.5, i-0.5), 1, 1,
                        facecolor="#808080", alpha=ag,
                        hatch=hp, edgecolor="white", lw=0, zorder=3))

        # Grid lines
        for j in range(nmets + 1):
            ax.axvline(j - 0.5, color="#DDDDDD", lw=0.4, zorder=2)
        for i in range(1, N_ROWS):
            lw = 2.0 if i == 3 else 0.8
            c  = "#444444" if i == 3 else "#AAAAAA"
            ax.axhline(i - 0.5, color=c, lw=lw, zorder=4)
        ax.axhline(-0.5,        color="#888888", lw=1.0, zorder=2)
        ax.axhline(N_ROWS-0.5,  color="#888888", lw=1.0, zorder=2)

        # ── LEFT y-axis: row short labels (M-R1 … Mi-R3) ───────────────
        ax.set_yticks(range(N_ROWS))
        ax.set_yticklabels(row_short, fontsize=9.5, fontweight="bold",
                           color="black")
        ax.tick_params(axis="y", length=0, pad=4)

        # ── RIGHT y-axis: strain descriptions ───────────────────────────
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(N_ROWS))
        ax2.set_yticklabels(row_desc, fontsize=7.5, color="#333333")
        ax2.tick_params(axis="y", length=0, pad=4)
        for sp in ax2.spines.values():
            sp.set_visible(False)

        # ── MAIN CHAIN / MIRROR CHAIN badges on the left ────────────────
        # Draw as coloured rectangles just to the left of the heatmap axes,
        # using figure coordinates converted from data coordinates.
        badge_x0 = -0.5   # in data x units (just left of first cell)
        badge_w  = 0.0     # width irrelevant; we draw in axes-fraction space

        # Use axes transform: position in (axes-fraction-x, data-y) space
        trans_data = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData)

        # Badge horizontal extent in axes-fraction
        badge_ax_x0 = -(LEFT_MARGIN - 0.15) / heatmap_w  # left edge of badge
        badge_ax_x1 = -0.02                               # right edge (just left of yticks)
        badge_ax_w  = badge_ax_x1 - badge_ax_x0

        for (label, color_face, color_edge, y_lo, y_hi) in [
            ("MAIN\nCHAIN",   "#E8F5E9", "#1B5E20", -0.48,  2.48),
            ("MIRROR\nCHAIN", "#E3F2FD", "#0D47A1",  2.52,  5.48),
        ]:
            ax.add_patch(mpatches.FancyBboxPatch(
                (badge_ax_x0, y_lo),
                badge_ax_w, y_hi - y_lo,
                boxstyle="round,pad=0.02",
                transform=trans_data,
                facecolor=color_face, edgecolor=color_edge, lw=1.0,
                clip_on=False, zorder=5))
            ax.text((badge_ax_x0 + badge_ax_x1) / 2,
                    (y_lo + y_hi) / 2,
                    label,
                    transform=trans_data,
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=color_edge, zorder=6, clip_on=False)

        # ── X-axis: metabolite names ─────────────────────────────────────
        ax.set_xticks(range(nmets))
        x_labels = [casic_map.get(m, m) for m in sorted_mets]
        # Truncate very long names
        x_labels = [lb[:28] if len(lb) > 28 else lb for lb in x_labels]
        ax.set_xticklabels(x_labels, rotation=55, ha="right",
                           fontsize=7.5, color="black")
        ax.tick_params(axis="x", length=0, pad=2)
        for xt, met in zip(ax.get_xticklabels(), sorted_mets):
            xt.set_color(CLASS_COLORS.get(classmap.get(met, "Unknown"), "#555555"))

        # ── Colorbar ─────────────────────────────────────────────────────
        cbar = fig.colorbar(im, cax=ax_cb, orientation="vertical")
        cbar.set_label("log₂ FC (±6)", color="black", fontsize=8)
        cbar.ax.tick_params(colors="black", labelsize=7)

        # ── Title (simplified) ───────────────────────────────────────────
        title_str = f"{chain_a} vs {chain_b}  |  {medium}"
        ax_lab.set_title(title_str, color="black", fontsize=11,
                         fontweight="bold", pad=5)

        # ── Legends (placed in figure coords below heatmap) ──────────────
        legend_ax = fig.add_axes([lm, 0.0, hmw + rm * 0.6, lgh * 0.98])
        legend_ax.axis("off")

        hp_patches = [
            mpatches.Patch(facecolor="#D62728", label="Produced (+FC)"),
            mpatches.Patch(facecolor="#3182BD", label="Consumed (−FC)"),
            mpatches.Patch(facecolor="#C0C0C0", hatch="////",
                           label=f"|FC|<{FC_THRESH:.1f} or CV>{int(CV_THRESH_MEDIUM*100)}%"),
            mpatches.Patch(facecolor="#BBBBBB", alpha=0.35, hatch="//",
                           label=f"CV {int(CV_THRESH_LIGHT*100)}–{int(CV_THRESH_MEDIUM*100)}% borderline"),
            mpatches.Patch(facecolor="white", edgecolor="#888",
                           label=f"CV≤{int(CV_THRESH_LIGHT*100)}% reliable"),
        ]
        leg1 = legend_ax.legend(
            handles=hp_patches,
            loc="upper left", bbox_to_anchor=(0.0, 1.0),
            ncol=5, fontsize=7.5,
            facecolor="white", edgecolor="#CCCCCC",
            labelcolor="black", framealpha=0.95,
            title="Cell encoding", title_fontsize=7.5)
        legend_ax.add_artist(leg1)

        used = [c for c in CF_CASE_ORDER if c in col_cases]
        case_patches = []
        for case in used:
            tc, bg, desc = CF_CASE_STYLES[case]
            short = case.replace("↔","=").replace("→",">")
            case_patches.append(
                mpatches.Patch(facecolor=bg, edgecolor=tc, lw=1.2,
                               label=f"{short}  —  {desc}"))
        if case_patches:
            legend_ax.legend(
                handles=case_patches,
                loc="upper left", bbox_to_anchor=(0.0, 0.52),
                ncol=min(3, len(case_patches)),
                fontsize=7.5, facecolor="white",
                edgecolor="#CCCCCC", labelcolor="black",
                framealpha=0.95,
                title="Column case labels (top bar)",
                title_fontsize=7.5)

        # ── Save ─────────────────────────────────────────────────────────
        safe     = f"{chain_a}_vs_{chain_b}".replace("/","_").replace(" ","_")
        out_path = Path(out_dir) / f"CF_Mirror_{safe}_{medium}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out_path.name}")
        cnt = dict(Counter(c for c in col_cases if c))
        print("  Cases: " + "  |  ".join(
            f"{c.replace(chr(8596),'=').replace(chr(8594),'>')}: {cnt[c]}"
            for c in CF_CASE_ORDER if c in cnt))

def fig_mirror_explanation(out_dir):
    """
    Standalone colour-interpretation guide — v1.4 (6-row extended chain).

    New vs v1.3:
    - Updated to 6-row layout (M-R1..M-R3 / Mi-R1..Mi-R3)
    - 2-level hatch encoding documented in the colour key
    - Group 6: 20h Return Signal (what M-R3 tells you about nutrient adaptation)
    - Group 7: Mirror-chain interpretation patterns
    """
    R = "#D62728"
    B = "#3182BD"
    G = "#CCCCCC"

    groups = [
      {
        "name": "Cross-Feeding  P\u2192E  (P produces X, E consumes X)",
        "color": "#1B5E20",
        "scenarios": [
          {
            "title": "True CF\n(Score 4) \u2605",
            "colors": [R, B, R,  G, G, G],
            "interp": [
              "M-R1: P accumulates X alone",
              "M-R2: X drops when E arrives",
              "M-R3: P still produces X at 20h",
              "Mi: E alone does NOT make X",
              "\u2192 Genuine CF: P\u2192X\u2192E confirmed",
            ],
          },
          {
            "title": "Putative CF\n(Score 2-3)",
            "colors": [R, B, R,  R, G, G],
            "interp": [
              "M-R1: P accumulates X alone",
              "M-R2: X drops when E arrives",
              "Mi-R1: E ALSO makes X alone",
              "\u2192 CF likely; mirror unconfirmed",
            ],
          },
          {
            "title": "CF + E also\ndepletes X alone",
            "colors": [R, B, B,  G, B, G],
            "interp": [
              "M-R1: P accumulates X alone",
              "M-R2: E consumes X in co-culture",
              "Mi-R2: E also depletes X alone",
              "\u2192 E has constitutive affinity for X",
            ],
          },
        ],
      },
      {
        "name": "Cross-Feeding  E\u2192P  (E produces X in co-culture context)",
        "color": "#0D47A1",
        "scenarios": [
          {
            "title": "E produces\nfor P (CF only)",
            "colors": [G, R, G,  G, G, G],
            "interp": [
              "M-R1: P does not produce X",
              "M-R2: X rises when E is present",
              "Mi-R1: E does NOT produce X alone",
              "\u2192 E produces X only in co-culture",
            ],
          },
          {
            "title": "E produces\nalways",
            "colors": [G, R, G,  R, G, G],
            "interp": [
              "M-R1: P does not produce X",
              "M-R2: X rises when E is present",
              "Mi-R1: E also produces X alone",
              "\u2192 E is a constitutive X producer",
            ],
          },
          {
            "title": "P produces +\nCF amplifies",
            "colors": [R, R, G,  G, G, G],
            "interp": [
              "M-R1: P produces X alone",
              "M-R2: X increases further in CF",
              "Mi: no solo mirror production",
              "\u2192 CF stimulates more X from P",
            ],
          },
          {
            "title": "All conditions\nproduce X",
            "colors": [R, R, R,  R, R, R],
            "interp": [
              "All 6 rows show accumulation",
              "Both strains produce X alone",
              "\u2192 Non-specific; not CF-driven",
            ],
          },
        ],
      },
      {
        "name": "Independent Metabolism  (no CF signal)",
        "color": "#E65100",
        "scenarios": [
          {
            "title": "P produces,\nno E interaction",
            "colors": [R, G, R,  G, G, G],
            "interp": [
              "M-R1: P produces X alone",
              "M-R2: no change when E arrives",
              "M-R3: X persists at 20h",
              "\u2192 P makes X; E does not use it",
            ],
          },
          {
            "title": "E produces,\nno P interaction",
            "colors": [G, G, G,  R, G, R],
            "interp": [
              "Mi-R1: E produces X alone",
              "Mi-R3: X persists at 20h mirror",
              "M-R2: no co-culture effect",
              "\u2192 E makes X; P does not use it",
            ],
          },
          {
            "title": "Both produce\nindependently",
            "colors": [R, G, R,  R, G, R],
            "interp": [
              "P and E both produce X alone",
              "No CF signal in either direction",
              "\u2192 Both make X; not CF-specific",
            ],
          },
        ],
      },
      {
        "name": "Depletion Patterns  (X decreases vs reference condition)",
        "color": "#4A148C",
        "scenarios": [
          {
            "title": "Co-culture\nconsumption",
            "colors": [G, B, G,  G, G, G],
            "interp": [
              "M-R1: P does not produce X",
              "M-R2: X drops when E arrives",
              "Mi: E alone does not deplete X",
              "\u2192 X consumed in CF; source unclear",
            ],
          },
          {
            "title": "E makes then\nconsumes X",
            "colors": [G, B, G,  R, G, G],
            "interp": [
              "M-R1: P does not produce X",
              "M-R2: X drops when E arrives",
              "Mi-R1: E produces X alone",
              "\u2192 E makes X; reabsorbs it in CF",
            ],
          },
          {
            "title": "P & CF both\ndeplete X",
            "colors": [B, B, G,  G, G, G],
            "interp": [
              "M-R1: P depletes X vs blank",
              "M-R2: CF depletes X further",
              "Mi: no mirror depletion alone",
              "\u2192 P uses medium X; E amplifies",
            ],
          },
          {
            "title": "All conditions\ndeplete X",
            "colors": [B, B, B,  B, B, B],
            "interp": [
              "All 6 rows show depletion",
              "X consumed from medium by both",
              "\u2192 X is consumed from medium",
            ],
          },
        ],
      },
      {
        "name": "20h Return Signal  (what M-R3 / Mi-R3 reveals about nutrient adaptation)",
        "color": "#006064",
        "scenarios": [
          {
            "title": "Strain returns,\nstill produces",
            "colors": [R, B, R,  G, G, G],
            "interp": [
              "M-R3 = Red: P rebounds at 20h",
              "X produced again despite E's use",
              "\u2192 Scarce nutrients did NOT stop",
              "  P's production of X",
            ],
          },
          {
            "title": "Strain returns,\nnow consumes X",
            "colors": [R, B, B,  G, G, G],
            "interp": [
              "M-R3 = Blue: P switches to uptake",
              "X is now scarce (E depleted it)",
              "\u2192 Nutrient scarcity flipped P's",
              "  metabolic strategy at 20h",
            ],
          },
          {
            "title": "Strain returns,\nno change (grey)",
            "colors": [R, B, G,  G, G, G],
            "interp": [
              "M-R3 = Grey: no signal at 20h",
              "X neither made nor consumed",
              "\u2192 Ambiguous; X may be absent",
            ],
          },
          {
            "title": "Mirror also\nshows return",
            "colors": [R, B, R,  B, R, B],
            "interp": [
              "Both chains show oscillation",
              "Main: P makes, E takes, P rebounds",
              "Mirror: E takes, P makes, E takes",
              "\u2192 Reciprocal cycling of X",
            ],
          },
        ],
      },
      {
        "name": "No Signal / Below Threshold  (grey = |FC|<1.0 or CV>50%)",
        "color": "#616161",
        "scenarios": [
          {
            "title": "No CF\n(all grey)",
            "colors": [G, G, G,  G, G, G],
            "interp": [
              "All 6 cells grey / hatched",
              "|log\u2082FC|<1.0 or CV>50% everywhere",
              "\u2192 No reliable CF signal",
            ],
          },
          {
            "title": "P depletes\nmedium only",
            "colors": [B, G, G,  G, G, G],
            "interp": [
              "M-R1: P depletes X vs blank",
              "No CF effect in other rows",
              "\u2192 A uses medium X; not CF",
            ],
          },
          {
            "title": "E depletes\nmedium only",
            "colors": [G, G, G,  B, G, G],
            "interp": [
              "Mi-R1: E depletes X alone",
              "No CF effect in main rows",
              "\u2192 E uses medium X; not CF",
            ],
          },
        ],
      },
    ]

    CELL_W   = 0.30
    CELL_H   = 0.26
    CELL_GAP = 0.03
    TITLE_H  = 0.55
    TEXT_H   = 0.95
    COL_GAP  = 2.90
    GRP_PAD  = 0.40

    SWATCH_H = 6 * CELL_H + 5 * CELL_GAP
    BLOCK_H  = TITLE_H + SWATCH_H + TEXT_H

    max_cols = max(len(g["scenarios"]) for g in groups)
    n_groups = len(groups)

    FIG_W = max_cols * COL_GAP + 1.6
    FIG_H = n_groups * (BLOCK_H + GRP_PAD) + 3.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax0 = fig.add_axes([0, 0, 1, 1])
    ax0.set_xlim(0, FIG_W)
    ax0.set_ylim(0, FIG_H)
    ax0.axis("off")

    ax0.text(FIG_W / 2, FIG_H - 0.18,
             "Mirror Cross-Feeding Plot \u2014 Complete Colour Interpretation Guide (6-Row Design)",
             ha="center", va="top", fontsize=13, fontweight="bold", color="black")

    key_y = FIG_H - 0.44
    key_lines = [
        "Row   legend    \u2502  Main-chain block (top 3 rows):",
        "M-R1  (top)    = Strain A solo vs blank       \u2022 what does A produce alone?",
        "M-R2  (middle) = Strain B in A\u2019s spent medium  \u2022 what does B do when it arrives?",
        "M-R3           = Strain A returns 20h         \u2022 does A still produce? or switch to consume under scarce nutrients?",
        "Mirror-chain block (bottom 3 rows):",
        "Mi-R1          = Strain B solo vs blank       \u2022 does B make it alone?",
        "Mi-R2          = Strain A in B\u2019s spent medium  \u2022 reciprocal CF direction check",
        "Mi-R3          = Strain B returns 20h         \u2022 mirror return signal",
    ]
    for ki, kl in enumerate(key_lines):
        ax0.text(0.18, key_y - ki * 0.155, kl,
                 ha="left", va="top", fontsize=6.8, color="#222222",
                 fontfamily="monospace")

    hatch_y = key_y - len(key_lines) * 0.155 - 0.30
    hatch_items = [
        ("#D62728", "",     "Red  = produced / accumulated (+log\u2082FC)"),
        ("#3182BD", "",     "Blue = consumed / depleted (\u2212log\u2082FC)"),
        ("#C0C0C0", "////", f"Grey/////  = |FC|<{FC_THRESH:.1f} or CV>{int(CV_THRESH_MEDIUM*100)}%"),
        ("#AAAAAA", "//",   f"//  = CV {int(CV_THRESH_LIGHT*100)}\u2013{int(CV_THRESH_MEDIUM*100)}% (borderline, colour visible)"),
        ("white",   "",     f"No hatch  = CV \u2264{int(CV_THRESH_LIGHT*100)}% (reliable)"),
    ]
    for hi, (fc_col, ht, lbl) in enumerate(hatch_items):
        bx = 0.18 + hi * 2.08
        rect = plt.Rectangle((bx, hatch_y), 0.28, 0.22,
                              facecolor=fc_col, hatch=ht,
                              edgecolor="#666666", lw=0.8)
        ax0.add_patch(rect)
        ax0.text(bx + 0.32, hatch_y + 0.10, lbl,
                 ha="left", va="center", fontsize=6.8, color="#222222")

    y_cursor = hatch_y - 0.42

    for grp in groups:
        y_cursor -= 0.08
        ax0.add_patch(plt.Rectangle(
            (0.08, y_cursor - 0.24), FIG_W - 0.16, 0.24,
            facecolor=grp["color"], alpha=0.13, lw=0, zorder=0))
        ax0.text(0.16, y_cursor - 0.07, grp["name"],
                 ha="left", va="center", fontsize=9.5,
                 fontweight="bold", color=grp["color"])

        y_cursor -= 0.32

        for sc_i, sc in enumerate(grp["scenarios"]):
            bx = 0.18 + sc_i * COL_GAP
            ax0.text(bx + CELL_W / 2, y_cursor, sc["title"],
                     ha="center", va="top", fontsize=7.2,
                     fontweight="bold", color="#222222")

            sw_top = y_cursor - TITLE_H
            for ri, col in enumerate(sc["colors"]):
                cy = sw_top - ri * (CELL_H + CELL_GAP)
                ax0.add_patch(plt.Rectangle(
                    (bx, cy - CELL_H), CELL_W, CELL_H,
                    facecolor=col, edgecolor="#888888", lw=0.6, zorder=4))
                if ri == 3:
                    ax0.plot([bx, bx + CELL_W], [cy, cy],
                             color="#444444", lw=1.5,
                             solid_capstyle="butt", zorder=5)
                lbl = f"M-R{ri+1}" if ri < 3 else f"Mi-R{ri-2}"
                ax0.text(bx - 0.03, cy - CELL_H / 2, lbl,
                         ha="right", va="center",
                         fontsize=5.5, color="#666666")

            txt_top = sw_top - SWATCH_H - 0.08
            for ti, line in enumerate(sc["interp"]):
                ax0.text(bx, txt_top - ti * 0.175, line,
                         ha="left", va="top", fontsize=6.2,
                         color="#333333",
                         style="italic" if line.startswith("\u2192") else "normal")

        y_cursor -= BLOCK_H + GRP_PAD

    plt.savefig(Path(out_dir) / "CF_Mirror_Explanation_v2.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: CF_Mirror_Explanation_v2.png")


# =============================================================================
# NEW FIGURE 12 — Per-Metabolite Cross-Feeding Summary Table
# =============================================================================

def fig_candidate_table(scored, casic_map, class_map, medium, out_dir,
                         max_rows=20, min_score=2):
    """
    NEW (v1.6, fix #12): Publication-ready table figure.
    Columns: Metabolite | Class | Producer | Consumer | Score | Category |
             FC produced | FC consumed | Mirror confirmed
    Rows colour-coded: green=4, orange=3, yellow=2.
    Max 20 rows (top by score then |FC|).
    Saved as: CF_CandidateTable_[medium].png
    """
    if scored is None or scored.empty:
        return

    df = scored[scored["score"] >= min_score].copy()
    if df.empty:
        return

    # Direction A = consumed_by_partner: producer=solo strain, consumer=partner
    # Use direction A for fc_produced / fc_consumed
    df_a = df[df["direction"] == "consumed_by_partner"].copy()
    df_b = df[df["direction"] == "produced_by_partner"].copy()
    df_use = pd.concat([df_a, df_b], ignore_index=True)

    if df_use.empty:
        return

    # Sort: score desc, then |fc_partner| desc
    df_use["_abs_fc"] = df_use["fc_partner"].abs()
    df_use = df_use.sort_values(["score", "_abs_fc"], ascending=[False, False])
    df_use = df_use.drop_duplicates(subset=["metabolite", "chain", "direction"])
    df_use = df_use.head(max_rows).reset_index(drop=True)

    if df_use.empty:
        return

    # Build table rows
    # NEW — keys match the v7 chain names
    def get_producer_consumer(row):
        cname = row["chain"]
        _strains = {
            "CLN-513p": ("Tmp05", "Tme13"),
            "CLN-513e": ("Tme13", "Tmp05"),
            "CLN-614p": ("Tmp06", "Tme14"),
            "CLN-614e": ("Tme14", "Tmp06"),
            "CTR-1p": ("PA01", "K12"),
            "CTR-1e": ("K12", "PA01"),
        }
        prod, cons = _strains.get(cname, (cname, "?"))  # fallback shows chain name not "?"
        if row["direction"] == "produced_by_partner":
            prod, cons = cons, prod  # dir B: partner produces, solo consumes at 20h
        return prod, cons

    table_rows = []
    for _, row in df_use.iterrows():
        prod, cons = get_producer_consumer(row)
        met_name   = casic_map.get(row["metabolite"], row["metabolite"])[:28]
        cls        = class_map.get(row["metabolite"], "Unknown")
        score      = int(row["score"])
        cat        = CATEGORY_DISPLAY.get(row["category"], row["category"])
        fc_prod    = f"{row.get('fc_solo', float('nan')):.2f}" if not (
            isinstance(row.get("fc_solo"), float) and np.isnan(row.get("fc_solo", np.nan))) else "—"
        fc_cons    = f"{row['fc_partner']:.2f}"
        mirror_ok  = "✓" if row.get("no_mirror_solo", False) else "✗"
        table_rows.append([met_name, cls, prod, cons, score, cat, fc_prod, fc_cons, mirror_ok])

    if not table_rows:
        return

    col_headers = ["Metabolite", "Class", "Producer", "Consumer",
                   "Score", "Category", "FC prod", "FC cons", "Mirror?"]

    # Row background colors
    row_bg = {4: "#C8E6C9", 3: "#FFE0B2", 2: "#FFF9C4"}   # green, orange, yellow

    n_rows = len(table_rows)
    n_cols = len(col_headers)
    cell_h = 0.38
    fh = max(3.5, n_rows * cell_h + 1.5)
    fw = 18

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")   # FIX #10
    ax.set_axis_off()

    col_widths = [0.22, 0.10, 0.07, 0.07, 0.05, 0.16, 0.07, 0.07, 0.07]
    # Normalise to sum=1
    total_w = sum(col_widths)
    col_widths = [w / total_w for w in col_widths]

    # Draw header
    header_y = 1.0
    row_height = 1.0 / (n_rows + 1.5)
    header_y = 1.0 - row_height * 0.5

    # Use matplotlib table
    cell_text = [[str(v) for v in r] for r in table_rows]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#263238")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("#455A64")

    # Style data rows
    for i, row_data in enumerate(table_rows):
        score_val = row_data[4]
        bg = row_bg.get(score_val, "#FFFFFF")
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#B0BEC5")
            cell.set_text_props(color="black", fontsize=8)
            # Colour metabolite name by class
            if j == 0:
                cls  = row_data[1]
                ccol = CLASS_COLORS.get(cls, "#222222")
                cell.set_text_props(color=ccol, fontweight="bold", fontsize=7.5)
            # Colour score cell
            if j == 4:
                cell.set_text_props(fontweight="bold", fontsize=9,
                                    color=CAT_MARKER_COLORS.get(
                                        {4: "TRUE_CROSSFEED",
                                         3: "ORDER_DEPENDENT",
                                         2: "ORDER_DEPENDENT"}.get(score_val, "OPPORTUNISTIC"),
                                        "#333"))
            # Mirror confirmed: green tick / red cross
            if j == 8:
                cell.set_text_props(color="#2E7D32" if row_data[8] == "✓" else "#C62828",
                                    fontweight="bold", fontsize=9)

    # Set column widths
    for j, cw in enumerate(col_widths):
        for i in range(n_rows + 1):
            tbl[i, j].set_width(cw)

    ax.set_title(
        f"{medium} | Cross-Feeding Candidate Table  (score ≥ {min_score}, top {n_rows})\n"
        f"Row colour: green=score 4 (True CF), orange=3, yellow=2 (Putative CF)",
        color="black", fontsize=11, fontweight="bold", pad=14,
        transform=fig.transFigure, y=0.97, x=0.5, ha="center")

    plt.tight_layout()
    out = Path(out_dir) / f"CF_CandidateTable_{medium}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")

def fig_normalisation_qc(raw_is_meandf, raw_bio_meandf,
                          mean_df, cvdf,
                          bio_feats, is_feats, casic_map, classmap,
                          medium, out_dir,
                          is_stability_thresh=0.25):
    """
    6-panel IS Normalisation Quality Control figure (v2).

    Changes vs v1
    -------------
    - All IS labels use casic_map (CASIC/common name) not raw column identifiers
    - Panel 3 replaced: correction-factor bar → IS peak-area box plots
      (distribution of each IS across all samples; exposes outlier injections)
    - IS Stability panel sorted CV ascending (most stable IS at left)
    - Supra-title summary: n stable IS used, n excluded, n samples

    Parameters
    ----------
    raw_is_meandf  : DataFrame (sample × IS columns), RAW peak areas, per-sample mean
    raw_bio_meandf : DataFrame (sample × bio columns), RAW peak areas, per-sample mean
    mean_df        : DataFrame (sample × bio columns), IS-normalised mean
    cvdf           : DataFrame (sample × bio columns), CV of tech reps (0-1)
    bio_feats      : list of biological feature column names
    is_feats       : list of IS column names
    casic_map      : {col_name: CASIC metabolite name}
    classmap       : {col_name: metabolite class string}
    medium         : str — medium name (e.g. 'AUM')
    out_dir        : str or Path — output directory
    is_stability_thresh : float — CV threshold for IS stability (default 0.25 = 25 %)

    Panels
    ------
    [0,0] IS Peak Areas heatmap      — injection quality per IS per sample
    [0,1] IS Stability bar chart     — which IS are stable enough to use
    [1,0] IS Peak-Area box plots     — spread / outliers per IS across samples  (NEW)
    [1,1] Median CV before vs after  — normalisation effect per metabolite class
    [2,0] PCA before normalisation   — raw data structure
    [2,1] PCA after normalisation    — IS-corrected data structure
    """
    if raw_is_meandf is None or raw_is_meandf.empty:
        print(f"  NormQC [{medium}]: no IS data — skipping.")
        return

    # ── resolve IS columns present in this sheet ──────────────────────────
    present_is = [f for f in is_feats if f in raw_is_meandf.columns]
    if not present_is:
        print(f"  NormQC [{medium}]: no IS columns found — skipping.")
        return

    # ── CASIC label helper ────────────────────────────────────────────────
    def is_label(col, maxlen=30):
        """Return CASIC name for an IS column, stripped of leading '(IS)' tag."""
        raw = casic_map.get(col, col)
        for prefix in ["(IS) ", "(IS)", "(is) "]:
            if raw.upper().startswith(prefix.upper()):
                raw = raw[len(prefix):].strip()
                break
        return raw[:maxlen]

    # ── IS stability: CV across samples ───────────────────────────────────
    is_cv = {}
    for col in present_is:
        vals = raw_is_meandf[col].replace(0, np.nan).dropna()
        if len(vals) >= 2 and vals.mean() != 0:
            is_cv[col] = float(vals.std(ddof=1) / vals.mean())
        else:
            is_cv[col] = np.nan

    sorted_is   = sorted(present_is, key=lambda c: (is_cv.get(c) or 9))
    stable_is   = [c for c in sorted_is if (is_cv.get(c) or 9) <= is_stability_thresh]
    unstable_is = [c for c in sorted_is if c not in stable_is]
    n_stable    = len(stable_is)
    n_total     = len(sorted_is)
    n_samples   = len(raw_is_meandf)

    # ── figure layout: 3×2 ────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18), facecolor="#FFFFFF")
    gs  = fig.add_gridspec(3, 2,
                           hspace=0.52, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.06)

    ax_hm   = fig.add_subplot(gs[0, 0])   # IS Peak Areas heatmap
    ax_stab = fig.add_subplot(gs[0, 1])   # IS Stability bars
    ax_box  = fig.add_subplot(gs[1, 0])   # IS box plots  (NEW)
    ax_cv   = fig.add_subplot(gs[1, 1])   # CV before / after
    ax_pca0 = fig.add_subplot(gs[2, 0])   # PCA before
    ax_pca1 = fig.add_subplot(gs[2, 1])   # PCA after

    for ax in (ax_hm, ax_stab, ax_box, ax_cv, ax_pca0, ax_pca1):
        ax.set_facecolor("#F8F9FA")
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")
        ax.tick_params(colors="#333333", labelsize=8)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 0,0 — IS Peak Areas heatmap  (median-scaled per IS)
    # ════════════════════════════════════════════════════════════════════════
    is_matrix = raw_is_meandf[sorted_is].replace(0, np.nan).astype(float)
    is_median  = is_matrix.median(axis=0).replace(0, np.nan)
    is_scaled  = is_matrix.div(is_median, axis=1)

    im = ax_hm.imshow(is_scaled.T.values,
                      aspect="auto", cmap="RdYlGn",
                      vmin=0.6, vmax=1.4, interpolation="nearest")

    sample_labels = [s.replace(f"{medium}_", "") for s in is_matrix.index]
    ax_hm.set_xticks(range(len(sample_labels)))
    ax_hm.set_xticklabels(sample_labels, rotation=65, ha="right",
                          fontsize=6.0, color="black")
    ax_hm.set_yticks(range(len(sorted_is)))
    ax_hm.set_yticklabels(
        [f"{is_label(c)}  {'✓' if c in stable_is else '✗'}"
         for c in sorted_is],
        fontsize=7.5, color="black")

    for yi, col in enumerate(sorted_is):
        bg = "#FFF0F0" if col in unstable_is else "#F0FFF0"
        ax_hm.add_patch(plt.Rectangle(
            (-0.5, yi - 0.5), len(sample_labels), 1,
            facecolor=bg, alpha=0.30, zorder=0))

    cbar_hm = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.01,
                           orientation="vertical", aspect=20)
    cbar_hm.set_label("Relative to median\n(1.0 = on-target)",
                      fontsize=7, color="black")
    cbar_hm.ax.tick_params(colors="black", labelsize=6.5)

    ax_hm.set_title(
        f"{medium}  |  IS Peak Areas  (median-scaled per IS)\n"
        f"✓ = stable (used)   ✗ = deviant (excluded if CV > "
        f"{int(is_stability_thresh * 100)}\u202f%)",
        fontsize=9, fontweight="bold", color="black", pad=8)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 0,1 — IS Stability bar chart  (sorted by CV ascending)
    # ════════════════════════════════════════════════════════════════════════
    cvs_pct = [(is_cv.get(c) or np.nan) * 100 for c in sorted_is]
    bar_cols = ["#2E7D32" if c in stable_is else "#C62828" for c in sorted_is]

    bars = ax_stab.bar(range(len(sorted_is)), cvs_pct,
                       color=bar_cols, edgecolor="white", linewidth=0.5)
    for bar, cv_val in zip(bars, cvs_pct):
        if not np.isnan(cv_val):
            ax_stab.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.4,
                         f"{cv_val:.1f}\u202f%",
                         ha="center", va="bottom", fontsize=7,
                         color="black", fontweight="bold")

    ax_stab.axhline(is_stability_thresh * 100, color="#E53935",
                    lw=1.5, ls="--",
                    label=f"{int(is_stability_thresh * 100)}\u202f% stability threshold")
    ax_stab.legend(fontsize=7.5, framealpha=0.9, labelcolor="black",
                   facecolor="white", edgecolor="#CCCCCC")
    ax_stab.set_xticks(range(len(sorted_is)))
    ax_stab.set_xticklabels([is_label(c) for c in sorted_is],
                             rotation=40, ha="right", fontsize=7.5, color="black")
    ax_stab.set_ylabel("CV across samples (%)", fontsize=8.5, color="black")
    ax_stab.set_xlim(-0.5, len(sorted_is) - 0.5)
    valid_cvs = [v for v in cvs_pct if not np.isnan(v)]
    ax_stab.set_ylim(0, max(max(valid_cvs) * 1.22, 30) if valid_cvs else 30)
    ax_stab.set_title(
        f"{medium}  |  IS Stability  (CV across samples)\n"
        f"Green = stable (used)   Red = unstable (excluded if > "
        f"{int(is_stability_thresh * 100)}\u202f%)",
        fontsize=9, fontweight="bold", color="black", pad=8)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1,0 — IS Peak-Area box plots  (NEW — replaces correction-factor bars)
    # One box per IS across all samples — exposes outlier injections
    # ════════════════════════════════════════════════════════════════════════
    bp_data   = [is_matrix[col].dropna().values for col in sorted_is]
    bp_cols   = ["#A5D6A7" if c in stable_is else "#EF9A9A" for c in sorted_is]

    bplot = ax_box.boxplot(
        bp_data,
        patch_artist=True,
        medianprops=dict(color="#333333", lw=1.8),
        whiskerprops=dict(color="#666666", lw=1.0),
        capprops=dict(color="#666666", lw=1.0),
        flierprops=dict(marker="o", markerfacecolor="#E53935",
                        markersize=3.5, markeredgewidth=0, alpha=0.7),
        boxprops=dict(lw=0.8))

    for patch, col_color in zip(bplot["boxes"], bp_cols):
        patch.set_facecolor(col_color)
        patch.set_alpha(0.85)

    ax_box.set_xticks(range(1, len(sorted_is) + 1))
    ax_box.set_xticklabels([is_label(c) for c in sorted_is],
                            rotation=40, ha="right", fontsize=7.5, color="black")
    ax_box.set_ylabel("Raw peak area", fontsize=8.5, color="black")
    ax_box.set_title(
        f"{medium}  |  IS Peak Area Distribution  (each box = all samples)\n"
        "Red dots = outlier injections   "
        "Green = stable IS   Pink = excluded IS",
        fontsize=9, fontweight="bold", color="black", pad=8)
    ax_box.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6
            else (f"{x / 1e3:.0f}k" if x >= 1e3 else f"{x:.0f}")))

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1,1 — Median CV before vs after IS normalisation per class
    # ════════════════════════════════════════════════════════════════════════
    feats = [f for f in bio_feats
             if f in mean_df.columns and f in raw_bio_meandf.columns]

    classes = sorted({classmap.get(f, "Unknown") for f in feats})
    cv_before, cv_after, cv_delta_labels = [], [], []

    for cls in classes:
        cls_feats = [f for f in feats if classmap.get(f, "Unknown") == cls]
        if not cls_feats:
            continue

        raw_vals  = raw_bio_meandf[cls_feats].replace(0, np.nan).astype(float)
        norm_vals = mean_df[cls_feats].replace(0, np.nan).astype(float)

        raw_cv = (raw_vals.std(axis=0, ddof=1) /
                  raw_vals.mean(axis=0).abs()).replace([np.inf, -np.inf], np.nan)
        norm_cv = (norm_vals.std(axis=0, ddof=1) /
                   norm_vals.mean(axis=0).abs()).replace([np.inf, -np.inf], np.nan)

        mb = float(raw_cv.median()) * 100 if len(raw_cv.dropna()) else np.nan
        ma = float(norm_cv.median()) * 100 if len(norm_cv.dropna()) else np.nan

        if np.isnan(mb) or np.isnan(ma):
            continue

        cv_before.append(mb)
        cv_after.append(ma)
        delta = ma - mb
        sign  = "\u2193" if delta < 0 else "\u2191"
        cv_delta_labels.append((cls[:18], f"{sign}{abs(delta):.1f}\u202f%"))

    if cv_delta_labels:
        x_cls     = np.arange(len(cv_delta_labels))
        w         = 0.38
        cls_names = [t[0] for t in cv_delta_labels]
        deltas    = [t[1] for t in cv_delta_labels]

        ax_cv.bar(x_cls - w / 2, cv_before, w,
                  color="#EF9A9A", label="Before normalisation",
                  edgecolor="white", lw=0.5)
        ax_cv.bar(x_cls + w / 2, cv_after,  w,
                  color="#A5D6A7", label="After normalisation",
                  edgecolor="white", lw=0.5)

        for xi, (bf, af, dl) in enumerate(zip(cv_before, cv_after, deltas)):
            ax_cv.text(xi, max(bf, af) + 1.5, dl,
                       ha="center", va="bottom", fontsize=6.8,
                       color="#333333", fontweight="bold")

        ax_cv.axhline(50, color="#E53935", lw=1.2, ls="--", alpha=0.7,
                      label="Analysis CV threshold (50\u202f%)")
        ax_cv.legend(fontsize=7.5, framealpha=0.9, labelcolor="black",
                     facecolor="white", edgecolor="#CCCCCC", loc="upper right")
        ax_cv.set_xticks(x_cls)
        ax_cv.set_xticklabels(cls_names, rotation=30, ha="right",
                               fontsize=7.5, color="black")
        ax_cv.set_ylabel("Median CV across samples (%)", fontsize=8.5, color="black")
        ax_cv.set_title(
            f"{medium}  |  Median CV Before vs After IS Normalisation\n"
            "\u2193 = improvement   \u2191 = worsened  "
            "(expected for high-CV IS)",
            fontsize=9, fontweight="bold", color="black", pad=8)
    else:
        ax_cv.text(0.5, 0.5, "Insufficient data for CV comparison",
                   transform=ax_cv.transAxes, ha="center", va="center",
                   fontsize=10, color="#888888")

    # ════════════════════════════════════════════════════════════════════════
    # Panels 2,0 and 2,1 — PCA before / after IS normalisation
    # ════════════════════════════════════════════════════════════════════════
    # Step keywords present in sample names
    STEP_META = {
        "medium": ("#E53935", "s",  100, "Blank"),
        "3H":     ("#1565C0", "o",  170, "3h solo"),
        "5H":     ("#E65100", "^",  200, "5h cross-feed"),
        "20H":    ("#2E7D32", "D",  240, "20h return"),
    }

    def _pca_panel(ax, data_df, title_suffix):
        d = data_df[[f for f in bio_feats if f in data_df.columns]].copy()
        valid = d.dropna(how="all")
        if valid.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            return

        imp = SimpleImputer(strategy="median")
        X   = np.log1p(np.maximum(imp.fit_transform(valid.values), 0))
        Xs  = StandardScaler().fit_transform(X)
        nc  = min(3, Xs.shape[0] - 1, Xs.shape[1])
        if nc < 2:
            ax.text(0.5, 0.5, "Too few samples for PCA",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888888")
            return

        pca    = PCA(n_components=nc)
        scores = pca.fit_transform(Xs)
        ev     = pca.explained_variance_ratio_ * 100
        pc_df  = pd.DataFrame(scores[:, :2],
                              index=valid.index,
                              columns=["PC1", "PC2"])

        for step_key, (color, mk, ms, lbl) in STEP_META.items():
            idxs = [i for i in pc_df.index
                    if step_key.lower() in i.lower()]
            if not idxs:
                continue
            sub = pc_df.loc[idxs]
            ax.scatter(sub["PC1"], sub["PC2"],
                       marker=mk, s=ms, c=color,
                       edgecolors="black", linewidths=1.2,
                       zorder=5, label=lbl)

        ax.axhline(0, color="#888888", lw=0.6)
        ax.axvline(0, color="#888888", lw=0.6)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}\u202f%)", color="black", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}\u202f%)", color="black", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.9, labelcolor="black",
                  facecolor="white", edgecolor="#CCCCCC",
                  loc="best", markerscale=0.85)
        ax.set_title(
            f"{medium}  |  PCA {title_suffix}\n"
            "log1p + StandardScaler",
            fontsize=9, fontweight="bold", color="black", pad=7)

    _pca_panel(ax_pca0, raw_bio_meandf, "BEFORE normalisation")
    _pca_panel(ax_pca1, mean_df,        "AFTER normalisation")

    # ════════════════════════════════════════════════════════════════════════
    # Supra-title summary line
    # ════════════════════════════════════════════════════════════════════════
    is_used_labels = "  ·  ".join([is_label(c) for c in stable_is]) or "none"
    if len(is_used_labels) > 110:
        is_used_labels = is_used_labels[:107] + "…"

    excluded_labels = (("  ✗  ".join([is_label(c) for c in unstable_is]))
                       if unstable_is else "none excluded")

    fig.suptitle(
        f"{medium}  —  IS Normalisation Quality Control\n"
        f"IS used ({n_stable}/{n_total}):  {is_used_labels}\n"
        f"Stability filter: CV \u2264 {int(is_stability_thresh * 100)}\u202f%"
        f"   \u2502   Correction: geometric mean"
        f"   \u2502   {n_samples} samples",
        fontsize=9, fontweight="bold", color="black",
        y=0.975, va="top")

    out = Path(out_dir) / f"NormalisationQC_{medium}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")

# =============================================================================
# PGM-BASED FIGURES  —  added after score-based figures (do NOT replace them).
#
# Rationale
# ---------
# The original v2.0 score-based figures (fig_evidence_heatmap, fig_trajectory_
# panels, fig_cross_media_summary, fig_candidate_table) consume the pattern-
# based integer `score` column (0-4) and the 3-category `category` column.
# They do NOT consume the probabilistic PGM fields emitted by scoring.py
# (cf_probability, cf_case, classification, p_prod_A/B/consB).
#
# These four functions visualise the PGM output directly, side-by-side with
# the originals. The user explicitly requested a "CF trajectory plot using
# PGM in addition to the existing scoring" (session of 2026-04-17), so the
# naming convention `fig_pgm_*` mirrors the existing functions and makes the
# two systems visually comparable.
#
# All four figures reuse:
#   CF_CASE_STYLES  — 9 biological-case colour/border/tooltip triples
#   CF_CASE_ORDER   — canonical legend ordering
#   _apply_white_style, _TRAJ_CLINICAL, CHAIN_COLORS, CLASS_COLORS
# =============================================================================

def _sig_label(p):
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def _draw_bracket(ax, x1, x2, y, label, color="#333333", fs=7.5, h=0.02):
    yr = ax.get_ylim()
    span = yr[1] - yr[0]
    tick_h = span * h
    ax.plot([x1, x1, x2, x2],
            [y, y + tick_h, y + tick_h, y],
            lw=0.9, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + tick_h * 1.1, label,
            ha="center", va="bottom", fontsize=fs, color=color,
            clip_on=False)


def _fc_annotations(means, fc_star_thresh=2.0, fc_flag_thresh=1.5):
    blank     = float(means[0])
    fc_labels = ["ctrl"]
    fc_stars  = [""]
    fc_vals   = np.full(7, np.nan)

    for i in range(1, 7):
        if blank > 0:
            fc = float(means[i]) / blank
        else:
            fc = np.nan
        fc_vals[i] = fc
        # Smart FC label: avoid ugly "12000.0×" — use compact notation
        if not np.isnan(fc):
            if fc >= 1000:
                lbl = f"{fc/1000:.1f}k×"
            elif fc >= 100:
                lbl = f"{fc:.0f}×"
            elif fc >= 10:
                lbl = f"{fc:.1f}×"
            else:
                lbl = f"{fc:.1f}×"
        else:
            lbl = "n/a"
        fc_labels.append(lbl)
        if not np.isnan(fc):
            if fc >= fc_star_thresh:
                fc_stars.append("★")
            elif fc >= fc_flag_thresh:
                fc_stars.append("◆")
            else:
                fc_stars.append("")
        else:
            fc_stars.append("")

    return fc_labels, fc_stars, fc_vals


def fig_abundance_bars(
        means=None,
        sds=None,
        mean_df=None,
        sd_df=None,
        cv_df=None,
        cvs=None,
        chain_p_name="CLN-513p",
        chain_e_name="CLN-513e",
        metabolite="metabolite",
        pathway="Pathway",
        condition="Condition",
        output_path="output/abundance_bars.png",
        gap_multiplier=3.0,      # unused now, kept for API compatibility
        outlier_ratio=8.0,       # unused now, kept for API compatibility
        dpi=160,
        cv_warn_thresh=0.50,
        cv_bad_thresh=0.75,
        fc_star_thresh=2.0,
        fc_flag_thresh=1.5,
        show_fc_labels=True,
        show_fc_symbols=True,
        **kwargs,
):
    # ── 1. Resolve inputs ────────────────────────────────────────────────────
    if means is None and mean_df is not None:
        arr = np.asarray(mean_df, dtype=float).flatten()
        if len(arr) == 7:
            means = arr
        else:
            print(f"  [fig_abundance_bars] Skipped: mean_df flattens to "
                  f"{len(arr)} values (expected 7).")
            return None

    if sds is None and sd_df is not None:
        arr = np.asarray(sd_df, dtype=float).flatten()
        sds = arr if len(arr) == 7 else np.zeros(7)

    if means is None:
        print("  [fig_abundance_bars] Skipped: no means provided.")
        return None
    if sds is None:
        sds = np.zeros(7)

    means = np.asarray(means, dtype=float)
    sds   = np.asarray(sds,   dtype=float)

    if len(means) != 7 or len(sds) != 7:
        print(f"  [fig_abundance_bars] Skipped: need 7 values.")
        return None

    # ── 2. Resolve CV array ──────────────────────────────────────────────────
    if cvs is None and cv_df is not None:
        arr = np.asarray(cv_df, dtype=float).flatten()
        cvs = arr if len(arr) == 7 else None
    if cvs is not None:
        cvs = np.asarray(cvs, dtype=float)
        if len(cvs) != 7:
            cvs = None

    # ── 3. Fold-change annotations ───────────────────────────────────────────
    fc_labels, fc_stars, fc_vals = _fc_annotations(
        means, fc_star_thresh=fc_star_thresh, fc_flag_thresh=fc_flag_thresh)

    # ── 4. Colour scheme ─────────────────────────────────────────────────────
    C_BLANK = "#4CAF50"
    C_P     = "#1565C0"
    C_E     = "#E57373"
    colors  = [C_BLANK, C_P, C_E, C_P,  C_E, C_P, C_E]

    x_pos       = np.array([0, 1, 2, 3,  4.8, 5.8, 6.8])
    step_labels = ["Medium", "3h solo", "5h CF", "20h ret",
                   "3h solo", "5h CF",  "20h ret"]
    bar_width   = 0.65

    # ── 5. Simple y-limit computation (no broken axis) ───────────────────────
    bar_tops = np.maximum(means + sds, 0)
    y_top = float(bar_tops.max()) * 1.22
    if y_top == 0:
        y_top = 1.0

    # ── 6. Figure layout (always single axis) ────────────────────────────────
    fig, ax_single = plt.subplots(figsize=(9.5, 5.8), facecolor="white")
    axes_all = [ax_single]

    # ── 7. Axis cosmetics ────────────────────────────────────────────────────
    for ax in axes_all:
        ax.set_facecolor("#F7F8FA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AAAAAA")
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.tick_params(colors="#444", length=3, width=0.8)
        ax.axvline(4.15, color="#BBBBBB", lw=0.8, ls="--", zorder=2)

    # ── 8. Set axis limits ───────────────────────────────────────────────────
    ax_single.set_ylim(0, y_top)

    # ── 9. Draw bars + error bars + CV hatching ──────────────────────────────
    for ax in axes_all:
        for i, (x, m, s, c) in enumerate(zip(x_pos, means, sds, colors)):
            cv_val  = float(cvs[i]) if cvs is not None else 0.0
            cv_warn = cv_val >= cv_warn_thresh
            cv_bad  = cv_val >= cv_bad_thresh

            edge_col = "#CC0000" if cv_bad  else \
                       "#888888" if cv_warn else "white"
            edge_lw  = 1.8      if cv_bad  else \
                       1.0      if cv_warn else 0.5
            hatch    = "////"   if (cv_warn or cv_bad) else None

            ax.bar(x, m, width=bar_width, color=c,
                   edgecolor=edge_col, linewidth=edge_lw,
                   zorder=3, alpha=0.88)

            if hatch:
                ax.bar(x, m, width=bar_width,
                       facecolor="none", edgecolor=edge_col,
                       linewidth=edge_lw, hatch=hatch,
                       zorder=4, alpha=0.55)

            if s > 0:
                ax.errorbar(x, m, yerr=s, fmt="none",
                            elinewidth=1.2, ecolor="#222",
                            capsize=3.5, zorder=5)

    # ── 10. FC labels + symbols above bars ───────────────────────────────────
    label_offset_frac = 0.030

    def _annotate_bar(ax, i, x, m, s, fc_lbl, fc_sym):
        ylim    = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        bar_tip = m + s
        bar_tip = max(bar_tip, ylim[0])
        y_lbl   = bar_tip + y_range * label_offset_frac
        y_sym   = y_lbl   + y_range * 0.060

        if show_fc_labels:
            lbl_color = "#333333" if i == 0 else "#555555"
            ax.text(x, y_lbl, fc_lbl,
                    ha="center", va="bottom", fontsize=7.5,
                    color=lbl_color, zorder=6, clip_on=False)

        if show_fc_symbols and fc_sym:
            sym_color = "#B00000" if fc_sym == "★" else "#7B5EA7"
            ax.text(x, y_sym, fc_sym,
                    ha="center", va="bottom", fontsize=9,
                    color=sym_color, fontweight="bold", zorder=6, clip_on=False)

    for i, (x, m, s) in enumerate(zip(x_pos, means, sds)):
        _annotate_bar(ax_single, i, x, m, s, fc_labels[i], fc_stars[i])

    # ── 12. Y-axis formatters + label ────────────────────────────────────────
    def _make_yfmt(axis_max):
        """Return a tick formatter appropriate for the given axis maximum."""
        def _yfmt(v, _):
            amax = axis_max
            if amax >= 1e6:
                return f"{v/1e6:.1f}M" if v >= 1e5 else (
                       f"{v/1e3:.1f}k" if v >= 1e3 else f"{v:.0f}")
            if amax >= 1e3:
                return f"{v/1e3:.1f}k" if v >= 500 else f"{v:.0f}"
            if amax >= 1000:
                return f"{v:.0f}"
            if amax >= 100:
                return f"{v:.0f}"
            if amax >= 10:
                return f"{v:.1f}"
            return f"{v:.2f}"
        return _yfmt

    ax_single.yaxis.set_major_formatter(plt.FuncFormatter(_make_yfmt(y_top)))
    ax_single.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=4))

    ax_single.set_ylabel("IS-normalised abundance (peak area)",
                         fontsize=9, color="#333", labelpad=5)

    # ── 13. X-axis + chain brackets ──────────────────────────────────────────
    ax_single.set_xticks(x_pos)
    ax_single.set_xticklabels(step_labels, fontsize=8.5, color="#333")
    ax_single.set_xlim(-0.6, 7.5)

    trans = ax_single.get_xaxis_transform()
    for xmin, xmax, label, col in [
            (-0.45, 3.45, chain_p_name, C_P),
            ( 4.35, 7.45, chain_e_name, C_E)]:
        ax_single.annotate("",
            xy=(xmax, -0.12), xytext=(xmin, -0.12),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="#888", lw=1.0))
        ax_single.text((xmin + xmax) / 2, -0.16, label,
                       transform=trans, ha="center", va="top",
                       fontsize=9, fontweight="bold", color=col, clip_on=False)

    # ── 14. Title (metabolite, pathway, condition) ───────────────────────────
    ax_single.set_title(
        f"{metabolite}   [{pathway}]   {condition}",
        fontsize=10, fontweight="bold", color="#222",
        pad=10, loc="left", x=0.06)

    # ── 15. Legend ───────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=C_BLANK, label="Medium blank  (control)"),
        mpatches.Patch(color=C_P,     label=f"P org. / {chain_p_name} return"),
        mpatches.Patch(color=C_E,     label=f"E org. / {chain_e_name} return"),
    ]

    if cvs is not None:
        if np.any(cvs >= cv_warn_thresh):
            patches.append(
                mpatches.Patch(
                    facecolor="#DDDDDD", edgecolor="#888888",
                    hatch="////", linewidth=1.0,
                    label=f"CV ≥ {int(cv_warn_thresh*100)}\u202f%  (unreliable)"))
        if np.any(cvs >= cv_bad_thresh):
            patches.append(
                mpatches.Patch(
                    facecolor="#DDDDDD", edgecolor="#CC0000",
                    hatch="////", linewidth=1.8,
                    label=f"CV ≥ {int(cv_bad_thresh*100)}\u202f%  (very unreliable)"))

    if show_fc_symbols:
        any_star = any(s == "★" for s in fc_stars)
        any_flag = any(s == "◆" for s in fc_stars)
        if any_star:
            patches.append(
                mpatches.Patch(color="none",
                    label=f"★  FC ≥ {fc_star_thresh:.1f}× vs blank  (large effect)"))
        if any_flag:
            patches.append(
                mpatches.Patch(color="none",
                    label=f"◆  FC ≥ {fc_flag_thresh:.1f}× vs blank  (moderate effect)"))

    ncol_leg = min(len(patches), 3)
    fig.legend(handles=patches, loc="lower center", ncol=ncol_leg,
               fontsize=8.0, bbox_to_anchor=(0.53, -0.03),
               framealpha=0.95, facecolor="white", edgecolor="#CCCCCC")

    # ── 16. Footnote ─────────────────────────────────────────────────────────
    if show_fc_symbols:
        fig.text(
            0.5, -0.07,
            "★◆ denote fold-change magnitude vs blank (n\u202f=\u202f2 "
            "biological replicates; not statistical significance tests).",
            ha="center", va="top", fontsize=7.5,
            color="#777777", style="italic",
            transform=fig.transFigure)

    # ── 17. Save ─────────────────────────────────────────────────────────────
    plt.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.22)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)
    return fig

# CHAIN_LABELS already imported above from .constants


def _infer_chain_id_from_strain_v2(strain_id: str) -> str:
    """
    Map a strain ID (Tmp05, Tmp06, Tme13, Tme14)
    to its chain ID (CLN-513p/e, CLN-614p/e) using CHAIN_LABELS.
    Legacy duplicate (kept for back-compat).
    """
    for chain_id, label in CHAIN_LABELS.items():
        if strain_id in label:
            return chain_id
    return strain_id


def fig_abundance_bars_from_strains(
        p_strain,
        e_strain,
        means,
        sds,
        cvs=None,
        metabolite="metabolite",
        pathway="Pathway",
        condition="Condition",
        output_path="output/abundance_bars.png",
        **kwargs,
):
    """
    Use strain IDs (Tmp05, Tmp06, Tme13, Tme14)
    but show chain names (CLN-513p/e, CLN-614p/e) on the plot.
    """
    # Turn Tmp05/Tmp06/Tme13/Tme14 into CLN-513p/e or CLN-614p/e
    p_chain_id = infer_chain_id_from_strain(p_strain)
    e_chain_id = infer_chain_id_from_strain(e_strain)

    return fig_abundance_bars(
        means=means,
        sds=sds,
        cvs=cvs,
        chain_p_name=p_chain_id,
        chain_e_name=e_chain_id,
        metabolite=metabolite,
        pathway=pathway,
        condition=condition,
        output_path=output_path,
        **kwargs,
    )

if __name__ == "__main__":
    import os

    os.makedirs("/home/user/workspace/test_output", exist_ok=True)

    # ── Test case 1: broken axis / sorbitol-like (one extreme outlier bar) ────
    means1 = [50000, 48000, 47000, 52000,  45000, 49000, 280000]
    sds1   = [3000,  4000,  3500,  2500,   5000,  3000,  15000]

    fig1 = fig_abundance_bars(
        means=means1,
        sds=sds1,
        metabolite="Sorbitol",
        pathway="Polyol",
        condition="Test1",
        output_path="/home/user/workspace/test_output/test_broken_axis.png",
    )
    print(f"Test 1 (broken axis): saved={'yes' if fig1 is not None else 'no'}")

    # ── Test case 2: blank near zero, big first bar (lactic acid-like) ────────
    means2 = [200,  2400000,  950000,  170000,  780000,  450000,  50000]
    sds2   = [80,   120000,   60000,   25000,   45000,   80000,   8000]

    fig2 = fig_abundance_bars(
        means=means2,
        sds=sds2,
        metabolite="Lactic acid",
        pathway="Glycolysis",
        condition="Test2",
        output_path="/home/user/workspace/test_output/test_blank_near_zero.png",
    )
    print(f"Test 2 (blank near zero): saved={'yes' if fig2 is not None else 'no'}")

    # ── Test case 3: moderate variation, no outlier (malic acid-like) ─────────
    means3 = [150000, 240000, 710000, 320000,  870000, 530000, 285000]
    sds3   = [12000,  25000,  48000,  31000,   65000,  42000,  22000]

    fig3 = fig_abundance_bars(
        means=means3,
        sds=sds3,
        metabolite="Malic acid",
        pathway="TCA cycle",
        condition="Test3",
        output_path="/home/user/workspace/test_output/test_normal.png",
    )
    print(f"Test 3 (normal, no break): saved={'yes' if fig3 is not None else 'no'}")

    print("\nDone. Check /home/user/workspace/test_output/ for PNGs.")


def fig_pgm_trajectory_panels(scored, mean_df, cv_df, casic_map, class_map,
                               chains, medium, out_dir, ntop=12):
    """
    PGM-driven trajectory panels — same 4-step layout as fig_trajectory_panels,
    but top metabolites are ranked by the highest PGM `cf_probability` across
    clinical chains (rather than the pattern-based `score`), and each chain
    line is coloured/styled by its assigned `cf_case`. The panel title shows
    cf_case + cf_probability instead of integer-score + category.

    Parameters mirror fig_trajectory_panels exactly, so wiring in pipeline.py
    is a one-line addition.
    """
    if scored is None or scored.empty or mean_df.empty:
        return
    if "cf_probability" not in scored.columns or "cf_case" not in scored.columns:
        print(f"  [PGM] scoring DataFrame is missing cf_probability/cf_case; skipping {medium}.")
        return

    clin_scored = scored[scored["chain"].isin(_TRAJ_CLINICAL)]
    if clin_scored.empty:
        print(f"  [PGM] no clinical-chain evidence for {medium}, skipping PGM trajectory.")
        return

    # Rank metabolites by best PGM probability across clinical chains
    top_mets = (clin_scored.groupby("metabolite")["cf_probability"].max()
                .sort_values(ascending=False).head(ntop).index.tolist())
    if not top_mets:
        return

    ncols = min(4, len(top_mets))
    nrows = int(np.ceil(len(top_mets) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.4),
                             facecolor="#FFFFFF")
    fig.subplots_adjust(hspace=0.85, wspace=0.38)
    axes = np.array(axes).flatten()
    step_labs = ["Blank", "3h", "5h-", "20h"]

    for idx, met in enumerate(top_mets):
        ax = axes[idx]
        ax.set_facecolor("#F8F9FA")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3A4A5A")
        ax.axvspan(1.5, 2.5, alpha=0.08, color="#EF5350", zorder=0)

        # shared y-range (log2+1 space)
        all_log_vals = []
        for ch in _TRAJ_CLINICAL:
            for s in chains.get(ch, []):
                if s in mean_df.index and met in mean_df.columns:
                    v = mean_df.loc[s, met]
                    if not (isinstance(v, float) and np.isnan(v)) and v > 0:
                        all_log_vals.append(np.log2(float(v) + 1))
        if not all_log_vals:
            ax.set_visible(False)
            continue
        ymin = min(all_log_vals)
        ymax = max(all_log_vals)
        yrng = ymax - ymin if ymax > ymin else 1.0

        plotted_any = False
        for chain_name in _TRAJ_CLINICAL:
            steps = chains.get(chain_name, [])
            if not steps:
                continue

            sub = scored[(scored["metabolite"] == met) & (scored["chain"] == chain_name)]
            # pick the max-probability row for this chain (if any)
            if not sub.empty:
                best_row = sub.loc[sub["cf_probability"].idxmax()]
                cf_case  = str(best_row.get("cf_case", "UNCLASSIFIED"))
                cf_prob  = float(best_row.get("cf_probability", 0.0))
                classif  = str(best_row.get("classification", "LOW_CONFIDENCE"))
                passed   = cf_prob >= 0.5
            else:
                cf_case, cf_prob, classif, passed = "UNCLASSIFIED", 0.0, "LOW_CONFIDENCE", False

            border, fill, _tooltip = _cf_case_colors(cf_case)
            # Use the cf_case border as the line colour so the figure speaks
            # the PGM language. Keep p/e distinction via line style.
            color  = border if passed else "#B0BEC5"
            ls     = "--" if chain_name.endswith("e") else "-"
            lw     = 2.4 if passed else 1.6
            marker = "o" if passed else "^"
            m_size = 6.5 if passed else 6.0
            alpha_val = 0.95 if passed else 0.45

            leg_label = (f"{chain_name} · {cf_case} · p={cf_prob:.2f}"
                         if passed else f"{chain_name} · no PGM evidence")

            ys_norm, sd_norm, xs_valid = [], [], []
            for xi, s in enumerate(steps):
                if s in mean_df.index and met in mean_df.columns:
                    v = mean_df.loc[s, met]
                    if not (isinstance(v, float) and np.isnan(v)) and v > 0:
                        yn = (np.log2(float(v) + 1) - ymin) / yrng
                        ys_norm.append(yn)
                        cv_val = np.nan
                        if cv_df is not None and s in cv_df.index and met in cv_df.columns:
                            try:
                                cv_val = float(cv_df.loc[s, met])
                            except (TypeError, ValueError):
                                pass
                        is_sentinel = (not np.isnan(cv_val) and abs(cv_val - CV_SENTINEL) < 0.01)
                        is_zero_cv  = (not np.isnan(cv_val) and cv_val == 0.0)
                        if (not np.isnan(cv_val) and not is_sentinel
                                and not is_zero_cv and float(v) > 0):
                            sd_raw = cv_val * float(v)
                            sd_log = sd_raw / (np.log(2) * (float(v) + 1))
                            sd_n   = sd_log / yrng
                        else:
                            sd_n = np.nan
                        sd_norm.append(sd_n)
                    else:
                        ys_norm.append(np.nan)
                        sd_norm.append(np.nan)
                else:
                    ys_norm.append(np.nan)
                    sd_norm.append(np.nan)
                xs_valid.append(xi)

            if sum(1 for y in ys_norm if not np.isnan(y)) < 2:
                continue

            ax.plot(xs_valid, ys_norm,
                    color=color, ls=ls, lw=lw,
                    marker=marker, markersize=m_size,
                    markeredgecolor="white", markeredgewidth=0.8,
                    alpha=alpha_val, zorder=3, label=leg_label)

            # SD error bars (only for reliable replicates)
            y_arr  = np.array(ys_norm, dtype=float)
            sd_arr = np.array(sd_norm, dtype=float)
            mask   = np.isfinite(y_arr) & np.isfinite(sd_arr)
            if mask.sum() >= 2:
                ax.errorbar(xs_valid, ys_norm, yerr=sd_norm,
                            fmt='none', ecolor=color, elinewidth=1.0,
                            capsize=3, capthick=1.0, alpha=0.6, zorder=4)

            plotted_any = True

        if not plotted_any:
            ax.set_visible(False)
            continue

        # Best row across all clinical chains → title
        met_rows   = clin_scored[clin_scored["metabolite"] == met]
        best_idx   = met_rows["cf_probability"].idxmax()
        best_case  = str(met_rows.loc[best_idx, "cf_case"])
        best_prob  = float(met_rows.loc[best_idx, "cf_probability"])
        best_classif = str(met_rows.loc[best_idx, "classification"]) \
            if "classification" in met_rows.columns else ""
        title_col  = _cf_case_colors(best_case)[0]

        ax.set_title(
            f"{casic_map.get(met, met)[:22]}\n"
            f"{best_case}  ·  p={best_prob:.2f}  ·  {best_classif}",
            fontsize=7.5, color=title_col, fontweight="bold", pad=4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(step_labs, fontsize=6.5, color="#333")
        ax.set_ylabel("Normalised abundance\n(0-1)", fontsize=6.0, color="#555")
        ax.set_ylim(-0.05, 1.10)
        ax.tick_params(colors="#333", labelsize=6.5, length=2)
        ax.spines["left"].set_edgecolor(
            CLASS_COLORS.get(class_map.get(met, "Unknown"), "#999999"))
        ax.spines["left"].set_linewidth(2.5)
        ax.legend(fontsize=5.2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.32), ncol=1,
                  framealpha=0.90, labelcolor="black",
                  facecolor="white", edgecolor="#CCCCCC",
                  handlelength=2.2, borderpad=0.5, handletextpad=0.4)

    for i in range(len(top_mets), len(axes)):
        axes[i].set_visible(False)

    # Global legend: 9 CF cases + p/e line style + reliability markers
    case_handles = []
    for label in CF_CASE_ORDER:
        border, fill, tooltip = CF_CASE_STYLES.get(
            label, ("#607D8B", "#ECEFF1", f"Unclassified: {label}")
        )
        case_handles.append(mpatches.Patch(facecolor=fill, edgecolor=border,
                                           linewidth=1.5, label=label))
    case_handles.append(mpatches.Patch(facecolor="#ECEFF1", edgecolor="#607D8B",
                                       linewidth=1.5, label="UNCLASSIFIED"))
    style_handles = [
        Line2D([0], [0], color="#666666", ls="-",  lw=2.0,
               label="Pseudomonas first (p) solid"),
        Line2D([0], [0], color="#666666", ls="--", lw=2.0,
               label="E. coli first (e) dashed"),
        Line2D([0], [0], color="#666666", ls="",
               marker="o", markersize=7,
               markeredgecolor="white", markeredgewidth=0.8,
               label="cf_probability ≥ 0.5"),
        Line2D([0], [0], color="#B0BEC5", ls="",
               marker="^", markersize=7,
               markeredgecolor="white", markeredgewidth=0.8,
               label="cf_probability < 0.5"),
    ]
    fig.legend(handles=case_handles + style_handles,
               loc="lower center", ncol=5,
               fontsize=7.0, framealpha=0.95, labelcolor="black",
               facecolor="white", edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, -0.06))

    plt.suptitle(
        f"{medium} — Clinical Cross-Feeding Trajectories (PGM)\n"
        f"Top {len(top_mets)} metabolites by cf_probability  |  "
        f"Line colour = cf_case  |  Solid = P first  ·  Dashed = E first",
        fontsize=9.5, fontweight="bold", color="black", y=1.02)

    plt.tight_layout()
    out = Path(out_dir) / f"CFTrajectory_PGM_{medium}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_cf_probability_heatmap(scored, casic_map, class_map, medium, out_dir,
                                ntop=30):
    """
    Continuous heatmap of cf_probability across (metabolite × chain).
    Complements fig_evidence_heatmap (which uses the integer score).

    Colourmap: white → orange → deep green (YlGn-like sequential).
    Metabolites ranked by max cf_probability across all chains.
    """
    if scored is None or scored.empty:
        return
    if "cf_probability" not in scored.columns:
        return

    grp = scored.groupby(["metabolite", "chain"])["cf_probability"].max().reset_index()
    if grp.empty:
        return

    top_mets = (grp.groupby("metabolite")["cf_probability"].max()
                .sort_values(ascending=False).head(ntop).index.tolist())
    if not top_mets:
        return

    grp = grp[grp["metabolite"].isin(top_mets)]
    pivot = (grp.pivot(index="metabolite", columns="chain", values="cf_probability")
               .reindex(columns=list(CHAIN_COLORS.keys()))
               .fillna(0.0)
               .loc[top_mets])

    # Best cf_case per metabolite (for the left annotation circle)
    best_case = {}
    for met in top_mets:
        sub = scored[scored["metabolite"] == met]
        if sub.empty:
            best_case[met] = "UNCLASSIFIED"
        else:
            best_case[met] = str(sub.loc[sub["cf_probability"].idxmax(), "cf_case"])

    chains_cols = list(CHAIN_COLORS.keys())
    n_rows = len(top_mets)
    n_cols = len(chains_cols)
    data   = pivot.values.astype(float)

    # Continuous sequential cmap
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    prob_cmap = LinearSegmentedColormap.from_list(
        "cf_prob", ["#FFFFFF", "#FFF3E0", "#FFB74D", "#66BB6A", "#1B5E20"]
    )
    prob_norm = Normalize(vmin=0.0, vmax=1.0)

    CELL    = max(0.45, min(0.8, 12.0 / max(n_rows, n_cols)))
    LEFT_W, RIGHT_W, TOP_H, BOT_H = 3.5, 3.0, 0.6, 1.2
    hm_w = n_cols * CELL
    hm_h = n_rows * CELL
    fw = LEFT_W + hm_w + RIGHT_W
    fh = max(4.0, TOP_H + hm_h + BOT_H)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    ax.set_position([LEFT_W / fw, BOT_H / fh, hm_w / fw, hm_h / fh])

    im = ax.imshow(data, aspect="equal", cmap=prob_cmap, norm=prob_norm,
                   interpolation="nearest")

    # Grid
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="#CCCCCC", lw=0.5, zorder=2)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="#CCCCCC", lw=0.5, zorder=2)

    # Numeric overlay for cells with probability ≥ 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            p = data[i, j]
            if p >= 0.5:
                text_col = "#FFFFFF" if p >= 0.75 else "#1B1B1B"
                ax.text(j, i, f"{p:.2f}",
                        ha="center", va="center",
                        fontsize=5.2, color=text_col, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(chains_cols, rotation=50, ha="right",
                       fontsize=6.5, fontweight="bold")
    for xt, ch in zip(ax.get_xticklabels(), chains_cols):
        xt.set_color(CHAIN_COLORS.get(ch, "#333"))
    ax.tick_params(axis="x", length=0)

    ax.set_yticks([])
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, met in enumerate(top_mets):
        border, _fill, _t = _cf_case_colors(best_case.get(met, "UNCLASSIFIED"))
        class_col = CLASS_COLORS.get(class_map.get(met, "Unknown"), "#333333")
        label = casic_map.get(met, met)[:24]
        ax.text(-0.02, i, "●", transform=trans, fontsize=6,
                color=border, ha="right", va="center", clip_on=False)
        ax.text(-0.04, i, label, transform=trans, fontsize=6,
                color=class_col, ha="right", va="center", clip_on=False,
                path_effects=[pe.withStroke(linewidth=0.3, foreground="#F0F0F0")])

    # Colorbar (probability scale)
    cax = fig.add_axes([(LEFT_W + hm_w + 0.15) / fw,
                        (BOT_H + hm_h * 0.45) / fh,
                        0.015, hm_h * 0.5 / fh])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("cf_probability (PGM)", fontsize=6.5, color="#333")
    cbar.ax.tick_params(labelsize=5.8, colors="#333")

    ax.set_title(
        f"{medium} | PGM Cross-Feeding Probability  ×  Chain",
        color="black", fontsize=8, fontweight="bold", pad=7,
        transform=ax.transAxes,
        x=0.5, y=1 + 0.35 / hm_h
    )

    out = Path(out_dir) / f"CF_ProbabilityHeatmap_{medium}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_cf_case_matrix(scored, casic_map, class_map, medium, out_dir, ntop=30):
    """
    Categorical matrix where every (metabolite × chain) cell is coloured by
    its assigned `cf_case`. Legend on the right lists the 9 biological cases.

    Pairs with fig_cf_probability_heatmap: the probability heatmap answers
    "how strong is the evidence?", this matrix answers "which biological
    case does the PGM assign?".
    """
    if scored is None or scored.empty:
        return
    if "cf_case" not in scored.columns or "cf_probability" not in scored.columns:
        return

    # For each (met, chain), keep the row with the highest cf_probability
    idx = scored.groupby(["metabolite", "chain"])["cf_probability"].idxmax()
    best = scored.loc[idx, ["metabolite", "chain", "cf_case", "cf_probability"]]

    top_mets = (best.groupby("metabolite")["cf_probability"].max()
                .sort_values(ascending=False).head(ntop).index.tolist())
    if not top_mets:
        return

    best = best[best["metabolite"].isin(top_mets)]
    chains_cols = list(CHAIN_COLORS.keys())

    case_pivot = best.pivot(index="metabolite", columns="chain",
                             values="cf_case").reindex(
        index=top_mets, columns=chains_cols)
    prob_pivot = best.pivot(index="metabolite", columns="chain",
                             values="cf_probability").reindex(
        index=top_mets, columns=chains_cols).fillna(0.0)

    n_rows = len(top_mets)
    n_cols = len(chains_cols)

    CELL    = max(0.48, min(0.85, 12.0 / max(n_rows, n_cols)))
    LEFT_W, RIGHT_W, TOP_H, BOT_H = 3.5, 3.6, 0.6, 1.2
    hm_w = n_cols * CELL
    hm_h = n_rows * CELL
    fw = LEFT_W + hm_w + RIGHT_W
    fh = max(4.0, TOP_H + hm_h + BOT_H)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    ax.set_position([LEFT_W / fw, BOT_H / fh, hm_w / fw, hm_h / fh])

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    for i, met in enumerate(top_mets):
        for j, ch in enumerate(chains_cols):
            case = case_pivot.loc[met, ch] if ch in case_pivot.columns else None
            prob = prob_pivot.loc[met, ch] if ch in prob_pivot.columns else 0.0
            if pd.isna(case) or case is None or prob < 0.05:
                # empty cell → faint background
                rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          facecolor="#FFFFFF",
                                          edgecolor="#E0E0E0", linewidth=0.4)
                ax.add_patch(rect)
                continue
            border, fill, _tt = _cf_case_colors(str(case))
            # Alpha scaled by cf_probability so low-confidence cells fade
            alpha_val = 0.3 + 0.7 * min(1.0, float(prob))
            rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      facecolor=fill,
                                      edgecolor=border, linewidth=1.2,
                                      alpha=alpha_val)
            ax.add_patch(rect)
            # Show cf_case label (abbreviated) in cell
            txt = str(case).replace("CF-", "").replace("IND-", "I:") \
                          .replace("DEP-", "D:").replace("SHIFT-", "S:")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=5.2, color=border, fontweight="bold")

    # Grid
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="#CCCCCC", lw=0.5, zorder=1)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="#CCCCCC", lw=0.5, zorder=1)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(chains_cols, rotation=50, ha="right",
                       fontsize=6.5, fontweight="bold")
    for xt, ch in zip(ax.get_xticklabels(), chains_cols):
        xt.set_color(CHAIN_COLORS.get(ch, "#333"))
    ax.tick_params(axis="x", length=0)

    ax.set_yticks([])
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, met in enumerate(top_mets):
        class_col = CLASS_COLORS.get(class_map.get(met, "Unknown"), "#333333")
        label = casic_map.get(met, met)[:24]
        ax.text(-0.02, i, label, transform=trans, fontsize=6,
                color=class_col, ha="right", va="center", clip_on=False,
                path_effects=[pe.withStroke(linewidth=0.3, foreground="#F0F0F0")])

    # Legend: 9 CF cases + tooltip text
    leg_handles = []
    for label in CF_CASE_ORDER:
        border, fill, tooltip = CF_CASE_STYLES[label]
        leg_handles.append(mpatches.Patch(
            facecolor=fill, edgecolor=border, linewidth=1.5,
            label=f"{label}  —  {tooltip[:48]}"
        ))

    leg_ax = fig.add_axes([(LEFT_W + hm_w + 0.25) / fw,
                           BOT_H / fh,
                           (RIGHT_W - 0.35) / fw,
                           hm_h * 0.7 / fh])
    leg_ax.set_axis_off()
    leg_ax.legend(handles=leg_handles,
                  title="PGM cross-feeding cases",
                  title_fontsize=6.5, fontsize=5.4,
                  loc="upper left",
                  facecolor="white", edgecolor="#CCCCCC",
                  framealpha=0.95, labelcolor="black",
                  handlelength=0.9, handleheight=0.7,
                  borderpad=0.4, labelspacing=0.35)

    ax.set_title(
        f"{medium} | PGM Cross-Feeding Case Assignment  ×  Chain\n"
        f"Cell alpha ∝ cf_probability",
        color="black", fontsize=8, fontweight="bold", pad=7,
        transform=ax.transAxes,
        x=0.5, y=1 + 0.35 / hm_h
    )

    out = Path(out_dir) / f"CF_CaseMatrix_{medium}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_pgm_cross_media_summary(all_scored, casic_map, class_map, out_dir,
                                  min_media=1, ntop=40):
    """
    Cross-media PGM summary.
    Row = metabolite, Column = medium (all 5), Cell = best cf_probability
    observed across any chain in that medium. A small square inside each
    cell is coloured by the dominant cf_case.

    Complements fig_cross_media_summary, which uses the integer score.
    """
    if all_scored is None or all_scored.empty:
        return
    if "medium" not in all_scored.columns:
        return
    if "cf_probability" not in all_scored.columns or "cf_case" not in all_scored.columns:
        return

    all_media = MEDIA   # always all 5 columns

    # Best row per (metabolite, medium)
    idx = all_scored.groupby(["metabolite", "medium"])["cf_probability"].idxmax()
    best = all_scored.loc[idx, ["metabolite", "medium", "cf_case", "cf_probability"]]

    met_media = all_scored.groupby("metabolite")["medium"].nunique()
    valid_mets = met_media[met_media >= min_media].index.tolist()
    if not valid_mets:
        print("  No metabolites found for PGM cross-media summary.")
        return

    # Rank metabolites by mean cf_probability across media they appear in
    rank = (best[best["metabolite"].isin(valid_mets)]
              .groupby("metabolite")["cf_probability"].mean()
              .sort_values(ascending=False))
    top_mets = rank.head(ntop).index.tolist()

    prob_pivot = (best[best["metabolite"].isin(top_mets)]
                   .pivot(index="metabolite", columns="medium",
                          values="cf_probability")
                   .reindex(index=top_mets, columns=all_media)
                   .fillna(0.0))
    case_pivot = (best[best["metabolite"].isin(top_mets)]
                   .pivot(index="metabolite", columns="medium",
                          values="cf_case")
                   .reindex(index=top_mets, columns=all_media))

    n_rows = len(top_mets)
    n_cols = len(all_media)

    from matplotlib.colors import LinearSegmentedColormap, Normalize
    prob_cmap = LinearSegmentedColormap.from_list(
        "cf_prob", ["#FFFFFF", "#FFF3E0", "#FFB74D", "#66BB6A", "#1B5E20"]
    )
    prob_norm = Normalize(vmin=0.0, vmax=1.0)

    CELL    = max(0.65, min(1.0, 14.0 / max(n_rows, n_cols)))
    LEFT_W, RIGHT_W, TOP_H, BOT_H = 4.0, 4.2, 0.8, 1.2
    hm_w = n_cols * CELL * 1.3
    hm_h = n_rows * CELL
    fw = LEFT_W + hm_w + RIGHT_W
    fh = max(5.0, TOP_H + hm_h + BOT_H)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    ax.set_position([LEFT_W / fw, BOT_H / fh, hm_w / fw, hm_h / fh])

    im = ax.imshow(prob_pivot.values.astype(float),
                   aspect="auto", cmap=prob_cmap, norm=prob_norm,
                   interpolation="nearest",
                   extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])

    # Grid
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="#CCCCCC", lw=0.5, zorder=2)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="#CCCCCC", lw=0.5, zorder=2)

    # Inner square coloured by cf_case (only if prob ≥ 0.5)
    inner_w = 0.38
    for i, met in enumerate(top_mets):
        for j, med in enumerate(all_media):
            p = float(prob_pivot.loc[met, med]) if med in prob_pivot.columns else 0.0
            c = case_pivot.loc[met, med] if med in case_pivot.columns else None
            if p >= 0.5 and not pd.isna(c) and c is not None:
                border, fill, _t = _cf_case_colors(str(c))
                rect = mpatches.Rectangle(
                    (j - inner_w / 2, i - inner_w / 2),
                    inner_w, inner_w,
                    facecolor=fill, edgecolor=border,
                    linewidth=0.9, zorder=4)
                ax.add_patch(rect)
                # Numeric p value just to the right
                ax.text(j + inner_w / 2 + 0.05, i, f"{p:.2f}",
                        ha="left", va="center", fontsize=5.6,
                        color="#1B1B1B", zorder=5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(all_media, fontsize=8, fontweight="bold", color="#333")
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])

    # Metabolite labels (left, coloured by class)
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, met in enumerate(top_mets):
        class_col = CLASS_COLORS.get(class_map.get(met, "Unknown"), "#333333")
        label = casic_map.get(met, met)[:28]
        ax.text(-0.01, i, label, transform=trans, fontsize=6.5,
                color=class_col, ha="right", va="center", clip_on=False,
                path_effects=[pe.withStroke(linewidth=0.3, foreground="#F0F0F0")])

    # Colorbar
    cax = fig.add_axes([(LEFT_W + hm_w + 0.2) / fw,
                        (BOT_H + hm_h * 0.55) / fh,
                        0.015, hm_h * 0.4 / fh])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("cf_probability (PGM)", fontsize=7, color="#333")
    cbar.ax.tick_params(labelsize=6, colors="#333")

    # Case legend below colorbar
    leg_handles = []
    for label in CF_CASE_ORDER:
        border, fill, _t = CF_CASE_STYLES[label]
        leg_handles.append(mpatches.Patch(
            facecolor=fill, edgecolor=border, linewidth=1.2, label=label))
    leg_ax = fig.add_axes([(LEFT_W + hm_w + 0.2) / fw,
                           BOT_H / fh,
                           (RIGHT_W - 0.35) / fw,
                           hm_h * 0.45 / fh])
    leg_ax.set_axis_off()
    leg_ax.legend(handles=leg_handles,
                  title="cf_case", title_fontsize=7,
                  loc="upper left", fontsize=5.8,
                  facecolor="white", edgecolor="#CCCCCC",
                  framealpha=0.95, labelcolor="black",
                  handlelength=0.9, handleheight=0.7,
                  borderpad=0.4, labelspacing=0.35)

    ax.set_title(
        "Cross-Media PGM Cross-Feeding Landscape\n"
        "Cell background = best cf_probability  |  "
        "Inner square = dominant cf_case  (only shown if p ≥ 0.5)",
        color="black", fontsize=9, fontweight="bold", pad=9,
        transform=ax.transAxes,
        x=0.5, y=1 + 0.35 / hm_h
    )

    out = Path(out_dir) / "CrossMediaSummary_PGM.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out.name}")
