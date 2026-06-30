"""
viz_abundance.py
================
Drop-in replacement for the abundance-bar plot.

Fixes vs the legacy `visualization.fig_abundance_bars`:

* Y-axis: NO broken axis. Linear by default; opt-in symlog when the
  largest bar is ≥ 50× the smallest non-zero bar. Tick formatter
  matches the actual axis maximum so labels never show e.g. "0.1"
  when bars are in the millions.
* Y-axis numbers honour the data magnitude — k / M suffixes with
  consistent precision; integer floor for tiny ranges.
* Title and bar bracket labels always use the chain display names
  (CLN-513p / CLN-513e / CTR-1p / CTR-1e ...) instead of the raw
  strain names (Tmp05, K12, ...).
* Metabolite label prefers CASIC_Result, then NIST_Name, then raw
  column name. Pathway / Metabolite_Class is always shown — never
  blank ("Unknown" fallback).
* Optional fold-change vs blank annotations are supported:
  text labels (e.g. 2.3×) and marker symbols (◆ / ★) above bars.
* Saved at publication DPI (300) with bbox_inches="tight".

The function deliberately ignores any leftover broken-axis kwargs
(`gap_multiplier`, `outlier_ratio`) from the old API.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── chain display names (raw strain → chain label) ──────────────────────────
STRAIN_TO_CHAIN_BASE = {
    "Tmp05": "CLN-513p", "Tme13": "CLN-513e",
    "Tmp06": "CLN-614p", "Tme14": "CLN-614e",
    "PA01": "CTR-1p", "K12": "CTR-1e",
}

# ── colour scheme — sober, publication-friendly ─────────────────────────────
C_BLANK = "#4CAF50"  # green for medium control
C_P = "#1565C0"      # Pseudomonas (blue)
C_E = "#E57373"      # E. coli (red)


def _yfmt(axis_max: float):
    """Return a tick formatter that matches the actual axis maximum."""
    def fmt(v, _pos):
        v_abs = abs(v)
        if axis_max >= 1e6:
            if v_abs >= 1e6:
                return f"{v/1e6:.2g}M"
            if v_abs >= 1e3:
                return f"{v/1e3:.0f}k"
            return f"{v:.0f}"
        if axis_max >= 1e3:
            if v_abs >= 1e3:
                return f"{v/1e3:.1f}k".rstrip("0").rstrip(".")
            return f"{v:.0f}"
        if axis_max >= 10:
            return f"{v:.1f}"
        if axis_max >= 1:
            return f"{v:.2f}"
        if axis_max >= 0.1:
            return f"{v:.3f}"
        if axis_max >= 0.01:
            return f"{v:.4f}"
        # very small values — use scientific notation
        return f"{v:.2e}"
    return ticker.FuncFormatter(fmt)


def _resolve_chain_label(raw: str | None, fallback: str | None = None) -> str:
    """Translate raw strain → chain ID; pass through if already a chain ID."""
    if raw is None:
        return fallback or ""
    raw = str(raw)
    if raw in STRAIN_TO_CHAIN_BASE:
        return STRAIN_TO_CHAIN_BASE[raw]
    return raw


def _metab_display(
    metabolite: str,
    casic_name: str | None,
    metab_class: str | None,
) -> tuple[str, str]:
    """
    Build (title_metab, pathway_label).

    Title prefers the CASIC name, then the raw column name (stripping
    a trailing " 1" / " 2" suffix that the Excel headers carry).
    Pathway falls back to "Unknown class" if empty/None.
    """
    raw_clean = (metabolite or "").strip()
    for suf in (" 1", " 2", " 1", " 2"):
        if raw_clean.endswith(suf):
            raw_clean = raw_clean[: -len(suf)].strip()
            break

    title = (casic_name or "").strip() or raw_clean or "(unknown metabolite)"
    cls = (metab_class or "").strip() or "Unknown class"
    return title, cls


def _fc_annotations(
    means: np.ndarray,
    fc_star_thresh: float = 2.0,
    fc_flag_thresh: float = 1.5,
) -> tuple[list[str], list[str], np.ndarray]:
    """Compute fold-change annotations versus the blank (bar 0)."""
    means = np.asarray(means, dtype=float)
    if means.size != 7:
        return [""] * means.size, [""] * means.size, np.full(means.size, np.nan)

    blank = means[0]
    fc_vals = np.full_like(means, np.nan, dtype=float)
    if blank > 0:
        fc_vals = means / blank

    fc_labels: list[str] = []
    fc_symbols: list[str] = []

    for i, fc in enumerate(fc_vals):
        if i == 0:
            fc_labels.append("1.0×" if blank > 0 else "")
            fc_symbols.append("")
            continue
        if not np.isfinite(fc) or fc <= 0:
            fc_labels.append("")
            fc_symbols.append("")
            continue

        if fc >= 10:
            lbl = f"{fc:.0f}×"
        elif fc >= 1:
            lbl = f"{fc:.1f}×"
        else:
            lbl = f"{fc:.2f}×"
        fc_labels.append(lbl)

        if fc >= fc_star_thresh:
            sym = "★"
        elif fc >= fc_flag_thresh:
            sym = "◆"
        else:
            sym = ""
        fc_symbols.append(sym)

    return fc_labels, fc_symbols, fc_vals


def fig_abundance_bars(
    means: Sequence[float],
    sds: Sequence[float] | None = None,
    cvs: Sequence[float] | None = None,
    *,
    chain_p_name: str | None = None,
    chain_e_name: str | None = None,
    metabolite: str = "metabolite",
    casic_name: str | None = None,
    metab_class: str | None = None,
    condition: str = "",
    output_path: str | Path = "abundance_bars.png",
    cv_warn_thresh: float = 0.50,
    cv_bad_thresh: float = 0.75,
    yscale: str = "auto",  # "linear" | "symlog" | "auto"
    dpi: int = 300,
    fc_star_thresh: float = 2.0,
    fc_flag_thresh: float = 1.5,
    show_fc_labels: bool = True,
    show_fc_symbols: bool = True,
    **kwargs,
) -> Path | None:
    """
    Plot the 7-bar abundance figure for one metabolite, one medium,
    one (P-chain, E-chain) pair.

    The 7 bars are, in order:

    0 Medium blank
    1 3h P solo
    2 5h P→E
    3 20h P-return
    4 3h E solo
    5 5h E→P
    6 20h E-return

    Parameters
    ----------
    means : 7 floats
        Per-sample mean (IS-normalised) for the seven bars.
    sds : 7 floats or None
        Per-sample SD (error bars). 0 / NaN → no error bar.
    cvs : 7 floats or None
        Per-sample CV. Bars with CV > thresh are hatched.
    chain_p_name, chain_e_name : str
        Chain ID for the P-first and E-first chains (e.g. 'CLN-513p').
        If a raw strain ID is passed, it is translated automatically.
    metabolite, casic_name, metab_class : str
        Labels for the title. CASIC takes precedence; class falls back
        to 'Unknown class'.
    condition : str
        Medium name (AUM, DEX, ...).
    output_path : path-like
        Output PNG.
    yscale : 'linear' | 'symlog' | 'auto'
        'auto' uses symlog when max/min(non-zero) ratio ≥ 50.
    fc_star_thresh : float
        Threshold for ★ fold-change symbol versus blank.
    fc_flag_thresh : float
        Threshold for ◆ fold-change symbol versus blank.
    show_fc_labels, show_fc_symbols : bool
        Whether to show fold-change text labels and symbols.

    Returns
    -------
    Path of saved figure, or None on failure.
    """
    means = np.asarray(means, dtype=float)
    if means.size != 7:
        print(f" [fig_abundance_bars] Skipped {metabolite}: expected 7 means, got {means.size}")
        return None

    sds = np.asarray(sds, dtype=float) if sds is not None else np.zeros(7)
    if sds.size != 7:
        sds = np.zeros(7)
    sds = np.nan_to_num(sds, nan=0.0)

    cvs = np.asarray(cvs, dtype=float) if cvs is not None else None
    if cvs is not None and cvs.size != 7:
        cvs = None

    chain_p_label = _resolve_chain_label(chain_p_name, "P-chain")
    chain_e_label = _resolve_chain_label(chain_e_name, "E-chain")
    title_metab, pathway_label = _metab_display(metabolite, casic_name, metab_class)

    finite_means = means[np.isfinite(means)]
    if finite_means.size == 0 or np.nanmax(finite_means) <= 0:
        print(f" [fig_abundance_bars] Skipped {metabolite}: all means are zero / NaN")
        return None

    fc_labels, fc_symbols, fc_vals = _fc_annotations(
        means,
        fc_star_thresh=fc_star_thresh,
        fc_flag_thresh=fc_flag_thresh,
    )

    bar_tops = np.maximum(means + sds, 0)
    bar_top_max = float(np.nanmax(bar_tops))
    nonzero_min = float(np.nanmin(finite_means[finite_means > 0])) if (finite_means > 0).any() else 0.0
    dyn_range = (bar_top_max / nonzero_min) if nonzero_min > 0 else 1.0

    use_symlog = (yscale == "symlog") or (yscale == "auto" and dyn_range >= 50.0)

    x_pos = np.array([0, 1, 2, 3, 4.8, 5.8, 6.8])
    step_labels = ["Medium", "3h solo", "5h CF", "20h ret", "3h solo", "5h CF", "20h ret"]
    colors = [C_BLANK, C_P, C_E, C_P, C_E, C_P, C_E]
    bar_width = 0.65

    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.tick_params(colors="#333", length=3, width=0.8)
    ax.axvline(4.15, color="#BBBBBB", lw=0.8, ls="--", zorder=2)

    if use_symlog:
        linthresh = max(nonzero_min * 0.5, 1e-6) if nonzero_min > 0 else 1e-6
        ax.set_yscale("symlog", linthresh=linthresh, base=10)
        # Use actual bar top + 22% headroom, same as linear branch
        upper = bar_top_max * 1.22 if bar_top_max > 0 else 1.0
        ax.set_ylim(0, upper)
        ax.yaxis.set_major_formatter(_yfmt(upper))
    else:
        y_top = bar_top_max * 1.22 if bar_top_max > 0 else 1.0
        ax.set_ylim(0, y_top)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=4))
        ax.yaxis.set_major_formatter(_yfmt(y_top))

    for i, (x, m, s, c) in enumerate(zip(x_pos, means, sds, colors)):
        cv_val = float(cvs[i]) if (cvs is not None and np.isfinite(cvs[i])) else 0.0
        cv_warn = cv_val >= cv_warn_thresh
        cv_bad = cv_val >= cv_bad_thresh

        edge = "#B00000" if cv_bad else ("#777777" if cv_warn else "white")
        elw = 1.8 if cv_bad else (1.0 if cv_warn else 0.5)
        hatch = "////" if (cv_warn or cv_bad) else None

        ax.bar(x, m, width=bar_width, color=c, edgecolor=edge,
               linewidth=elw, zorder=3, alpha=0.9)
        if hatch:
            ax.bar(x, m, width=bar_width, facecolor="none",
                   edgecolor=edge, linewidth=elw, hatch=hatch,
                   zorder=4, alpha=0.5)
        if s > 0 and np.isfinite(s):
            ax.errorbar(x, m, yerr=s, fmt="none", elinewidth=1.2,
                        ecolor="#222", capsize=3.5, zorder=5)

    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    label_offset_frac = 0.03

    for i, (x, m, s) in enumerate(zip(x_pos, means, sds)):
        if not np.isfinite(m):
            continue

        bar_tip = m + (s if np.isfinite(s) else 0.0)
        bar_tip = max(bar_tip, ylim[0])
        y_fc = bar_tip + span * label_offset_frac
        y_sym = y_fc + span * 0.06

        if show_fc_labels and i < len(fc_labels) and fc_labels[i]:
            lbl_color = "#333333" if i == 0 else "#555555"
            ax.text(x, y_fc, fc_labels[i], ha="center", va="bottom",
                    fontsize=7.5, color=lbl_color, zorder=6, clip_on=False)

        if show_fc_symbols and i < len(fc_symbols):
            sym = fc_symbols[i]
            if sym:
                sym_color = "#B00000" if sym == "★" else "#7B5EA7"
                ax.text(x, y_sym, sym, ha="center", va="bottom",
                        fontsize=9, color=sym_color, fontweight="bold",
                        zorder=6, clip_on=False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(step_labels, fontsize=8.5, color="#333")
    ax.set_xlim(-0.6, 7.5)

    trans = ax.get_xaxis_transform()
    for xmin, xmax, label, col in [
        (-0.45, 3.45, chain_p_label, C_P),
        (4.35, 7.45, chain_e_label, C_E),
    ]:
        ax.annotate(
            "",
            xy=(xmax, -0.12), xytext=(xmin, -0.12),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="#888", lw=1.0),
        )
        ax.text((xmin + xmax) / 2, -0.16, label,
                transform=trans, ha="center", va="top",
                fontsize=10, fontweight="bold", color=col, clip_on=False)

    title_parts = [title_metab, f"[{pathway_label}]"]
    if condition:
        title_parts.append(condition)
    ax.set_title(" ".join(title_parts), fontsize=11, fontweight="bold", color="#222", pad=10, loc="left")

    ax.set_ylabel("IS-normalised abundance (peak area)", fontsize=10, color="#333", labelpad=5)

    patches = [
        mpatches.Patch(color=C_BLANK, label="Medium blank (control)"),
        mpatches.Patch(color=C_P, label=f"Pseudomonas ({chain_p_label})"),
        mpatches.Patch(color=C_E, label=f"E. coli ({chain_e_label})"),
    ]

    if cvs is not None:
        if np.any(cvs >= cv_warn_thresh):
            patches.append(mpatches.Patch(
                facecolor="#DDD", edgecolor="#777", hatch="////",
                label=f"CV ≥ {int(cv_warn_thresh*100)}% (less reliable)"))
        if np.any(cvs >= cv_bad_thresh):
            patches.append(mpatches.Patch(
                facecolor="#DDD", edgecolor="#B00000", hatch="////",
                label=f"CV ≥ {int(cv_bad_thresh*100)}% (unreliable)"))

    if show_fc_symbols:
        any_star = any(s == "★" for s in fc_symbols)
        any_flag = any(s == "◆" for s in fc_symbols)
        if any_star:
            patches.append(mpatches.Patch(
                facecolor="none", edgecolor="none",
                label=f"★  FC ≥ {fc_star_thresh:.1f}× vs blank (large effect)"))
        if any_flag:
            patches.append(mpatches.Patch(
                facecolor="none", edgecolor="none",
                label=f"◆  FC ≥ {fc_flag_thresh:.1f}× vs blank (moderate effect)"))

    fig.legend(handles=patches, loc="lower center",
               ncol=min(len(patches), 3), fontsize=8.5,
               bbox_to_anchor=(0.53, -0.02),
               framealpha=0.95, facecolor="white", edgecolor="#CCC")

    if show_fc_symbols:
        fig.text(0.5, -0.08,
                 "★/◆ denote fold-change magnitude vs blank (not formal statistical significance).",
                 ha="center", va="top", fontsize=7.0,
                 color="#777777", style="italic",
                 transform=fig.transFigure)

    if use_symlog:
        fig.text(0.99, 0.005, "y-axis: symlog scale",
                 ha="right", va="bottom", fontsize=7,
                 color="#888", style="italic")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.24)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
