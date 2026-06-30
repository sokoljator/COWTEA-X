"""
constants.py
============
Single source of truth for every shared constant used across the pipeline.

Extracted verbatim from crossfeeding_pipeline_v2.0.py so that io.py,
preprocessing.py, analysis.py, scoring.py, export.py and visualization.py
can all import the same values instead of each duplicating or (worse)
disagreeing about them.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
MEDIA = ["AUM", "DEX", "GLY", "HisGly", "SUC"]

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY / FC THRESHOLDS
#   Re-tuned in v2.0 (see memory of 2026-04-08):
#   FC_THRESH = 1.0          → 2-fold minimum for primary signal
#   FC_THRESH_RETURN = 0.75  → 1.68-fold for the (weaker) 20h return step
#   CV_THRESH = 0.50         → relaxed from 0.25 because the experiment has
#                              only 2 technical reps (no bio reps).
# ─────────────────────────────────────────────────────────────────────────────
CV_THRESH        = 0.50
FC_THRESH        = 1.0
FC_THRESH_RETURN = 0.75

# Split-level CV bands used by mirror plot & ranked-bar hatching
CV_THRESH_LIGHT  = 0.25     # analytically clean
CV_THRESH_MEDIUM = 0.50     # borderline (hatched)

# Sentinel value injected by preprocessing when a sample has a single
# replicate. sqrt(2) was chosen because it cannot collide with a real CV
# (real CVs are <= 1 in practice) while still being distinguishable from
# NaN. Detection uses abs(cv - CV_SENTINEL) < 0.001.
CV_SENTINEL = np.sqrt(2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP LABELS / MARKERS  — order matches [blank, 3h, 5h, 20h] in build_chains
# ─────────────────────────────────────────────────────────────────────────────
STEP_MARKERS = ["s", "o", "^", "D"]
STEP_LABELS  = ["Blank", "3h solo", "5h cross-feed", "20h return"]

# ─────────────────────────────────────────────────────────────────────────────
# METABOLITE-CLASS COLOURS (used by every heatmap / barplot / PCA)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Amino Acids":          "#1565C0",
    "Carbohydrates":        "#2E7D32",
    "Sugar Alcohols":       "#00838F",
    "Sugar Acids":          "#006064",
    "Organic Acids":        "#E65100",
    "TCA Cycle":            "#B71C1C",
    "Lipids & Fatty Acids": "#6A1B9A",
    "Polyamines":           "#AD1457",
    "Nitrogen Metabolism":  "#827717",
    "Inorganic":            "#4E342E",
    "Internal Standard":    "#0277BD",
    "Unknown":              "#424242",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHAIN COLOURS  — Okabe-Ito palette, max contrast on white
# Pairs share the same colour; p/e differ via line style in figures.
# ─────────────────────────────────────────────────────────────────────────────
CHAIN_COLORS = {
    "CLN-513p": "#D55E00",   # vermillion  (clinical pair 1: Tmp05 × Tme13)
    "CLN-513e": "#D55E00",
    "CLN-614p": "#0072B2",   # deep blue   (clinical pair 2: Tmp06 × Tme14)
    "CLN-614e": "#0072B2",
    "CTR-1p":   "#009E73",   # bluish green (PA01 × K12)
    "CTR-1e":   "#009E73",
}

CHAIN_LABELS = {
    "CLN-513p": "CLN-513p  Tmp05\u2192Tme13\u2192Tmp05  (Pseudomonas first)",
    "CLN-513e": "CLN-513e  Tme13\u2192Tmp05\u2192Tme13  (E. coli first)",
    "CLN-614p": "CLN-614p  Tmp06\u2192Tme14\u2192Tmp06  (Pseudomonas first)",
    "CLN-614e": "CLN-614e  Tme14\u2192Tmp06\u2192Tme14  (E. coli first)",
    "CTR-1p":   "CTR-1p   PA01\u2192K12\u2192PA01     (Pseudomonas first)",
    "CTR-1e":   "CTR-1e   K12\u2192PA01\u2192K12      (E. coli first)",
}

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-FEEDING CASE STYLES  (label → border, fill, tooltip/legend text)
# ─────────────────────────────────────────────────────────────────────────────
CF_CASE_STYLES = {
    "CF-P\u2194E":   ("#1B5E20", "#C8E6C9",
                      "Bidirectional CF — P\u2194E exchange confirmed in both chains"),
    "CF-P\u2192E":   ("#2E7D32", "#DCEDC8",
                      "P constitutively produces X (M-R1+); E consumes X (M-R2\u2212)"),
    "CF-E\u2192P":   ("#0D47A1", "#BBDEFB",
                      "E constitutively produces X (Mi-R1+); P consumes X (Mi-R2\u2212)"),
    "IND-E-byP":     ("#E65100", "#FFE0B2",
                      "E produces X only when induced by P's spent medium (M-R1 grey, M-R2+)"),
    "IND-P-byE":     ("#BF360C", "#FFCCBC",
                      "P produces X only when induced by E's spent medium (Mi-R1 grey, Mi-R2+)"),
    "DEP-byE":       ("#006064", "#B2EBF2",
                      "E depletes X only when entering P's spent medium (M-R1 grey, M-R2\u2212)"),
    "DEP-byP":       ("#00695C", "#B2DFDB",
                      "P depletes X only when entering E's spent medium (Mi-R1 grey, Mi-R2\u2212)"),
    "SHIFT-P-20h":   ("#6A1B9A", "#E1BEE7",
                      "P's role switches at 20h return (producer\u2192consumer or vice versa)"),
    "SHIFT-E-20h":   ("#4527A0", "#D1C4E9",
                      "E's role switches at 20h return (mirror chain)"),
}

CF_CASE_ORDER = [
    "CF-P\u2194E", "CF-P\u2192E", "CF-E\u2192P",
    "IND-E-byP",  "IND-P-byE",
    "DEP-byE",    "DEP-byP",
    "SHIFT-P-20h", "SHIFT-E-20h",
]

# ─────────────────────────────────────────────────────────────────────────────
# MIRROR PAIRS  (each chain → its reciprocal-order mirror)
# ─────────────────────────────────────────────────────────────────────────────
MIRROR_PAIRS = {
    "CTR-1p":   "CTR-1e",   "CTR-1e":   "CTR-1p",
    "CLN-513p": "CLN-513e", "CLN-513e": "CLN-513p",
    "CLN-614p": "CLN-614e", "CLN-614e": "CLN-614p",
}

MIRROR_PAIR_NAMES = [
    ("CLN-513p", "CLN-513e"),
    ("CLN-614p", "CLN-614e"),
    ("CTR-1p",   "CTR-1e"),
]

# Clinical chains shown in the primary trajectory figure
TRAJ_CLINICAL  = ["CLN-513p", "CLN-513e", "CLN-614p", "CLN-614e"]
_TRAJ_CLINICAL = TRAJ_CLINICAL    # legacy alias used inside visualization.py

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY MAPPING  (score band → category → display label / explanation)
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLORS = {
    "TRUE_CROSSFEED":  "#1B5E20",
    "ORDER_DEPENDENT": "#E65100",
    "OPPORTUNISTIC":   "#B71C1C",
}

CAT_MARKER_COLORS = {
    "TRUE_CROSSFEED":  "#2E7D32",
    "ORDER_DEPENDENT": "#BF360C",
    "OPPORTUNISTIC":   "#880E4F",
}

SCORE_CELL_COLORS = {
    0: "#FFFFFF",
    1: "#F8BBD0",
    2: "#FFF176",
    3: "#FFB74D",
    4: "#66BB6A",
}

CATEGORY_DISPLAY = {
    "TRUE_CROSSFEED":  "True Cross-Feed",
    "ORDER_DEPENDENT": "Putative Cross-Feed",
    "OPPORTUNISTIC":   "Unspecific change",
}

CATEGORY_EXPLAIN = {
    "TRUE_CROSSFEED": (
        "True Cross-Feed (score=4)\n"
        "Producer+consumer pattern confirmed by mirror chain.\n"
        "Example: Strain P makes metabolite X alone;\n"
        "X drops when E arrives; mirror chain shows X does\n"
        "NOT rise when B goes solo."
    ),
    "ORDER_DEPENDENT": (
        "Putative Cross-Feed (score 2-3)\n"
        "Producer+consumer pattern detected, but NOT fully\n"
        "confirmed by mirror-chain control. The exchange\n"
        "may depend on which strain arrives first."
    ),
    "OPPORTUNISTIC": (
        "Unspecific change (score <2)\n"
        "FC changes do NOT fit a producer+consumer pattern.\n"
        "Likely reflects individual metabolism, not genuine\n"
        "cross-feeding."
    ),
}

# PCA / trajectory node sizes (one per step index)
NODE_SIZES = {
    0: 170,   # blank
    1: 200,   # 3h
    2: 200,   # 5h
    3: 240,   # 20h
}
