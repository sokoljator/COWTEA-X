"""
gui_methods_writer.py
=====================
GUI helper that turns a COWTEA-X run's `debug_report.json` +
`All_ScoredCandidates*.csv` into a concise manuscript-style Methods
paragraph. The user can preview and edit before saving.

Output:
  * `Methods.docx` if `python-docx` is available
  * `Methods.txt`  fallback

Running:
    python -m crossfeeding_pipeline.gui_methods_writer
"""

from __future__ import annotations
from pathlib import Path
import json
import sys

import pandas as pd

try:
    from docx import Document            # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False


# ─────────────────────────────────────────────────────────────────────────────
# Methods drafting
# ─────────────────────────────────────────────────────────────────────────────
def draft_methods_text(folder: Path) -> str:
    """
    Read debug_report.json + the enhanced-scoring CSV from `folder`
    and return a default Methods-section draft as plain text. Lines
    are kept short so they wrap well in a manuscript.
    """
    folder = Path(folder)
    debug_path = folder / "debug_report.json"
    debug = {}
    if debug_path.exists():
        debug = json.loads(debug_path.read_text(encoding="utf-8"))

    summary = debug.get("summary_stats", {}) or {}
    config  = debug.get("config_snapshot", {}) or {}
    schema  = debug.get("input_schema", {}) or {}
    media   = list(schema.keys()) or config.get("media") or []
    scorer  = config.get("scorer", "classic")

    scored_enh = folder / "All_ScoredCandidates_enhanced.csv"
    scored_clas = folder / "All_ScoredCandidates.csv"
    n_strong = summary.get("n_strong_evidence", "n/a")
    n_moderate = summary.get("n_moderate_evidence", "n/a")
    n_metab = summary.get("n_unique_metabolites", "n/a")

    media_str = ", ".join(media) if media else "five defined media (AUM, DEX, GLY, HisGly, SUC)"
    cohort_str = ("Pseudomonas putida (CLN-513p, CLN-614p) and Escherichia coli "
                  "(CLN-513e, CLN-614e) co-culture chains plus the laboratory "
                  "control pair (CTR-1p, CTR-1e)")

    lines = []
    lines.append("Metabolomics data analysis — COWTEA-X pipeline")
    lines.append("")
    lines.append(
        "Sequential co-culture GC-MS metabolomics data were analysed with the "
        f"in-house Python pipeline COWTEA-X. Five media ({media_str}) were "
        "processed in parallel. For each medium, peak areas were normalised "
        "using stable internal standards (CV ≤ 25%); per-sample means, "
        "standard deviations and coefficients of variation were computed "
        "across biological replicates."
    )
    lines.append("")
    lines.append(
        "Sequential culturing chains were defined as Pseudomonas-first "
        "(3h producer → 5h E. coli on producer-spent medium → 20h "
        "Pseudomonas return) and the mirror E. coli-first arrangement. "
        "Chains were labelled with the display IDs CLN-513p / CLN-513e for "
        "the Tmp05/Tme13 pair, CLN-614p / CLN-614e for Tmp06/Tme14, and "
        "CTR-1p / CTR-1e for the PA01/K12 laboratory controls."
    )
    lines.append("")
    if scorer == "enhanced" or scored_enh.exists():
        lines.append(
            "Cross-feeding evidence was scored with a multi-component scheme "
            "rather than an all-or-nothing rule. For every (medium, chain, "
            "metabolite) triple, eight bounded evidence components were "
            "computed and summed (maximum 12): strict paired-chain pattern "
            "(producer step + consumer step + mirror non-rise + mirror "
            "return rise, up to 4 points); single-chain 3h→5h or 5h→20h "
            "pattern (up to 2 points); producer-leaning behaviour "
            "(1 point); consumer-leaning behaviour (1 point); genus-level "
            "partial support from the sister CLN strain (1 point); media "
            "consistency across ≥ 2 media (1 point); replicate confidence "
            "with all involved CVs ≤ 0.30 (1 point); and a late-time "
            "20-h metabolic shift (1 point). A conflict penalty of "
            "2 points was subtracted when the mirror chain showed a "
            "matching producer signal that would compromise specificity. "
            "Scores were thresholded as STRONG (≥ 8), MODERATE (≥ 5) and "
            "WEAK (≥ 3), and each row was annotated with a stable pattern "
            "label (PE_FULL, PE_3H5H, PE_5H20H, GENUS_HALF, PROD_ONLY, "
            "CONS_ONLY or NONE)."
        )
        lines.append("")
    else:
        lines.append(
            "Cross-feeding candidates were identified by step-wise "
            "log2-fold-change thresholds within each chain, with a strict "
            "paired producer-then-consumer rule and a mirror-chain "
            "complementarity check."
        )
        lines.append("")
    lines.append(
        f"Across the {len(media) or 'five'} processed media, "
        f"{n_metab} unique metabolites were scored; {n_strong} reached the "
        f"STRONG evidence band and {n_moderate} the MODERATE band. All "
        "outputs — per-medium IS-normalised tables, scored candidate "
        "tables, evidence heatmaps, abundance-bar figures and a JSON "
        "debug report — are written into a single timestamped output "
        "directory for traceability."
    )
    lines.append("")
    lines.append(
        "Figures were rendered with matplotlib at 300 DPI; abundance "
        "bar plots use a continuous y-axis (linear by default, symlog "
        "when the bar dynamic range exceeded 50-fold) with CV-based "
        "reliability hatching."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
def save_methods(text: str, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if HAS_DOCX:
        out = folder / "Methods.docx"
        doc = Document()
        for para in text.split("\n\n"):
            doc.add_paragraph(para)
        doc.save(out)
        return out
    out = folder / "Methods.txt"
    out.write_text(text, encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("COWTEA-X Methods Writer")
    root.geometry("820x620")

    state = {"folder": None}

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="Results folder:").pack(side="left")
    folder_var = tk.StringVar()
    ttk.Entry(top, textvariable=folder_var, width=70).pack(side="left", padx=4)

    def pick():
        f = filedialog.askdirectory(title="Pick a COWTEA-X output folder")
        if not f:
            return
        folder_var.set(f)
        state["folder"] = Path(f)
        try:
            text = draft_methods_text(state["folder"])
        except Exception as e:
            messagebox.showerror("Draft failed", str(e))
            return
        editor.delete("1.0", "end")
        editor.insert("1.0", text)
        status_var.set(f"Drafted from {f}")

    ttk.Button(top, text="Browse...", command=pick).pack(side="left")
    ttk.Button(top, text="Re-draft",
               command=lambda: pick() if not state["folder"]
               else editor.replace("1.0", "end",
                                   draft_methods_text(state["folder"]))
               ).pack(side="left", padx=4)

    editor = tk.Text(root, wrap="word")
    editor.pack(fill="both", expand=True, padx=8, pady=4)

    bottom = ttk.Frame(root, padding=8)
    bottom.pack(fill="x")

    def save():
        if state["folder"] is None:
            messagebox.showinfo("Pick folder first", "Choose a folder, then save.")
            return
        text = editor.get("1.0", "end").rstrip()
        out = save_methods(text, state["folder"])
        messagebox.showinfo(
            "Saved", f"Wrote: {out}\n"
                     f"({'DOCX' if HAS_DOCX else 'TXT — python-docx not installed'})"
        )

    ttk.Button(bottom, text="Save", command=save).pack(side="right")

    status_var = tk.StringVar(value=("python-docx available." if HAS_DOCX
                                     else "python-docx NOT installed — will save .txt"))
    ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w")\
        .pack(side="bottom", fill="x")

    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
