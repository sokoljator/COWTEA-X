# COWTEA-X — Handoff

## How to run

```bash
# Full pipeline (5 media, classic + enhanced scoring, all figures, debug report)
python run_pipeline.py config.json

# Smoke tests
python tests_smoke.py

# GUIs (need a display)
python -m crossfeeding_pipeline.gui_figure_selector
python -m crossfeeding_pipeline.gui_methods_writer
```

Outputs land in `output/CF_results_<YYMMDD_HHMMSS>/`.

## Files created

| File | Role |
| --- | --- |
| `crossfeeding_pipeline/__init__.py` | Public API of the cleaned package |
| `crossfeeding_pipeline/config.py` | Cross-platform config loader (workspace/Windows paths, fuzzy file resolution) |
| `crossfeeding_pipeline/pipeline.py` | Re-orchestrated pipeline (uses new plotter, enhanced scorer, debug report) |
| `crossfeeding_pipeline/viz_abundance.py` | **New** abundance-bars plot (fixed y-axis, CASIC + class labels, chain names) |
| `crossfeeding_pipeline/scoring_enhanced.py` | **New** multi-evidence scorer (8 components + media consistency post-pass) |
| `crossfeeding_pipeline/debug_report.py` | **New** run reporting → `debug_report.json` + `.md` |
| `crossfeeding_pipeline/gui_figure_selector.py` | **New** Tkinter GUI: load results, filter, render figures |
| `crossfeeding_pipeline/gui_methods_writer.py` | **New** Tkinter GUI: draft + edit + save Methods (.docx / .txt fallback) |
| `crossfeeding_pipeline/visualization.py` | Patched copy of the original (removed two crashing `from constants import` lines) |
| `run_pipeline.py` | Workspace entry-point that defaults to `config-2.json` or `config.json` |
| `config.json` | New clean cross-platform config (`xlsx_path` relative, `scorer: enhanced`) |
| `tests_smoke.py` | 11 regression tests (imports, config, parsing, plotting, scoring, GUI importability, debug report) |
| `HANDOFF.md` | This document |

## Files modified

| File | Change |
| --- | --- |
| `crossfeeding_pipeline/visualization.py` | Replaced two `from constants import CHAIN_LABELS` lines (would crash on import outside source folder) with a comment. Second duplicate `infer_chain_id_from_strain` renamed `_infer_chain_id_from_strain_v2`. The rest of the file is verbatim. |
| `crossfeeding_pipeline/pipeline.py` | Re-written from `pipeline-10.py`. Uses the new plotter, scorer, debug report. |
| `run_pipeline.py` | Replaced the original entry point with one that puts the workspace on `sys.path` and falls back across `config-2.json` / `config.json`. |

## Files untouched (no changes needed)

The package copies of these files are byte-identical to the originals; only their
names were normalized (e.g. `analysis-3.py` → `crossfeeding_pipeline/analysis.py`):

- `analysis-3.py` → `analysis.py`
- `cli-4.py` → `cli.py`
- `config-5.py` → no longer used (replaced by `config.py` because the original hard-coded Windows paths)
- `constants-6.py` → `constants.py`
- `export-7.py` → `export.py`
- `io-8.py` → `io.py`
- `pgm-9.py` → `pgm.py`
- `preprocessing-11.py` → `preprocessing.py`
- `scoring-12.py` → `scoring.py` (still called in parallel with the new enhanced scorer)
- `utils-13.py` → `utils.py`
- `init__-2.py` → not used (we wrote a fresh `__init__.py`)

The original suffixed `*-N.py` files are preserved in `/home/user/workspace/` and were not deleted.

## What changed by task

### 1. Module inspection and redundancy

* Two crashing imports in `visualization.py` removed (`from constants import CHAIN_LABELS` outside any try/except).
* Duplicate definition of `infer_chain_id_from_strain` deduped.
* Per-medium runner now wraps each figure call in try/except so a single broken figure no longer aborts the whole medium.

### 2. Unified config & cross-platform paths

* `crossfeeding_pipeline/config.py` resolves `xlsx_path` and `out_dir` against
  (in order) the config-file's directory, the current working directory,
  `/home/user/workspace`, and `$COWTEAX_ROOT`.
* Fuzzy basename matching handles cases like `…final.xlsx` vs `…final-14.xlsx`.
* Windows-absolute output paths (`C:/Users/…/output/CF_results`) are rewritten to the trailing two path segments under the first writable root.
* Verified: both `config.json` (new clean form) and `config-2.json` (original Windows paths) resolve correctly to the workspace dataset (regression-tested).

### 3. Fixed abundance bar plots (`viz_abundance.py`)

* No broken axis. Linear by default; `symlog` only when max/min ratio ≥ 50.
* Tick formatter adapts to the actual data magnitude (k / M suffixes).
* Metabolite label is `CASIC_Result` → `NIST_Name` → cleaned column header.
* `Metabolite_Class` is shown in brackets in the title; falls back to "Unknown class".
* Chain bracket annotations always show chain IDs (CLN-513p / CLN-513e / CTR-1p / CTR-1e); raw strain names (Tmp05 / K12 / …) translate automatically.
* CV-based hatching with two thresholds (warn ≥ 0.50, bad ≥ 0.75).
* DPI 300, `bbox_inches='tight'`.
* See sample rendering in `output/CF_results_260519_111950/AbundanceBars_AUM_CLN-513p_vs_CLN-513e_glycine.png`.

### 4. Multi-evidence scoring (`scoring_enhanced.py`)

Eight bounded sub-scores (sum capped at 12):

| Column | Range | Meaning |
| --- | --- | --- |
| `e_paired_strict` | 0–4 | Classic 4-bar producer + consumer + mirror non-rise + mirror return rise |
| `e_single_chain` | 0–2 | 3h↑→5h↓ OR 5h↑→20h↓ — no mirror needed |
| `e_producer` | 0–1 | At least one step looks like a producer |
| `e_consumer` | 0–1 | At least one step looks like a consumer |
| `e_genus_partial` | 0–1 | Sister CLN strain (513↔614, CTR alone) supports same direction |
| `e_replicate_conf` | 0–1 | All involved CVs ≤ 0.30 |
| `e_late_shift` | 0–1 | 20h flips direction (nutrient scarcity / metabolic shift) |
| `e_media_consistency` | 0–1 | Same pattern fires in ≥ 2 media |
| `p_conflict` | 0..−2 | Mirror chain also produces — specificity penalty |

Final fields:
* `score` (0–12), `confidence` ∈ {STRONG_EVIDENCE ≥ 8, MODERATE_EVIDENCE ≥ 5, WEAK_EVIDENCE ≥ 3, INSUFFICIENT}.
* `pattern_label` ∈ {PE_FULL, PE_3H5H, PE_5H20H, GENUS_HALF, PROD_ONLY, CONS_ONLY, NONE} — labels asymmetric and partial cases distinctly.
* Classic scorer (`scoring.py :: score_crossfeeding`) is still called in parallel — both outputs are exported, so nothing the user was relying on disappeared.
* Run on the supplied workbook: 723 rows, 14 STRONG / 25 MODERATE / 42 WEAK / 642 INSUFFICIENT. Top STRONG hits: Aminomalonic acid in CLN-614p (GLY + AUM, pattern `PE_FULL`, score 10), Succinic acid in CLN-614e (DEX, score 10), Uric acid in CLN-614e (AUM, score 9), Pyroglutamic acid in CLN-614p (GLY + DEX, score 8).

### 5. Debug / reporting module (`debug_report.py`)

Each run writes `debug_report.json` + `debug_report.md` into the output folder with:

* `run_meta` (host, platform, python, timestamps, duration)
* `config_snapshot` (resolved paths)
* `input_schema` per medium (n_features_bio / _is, n_samples, missing chain keys)
* `mapping_checks` (CASIC / class coverage %)
* `warnings`, `plot_warnings`
* `scoring_summary` per medium (row counts, score / pattern / confidence histograms)
* `exceptions` (per-medium captured tracebacks; the pipeline keeps going)
* `summary_stats` (n_metabolites, n_strong, n_moderate, duration, etc.)

### 6 & 7. GUIs

* `gui_figure_selector.py` — load a results folder, filter on min score / medium / chain / pattern label, pick a row, choose plot method (`abundance_bars` re-rendered at user-chosen DPI, or open an existing PNG). Tk is only imported inside `main()`; the module imports cleanly without a display.
* `gui_methods_writer.py` — drafts a Methods paragraph from `debug_report.json` + scoring CSVs, allows editing, saves `.docx` (python-docx) or `.txt`. The drafter function (`draft_methods_text`) is callable headless and is exercised by the tests.

### 8. Validation

End-to-end run on the supplied workbook completed in 76 seconds with **0 exceptions and 0 plot warnings**:

```
n_media_processed:    5  (AUM / DEX / GLY / HisGly / SUC)
n_unique_metabolites: 274 scored across all chains
n_strong_evidence:    28 (across enhanced scoring rows)
n_moderate_evidence:  49
duration_seconds:     76.2
files written:        181
abundance bar plots:  93 (publication DPI, chain labels, class labels)
```

11 / 11 smoke tests pass:

```
PASS  imports
PASS  config_resolution_workspace
PASS  config_resolution_windows         (Windows path in config-2.json fuzzy-resolves)
PASS  load_sheet                        (CASIC + class maps populated)
PASS  chain_labels                      (build_chains uses CLN-/CTR- IDs)
PASS  abundance_plot_basic              (symlog auto-triggered, file > 30 kB)
PASS  abundance_plot_skips_empty        (all-zero data returns None safely)
PASS  chain_label_translation           (Tmp05→CLN-513p, K12→CTR-1e, etc.)
PASS  scoring_enhanced_components       (all 8 evidence columns present, PE_FULL detected)
PASS  debug_report_roundtrip            (JSON + markdown written)
PASS  gui_modules_importable            (no Tk display required at import time)
```

## Limitations / open items

* The `scoring_enhanced` thresholds (`FC_THRESH`, `STRICT_CV`, `W_*` weights) are module-local. They should probably be promoted to the config file when the biology is finalised — currently editing them requires opening the .py.
* The GUI figure-selector relies on `IS_normalized_<medium>.xlsx` exports to re-render abundance bars. If `export_excel` is ever changed to not emit these, the GUI's "abundance_bars" method will fall back to opening the pre-saved PNG.
* The figure-selector GUI's plot menu currently exposes only `abundance_bars` and `open_existing_PNG`. The other figure types from `visualization.py` (PCA, FC heatmap, mirror comparison, etc.) are still produced automatically by the pipeline but are not yet wired into the GUI's "Render" button — they can be added by replicating the abundance-bars helper pattern.
* `gui_methods_writer` writes a `.docx` only when `python-docx` is installed; otherwise it silently falls back to `.txt`. The "tone" of the Methods paragraph is a single template — the user is expected to edit it inline in the text box before saving.
* `Single replicate` cells in the workbook surface a `CV_SENTINEL` value (≈ 0.95); the enhanced scorer detects this and disqualifies the row from the replicate-confidence bonus, but currently does not down-weight other components — that is a deliberate conservative choice.
* Did not modify the original suffixed source files (`*-N.py`); the package versions live alongside them.
