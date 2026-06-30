<img width="1199" height="1312" alt="b945399e-d781-4542-97fa-67589699d0fc" src="https://github.com/user-attachments/assets/b4ada148-7a48-4182-bcec-d861916e70c9" />



# COWTEA-X — Cross-Feeding Analysis Pipeline

COWTEA-X is a Python-based analysis and visualization pipeline for GC-MS metabolomics data, designed as the continuation and logical extension of the original COWTEA pipeline. It operates on MetHub/CASIC-processed, QC-filtered metabolite matrices and focuses on cross-feeding patterns and treatment effects in bacterial biofilm, planktonic cell, and spent medium experiments.

## Project Layout

The COWTEA-X folder is structured as:

```
COWTEA-X/
├── output/                      # Results written by the pipeline (Excel, figures, reports)
├── config.json                  # Configuration file (paths, thresholds, options)
├── HANDOFF.md                   # Notes for handing the pipeline to other users/developers
├── run_pipeline.py              # Main entry point; run this script to execute the pipeline
├── test_smoke.py                # Lightweight smoke test to check environment and imports
└── crossfeeding_pipeline/       # Core Python package with all modules
    ├── __init__.py
    ├── analysis.py
    ├── cli.py
    ├── config.py
    ├── constants.py
    ├── debug_report.py
    ├── export.py
    ├── gui_figure_selector.py
    ├── gui_methods_writer.py
    ├── io.py
    ├── pgm.py
    ├── pipeline.py
    ├── preprocessing.py
    ├── scoring.py
    ├── scoring_enhanced.py
    ├── utils.py
    ├── visualisation.py
    └── viz_abundance.py
```

No additional folders are required beyond these; documentation files (this `README.md`, `LICENSE`, `CITATION.cff`, `requirements.txt`, `.gitignore`) live directly in the top-level `COWTEA-X` directory alongside `run_pipeline.py`.

## Input file location

Place the Excel input file(s) that COWTEA-X should analyze in the **same folder** as `run_pipeline.py` (the top-level `COWTEA-X` directory). The pipeline expects to find input files relative to this folder, and all results (Excel outputs, plots, reports) will be written automatically into the `output/` subfolder.

## Workflow Position

COWTEA-X is intended to be used after:

1. **Metabolites Cleaner** (external R tool by Alex Ilchenko) has cleaned NIST and Excel metabolite names.
2. **MetHub/CASIC** has produced harmonized annotation tables with confidence scores, quality flags, and standardized metabolite IDs.

COWTEA-X then performs:

- IS normalization v2.0 using the geometric mean of stable internal standards selected by CV < 25%.
- Appropriate transformations (e.g. log2(x+1)) and scaling of intensities.
- Univariate statistical testing and fold-change based candidate scoring.
- Visualization of cross-feeding trajectories and heatmaps for selected metabolites.

## Installation

From the `COWTEA-X` folder:

```bash
pip install -r requirements.txt
```

Requirements:

- Python 3.9+
- `pandas`, `numpy` — data handling and numerical operations
- `scipy`, `statsmodels` — statistical tests and normality checks
- `matplotlib`, `seaborn` — CF trajectories, heatmaps, and other plots
- `openpyxl` — Excel I/O

## Basic Usage

1. Prepare CASIC/MetHub-processed metabolite matrices and configuration in `config.json`. Ensure the Excel input file(s) are placed next to `run_pipeline.py` in the top-level folder.
2. Run the pipeline from the command line:

   ```bash
   python run_pipeline.py --config config.json
   ```

   or, if you have CLI options defined in `crossfeeding_pipeline/cli.py`, use:

   ```bash
   python run_pipeline.py --help
   ```

3. Inspect results written to the `output/` folder:

   - Excel tables with statistics and candidate lists
   - Cross-feeding trajectory plots (e.g. PNG/PDF)
   - Heatmaps of selected metabolites
   - Optional debug or methods reports

## Testing

Use the smoke test to verify that dependencies and imports work:

```bash
python test_smoke.py
```

## Citation

If you use COWTEA-X in your research, please cite:

> Sokol, D. (2026). COWTEA-X: Cross-feeding analysis pipeline (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.XXXXXXX

## Author

**Dmytro Sokol (SokolD)**  
PhD Student — Chemistry / Metabolomics  
Umeå University, Sweden

## License

Released under the MIT License (see `LICENSE`).
