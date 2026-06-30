"""
tests_smoke.py
==============
Minimal regression tests for the COWTEA-X package. Run with:

    python tests_smoke.py

The script returns non-zero on any failure so it can be wired into CI.
It deliberately avoids pytest to keep the dependency surface small.
"""

from __future__ import annotations
from pathlib import Path
import sys, traceback, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _ok(name):  print(f"  PASS  {name}")
def _fail(name, e):
    print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    traceback.print_exc()


def test_imports():
    import crossfeeding_pipeline as cp
    assert callable(cp.run_pipeline)
    assert callable(cp.score_crossfeeding_enhanced)
    assert callable(cp.fig_abundance_bars)


def test_config_resolution_workspace():
    from crossfeeding_pipeline import load_config
    cfg = load_config(ROOT / "config.json")
    assert Path(cfg["xlsx_path"]).exists(), cfg["xlsx_path"]
    assert "output" in cfg["out_dir"]


def test_config_resolution_windows_paths():
    """Windows-style paths in config-2.json should fuzzy-resolve."""
    from crossfeeding_pipeline import load_config
    cfg = load_config(ROOT / "config-2.json")
    assert Path(cfg["xlsx_path"]).exists()
    # Out_dir must not still contain C:
    assert "C:" not in cfg["out_dir"]


def test_load_sheet():
    from crossfeeding_pipeline import load_sheet
    raw_df, casic, cls, bio, is_ = load_sheet(
        str(ROOT / "Consolidated_CASIC_5media_withClass_final-14.xlsx"),
        "AUM", media_prefixes=["AUM","DEX","GLY","HisGly","SUC"])
    assert raw_df.shape[0] > 10
    assert len(bio) > 0
    assert len(is_) > 0
    assert any(casic.get(b) for b in bio), "no CASIC_Result mapping found"
    assert any(cls.get(b) for b in bio), "no Metabolite_Class mapping found"


def test_chain_labels():
    from crossfeeding_pipeline import build_chains
    chains = build_chains("AUM")
    expected = {"CLN-513p","CLN-513e","CLN-614p","CLN-614e","CTR-1p","CTR-1e"}
    assert set(chains) == expected
    # blank+3+5+20 sample keys
    for c, steps in chains.items():
        assert len(steps) == 4, f"{c}: {steps}"
        assert steps[0] == "AUM_medium"


def test_abundance_plot_basic(tmpdir: Path):
    from crossfeeding_pipeline import fig_abundance_bars
    out = tmpdir / "abund.png"
    means = [1e3, 1e6, 5e5, 1e4, 8e5, 6e5, 1e3]
    sds   = [50, 5e4, 1e4, 500, 4e4, 3e4, 100]
    cvs   = [0.05, 0.10, 0.40, 0.55, 0.08, 0.20, 0.10]
    p = fig_abundance_bars(
        means, sds, cvs,
        chain_p_name="Tmp05", chain_e_name="Tme13",
        metabolite="lactic acid  1", casic_name="lactic acid",
        metab_class="Carboxylic acids", condition="AUM",
        output_path=str(out), yscale="auto",
    )
    assert p is not None and Path(p).exists()
    # Should have triggered symlog (dynamic range ~1000)
    # (we can't assert internal state, but file should be > 30kB)
    assert Path(p).stat().st_size > 30_000


def test_abundance_plot_skips_empty(tmpdir: Path):
    from crossfeeding_pipeline import fig_abundance_bars
    out = tmpdir / "empty.png"
    means = [0,0,0,0,0,0,0]
    p = fig_abundance_bars(means, output_path=str(out),
                           chain_p_name="Tmp05", chain_e_name="Tme13",
                           metabolite="x", casic_name="x", metab_class="y",
                           condition="AUM")
    assert p is None, "Plot should be skipped for all-zero data"


def test_chain_label_translation(tmpdir: Path):
    """Plot must use chain labels (CLN-513p) not raw strain names (Tmp05)."""
    from crossfeeding_pipeline.viz_abundance import _resolve_chain_label
    assert _resolve_chain_label("Tmp05") == "CLN-513p"
    assert _resolve_chain_label("Tme13") == "CLN-513e"
    assert _resolve_chain_label("Tmp06") == "CLN-614p"
    assert _resolve_chain_label("K12")   == "CTR-1e"
    # Pass-through for already-chain IDs
    assert _resolve_chain_label("CLN-513p") == "CLN-513p"


def test_scoring_enhanced_emits_components():
    from crossfeeding_pipeline import build_chains, score_crossfeeding_enhanced
    # Synthetic minimal frame
    samples = build_chains("AUM")
    rows = []
    for chain, ss in samples.items():
        for s in ss: rows.append(s)
    rows = sorted(set(rows))
    cols = ["fakeA"]
    mean = pd.DataFrame(1.0, index=rows, columns=cols)
    # Set a clean PE_FULL pattern on CLN-513p / fakeA
    mean.loc["AUM_3H_Tmp05",            "fakeA"] = 10.0   # producer
    mean.loc["AUM_5H_Tmp05_Tme13",      "fakeA"] = 0.5    # consumer
    mean.loc["AUM_20H_Tmp05_Tme13_Tmp05","fakeA"] = 8.0   # mirror_return ↑
    # mirror chain has no solo rise
    mean.loc["AUM_3H_Tme13",            "fakeA"] = 1.1
    cv = pd.DataFrame(0.10, index=rows, columns=cols)
    out = score_crossfeeding_enhanced(mean, cv, samples, "AUM")
    sub = out[(out["chain"]=="CLN-513p") & (out["metabolite"]=="fakeA")]
    assert not sub.empty
    r = sub.iloc[0]
    assert r["pattern_label"] in ("PE_FULL", "PE_3H5H")
    assert r["score"] >= 4
    assert r["e_paired_strict"] >= 2
    # Required evidence columns present
    for col in ["e_paired_strict","e_single_chain","e_producer","e_consumer",
                "e_genus_partial","e_replicate_conf","e_late_shift",
                "p_conflict","pattern_label","confidence","score"]:
        assert col in out.columns, f"missing column {col}"


def test_debug_report_roundtrip(tmpdir: Path):
    from crossfeeding_pipeline import DebugReport
    d = DebugReport()
    d.record_config({"xlsx_path":"x","out_dir":"y","media":["AUM"]})
    d.record_input_schema("AUM", n_features_bio=34, n_features_is=7,
                          n_samples_unique=19, n_rows_raw=38,
                          missing_chain_keys=[])
    d.record_mapping("AUM", casic_map={"A":"acetate"}, class_map={"A":"acid"},
                     bio_feats=["A"])
    d.record_warning("test warning")
    d.record_scoring("AUM", pd.DataFrame({
        "metabolite":["a"], "score":[8], "pattern_label":["PE_FULL"],
        "confidence":["STRONG_EVIDENCE"],
    }), scorer="enhanced")
    json_path = d.write_report(tmpdir)
    assert json_path.exists()
    md = (tmpdir / "debug_report.md").read_text()
    assert "Summary statistics" in md
    assert "AUM" in md


def test_gui_modules_importable():
    """GUIs must import without launching Tk."""
    import crossfeeding_pipeline.gui_figure_selector as g1
    import crossfeeding_pipeline.gui_methods_writer as g2
    assert callable(g1.main)
    assert callable(g2.main)
    # Methods drafter must work standalone
    text = g2.draft_methods_text(Path("output/CF_results_260519_111950"))
    assert "COWTEA-X" in text
    assert "CLN-513" in text


def main() -> int:
    import tempfile
    tests = [
        ("imports",                      test_imports,                   False),
        ("config_resolution_workspace",  test_config_resolution_workspace, False),
        ("config_resolution_windows",    test_config_resolution_windows_paths, False),
        ("load_sheet",                   test_load_sheet,                False),
        ("chain_labels",                 test_chain_labels,              False),
        ("abundance_plot_basic",         test_abundance_plot_basic,      True),
        ("abundance_plot_skips_empty",   test_abundance_plot_skips_empty, True),
        ("chain_label_translation",      test_chain_label_translation,   True),
        ("scoring_enhanced_components",  test_scoring_enhanced_emits_components, False),
        ("debug_report_roundtrip",       test_debug_report_roundtrip,    True),
        ("gui_modules_importable",       test_gui_modules_importable,    False),
    ]
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for name, fn, needs_tmp in tests:
            try:
                if needs_tmp:
                    fn(td_path)
                else:
                    fn()
                _ok(name)
            except Exception as e:
                _fail(name, e)
                fails += 1
    print()
    print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}  ({len(tests)} tests)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
