"""
debug_report.py
===============
Run-time debugging / reporting helper.

Each pipeline run instantiates one `DebugReport`. The pipeline calls
`debug.record_*` methods at every stage; at the end of the run
`debug.write_report(out_dir)` emits two files in the timestamped
output folder:

    debug_report.json     — machine-readable
    debug_report.md       — human-readable summary

Recorded categories
-------------------
* run_meta              host, user, python version, timestamps, duration
* config_snapshot       resolved config (xlsx_path, out_dir, media, scorer)
* input_schema          per-sheet row / column counts, IS feature counts,
                        per-medium sample-key coverage
* mapping_checks        casic / class map completeness; missing translations
* warnings              free-form warnings collected anywhere
* scoring_summary       per-medium row counts, score-band breakdown,
                        pattern-label histogram
* plot_warnings         per-figure skip reasons (e.g. "all means zero")
* exceptions            per-medium captured tracebacks (does not raise)
* summary_stats         aggregate counts: n_metabolites, n_strong, etc.

This module has no required external dependencies beyond the standard
library + pandas (which is already a hard pipeline dependency).
"""

from __future__ import annotations
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class DebugReport:
    """In-memory debug record. Call methods during the run, then
    `write_report(out_dir)` at the end."""

    def __init__(self):
        self.start_ts = time.time()
        self.run_meta = {
            "host":      platform.node(),
            "platform":  platform.platform(),
            "python":    sys.version.split()[0],
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.config_snapshot: dict[str, Any] = {}
        self.input_schema:    dict[str, Any] = {}
        self.mapping_checks:  dict[str, Any] = {}
        self.warnings:        list[str] = []
        self.scoring_summary: dict[str, Any] = {}
        self.plot_warnings:   list[str] = []
        self.exceptions:      list[dict] = []
        self.summary_stats:   dict[str, Any] = {}

    # ── recording API ───────────────────────────────────────────────
    def record_config(self, config: dict) -> None:
        self.config_snapshot = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in config.items()
        }

    def record_input_schema(self, medium: str, *,
                            n_features_bio: int, n_features_is: int,
                            n_samples_unique: int, n_rows_raw: int,
                            missing_chain_keys: list[str]) -> None:
        self.input_schema[medium] = {
            "n_features_bio":    n_features_bio,
            "n_features_is":     n_features_is,
            "n_samples_unique":  n_samples_unique,
            "n_rows_raw":        n_rows_raw,
            "missing_chain_keys": missing_chain_keys,
        }

    def record_mapping(self, medium: str, *,
                       casic_map: dict, class_map: dict,
                       bio_feats: list[str]) -> None:
        n_total = len(bio_feats)
        n_casic = sum(1 for f in bio_feats if casic_map.get(f, "").strip())
        n_class = sum(1 for f in bio_feats
                      if class_map.get(f, "").strip()
                      and class_map.get(f) != "Unknown")
        self.mapping_checks[medium] = {
            "n_bio_features":      n_total,
            "n_with_casic_name":   n_casic,
            "n_with_class":        n_class,
            "frac_with_casic":     round(n_casic / max(n_total, 1), 3),
            "frac_with_class":     round(n_class / max(n_total, 1), 3),
        }

    def record_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def record_plot_warning(self, msg: str) -> None:
        self.plot_warnings.append(msg)

    def record_scoring(self, medium: str, scored: pd.DataFrame,
                       *, scorer: str = "classic") -> None:
        if scored is None or scored.empty:
            self.scoring_summary[medium] = {
                "scorer": scorer, "n_rows": 0,
                "n_unique_metabolites": 0,
            }
            return
        entry: dict[str, Any] = {
            "scorer": scorer,
            "n_rows": int(len(scored)),
            "n_unique_metabolites": int(scored["metabolite"].nunique()),
        }
        if "score" in scored.columns:
            entry["score_distribution"] = (
                scored["score"].value_counts().sort_index().to_dict()
            )
        if "pattern_label" in scored.columns:
            entry["pattern_label_counts"] = (
                scored["pattern_label"].value_counts().to_dict()
            )
        if "category" in scored.columns:
            entry["category_counts"] = (
                scored["category"].value_counts().to_dict()
            )
        if "confidence" in scored.columns:
            entry["confidence_counts"] = (
                scored["confidence"].value_counts().to_dict()
            )
        self.scoring_summary[medium] = entry

    def record_exception(self, where: str, exc: BaseException) -> None:
        self.exceptions.append({
            "where":   where,
            "type":    type(exc).__name__,
            "message": str(exc),
            "trace":   traceback.format_exc(),
        })

    # ── finalize ────────────────────────────────────────────────────
    def _build_summary_stats(self) -> None:
        n_strong = n_moderate = n_total = 0
        for med, s in self.scoring_summary.items():
            n_total += s.get("n_unique_metabolites", 0)
            conf = s.get("confidence_counts", {}) or {}
            n_strong   += conf.get("STRONG_EVIDENCE", 0)
            n_moderate += conf.get("MODERATE_EVIDENCE", 0)
        self.summary_stats = {
            "n_media_processed":       len(self.scoring_summary),
            "n_unique_metabolites":    n_total,
            "n_strong_evidence":       n_strong,
            "n_moderate_evidence":     n_moderate,
            "n_warnings":              len(self.warnings),
            "n_plot_warnings":         len(self.plot_warnings),
            "n_exceptions":            len(self.exceptions),
            "duration_seconds":        round(time.time() - self.start_ts, 1),
        }

    def write_report(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._build_summary_stats()

        payload = {
            "run_meta":         self.run_meta | {"duration_seconds":
                                                 self.summary_stats["duration_seconds"]},
            "config_snapshot":  self.config_snapshot,
            "input_schema":     self.input_schema,
            "mapping_checks":   self.mapping_checks,
            "warnings":         self.warnings,
            "scoring_summary":  self.scoring_summary,
            "plot_warnings":    self.plot_warnings,
            "exceptions":       self.exceptions,
            "summary_stats":    self.summary_stats,
        }

        json_path = out / "debug_report.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        md_path = out / "debug_report.md"
        md_path.write_text(self._render_md(payload), encoding="utf-8")

        return json_path

    # ── markdown rendering ──────────────────────────────────────────
    def _render_md(self, p: dict) -> str:
        lines = ["# COWTEA-X Run Report", ""]
        lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
        lines.append("")
        lines.append("## Summary statistics")
        for k, v in p["summary_stats"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append("## Run metadata")
        for k, v in p["run_meta"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append("## Config snapshot")
        for k, v in p["config_snapshot"].items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
        lines.append("## Input schema (per medium)")
        for med, s in p["input_schema"].items():
            lines.append(f"### {med}")
            for k, v in s.items():
                lines.append(f"- {k}: `{v}`")
            lines.append("")
        lines.append("## Mapping completeness")
        for med, m in p["mapping_checks"].items():
            lines.append(f"- **{med}**: {m['n_with_casic_name']}/"
                         f"{m['n_bio_features']} CASIC names "
                         f"({m['frac_with_casic']*100:.0f}%); "
                         f"{m['n_with_class']}/{m['n_bio_features']} "
                         f"class labels "
                         f"({m['frac_with_class']*100:.0f}%)")
        lines.append("")
        lines.append("## Scoring summary")
        for med, s in p["scoring_summary"].items():
            lines.append(f"### {med}")
            for k, v in s.items():
                lines.append(f"- {k}: `{v}`")
            lines.append("")
        if p["warnings"]:
            lines.append("## Warnings")
            for w in p["warnings"]:
                lines.append(f"- {w}")
            lines.append("")
        if p["plot_warnings"]:
            lines.append("## Plot warnings")
            for w in p["plot_warnings"]:
                lines.append(f"- {w}")
            lines.append("")
        if p["exceptions"]:
            lines.append("## Captured exceptions")
            for e in p["exceptions"]:
                lines.append(f"### {e['where']} — {e['type']}")
                lines.append(f"> {e['message']}")
                lines.append("```")
                lines.append(e["trace"])
                lines.append("```")
        return "\n".join(lines)
