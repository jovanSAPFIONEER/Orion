#!/usr/bin/env python3
"""
Batch-run the fixed CAI significance analysis over all run subfolders that contain a `cai/` directory.
Produces: one JSON per run at <run>/cai_signif_fixed.json and a summary CSV at runs/cai_signif_fixed_summary.csv.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SCRIPT = ROOT / "scripts" / "eval_cai_significance_fixed.py"


def main():
    runs = []
    for child in sorted(RUNS.iterdir()):
        if child.is_dir():
            cai_dir = child / "cai"
            if cai_dir.exists() and any(cai_dir.glob("*.json")):
                runs.append(child)
    if not runs:
        print("No runs with cai/ found.")
        return

    summary_rows = []

    for r in runs:
        out_json = r / "cai_signif_fixed.json"
        print(f"Evaluating: {r.name}")
        cmd = [sys.executable, str(SCRIPT), "--cai_dir", str(r / "cai"), "--outfile", str(out_json), "--B", "1000", "--R", "2000", "--by_soa"]
        subprocess.run(cmd, check=True)

        # Read back metrics for summary
        with out_json.open("r", encoding="utf-8") as f:
            d = json.load(f)
        summary_rows.append({
            "run": r.name,
            "n": d.get("n"),
            "AUC": d.get("AUC"),
            "AUC_lo": d.get("AUC_CI95", [None, None])[0],
            "AUC_hi": d.get("AUC_CI95", [None, None])[1],
            "p_value": d.get("p_value"),
            "effect_size": d.get("effect_size"),
            "orientation": d.get("orientation"),
            "significant": d.get("significant"),
        })

    # Write summary CSV
    out_csv = RUNS / "cai_signif_fixed_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"\nWrote summary: {out_csv}")


if __name__ == "__main__":
    main()
