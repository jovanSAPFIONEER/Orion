#!/usr/bin/env python3
"""
Batch-run fixed CAI significance over sweep folders (e.g., causal_bcast_sweep_ext/gain_*).
Outputs one JSON per gain dir and a summary CSV with gain and overall metrics.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SCRIPT = ROOT / "scripts" / "eval_cai_significance_fixed.py"

GAIN_DIR_RE = re.compile(r"^gain_([0-9]+)p([0-9]+)$")


def discover_gain_dirs() -> list[tuple[str, Path, float]]:
    results: list[tuple[str, Path, float]] = []
    for sweep_name in ("causal_bcast_sweep_ext", "causal_bcast_sweep"):
        sweep_dir = RUNS / sweep_name
        if not sweep_dir.exists():
            continue
        for child in sorted(sweep_dir.iterdir()):
            if not child.is_dir():
                continue
            m = GAIN_DIR_RE.match(child.name)
            if not m:
                continue
            whole, frac = m.group(1), m.group(2)
            try:
                gain = float(f"{int(whole)}.{int(frac)}")
            except Exception:
                continue
            cai_dir = child / "cai"
            if cai_dir.exists() and any(cai_dir.glob("*.json")):
                results.append((sweep_name, child, gain))
    return results


def main():
    targets = discover_gain_dirs()
    if not targets:
        print("No sweep gain_* directories with cai/ found.")
        return

    summary_rows = []

    for sweep_name, path, gain in targets:
        out_json = path / "cai_signif_fixed.json"
        print(f"Evaluating: {sweep_name}/{path.name} (gain={gain})")
        cmd = [sys.executable, str(SCRIPT), "--cai_dir", str(path / "cai"), "--outfile", str(out_json), "--B", "800", "--R", "1600"]
        subprocess.run(cmd, check=True)
        with out_json.open("r", encoding="utf-8") as f:
            d = json.load(f)
        summary_rows.append({
            "sweep": sweep_name,
            "dir": path.name,
            "gain": gain,
            "n": d.get("n"),
            "AUC": d.get("AUC"),
            "AUC_lo": d.get("AUC_CI95", [None, None])[0],
            "AUC_hi": d.get("AUC_CI95", [None, None])[1],
            "p_value": d.get("p_value"),
            "effect_size": d.get("effect_size"),
            "orientation": d.get("orientation"),
            "significant": d.get("significant"),
        })

    # Sort by sweep then gain
    summary_rows.sort(key=lambda r: (r["sweep"], r["gain"]))

    out_csv = RUNS / "cai_signif_fixed_bcast_sweep.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"Wrote sweep summary: {out_csv}")


if __name__ == "__main__":
    main()
