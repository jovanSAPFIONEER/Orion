#!/usr/bin/env python3
"""
Plot AUC vs. broadcast_gain with 95% CIs, marking significance and orientation.
Reads runs/cai_signif_fixed_bcast_sweep.csv and optional BH CSV, writes figures/cai_auc_vs_gain.[png|pdf].
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"
CSV_IN = RUNS / "cai_signif_fixed_bcast_sweep.csv"
CSV_BH = RUNS / "cai_signif_fixed_bcast_sweep_bh.csv"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main():
    if not CSV_IN.exists():
        raise SystemExit(f"Missing input: {CSV_IN}")
    rows = load_rows(CSV_IN)
    bh_rows = load_rows(CSV_BH) if CSV_BH.exists() else []
    # Index BH by (sweep, dir)
    bh_map = {(r["sweep"], r["dir"]): r for r in bh_rows}

    # Prefer the _ext sweep for plotting if present; else use base
    groups = {}
    for r in rows:
        groups.setdefault(r["sweep"], []).append(r)
    sweep = "causal_bcast_sweep_ext" if "causal_bcast_sweep_ext" in groups else "causal_bcast_sweep"
    data = groups[sweep]

    # Sort by gain
    data.sort(key=lambda d: float(d["gain"]))

    gains = [float(d["gain"]) for d in data]
    aucs = [float(d["AUC"]) for d in data]
    lo = [float(d["AUC_lo"]) for d in data]
    hi = [float(d["AUC_hi"]) for d in data]
    yerr = [[a - l for a, l in zip(aucs, lo)], [h - a for a, h in zip(aucs, hi)]]

    sig = [d.get("significant", "False") == "True" for d in data]
    bh_sig = []
    for d in data:
        key = (d["sweep"], d["dir"])
        bh_sig.append(bh_map.get(key, {}).get("significant_bh", "False") == "True")
    ori = [d.get("orientation", "normal") for d in data]

    FIGS.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.0), dpi=150)
    for i, (g, a, yl, yh, s, bs, o) in enumerate(zip(gains, aucs, yerr[0], yerr[1], sig, bh_sig, ori)):
        marker = "^" if o == "inverted" else "o"
        face = "#d62728" if bs else ("#1f77b4" if s else "white")
        edge = "#d62728" if bs else ("#1f77b4" if s else "#7f7f7f")
        plt.errorbar([g], [a], yerr=[[yl], [yh]], fmt=marker, ms=7, mfc=face, mec=edge, ecolor="#bbbbbb", elinewidth=1, capsize=3)
    plt.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    plt.xlabel("broadcast_gain")
    plt.ylabel("AUC (two-sided, 95% CI)")
    plt.title(f"CAI discrimination vs broadcast_gain ({sweep})")
    plt.ylim(0.35, 0.70)
    plt.xlim(min(gains) - 0.05, max(gains) + 0.05)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"cai_auc_vs_gain.{ext}"
        plt.savefig(out)
    plt.close()
    print("Wrote:", FIGS / "cai_auc_vs_gain.png")


if __name__ == "__main__":
    main()
