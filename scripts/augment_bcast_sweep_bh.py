#!/usr/bin/env python3
"""
Augment broadcast_gain sweep summary with BH/FDR-adjusted p-values.
Reads runs/cai_signif_fixed_bcast_sweep.csv and writes runs/cai_signif_fixed_bcast_sweep_bh.csv.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "runs" / "cai_signif_fixed_bcast_sweep.csv"
OUT_CSV = ROOT / "runs" / "cai_signif_fixed_bcast_sweep_bh.csv"


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg step-up FDR adjustment with proper monotonicity.
    Returns adjusted p-values in the original order.
    """
    m = len(pvals)
    pairs = sorted([(p, i) for i, p in enumerate(pvals)], key=lambda t: t[0])
    adj_raw = [0.0] * m
    # Compute raw adjusted p-values p * m / rank for sorted p
    for rank, (p, idx_sorted) in enumerate(pairs, start=1):
        adj_raw[rank - 1] = min(p * m / rank, 1.0)
    # Enforce monotonicity from the end (non-decreasing in rank)
    for k in range(m - 2, -1, -1):
        adj_raw[k] = min(adj_raw[k], adj_raw[k + 1])
    # Map back to original indices
    adj = [0.0] * m
    for (p, idx_sorted), val in zip(pairs, adj_raw):
        adj[idx_sorted] = val
    return adj


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input CSV: {IN_CSV}")
    rows = []
    with IN_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    # Compute BH within each sweep group
    by_sweep = {}
    for row in rows:
        by_sweep.setdefault(row["sweep"], []).append(row)
    out_rows = []
    for sweep, group in by_sweep.items():
        pvals = [float(g["p_value"]) for g in group]
        pads = bh_adjust(pvals)
        for g, padj in zip(group, pads):
            g2 = dict(g)
            g2["p_adj_bh"] = f"{padj:.6f}"
            g2["significant_bh"] = str(padj < 0.05)
            out_rows.append(g2)
    # Sort and write
    out_rows.sort(key=lambda r: (r["sweep"], float(r["gain"])))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
