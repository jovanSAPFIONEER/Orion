import os
import argparse
from typing import Any, List, Dict, Tuple
import pandas as pd
import numpy as np

"""
Show progress and provisional stats for a rewire sweep directory using partial CSVs.
Reads <dir>/rewire_sweep_thresholds_partial.csv and prints per-RewireP status:
  - seeds_done, mean threshold (provisional), and 95% CI if >=4 seeds.
"""


def ci95(vals: List[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    mu = float(np.mean(v))
    se = float(np.std(v, ddof=1) / max(1.0, np.sqrt(v.size)))
    half = 1.96 * se
    return (mu - half, mu + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="runs/rewire_sweep_dense",
                    help="Sweep output directory containing partial CSVs")
    ap.add_argument("--total_seeds", type=int, default=8, help="Planned seeds per rewire level")
    args = ap.parse_args()

    thr_path = os.path.join(args.dir, "rewire_sweep_thresholds_partial.csv")
    if not os.path.isfile(thr_path):
        print("No partial thresholds CSV found:", thr_path)
        raise SystemExit(1)
    df = pd.read_csv(thr_path)  # type: ignore[call-overload]
    if df.empty:
        print("Partial thresholds CSV is empty")
        raise SystemExit(1)

    # keep per-seed rows that have Seed and Threshold50_logistic
    per_seed = df.copy()
    if "Seed" in per_seed.columns:
        per_seed = per_seed.dropna(subset=["Seed"])  # type: ignore[call-overload]
    if "Threshold50_logistic" not in per_seed.columns:
        print("Threshold50_logistic column missing in partial CSV.")
        raise SystemExit(1)
    per_seed = per_seed.dropna(subset=["Threshold50_logistic", "RewireP"])  # type: ignore[call-overload]

    # group by RewireP
    groups = per_seed.groupby("RewireP")  # type: ignore[call-overload]
    rows: List[Dict[str, Any]] = []
    for rp, g in groups:  # type: ignore[misc]
        vals = [float(x) for x in g["Threshold50_logistic"].tolist() if np.isfinite(x)]
        n = len(vals)
        mu = float(np.mean(vals)) if n else float("nan")
        lo, hi = ci95(vals) if n >= 4 else (float("nan"), float("nan"))
        try:
            rp_val = float(rp)  # type: ignore[arg-type]
        except Exception:
            rp_val = float("nan")
        rows.append({
            "RewireP": rp_val,
            "Seeds_done": int(n),
            "Seeds_planned": int(args.total_seeds),
            "Threshold_mean_prov": mu,
            "Threshold_lo_prov": lo,
            "Threshold_hi_prov": hi,
            "Progress": f"{n}/{args.total_seeds}",
        })

    out = pd.DataFrame(rows).sort_values("RewireP")  # type: ignore[call-overload]
    if out.empty:
        print("No per-seed rows available yet.")
        raise SystemExit(0)

    # Print a compact table
    cols = ["RewireP", "Progress", "Threshold_mean_prov", "Threshold_lo_prov", "Threshold_hi_prov"]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))  # type: ignore[call-overload]


if __name__ == "__main__":
    main()
