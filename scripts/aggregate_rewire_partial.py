import os
import glob
import argparse
from typing import Any, Dict, List, Tuple, Set
import numpy as np
import pandas as pd

"""
Aggregate partial rewire sweep outputs into provisional CSVs for plotting and fitting.
Inputs (in --dir):
  - rewire_sweep_thresholds_partial.csv (mixed aggregated + per-seed rows)
  - rewire_sweep_metrics_partial.csv    (per-seed metric rows)

Outputs (written to the same --dir):
  - rewire_sweep_thresholds_current.csv        (aggregated rows only)
  - rewire_sweep_with_metrics_current.csv      (aggregated rows + mean metrics by rewire)
  - rewire_sweep_thresholds_current_combined.csv (aggregated rows + per-seed rows)
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
                    help="Directory containing partial CSVs")
    args = ap.parse_args()

    thr_path = os.path.join(args.dir, "rewire_sweep_thresholds_partial.csv")
    met_path = os.path.join(args.dir, "rewire_sweep_metrics_partial.csv")
    if not os.path.isfile(thr_path):
        print("Missing partial thresholds:", thr_path)
        raise SystemExit(1)
    df = pd.read_csv(thr_path)  # type: ignore[call-overload]
    if df.empty:
        print("Partial thresholds CSV is empty")
        raise SystemExit(1)

    # Per-seed rows
    per_seed = df.copy()
    if "Seed" in per_seed.columns:
        per_seed = per_seed.dropna(subset=["Seed"])  # type: ignore[call-overload]
    need_cols = ["RewireP", "Threshold50_logistic"]
    for c in need_cols:
        if c not in per_seed.columns:
            print("Missing column in partial thresholds:", c)
            raise SystemExit(1)
    per_seed = per_seed.dropna(subset=need_cols)  # type: ignore[call-overload]

    # Also scan per-seed subdirectories (rp_*_seed_*) to pick up older runs
    # and ensure we aggregate across multiple invocations.
    seen_pairs: Set[Tuple[float, int]] = set()
    if not per_seed.empty and "Seed" in per_seed.columns:
        try:
            for _, r in per_seed.iterrows():
                seen_pairs.add((float(r["RewireP"]), int(r["Seed"])) )
        except Exception:
            pass
    rows_seed_from_dirs: List[Dict[str, Any]] = []
    for sub in glob.glob(os.path.join(args.dir, "rp_*_seed_*")):
        thr_csv = os.path.join(sub, "multi_size_thresholds.csv")
        if not os.path.isfile(thr_csv):
            continue
        try:
            df_thr = pd.read_csv(thr_csv)  # type: ignore[call-overload]
            if df_thr.empty:
                continue
            r = df_thr.iloc[0].to_dict()
            rp = float(r.get("RewireP", np.nan))
            thr = float(r.get("Threshold50_logistic", np.nan))
            # parse seed from folder name
            base = os.path.basename(sub)
            try:
                seed = int(base.split("seed_")[-1])
            except Exception:
                seed = int(r.get("Seed", np.nan)) if pd.notna(r.get("Seed", np.nan)) else -1
            if np.isfinite(rp) and np.isfinite(thr):
                if (rp, seed) not in seen_pairs and seed != -1:
                    row: Dict[str, Any] = {
                        "RewireP": rp,
                        "Seed": seed,
                        "Threshold50_logistic": thr,
                    }
                    # carry Size/N if available
                    for cand in ("Size", "N"):
                        if cand in r and pd.notna(r[cand]):
                            row[cand] = r[cand]
                    rows_seed_from_dirs.append(row)
        except Exception:
            continue
    if rows_seed_from_dirs:
        per_seed = pd.concat([per_seed, pd.DataFrame(rows_seed_from_dirs)], ignore_index=True, sort=False)  # type: ignore[call-overload]

    # Build aggregated rows per RewireP
    rows: List[Dict[str, Any]] = []
    for rp, g in per_seed.groupby("RewireP"):  # type: ignore[call-overload]
        rp_val = float(rp)  # type: ignore[arg-type]
        vals = g["Threshold50_logistic"].astype(float).tolist()
        vals = [float(x) for x in vals if np.isfinite(x)]
        n = len(vals)
        mu = float(np.mean(vals)) if n else float("nan")
        lo, hi = ci95(vals) if n >= 4 else (float("nan"), float("nan"))
        # Try to get Size/N if present
        size_val = float("nan")
        for cand in ("Size", "N"):
            if cand in g.columns and g[cand].notna().any():  # type: ignore[call-overload]
                try:
                    size_val = float(pd.to_numeric(g[cand], errors="coerce").dropna().iloc[0])  # type: ignore[call-overload]
                    break
                except Exception:
                    pass
        rows.append({
            "RewireP": rp_val,
            "Size": size_val,
            "Seeds": int(n),
            "Threshold_mean": mu,
            "Threshold_lo": lo,
            "Threshold_hi": hi,
        })

    agg = pd.DataFrame(rows).sort_values("RewireP")  # type: ignore[call-overload]
    out_thr = os.path.join(args.dir, "rewire_sweep_thresholds_current.csv")
    agg.to_csv(out_thr, index=False)
    print("Wrote", out_thr)

    # Metrics aggregation (mean over seeds per rewire)
    with_met = agg.copy()
    if os.path.isfile(met_path):
        met = pd.read_csv(met_path)  # type: ignore[call-overload]
        if not met.empty and "RewireP" in met.columns:
            metric_cols = [c for c in [
                "Clustering","CharPath","GlobalEff","Fiedler","WeightedEff","CommMean","LongRangeFrac"
            ] if c in met.columns]
            if metric_cols:
                met_grp = met.groupby("RewireP", as_index=False)[metric_cols].mean()  # type: ignore[call-overload]
                with_met = pd.merge(with_met, met_grp, on="RewireP", how="left")  # type: ignore[call-overload]
    out_thr_met = os.path.join(args.dir, "rewire_sweep_with_metrics_current.csv")
    with_met.to_csv(out_thr_met, index=False)
    print("Wrote", out_thr_met)

    # Combined file: aggregated rows + per-seed rows (for bootstrap convenience)
    combined = pd.concat([agg, per_seed], ignore_index=True, sort=False)  # type: ignore[call-overload]
    out_combined = os.path.join(args.dir, "rewire_sweep_thresholds_current_combined.csv")
    combined.to_csv(out_combined, index=False)
    print("Wrote", out_combined)


if __name__ == "__main__":
    main()
