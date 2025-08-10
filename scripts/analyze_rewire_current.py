import os
import argparse
import subprocess
import sys
import pandas as pd

"""
Orchestrate interim analysis for a rewire sweep directory using the *_current.csv files
produced by aggregate_rewire_partial.py. It will:
  - If >= 2 rewire levels present: plot threshold curve and metrics.
  - If >= 4 rewire levels: fit spline minimum and run bootstrap CI.
  - If metrics present with per-seed data: run PCA and mixed-effects.
"""


def run(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="runs/rewire_sweep_dense")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--smoothing", type=float, default=0.01)
    ap.add_argument("--grid", type=int, default=1201)
    ap.add_argument("--drop_exact_fit", type=str, default=None,
                    help="Comma-separated aggregated Threshold_mean values to drop in spline fit (e.g., '48')")
    ap.add_argument("--drop_at_or_above_fit", type=float, default=None,
                    help="Drop aggregated Threshold_mean >= this value in spline fit")
    ap.add_argument("--drop_exact_bs", type=str, default=None,
                    help="Comma-separated per-seed Threshold values to drop in bootstrap (e.g., '48')")
    ap.add_argument("--drop_at_or_above_bs", type=float, default=None,
                    help="Drop per-seed Threshold >= this value in bootstrap")
    args = ap.parse_args()

    thr_cur = os.path.join(args.dir, "rewire_sweep_thresholds_current.csv")
    met_cur = os.path.join(args.dir, "rewire_sweep_with_metrics_current.csv")
    met_final = os.path.join(args.dir, "rewire_sweep_with_metrics.csv")
    comb_cur = os.path.join(args.dir, "rewire_sweep_thresholds_current_combined.csv")

    if not os.path.isfile(thr_cur):
        print("No current thresholds CSV found. Run aggregate_rewire_partial.py first.")
        sys.exit(1)

    df_thr = pd.read_csv(thr_cur)
    n_levels = df_thr["RewireP"].nunique() if "RewireP" in df_thr.columns else 0
    print(f"Rewire levels available: {n_levels}")

    # Plot if >= 2 rewire levels
    if n_levels >= 2 and os.path.isfile(met_cur):
        run([args.python, "plot_rewire_curve.py", "--csv", met_cur, "--outdir", args.dir])
    elif n_levels >= 2:
        # plot thresholds only if no metrics yet
        run([args.python, "plot_rewire_curve.py", "--csv", thr_cur, "--outdir", args.dir])
    else:
        print("Skip plotting (need >=2 levels).")

    # Spline fit and bootstrap if >= 4 rewire levels
    if n_levels >= 4:
        fit_cmd = [args.python, "fit_rewire_curve.py", "--csv", thr_cur, "--outdir", args.dir, "--smoothing", str(args.smoothing), "--grid", str(args.grid)]
        if args.drop_exact_fit:
            fit_cmd += ["--drop_exact", args.drop_exact_fit]
        if args.drop_at_or_above_fit is not None:
            fit_cmd += ["--drop_at_or_above", str(args.drop_at_or_above_fit)]
        run(fit_cmd)
        # Use the combined file so bootstrap has per-seed thresholds
        if os.path.isfile(comb_cur):
            bs_cmd = [args.python, "bootstrap_rewire_min.py", "--csv", comb_cur, "--outdir", args.dir, "--B", str(args.B), "--smoothing", str(args.smoothing), "--grid", str(args.grid)]
            if args.drop_exact_bs:
                bs_cmd += ["--drop_exact", args.drop_exact_bs]
            if args.drop_at_or_above_bs is not None:
                bs_cmd += ["--drop_at_or_above", str(args.drop_at_or_above_bs)]
            run(bs_cmd)
        else:
            print("Skip bootstrap: combined CSV not found.")
    else:
        print("Skip spline/bootstrap (need >=4 levels).")

    # PCA and mixed-effects if metrics available and per-seed rows exist
    # Prefer current with metrics; fall back to final merged with metrics if needed
    chosen_met = met_cur if os.path.isfile(met_cur) else (met_final if os.path.isfile(met_final) else None)
    if chosen_met:
        # Try PCA on aggregated rows if enough levels
        if n_levels >= 4:
            run([args.python, "pca_metric_analysis.py", "--csv", chosen_met, "--out", os.path.join(args.dir, "pca_metric_analysis.txt"), "--use_mean"])
        # Mixed-effects requires per-seed metrics; we can reuse merged current file if it has per-seed rows
        # For now, run only when n_levels >= 4 to avoid degenerate models
        if n_levels >= 4:
            # mixed_effects expects per-seed rows; prefer final with per-seed rows if available
            me_csv = met_final if os.path.isfile(met_final) else met_cur
            run([args.python, "mixed_effects_rewire.py", "--csv", me_csv, "--out", os.path.join(args.dir, "mixed_effects_rewire.txt"), "--ols_only"])
    else:
        print("Metrics file not found; skipping PCA/mixed-effects for now.")


if __name__ == "__main__":
    main()
