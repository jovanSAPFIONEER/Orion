import os, argparse, math
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Sequence

# Import analysis and metrics from project modules
from multi_size_masking import analyze_sizes
from pattern_graph_metrics import compute_metrics

def ci95(vals: Sequence[float]) -> Tuple[float, float]:
    """Approximate 95% CI for the mean via normal approximation.
    Returns (lo, hi); NaNs if insufficient data.
    """
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float('nan'), float('nan'))
    mu = float(np.mean(v))
    se = float(np.std(v, ddof=1) / max(1.0, np.sqrt(v.size)))
    half = 1.96 * se
    return (mu - half, mu + half)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--degree_frac', type=float, default=0.38)
    ap.add_argument('--rewire_ps', nargs='+', type=float, default=[0.0,0.05,0.10,0.15,0.18,0.25,0.35,0.5])
    ap.add_argument('--soas', nargs='+', type=int, default=[0,1,2,3,4,6,8,10,12])
    ap.add_argument('--trials', type=int, default=35)
    ap.add_argument('--seeds', nargs='+', type=int, default=[12000,12001,12002,12003,12004])
    ap.add_argument('--boots', type=int, default=140)
    ap.add_argument('--amp_t', type=float, default=0.9)
    ap.add_argument('--amp_m', type=float, default=2.0)
    ap.add_argument('--outdir', type=str, default='runs/rewire_sweep')
    ap.add_argument('--backward_only', action='store_true')
    ap.add_argument('--mask_width', type=int, default=1)
    ap.add_argument('--target_width', type=int, default=1)
    ap.add_argument('--gw_tau', type=float, default=0.9)
    ap.add_argument('--gw_theta', type=float, default=0.55)
    ap.add_argument('--noise_scale', type=float, default=1.0)
    ap.add_argument('--hit_window_radius', type=int, default=5)
    ap.add_argument('--lat_search_max', type=int, default=22)
    ap.add_argument('--slope_min_div', type=float, default=50.0)
    ap.add_argument('--slope_max_factor', type=float, default=0.6)
    ap.add_argument('--save_partial', action='store_true', help='Write partial CSV after each rewire level')
    ap.add_argument('--extend_if_ceiling', action='store_true', help='If threshold hits max SOA, extend SOAs and retry')
    ap.add_argument('--max_soa_limit', type=int, default=24, help='Upper limit for adaptive SOA extension')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_thresh_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    for rp in args.rewire_ps:
        print(f'[rewire_sweep] Starting rewire_p={rp:.3f}')
        per_seed_thresholds: List[float] = []
        per_seed_rows: List[Dict[str, Any]] = []
        mean_thr = math.nan

        for seed in args.seeds:
            run_out = os.path.join(args.outdir, f"rp_{rp:.3f}_seed_{seed}")
            _, df_thr, _, _ = analyze_sizes(
                sizes=[args.size], soas=args.soas, trials=args.trials, seed0=seed,
                outdir=run_out, boots=args.boots, calibrate=False,
                amp_t=args.amp_t, amp_m=args.amp_m, ignition_thresh=0.5, fast_boot=False,
                debug_ign=False, debug_ign_limit=3, noise_scale=args.noise_scale,
                hit_window_radius=args.hit_window_radius, lat_search_max=args.lat_search_max,
                slope_min_div=args.slope_min_div, slope_max_factor=args.slope_max_factor,
                pattern='small_world', modules=4, inter_frac=0.05, gw_tau=args.gw_tau, gw_theta=args.gw_theta,
                pmax_latency_percentile=75.0, pmax_latency_min=0.25, backward_only=args.backward_only,
                mask_width=args.mask_width, target_width=args.target_width,
                degree_frac=args.degree_frac, rewire_p=rp)

            # Adaptive extension if ceiling threshold encountered (iterate until resolved or limit)
            if args.extend_if_ceiling:
                try:
                    thr = float(df_thr.iloc[0]['Threshold50_logistic']) if len(df_thr) > 0 else float('nan')
                except Exception:
                    thr = float('nan')
                last_soa = max(args.soas) if args.soas else 0
                ext_soas = list(args.soas)
                while True:
                    need_extend = (not np.isfinite(thr)) or (thr >= (max(ext_soas) if ext_soas else last_soa) - 1e-9)
                    if (not need_extend):
                        break
                    cur_max = max(ext_soas) if ext_soas else last_soa
                    if cur_max >= args.max_soa_limit:
                        break
                    new_max = min(args.max_soa_limit, cur_max + 4)
                    new_range = list(range(cur_max + 2, new_max + 1, 2))
                    ext_soas = sorted(set(list(ext_soas) + new_range))
                    print(f"[rewire_sweep] Extending SOAs {sorted(set(args.soas))} -> {ext_soas} for rp={rp:.3f} seed={seed}")
                    _, df_thr, _, _ = analyze_sizes(
                        sizes=[args.size], soas=ext_soas, trials=args.trials, seed0=seed,
                        outdir=run_out, boots=args.boots, calibrate=False,
                        amp_t=args.amp_t, amp_m=args.amp_m, ignition_thresh=0.5, fast_boot=False,
                        debug_ign=False, debug_ign_limit=3, noise_scale=args.noise_scale,
                        hit_window_radius=args.hit_window_radius, lat_search_max=args.lat_search_max,
                        slope_min_div=args.slope_min_div, slope_max_factor=args.slope_max_factor,
                        pattern='small_world', modules=4, inter_frac=0.05, gw_tau=args.gw_tau, gw_theta=args.gw_theta,
                        pmax_latency_percentile=75.0, pmax_latency_min=0.25, backward_only=args.backward_only,
                        mask_width=args.mask_width, target_width=args.target_width,
                        degree_frac=args.degree_frac, rewire_p=rp)
                    try:
                        thr = float(df_thr.iloc[0]['Threshold50_logistic']) if len(df_thr) > 0 else float('nan')
                    except Exception:
                        thr = float('nan')

            if len(df_thr) > 0:
                row = {col: df_thr.iloc[0][col] for col in df_thr.columns}
                row['Seed'] = seed
                per_seed_rows.append(row)
                val = row.get('Threshold50_logistic', math.nan)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    per_seed_thresholds.append(float(val))

            # metrics
            try:
                mrow = compute_metrics(args.size, 'small_world', seed, args.degree_frac, rp, 4, 0.05, density=0.12)
                mrow['RewireP'] = rp
                mrow['Seed'] = seed
                metrics_rows.append(mrow)
            except Exception as e:
                print('[warn] metric fail rp', rp, 'seed', seed, e)

        # aggregate threshold across seeds
        if per_seed_rows:
            thr_vals = np.array(per_seed_thresholds, dtype=float)
            mean_thr = float(np.nanmean(thr_vals)) if len(thr_vals) > 0 else math.nan
            lo_thr, hi_thr = ci95(thr_vals.tolist()) if len(thr_vals) > 3 else (math.nan, math.nan)
            all_thresh_rows.append({
                'RewireP': rp,
                'Size': args.size,
                'Seeds': len(per_seed_rows),
                'Threshold_mean': mean_thr,
                'Threshold_lo': lo_thr,
                'Threshold_hi': hi_thr,
                'DegreeFrac': args.degree_frac,
                'BackwardOnly': args.backward_only,
                'Trials': args.trials,
            })
            # Also store raw per-seed thresholds
            for r in per_seed_rows:
                r2 = r.copy(); r2['RewireP'] = rp
                all_thresh_rows.append(r2)

        try:
            mt = float(mean_thr)
        except Exception:
            mt = float('nan')
        print(f'[rewire_sweep] Finished rewire_p={rp:.3f} mean_thr={mt:.3f}')
        if args.save_partial and all_thresh_rows:
            thresh_df_partial = pd.DataFrame(all_thresh_rows)
            thresh_df_partial.to_csv(os.path.join(args.outdir, 'rewire_sweep_thresholds_partial.csv'), index=False)
            if metrics_rows:
                met_df_partial = pd.DataFrame(metrics_rows)
                met_df_partial.to_csv(os.path.join(args.outdir, 'rewire_sweep_metrics_partial.csv'), index=False)
            print('[rewire_sweep] Wrote partial progress CSVs')

    # Merge mean metrics per rewire
    met_df = pd.DataFrame(metrics_rows)
    # Aggregate metrics across seeds per rewire
    rp_met_df = pd.DataFrame()
    if not met_df.empty:
        met_df = met_df.copy()
        met_df['RewireP'] = pd.to_numeric(met_df['RewireP'], errors='coerce')  # type: ignore[call-overload]
        metric_cols = ['Clustering','CharPath','GlobalEff','Fiedler','WeightedEff','CommMean','LongRangeFrac']
        metric_cols = [c for c in metric_cols if c in met_df.columns]
        rp_met_df = met_df.groupby('RewireP', as_index=False)[metric_cols].mean()  # type: ignore[call-overload]

    thresh_df = pd.DataFrame(all_thresh_rows)
    thresh_path = os.path.join(args.outdir, 'rewire_sweep_thresholds.csv')
    thresh_df.to_csv(thresh_path, index=False)
    print('Wrote', thresh_path)
    if not rp_met_df.empty:
        merged = pd.merge(thresh_df, rp_met_df, on='RewireP', how='left')  # type: ignore[call-overload]
        merged_path = os.path.join(args.outdir, 'rewire_sweep_with_metrics.csv')
        merged.to_csv(merged_path, index=False)
        print('Wrote', merged_path)

if __name__ == '__main__':
    main()
