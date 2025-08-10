import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

"""
Bootstrap the minimum of Threshold vs RewireP by resampling per-seed thresholds.
Reads runs/rewire_sweep/rewire_sweep_thresholds.csv which contains both aggregated
and per-seed rows (with Threshold50_logistic). Outputs:
 - Text report with point estimate and 95% CI for rewire_p* and threshold*
 - Figure showing bootstrap distribution and the spline fit overlay
"""

def fit_min_spline(x: np.ndarray, y: np.ndarray, s: float, grid: int) -> tuple[float, float]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # Ensure strictly increasing x for spline; jitter duplicates slightly
    x_adj = x.copy()
    for i in range(1, len(x_adj)):
        if x_adj[i] <= x_adj[i-1]:
            x_adj[i] = x_adj[i-1] + 1e-9
    try:
        spl = UnivariateSpline(x_adj, y, s=s, k=3)
    except Exception:
        spl = UnivariateSpline(x_adj, y, s=max(1e-8, s), k=3)
    xx = np.linspace(float(x_adj.min()), float(x_adj.max()), grid)
    yy = spl(xx)
    i_min = int(np.argmin(yy))
    return float(xx[i_min]), float(yy[i_min])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='runs/rewire_sweep/rewire_sweep_thresholds.csv')
    ap.add_argument('--outdir', type=str, default='runs/rewire_sweep')
    ap.add_argument('--B', type=int, default=600)
    ap.add_argument('--smoothing', type=float, default=1e-4)
    ap.add_argument('--grid', type=int, default=1001)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--drop_at_or_above', type=float, default=None,
                    help='If set, drop per-seed thresholds >= this value (treat as ceiling)')
    ap.add_argument('--drop_exact', type=str, default=None,
                    help='Comma-separated list of exact threshold values to drop (e.g., "16,24")')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Extract per-seed rows with Threshold50_logistic and RewireP
    if 'Threshold50_logistic' not in df.columns:
        raise SystemExit('Per-seed Threshold50_logistic not found in CSV.')
    per_seed = df.dropna(subset=['Threshold50_logistic', 'RewireP'])
    # Exclude aggregated rows; keep those with Seed present
    if 'Seed' in per_seed.columns:
        per_seed = per_seed.dropna(subset=['Seed'])

    # Build mapping rp -> list of thresholds, optionally dropping ceiling values
    groups: dict[float, list[float]] = {}
    ceiling_cut = args.drop_at_or_above
    drop_exact: set[float] | None = None
    if args.drop_exact:
        try:
            drop_exact = {float(x.strip()) for x in args.drop_exact.split(',') if x.strip()}
        except Exception:
            drop_exact = None
    for _, r in per_seed.iterrows():
        rp = float(r['RewireP'])
        thr = float(r['Threshold50_logistic'])
        if not np.isfinite(rp) or not np.isfinite(thr):
            continue
        if drop_exact is not None and any(abs(thr - v) <= 1e-9 for v in drop_exact):
            continue
        if ceiling_cut is not None and thr >= ceiling_cut:
            # treat as censored ceiling; exclude from bootstrap/resampling
            continue
        groups.setdefault(rp, []).append(thr)

    # Drop rewire levels that have no usable per-seed values after filtering
    groups = {rp: vals for rp, vals in groups.items() if len(vals) > 0}

    if len(groups) < 4:
        raise SystemExit('Need at least 4 distinct rewire levels for spline fitting (after filtering).')

    rps = np.array(sorted(groups.keys()), dtype=float)
    # Point estimate: use per-rewire mean of per-seed thresholds
    y_mean = np.array([float(np.mean(groups[rp])) for rp in rps], dtype=float)
    x_hat, y_hat = fit_min_spline(rps, y_mean, s=args.smoothing, grid=args.grid)

    # Bootstrap
    rng = np.random.default_rng(args.seed)
    bs_x = np.empty(args.B, dtype=float)
    bs_y = np.empty(args.B, dtype=float)
    for b in range(args.B):
        # resample per rewire level
        yb = []
        for rp in rps:
            vals = groups[rp]
            if len(vals) == 0:
                yb.append(np.nan)
            elif len(vals) == 1:
                yb.append(float(vals[0]))
            else:
                idx = rng.integers(0, len(vals), size=len(vals))
                yb.append(float(np.mean([vals[i] for i in idx])))
        yb = np.array(yb, dtype=float)
        if np.isnan(yb).any():
            bs_x[b] = np.nan
            bs_y[b] = np.nan
            continue
        xm, ym = fit_min_spline(rps, yb, s=args.smoothing, grid=args.grid)
        bs_x[b] = xm
        bs_y[b] = ym

    # Clean and compute CI
    valid = np.isfinite(bs_x) & np.isfinite(bs_y)
    bs_x = bs_x[valid]; bs_y = bs_y[valid]
    if bs_x.size < max(50, args.B // 5):
        print('[warn] Few valid bootstrap samples; CI may be unstable.')
    lo_x, hi_x = np.percentile(bs_x, [2.5, 97.5])
    lo_y, hi_y = np.percentile(bs_y, [2.5, 97.5])

    # Save report
    lines = []
    lines.append('Bootstrap spline minimum for Threshold vs RewireP')
    lines.append('='*60)
    lines.append(f'B = {args.B}  smoothing={args.smoothing}  grid={args.grid}  seed={args.seed}')
    if ceiling_cut is not None:
        lines.append(f'Filtered per-seed thresholds >= {ceiling_cut} (treated as ceiling)')
    if drop_exact:
        lines.append(f'Filtered exact per-seed thresholds equal to: {sorted(drop_exact)}')
    lines.append(f'Rewire levels: {len(rps)}  Seeds per level (min/median/max): '
                 f"{min(len(groups[rp]) for rp in rps)}/{int(np.median([len(groups[rp]) for rp in rps]))}/{max(len(groups[rp]) for rp in rps)}")
    lines.append('Point estimate:')
    lines.append(f'  rewire_p* ~ {x_hat:.3f}  threshold* ~ {y_hat:.3f}')
    lines.append('95% CI (bootstrap percentiles):')
    lines.append(f'  rewire_p*: [{lo_x:.3f}, {hi_x:.3f}]')
    lines.append(f'  threshold*: [{lo_y:.3f}, {hi_y:.3f}]')
    out_txt = os.path.join(args.outdir, 'spline_min_bootstrap.txt')
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))
    print('Wrote', out_txt)

    # Figure: histogram of bs_x and overlay point estimate and CI
    plt.figure(figsize=(7.5, 4.0))
    plt.hist(bs_x, bins=24, color='#89a1ef', edgecolor='white', alpha=0.9)
    plt.axvline(x_hat, color='#d34e24', lw=2, label=f'Point est: {x_hat:.3f}')
    plt.axvspan(lo_x, hi_x, color='#d34e24', alpha=0.2, label=f'95% CI [{lo_x:.3f}, {hi_x:.3f}]')
    plt.xlabel('Bootstrap rewire_p*')
    plt.ylabel('Count')
    plt.title('Bootstrap distribution of optimal rewire probability')
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.outdir, 'spline_min_bootstrap.png')
    out_pdf = os.path.join(args.outdir, 'spline_min_bootstrap.pdf')
    plt.savefig(out_png, dpi=200); plt.savefig(out_pdf)
    print('Wrote figures:', out_png, out_pdf)

if __name__ == '__main__':
    main()
