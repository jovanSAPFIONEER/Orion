import os, argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

"""
Mixed-effects model: threshold ~ PC1 (latent structure) with random intercepts by Seed.
Loads per-seed rows from runs/rewire_sweep/rewire_sweep_with_metrics.csv and computes PC1 from metrics.
Writes a brief report to runs/mixed_effects_rewire.txt.
"""

METRIC_COLS = ['CommMean','Fiedler','LongRangeFrac','WeightedEff']


def compute_pc1(df: pd.DataFrame, metrics: list[str]) -> pd.Series:
    Z = df[metrics].apply(lambda s: (s - s.mean())/(s.std(ddof=0)+1e-12))
    M = Z.values
    # Center columns
    M = M - M.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    pc_scores = U[:, 0] * S[0]
    return pd.Series(pc_scores, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='runs/rewire_sweep/rewire_sweep_with_metrics.csv')
    ap.add_argument('--out', type=str, default='runs/mixed_effects_rewire.txt')
    ap.add_argument('--thr_col', type=str, default='Threshold50_logistic', help='Use per-seed logistic thresholds')
    ap.add_argument('--ols_only', action='store_true', help='Skip MixedLM and only run FE OLS with clustered SE')
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f'CSV not found: {args.csv}')
    df = pd.read_csv(args.csv)

    # Filter to per-seed rows (have the specified per-seed threshold column)
    if args.thr_col not in df.columns:
        raise SystemExit(f'{args.thr_col} not found in CSV. Please run with per-seed data present.')
    base = df.dropna(subset=[args.thr_col]).copy()

    metrics = [m for m in METRIC_COLS if m in base.columns]
    if len(metrics) < 2:
        raise SystemExit('Not enough metrics present to compute PC1.')

    # Compute PC1 on available rows
    base['PC1'] = compute_pc1(base, metrics)
    # Clean rows: require finite threshold, PC1, and Seed
    cols_needed = [args.thr_col, 'PC1', 'Seed']
    base = base.dropna(subset=cols_needed)
    base = base[np.isfinite(base[args.thr_col]) & np.isfinite(base['PC1'])]

    # Prepare MixedLM: threshold ~ PC1, random intercept by Seed
    # Drop rows without Seed
    if 'Seed' not in base.columns:
        raise SystemExit('Seed column not found in data.')
    base = base.dropna(subset=['Seed'])

    endog = base[args.thr_col].astype(float).to_numpy()
    exog = sm.add_constant(base[['PC1']].astype(float))
    exog = exog.to_numpy(dtype=float)
    groups = base['Seed'].astype(int).to_numpy()

    lines = []
    lines.append('Mixed-effects model: threshold ~ PC1 + (1 | Seed)')
    lines.append('='*60)
    lines.append(f'Rows (per-seed): {len(base)}  Seeds: {base.Seed.nunique()}  Rewire levels: {base.RewireP.nunique()}')
    lines.append('Metrics used for PC1: ' + ', '.join(metrics))

    # Try MixedLM with REML first; if it fails or is singular, try alternative optimizer; then fall back to OLS with seed FE + clustered SE
    mixed_ok = False
    mixed_err = None
    if not args.ols_only:
        try:
            model = MixedLM(endog, exog, groups=groups)
            try:
                result = model.fit(reml=True, method='lbfgs')
                mixed_ok = True
            except Exception as e1:
                # Try a derivative-free optimizer
                try:
                    result = model.fit(reml=True, method='nm')
                    mixed_ok = True
                except Exception as e2:
                    mixed_err = f'MixedLM failed: lbfgs={e1}; nm={e2}'
        except Exception as e:
            mixed_err = f'MixedLM construction failed: {e}'

        if mixed_ok:
            lines.append('\nMixedLM summary:')
            lines.append(str(result.summary()))
        else:
            lines.append('\nMixedLM unavailable (singular or failed). Reason:')
            lines.append(str(mixed_err))

    # Always provide robust fixed-effects alternative: OLS with Seed dummies and cluster-robust SE by Seed
    seed_fe = pd.get_dummies(base['Seed'].astype(int), prefix='Seed', drop_first=True)
    X_fe_df = pd.concat([base[['PC1']].astype(float), seed_fe], axis=1)
    X_fe = sm.add_constant(X_fe_df)
    X_fe_arr = X_fe.to_numpy(dtype=float)
    ols_fe = sm.OLS(endog, X_fe_arr).fit(cov_type='cluster', cov_kwds={'groups': base['Seed'].astype(int)})
    lines.append('\nFixed-effects OLS with Seed dummies (cluster-robust by Seed):')
    lines.append(str(ols_fe.summary()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
