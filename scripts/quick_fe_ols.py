import os, argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

METRICS = ['CommMean','Fiedler','LongRangeFrac','WeightedEff']

def compute_pc1(df: pd.DataFrame) -> pd.Series:
    cols = [m for m in METRICS if m in df.columns]
    if len(cols) < 2:
        raise SystemExit('Not enough metrics to compute PC1.')
    X = df[cols].astype(float).to_numpy()
    # z-score by column
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    # replace non-finite
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # SVD
    U, S, VT = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    pc1 = U[:, 0] * S[0]
    return pd.Series(pc1, index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--thr_col', type=str, default='Threshold50_logistic')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    base = df.dropna(subset=[args.thr_col]).copy()
    base = base.dropna(subset=['RewireP'])
    # Ensure per-seed
    if 'Seed' not in base.columns:
        raise SystemExit('Seed column missing')
    # Keep rows with metrics
    have_metrics = [m for m in METRICS if m in base.columns]
    if len(have_metrics) < 2:
        raise SystemExit('Metrics not found in CSV (need at least 2 of: %s)' % ', '.join(METRICS))
    base = base.dropna(subset=have_metrics)

    base['PC1'] = compute_pc1(base)

    y = base[args.thr_col].astype(float).to_numpy()
    X = sm.add_constant(base[['PC1']].astype(float))
    X_arr = X.to_numpy(dtype=float)

    # Seed FE with clustered SE by Seed
    seed_dum = pd.get_dummies(base['Seed'].astype(int), prefix='Seed', drop_first=True)
    X_fe = sm.add_constant(pd.concat([base[['PC1']].astype(float), seed_dum], axis=1)).to_numpy(dtype=float)
    ols_fe = sm.OLS(y, X_fe).fit(cov_type='cluster', cov_kwds={'groups': base['Seed'].astype(int)})

    lines = []
    lines.append('Fixed-effects OLS: threshold ~ PC1 + Seed dummies (clustered by Seed)')
    lines.append('='*72)
    lines.append(f'Rows: {len(base)}  Seeds: {base.Seed.nunique()}  Rewire levels: {base.RewireP.nunique()}')
    lines.append('Metrics used for PC1: ' + ', '.join(have_metrics))
    lines.append('\nOLS FE summary:')
    lines.append(str(ols_fe.summary()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
