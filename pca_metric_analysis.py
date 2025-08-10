import os, argparse, math
import pandas as pd
import numpy as np
from typing import List
import statsmodels.api as sm

"""PCA and nonlinear (quadratic) regression of masking threshold on latent structural factor.
Expects input CSV (e.g., rewire_sweep_with_metrics.csv) with columns:
  RewireP, Threshold_mean or per-seed Threshold50_logistic, CommMean, Fiedler, LongRangeFrac, WeightedEff
Outputs:
  - PCA loadings
  - Regression (linear + quadratic) of threshold on PC1 (and optionally PC2)
  - Model comparison AIC
  - Report text file
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='runs/rewire_sweep/rewire_sweep_with_metrics.csv')
    ap.add_argument('--out', type=str, default='runs/pca_metric_analysis.txt')
    ap.add_argument('--use_mean', action='store_true', help='Use aggregated Threshold_mean if available')
    ap.add_argument('--min_rewire', type=int, default=4, help='Minimum distinct rewire levels required')
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f'Input CSV not found: {args.csv}')
    df = pd.read_csv(args.csv)

    # Determine threshold column
    thr_col = None
    if args.use_mean and 'Threshold_mean' in df.columns:
        thr_col = 'Threshold_mean'
        base = df.dropna(subset=[thr_col])
    else:
        # prefer per-seed logistic thresholds
        cand = 'Threshold50_logistic'
        if cand not in df.columns:
            raise SystemExit('No threshold column found.')
        base = df.dropna(subset=[cand])
        thr_col = cand

    metrics = ['CommMean','Fiedler','LongRangeFrac','WeightedEff']
    metrics = [m for m in metrics if m in base.columns]
    if len(metrics) < 2:
        raise SystemExit('Not enough metrics for PCA.')

    # Aggregate to unique rewire levels if using per-seed
    if thr_col != 'Threshold_mean':
        grp = base.groupby('RewireP')
        agg_rows = []
        for rp, g in grp:
            row = {'RewireP': rp, 'Threshold_mean': g[thr_col].mean()}
            for m in metrics:
                row[m] = g[m].mean()
            agg_rows.append(row)
        base = pd.DataFrame(agg_rows)
        thr_col = 'Threshold_mean'

    # Drop rows with missing metric values to avoid SVD NaN issues
    base = base.dropna(subset=metrics)
    if base['RewireP'].nunique() < args.min_rewire:
        raise SystemExit('Insufficient distinct rewire levels.')

    # Z-score metrics
    Z = base[metrics].apply(lambda s: (s - s.mean())/(s.std(ddof=0)+1e-12))
    # PCA via SVD
    M = Z.values
    # Ensure finite matrix
    M = np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, VT = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
    pcs = U * S  # scores
    loadings = VT.T
    expl_var = (S**2) / (len(M)-1)
    expl_ratio = expl_var / expl_var.sum()

    base['PC1'] = pcs[:,0]
    if pcs.shape[1] > 1:
        base['PC2'] = pcs[:,1]

    y = base[thr_col].values
    X1 = sm.add_constant(base[['PC1']])
    m_lin = sm.OLS(y, X1).fit()

    # Quadratic on PC1
    base['PC1_sq'] = base['PC1']**2
    Xq = sm.add_constant(base[['PC1','PC1_sq']])
    m_quad = sm.OLS(y, Xq).fit()

    # Optional add PC2 linear
    m2 = None
    if 'PC2' in base.columns:
        X2 = sm.add_constant(base[['PC1','PC2']])
        m2 = sm.OLS(y, X2).fit()

    lines = []
    lines.append('PCA Structural Metric Analysis')
    lines.append('='*50)
    lines.append(f'Rows (rewire levels): {len(base)}')
    lines.append('Metrics used: ' + ', '.join(metrics))
    lines.append('\nExplained variance ratios:')
    for i,(ev, er) in enumerate(zip(expl_var, expl_ratio)):
        lines.append(f' PC{i+1}: var={ev:.4f} ratio={er:.4f}')
    lines.append('\nLoadings (columns=PCs):')
    load_df = pd.DataFrame(loadings, index=metrics, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
    lines.append(load_df.to_string())

    lines.append('\nLinear PC1 model:')
    lines.append(str(m_lin.summary()))
    lines.append('\nQuadratic PC1 model:')
    lines.append(str(m_quad.summary()))
    if m2 is not None:
        lines.append('\nPC1+PC2 model:')
        lines.append(str(m2.summary()))
    lines.append('\nModel AIC: linear={:.2f} quad={:.2f} PC1+PC2={}'.format(m_lin.aic, m_quad.aic, ('{:.2f}'.format(m2.aic) if m2 else 'NA')))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
