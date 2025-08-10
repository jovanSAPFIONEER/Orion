import os, argparse, math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations

"""
Analyze relationship between masking threshold (Threshold50_logistic) and
advanced structural metrics derived in pattern_graph_metrics.py.
Performs:
 1. Data load & cleaning
 2. Correlation matrix & VIF diagnostics
 3. OLS with cluster-robust SE (by Seed) if available
 4. Standardized coefficients
 5. Partial correlations (pairwise, controlling for remaining predictors)
Outputs plain-text report and CSV of coefficient table.
"""

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def vif(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        y = df[col]
        X = df.drop(columns=[col])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared
        v = np.inf if (1 - r2) < 1e-12 else 1.0 / (1 - r2)
        rows.append({'Predictor': col, 'VIF': v})
    return pd.DataFrame(rows)

def partial_corr(df: pd.DataFrame, x: str, y: str, controls: list) -> float:
    # Regress x, y on controls, then correlate residuals.
    if controls:
        Xc = sm.add_constant(df[controls])
        rx = sm.OLS(df[x], Xc).fit().resid
        ry = sm.OLS(df[y], Xc).fit().resid
        a, b = rx.values, ry.values
    else:
        a, b = df[x].values, df[y].values
    am, bm = a.mean(), b.mean()
    num = ((a-am)*(b-bm)).sum()
    den = math.sqrt(((a-am)**2).sum() * ((b-bm)**2).sum()) + 1e-12
    return float(num/den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics_csv', type=str, default='runs/pattern_metrics_with_thresholds.csv')
    ap.add_argument('--out_txt', type=str, default='runs/pattern_metric_regression.txt')
    ap.add_argument('--out_coef_csv', type=str, default='runs/pattern_metric_regression_coefs.csv')
    ap.add_argument('--predictors', nargs='+', default=['CommMean','Fiedler','LongRangeFrac','WeightedEff'])
    ap.add_argument('--min_rows', type=int, default=8)
    args = ap.parse_args()

    if not os.path.isfile(args.metrics_csv):
        raise SystemExit(f"Metrics CSV not found: {args.metrics_csv}")
    df = pd.read_csv(args.metrics_csv)
    # Basic filters
    needed_cols = ['Threshold50_logistic','Pattern','Seed'] + args.predictors
    for c in needed_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in metrics CSV")
    df = df.dropna(subset=['Threshold50_logistic'])
    if len(df) < args.min_rows:
        raise SystemExit(f"Not enough rows after filtering: {len(df)} < {args.min_rows}")

    # Keep only predictors with variability
    active_preds = []
    for p in args.predictors:
        if df[p].nunique() > 1:
            active_preds.append(p)
        else:
            print(f'[info] Dropping predictor with no variance: {p}')
    if not active_preds:
        raise SystemExit('No predictors with variance remain.')

    # Z-score predictors for standardized coefficients
    Z = df[active_preds].apply(zscore)
    y = df['Threshold50_logistic']

    X = sm.add_constant(Z)
    # Cluster robust by Seed if multiple seeds present; else by Pattern if multiple patterns; else none.
    clusters = None
    cluster_name = None
    if df['Seed'].nunique() > 1:
        clusters = df['Seed']
        cluster_name = 'Seed'
    elif df['Pattern'].nunique() > 1:
        clusters = df['Pattern']
        cluster_name = 'Pattern'

    if clusters is not None and len(set(clusters)) >= 3:
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': clusters})
    else:
        model = sm.OLS(y, X).fit()
        cluster_name = 'None'

    # Build coefficient table with standardized betas
    coefs = []
    for name, val in model.params.items():
        if name == 'const':
            continue
        se = model.bse[name]
        tval = model.tvalues[name]
        pval = model.pvalues[name]
        coefs.append({'Predictor': name, 'Beta_std': val, 'SE': se, 't': tval, 'p': pval})
    df_coefs = pd.DataFrame(coefs)

    # Correlation matrix
    corr_mat = df[active_preds + ['Threshold50_logistic']].corr()

    # VIF
    vif_df = vif(Z[active_preds]) if len(active_preds) >= 2 else pd.DataFrame()

    # Partial correlations (predictor with threshold controlling others)
    pc_rows = []
    for p in active_preds:
        others = [o for o in active_preds if o != p]
        r = partial_corr(df[['Threshold50_logistic'] + active_preds], p, 'Threshold50_logistic', others)
        pc_rows.append({'Predictor': p, 'PartialR': r})
    pc_df = pd.DataFrame(pc_rows)

    # Pairwise partial correlations between predictors controlling others
    pair_pc_rows = []
    if len(active_preds) >= 3:
        for a, b in combinations(active_preds, 2):
            others = [o for o in active_preds if o not in (a,b)]
            r = partial_corr(df[active_preds], a, b, others)
            pair_pc_rows.append({'A': a, 'B': b, 'PartialR_ctrl_others': r})
    pair_pc_df = pd.DataFrame(pair_pc_rows)

    # Write report
    lines = []
    lines.append('Masking Threshold Metric Regression Report')
    lines.append('='*50)
    lines.append(f'Rows: {len(df)}  Patterns: {df["Pattern"].nunique()}  Seeds: {df["Seed"].nunique()}  Cluster: {cluster_name}')
    lines.append('Active Predictors: ' + ', '.join(active_preds))
    lines.append('\nOLS Summary (cluster-robust if applicable):')
    lines.append(str(model.summary()))
    lines.append('\nStandardized Coefficients:')
    lines.append(df_coefs.to_string(index=False))
    if not vif_df.empty:
        lines.append('\nVIF:')
        lines.append(vif_df.to_string(index=False))
    lines.append('\nPredictor-Threshold Partial Correlations (controlling others):')
    lines.append(pc_df.to_string(index=False))
    if not pair_pc_df.empty:
        lines.append('\nPredictor-Predictor Partial Correlations (controlling remaining predictors):')
        lines.append(pair_pc_df.to_string(index=False))
    lines.append('\nCorrelation Matrix:')
    lines.append(corr_mat.to_string())

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, 'w') as f:
        f.write('\n'.join(lines))
    df_coefs.to_csv(args.out_coef_csv, index=False)

    print('Wrote', args.out_txt)
    print('Wrote', args.out_coef_csv)

if __name__ == '__main__':
    main()
