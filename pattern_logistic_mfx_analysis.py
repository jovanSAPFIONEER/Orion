import os, glob
import pandas as pd
import numpy as np
import statsmodels.api as sm

RUN_GLOB = 'runs/*_backward_harsh'
BASE_SOAS = [0,1,2,3,4,5,6,7,8]

# We'll fit a logistic regression of hit ~ C(Pattern) * SOA treating each trial as a row.
# Since we only have one size and (likely) one seed per pattern, we can't add proper random effects;
# we will cluster robust standard errors by pattern to get conservative inference.

def load_trials():
    frames = []
    for path in glob.glob(RUN_GLOB):
        trial_csv = os.path.join(path, 'multi_size_trial_level.csv')
        thr_csv = os.path.join(path, 'multi_size_thresholds.csv')
        if not (os.path.isfile(trial_csv) and os.path.isfile(thr_csv)):
            continue
        df_t = pd.read_csv(trial_csv)
        df_th = pd.read_csv(thr_csv)
        if df_th.empty: continue
        pattern = df_th.loc[0, 'Pattern'] if 'Pattern' in df_th.columns else os.path.basename(path)
        df_t['Pattern'] = pattern
        frames.append(df_t)
    if not frames:
        raise SystemExit('No trial data found.')
    df = pd.concat(frames, ignore_index=True)
    # Limit to base SOAs if necessary
    df = df[df['SOA'].isin(BASE_SOAS)].copy()
    return df

def prepare_design(df: pd.DataFrame):
    # Encode pattern categorical (baseline = small_world if present else first alphabetically)
    pats = sorted(df['Pattern'].unique())
    baseline = 'small_world' if 'small_world' in pats else pats[0]
    df['Pattern'] = pd.Categorical(df['Pattern'], categories=[baseline]+[p for p in pats if p!=baseline])
    # Center SOA for stability
    df['SOA_c'] = df['SOA'] - df['SOA'].mean()
    # Design matrix: intercept + pattern dummies + SOA_c + interactions
    # One-hot encode pattern only (drop_first=True omits baseline dummy) to avoid collinearity
    patt_dummies = pd.get_dummies(df['Pattern'], prefix='Pattern', drop_first=True)
    X = pd.concat([patt_dummies, df[['SOA_c']].reset_index(drop=True)], axis=1)
    # Get interaction terms manually: Pattern_x * SOA_c for non-baseline patterns
    for p in list(X.columns):
        if p.startswith('Pattern_'):
            pname = p.split('Pattern_')[1]
            inter_col = f'inter_{pname}_SOA_c'
            X[inter_col] = (X[p].astype(float) * df['SOA_c'].astype(float)).astype(float)
    # Add intercept if not present
    # Ensure float dtype
    X = X.astype(float)
    if 'const' not in X.columns:
        X = sm.add_constant(X, has_constant='add')
    y = df['hit'].astype(int)
    clusters = df['Pattern']
    return X, y, clusters, baseline

def fit_logit(X, y, clusters):
    model = sm.Logit(y, X)
    # statsmodels Logit fit can accept cov_type & cov_kwds for clustered SEs
    res = model.fit(disp=False, cov_type='cluster', cov_kwds={'groups':clusters})
    return res

def marginal_threshold(res, baseline: str, patterns: list, soas: np.ndarray):
    # For each pattern, compute predicted accuracy over SOAs and find where it crosses 0.5 by linear interpolation.
    th_rows = []
    for pat in patterns:
        params = res.params.copy()
        is_base = (pat == baseline)
        pat_add = 0.0 if is_base else params.get(f'Pattern_{pat}', 0.0)
        base_slope = params.get('SOA_c', 0.0)
        inter_term = 0.0 if is_base else params.get(f'inter_{pat}_SOA_c', 0.0)
        preds = []
        for s in soas:
            s_c = s - soas.mean()
            slope = base_slope + inter_term
            logit = params.get('const', 0.0) + pat_add + slope * s_c
            p = 1/(1+np.exp(-logit))
            preds.append((s,p))
        # Find threshold
        threshold = np.nan
        for (s1,p1),(s2,p2) in zip(preds[:-1], preds[1:]):
            if (p1-0.5)*(p2-0.5) <= 0 and p1 != p2:
                t = (0.5 - p1)/(p2 - p1)
                threshold = s1 + t*(s2 - s1)
                break
        th_rows.append({'Pattern':pat,'ModelThreshold':threshold})
    return pd.DataFrame(th_rows)

def main():
    df = load_trials()
    X, y, clusters, baseline = prepare_design(df)
    res = fit_logit(X, y, clusters)
    print('Baseline pattern:', baseline)
    print('\nCluster-robust coefficient table:')
    summ = res.summary2().tables[1]
    print(summ.to_string())
    pats = list(X.filter(regex='Pattern_').columns)
    # Derive pattern list from categorical order
    patterns = [baseline] + [c.split('Pattern_')[1] for c in pats if c != f'Pattern_{baseline}']
    soas = np.array(sorted(df['SOA'].unique()), dtype=float)
    th_df = marginal_threshold(res, baseline, patterns, soas)
    th_df.to_csv('runs/pattern_logit_model_thresholds.csv', index=False)
    print('\nModel-based threshold estimates written to runs/pattern_logit_model_thresholds.csv')
    print(th_df.to_string(index=False))

if __name__ == '__main__':
    main()
