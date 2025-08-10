import os, glob, json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Reuse threshold fitting from masking module
from multi_size_masking import fit_logistic_curve, estimate_threshold

RUN_GLOB = 'runs/*_backward_harsh'
BOOT = 800  # number of bootstrap samples per pattern (can adjust for precision)
SLOPE_MIN_DIV = 50.0
SLOPE_MAX_FACTOR = 0.6
TARGET_ACC = 0.5
RANDOM_SEED = 7771

# Patterns we expect; order for comparisons
PATTERN_ORDER = [
    'small_world', 'line', 'modular', 'random'
]


def load_trial_data(path: str) -> pd.DataFrame:
    f = os.path.join(path, 'multi_size_trial_level.csv')
    if not os.path.isfile(f):
        raise FileNotFoundError(f)
    df = pd.read_csv(f)
    return df


def build_hits_dict(df: pd.DataFrame) -> Dict[int, List[int]]:
    d: Dict[int, List[int]] = {}
    for soa, sub in df.groupby('SOA'):
        d[int(soa)] = sub['hit'].astype(int).tolist()
    return d


def sample_threshold(hits: Dict[int, List[int]], rng: np.random.Generator, logistic: bool = True) -> float:
    soas = sorted(hits.keys())
    accs = []
    for s in soas:
        h = hits[s]
        if len(h) == 0:
            accs.append(np.nan)
        else:
            idx = rng.integers(0, len(h), size=len(h))
            accs.append(float(np.mean([h[i] for i in idx])))
    if logistic and len(soas) >= 3 and not np.all(np.isnan(accs)):
        th, _, _ = fit_logistic_curve(soas, accs, slope_min_div=SLOPE_MIN_DIV, slope_max_factor=SLOPE_MAX_FACTOR)
        # guard like original: if all accs >= target, set to max SOA
        if all(a >= TARGET_ACC for a in accs if not np.isnan(a)):
            th = float(soas[-1])
    else:
        th = estimate_threshold(soas, accs, target=TARGET_ACC)
    return float(th)


def bootstrap_thresholds(hits: Dict[int, List[int]], B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        vals.append(sample_threshold(hits, rng, logistic=True))
    return np.array(vals, dtype=float)


def main():
    # Locate pattern runs
    runs = {}
    for path in glob.glob(RUN_GLOB):
        thr_csv = os.path.join(path, 'multi_size_thresholds.csv')
        if not os.path.isfile(thr_csv):
            continue
        df_thr = pd.read_csv(thr_csv)
        if df_thr.empty:
            continue
        pattern = str(df_thr.loc[0, 'Pattern']) if 'Pattern' in df_thr.columns else os.path.basename(path)
        if pattern not in runs:
            runs[pattern] = path
    if len(runs) == 0:
        print('No runs found for glob', RUN_GLOB)
        return
    print('Found patterns:', ', '.join(sorted(runs.keys())))

    # Bootstrap each pattern
    all_boot = {}
    results_rows = []
    for pat, path in runs.items():
        df_trials = load_trial_data(path)
        hits = build_hits_dict(df_trials)
        boots = bootstrap_thresholds(hits, BOOT, RANDOM_SEED + hash(pat) % 10000)
        boots = boots[np.isfinite(boots)]
        if len(boots) == 0:
            continue
        all_boot[pat] = boots
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
        mean = float(np.mean(boots))
        results_rows.append({'Pattern': pat, 'BootMean': mean, 'BootLo': lo, 'BootHi': hi, 'Nboots': len(boots)})
    df_pat = pd.DataFrame(results_rows)
    df_pat.to_csv('runs/pattern_bootstrap_thresholds.csv', index=False)
    print('Wrote runs/pattern_bootstrap_thresholds.csv')

    # Pairwise differences A - B
    diff_rows = []
    pats = [p for p in PATTERN_ORDER if p in all_boot]
    for i in range(len(pats)):
        for j in range(i+1, len(pats)):
            a, b = pats[i], pats[j]
            A = all_boot[a]; B = all_boot[b]
            # Align lengths by random resample if needed
            m = min(len(A), len(B))
            if m == 0:
                continue
            Ai = np.random.default_rng(123).choice(A, size=m, replace=True)
            Bi = np.random.default_rng(456).choice(B, size=m, replace=True)
            D = Ai - Bi
            diff_rows.append({
                'A': a, 'B': b,
                'DiffMean': float(D.mean()),
                'DiffLo': float(np.percentile(D, 2.5)),
                'DiffHi': float(np.percentile(D, 97.5)),
                'Significant': bool((np.percentile(D,2.5) > 0) or (np.percentile(D,97.5) < 0))
            })
    df_diff = pd.DataFrame(diff_rows)
    df_diff.to_csv('runs/pattern_threshold_differences.csv', index=False)
    print('Wrote runs/pattern_threshold_differences.csv')

    # Print concise tables
    print('\nBootstrap threshold means (logistic):')
    print(df_pat.to_string(index=False))
    print('\nPairwise differences (A - B):')
    print(df_diff.to_string(index=False))

if __name__ == '__main__':
    main()
