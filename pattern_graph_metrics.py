import os, argparse, json, math
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from multi_size_masking import make_connectivity
import math
from numpy.linalg import eigvalsh

# ---- Advanced weighted metrics ----
def symmetrize_weights(W):  # type: ignore
    A = np.abs(W)
    return 0.5*(A + A.T)

def algebraic_connectivity(Ws) -> float:  # type: ignore
    n = Ws.shape[0]
    d = Ws.sum(axis=1)
    L = np.diag(d) - Ws
    vals = eigvalsh(L)
    if len(vals) < 2:
        return float('nan')
    return float(sorted(vals)[1])  # Fiedler value

def weighted_global_efficiency(Ws, eps: float = 1e-9) -> float:  # type: ignore
    # Treat larger weights as stronger/shorter; define distance = 1/(w+eps)
    n = Ws.shape[0]
    Dmat = np.full((n,n), np.inf)
    np.fill_diagonal(Dmat, 0.0)
    # Use Dijkstra over dense graph (O(n^3)) acceptable for modest n.
    invW = 1.0/(Ws + eps)
    for s in range(n):
        dist = Dmat[s]
        visited = np.zeros(n, dtype=bool)
        while True:
            # pick unvisited minimum
            idx = -1; best = np.inf
            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]; idx = i
            if idx == -1:
                break
            visited[idx] = True
            # relax
            for j in range(n):
                if visited[j]:
                    continue
                w = Ws[idx,j]
                if w <= 0:  # no direct edge
                    continue
                nd = dist[idx] + invW[idx,j]
                if nd < dist[j]:
                    dist[j] = nd
    eff_sum = 0.0; cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            d = Dmat[i,j]
            if np.isfinite(d) and d > 0:
                eff_sum += 1.0/d
                cnt += 1
    return float(eff_sum/cnt) if cnt>0 else float('nan')

def communicability_mean(Ws, order: int = 10) -> float:  # type: ignore
    # Approximate exp(Ws) via truncated series sum_{k=0..order} Ws^k / k!
    n = Ws.shape[0]
    acc = np.eye(n)
    term = np.eye(n)
    for k in range(1, order+1):
        term = term @ Ws / k
        acc += term
    # communicability centrality per node = sum_j acc_ij
    comm = acc.sum(axis=1)
    return float(np.mean(comm))

def long_range_fraction(W) -> float:  # type: ignore
    # Heuristic: treat indices as ring; an edge is long-range if circular distance > N/6
    n = W.shape[0]
    A = np.abs(W)
    lr = 0; tot = 0
    limit = n/6.0
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if A[i,j] != 0:
                d = abs(i-j)
                d = min(d, n-d)  # ring distance
                tot += 1
                if d > limit:
                    lr += 1
    return float(lr/max(1,tot))

# ---- Graph metric helpers (undirected simplification) ----

def _binarize(W, density: float = 0.12):
    A = np.abs(W).copy()
    n = A.shape[0]
    # Choose threshold so that roughly desired density retained (excluding self)
    flat = A[~np.eye(n, dtype=bool)]
    k = int(max(1, min(len(flat)-1, round(density * len(flat)))))
    thr = np.partition(flat, -k)[-k]
    B = (A >= thr).astype(int)
    np.fill_diagonal(B, 0)
    # Symmetrize (undirected approximation)
    B = ((B + B.T) > 0).astype(int)
    return B

def clustering_coeff(B: np.ndarray) -> float:
    n = B.shape[0]
    tri = 0
    closed = 0
    for i in range(n):
        nbrs = np.where(B[i] > 0)[0]
        k = len(nbrs)
        if k < 2:
            continue
        # count edges among neighbors
        sub = B[np.ix_(nbrs, nbrs)]
        e = (np.sum(sub) - np.trace(sub)) // 2
        closed += e
        tri += k*(k-1)/2
    return float(closed/tri) if tri > 0 else 0.0

def shortest_path_metrics(B: np.ndarray) -> Dict[str, float]:
    n = B.shape[0]
    # BFS from each node (unweighted)
    dsum = 0.0
    cnt = 0
    eff_sum = 0.0
    for s in range(n):
        dist = -np.ones(n, dtype=int)
        dist[s] = 0
        q = [s]
        for v in q:
            for w in np.where(B[v] > 0)[0]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
        for t in range(s+1, n):
            d = dist[t]
            if d > 0:
                dsum += d
                eff_sum += 1.0/d
                cnt += 1
    char_path = dsum/cnt if cnt>0 else float('nan')
    glob_eff = eff_sum/cnt if cnt>0 else float('nan')
    return {"CharPath": float(char_path), "GlobalEff": float(glob_eff)}

def degree_stats(B: np.ndarray) -> Dict[str, float]:
    deg = B.sum(axis=0)
    return {"MeanDeg": float(np.mean(deg)), "DegStd": float(np.std(deg))}

# ---- Main metric computation ----

def compute_metrics(size: int, pattern: str, seed: int, degree_frac: float, rewire_p: float, modules: int, inter_frac: float, density: float) -> Dict[str, Any]:
    W = make_connectivity(size, pattern, base_coupling=1.0, seed=seed,
                          degree_frac=degree_frac, rewire_p=rewire_p,
                          modules=modules, inter_frac=inter_frac)
    B = _binarize(W, density=density)
    cc = clustering_coeff(B)
    sp = shortest_path_metrics(B)
    ds = degree_stats(B)
    Ws = symmetrize_weights(W)
    fied = algebraic_connectivity(Ws)
    w_eff = weighted_global_efficiency(Ws)
    comm = communicability_mean(Ws)
    lrf = long_range_fraction(W)
    row = {
        'N': size,
        'Pattern': pattern,
        'Seed': seed,
        'Clustering': cc,
        **sp,
        **ds,
        'EdgeDensity': float(B.sum() / (size*(size-1))),
        'Fiedler': fied,
        'WeightedEff': w_eff,
        'CommMean': comm,
        'LongRangeFrac': lrf
    }
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', nargs='+', type=int, default=[128], help='Network sizes')
    ap.add_argument('--patterns', nargs='+', type=str, default=['small_world','line','modular','random'], help='Patterns to evaluate')
    ap.add_argument('--seeds', nargs='+', type=int, default=[12000], help='Base seeds for connectivity sampling')
    ap.add_argument('--degree_frac', type=float, default=0.38, help='Degree fraction for small_world generator')
    ap.add_argument('--rewire_p', type=float, default=0.18, help='Rewire probability for small_world generator')
    ap.add_argument('--modules', type=int, default=4, help='Modules for modular pattern')
    ap.add_argument('--inter_frac', type=float, default=0.05, help='Inter-module fraction')
    ap.add_argument('--density', type=float, default=0.12, help='Target edge density for binarization')
    ap.add_argument('--out', type=str, default='runs/pattern_structure_metrics.csv', help='Output CSV file')
    ap.add_argument('--merge_thresholds', action='store_true', help='If set, merge with existing threshold CSVs from runs/*_backward_harsh')
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for N in args.sizes:
        for pat in args.patterns:
            for s in args.seeds:
                try:
                    rows.append(compute_metrics(N, pat, s, args.degree_frac, args.rewire_p, args.modules, args.inter_frac, args.density))
                except Exception as e:
                    print(f'[warn] metric failed N={N} pattern={pat} seed={s}: {e}')
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print('Wrote', args.out)

    if args.merge_thresholds:
        # Collect thresholds
        thr_rows = []
        for run_dir in os.listdir('runs'):
            if not run_dir.endswith('_backward_harsh'):
                continue
            thr_path = os.path.join('runs', run_dir, 'multi_size_thresholds.csv')
            if os.path.isfile(thr_path):
                try:
                    dthr = pd.read_csv(thr_path)
                    if not dthr.empty:
                        thr_rows.append(dthr.iloc[0])
                except Exception as e:
                    print('[warn] threshold read failed', run_dir, e)
        if thr_rows:
            df_thr = pd.DataFrame(thr_rows)
            # Merge on Pattern & N
            merged = pd.merge(df, df_thr[['Pattern','N','Threshold50_logistic','Threshold50_lo','Threshold50_hi']], on=['Pattern','N'], how='left')
            merged_out = os.path.join('runs', 'pattern_metrics_with_thresholds.csv')
            merged.to_csv(merged_out, index=False)
            print('Wrote', merged_out)
            # Quick correlation output
            if 'Threshold50_logistic' in merged.columns:
                valid = merged.dropna(subset=['Threshold50_logistic'])
                if len(valid) >= 3:
                    def corr(a,b):
                        a=np.array(a,dtype=float); b=np.array(b,dtype=float)
                        am=a.mean(); bm=b.mean()
                        num=((a-am)*(b-bm)).sum(); den=math.sqrt(((a-am)**2).sum()*((b-bm)**2).sum())+1e-12
                        return num/den
                    print('Corr Threshold~Clustering     :', corr(valid['Threshold50_logistic'], valid['Clustering']))
                    print('Corr Threshold~CharPath       :', corr(valid['Threshold50_logistic'], valid['CharPath']))
                    print('Corr Threshold~GlobalEff(bin) :', corr(valid['Threshold50_logistic'], valid['GlobalEff']))
                    if 'Fiedler' in valid.columns:
                        print('Corr Threshold~Fiedler        :', corr(valid['Threshold50_logistic'], valid['Fiedler']))
                    if 'WeightedEff' in valid.columns:
                        print('Corr Threshold~WeightedEff    :', corr(valid['Threshold50_logistic'], valid['WeightedEff']))
                    if 'CommMean' in valid.columns:
                        print('Corr Threshold~CommMean       :', corr(valid['Threshold50_logistic'], valid['CommMean']))
                    if 'LongRangeFrac' in valid.columns:
                        print('Corr Threshold~LongRangeFrac  :', corr(valid['Threshold50_logistic'], valid['LongRangeFrac']))

if __name__ == '__main__':
    main()
