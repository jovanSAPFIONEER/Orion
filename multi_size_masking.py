#!/usr/bin/env python3
"""
multi_size_masking.py

Run masking paradigm across multiple network sizes and SOAs with replicated trials,
computing accuracy, confidence, ignition latency summaries, and estimating a
threshold SOA (linear interpolation where accuracy crosses 0.5).

Outputs:
  - CSV with per-size, per-SOA metrics (accuracy + Wilson CI, confidence mean/CI, ignition latency mean/CI)
  - CSV with per-size threshold summary
  - PNG/PDF figure of masking curves across sizes

Example quick run:
  python multi_size_masking.py --out runs/multi_size_quick --sizes 32 64 --soas 1 2 3 4 6 8 --trials 25

Heavy run (more trials):
  python multi_size_masking.py --out runs/multi_size_full --sizes 32 64 128 256 --trials 120
"""

import os, sys, argparse, json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from overnight_full_run import (
    make_small_world_W, GlobalWorkspace, tanh, masking_specs,
    LABEL_TO_ID, VOCAB, bootstrap_mean_ci, wilson_interval,
    reliability_bins, ece
)

# ---- Connectivity pattern variants ----
def make_connectivity(N: int, pattern: str, base_coupling: float, seed: int,
                      degree_frac: float = 0.38, rewire_p: float = 0.18,
                      modules: int = 4, inter_frac: float = 0.05) -> np.ndarray:
    """Factory for different structural patterns to test communication hypothesis.

    Patterns:
      small_world (default): existing generator
      modular: dense intra-module, sparse inter-module gaussian weights
      random: iid gaussian normalized columns
      line: 1-D chain with bidirectional nearest-neighbor
    """
    rng = np.random.default_rng(seed)
    if pattern == 'small_world':
        return make_small_world_W(N, base_coupling, degree_frac, rewire_p, seed)
    if pattern == 'random':
        W = rng.normal(0.0, 1.0, size=(N, N))
        cn = np.linalg.norm(W, axis=0)+1e-8
        return (W/cn)*base_coupling
    if pattern == 'line':
        W = np.zeros((N,N))
        for i in range(N):
            if i>0: W[i-1,i] = rng.normal(0,1)
            if i<N-1: W[i+1,i] = rng.normal(0,1)
        cn = np.linalg.norm(W, axis=0)+1e-8
        return (W/cn)*base_coupling
    if pattern == 'modular':
        m = max(1, modules)
        sizes = [N//m]*m
        for i in range(N - sum(sizes)): sizes[i]+=1
        idxs=[]; s=0
        for sz in sizes:
            idxs.append(list(range(s, s+sz))); s+=sz
        W = np.zeros((N,N))
        # intra-module dense
        for ids in idxs:
            for j in ids:
                W[np.ix_(ids,[j])] = rng.normal(0,1,size=(len(ids),1))
        # sparse inter-module
        for a in range(m):
            for b in range(m):
                if a==b: continue
                if rng.random() < inter_frac:
                    src = rng.choice(idxs[b]); dst = rng.choice(idxs[a])
                    W[dst, src] = rng.normal(0,1)
        cn = np.linalg.norm(W, axis=0)+1e-8
        return (W/cn)*base_coupling
    # fallback
    return make_small_world_W(N, base_coupling, degree_frac, rewire_p, seed)

def _temperature_scale(conf: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to confidence values interpreted as probabilities.
    We convert to logits, divide by T, then back to probs. Clamp to avoid 0/1 extremes.
    """
    eps = 1e-6
    conf_c = np.clip(conf, eps, 1 - eps)
    logits = np.log(conf_c) - np.log(1 - conf_c)
    scaled = 1 / (1 + np.exp(-logits / T))
    return np.clip(scaled, 0.0, 1.0)

def optimize_temperature(conf: np.ndarray, hit: np.ndarray, grid: Tuple[float,float,int]=(0.3, 3.5, 140)) -> float:
    """Find temperature minimizing negative log-likelihood for binary correctness.
    Returns best temperature; falls back to 1.0 if degenerate."""
    if len(conf) == 0 or np.all(hit==hit[0]):  # no variation
        return 1.0
    lo, hi, K = grid
    temps = np.linspace(lo, hi, K)
    best_T = 1.0
    best_nll = float('inf')
    eps = 1e-9
    for T in temps:
        p = _temperature_scale(conf, T)
        nll = -np.sum(hit * np.log(p+eps) + (1-hit)*np.log(1-p+eps))
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)

def run_trial_custom(N: int, T: int, burn: int, seed: int, specs, noise: float = 0.10,
                     pattern: str = 'small_world', modules: int = 4, inter_frac: float = 0.05,
                     gw_tau: float = 0.9, gw_theta: float = 0.55,
                     degree_frac: float = 0.38, rewire_p: float = 0.18):
    """Run a single masking trial with custom network size N."""
    r = np.random.default_rng(seed); g = 0.9
    W = make_connectivity(N, pattern, base_coupling=1.0, seed=seed,
                          degree_frac=degree_frac, rewire_p=rewire_p,
                          modules=modules, inter_frac=inter_frac)
    Win_v = r.normal(0.0, 1.0, size=N)
    Win_a = r.normal(0.0, 1.0, size=N)
    Win_c = r.normal(0.0, 1.0, size=N)
    proj_R = r.normal(0.0, 1.0, size=(N, 6)); proj_R /= (np.linalg.norm(proj_R, axis=0, keepdims=True)+1e-8)
    x = np.zeros((N, T)); x[:,0] = r.normal(0, 0.1, size=N)
    gw = GlobalWorkspace(N=N, K=len(VOCAB), tau=gw_tau, theta=gw_theta, seed=seed+1)
    if specs is None:
        inputs = {"V":np.zeros(T),"A":np.zeros(T),"C":np.zeros(T),"y":np.zeros(T,dtype=int)}
    else:
        from overnight_full_run import make_env_inputs
        inputs = make_env_inputs(T, specs)
    y = inputs["y"]
    tokens=[]; igns=[]; probs=[]; labels=[]; pmax_all=[]
    for t in range(T-1):
        z_int = g * (W @ x[:, t])
        z_ext = 0.24*(Win_v*inputs["V"][t] + Win_a*inputs["A"][t] + Win_c*inputs["C"][t]) + \
                noise*np.random.default_rng(900+t).normal(0,1,size=N)
        x[:, t+1] = tanh(z_int + z_ext)
        bcast, p, k, ign = gw.step(x[:, t+1])
        x[:, t+1] = tanh(x[:, t+1] + 0.22*bcast)
        # record per-step after burn slicing later
        tokens.append(k); igns.append(float(ign)); probs.append(float(p[k])); labels.append(int(y[t])); pmax_all.append(float(p.max()))
    sl = slice(burn, T-1)
    return {
        "tokens": np.array(tokens[sl], dtype=int),
        "probs": np.array(probs[sl], dtype=float),
        "ignitions": np.array(igns[sl], dtype=float),
        "labels": np.array(labels[sl], dtype=int),
        "pmax": np.array(pmax_all[sl], dtype=float),
    }

def estimate_threshold(soas: List[int], accs: List[float], target: float = 0.5) -> float:
    xs = np.array(soas, dtype=float); ys = np.array(accs, dtype=float)
    order = np.argsort(xs); xs = xs[order]; ys = ys[order]
    for i in range(len(xs)-1):
        y1, y2 = ys[i], ys[i+1]
        if (y1 - target) * (y2 - target) <= 0 and y1 != y2:
            t = (target - y1) / (y2 - y1)
            return float(xs[i] + t*(xs[i+1]-xs[i]))
        if y1 == target:
            return float(xs[i])
    if ys[-1] == target:
        return float(xs[-1])
    return float('nan')

def fit_logistic_curve(soas: List[int], accs: List[float], th_points: int = 120, slope_points: int = 50,
                       slope_min_div: float = 50.0, slope_max_factor: float = 0.333):
    """Grid search 2-parameter logistic p=1/(1+exp(-(x-th)/s)). Returns th, s, preds.
    slope_min_div: lower slope bound = span / slope_min_div
    slope_max_factor: upper slope bound = span * slope_max_factor
    Increase slope_max_factor (>0.333) to allow steeper curves; decrease slope_min_div for shallower lower bound."""
    xs = np.array(soas, dtype=float); ys = np.array(accs, dtype=float)
    if len(xs) < 3 or np.any(np.isnan(ys)):
        return float('nan'), float('nan'), ys
    th_grid = np.linspace(xs.min(), xs.max(), th_points)
    span = xs.max() - xs.min() + 1e-6
    s_lo = span / max(1.0, slope_min_div)
    s_hi = span * max(1e-6, slope_max_factor)
    if s_hi <= s_lo:  # fallback
        s_hi = span / 3.0
    s_grid = np.linspace(s_lo, s_hi, slope_points)
    # Vectorized evaluation: for each s, compute preds for all th via broadcasting
    best_sse = float('inf'); best_th = float('nan'); best_s = float('nan'); best_preds = None
    for s in s_grid:
        z_all = xs[None,:] - th_grid[:,None]  # th_grid x len(xs)
        preds_all = 1/(1+np.exp(-z_all/s))
        sse_all = np.sum((preds_all - ys)**2, axis=1)
        idx = int(np.argmin(sse_all))
        if sse_all[idx] < best_sse:
            best_sse = float(sse_all[idx]); best_th = float(th_grid[idx]); best_s = float(s); best_preds = preds_all[idx]
    return float(best_th), float(best_s), best_preds if best_preds is not None else ys

def analyze_sizes(sizes: List[int], soas: List[int], trials: int, seed0: int, outdir: str, boots: int, calibrate: bool=False,
                  amp_t: float = 0.9, amp_m: float = 2.0, ignition_thresh: float = 0.5, fast_boot: bool=False,
                  debug_ign: bool=False, debug_ign_limit: int=3,
                  noise_scale: float = 1.0, hit_window_radius: int = 5, lat_search_max: int = 22,
                  slope_min_div: float = 50.0, slope_max_factor: float = 0.333,
                  pattern: str = 'small_world', modules: int = 4, inter_frac: float = 0.05,
                  gw_tau: float = 0.9, gw_theta: float = 0.55,
                  record_module_sync: bool = True,
                  pmax_latency_percentile: float = 75.0, pmax_latency_min: float = 0.25,
                  backward_only: bool=False, mask_width: int=1, target_width: int=1,
                  degree_frac: float = 0.38, rewire_p: float = 0.18):
    rows: List[Dict[str, Any]] = []
    thresh_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    ign_debug_rows: List[Dict[str, Any]] = []
    for N in sizes:
        acc_curve: List[float] = []
        soas_sorted: List[int] = []
        size_conf_all: List[float] = []
        size_hit_all: List[int] = []
        module_sync_vals: List[float] = []
        for soa in soas:
            hits=[]; confs=[]; lats=[]
            for tr in range(trials):
                T=360; pos=180
                if backward_only:
                    # Single backward mask only
                    mpos = min(T-1, pos + max(0, soa))
                    specs = [
                        {"kind":"vis_tgt","t":pos,"width":target_width,"amp":amp_t},
                        {"kind":"mask","t":mpos,"width":mask_width,"amp":amp_m},
                    ]
                else:
                    specs = masking_specs(T, pos, soa, amp_t=amp_t, amp_m=amp_m)
                out = run_trial_custom(N, T=T, burn=80, seed=seed0+N*1000+soa*300+tr, specs=specs, noise=0.10*noise_scale,
                                       pattern=pattern, modules=modules, inter_frac=inter_frac,
                                       gw_tau=gw_tau, gw_theta=gw_theta,
                                       degree_frac=degree_frac, rewire_p=rewire_p)
                center = pos-80  # adjusted for burn already (burn=80)
                r = hit_window_radius
                hit = 1 if np.any(out["tokens"][max(0,center-r):center+r+1] == LABEL_TO_ID["VIS_TGT"]) else 0
                a = max(0, center-r); b = min(len(out["probs"])-1, center+r)
                conf = float(np.max(out["probs"][a:b+1])) if b>=a else 0.0
                lat=None
                # Binary ignition-based latency
                for dt in range(lat_search_max):
                    idx=center+dt
                    if 0 <= idx < len(out["ignitions"]) and out["ignitions"][idx] > ignition_thresh:
                        lat=dt; break
                # Dynamic latency via pmax percentile if still missing
                if lat is None and 'pmax' in out:
                    p_slice = out['pmax'][center:center+lat_search_max]
                    if len(p_slice) > 3:
                        dyn_thr = max(pmax_latency_min, float(np.percentile(out['pmax'], pmax_latency_percentile)))
                        for dt2, val in enumerate(p_slice):
                            if val >= dyn_thr:
                                lat = dt2; break
                if lat is not None:
                    lats.append(lat)
                hits.append(hit); confs.append(conf)
                size_conf_all.append(conf); size_hit_all.append(hit)
                row_base = {"N":N, "SOA":soa, "trial":tr, "hit":hit, "conf":conf, "lat":lat}
                # Module sync proxy: variance of token ids in window vs overall (lower variance may indicate dominance/broadcast)
                if record_module_sync:
                    # use token slice window
                    win_tokens = out["tokens"][max(0,center-5):min(len(out["tokens"]), center+6)]
                    if len(win_tokens)>1:
                        sync = 1.0 - (np.unique(win_tokens).size / len(win_tokens))
                    else:
                        sync = float('nan')
                    row_base['sync'] = sync
                    module_sync_vals.append(sync)
                trial_rows.append(row_base)
                if debug_ign and tr < debug_ign_limit:
                    # capture ignition trace window around center
                    w_lo = max(0, center-10); w_hi = min(len(out["ignitions"])-1, center+30)
                    for t_rel, idx in enumerate(range(w_lo, w_hi+1)):
                        ign_debug_rows.append({
                            "N":N, "SOA":soa, "trial":tr, "t_rel":idx-center, "ignition":float(out["ignitions"][idx])
                        })
            k = int(np.sum(hits)); n=len(hits); p=k/n; lo, hi = wilson_interval(k, n)
            mu_conf, conf_lo, conf_hi = bootstrap_mean_ci(confs, B=min(1200, max(400, 50*trials)), seed=seed0+N+soa)
            if len(lats)>0:
                mu_lat, lat_lo, lat_hi = bootstrap_mean_ci(lats, B=min(1200, max(400, 50*trials)), seed=seed0+N+soa+77)
            else:
                mu_lat=lat_lo=lat_hi=float('nan')
            rows.append({
                "N":N, "SOA":soa, "Acc":p, "Acc_lo":lo, "Acc_hi":hi,
                "Conf_mean":mu_conf, "Conf_lo":conf_lo, "Conf_hi":conf_hi,
                "IgnLat_mean":mu_lat, "IgnLat_lo":lat_lo, "IgnLat_hi":lat_hi,
                "Trials":n
            })
            acc_curve.append(p); soas_sorted.append(soa)
        # --- Per-size threshold estimation ---
        th_fit, slope_fit, preds = fit_logistic_curve(soas_sorted, acc_curve,
                                                     slope_min_div=slope_min_div,
                                                     slope_max_factor=slope_max_factor)
        # Guard: if all accs above 0.5, set threshold to max SOA (mark unbracketed)
        if len(acc_curve)>0 and all(a>=0.5 for a in acc_curve):
            th_fit = float(soas_sorted[-1])
        boot_th: List[float] = []
        rng = np.random.default_rng(seed0+N+999)
        if boots > 0 and not np.isnan(th_fit):
            per_soa_hits = {s: [] for s in soas_sorted}
            for trr in trial_rows:
                if trr["N"]==N:
                    per_soa_hits[int(trr["SOA"])].append(trr["hit"])
            for _ in range(min(boots, 400)):
                accs_boot = []
                for s in soas_sorted:
                    hlist = per_soa_hits[s]
                    if len(hlist)==0:
                        accs_boot.append(np.nan)
                    else:
                        idx = rng.integers(0, len(hlist), size=len(hlist))
                        accs_boot.append(float(np.mean([hlist[i] for i in idx])))
                if fast_boot:
                    th_b = estimate_threshold(soas_sorted, accs_boot)
                else:
                    th_b, _, _ = fit_logistic_curve(soas_sorted, accs_boot, th_points=60, slope_points=30)
                if not np.isnan(th_b):
                    boot_th.append(th_b)
        if len(boot_th)>5:
            th_lo = float(np.percentile(boot_th, 2.5)); th_hi = float(np.percentile(boot_th,97.5))
        else:
            th_lo=th_hi=float('nan')
        th_linear = estimate_threshold(soas_sorted, acc_curve) if len(soas_sorted) >= 2 else float('nan')
        if len(soas_sorted) < 3 and np.isnan(th_fit):
            slope_fit = float('nan')
        if len(module_sync_vals)==0 and record_module_sync:
            # recover sync from trial_rows if missing
            extracted=[trr.get('sync') for trr in trial_rows if trr['N']==N and trr.get('sync') is not None]
            if extracted:
                module_sync_vals = extracted
        thresh_rows.append({"N":N, "Threshold50_linear":th_linear,
                             "Threshold50_logistic":th_fit, "Threshold50_lo":th_lo, "Threshold50_hi":th_hi,
                             "LogisticSlope":slope_fit,
                             "Pattern":pattern, "Modules":modules, "InterFrac":inter_frac,
                             "GW_tau":gw_tau, "GW_theta":gw_theta,
                             "DegreeFrac":degree_frac, "RewireP":rewire_p,
                             "Sync_mean": (float(np.nanmean(module_sync_vals)) if len(module_sync_vals)>0 else float('nan'))})
        if len(size_conf_all)>0:
            conf_arr = np.array(size_conf_all); hit_arr = np.array(size_hit_all)
            mids, accs, counts = reliability_bins(conf_arr, hit_arr, M=10)
            size_ece = ece(conf_arr, hit_arr, M=10)
            row = {"N":N, "ECE":float(size_ece), "bins_mids":mids.tolist(),
                   "bins_accs":np.nan_to_num(accs).tolist(), "bins_counts":counts.tolist()}
            if calibrate:
                Topt = optimize_temperature(conf_arr, hit_arr)
                conf_cal = _temperature_scale(conf_arr, Topt)
                mids2, accs2, counts2 = reliability_bins(conf_cal, hit_arr, M=10)
                size_ece2 = ece(conf_cal, hit_arr, M=10)
                row.update({"Temp":Topt, "ECE_cal":float(size_ece2),
                            "bins_mids_cal":mids2.tolist(),
                            "bins_accs_cal":np.nan_to_num(accs2).tolist(),
                            "bins_counts_cal":counts2.tolist()})
                idx_c = 0
                for trr in trial_rows:
                    if trr["N"]==N:
                        tr_conf = conf_arr[idx_c]
                        trr["conf_cal"] = float(_temperature_scale(np.array([tr_conf]), Topt)[0])
                        idx_c += 1
            calib_rows.append(row)
    df = pd.DataFrame(rows); df_thr = pd.DataFrame(thresh_rows); df_trials = pd.DataFrame(trial_rows); df_cal = pd.DataFrame(calib_rows)
    if debug_ign and len(ign_debug_rows)>0:
        df_ign = pd.DataFrame(ign_debug_rows)
    else:
        df_ign = pd.DataFrame()
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "multi_size_masking_detailed.csv"), index=False)
    df_thr.to_csv(os.path.join(outdir, "multi_size_thresholds.csv"), index=False)
    df_trials.to_csv(os.path.join(outdir, "multi_size_trial_level.csv"), index=False)
    df_cal.to_json(os.path.join(outdir, "multi_size_calibration.json"), orient="records", indent=2)
    if debug_ign and len(df_ign)>0:
        df_ign.to_csv(os.path.join(outdir, "multi_size_ignition_debug.csv"), index=False)
    return df, df_thr, df_trials, df_cal

def sweep_amplitudes(size: int, soas: List[int], amp_t_list: List[float], amp_m_list: List[float], trials: int, seed0: int,
                     outdir: str, ignition_thresh: float=0.5, noise_scale: float = 1.0, hit_window_radius: int = 5) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for at in amp_t_list:
        for am in amp_m_list:
            for soa in soas:
                hits=[]
                for tr in range(trials):
                    T=360; pos=180
                    specs = masking_specs(T, pos, soa, amp_t=at, amp_m=am)
                    out = run_trial_custom(size, T=T, burn=80, seed=seed0+int(size*1000+soa*300+tr+abs(hash((at,am)))%10000), specs=specs, noise=0.10*noise_scale)
                    center = pos-80
                    r = hit_window_radius
                    hit = 1 if np.any(out["tokens"][max(0,center-r):center+r+1] == LABEL_TO_ID["VIS_TGT"]) else 0
                    hits.append(hit)
                k = int(np.sum(hits)); n=len(hits); p = k/n; lo, hi = wilson_interval(k,n)
                rows.append({"N":size, "SOA":soa, "amp_t":at, "amp_m":am, "Acc":p, "Acc_lo":lo, "Acc_hi":hi, "Trials":n})
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "amplitude_sweep.csv"), index=False)
    return df

def plot_curves(df: pd.DataFrame, sizes: List[int], outdir: str):
    plt.figure(figsize=(10,6))
    colors = plt.cm.viridis(np.linspace(0,1,len(sizes)))
    for c, N in zip(colors, sizes):
        sub = df[df.N==N].sort_values("SOA")
        if len(sub)==0: continue
        plt.plot(sub.SOA, sub.Acc, marker='o', color=c, label=f"N={N}")
        plt.fill_between(sub.SOA, sub.Acc_lo, sub.Acc_hi, color=c, alpha=0.18)
    plt.xlabel("SOA"); plt.ylabel("Report Accuracy")
    plt.title("Masking Curves Across Network Sizes")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    png = os.path.join(outdir, "multi_size_masking_curves.png")
    pdf = os.path.join(outdir, "multi_size_masking_curves.pdf")
    plt.savefig(png, dpi=300); plt.savefig(pdf)
    plt.close()
    return png, pdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=str, help='Output directory')
    ap.add_argument('--sizes', nargs='+', type=int, default=[32,64,128,256], help='Network sizes')
    ap.add_argument('--soas', nargs='+', type=int, default=[0,1,2,3,4,6,8,10,12], help='SOA values (include 0 for simultaneous)')
    ap.add_argument('--trials', type=int, default=40, help='Trials per (size, SOA)')
    ap.add_argument('--seed', type=int, default=12000, help='Base seed')
    ap.add_argument('--boots', type=int, default=200, help='Bootstrap iterations for threshold CI (per size)')
    ap.add_argument('--calibrate', action='store_true', help='Apply per-size temperature scaling to confidence and report post-calibration ECE')
    ap.add_argument('--amp_t', type=float, default=0.9, help='Target amplitude')
    ap.add_argument('--amp_m', type=float, default=2.0, help='Mask amplitude')
    ap.add_argument('--ignition_thresh', type=float, default=0.5, help='Ignition threshold for latency detection')
    ap.add_argument('--meta', action='store_true', help='Write metadata JSON')
    ap.add_argument('--fast_boot', action='store_true', help='Use faster linear threshold only inside bootstraps')
    ap.add_argument('--debug_ign', action='store_true', help='Dump ignition window samples to CSV')
    ap.add_argument('--debug_ign_limit', type=int, default=3, help='Trials per (size,SOA) to dump ignition debug')
    ap.add_argument('--sweep_size', type=int, help='Run amplitude sweep for a single size instead of full analysis')
    ap.add_argument('--sweep_amp_t', nargs='+', type=float, help='Target amplitude values to sweep')
    ap.add_argument('--sweep_amp_m', nargs='+', type=float, help='Mask amplitude values to sweep')
    ap.add_argument('--sweep_trials', type=int, default=12, help='Trials per amplitude pair (SOA) during sweep')
    ap.add_argument('--noise_scale', type=float, default=1.0, help='Multiplier for base external noise std (0.10 * scale)')
    ap.add_argument('--hit_window_radius', type=int, default=5, help='Half-width of temporal window (in steps) to count a hit')
    ap.add_argument('--lat_search_max', type=int, default=22, help='Max steps after target onset to search for ignition latency')
    ap.add_argument('--sweep_only', action='store_true', help='Run only the amplitude sweep (skip full size/SOA analysis)')
    ap.add_argument('--auto_from_sweep', action='store_true', help='After sweep, pick amplitude pair with Acc just below 0.5 (closest) and reuse for full analysis')
    ap.add_argument('--slope_min_div', type=float, default=50.0, help='Lower logistic slope bound = span / slope_min_div')
    ap.add_argument('--slope_max_factor', type=float, default=0.6, help='Upper logistic slope bound = span * slope_max_factor')
    ap.add_argument('--temp_lo', type=float, default=0.3, help='Temperature grid lower bound')
    ap.add_argument('--temp_hi', type=float, default=6.0, help='Temperature grid upper bound')
    ap.add_argument('--temp_points', type=int, default=160, help='Temperature grid points')
    ap.add_argument('--pattern', type=str, default='small_world', choices=['small_world','modular','random','line'], help='Connectivity pattern')
    ap.add_argument('--modules', type=int, default=4, help='Number of modules (modular pattern)')
    ap.add_argument('--inter_frac', type=float, default=0.05, help='Inter-module connection fraction (modular pattern)')
    ap.add_argument('--gw_tau', type=float, default=0.9, help='GW competition temperature')
    ap.add_argument('--gw_theta', type=float, default=0.55, help='GW ignition probability threshold')
    ap.add_argument('--pmax_latency_percentile', type=float, default=75.0, help='Percentile of pmax distribution for dynamic latency threshold')
    ap.add_argument('--pmax_latency_min', type=float, default=0.25, help='Minimum absolute pmax threshold for latency')
    ap.add_argument('--backward_only', action='store_true', help='Use single backward mask only (no forward mask)')
    ap.add_argument('--mask_width', type=int, default=1, help='Mask temporal width (steps) when using backward_only')
    ap.add_argument('--target_width', type=int, default=1, help='Target temporal width (steps)')
    ap.add_argument('--degree_frac', type=float, default=0.38, help='Degree fraction for small_world connectivity')
    ap.add_argument('--rewire_p', type=float, default=0.18, help='Rewire probability for small_world connectivity')
    args = ap.parse_args()

    # Amplitude sweep mode
    if args.sweep_size and args.sweep_amp_t and args.sweep_amp_m:
        print("Running amplitude sweep...")
        sweep_df = sweep_amplitudes(args.sweep_size, args.soas, args.sweep_amp_t, args.sweep_amp_m,
                                    args.sweep_trials, args.seed, args.out, ignition_thresh=args.ignition_thresh,
                                    noise_scale=args.noise_scale, hit_window_radius=args.hit_window_radius)
        print("Sweep saved:", os.path.join(args.out, "amplitude_sweep.csv"))
        if args.auto_from_sweep:
            # Prefer rows with Acc < 0.5 but closest to 0.5; else closest overall
            sub = sweep_df.copy()
            sub['dist'] = np.abs(sub['Acc'] - 0.5)
            below = sub[sub['Acc'] < 0.5]
            if len(below) > 0:
                pick = below.iloc[below['dist'].argmin()]
            else:
                pick = sub.iloc[sub['dist'].argmin()]
            args.amp_t = float(pick['amp_t']); args.amp_m = float(pick['amp_m'])
            print(f"Auto-selected amplitude pair from sweep: amp_t={args.amp_t} amp_m={args.amp_m} (Acc={pick['Acc']:.3f} at SOA={pick['SOA']})")
        if args.sweep_only:
            return

    # Extend temperature scaling grid dynamically
    orig_opt_temp = optimize_temperature
    def opt_temp_wrapper(conf_arr, hit_arr):
        return orig_opt_temp(conf_arr, hit_arr, (args.temp_lo, args.temp_hi, args.temp_points))
    globals()['optimize_temperature'] = opt_temp_wrapper  # monkey patch inside analyze if used
    df, df_thr, df_trials, df_cal = analyze_sizes(
        args.sizes, args.soas, args.trials, args.seed, args.out, args.boots,
        calibrate=args.calibrate, amp_t=args.amp_t, amp_m=args.amp_m,
        ignition_thresh=args.ignition_thresh, fast_boot=args.fast_boot,
        debug_ign=args.debug_ign, debug_ign_limit=args.debug_ign_limit,
        noise_scale=args.noise_scale, hit_window_radius=args.hit_window_radius,
        lat_search_max=args.lat_search_max, slope_min_div=args.slope_min_div,
        slope_max_factor=args.slope_max_factor, pattern=args.pattern, modules=args.modules,
        inter_frac=args.inter_frac, gw_tau=args.gw_tau, gw_theta=args.gw_theta,
        pmax_latency_percentile=args.pmax_latency_percentile, pmax_latency_min=args.pmax_latency_min,
        backward_only=args.backward_only, mask_width=args.mask_width, target_width=args.target_width,
        degree_frac=args.degree_frac, rewire_p=args.rewire_p)
    globals()['optimize_temperature'] = orig_opt_temp
    png, pdf = plot_curves(df, args.sizes, args.out)

    if args.meta:
        with open(os.path.join(args.out, 'meta.json'), 'w') as f:
            json.dump({"sizes":args.sizes, "soas":args.soas, "trials":args.trials, "seed":args.seed}, f, indent=2)

    print("Saved detailed metrics:", os.path.join(args.out, "multi_size_masking_detailed.csv"))
    print("Saved thresholds:", os.path.join(args.out, "multi_size_thresholds.csv"))
    print("Saved figures:", png, pdf)
    print("Threshold estimates (logistic & linear):")
    print(df_thr.to_string(index=False))
    print("Calibration JSON entries:", len(df_cal))
    if args.calibrate and 'ECE_cal' in df_cal.columns:
        print("Per-size calibration improvement (ECE -> ECE_cal):")
        for _, r in df_cal.iterrows():
            if not np.isnan(r.get('ECE_cal', np.nan)):
                print(f" N={int(r.N)} Temp={r.get('Temp', float('nan')):.3f} ECE {r.ECE:.4f} -> {r.ECE_cal:.4f}")
    # Ignition latency correlations near threshold
    try:
        if 'IgnLat_mean' in df.columns and 'Threshold50_logistic' in df_thr.columns:
            merged = df.merge(df_thr[['N','Threshold50_logistic']], on='N', how='left')
            band_rows = []
            for N in merged.N.unique():
                sub = merged[merged.N==N]
                th = sub['Threshold50_logistic'].iloc[0]
                if np.isnan(th):
                    continue
                band = sub[np.abs(sub.SOA - th) <= 1.5]
                if len(band)==0:
                    continue
                w = band.Trials.fillna(0).values
                if w.sum()==0:
                    w = np.ones_like(w)
                lat_vals = band.IgnLat_mean.values
                if np.isfinite(lat_vals).any():
                    lat_near = float(np.nanmean(lat_vals))
                else:
                    lat_near = float('nan')
                band_rows.append({'N':int(N), 'Thresh':float(th), 'NearLat':lat_near})
            if len(band_rows) >= 2:
                corr_df = pd.DataFrame(band_rows)
                def pearson(a,b):
                    a=np.array(a,dtype=float); b=np.array(b,dtype=float)
                    m1=np.nanmean(a); m2=np.nanmean(b)
                    num=np.nansum((a-m1)*(b-m2)); den=np.sqrt(np.nansum((a-m1)**2)*np.nansum((b-m2)**2))+1e-12
                    return float(num/den)
                if corr_df['NearLat'].notna().sum()>=2:
                    c_thr_lat = pearson(corr_df['Thresh'], corr_df['NearLat'])
                    c_N_thr = pearson(corr_df['N'], corr_df['Thresh'])
                    c_N_lat = pearson(corr_df['N'], corr_df['NearLat'])
                    print(f"Ignition correlations: Thresh~Lat={c_thr_lat:.3f} N~Thresh={c_N_thr:.3f} N~Lat={c_N_lat:.3f}")
    except Exception as e:
        print('[Warn] Ignition correlation failed:', e)

if __name__ == '__main__':
    main()
