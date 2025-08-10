#!/usr/bin/env python3
"""
overnight_full_run.py

High-powered (overnight) run for GW awareness tasks + no-report metrics.
Produces: CSVs, PDF report, calibration (reliability diagrams + ECE per SOA), and logs.

This module implements the core Global Workspace (GW) consciousness model with:
- Small-world network dynamics for neural simulation
- Global Workspace Theory implementation with ignition dynamics
- Four experimental paradigms: masking, attentional blink, change blindness, dual-task
- Comprehensive analysis including ROC curves, calibration metrics, and bootstrap CIs

Usage (examples):
  python overnight_full_run.py --out ./runs/full_01 --full
  nohup python overnight_full_run.py --out ./runs/full_overnight --n_mask 160 --n_blink 120 --n_cb 96 --n_dual 120 --boots 1500 > full.log 2>&1 &

Notes:
- CPU-only (numpy). Parallelization optional (per-SOA) via simple chunking; default is sequential for portability.
- Checkpoint-friendly: writes CSVs per section as it goes; safe to re-run (will overwrite).
"""

import os, sys, time, math, argparse, json
from typing import List, Dict, Any, Tuple, Optional, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import type aliases
from gw_typing import NDArrayF, DataFrame, CIBounds, TrialData, ModelParams

# ---------------- Utilities ----------------
def tanh(x: Union[float, NDArrayF]) -> Union[float, NDArrayF]:
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion confidence intervals.
    
    Args:
        k: Number of successes
        n: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n == 0: return (0.0, 0.0)
    p = k / n
    z = 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    return (float(lo), float(hi))

def bootstrap_mean_ci(samples: Union[Sequence[float], NDArrayF], B: int = 1000, alpha: float = 0.05, seed: int = 1234) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.
    
    Args:
        samples: Input data samples
        B: Number of bootstrap resamples
        alpha: Significance level
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if len(samples)==0: return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    samples_arr = np.asarray(samples); n=len(samples_arr)
    means: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(samples_arr[idx])))
    lo = float(np.percentile(means, 100*alpha/2))
    hi = float(np.percentile(means, 100*(1-alpha/2)))
    return (float(np.mean(samples_arr)), lo, hi)

def ece(probs: NDArrayF, labels: NDArrayF, M: int = 12) -> float:
    """
    Expected Calibration Error (ECE) for probabilistic predictions.
    
    Args:
        probs: Predicted probabilities [0,1]
        labels: True binary labels {0,1}
        M: Number of bins for calibration
        
    Returns:
        ECE value
    """
    probs = np.asarray(probs).ravel(); labels = np.asarray(labels).ravel().astype(int)
    bins = np.linspace(0,1,M+1); e=0.0
    for i in range(M):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if np.any(m):
            e += (m.mean()) * abs(labels[m].mean() - probs[m].mean())
    return float(e)

def reliability_bins(probs: NDArrayF, labels: NDArrayF, M: int = 12) -> Tuple[NDArrayF, NDArrayF, NDArrayF]:
    """
    Compute reliability diagram bins for calibration analysis.
    
    Args:
        probs: Predicted probabilities
        labels: True binary labels
        M: Number of bins
        
    Returns:
        Tuple of (bin_midpoints, accuracies, counts)
    """
    probs = np.asarray(probs).ravel(); labels = np.asarray(labels).ravel().astype(int)
    bins = np.linspace(0,1,M+1); mids = 0.5*(bins[:-1]+bins[1:])
    accs: List[float] = []; counts: List[int] = []
    for i in range(M):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if np.any(m): accs.append(float(labels[m].mean())); counts.append(int(m.sum()))
        else: accs.append(float('nan')); counts.append(0)
    return mids, np.array(accs), np.array(counts)

def auc_pairwise(scores: NDArrayF, labels: NDArrayF) -> float:
    """
    Area Under the ROC Curve computed via pairwise comparisons.
    
    Args:
        scores: Prediction scores/probabilities
        labels: True binary labels
        
    Returns:
        AUC value [0,1]
    """
    s = np.asarray(scores).ravel(); y = np.asarray(labels).ravel().astype(int)
    pos = s[y==1]; neg = s[y==0]
    if len(pos)==0 or len(neg)==0: return 0.5
    wins = 0.0
    for p in pos: 
        wins += float(np.mean(p > neg)) + 0.5 * float(np.mean(p == neg))
    return float(wins / len(pos))

def roc_curve(scores: NDArrayF, labels: NDArrayF, grid: int = 201) -> Tuple[NDArrayF, NDArrayF]:
    """
    Compute ROC curve points.
    
    Args:
        scores: Prediction scores
        labels: True binary labels
        grid: Number of threshold points
        
    Returns:
        Tuple of (false_positive_rates, true_positive_rates)
    """
    s = np.asarray(scores); y = np.asarray(labels).astype(int)
    ths = np.linspace(float(s.min()), float(s.max()), grid)
    tpr: List[float] = []; fpr: List[float] = []
    for t in ths:
        yp = (s >= t).astype(int)
        tp = int(np.sum((yp==1) & (y==1))); fp = int(np.sum((yp==1) & (y==0)))
        tn = int(np.sum((yp==0) & (y==0))); fn = int(np.sum((yp==0) & (y==1)))
        TPR = tp/(tp+fn+1e-8); FPR = fp/(fp+tn+1e-8)
        tpr.append(float(TPR)); fpr.append(float(FPR))
    return np.array(fpr), np.array(tpr)

# ---------------- Model & Stimuli ----------------
def make_small_world_W(N: int, base_coupling: float, degree_frac: float, rewire_p: float, seed: int) -> NDArrayF:
    """
    Generate small-world connectivity matrix using Watts-Strogatz algorithm.
    
    Args:
        N: Number of nodes
        base_coupling: Base connection strength
        degree_frac: Fraction of possible connections for initial regular lattice
        rewire_p: Rewiring probability [0,1]
        seed: Random seed
        
    Returns:
        N×N weight matrix with small-world topology
    """
    r = np.random.default_rng(seed); k = max(2, int(degree_frac * (N-1)));  k += (k % 2)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for m in range(1, k//2 + 1):
            j1 = (i + m) % N; j2 = (i - m) % N
            A[j1, i] = 1; A[j2, i] = 1
    for i in range(N):
        targets = np.where(A[:, i] == 1)[0]
        for t in targets:
            if r.random() < rewire_p:
                A[t, i] = 0
                candidates = [c for c in range(N) if c != i and A[c, i] == 0]
                if candidates: A[r.choice(candidates), i] = 1
    W = np.zeros((N, N), dtype=float)
    nz = np.where(A == 1); W[nz] = r.normal(0.0, 1.0, size=len(nz[0]))
    col_norms = np.linalg.norm(W, axis=0) + 1e-8
    return (W / col_norms) * base_coupling

class GlobalWorkspace:
    """
    Global Workspace Theory implementation with competitive ignition dynamics.
    
    Implements a neural global workspace where:
    - Multiple specialized modules project to a global workspace
    - Winner-take-all competition determines which content becomes globally available
    - Global ignition occurs when winning module exceeds threshold
    - Global broadcast amplifies the winning signal across the network
    """
    
    def __init__(self, N: int, K: int = 5, tau: float = 0.9, theta: float = 0.55, seed: int = 0):
        """
        Initialize Global Workspace.
        
        Args:
            N: Number of neural units
            K: Number of workspace modules (vocabulary size)
            tau: Temperature parameter for softmax competition
            theta: Ignition threshold for global access
            seed: Random seed for reproducible initialization
        """
        r = np.random.default_rng(seed)
        self.K = K; self.tau = tau; self.theta = theta
        self.proj = r.normal(0.0, 1.0, size=(K, N))
        self.proj /= (np.linalg.norm(self.proj, axis=1, keepdims=True)+1e-8)
        self.broadcasts = r.normal(0.0, 1.0, size=(K, N))
        self.broadcasts /= (np.linalg.norm(self.broadcasts, axis=1, keepdims=True)+1e-8)
    
    def step(self, x: NDArrayF) -> Tuple[NDArrayF, NDArrayF, int, float]:
        """
        Execute one Global Workspace step: competition → ignition → broadcast.
        
        Args:
            x: Current neural state vector (N,)
            
        Returns:
            Tuple of (broadcast_signal, all_probabilities, winning_module_id, ignition_binary)
        """
        xn = x / (np.linalg.norm(x)+1e-8)
        s = self.proj @ xn
        # Use configured temperature (tau) instead of hard-coded constant
        logits = s / (self.tau if getattr(self, 'tau', None) else 0.9)
        m = logits.max(); p = np.exp(logits - m); p = p / (p.sum()+1e-8)
        k = int(np.argmax(p)); ignited = float(p[k] > 0.55)
        # Use configured ignition threshold (theta) instead of fixed 0.55 if available
        if getattr(self, 'theta', None) is not None:
            ignited = float(p[k] > self.theta)
        return self.broadcasts[k], p, k, ignited

VOCAB = ["SIL", "VIS_TGT", "MASK", "AUD_TONE", "CHANGE"]
LABEL_TO_ID = {n:i for i,n in enumerate(VOCAB)}

def make_env_inputs(T: int, specs: List[Dict[str, Any]]) -> Dict[str, NDArrayF]:
    """
    Generate environmental input timeseries from stimulus specifications.
    
    Args:
        T: Total time steps
        specs: List of stimulus specifications with keys: kind, t, width, amp
        
    Returns:
        Dictionary with keys V (visual), A (auditory), C (change), y (labels)
    """
    V = np.zeros(T); A = np.zeros(T); C = np.zeros(T); y = np.zeros(T, dtype=int)
    for sp in specs:
        t = sp["t"]; width = sp.get("width", 3); amp = sp.get("amp", 1.0)
        rngt = range(max(0, t-width), min(T, t+width+1))
        if sp["kind"] == "vis_tgt":
            for k in rngt: V[k] += amp; y[k] = LABEL_TO_ID["VIS_TGT"]
        elif sp["kind"] == "mask":
            for k in rngt: V[k] += amp*1.6; y[k] = LABEL_TO_ID["MASK"]
        elif sp["kind"] == "aud_tone":
            for k in rngt: A[k] += amp; y[k] = LABEL_TO_ID["AUD_TONE"]
        elif sp["kind"] == "change":
            for k in rngt: C[k] += amp; y[k] = LABEL_TO_ID["CHANGE"]
    return {"V":V, "A":A, "C":C, "y":y}

def masking_specs(T: int, pos: int, soa: int, amp_t: float = 1.0, amp_m: float = 1.6) -> List[Dict[str, Any]]:
    """
    Generate stimulus specifications for visual masking paradigm.
    
    Args:
        T: Total time steps
        pos: Target position
        soa: Stimulus Onset Asynchrony (target-mask delay)
        amp_t: Target amplitude
        amp_m: Mask amplitude
        
    Returns:
        List of stimulus specifications for target and forward/backward masks
    """
    fm = max(0, pos - max(1, soa//2)); bm = min(T-1, pos + max(1, soa//2))
    return [
        {"kind":"vis_tgt","t":pos,"width":1,"amp":amp_t},
        {"kind":"mask","t":fm,"width":1,"amp":amp_m},
        {"kind":"mask","t":bm,"width":1,"amp":amp_m},
    ]

def blink_specs(T: int, pos1: int, lag: int, amp: float = 1.0) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate stimulus specifications for attentional blink paradigm.
    
    Args:
        T: Total time steps
        pos1: First target position
        lag: Inter-target lag
        amp: Stimulus amplitude
        
    Returns:
        Tuple of (stimulus_specs, second_target_position)
    """
    pos2 = min(T-1, pos1 + lag)
    return [
        {"kind":"vis_tgt","t":pos1,"width":1,"amp":amp},
        {"kind":"vis_tgt","t":pos2,"width":1,"amp":amp},
    ], pos2

def change_blind_specs(T: int, start: int, period: int, delta: float = 0.6) -> List[Dict[str, Any]]:
    """
    Generate stimulus specifications for change blindness paradigm.
    
    Args:
        T: Total time steps
        start: Starting time for changes
        period: Period between changes
        delta: Change amplitude
        
    Returns:
        List of stimulus specifications for alternating changes
    """
    specs = []; flip=False
    for t in range(start, T, 1):
        if (t-start) % period == 0: flip = not flip
        if flip: specs.append({"kind":"change","t":t,"width":0,"amp":delta})
    return specs

# Random projection features (6D) for speed
def run_trial_fast(T: int = 360, burn: int = 80, seed: int = 0, specs: Optional[List[Dict[str, Any]]] = None, noise: float = 0.10) -> Dict[str, NDArrayF]:
    """
    Run a fast neural simulation trial with random feature projections.
    
    Args:
        T: Total timesteps
        burn: Burn-in period to discard
        seed: Random seed
        specs: Stimulus specifications (None for spontaneous activity)
        noise: Noise level for external inputs
        
    Returns:
        Dictionary with trial data: tokens, probs, ignitions, labels, feats
    """
    r = np.random.default_rng(seed); N = 32; g = 0.9
    W = make_small_world_W(N, 1.0, 0.38, 0.18, seed)
    Win_v = r.normal(0.0, 1.0, size=N)
    Win_a = r.normal(0.0, 1.0, size=N)
    Win_c = r.normal(0.0, 1.0, size=N)
    proj_R = r.normal(0.0, 1.0, size=(N, 6)); proj_R /= (np.linalg.norm(proj_R, axis=0, keepdims=True)+1e-8)

    x = np.zeros((N, T)); x[:,0] = r.normal(0, 0.1, size=N)
    gw = GlobalWorkspace(N=N, K=len(VOCAB), tau=0.9, theta=0.55, seed=seed+1)

    inputs = make_env_inputs(T, specs) if specs is not None else {"V":np.zeros(T),"A":np.zeros(T),"C":np.zeros(T),"y":np.zeros(T, dtype=int)}
    y = inputs["y"]
    tokens: List[int] = []; ignitions: List[float] = []; probs: List[float] = []; feats: List[NDArrayF] = []; labels: List[int] = []

    for t in range(T-1):
        z_int = g * (W @ x[:, t])
        z_ext = 0.24*(Win_v*inputs["V"][t] + Win_a*inputs["A"][t] + Win_c*inputs["C"][t]) + noise*np.random.default_rng(900+t).normal(0,1,size=N)
        x[:, t+1] = tanh(z_int + z_ext)

        feat = (proj_R.T @ x[:, t+1]).ravel()
        feats.append(feat); labels.append(int(y[t]))

        bcast, p, k, ign = gw.step(x[:, t+1])
        x[:, t+1] = tanh(x[:, t+1] + 0.22*bcast)
        tokens.append(k); ignitions.append(float(ign)); probs.append(float(p[k]))

    sl = slice(burn, T-1)
    return {
        "tokens": np.array(tokens[sl], dtype=int),
        "probs": np.array(probs[sl], dtype=float),
        "ignitions": np.array(ignitions[sl], dtype=float),
        "labels": np.array(labels[sl], dtype=int),
        "feats": np.array(feats[sl], dtype=float),  # (T',6)
    }

def agg_features(feats: NDArrayF, ign: NDArrayF, center: int, w: int = 5) -> NDArrayF:
    """
    Aggregate neural features around a time window for classification.
    
    Args:
        feats: Feature matrix (T, D)
        ign: Ignition signals (T,)
        center: Center timepoint
        w: Window half-width
        
    Returns:
        Aggregated feature vector (14-dim: mean_abs + max + ignition_stats)
    """
    a = max(0, center-w); b = min(len(ign)-1, center+w)
    F = feats[a:b+1]; g = ign[a:b+1]
    mean_abs = np.mean(np.abs(F), axis=0)
    maxi = np.max(F, axis=0)
    ig_mean = np.mean(g); ig_max = np.max(g)
    return np.hstack([mean_abs, maxi, [ig_mean, ig_max]])  # 14 dims

def fit_logistic(X: NDArrayF, y: NDArrayF, iters: int = 450, lr: float = 0.1, l2: float = 1e-3) -> Tuple[NDArrayF, float]:
    """
    Fit logistic regression with L2 regularization using gradient descent.
    
    Args:
        X: Feature matrix (N, D)
        y: Binary labels (N,)
        iters: Number of iterations
        lr: Learning rate
        l2: L2 regularization strength
        
    Returns:
        Tuple of (weights, bias)
    """
    X = np.asarray(X); y = np.asarray(y).astype(int)
    n, d = X.shape; w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X@w + b; p = 1/(1+np.exp(-z))
        dw = (X.T @ (p - y))/n + l2*w; db = float(np.mean(p - y))
        w -= lr*dw; b -= lr*db
    return w, b

def section_masking(outdir, SOAs, N_TRIALS, BOOTS, seed0=11000):
    rows = []; roc_pages = []; calib_rows = []
    for soa in SOAs:
        hits=[]; confs=[]; ign_lats=[]
        Xnr=[]; ynr=[]; nr_probs=[]; nr_labels=[]
        for tr in range(N_TRIALS):
            T=360; pos=180
            specs = masking_specs(T, pos, soa, amp_t=0.9, amp_m=2.0)
            out = run_trial_fast(T=T, burn=80, seed=seed0+soa*300+tr, specs=specs, noise=0.10)
            center = pos-80
            # report hit near center
            hits.append(1 if np.any(out["tokens"][center-5:center+6] == LABEL_TO_ID["VIS_TGT"]) else 0)
            a = max(0, center-5); b = min(len(out["probs"])-1, center+5)
            confs.append(float(np.max(out["probs"][a:b+1])) if b>=a else 0.0)
            # ignition latency
            lat=None
            for dt in range(22):
                idx=center+dt
                if 0 <= idx < len(out["ignitions"]) and out["ignitions"][idx]>0.5:
                    lat=dt; break
            if lat is not None: ign_lats.append(lat)
            # no-report pos window (present=1) and a negative off-window (0)
            Xnr.append(agg_features(out["feats"], out["ignitions"], center, w=5)); ynr.append(1)
            off=center+40
            if off+5 < len(out["ignitions"]):
                Xnr.append(agg_features(out["feats"], out["ignitions"], off, w=5)); ynr.append(0)
        # summarize
        k = int(np.sum(hits)); n = len(hits); p = k/n; lo, hi = wilson_interval(k, n)
        mu_conf, lo_conf, hi_conf = bootstrap_mean_ci(confs, B=BOOTS)
        mu_lat, lo_lat, hi_lat = bootstrap_mean_ci(ign_lats, B=BOOTS) if len(ign_lats)>0 else (float('nan'), float('nan'), float('nan'))
        rows.append({"SOA":soa, "ReportAcc":p, "ReportAcc_lo":lo, "ReportAcc_hi":hi,
                     "Conf_mean":mu_conf, "Conf_lo":lo_conf, "Conf_hi":hi_conf,
                     "IgnLat_mean":mu_lat, "IgnLat_lo":lo_lat, "IgnLat_hi":hi_lat})

        # no-report AUC (5-fold CV) and calibration bins/ECE on held-out folds
        Xnr = np.vstack(Xnr); ynr = np.array(ynr)
        idx = np.arange(len(ynr)); np.random.default_rng(soa*77+5).shuffle(idx)
        folds = np.array_split(idx, 5)
        aucs=[]; curves=[]; all_probs=[]; all_labels=[]
        for i in range(5):
            va = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j!=i])
            mu = Xnr[tr].mean(axis=0); sd = Xnr[tr].std(axis=0)+1e-8
            Xtr = (Xnr[tr]-mu)/sd; Xva=(Xnr[va]-mu)/sd
            w,b = fit_logistic(Xtr, ynr[tr], iters=450, lr=0.1, l2=1e-3)
            scores = Xva@w + b
            # probs for calibration
            probs = 1/(1+np.exp(-(scores)))
            all_probs.extend(probs.tolist()); all_labels.extend(ynr[va].tolist())
            fpr, tpr = roc_curve(scores, ynr[va], grid=201)
            curves.append((fpr, tpr))
            aucs.append(auc_pairwise(scores, ynr[va]))
        auc_mean = float(np.mean(aucs)); auc_std = float(np.std(aucs))
        roc_pages.append({"SOA":soa, "curves":curves, "AUC_mean":auc_mean, "AUC_std":auc_std})

        mids, accs, counts = reliability_bins(np.array(all_probs), np.array(all_labels), M=12)
        e = ece(np.array(all_probs), np.array(all_labels), M=12)
        calib_rows.append({"SOA":soa, "ECE":float(e), "bins_mids":mids.tolist(), "bins_accs":np.nan_to_num(accs).tolist(), "bins_counts":counts.tolist()})

    df = pd.DataFrame(rows)
    df_auc = pd.DataFrame([{"SOA":p["SOA"], "AUC_mean":p["AUC_mean"], "AUC_std":p["AUC_std"]} for p in roc_pages])
    df_cal = pd.DataFrame(calib_rows)
    df.to_csv(os.path.join(outdir, "masking_curve_ci.csv"), index=False)
    df_auc.to_csv(os.path.join(outdir, "noreport_auc_masking.csv"), index=False)
    df_cal.to_json(os.path.join(outdir, "noreport_calibration_masking.json"), orient="records", indent=2)
    return df, roc_pages, calib_rows

def section_blink(outdir, LAGS, N_TRIALS, seed0=22000):
    rows=[]
    for lag in LAGS:
        acc=[]
        for tr in range(N_TRIALS):
            T=340; pos1=150
            specs, pos2 = blink_specs(T, pos1, lag, amp=1.0)
            out = run_trial_fast(T=T, burn=80, seed=seed0+lag*300+tr, specs=specs, noise=0.10)
            center = pos2-80
            acc.append(1 if np.any(out["tokens"][center-4:center+5] == LABEL_TO_ID["VIS_TGT"]) else 0)
        k = int(np.sum(acc)); n=len(acc); p=k/n; lo,hi = wilson_interval(k,n)
        rows.append({"Lag":lag, "T2Acc":p, "T2_lo":lo, "T2_hi":hi})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir,"blink_curve_ci.csv"), index=False); return df

def section_change_blind(outdir, PERIODS, N_TRIALS, BOOTS, seed0=33000):
    rows=[]
    for period in PERIODS:
        det=[]
        for tr in range(N_TRIALS):
            T=360; start=90
            specs = change_blind_specs(T, start, period, delta=0.6)
            out = run_trial_fast(T=T, burn=80, seed=seed0+period*300+tr, specs=specs, noise=0.10)
            start_idx = start-80
            d=None
            for t in range(start_idx+3, len(out["tokens"])):
                if out["tokens"][t] == LABEL_TO_ID["CHANGE"]:
                    d = t - start_idx; break
            if d is None: d = len(out["tokens"]) - start_idx
            det.append(d)
        mu, lo, hi = bootstrap_mean_ci(det, B=BOOTS)
        rows.append({"Period":period, "DetectTime":mu, "Detect_lo":lo, "Detect_hi":hi})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir,"change_blind_curve_ci.csv"), index=False); return df

def section_dual(outdir, N_TRIALS, seed0=44000):
    rows=[]
    for dual in [False, True]:
        acc_vis=[]; acc_change=[]; acc_aud=[]
        for tr in range(N_TRIALS):
            T=340; pos=T//2
            specs = [{"kind":"vis_tgt","t":pos-10,"width":2,"amp":1.0},
                     {"kind":"mask","t":pos-6,"width":1,"amp":1.6},
                     {"kind":"change","t":pos+16,"width":2,"amp":0.9}]
            if dual: specs.append({"kind":"aud_tone","t":pos+4,"width":2,"amp":1.0})
            out = run_trial_fast(T=T, burn=80, seed=seed0+int(dual)*500+tr, specs=specs, noise=0.10)
            acc_vis.append(1 if np.any(out["tokens"][pos-80-5:pos-80+6] == LABEL_TO_ID["VIS_TGT"]) else 0)
            acc_change.append(1 if np.any(out["tokens"][pos+16-80-5:pos+16-80+6] == LABEL_TO_ID["CHANGE"]) else 0)
            if dual: acc_aud.append(1 if np.any(out["tokens"][pos+4-80-5:pos+4-80+6] == LABEL_TO_ID["AUD_TONE"]) else 0)
        def wilson_row(v):
            k, n = int(np.sum(v)), len(v); p=k/n; lo, hi = wilson_interval(k, n); return p, lo, hi
        p,lo,hi = wilson_row(acc_vis); row={"condition":"dual" if dual else "single", "VIS_Acc":p, "VIS_lo":lo, "VIS_hi":hi}
        p,lo,hi = wilson_row(acc_change); row.update({"CHANGE_Acc":p, "CHANGE_lo":lo, "CHANGE_hi":hi})
        if dual:
            p,lo,hi = wilson_row(acc_aud); row.update({"AUD_Acc":p, "AUD_lo":lo, "AUD_hi":hi})
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir,"dualtask_ci.csv"), index=False); return df

# ---------------- Report ----------------
def build_pdf(outdir, mask_df, roc_pages, calib_rows, blink_df, cb_df, dual_df):
    pdf_path = os.path.join(outdir, "full_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Masking accuracy
        plt.figure(figsize=(6,4))
        xs=mask_df["SOA"].values; ys=mask_df["ReportAcc"].values
        plt.plot(xs, ys, marker='o'); plt.fill_between(xs, mask_df["ReportAcc_lo"].values, mask_df["ReportAcc_hi"].values, alpha=0.2)
        plt.xlabel("SOA"); plt.ylabel("Report accuracy"); plt.title("Masking (CIs)")
        plt.tight_layout(); pdf.savefig(); plt.close()
        # Masking ignition latency
        plt.figure(figsize=(6,4))
        xs=mask_df["SOA"].values; ys=mask_df["IgnLat_mean"].values
        plt.plot(xs, ys, marker='o'); plt.fill_between(xs, mask_df["IgnLat_lo"].values, mask_df["IgnLat_hi"].values, alpha=0.2)
        plt.xlabel("SOA"); plt.ylabel("Ignition latency"); plt.title("Masking: ignition latency")
        plt.tight_layout(); pdf.savefig(); plt.close()
        # No-report ROC per SOA
        for rp in roc_pages:
            plt.figure(figsize=(5,5))
            for fpr, tpr in rp["curves"]: plt.plot(fpr, tpr)
            plt.plot([0,1],[0,1],'--')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"No-report ROC (masking, SOA={rp['SOA']}) AUC≈{rp['AUC_mean']:.2f}")
            plt.tight_layout(); pdf.savefig(); plt.close()
        # Calibration pages
        for cr in calib_rows:
            mids = np.array(cr["bins_mids"]); accs = np.array(cr["bins_accs"]); counts = np.array(cr["bins_counts"])
            plt.figure(figsize=(5,5))
            plt.plot([0,1],[0,1],'--')
            sel = ~np.isnan(accs)
            plt.scatter(mids[sel], accs[sel], s=20+2*counts[sel])
            plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy")
            plt.title(f"Reliability (SOA={cr['SOA']})  ECE≈{cr['ECE']:.3f}")
            plt.tight_layout(); pdf.savefig(); plt.close()
        # Blink
        plt.figure(figsize=(6,4))
        xs=blink_df["Lag"].values; ys=blink_df["T2Acc"].values
        plt.plot(xs, ys, marker='o'); plt.fill_between(xs, blink_df["T2_lo"].values, blink_df["T2_hi"].values, alpha=0.2)
        plt.xlabel("Lag"); plt.ylabel("T2 report accuracy"); plt.title("Attentional blink (CIs)")
        plt.tight_layout(); pdf.savefig(); plt.close()
        # Change blindness
        plt.figure(figsize=(6,4))
        xs=cb_df["Period"].values; ys=cb_df["DetectTime"].values
        plt.plot(xs, ys, marker='o'); plt.fill_between(xs, cb_df["Detect_lo"].values, cb_df["Detect_hi"].values, alpha=0.2)
        plt.xlabel("Flicker period"); plt.ylabel("Detection time"); plt.title("Change blindness (CIs)")
        plt.tight_layout(); pdf.savefig(); plt.close()
        # Dual-task
        plt.figure(figsize=(6,4))
        xs=np.arange(len(dual_df))
        plt.plot(xs, dual_df["VIS_Acc"].values, marker='o')
        plt.plot(xs, dual_df["CHANGE_Acc"].values, marker='s')
        if "AUD_Acc" in dual_df.columns:
            plt.plot(xs, dual_df["AUD_Acc"].fillna(0).values, marker='^')
        plt.xticks(xs, dual_df["condition"].values); plt.xlabel("Condition"); plt.ylabel("Accuracy"); plt.title("Dual-task (CIs)")
        plt.tight_layout(); pdf.savefig(); plt.close()
    return pdf_path

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True, help='Output directory')
    ap.add_argument('--seed', type=int, default=7777)
    ap.add_argument('--boots', type=int, default=1200, help='Bootstrap resamples')
    ap.add_argument('--n_mask', type=int, default=140)
    ap.add_argument('--n_blink', type=int, default=100)
    ap.add_argument('--n_cb', type=int, default=80)
    ap.add_argument('--n_dual', type=int, default=100)
    ap.add_argument('--full', action='store_true', help='Use preset heavier counts (overrides)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f: json.dump(vars(args), f, indent=2)

    if args.full:
        args.n_mask = max(args.n_mask, 200)
        args.n_blink = max(args.n_blink, 140)
        args.n_cb = max(args.n_cb, 120)
        args.n_dual = max(args.n_dual, 140)
        args.boots = max(args.boots, 2000)

    SOAs = [1,2,3,4,6,8]
    BLINK_LAGS = [2,3,4,6,8]
    CB_PERIODS = [10,16,24,36]

    print("=== Starting FULL run ==="); sys.stdout.flush()
    t0 = time.time()

    print("[1/4] Masking with no-report ROC + calibration..."); sys.stdout.flush()
    mask_df, roc_pages, calib_rows = section_masking(args.out, SOAs, args.n_mask, args.boots)
    print("  -> saved masking_curve_ci.csv, noreport_auc_masking.csv, noreport_calibration_masking.json")

    print("[2/4] Attentional blink..."); sys.stdout.flush()
    blink_df = section_blink(args.out, BLINK_LAGS, args.n_blink)
    print("  -> saved blink_curve_ci.csv")

    print("[3/4] Change blindness..."); sys.stdout.flush()
    cb_df = section_change_blind(args.out, CB_PERIODS, args.n_cb, args.boots)
    print("  -> saved change_blind_curve_ci.csv")

    print("[4/4] Dual-task interference..."); sys.stdout.flush()
    dual_df = section_dual(args.out, args.n_dual)
    print("  -> saved dualtask_ci.csv")

    print("Building PDF..."); sys.stdout.flush()
    pdf_path = build_pdf(args.out, mask_df, roc_pages, calib_rows, blink_df, cb_df, dual_df)
    print("Saved:", pdf_path)

    print("Done in {:.1f} minutes".format((time.time()-t0)/60.0))

if __name__ == "__main__":
    main()
