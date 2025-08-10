#!/usr/bin/env python3
"""
Fixed CAI significance testing with proper orientation detection and two-sided permutation test on AUC.

Usage (Windows PowerShell):
    python scripts/eval_cai_significance_fixed.py --cai_dir .\\runs\\mini\\cai --outfile .\\runs\\mini\\cai_signif_fixed.json --B 1000 --R 2000 --by_soa

Outputs a JSON with overall AUC, 95% bootstrap CI, two-sided permutation p-value, effect size, ECE, orientation, and optional per‑SOA stats.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys as _sys
from typing import Any, Dict, Tuple
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, KFold

# Ensure project root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from scripts.cai_features import trial_to_features


def auc_pairwise(scores: Any, labels: Any) -> float:
    """Compute AUC using pairwise comparisons (equivalent to Mann–Whitney U)."""
    s = np.asarray(scores).ravel()
    y = np.asarray(labels).ravel().astype(int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.mean(p > neg)) + 0.5 * float(np.mean(p == neg))
    return float(wins / max(len(pos), 1))


def detect_orientation(probs: Any, y: Any, n_folds: int = 5) -> int:
    """
    Detect if scores are correctly oriented via cross-validation on the scores themselves.
    Returns 1 if orientation is normal (AUC>=0.5), -1 if inverted.
    """
    probs = np.asarray(probs).ravel()
    y = np.asarray(y).ravel().astype(int)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    for _, val_idx in kf.split(probs):
        auc_val = auc_pairwise(probs[val_idx], y[val_idx])
        aucs.append(auc_val)
    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    return 1 if mean_auc >= 0.5 else -1


def bootstrap_ci_auc(probs: Any, y: Any, B: int = 1000, seed: int = 123) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for AUC."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs).ravel()
    y = np.asarray(y).ravel().astype(int)
    auc = auc_pairwise(probs, y)
    n = len(y)
    if n == 0:
        return auc, auc, auc
    bs_aucs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        auc_b = auc_pairwise(probs[idx], y[idx])
        bs_aucs.append(auc_b)
    lo, hi = float(np.percentile(bs_aucs, 2.5)), float(np.percentile(bs_aucs, 97.5))
    return float(auc), lo, hi


def permutation_test_auc(probs: Any, y: Any, R: int = 2000, seed: int = 2025) -> Tuple[float, float]:
    """
    Two-sided permutation test for H0: AUC == 0.5 against H1: AUC != 0.5.
    Returns (p_value, standardized_effect_size).
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs).ravel()
    y = np.asarray(y).ravel().astype(int)
    auc_obs = auc_pairwise(probs, y)
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    null_aucs = []
    for _ in range(R):
        y_perm = rng.permutation(y)
        null_aucs.append(auc_pairwise(probs, y_perm))
    null_aucs = np.asarray(null_aucs)
    distance_obs = abs(auc_obs - 0.5)
    distances_null = np.abs(null_aucs - 0.5)
    # Add-one smoothing for stability in extreme cases
    ge = int(np.sum(distances_null >= distance_obs))
    p_value = float((ge + 1) / (R + 1))
    effect_size = float((auc_obs - 0.5) / (np.std(null_aucs) + 1e-12))
    return p_value, effect_size


def cv_out_of_fold_probs(X: Any, y: Any, n_splits: int = 5, seed: int = 42) -> Tuple[Any, Any]:
    """Get calibrated out-of-fold probabilities reconstructed in original sample order."""
    x_arr = np.asarray(X)
    y_arr = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y_arr), dtype=float)
    y_out = np.zeros(len(y_arr), dtype=int)
    for tr, te in skf.split(x_arr, y_arr):
        base = LogisticRegression(max_iter=200, solver="lbfgs", random_state=seed)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(x_arr[tr], y_arr[tr])
        p_te = clf.predict_proba(x_arr[te])[:, 1]
        probs[te] = p_te
        y_out[te] = y_arr[te]
    return probs, y_out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cai_dir", required=True, help="Directory of CAI-ready trial JSONs")
    p.add_argument("--outfile", required=True, help="Output JSON path for metrics")
    p.add_argument("--B", type=int, default=1000, help="Bootstrap resamples for CI")
    p.add_argument("--R", type=int, default=2000, help="Permutation iterations for p-value")
    p.add_argument("--by_soa", action="store_true", help="Also compute per-SOA stats if SOA meta is present")
    args = p.parse_args()

    cai_dir = Path(args.cai_dir)
    files = sorted(cai_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {cai_dir}")

    rows = []
    metas = []
    for fp in files:
        trial = json.loads(fp.read_text())
        feats = trial_to_features(trial)
        feats["_file"] = fp.name
        rows.append(feats)
        metas.append(trial.get("meta", {}))
    df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metas)

    y = df["label_report"].astype(int).to_numpy()
    X = df.drop(columns=["label_report", "_file"]).to_numpy(dtype=float)

    # Out-of-fold calibrated probabilities in original order
    probs, y_true = cv_out_of_fold_probs(X, y, n_splits=5, seed=42)

    # Detect and fix orientation once
    orientation = detect_orientation(probs, y_true, n_folds=5)
    if orientation == -1:
        probs = 1.0 - probs

    # Metrics
    auc, lo, hi = bootstrap_ci_auc(probs, y_true, B=args.B, seed=123)
    p_val, effect = permutation_test_auc(probs, y_true, R=args.R, seed=2025)
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    out: Dict[str, Any] = {
        "n": int(len(y_true)),
        "AUC": float(auc),
        "AUC_CI95": [float(lo), float(hi)],
        "p_value": float(p_val),
        "effect_size": float(effect),
        "ECE": ece,
        "orientation": "inverted" if orientation == -1 else "normal",
        "significant": bool(p_val < 0.05),
    }

    # Optional per-SOA analysis (uses original order alignment)
    if args.by_soa and not meta_df.empty and "SOA" in meta_df.columns:
        soa_vals = meta_df["SOA"].to_numpy()
        by = []
        unique_soas = sorted({int(v) for v in soa_vals.tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))})
        for s in unique_soas:
            m = (soa_vals == s)
            n_s = int(m.sum())
            if n_s >= 30:
                auc_s, lo_s, hi_s = bootstrap_ci_auc(probs[m], y_true[m], B=max(300, args.B // 2), seed=123 + int(s))
                p_s, eff_s = permutation_test_auc(probs[m], y_true[m], R=max(400, args.R // 2), seed=2025 + int(s))
                by.append({
                    "SOA": int(s),
                    "n": n_s,
                    "AUC": float(auc_s),
                    "AUC_CI95": [float(lo_s), float(hi_s)],
                    "p_value": float(p_s),
                    "effect_size": float(eff_s),
                    "significant": bool(p_s < 0.05),
                })
        out["by_SOA"] = by

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.outfile).write_text(json.dumps(out, indent=2))

    # Console summary
    print("\n=== CAI Significance Results (AUC) ===")
    print(f"N: {out['n']}")
    print(f"AUC: {out['AUC']:.3f} [{out['AUC_CI95'][0]:.3f}, {out['AUC_CI95'][1]:.3f}]")
    print(f"P-value: {out['p_value']:.3f}; Effect size: {out['effect_size']:.2f}; Orientation: {out['orientation']}")
    if 'by_SOA' in out:
        print("By SOA:")
        for s in out['by_SOA']:
            star = '**' if s['significant'] else ''
            print(f"  SOA {s['SOA']}: n={s['n']}, AUC={s['AUC']:.3f} [{s['AUC_CI95'][0]:.3f}, {s['AUC_CI95'][1]:.3f}], p={s['p_value']:.3f} {star}")


if __name__ == "__main__":
    main()
