import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Sequence, List

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.metrics_pci import pci_like  # type: ignore

"""
Compute a Conscious Access Index (CAI) from saved trial activations.

Expected input JSON (per trial):
{
    "module_activations": [[...], ...],  # [T, K] module probability vectors (rows typically sum to 1)
    "baseline_activations": [[...], ...],  # optional [T, K]
}

Robust CAI for softmax-like activations:
    Let m_t = max_k module_activations[t, k]. Define features on m_t using an adaptive threshold thr:
        - thr = max(0.3, 1/K + 0.08, percentile_60(m)) where K is number of modules.
        - ign_frac: fraction of timesteps with m_t >= thr
        - run_norm: longest consecutive run of (m_t >= thr) divided by T
        - span_norm: duration from first to last (m_t >= thr) divided by T
    Base score = 0.5*ign_frac + 0.3*run_norm + 0.2*span_norm
    Add small PCI term if baseline_activations present: + 0.05*PCI
    Fallback (if all features zero): use normalized mean(m): (mean(m) - 1/K) / (1 - 1/K)
    Finally, clip to [0,1].
"""


def _series_features(m: Sequence[float], thr: float) -> Dict[str, float]:
    m_arr = np.asarray(m, dtype=float)
    T = int(max(len(m_arr), 1))
    if T == 0:
        return {"ign_frac": 0.0, "run_norm": 0.0, "span_norm": 0.0}
    b = (m_arr >= thr).astype(np.uint8)
    # ignited fraction
    ign_frac = float(b.mean())
    # longest consecutive run
    max_run = 0
    cur = 0
    for val in b:
        if val:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    run_norm = float(max_run / T)
    # span from first to last ignition
    if b.any():
        idx = np.where(b == 1)[0]
        span = int(idx[-1] - idx[0] + 1)
    else:
        span = 0
    span_norm = float(span / T)
    return {"ign_frac": ign_frac, "run_norm": run_norm, "span_norm": span_norm}


def compute_cai(trial: Dict[str, Any]) -> float:
    arr = np.array(trial.get("module_activations", []), dtype=float)
    if arr.size == 0:
        return 0.0
    # m_t is the max module probability per timestep
    if arr.ndim == 1:
        m = arr
    else:
        m = arr.max(axis=1)
    # Adaptive threshold based on module count and distribution
    K = int(arr.shape[1]) if arr.ndim == 2 else max(int(trial.get("meta", {}).get("K", 0)), 1)
    base_thr = 1.0 / max(K, 1)
    thr = float(max(0.3, base_thr + 0.08, np.percentile(m, 60)))
    feats = _series_features(m.tolist() if hasattr(m, 'tolist') else list(m), thr=thr)

    pci = 0.0
    if "baseline_activations" in trial:
        base = np.array(trial["baseline_activations"], dtype=float)
        if base.shape == arr.shape:
            pci = max(pci_like(base, arr, threshold=0.0), 0.0)

    cai = 0.5 * feats["ign_frac"] + 0.3 * feats["run_norm"] + 0.2 * feats["span_norm"] + 0.05 * float(pci)
    if cai <= 1e-6:
        # fallback on normalized mean of m above chance level 1/K
        norm = (m.mean() - base_thr) / max(1.0 - base_thr, 1e-6)
        cai = max(0.0, float(norm))
    return float(np.clip(cai, 0.0, 1.0))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, help="Path to trial JSON or directory of JSONs")
    p.add_argument("--outfile", required=False, help="Write per-trial CAI and summary")
    args = p.parse_args()

    path = Path(args.infile)
    trials: List[Dict[str, Any]] = []
    if path.is_dir():
        for fp in sorted(path.glob("*.json")):
            trials.append(json.loads(fp.read_text()))
    else:
        trials.append(json.loads(Path(args.infile).read_text()))

    cais: List[float] = [compute_cai(t) for t in trials]
    out: Dict[str, Any] = {
        "n_trials": len(cais),
        "cai_mean": float(np.mean(cais)) if cais else 0.0,
        "cai_std": float(np.std(cais)) if cais else 0.0,
        "trials": cais,
    }

    text = json.dumps(out, indent=2)
    if args.outfile:
        Path(args.outfile).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
