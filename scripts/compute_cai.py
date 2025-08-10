import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scripts.metrics_broadcasting import broadcasting_index, ignition_span
from scripts.metrics_pci import pci_like

"""
Compute a simple Conscious Access Index (CAI) from saved trial activations.

Expected input JSON (per trial):
{
  "module_activations": [[...], ...],  # [T, M] floats 0..1
  "baseline_activations": [[...], ...],  # optional [T, M]
}

CAI definition (example):
  CAI = 0.6 * BI + 0.3 * norm_ignition + 0.1 * PCI
Where:
  - BI: broadcasting_index(module_activations)
  - norm_ignition: ignition_span / T (0..1)
  - PCI: computed if baseline provided; else 0
"""


def compute_cai(trial: Dict[str, Any]) -> float:
    arr = np.array(trial["module_activations"], dtype=float)
    T = max(len(arr), 1)
    bi = broadcasting_index(arr, threshold=0.5, min_duration=3)
    ign = ignition_span(arr, threshold=0.5) / T

    pci = 0.0
    if "baseline_activations" in trial:
        base = np.array(trial["baseline_activations"], dtype=float)
        if base.shape == arr.shape:
            pci = pci_like(base, arr, threshold=0.0)

    return float(0.6 * bi + 0.3 * ign + 0.1 * pci)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, help="Path to trial JSON or directory of JSONs")
    p.add_argument("--outfile", required=False, help="Write per-trial CAI and summary")
    args = p.parse_args()

    path = Path(args.infile)
    trials = []
    if path.is_dir():
        for fp in sorted(path.glob("*.json")):
            trials.append(json.loads(fp.read_text()))
    else:
        trials.append(json.loads(Path(args.infile).read_text()))

    cais = [compute_cai(t) for t in trials]
    out = {
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
