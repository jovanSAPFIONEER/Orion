from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# Broadcasting Index (BI): fraction of modules active above threshold
# sustained for a minimum duration within a trial window.

def broadcasting_index(
    module_activations: np.ndarray,
    threshold: float = 0.5,
    min_duration: int = 3,
) -> float:
    """
    Compute a simple broadcasting index (BI).

    Args:
        module_activations: array of shape [T, M] with per-time, per-module activation (0..1 or arbitrary scale).
        threshold: activation threshold to consider a module "active".
        min_duration: number of consecutive time steps a module must stay above threshold to count as broadcasting.

    Returns:
        BI in [0,1]: fraction of modules that achieve the sustained activation criterion.
    """
    if module_activations.ndim != 2:
        raise ValueError("module_activations must be [T, M]")
    T, M = module_activations.shape
    if T == 0 or M == 0:
        return 0.0

    active = module_activations >= threshold  # [T, M]
    sustained = np.zeros(M, dtype=bool)
    run = np.zeros(M, dtype=int)
    for t in range(T):
        run = np.where(active[t], run + 1, 0)
        sustained = sustained | (run >= min_duration)
    return float(sustained.sum() / M)


def ignition_span(
    module_activations: np.ndarray,
    threshold: float = 0.5,
) -> int:
    """
    Duration (in time steps) from first time any module crosses threshold to the last time any module stays above.
    Returns 0 if never crosses.
    """
    active_any = (module_activations >= threshold).any(axis=1)  # [T]
    if not active_any.any():
        return 0
    idx = np.where(active_any)[0]
    return int(idx[-1] - idx[0] + 1)
