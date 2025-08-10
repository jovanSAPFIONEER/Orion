from __future__ import annotations
import numpy as np
from typing import Optional

# Simple PCI-like measure: perturb a subset of modules and
# compute Lempel–Ziv complexity of the evoked response vs baseline.


def _lempel_ziv_binary(seq: np.ndarray) -> int:
    """Compute LZ77-like complexity on a 1D binary sequence."""
    s = ''.join('1' if x else '0' for x in seq.astype(bool).tolist())
    i, c, k, l = 0, 1, 1, 1
    n = len(s)
    while True:
        if i + k > n:
            c += 1
            break
        if s[i:i+k] == s[l:l+k]:
            k += 1
        else:
            c += 1
            l += 1
            if l == i + k:
                i = l
                k = 1
                l = 1
        if i + k > n:
            break
    return c


def pci_like(
    baseline: np.ndarray,
    response: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Compute a crude PCI-like complexity measure from baseline vs response.

    Args:
        baseline: [T, M] array of module activations without perturbation.
        response: [T, M] array of activations after perturbation.
        threshold: optional binarization threshold (default 0 -> positive values are 1).

    Returns:
        Normalized difference in LZ complexity of flattened, binarized responses.
    """
    if baseline.shape != response.shape:
        raise ValueError("baseline and response must have the same shape")
    if baseline.ndim != 2:
        raise ValueError("inputs must be [T, M]")

    def binarize(x: np.ndarray) -> np.ndarray:
        return (x > threshold).astype(np.uint8)

    b = binarize(baseline).flatten()
    r = binarize(response).flatten()

    cb = _lempel_ziv_binary(b)
    cr = _lempel_ziv_binary(r)
    if cb == 0:
        return float(max(cr, 0.0))
    return float(max((cr - cb) / max(cb, 1), 0.0))
