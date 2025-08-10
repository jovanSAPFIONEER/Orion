import numpy as np
from scripts.metrics_broadcasting import broadcasting_index, ignition_span
from scripts.metrics_pci import pci_like

def test_broadcasting_index_basic():
    # 4 time steps, 3 modules; module 0/1 sustain >=3 steps, module 2 does not
    a = np.array([
        [0.1, 0.6, 0.0],
        [0.6, 0.7, 0.2],
        [0.7, 0.8, 0.4],
        [0.9, 0.2, 0.3],
    ])
    bi = broadcasting_index(a, threshold=0.5, min_duration=3)
    assert 0.0 <= bi <= 1.0
    # modules 0 and 1 sustained >=3, 2 did not -> 2/3
    assert abs(bi - (2/3)) < 1e-6


def test_ignition_span_basic():
    a = np.array([
        [0.1, 0.2],
        [0.6, 0.1],
        [0.7, 0.4],
        [0.2, 0.2],
    ])
    span = ignition_span(a, threshold=0.5)
    # Active at t=1 and t=2; none at t=3 -> duration 2
    assert span == 2


def test_pci_like_monotonic():
    base = np.zeros((5, 4))
    resp = np.zeros((5, 4))
    # induce some differences in response
    resp[1:4, 1:3] = 1.0
    val = pci_like(base, resp, threshold=0.0)
    assert val >= 0.0
