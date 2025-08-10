import csv
import json
import math
from pathlib import Path

# Inputs
CSV_PATH = Path(__file__).resolve().parents[1] / 'data' / 'masking_curves_all_sizes.csv'
N_SMALL = 32
N_LARGE = 512
SOA = 1
Z = 1.96


def wilson_ci(p: float, n: int, z: float = Z):
    if n == 0:
        return 0.0, 1.0
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_diff_ci(p1: float, n1: int, p2: float, n2: int, z: float = Z):
    l1, u1 = wilson_ci(p1, n1, z)
    l2, u2 = wilson_ci(p2, n2, z)
    low = l2 - u1
    high = u2 - l1
    return low, high


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def main():
    with CSV_PATH.open(newline='') as f:
        rows = list(csv.DictReader(f))
    sel = {(int(r['N_nodes']), int(r['SOA'])): r for r in rows}
    A = sel[(N_SMALL, SOA)]
    B = sel[(N_LARGE, SOA)]

    n1 = int(A['n_trials']); x1 = int(A['n_hits']); p1 = x1 / n1
    n2 = int(B['n_trials']); x2 = int(B['n_hits']); p2 = x2 / n2

    delta = p2 - p1
    low, high = newcombe_diff_ci(p1, n1, p2, n2)
    h = cohens_h(p1, p2)

    out = {
        'N1': N_SMALL,
        'N2': N_LARGE,
        'SOA': SOA,
        'p1': round(p1, 3),
        'p2': round(p2, 3),
        'delta': round(delta, 3),
        'delta_CI95': [round(low, 3), round(high, 3)],
        'cohens_h': round(h, 3)
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
