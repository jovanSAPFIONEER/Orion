#!/usr/bin/env python3
"""
test_enhanced_codebase.py

Comprehensive validation script for the enhanced GW awareness codebase.
Tests type safety, robustness, and reproducibility of core functions.

Usage:
    python test_enhanced_codebase.py --quick  # Fast tests only
    python test_enhanced_codebase.py --full   # Complete validation suite
"""

import os
import sys
import argparse
import traceback
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Import the enhanced modules
from gw_typing import NDArrayF, to_array, to_series, BreakpointResult, ModelParams
from overnight_full_run import (
    wilson_interval, bootstrap_mean_ci, auc_pairwise, 
    make_small_world_W, GlobalWorkspace, run_trial_fast
)

def test_type_utilities():
    """Test the type conversion utilities."""
    print("Testing type utilities...")
    # Test array conversion
    data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    array_result = to_array(data_list)
    assert isinstance(array_result, np.ndarray)
    assert array_result.shape == (5,)
    # Test series conversion
    series_result = to_series(data_list, name="test_data")
    assert isinstance(series_result, pd.Series)
    assert series_result.name == "test_data"
    assert len(series_result) == 5
    # Test with pandas Series input
    original_series = pd.Series([10, 20, 30])
    array_from_series = to_array(original_series)
    assert isinstance(array_from_series, np.ndarray)
    assert len(array_from_series) == 3
    print("✓ Type utilities tests passed")

def test_statistical_functions():
    """Test enhanced statistical functions."""
    print("Testing statistical functions...")
    lo, hi = wilson_interval(7, 10, alpha=0.05)
    assert 0 <= lo <= hi <= 1
    assert lo < hi
    np.random.seed(42)
    samples = np.random.normal(5.0, 1.0, 100)
    mean_est, lo_ci, hi_ci = bootstrap_mean_ci(samples, B=500, seed=42)
    assert abs(mean_est - 5.0) < 0.5
    assert lo_ci < mean_est < hi_ci
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.9])
    labels = np.array([0, 0, 1, 1, 1, 1])
    auc = auc_pairwise(scores, labels)
    assert 0.5 <= auc <= 1.0
    print("✓ Statistical functions tests passed")

def test_network_generation():
    """Test small-world network generation."""
    print("Testing network generation...")
    N = 32
    W = make_small_world_W(N, base_coupling=1.0, degree_frac=0.38, rewire_p=0.18, seed=42)
    assert W.shape == (N, N)
    assert np.all(np.isfinite(W))
    assert np.sum(np.abs(W)) > 0
    col_norms = np.linalg.norm(W, axis=0)
    expected_norm = 1.0
    assert np.allclose(col_norms[col_norms > 0], expected_norm, rtol=0.1)
    print("✓ Network generation tests passed")

def test_global_workspace():
    """Test Global Workspace implementation."""
    print("Testing Global Workspace...")
    N = 32; K = 5
    gw = GlobalWorkspace(N=N, K=K, tau=0.9, theta=0.55, seed=123)
    assert gw.proj.shape == (K, N)
    assert gw.broadcasts.shape == (K, N)
    assert gw.K == K
    assert gw.tau == 0.9
    assert gw.theta == 0.55
    x = np.random.normal(0, 1, size=N)
    broadcast, probs, winner_id, ignited = gw.step(x)
    assert broadcast.shape == (N,)
    assert probs.shape == (K,)
    assert 0 <= winner_id < K
    assert ignited in [0.0, 1.0]
    assert abs(np.sum(probs) - 1.0) < 1e-6
    print("✓ Global Workspace tests passed")

def test_trial_simulation():
    """Test neural trial simulation."""
    print("Testing trial simulation...")
    result = run_trial_fast(T=180, burn=40, seed=42, specs=None, noise=0.10)
    required_keys = ["tokens", "probs", "ignitions", "labels", "feats"]
    for key in required_keys:
        assert key in result
        assert isinstance(result[key], np.ndarray)
    T_effective = 180 - 40 - 1
    assert result["tokens"].shape == (T_effective,)
    assert result["probs"].shape == (T_effective,)
    assert result["ignitions"].shape == (T_effective,)
    assert result["labels"].shape == (T_effective,)
    assert result["feats"].shape == (T_effective, 6)
    assert np.all(result["tokens"] >= 0) and np.all(result["tokens"] < 5)
    assert np.all(result["probs"] >= 0) and np.all(result["probs"] <= 1)
    assert np.all(result["ignitions"] >= 0) and np.all(result["ignitions"] <= 1)
    print("✓ Trial simulation tests passed")

def test_reproducibility():
    """Test reproducibility of results."""
    print("Testing reproducibility...")
    seed = 12345
    np.random.seed(seed)
    samples = np.random.normal(0, 1, 50)
    result1 = bootstrap_mean_ci(samples, B=100, seed=seed)
    result2 = bootstrap_mean_ci(samples, B=100, seed=seed)
    assert result1 == result2
    W1 = make_small_world_W(16, 1.0, 0.3, 0.2, seed=seed)
    W2 = make_small_world_W(16, 1.0, 0.3, 0.2, seed=seed)
    assert np.allclose(W1, W2)
    gw1 = GlobalWorkspace(N=16, K=3, seed=seed)
    gw2 = GlobalWorkspace(N=16, K=3, seed=seed)
    assert np.allclose(gw1.proj, gw2.proj)
    assert np.allclose(gw1.broadcasts, gw2.broadcasts)
    print("✓ Reproducibility tests passed")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    empty_samples: List[float] = []
    mean_est, lo_ci, hi_ci = bootstrap_mean_ci(empty_samples)
    assert np.isnan(mean_est) and np.isnan(lo_ci) and np.isnan(hi_ci)
    lo, hi = wilson_interval(0, 0)
    assert lo == 0.0 and hi == 0.0
    lo, hi = wilson_interval(10, 10)
    assert 0.0 < lo < 1.0 and 0.9 < hi <= 1.0
    scores_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    labels_perfect = np.array([0, 0, 1, 1])
    auc_perfect = auc_pairwise(scores_perfect, labels_perfect)
    assert auc_perfect == 1.0
    scores_no_pos = np.array([0.1, 0.2, 0.3])
    labels_no_pos = np.array([0, 0, 0])
    auc_no_pos = auc_pairwise(scores_no_pos, labels_no_pos)
    assert auc_no_pos == 0.5
    print("✓ Edge cases tests passed")

def test_performance():
    """Test performance of key functions."""
    print("Testing performance...")
    import time
    start_time = time.time()
    large_sample = np.random.normal(0, 1, 1000)
    bootstrap_mean_ci(large_sample, B=1000, seed=42)
    bootstrap_time = time.time() - start_time
    assert bootstrap_time < 5.0, f"Bootstrap too slow: {bootstrap_time:.2f}s"
    start_time = time.time()
    make_small_world_W(64, 1.0, 0.4, 0.2, seed=42)
    network_time = time.time() - start_time
    assert network_time < 2.0, f"Network generation too slow: {network_time:.2f}s"
    start_time = time.time()
    run_trial_fast(T=200, burn=50, seed=42)
    trial_time = time.time() - start_time
    assert trial_time < 3.0, f"Trial simulation too slow: {trial_time:.2f}s"
    print("✓ Performance tests passed")

def run_test_suite(quick: bool = False) -> bool:
    """Run the complete test suite."""
    print("="*60)
    print("Enhanced GW Codebase Validation Suite")
    print("="*60)
    
    tests = [
        test_type_utilities,
        test_statistical_functions,
        test_network_generation,
        test_global_workspace,
        test_trial_simulation,
        test_reproducibility,
        test_edge_cases,
    ]
    
    if not quick:
        tests.append(test_performance)
    
    all_passed = True
    for test_func in tests:
        try:
            test_func()
        except AssertionError:
            all_passed = False
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            traceback.print_exc()
            all_passed = False
    print("="*60)
    if all_passed:
        print("All inline validation tests completed.")
    else:
        print("Some inline validation tests reported failures.")
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Test enhanced GW codebase")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--full", action="store_true", help="Run complete test suite")
    
    args = parser.parse_args()
    
    # Default to quick if neither specified
    quick_mode = args.quick or not args.full
    
    success = run_test_suite(quick=quick_mode)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
