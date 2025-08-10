#!/usr/bin/env python3
"""
test_node_scaling.py

Quick test to verify GlobalWorkspace works with different node counts
by running the masking task with different network sizes.
"""

import os, sys
import numpy as np
import pandas as pd
from overnight_full_run import *

def run_single_node_size(N_nodes: int, n_trials: int = 20):
    """Test masking task with specified node count."""
    print(f"Testing {N_nodes} nodes with {n_trials} trials...")
    
    # Override the hardcoded N=32 in run_trial_fast by monkey-patching
    original_run_trial_fast = globals()['run_trial_fast']
    
    def run_trial_fast_custom(T, burn, seed, specs, noise=0.10):
        """Modified run_trial_fast with custom node count."""
        r = np.random.default_rng(seed)
        N = N_nodes  # Use custom node count
        g = 0.9
        W = make_small_world_W(N, 1.0, 0.38, 0.18, seed)
        Win_v = r.normal(0.0, 1.0, size=N)
        Win_a = r.normal(0.0, 1.0, size=N)
        Win_c = r.normal(0.0, 1.0, size=N)
        proj_R = r.normal(0.0, 1.0, size=(N, 6))
        proj_R /= (np.linalg.norm(proj_R, axis=0, keepdims=True)+1e-8)

        x = np.zeros((N, T))
        x[:,0] = r.normal(0, 0.1, size=N)
        gw = GlobalWorkspace(N=N, K=len(VOCAB), tau=0.9, theta=0.55, seed=seed+1)

        inputs = make_env_inputs(T, specs) if specs is not None else {
            "V":np.zeros(T),"A":np.zeros(T),"C":np.zeros(T),"y":np.zeros(T, dtype=int)
        }
        y = inputs["y"]
        tokens = []
        ignitions = []
        probs = []
        feats = []
        labels = []

        for t in range(T-1):
            z_int = g * (W @ x[:, t])
            z_ext = 0.24*(Win_v*inputs["V"][t] + Win_a*inputs["A"][t] + Win_c*inputs["C"][t]) + \
                    noise*np.random.default_rng(900+t).normal(0,1,size=N)
            x[:, t+1] = tanh(z_int + z_ext)

            feat = (proj_R.T @ x[:, t+1]).ravel()
            feats.append(feat)
            labels.append(int(y[t]))

            bcast, p, k, ign = gw.step(x[:, t+1])
            x[:, t+1] = tanh(x[:, t+1] + 0.22*bcast)
            tokens.append(k)
            ignitions.append(float(ign))
            probs.append(float(p[k]))

        sl = slice(burn, T-1)
        return {
            "tokens": np.array(tokens[sl], dtype=int),
            "probs": np.array(probs[sl], dtype=float),
            "ignitions": np.array(ignitions[sl], dtype=float),
            "labels": np.array(labels[sl], dtype=int),
            "feats": np.array(feats[sl], dtype=float),
        }
    
    # Temporarily replace the function
    globals()['run_trial_fast'] = run_trial_fast_custom
    
    try:
        # Test a single SOA condition
        soa = 100  # Middle SOA value
        seed0 = 11000
        hits = []
        confs = []
        
        for tr in range(n_trials):
            T = 360
            pos = 180
            specs = masking_specs(T, pos, soa, amp_t=0.9, amp_m=2.0)
            out = run_trial_fast_custom(T=T, burn=80, seed=seed0+soa*300+tr, specs=specs, noise=0.10)
            center = pos - 80
            
            # Check for VIS_TGT token around target time (same logic as original)
            hit = 1 if np.any(out["tokens"][center-5:center+6] == LABEL_TO_ID["VIS_TGT"]) else 0
            hits.append(hit)
            
            # Get confidence 
            a = max(0, center-5)
            b = min(len(out["probs"])-1, center+5)
            conf = float(np.max(out["probs"][a:b+1])) if b>=a else 0.0
            confs.append(conf)
        
        # Calculate accuracy
        accuracy = np.mean(hits)
        mean_conf = np.mean(confs)
        
        result = {
            'N_nodes': N_nodes,
            'accuracy': accuracy,
            'mean_confidence': mean_conf,
            'n_trials': n_trials,
            'SOA': soa
        }
        
        print(f"  Accuracy: {accuracy:.3f}, Mean confidence: {mean_conf:.3f}")
        return result
        
    finally:
        # Restore original function
        globals()['run_trial_fast'] = original_run_trial_fast
    
def main():
    """Test multiple node sizes."""
    node_sizes = [32, 64, 128, 256]  # Start with 32 as reference
    results = []
    
    print("Testing node size scaling...")
    
    for N in node_sizes:
        result = run_single_node_size(N, n_trials=30)
        results.append(result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    print("\\nResults:")
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs('./runs/node_scaling', exist_ok=True)
    df.to_csv('./runs/node_scaling/node_scaling_test.csv', index=False)
    
    # Check if thresholds persist (non-zero accuracy)
    successful_sizes = df[df['accuracy'] > 0]['N_nodes'].tolist()
    print(f"\\nSizes with detectable thresholds: {successful_sizes}")
    
    if len(successful_sizes) > 1:
        print("✓ Threshold effects persist across multiple network sizes!")
    else:
        print("⚠ Threshold effects may be specific to certain network sizes")

def test_node_scaling_smoke():
    """Pytest smoke test: run scaling for a small subset to ensure it executes."""
    res = run_single_node_size(32, n_trials=5)
    assert 'accuracy' in res and 0.0 <= res['accuracy'] <= 1.0

if __name__ == '__main__':
    main()
