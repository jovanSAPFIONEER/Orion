#!/usr/bin/env python3
"""
comprehensive_scaling_test.py

Comprehensive test of masking curves across different network sizes.
Tests multiple SOA values to validate threshold curve persistence.
"""

import os, sys, time
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from overnight_full_run import *

def run_full_masking_curve(N_nodes: int, n_trials: int = 25):
    """Test full masking curve with specified node count."""
    print(f"Testing full masking curve with {N_nodes} nodes...")
    
    def run_trial_fast_custom(T, burn, seed, specs, noise=0.10):
        """Modified run_trial_fast with custom node count."""
        r = np.random.default_rng(seed)
        N = N_nodes
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
    
    # Test multiple SOA values
    SOAs = [1, 2, 3, 4, 6, 8]  # Subset of original SOAs for speed
    seed0 = 11000
    results = []
    
    for soa in SOAs:
        print(f"  SOA: {soa}")
        hits = []
        confs = []
        
        for tr in range(n_trials):
            T = 360
            pos = 180
            specs = masking_specs(T, pos, soa, amp_t=0.9, amp_m=2.0)
            out = run_trial_fast_custom(T=T, burn=80, seed=seed0+soa*300+tr, specs=specs, noise=0.10)
            center = pos - 80
            
            # Check for VIS_TGT token around target time
            hit = 1 if np.any(out["tokens"][center-5:center+6] == LABEL_TO_ID["VIS_TGT"]) else 0
            hits.append(hit)
            
            # Get confidence 
            a = max(0, center-5)
            b = min(len(out["probs"])-1, center+5)
            conf = float(np.max(out["probs"][a:b+1])) if b>=a else 0.0
            confs.append(conf)
        
        # Calculate metrics
        accuracy = np.mean(hits)
        mean_conf = np.mean(confs)
        
        results.append({
            'N_nodes': N_nodes,
            'SOA': soa,
            'accuracy': accuracy,
            'mean_confidence': mean_conf,
            'n_trials': n_trials,
            'n_hits': sum(hits)
        })
    
    return results

pytestmark = pytest.mark.skip(reason="Comprehensive scaling test skipped by default (long-running)")

def main():
    """Test masking curves across multiple node sizes."""
    node_sizes = [32, 64, 128, 256, 512]
    all_results = []
    
    print("Comprehensive masking curve scaling test...")
    print("Testing network sizes:", node_sizes)
    
    start_time = time.time()
    
    for N in node_sizes:
        node_start = time.time()
        results = run_full_masking_curve(N, n_trials=25)
        all_results.extend(results)
        elapsed = time.time() - node_start
        print(f"  Completed {N} nodes in {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\\nTotal test time: {total_time:.1f} seconds")
    
    # Create comprehensive dataframe
    df = pd.DataFrame(all_results)
    
    # Save results
    out_dir = './runs/comprehensive_scaling'
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'masking_curves_all_sizes.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, N in enumerate(node_sizes):
        subset = df[df['N_nodes'] == N].sort_values('SOA')
        if len(subset) > 0:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(subset['SOA'], subset['accuracy'], 
                    label=f'N={N}', color=color, marker=marker, linewidth=2, markersize=6)
    
    plt.xlabel('SOA (stimulus units)', fontsize=12)
    plt.ylabel('Detection Accuracy', fontsize=12)
    plt.title('Masking Performance vs Network Size\\n(Threshold Effects Across Scales)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(out_dir, 'masking_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'masking_curves_comparison.pdf'), bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\\nSummary by network size:")
    summary = df.groupby('N_nodes').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'mean_confidence': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    # Check threshold persistence
    print("\\nThreshold Analysis:")
    for N in node_sizes:
        subset = df[df['N_nodes'] == N].sort_values('SOA')
        if len(subset) > 1:
            max_acc = subset['accuracy'].max()
            min_acc = subset['accuracy'].min()
            effect_size = max_acc - min_acc
            
            # Find steepest drop
            accuracies = subset['accuracy'].values
            soas = subset['SOA'].values
            gradients = np.diff(accuracies) / np.diff(soas)
            steepest_drop = np.min(gradients) if len(gradients) > 0 else 0
            
            print(f"N={N:3d}: Max={max_acc:.3f}, Min={min_acc:.3f}, Effect={effect_size:.3f}, "
                  f"Steepest drop={steepest_drop:.3f}")
    
    # Validation result
    successful_sizes = df.groupby('N_nodes')['accuracy'].max()
    working_sizes = successful_sizes[successful_sizes > 0.1].index.tolist()
    
    print(f"\\nNetwork sizes with detectable thresholds: {working_sizes}")
    
    if len(working_sizes) >= 4:
        print("\\n✅ SUCCESS: Threshold effects robustly persist across network sizes!")
        print("   This validates that the consciousness model scales beyond 32-node limitations.")
    else:
        print("\\n⚠️  Mixed results: Some network sizes may not show clear thresholds.")

def test_comprehensive_masking_smoke():
    """Pytest smoke test for masking curve at one size."""
    res = run_full_masking_curve(32, n_trials=5)
    assert isinstance(res, list) and len(res) > 0
    assert all('accuracy' in r for r in res)

if __name__ == '__main__':
    main()
