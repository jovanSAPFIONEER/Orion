#!/usr/bin/env python3
"""
correlation_analysis.py

Analyze correlations between ReportAcc and info-flow metrics (TE, Granger, Participation, PCI_like)
with bootstrap confidence intervals across coupling strengths.

This script performs robust correlation analysis to assess the relationship between:
- Behavioral accuracy (ReportAcc) 
- Information flow metrics from neural dynamics
- Consciousness-related measures (ignition, participation)

The analysis includes bootstrap confidence intervals and handles missing data gracefully.

Usage:
  python correlation_analysis.py --full ./runs/full_infoflow_v2_heavy --out ./runs/correlation_analysis
"""

import os, argparse, json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

# Import type aliases
from gw_typing import NDArrayF, DataFrame, CIBounds

def bootstrap_correlation(x: NDArrayF, y: NDArrayF, B: int = 2000, seed: int = 42) -> Dict[str, float]:
    """
    Bootstrap confidence intervals for Pearson correlation coefficient.
    
    Args:
        x: First variable data
        y: Second variable data  
        B: Number of bootstrap resamples
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with keys 'r' (correlation), 'lo' (lower CI), 'hi' (upper CI)
    """
    if len(x) != len(y) or len(x) < 3:
        return {"r": np.nan, "lo": np.nan, "hi": np.nan}
    
    rng = np.random.default_rng(seed)
    n = len(x)
    
    # Original correlation
    r_orig = np.corrcoef(x, y)[0, 1]
    if np.isnan(r_orig):
        return {"r": np.nan, "lo": np.nan, "hi": np.nan}
    
    # Bootstrap
    rs: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        x_boot = x[idx]
        y_boot = y[idx]
        r_boot = np.corrcoef(x_boot, y_boot)[0, 1]
        if not np.isnan(r_boot):
            rs.append(float(r_boot))
    
    if len(rs) == 0:
        return {"r": float(r_orig), "lo": np.nan, "hi": np.nan}
    
    rs_array = np.array(rs)
    return {
        "r": float(r_orig),
        "lo": float(np.percentile(rs_array, 2.5)),
        "hi": float(np.percentile(rs_array, 97.5))
    }


def noise_ceiling(data_matrix: NDArrayF) -> Tuple[float, float]:
    """
    Compute upper and lower noise ceiling correlations following Ince et al. (2022).
    
    The noise ceiling provides bounds on the best possible model performance given
    subject-to-subject variability in the data.
    
    Args:
        data_matrix: Array of shape (n_subjects, n_conditions) containing accuracy/ratings
        
    Returns:
        Tuple of (upper_bound, lower_bound) noise ceiling correlations
    """
    n_subjects, n_conditions = data_matrix.shape
    
    if n_subjects < 2 or n_conditions < 2:
        return np.nan, np.nan
    
    # Grand average across all subjects
    grand_average = data_matrix.mean(axis=0)
    
    # Upper bound: correlation of each subject with grand average
    upper_correlations = []
    for i in range(n_subjects):
        if not (np.isnan(data_matrix[i]).any() or np.isnan(grand_average).any()):
            r = pearsonr(data_matrix[i], grand_average)[0]
            if not np.isnan(r):
                upper_correlations.append(r)
    
    upper_bound = np.mean(upper_correlations) if upper_correlations else np.nan
    
    # Lower bound: leave-one-subject-out correlations
    lower_correlations = []
    for i in range(n_subjects):
        # Average of all other subjects (leave subject i out)
        other_subjects = np.delete(data_matrix, i, axis=0)
        loo_average = other_subjects.mean(axis=0)
        
        if not (np.isnan(data_matrix[i]).any() or np.isnan(loo_average).any()):
            r = pearsonr(data_matrix[i], loo_average)[0]
            if not np.isnan(r):
                lower_correlations.append(r)
    
    lower_bound = np.mean(lower_correlations) if lower_correlations else np.nan
    
    return float(upper_bound), float(lower_bound)


def loso_lower_bound(data_matrix: NDArrayF) -> float:
    """
    Compute leave-one-subject-out (LOSO) lower bound accuracy.
    
    For each subject, predict their responses using the average of all other subjects,
    then compute the mean squared error. Lower MSE indicates better prediction.
    
    Args:
        data_matrix: Array of shape (n_subjects, n_conditions)
        
    Returns:
        Mean squared error across leave-one-out predictions
    """
    n_subjects, n_conditions = data_matrix.shape
    
    if n_subjects < 2:
        return np.nan
    
    mse_values = []
    for i in range(n_subjects):
        # Prediction: average of all other subjects
        other_subjects = np.delete(data_matrix, i, axis=0)
        prediction = other_subjects.mean(axis=0)
        
        # Actual: this subject's data
        actual = data_matrix[i]
        
        # MSE for this leave-one-out fold
        if not (np.isnan(actual).any() or np.isnan(prediction).any()):
            mse = np.mean((actual - prediction) ** 2)
            mse_values.append(mse)
    
    return float(np.mean(mse_values)) if mse_values else float(np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', type=str, required=True, help='Path to full_infoflow_v2 directory')
    ap.add_argument('--out', type=str, required=True, help='Output directory')
    ap.add_argument('--b_gain', type=float, default=0.22, help='Broadcast gain to analyze')
    ap.add_argument('--boots', type=int, default=2000, help='Bootstrap samples')
    ap.add_argument('--noise_ceiling', action='store_true', help='Compute noise ceiling and LOSO bounds')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Load sweep data
    sweep_file = os.path.join(args.full, "infoflow_pci_sweep.csv")
    if not os.path.exists(sweep_file):
        raise FileNotFoundError(f"Missing sweep file: {sweep_file}")
    
    sweep = pd.read_csv(sweep_file)
    
    # Filter to chosen broadcast gain
    df = sweep[sweep["b_gain"].round(6) == round(args.b_gain, 6)].copy()
    if len(df) == 0:
        raise ValueError(f"No data found for b_gain={args.b_gain}")
    
    df = df.sort_values("g_scale")
    
    # Define info-flow metrics to correlate with ReportAcc
    metrics = ["TE_bits", "Granger_logVR", "Participation", "PCI_like"]
    
    results = []
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: {metric} not found in data, skipping")
            continue
            
        # Get clean data
        clean_df = df[["g_scale", "ReportAcc", metric]].dropna()
        
        if len(clean_df) < 3:
            print(f"Warning: Insufficient data for {metric}, skipping")
            continue
        
        x = clean_df["ReportAcc"].values
        y = clean_df[metric].values
        
        # Bootstrap correlation
        corr_stats = bootstrap_correlation(x, y, B=args.boots)
        
        # Also correlate with g_scale for context
        g_scale = clean_df["g_scale"].values
        corr_g_report = bootstrap_correlation(g_scale, x, B=args.boots)
        corr_g_metric = bootstrap_correlation(g_scale, y, B=args.boots)
        
        results.append({
            "metric": metric,
            "n_points": len(clean_df),
            "g_range": f"{g_scale.min():.2f}-{g_scale.max():.2f}",
            "corr_ReportAcc": corr_stats["r"],
            "corr_ReportAcc_lo": corr_stats["lo"],
            "corr_ReportAcc_hi": corr_stats["hi"],
            "corr_g_ReportAcc": corr_g_report["r"],
            "corr_g_metric": corr_g_metric["r"],
            "significant": (corr_stats["lo"] > 0) or (corr_stats["hi"] < 0)
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(args.out, "correlation_summary.csv")
    results_df.to_csv(results_file, index=False)
    
    # Save detailed JSON
    json_file = os.path.join(args.out, "correlation_analysis.json")
    with open(json_file, "w") as f:
        json.dump({
            "b_gain": args.b_gain,
            "bootstrap_samples": args.boots,
            "results": results
        }, f, indent=2)
    
    # Compute noise ceiling and LOSO bounds if requested
    if args.noise_ceiling:
        print("\n=== COMPUTING NOISE CEILING & LOSO BOUNDS ===")
        
        # Check if we have appropriate data structure for noise ceiling
        g_values = sorted(df["g_scale"].unique())
        
        if len(g_values) < 4:
            print("WARNING: Insufficient g_scale diversity for meaningful noise ceiling analysis")
            print("Noise ceiling requires multiple 'subjects' (different parameter settings)")
            return
        
        # For noise ceiling, we need to restructure the data properly
        # We'll treat different g_scale values as "subjects" and create a condition matrix
        print(f"Using {len(g_values)} g_scale values as 'subjects' for noise ceiling analysis")
        print("NOTE: This treats different coupling strengths as independent observations")
        
        # For each metric, create subject x condition matrix  
        noise_ceiling_results = {}
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            # Create matrix where each g_scale is a "subject" and we use ReportAcc + metric as "conditions"
            matrix_data = []
            for g_val in g_values:
                g_subset = df[df["g_scale"] == g_val]
                if len(g_subset) > 0:
                    # For this analysis, we create multiple "conditions" from available metrics
                    row_data = [
                        g_subset["ReportAcc"].mean(),
                        g_subset[metric].mean() 
                    ]
                    matrix_data.append(row_data)
            
            if len(matrix_data) >= 4:  # Need at least 4 "subjects" for meaningful analysis
                data_matrix = np.array(matrix_data)
                
                # Check for sufficient variability
                if np.std(data_matrix[:, 0]) < 1e-6 or np.std(data_matrix[:, 1]) < 1e-6:
                    print(f"{metric}: Insufficient variability for noise ceiling analysis")
                    continue
                
                # Compute noise ceiling for this metric
                upper_nc, lower_nc = noise_ceiling(data_matrix)
                loso_mse = loso_lower_bound(data_matrix)
                
                # Sanity check results
                if abs(upper_nc) > 0.99 and abs(lower_nc) > 0.99:
                    print(f"{metric}: WARNING - Perfect correlations detected, check data structure")
                
                noise_ceiling_results[metric] = {
                    "upper_noise_ceiling": upper_nc,
                    "lower_noise_ceiling": lower_nc,
                    "loso_mse": loso_mse,
                    "n_subjects": len(matrix_data),
                    "data_std_reportacc": float(np.std(data_matrix[:, 0])),
                    "data_std_metric": float(np.std(data_matrix[:, 1]))
                }
                
                print(f"{metric}:")
                print(f"  Noise ceiling (upper/lower): {upper_nc:.3f} / {lower_nc:.3f}")
                print(f"  LOSO MSE: {loso_mse:.4f}")
                print(f"  Data variability (ReportAcc/Metric): {np.std(data_matrix[:, 0]):.4f} / {np.std(data_matrix[:, 1]):.4f}")
                print(f"  N parameter settings: {len(matrix_data)}")
        
        if not noise_ceiling_results:
            print("No valid noise ceiling results computed - check data structure and variability")
        else:
            # Add noise ceiling results to JSON output
            with open(json_file, "r") as f:
                json_data = json.load(f)
            json_data["noise_ceiling_analysis"] = noise_ceiling_results
            json_data["noise_ceiling_note"] = "g_scale values treated as independent subjects"
            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=2)
    
    # Create visualization
    pdf_file = os.path.join(args.out, "correlation_analysis.pdf")
    with PdfPages(pdf_file) as pdf:
        # Correlation matrix plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if metric not in df.columns:
                continue
                
            ax = axes[i]
            clean_df = df[["ReportAcc", metric]].dropna()
            
            if len(clean_df) >= 3:
                x = clean_df["ReportAcc"].values
                y = clean_df[metric].values
                
                ax.scatter(x, y, alpha=0.7, s=50)
                
                # Add correlation info
                result = next((r for r in results if r["metric"] == metric), None)
                if result:
                    r = result["corr_ReportAcc"]
                    lo = result["corr_ReportAcc_lo"]
                    hi = result["corr_ReportAcc_hi"]
                    sig_str = "***" if result["significant"] else ""
                    
                    ax.set_title(f'{metric} vs ReportAcc\nr = {r:.3f} [{lo:.3f}, {hi:.3f}] {sig_str}')
                
                # Fit line
                if not np.isnan(r) and len(x) > 2:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 50)
                    ax.plot(x_line, p(x_line), '--', alpha=0.8, color='red')
            
            ax.set_xlabel('ReportAcc')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Summary table plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = []
        table_data.append(['Metric', 'N', 'g_range', 'r(ReportAcc)', '95% CI', 'Significant'])
        
        for result in results:
            sig_mark = "***" if result["significant"] else ""
            ci_str = f"[{result['corr_ReportAcc_lo']:.3f}, {result['corr_ReportAcc_hi']:.3f}]"
            table_data.append([
                result["metric"],
                str(result["n_points"]),
                result["g_range"],
                f"{result['corr_ReportAcc']:.3f}",
                ci_str,
                sig_mark
            ])
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color significant results
        for i, result in enumerate(results):
            if result["significant"]:
                for j in range(len(table_data[0])):
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
        
        ax.set_title(f'Correlation Analysis: Info-Flow Metrics vs ReportAcc\n(b_gain={args.b_gain}, Bootstrap N={args.boots})', 
                    fontsize=14, pad=20)
        
        pdf.savefig(fig)
        plt.close()
    
    print(f"Correlation analysis complete!")
    print(f"Results saved to: {results_file}")
    print(f"Detailed analysis: {json_file}")
    print(f"Visualization: {pdf_file}")
    
    # Print summary
    print("\n=== CORRELATION SUMMARY ===")
    for result in results:
        sig_str = " ***SIGNIFICANT***" if result["significant"] else ""
        print(f"{result['metric']}: r={result['corr_ReportAcc']:.3f} "
              f"[{result['corr_ReportAcc_lo']:.3f}, {result['corr_ReportAcc_hi']:.3f}]{sig_str}")
    
    # Print noise ceiling summary if computed
    if args.noise_ceiling and 'noise_ceiling_results' in locals():
        print("\n=== NOISE CEILING & LOSO SUMMARY ===")
        for metric, nc_data in noise_ceiling_results.items():
            print(f"{metric}:")
            print(f"  Noise ceiling: [{nc_data['lower_noise_ceiling']:.3f}, {nc_data['upper_noise_ceiling']:.3f}]")
            print(f"  LOSO MSE: {nc_data['loso_mse']:.4f}")
            print(f"  Performance context: Model correlations should fall within noise ceiling bounds")

if __name__ == "__main__":
    main()
