#!/usr/bin/env python3
"""
make_figures.py

Generate publication-quality figures from experimental data.
This script creates Figure 1-3 for the manuscript with proper styling and error bars.

Usage:
    python make_figures.py --data_dir ./runs/advanced_full --output_dir ./manuscript/figures
    python make_figures.py --masking ./v6f/masking_curve_ci.csv --blink ./v6f/blink_curve_ci.csv --infoflow ./runs/full_infoflow/infoflow_pci_sweep.csv --output_dir ./figures
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })

def create_figure_1(masking_file: str, output_dir: str):
    """Create Figure 1: Visual masking threshold curves."""
    try:
        df = pd.read_csv(masking_file)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Plot main curve with error bars
        ax.errorbar(df['SOA'], df['ReportAcc'], 
                   yerr=[df['ReportAcc'] - df['ReportAcc_lo'], 
                         df['ReportAcc_hi'] - df['ReportAcc']], 
                   marker='o', linewidth=2, capsize=5, capthick=2,
                   color='#2E8B57', label='Report Accuracy')
        
        ax.set_xlabel('SOA (stimulus onset asynchrony)')
        ax.set_ylabel('Report Accuracy')
        ax.set_title('Figure 1: Visual Masking Threshold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add threshold annotation if clear transition visible
        if len(df) > 3:
            mid_idx = len(df) // 2
            ax.axvline(df.iloc[mid_idx]['SOA'], color='red', linestyle='--', alpha=0.7, label='Threshold region')
        
        ax.legend()
        plt.tight_layout()
        
        # Save in both formats
        png_path = os.path.join(output_dir, 'Figure_1_Masking_Threshold.png')
        pdf_path = os.path.join(output_dir, 'Figure_1_Masking_Threshold.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Figure 1 saved: {png_path}, {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error creating Figure 1: {e}")
        return False
    return True

def create_figure_2(blink_file: str, output_dir: str):
    """Create Figure 2: Attentional blink threshold curves."""
    try:
        df = pd.read_csv(blink_file)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Plot T2 accuracy with error bars
        ax.errorbar(df['Lag'], df['T2Acc'], 
                   yerr=[df['T2Acc'] - df['T2_lo'], 
                         df['T2_hi'] - df['T2Acc']], 
                   marker='s', linewidth=2, capsize=5, capthick=2,
                   color='#4169E1', label='T2 Accuracy')
        
        ax.set_xlabel('T1-T2 Lag (time steps)')
        ax.set_ylabel('T2 Report Accuracy')
        ax.set_title('Figure 2: Attentional Blink')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Highlight typical blink period
        if len(df) > 2:
            blink_region = df['Lag'].iloc[1:3]
            ax.axvspan(blink_region.min(), blink_region.max(), alpha=0.2, color='red', label='Blink period')
        
        ax.legend()
        plt.tight_layout()
        
        # Save in both formats
        png_path = os.path.join(output_dir, 'Figure_2_Blink_Threshold.png')
        pdf_path = os.path.join(output_dir, 'Figure_2_Blink_Threshold.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Figure 2 saved: {png_path}, {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error creating Figure 2: {e}")
        return False
    return True

def create_figure_3(infoflow_file: str, output_dir: str):
    """Create Figure 3: Information flow analysis."""
    try:
        df = pd.read_csv(infoflow_file)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        # Top panel: Transfer Entropy
        if 'TE_bits' in df.columns:
            ax1.plot(df['beta'], df['TE_bits'], marker='o', linewidth=2, color='#FF6347', label='Transfer Entropy')
            ax1.set_ylabel('Transfer Entropy (bits)')
        else:
            # Fallback if column name different
            te_col = [col for col in df.columns if 'TE' in col or 'transfer' in col.lower()]
            if te_col:
                ax1.plot(df['beta'], df[te_col[0]], marker='o', linewidth=2, color='#FF6347', label='Transfer Entropy')
                ax1.set_ylabel('Transfer Entropy')
        
        ax1.set_title('Figure 3A: Information Flow Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom panel: PCI or connectivity measure
        if 'PCI' in df.columns:
            ax2.plot(df['beta'], df['PCI'], marker='s', linewidth=2, color='#32CD32', label='Participation Coefficient')
            ax2.set_ylabel('Participation Coefficient')
        else:
            # Look for any connectivity-related measure
            conn_cols = [col for col in df.columns if any(term in col.lower() for term in ['pci', 'participation', 'connectivity', 'integration'])]
            if conn_cols:
                ax2.plot(df['beta'], df[conn_cols[0]], marker='s', linewidth=2, color='#32CD32', label=conn_cols[0])
                ax2.set_ylabel(conn_cols[0])
        
        ax2.set_xlabel('β (rewiring parameter)')
        ax2.set_title('Figure 3B: Network Integration')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add threshold line if clear transition
        if len(df) > 5:
            threshold_beta = 0.35  # Based on manuscript results
            ax1.axvline(threshold_beta, color='red', linestyle='--', alpha=0.7, label='Threshold (β≈0.35)')
            ax2.axvline(threshold_beta, color='red', linestyle='--', alpha=0.7, label='Threshold (β≈0.35)')
            ax1.legend()
            ax2.legend()
        
        plt.tight_layout()
        
        # Save in both formats
        png_path = os.path.join(output_dir, 'Figure_3_Information_Flow.png')
        pdf_path = os.path.join(output_dir, 'Figure_3_Information_Flow.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Figure 3 saved: {png_path}, {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error creating Figure 3: {e}")
        return False
    return True

def validate_files(*file_paths):
    """Validate that required data files exist."""
    missing = []
    for path in file_paths:
        if path and not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate publication figures from experimental data')
    
    # Option 1: Specify data directory (looks for standard filenames)
    parser.add_argument('--data_dir', type=str, help='Directory containing standard data files')
    
    # Option 2: Specify individual files
    parser.add_argument('--masking', type=str, help='Path to masking_curve_ci.csv file')
    parser.add_argument('--blink', type=str, help='Path to blink_curve_ci.csv file')
    parser.add_argument('--infoflow', type=str, help='Path to information flow data file')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.data_dir:
        masking_file = os.path.join(args.data_dir, 'masking_curve_ci.csv')
        blink_file = os.path.join(args.data_dir, 'blink_curve_ci.csv')
        # Look for information flow file with flexible naming
        infoflow_candidates = [
            os.path.join(args.data_dir, 'infoflow_pci_sweep.csv'),
            os.path.join(args.data_dir, 'information_flow.csv'),
            os.path.join(args.data_dir, 'connectivity_sweep.csv')
        ]
        infoflow_file = next((f for f in infoflow_candidates if os.path.exists(f)), None)
    else:
        masking_file = args.masking
        blink_file = args.blink
        infoflow_file = args.infoflow
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup matplotlib
    setup_matplotlib()
    
    print("🎨 Generating publication figures...")
    
    success_count = 0
    
    # Generate Figure 1
    if masking_file and validate_files(masking_file):
        if create_figure_1(masking_file, args.output_dir):
            success_count += 1
    else:
        print("⚠️  Skipping Figure 1: masking data not found")
    
    # Generate Figure 2
    if blink_file and validate_files(blink_file):
        if create_figure_2(blink_file, args.output_dir):
            success_count += 1
    else:
        print("⚠️  Skipping Figure 2: blink data not found")
    
    # Generate Figure 3
    if infoflow_file and validate_files(infoflow_file):
        if create_figure_3(infoflow_file, args.output_dir):
            success_count += 1
    else:
        print("⚠️  Skipping Figure 3: information flow data not found")
    
    print(f"\n✅ Generated {success_count}/3 figures successfully")
    
    if success_count > 0:
        print(f"📁 Figures saved to: {args.output_dir}")
        print("🎯 Ready for manuscript submission!")
    else:
        print("❌ No figures generated. Check data file paths.")
        sys.exit(1)

if __name__ == "__main__":
    main()
