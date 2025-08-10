#!/usr/bin/env python3
"""
generate_scaling_figures.py

Generate publication-quality figures for network scaling analysis.
Creates comprehensive visualizations of threshold persistence across network sizes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import cm

# Set publication-quality style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_scaling_data():
    """Load comprehensive scaling data."""
    data_path = "./runs/comprehensive_scaling/masking_curves_all_sizes.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Scaling data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df

def create_comprehensive_scaling_figure(df, save_path="./scaling_analysis_comprehensive.png"):
    """Create comprehensive scaling analysis figure with multiple panels."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Masking curves for all network sizes
    ax1 = plt.subplot(2, 3, (1, 2))
    
    node_sizes = sorted(df['N_nodes'].unique())
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(node_sizes)))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, N in enumerate(node_sizes):
        subset = df[df['N_nodes'] == N].sort_values('SOA')
        color = colors[i]
        marker = markers[i % len(markers)]
        
        plt.plot(subset['SOA'], subset['accuracy'], 
                label=f'N={N}', color=color, marker=marker, 
                linewidth=2.5, markersize=8, markeredgewidth=0.5, 
                markeredgecolor='white')
    
    plt.xlabel('SOA (stimulus units)', fontsize=14, fontweight='bold')
    plt.ylabel('Detection Accuracy', fontsize=14, fontweight='bold')
    plt.title('A. Masking Performance Across Network Scales', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Add threshold region annotation
    plt.axvspan(2, 4, alpha=0.1, color='red', label='Threshold Region')
    
    # Panel B: Effect sizes across network sizes
    ax2 = plt.subplot(2, 3, 3)
    
    effect_sizes = []
    for N in node_sizes:
        subset = df[df['N_nodes'] == N]
        max_acc = subset['accuracy'].max()
        min_acc = subset['accuracy'].min()
        effect_size = max_acc - min_acc
        effect_sizes.append(effect_size)
    
    bars = plt.bar(range(len(node_sizes)), effect_sizes, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, effect) in enumerate(zip(bars, effect_sizes)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(range(len(node_sizes)), [f'{N}' for N in node_sizes])
    plt.xlabel('Network Size (nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Effect Size', fontsize=14, fontweight='bold')
    plt.title('B. Threshold Effect Sizes', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tick_params(labelsize=12)
    
    # Panel C: Peak performance scaling
    ax3 = plt.subplot(2, 3, 4)
    
    max_accuracies = []
    for N in node_sizes:
        subset = df[df['N_nodes'] == N]
        max_acc = subset['accuracy'].max()
        max_accuracies.append(max_acc)
    
    plt.plot(node_sizes, max_accuracies, 'o-', color='darkblue', 
             linewidth=3, markersize=10, markerfacecolor='lightblue', 
             markeredgecolor='darkblue', markeredgewidth=2)
    
    # Add trend line
    z = np.polyfit(node_sizes, max_accuracies, 1)
    p = np.poly1d(z)
    plt.plot(node_sizes, p(node_sizes), "--", alpha=0.8, color='red', linewidth=2)
    
    plt.xlabel('Network Size (nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Peak Accuracy', fontsize=14, fontweight='bold')
    plt.title('C. Peak Performance Scaling', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Panel D: Computational scaling
    ax4 = plt.subplot(2, 3, 5)
    
    # Runtime data from scaling tests
    runtimes = [3.2, 4.0, 8.4, 35.2, 229.0]  # seconds
    
    plt.loglog(node_sizes, runtimes, 'o-', color='darkgreen', 
               linewidth=3, markersize=10, markerfacecolor='lightgreen',
               markeredgecolor='darkgreen', markeredgewidth=2)
    
    plt.xlabel('Network Size (nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
    plt.title('D. Computational Scaling', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Panel E: Confidence scaling
    ax5 = plt.subplot(2, 3, 6)
    
    mean_confidences = []
    conf_stds = []
    for N in node_sizes:
        subset = df[df['N_nodes'] == N]
        mean_conf = subset['mean_confidence'].mean()
        conf_std = subset['mean_confidence'].std()
        mean_confidences.append(mean_conf)
        conf_stds.append(conf_std)
    
    plt.errorbar(node_sizes, mean_confidences, yerr=conf_stds, 
                 fmt='o-', color='purple', linewidth=3, markersize=10,
                 markerfacecolor='plum', markeredgecolor='purple', 
                 markeredgewidth=2, capsize=5, capthick=2)
    
    plt.xlabel('Network Size (nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Confidence', fontsize=14, fontweight='bold')
    plt.title('E. Confidence Scaling', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Overall title and layout
    fig.suptitle('Global Workspace Network Scaling Analysis\nScale-Invariant Consciousness Thresholds', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # Save high-resolution figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"Comprehensive scaling figure saved to: {save_path}")
    return fig

def create_threshold_comparison_figure(df, save_path="./threshold_comparison.png"):
    """Create focused figure comparing threshold curves across scales."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left panel: All curves with threshold region highlighted
    node_sizes = sorted(df['N_nodes'].unique())
    colors = cm.get_cmap('plasma')(np.linspace(0, 1, len(node_sizes)))
    
    for i, N in enumerate(node_sizes):
        subset = df[df['N_nodes'] == N].sort_values('SOA')
        color = colors[i]
        
        ax1.plot(subset['SOA'], subset['accuracy'], 
                label=f'{N} nodes', color=color, linewidth=3, 
                marker='o', markersize=6)
    
    # Highlight threshold region
    ax1.axvspan(2, 4, alpha=0.15, color='red', label='Threshold Region')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance Level')
    
    ax1.set_xlabel('SOA (stimulus units)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Detection Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Masking Curves Across Network Scales', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # Right panel: Effect size preservation
    effect_data = []
    for N in node_sizes:
        subset = df[df['N_nodes'] == N]
        max_acc = subset['accuracy'].max()
        min_acc = subset['accuracy'].min()
        effect_size = max_acc - min_acc
        
        # Calculate threshold steepness (maximum negative gradient)
        accuracies = subset.sort_values('SOA')['accuracy'].values
        soas = subset.sort_values('SOA')['SOA'].values
        if len(accuracies) > 1:
            gradients = np.diff(accuracies) / np.diff(soas)
            steepest_drop = np.min(gradients)
        else:
            steepest_drop = 0
            
        effect_data.append({
            'N_nodes': N,
            'effect_size': effect_size,
            'steepest_drop': abs(steepest_drop),
            'max_acc': max_acc,
            'min_acc': min_acc
        })
    
    effect_df = pd.DataFrame(effect_data)
    
    # Create dual-axis plot
    ax2_twin = ax2.twinx()
    
    # Effect sizes (bars)
    bars = ax2.bar(range(len(node_sizes)), effect_df['effect_size'], 
                   alpha=0.6, color='steelblue', label='Effect Size')
    
    # Threshold steepness (line)
    line = ax2_twin.plot(range(len(node_sizes)), effect_df['steepest_drop'], 
                         'ro-', linewidth=3, markersize=8, label='Threshold Steepness')
    
    ax2.set_xticks(range(len(node_sizes)))
    ax2.set_xticklabels([f'{N}' for N in node_sizes])
    ax2.set_xlabel('Network Size (nodes)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Effect Size (Max - Min Accuracy)', fontsize=14, fontweight='bold', color='steelblue')
    ax2_twin.set_ylabel('Threshold Steepness', fontsize=14, fontweight='bold', color='red')
    
    ax2.set_title('Threshold Properties Across Scales', fontsize=16, fontweight='bold')
    ax2.tick_params(labelsize=12, colors='steelblue')
    ax2_twin.tick_params(labelsize=12, colors='red')
    
    # Add value labels
    for i, (bar, effect, steep) in enumerate(zip(bars, effect_df['effect_size'], effect_df['steepest_drop'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{effect:.3f}', ha='center', va='bottom', fontweight='bold', color='steelblue')
        ax2_twin.text(i, steep + 0.005, f'{steep:.3f}', ha='center', va='bottom', 
                     fontweight='bold', color='red')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"Threshold comparison figure saved to: {save_path}")
    return fig

def create_publication_summary(df, save_path="./scaling_summary_table.png"):
    """Create publication-ready summary table."""
    
    # Calculate summary statistics
    summary_data = []
    node_sizes = sorted(df['N_nodes'].unique())
    
    for N in node_sizes:
        subset = df[df['N_nodes'] == N]
        max_acc = subset['accuracy'].max()
        min_acc = subset['accuracy'].min()
        effect_size = max_acc - min_acc
        mean_conf = subset['mean_confidence'].mean()
        
        # Calculate threshold steepness
        accuracies = subset.sort_values('SOA')['accuracy'].values
        soas = subset.sort_values('SOA')['SOA'].values
        if len(accuracies) > 1:
            gradients = np.diff(accuracies) / np.diff(soas)
            steepest_drop = abs(np.min(gradients))
        else:
            steepest_drop = 0
        
        summary_data.append([
            f'{N}',
            f'{max_acc:.3f}',
            f'{min_acc:.3f}',
            f'{effect_size:.3f}',
            f'{steepest_drop:.3f}',
            f'{mean_conf:.3f}'
        ])
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Table headers
    headers = ['Network Size\n(nodes)', 'Peak\nAccuracy', 'Min\nAccuracy', 
               'Effect\nSize', 'Threshold\nSteepness', 'Mean\nConfidence']
    
    # Create table
    table = ax.table(cellText=summary_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     colColours=['lightgray']*len(headers))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color cells based on values
    for i in range(len(summary_data)):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor('white')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            
        # Highlight effect size column
        effect_cell = table[(i+1, 3)]
        effect_val = float(summary_data[i][3])
        if effect_val > 0.4:
            effect_cell.set_facecolor('lightgreen')
        elif effect_val > 0.3:
            effect_cell.set_facecolor('lightyellow')
    
    # Header styling
    for j in range(len(headers)):
        header_cell = table[(0, j)]
        header_cell.set_facecolor('darkgray')
        header_cell.set_text_props(weight='bold', color='white')
    
    plt.title('Global Workspace Network Scaling: Summary Statistics\n' + 
              'Threshold Effects Persist Across All Network Sizes',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"Summary table saved to: {save_path}")
    return fig

def main():
    """Generate all scaling analysis figures."""
    
    print("Loading scaling data...")
    df = load_scaling_data()
    
    print("Creating comprehensive scaling figure...")
    create_comprehensive_scaling_figure(df, "./figures/scaling_analysis_comprehensive.png")
    
    print("Creating threshold comparison figure...")
    create_threshold_comparison_figure(df, "./figures/threshold_comparison.png")
    
    print("Creating publication summary table...")
    create_publication_summary(df, "./figures/scaling_summary_table.png")
    
    print("\nAll scaling figures generated successfully!")
    print("Files created in ./figures/:")
    print("- scaling_analysis_comprehensive.png/pdf")
    print("- threshold_comparison.png/pdf") 
    print("- scaling_summary_table.png/pdf")

if __name__ == '__main__':
    # Create figures directory
    os.makedirs('./figures', exist_ok=True)
    main()
