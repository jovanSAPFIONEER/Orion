# Global Workspace Consciousness Model - User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Running Experiments](#running-experiments)
5. [Analysis Scripts](#analysis-scripts)
6. [Output Interpretation](#output-interpretation)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows PowerShell or Command Prompt

### Setup Instructions

1. **Clone or download the repository:**
   ```powershell
   git clone https://github.com/[username]/gw-consciousness-thresholds.git
   cd gw-consciousness-thresholds
   ```

2. **Install required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```powershell
   python test_environment.py
   ```

### Required Packages
- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing (for statistics)

## Quick Start

### Running a Basic Experiment

The simplest way to get started is running a quick test:

```powershell
python overnight_full_run.py --out ./runs/quick_test --n_reps 10
```

This will:
- Run all four consciousness paradigms (masking, blink, change blindness, dual-task)
- Use 10 repetitions (faster for testing)
- Save results to `./runs/quick_test/`
- Generate a PDF report with key findings

### Full Overnight Run

For publication-quality results:

```powershell
python overnight_full_run.py --out ./runs/full_experiment --n_reps 100
```

**Warning**: This will take 6-12 hours depending on your computer.

## Core Components

### Main Scripts

#### `overnight_full_run.py`
The primary script for running consciousness experiments.

**Key Parameters:**
- `--out`: Output directory for results
- `--n_reps`: Number of repetitions per condition (default: 100)
- `--betas`: Custom beta values for connectivity sweep
- `--seed`: Random seed for reproducibility

**Example Usage:**
```powershell
# Basic run
python overnight_full_run.py --out ./runs/experiment1

# Custom beta range
python overnight_full_run.py --out ./runs/custom --betas 0.1 0.2 0.3 0.4 0.5

# Reproducible run
python overnight_full_run.py --out ./runs/reproducible --seed 42
```

#### `compare_versions_effects_v2.py`
Compares results from different model versions or conditions.

**Usage:**
```powershell
python compare_versions_effects_v2.py --v6m ./v6m --v6f ./v6f --advanced ./runs/full_experiment --out ./runs/comparison
```

### Helper Scripts

#### `correlation_analysis.py`
Analyzes correlations between network properties and consciousness measures.

#### `enhanced_trial_analysis.py`
Detailed trial-level analysis with regression modeling.

#### `generate_variants.py`
Creates model variants for parameter exploration.

## Running Experiments

### Standard Workflow

1. **Initial Test Run** (5-10 minutes):
   ```powershell
   python overnight_full_run.py --out ./runs/test --n_reps 5
   ```

2. **Review Test Results**:
   - Check `./runs/test/` for output files
   - Open PDF report to verify functionality

3. **Full Experiment** (6-12 hours):
   ```powershell
   python overnight_full_run.py --out ./runs/full_experiment
   ```

4. **Analysis and Comparison**:
   ```powershell
   python compare_versions_effects_v2.py --v6m ./baseline --v6f ./enhanced --advanced ./runs/full_experiment --out ./runs/analysis
   ```

### Experiment Types

#### 1. Basic Consciousness Paradigms
- **Visual Masking**: Tests threshold-like access to consciousness
- **Attentional Blink**: Examines temporal limits of awareness
- **Change Blindness**: Probes sustained attention and change detection
- **Dual-Task**: Measures capacity limitations

#### 2. Connectivity Sweep
Systematically varies small-world connectivity parameter (β) from 0.0 to 1.0:
- β = 0.0: Regular lattice (high clustering, long paths)
- β = 0.5: Optimal small-world (balanced)
- β = 1.0: Random network (low clustering, short paths)

#### 3. Information Flow Analysis
Computes communication measures between network nodes:
- **Transfer Entropy**: Information flow from one node to another
- **Granger Causality**: Predictive relationships
- **Participation Coefficient**: Node integration measure

## Analysis Scripts

### Primary Analyses

#### Breakpoint Detection
Identifies sharp thresholds in connectivity-consciousness relationships:

```powershell
python enhanced_trial_analysis.py --data ./runs/full_experiment --out ./runs/breakpoint_analysis
```

#### Statistical Validation
Tests significance of threshold effects:

```powershell
python correlation_analysis.py --data ./runs/full_experiment --out ./runs/statistical_tests
```

### Secondary Analyses

#### Model Comparison
```powershell
python compare_versions_effects_v2.py --v6m ./baseline --v6f ./enhanced --advanced ./runs/full_experiment --out ./runs/comparison
```

#### Publication Figures
```powershell
python publication_summary.py --data ./runs/full_experiment --out ./manuscript/figures
```

## Output Interpretation

### Key Output Files

#### CSV Files
- `masking_curve_ci.csv`: Masking paradigm results with confidence intervals
- `blink_curve_ci.csv`: Attentional blink results
- `change_blind_curve_ci.csv`: Change blindness results
- `dualtask_ci.csv`: Dual-task interference results
- `infoflow_pci_sweep.csv`: Information flow measures across connectivity

#### Analysis Files
- `masking_trial_level.csv`: Trial-by-trial data for detailed analysis
- `masking_trial_level_regression.csv`: Regression coefficients and confidence intervals
- `noreport_calibration_masking.json`: Model calibration metrics

#### Visualization
- `full_infoflow_report.pdf`: Comprehensive results report
- Individual PNG files for each information flow measure

### Interpreting Results

#### Breakpoint Analysis
Look for sharp transitions in consciousness measures around β ≈ 0.35:
- **Below threshold (β < 0.3)**: Poor consciousness-related performance
- **At threshold (β ≈ 0.35)**: Sharp transition zone
- **Above threshold (β > 0.4)**: Robust consciousness-related performance

#### Information Flow Measures
- **Transfer Entropy**: Higher values indicate better information flow
- **Granger Causality**: Tests predictive relationships between regions
- **Participation Coefficient**: Measures how well nodes integrate across the network

#### Statistical Significance
- Confidence intervals that don't overlap suggest significant differences
- AIC model comparison favoring breakpoint over linear models indicates sharp thresholds

## Customization

### Modifying Experimental Parameters

#### Network Parameters
Edit the `GlobalWorkspace` class initialization in `overnight_full_run.py`:

```python
# Custom network size
gw = GlobalWorkspace(n_nodes=100)  # Default: 50

# Custom connectivity
gw = GlobalWorkspace(k_nearest=8)  # Default: 6
```

#### Task Parameters
Modify paradigm-specific settings:

```python
# Masking SOAs
MASKING_SOAS = [17, 33, 50, 67, 83, 100, 117, 133, 150]  # milliseconds

# Attentional blink lags
BLINK_LAGS = [100, 200, 300, 400, 500, 600, 700, 800]  # milliseconds
```

#### Statistical Parameters
Adjust analysis settings:

```python
# Bootstrap iterations
N_BOOTSTRAP = 1000  # Default, increase for more precision

# Confidence level
ALPHA = 0.05  # For 95% confidence intervals
```

### Adding New Paradigms

To add a custom consciousness paradigm:

1. **Create paradigm function**:
   ```python
   def run_custom_paradigm(gw: GlobalWorkspace, beta: float, n_trials: int) -> Dict[str, Any]:
       results = []
       for trial in range(n_trials):
           # Implement your paradigm logic
           result = custom_trial_logic(gw, beta)
           results.append(result)
       return aggregate_results(results)
   ```

2. **Add to main loop**:
   ```python
   # In overnight_full_run.py main function
   custom_results = run_custom_paradigm(gw, beta, n_trials)
   ```

3. **Include in output**:
   ```python
   # Save results
   custom_df = pd.DataFrame(custom_results)
   custom_df.to_csv(os.path.join(output_dir, "custom_paradigm.csv"))
   ```

### Custom Analysis Scripts

Create custom analysis following this template:

```python
#!/usr/bin/env python3
"""
Custom analysis script template
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gw_typing import NDArrayF, SeriesF

def custom_analysis(data_dir: str, output_dir: str) -> None:
    """Your custom analysis function."""
    # Load data
    df = pd.read_csv(f"{data_dir}/your_data.csv")
    
    # Perform analysis
    results = your_analysis_function(df)
    
    # Save results
    results.to_csv(f"{output_dir}/custom_results.csv")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results['x'], results['y'])
    plt.savefig(f"{output_dir}/custom_plot.png")

if __name__ == "__main__":
    custom_analysis("./runs/experiment", "./runs/custom_analysis")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'package_name'`

**Solution**:
```powershell
pip install package_name
# or
pip install -r requirements.txt
```

#### 2. Memory Issues
**Problem**: Script crashes with memory error

**Solutions**:
- Reduce `n_reps`: `--n_reps 50` instead of 100
- Reduce network size in code: `n_nodes=30`
- Close other applications
- Run on a machine with more RAM

#### 3. Long Runtime
**Problem**: Script takes too long

**Solutions**:
- Use fewer repetitions for testing: `--n_reps 10`
- Reduce beta sweep range: `--betas 0.1 0.3 0.5`
- Run overnight for full experiments
- Use faster computer if available

#### 4. File Permission Errors
**Problem**: Cannot write to output directory

**Solutions**:
```powershell
# Create directory first
mkdir ./runs/experiment

# Check permissions
# Run PowerShell as administrator if needed
```

#### 5. Inconsistent Results
**Problem**: Results vary between runs

**Solutions**:
- Use fixed random seed: `--seed 42`
- Increase repetitions: `--n_reps 200`
- Check for system-specific issues (different numpy versions)

### Performance Optimization

#### For Faster Development
```powershell
# Quick test with minimal repetitions
python overnight_full_run.py --out ./runs/dev_test --n_reps 3 --betas 0.1 0.3 0.5
```

#### For Production Runs
```powershell
# Full experiment with maximum precision
python overnight_full_run.py --out ./runs/production --n_reps 200 --seed 42
```

### Getting Help

1. **Check existing issues**: Review output logs and error messages
2. **Validate installation**: Run `python test_environment.py`
3. **Start simple**: Begin with quick test runs
4. **Documentation**: Refer to code comments and docstrings
5. **Community**: Contact the development team with specific error messages

### Advanced Usage

#### Parallel Processing
For faster computation on multi-core systems, you can run multiple beta values in parallel:

```powershell
# Terminal 1
python overnight_full_run.py --out ./runs/batch1 --betas 0.0 0.1 0.2 0.3

# Terminal 2  
python overnight_full_run.py --out ./runs/batch2 --betas 0.4 0.5 0.6 0.7

# Terminal 3
python overnight_full_run.py --out ./runs/batch3 --betas 0.8 0.9 1.0
```

Then combine results:
```powershell
python combine_results.py --inputs ./runs/batch1 ./runs/batch2 ./runs/batch3 --out ./runs/combined
```

#### Custom Network Topologies
To test different network architectures, modify the `make_small_world_W` function:

```python
def make_custom_network(n_nodes: int, custom_param: float) -> NDArrayF:
    """Create custom network topology."""
    # Implement your custom topology
    W = your_custom_topology_function(n_nodes, custom_param)
    return W
```

This comprehensive user guide should help researchers and practitioners effectively use the Global Workspace consciousness model for their own investigations.
