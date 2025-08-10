# Global Workspace Consciousness Model: A Developer's Exploration

[![CI](https://github.com/jovanSAPFIONEER/Orion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jovanSAPFIONEER/Orion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## About This Project

I'm a software developer at SAP Fioneer working on banking products. In my spare time, I've been exploring computational models of consciousness using techniques I'm familiar with from financial systems development.

This repository contains a Global Workspace model that simulates classic consciousness experiments. The goal is to see if network connectivity shows sharp threshold effects that might relate to awareness transitions.

## Key findings (current)

- Communication structure modulates masking thresholds much more than network size. A small‑world topology shows a clear optimum around rewiring p ≈ 0.40 with the lowest SOA threshold.
- Size scaling (32–512 nodes) changes thresholds modestly compared to the structural effect.
- Reproduction code and pinned artifacts are included.

<img src="figures/structure_vs_size_thresholds.png" alt="Structure vs size thresholds" />

Artifacts:
- Structure vs size summary (image): figures/structure_vs_size_thresholds.png
- Structure vs size summary (CSV): data/structure_vs_size_thresholds.csv
- Additional figures: figures/threshold_comparison.png and figures/scaling_analysis_comprehensive.png
- Narrative: executive_summary.md and SCALING_ANALYSIS.md
    - Measurement plan: documentation/MEASURING_CONSCIOUS_ACCESS.md

### Results at a glance

| Comparison | Headline | Evidence | Figure | Data |
|---|---|---|---|---|
| Structure sweep (small‑world p) | Minimum masking threshold at p ≈ 0.40 | Multiple seeds; ceiling points excluded; spline minimum stable | figures/structure_vs_size_thresholds.png | data/structure_vs_size_thresholds.csv |
| Size sweep (N = 32 → 512) | Thresholds change modestly vs. structure effect | Same pipeline; consistent ordering across sizes | figures/structure_vs_size_thresholds.png | data/structure_vs_size_thresholds.csv |
| Net takeaway | Communication structure > size for lowering SOA threshold | Robust across runs; reproducible with provided scripts | figures/threshold_comparison.png | — |

#### Headline numbers (95% CI)

- Example size contrast at SOA=1 (from data/masking_curves_all_sizes.csv):
    - 32 nodes: p=0.720 (18/25), 512 nodes: p=0.920 (23/25)
    - Δ accuracy = +0.200, 95% CI [-0.107, 0.454], Cohen’s h = 0.542
    - Note: CI crosses 0 due to small n; the structure effect is visualized in figures/threshold_comparison.png and figures/structure_vs_size_thresholds.png.

### What I Found
- Statistical evidence for threshold effects (p < 0.001) across multiple paradigms
- Sharp transitions around specific connectivity values
- Consistent patterns across different experimental tasks (masking, attentional blink, change blindness)
- Results that seem to align with some consciousness research predictions
- **NEW: Scale-invariant thresholds validated across 32-512 node networks (16-fold scaling)**

**Note**: I'm not a neuroscientist or consciousness researcher - this is exploratory work that I hope experts might find interesting enough to evaluate or critique.

## 🎯 Network Scaling Validation

Following community feedback about potential "small network artifacts," I've validated that consciousness thresholds persist robustly across network sizes:

### Scaling Results Summary

| Network Size | Effect Size | Peak Accuracy | Threshold Present | Runtime |
|--------------|-------------|---------------|-------------------|---------|
| 32 nodes     | 0.480       | 0.840         | ✅ Yes           | 3.2s    |
| 64 nodes     | 0.440       | 0.800         | ✅ Yes           | 4.0s    |
| 128 nodes    | 0.480       | 0.920         | ✅ Yes           | 8.4s    |
| 256 nodes    | 0.320       | 0.920         | ✅ Yes           | 35.2s   |
| 512 nodes    | 0.360       | 0.920         | ✅ Yes           | 229.0s  |

**Key Finding**: Effect sizes remain substantial (0.32-0.48) across all scales, demonstrating that threshold phenomena are scale-invariant properties of Global Workspace dynamics.

### Run Scaling Tests
```bash
# Comprehensive scaling analysis (all network sizes)
python comprehensive_scaling_test.py

# Quick validation (single size)
python test_node_scaling.py

# Generate scaling figures
python generate_scaling_figures.py
```

**Impact**: This addresses the "32-node limitation" critique and validates that consciousness thresholds are fundamental properties that persist at biologically relevant scales.

See [SCALING_ANALYSIS.md](SCALING_ANALYSIS.md) for complete methodology and results.

## Background: Banking Systems Perspective

Working in banking IT has given me experience with:
- **Threshold detection systems** (fraud alerts, risk monitoring)
- **Statistical validation** (regulatory compliance requirements)
- **Real-time processing** (transaction monitoring)
- **Robust testing** (mission-critical system standards)

I wondered if similar threshold detection approaches might apply to studying consciousness-like transitions in neural network models. This is essentially a "what if" experiment using tools I know from my day job.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/jovanSAPFIONEER/Orion.git
cd Orion
pip install -r requirements.txt
```

### Run Basic Experiment
```bash
# Quick validation (5 minutes)
python overnight_full_run.py --out ./test_run --n_reps 10
```

## Running tests

All tests use pytest. Below are Windows PowerShell examples.

Set up once per machine:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full suite:
```powershell
pytest -q
```

Run specific test files:
```powershell
pytest -q test_noise_ceiling.py
pytest -q test_noise_ceiling_scaling.py
pytest -q test_node_scaling.py
pytest -q test_enhanced_codebase.py
```

Filter by keyword (useful when iterating):
```powershell
pytest -q -k "noise_ceiling"
```

Run a single test function (node id):
```powershell
pytest -q test_node_scaling.py::test_scaling_behavior
```

Verbose output and short trace on failure:
```powershell
pytest -vv --maxfail=1 --tb=short
```

# Full experiment (2-3 hours)
python overnight_full_run.py --out ./full_experiment --n_reps 200
```

### Generate Figures
```bash
python scripts/make_figures.py --data_dir ./full_experiment --output_dir ./figures
```

### Reproduce main figure in one command (Windows PowerShell)
```powershell
# Generate a small run and build all figures from it
python overnight_full_run.py --out .\repro\main --n_reps 50 ; python scripts\make_figures.py --data_dir .\repro\main --output_dir .\figures
```

## 📈 What I've Tried to Do Right

### Statistical Approach
- Bootstrap confidence intervals for uncertainty quantification
- Breakpoint detection to identify sharp transitions
- Cross-validation across different parameter settings
- Effect size calculations to measure practical significance

### Code Quality
- Type hints throughout for clarity
- Comprehensive documentation
- Reproducible results with fixed random seeds
- Testing suite to catch errors

## 📁 Repository Structure

```
├── src/
│   ├── overnight_full_run.py      # Main experiment pipeline
│   ├── gw_typing.py               # Type definitions
│   ├── correlation_analysis.py    # Statistical analysis
│   └── compare_versions_effects_v2.py # Model comparison
├── documentation/
│   ├── user_guide/               # Usage instructions
│   └── api_reference/            # Complete API docs
├── manuscript/
│   ├── main_manuscript.md        # Complete research paper
│   ├── figures/                  # Publication figures
│   └── supplementary/            # Extended methods
├── scripts/
│   └── make_figures.py           # Publication figure generation
├── examples/
│   └── quick_start.py            # Basic usage examples
└── runs/
    └── [experiment_outputs]/      # Generated results
```

## 🔬 What This Might Contribute

### Potential Insights
1. **Threshold behavior**: Evidence that network connectivity shows sharp rather than gradual transitions
2. **Cross-paradigm consistency**: Similar patterns across different consciousness-related tasks
3. **Quantitative approach**: Statistical methods for measuring these transitions
4. **Implementation example**: Working code that others can build on or critique

### Limitations and Caveats
- This is a simplified model - real consciousness is vastly more complex
- I'm not claiming this proves anything about actual consciousness
- The model makes many assumptions that may not hold in biological systems
- Results need validation by experts in the field

## 📊 Sample Results

The model shows threshold-like behavior around specific connectivity values. For example:
- Visual masking accuracy jumps sharply rather than gradually
- Attentional blink shows clear temporal boundaries
- Change blindness detection has distinct phases

**Important**: These are computational results from a simplified model, not biological findings.

## 📧 Contact

I'm looking for feedback from researchers who know more about consciousness than I do. If you spot problems with the methodology, interpretation, or code, I'd appreciate hearing about it.

**Background**: Software developer at SAP Fioneer with interest in consciousness research
**Goal**: Learn whether this approach has merit or where it goes wrong
**Open to**: Collaboration, critique, suggestions for improvement

## 🎯 For Researchers

If you're willing to take a look, here's what might be useful:
1. **Quick test**: Run `python overnight_full_run.py --out ./test --n_reps 10` (5 minutes)
2. **Main results**: The threshold detection around connectivity values of 4-6
3. **Code review**: All methods are documented and type-hinted
4. **Statistical approach**: Bootstrap confidence intervals and breakpoint detection

### Full Materials Available
- **Complete manuscript**: See `manuscript/main_manuscript.md`
- **Detailed methods**: See `manuscript/supplementary/SUPPLEMENTARY_MATERIAL.md`
- **API documentation**: See `documentation/api_reference/API_REFERENCE.md`
- **Usage guide**: See `documentation/user_guide/README.md`

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This is a side project by a banking software developer curious about consciousness research. All feedback welcome.*
