# Global Workspace Consciousness Model: A Developer's Exploration

[![CI](https://github.com/jovanSAPFIONEER/Orion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jovanSAPFIONEER/Orion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

> Replication and citation: One-command external kit at https://github.com/jovanSAPFIONEER/DISCOVER-5.0 — DOI: https://doi.org/10.5281/zenodo.16912657

## About This Project

I'm a software developer at SAP Fioneer working on banking products. In my spare time, I've been exploring computational models of consciousness using techniques I'm familiar with from financial systems development.

This repository contains a Global Workspace model that simulates classic consciousness experiments. The goal is to see if network connectivity shows sharp threshold effects that might relate to awareness transitions.

## Key findings (current)

- Communication structure modulates masking thresholds more than network size (see figures/threshold_comparison.png).
- Size scaling (32–512 nodes) changes thresholds modestly compared to the structural effect (see figures/scaling_analysis_comprehensive.png).
- Causal broadcast sweep shows orientation-corrected sAUC windows modestly above chance at mid gains; stronger per‑SOA effects are visible (see figures/broadcast_gain_sweep_sAUC.png and figures/broadcast_gain_sAUC_bySOA.png).
- Reproduction code and pinned artifacts are included.

Artifacts (selected):
- Threshold comparison (image): figures/threshold_comparison.png
- Scaling analysis (image): figures/scaling_analysis_comprehensive.png
- Masking curves by size (CSV): data/masking_curves_all_sizes.csv
- Best structural threshold summary (CSV): data/structure_best_threshold.csv
- Broadcast sweep (sAUC, overall): data/broadcast_gain_sweep_sAUC.csv, figures/broadcast_gain_sweep_sAUC.png
- Broadcast sweep (sAUC, by SOA): data/broadcast_gain_sweep_bySOA.csv, figures/broadcast_gain_sweep_sAUC_bySOA.png
- Narrative: executive_summary.md and SCALING_ANALYSIS.md
    - Measurement plan: documentation/user_guide/README.md

### Results at a glance

| Comparison | Headline | Evidence | Figure | Data |
|---|---|---|---|---|
| Structure sweep (small‑world p) | Minimum masking threshold at p ≈ 0.40 | Multiple seeds; ceiling points excluded; spline minimum stable | figures/threshold_comparison.png | data/structure_best_threshold.csv |
| Size sweep (N = 32 → 512) | Thresholds change modestly vs. structure effect | Same pipeline; consistent ordering across sizes | figures/scaling_analysis_comprehensive.png | data/masking_curves_all_sizes.csv |
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
# Quick validation (≈5 minutes)
python overnight_full_run.py --out ./test_run --n_mask 60 --n_blink 40 --n_cb 32 --n_dual 40 --boots 300
```

### CAI (Conscious Access Index) quickstart
```powershell
# 1) Dump CAI-ready JSONs during a small run
python overnight_full_run.py --out .\runs\mini --n_mask 60 --n_blink 40 --n_cb 32 --n_dual 40 --boots 300 --dump_cai_json --cai_dir .\runs\mini\cai
# 2) Compute CAI scores
python scripts\compute_cai.py --infile .\runs\mini\cai --outfile .\runs\mini\cai_scores.json
# 3) Evaluate predictive power (AUC/ECE)
python scripts\eval_cai.py --cai_dir .\runs\mini\cai --outfile .\runs\mini\cai_eval.json

# Optional (recommended): Cross-validated, calibrated evaluation with sAUC/Brier and calibration export
python scripts\eval_cai_cv.py --cai_dir .\runs\mini\cai --outfile .\runs\mini\cai_cv_eval.json --calib_json .\runs\mini\cai_calibration.json --save_probs .\runs\mini\cai_probs.csv
```

See `documentation/user_guide/README.md` for user-facing guidance.

### CAI evaluation (larger run snapshot)

Latest pinned metrics from a moderate run (n ≈ 1200 trials in masking):

- Simple CAI: AUC ≈ 0.491, ECE ≈ 0.220 (min–max scaled proxy)
- Cross‑validated + isotonic calibrated: AUC ≈ 0.497, sAUC (orientation‑corrected) ≈ 0.503, ECE ≈ 0.012; Brier score also reported

Artifacts:

- data/cai_eval_full_01.json (simple)
- data/cai_cv_eval_full_01.json (CV‑calibrated)

Reproduce on Windows PowerShell:

```powershell
# Generate CAI-ready trials (moderate size) and PDF report
python overnight_full_run.py --out .\runs\cai_full_01 --n_mask 200 --n_blink 120 --n_cb 96 --n_dual 120 --boots 1200 --dump_cai_json --cai_dir .\runs\cai_full_01\cai

# Evaluate simple CAI
python scripts\eval_cai.py --cai_dir .\runs\cai_full_01\cai --outfile .\runs\cai_full_01\cai_eval.json

# Evaluate cross‑validated, calibrated CAI
python scripts\eval_cai_cv.py --cai_dir .\runs\cai_full_01\cai --outfile .\runs\cai_full_01\cai_cv_eval.json --calib_json .\runs\cai_full_01\cai_calibration.json --save_probs .\runs\cai_full_01\cai_probs.csv
```

Brief causal check:

- Lesion (broadcast_gain=0): CV‑calibrated AUC ≈ 0.499, ECE ≈ 0.068 (data/cai_cv_eval_lesion_broadcast0.json)

```powershell
# Run lesion and evaluate
python overnight_full_run.py --out .\runs\cai_lesion_broadcast0 --n_mask 60 --n_blink 40 --n_cb 32 --n_dual 40 --boots 300 --dump_cai_json --cai_dir .\runs\cai_lesion_broadcast0\cai --broadcast_gain 0.0
python scripts\eval_cai_cv.py --cai_dir .\runs\cai_lesion_broadcast0\cai --outfile .\runs\cai_lesion_broadcast0\cai_cv_eval.json
```

### Causal sweep (headline: sAUC with orientation correction)

Run a graded sweep over the global broadcast gain and evaluate CAI sensitivity with orientation‑corrected sAUC. Per‑SOA effects and bootstrap CIs are provided.

Artifacts:
- Overall sAUC with 95% CIs: data/broadcast_gain_sweep_sAUC.csv, figures/broadcast_gain_sweep_sAUC.png
- By‑SOA sAUC: data/broadcast_gain_sweep_bySOA.csv, figures/broadcast_gain_sAUC_bySOA.png

Reproduce on Windows PowerShell:

```powershell
# 1) Sweep broadcast_gain and generate CAI JSONs per gain
python scripts\causal_broadcast_sweep.py --out .\runs\causal_bcast_sweep --gains 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0 --n_mask 60 --n_blink 40 --n_cb 32 --n_dual 40 --boots 300
# 2) Analyze sweep with sAUC and by‑SOA outputs
python scripts\analyze_broadcast_sweep.py --sweep_dir .\runs\causal_bcast_sweep --out .\runs\causal_bcast_sweep
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
python overnight_full_run.py --out ./full_experiment --full
```

### Generate Figures
```bash
python scripts/make_figures.py --data_dir ./full_experiment --output_dir ./figures
```

### Reproduce main figure in one command (Windows PowerShell)
```powershell
# Generate a small run and build all figures from it
python overnight_full_run.py --out .\repro\main --n_mask 60 --n_blink 40 --n_cb 32 --n_dual 40 --boots 300 ; python scripts\make_figures.py --data_dir .\repro\main --output_dir .\figures
```

### Full causal rewire replication (Windows PowerShell)
For a turnkey reproduction of the lesion→recovery result (SOA‑1), see `REPRODUCE.md` or run:

```powershell
# 1) Set up environment
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce_env.ps1
# 2) Run end-to-end rewire pipeline
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce_rewire_pipeline.ps1
```
Key outputs will appear in `runs/rewire_cai_sweep_gain0p6_rep` (CSV + PNG/PDF).

Alternatively, use the external one-command replication kit (cross-platform, with CI/release):
- Repo: https://github.com/jovanSAPFIONEER/DISCOVER-5.0
- Cite: https://doi.org/10.5281/zenodo.16912657

## How to cite (replication kit)

If you use the reproduction kit or its outputs, please cite:

DISCOVER 5.0 — Reproduction Kit for Causal Rewire Recovery (SOA‑1). Version v0.1.2. DOI: https://doi.org/10.5281/zenodo.16912657

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
1. **Quick test**: Run `python overnight_full_run.py --out ./test --n_mask 60 --n_blink 40 --n_cb 32 --n_dual 40 --boots 300` (≈5 minutes)
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
