# Measuring Conscious Access: A Rigorous Validation Plan

Goal
- Establish a reproducible, falsifiable marker of conscious access in this model.
- Demonstrate convergent, discriminant, predictive, and causal validity across tasks and perturbations.

Scope and definitions
- Target: conscious access (reportable awareness), not "consciousness" writ large.
- Marker: a scalar index that predicts reportable access on a per-trial basis and generalizes across tasks and settings.

Validation contract
- Inputs: trial-wise model activity, per-module activations, optional perturbations, task labels, (optional) action outputs.
- Outputs: scalar Conscious Access Index (CAI) per trial.
- Success criteria (all must hold):
  1) Discrimination: CAI predicts report vs no-report with ROC-AUC ≥ 0.85 on held-out seeds, tasks, and sizes.
  2) Calibration: Expected Calibration Error (ECE) ≤ 0.05; Brier score improves over baselines.
  3) Generalization: Pre-registered thresholds hold on out-of-distribution (OOD) masking variants and sizes; no drop > 0.05 AUC.
  4) Causal sensitivity: Lesions/perturbations that reduce global broadcasting reduce CAI and report rates in tandem (monotonic trend, p < 0.01 after multiple-comparison correction).
  5) Specificity: CAI changes are not explained by overall performance or SNR alone (matched-accuracy controls remain separated by CAI; partial correlations significant, |r| ≥ 0.3, p < 0.01).

Marker candidates (to be combined via logistic/meta-learner if needed)
- Broadcasting index (BI): fraction of modules participating above threshold over a sustained window; includes duration and breadth (e.g., 95th-percentile span of global ignition).
- Participation coefficient / k-core ignition: per-trial mean participation coefficient and largest active k-core size.
- Perturbational complexity index (PCI-like): transiently perturb a subset (e.g., 5% nodes), compute Lempel–Ziv complexity (or entropy rate) of the evoked spatiotemporal response vs baseline; normalize per network.
- Effective connectivity gain: change in multivariate transfer entropy/Granger between modules during reports vs misses.
- Metacognitive signal: confidence head outputs; compute meta-d′ and type-2 ROC; integrate with BI/PCI.

Benchmark suite (convergent validity)
- Masking variants: backward (current), forward, and noise masking.
- Temporal limits: attentional blink, change blindness.
- Vigilance/task load: dual-task interference; sustained attention with distractors.
- Size/topology sweeps: N ∈ {32, 64, 128, 256, 512}, small-world p ∈ [0, 1].
- Closed loop: simple RL/control tasks where action depends on access under masking.
- Two-network coupling: information sharing across coupled agents; test whether CAI predicts inter-agent transfer under masking.

Controls (discriminant validity)
- Hold size constant; vary degree, path length, clustering; show CAI tracks broadcasting, not trivial graph confounds.
- Matched-performance controls: downsample or adjust gain so accuracy is equal across conditions; CAI should still differ for access vs no-access.
- Energy/noise confounds: vary energy budget and input SNR; CAI should not reduce to energy/SNR alone (control regressors; residual CAI remains predictive).
- Random topology and lattice baselines.

Causal tests
- Targeted lesions: remove hub edges or inter-module connections; expect monotonic drop in CAI and report rate.
- Stimulation: brief excitation to workspace hubs increases CAI and report probability; sham has no effect.
- Sedation/anesthesia proxy: reduce global gain; CAI and PCI both decline more than task accuracy under matched-accuracy controls.

Analysis plan (pre-registered)
- Train marker on a subset of seeds/tasks; lock hyperparameters.
- Evaluate on held-out seeds, sizes, and new masking variants.
- Primary endpoints: AUC, ECE; secondary: meta-d′, PCI difference, participation coefficient.
- Multiple-comparison correction: Holm–Bonferroni across tasks and endpoints.
- Report bootstrap 95% CIs (10k resamples) and cluster-robust SE across seeds.

Acceptance thresholds and "nail in the coffin"
- All five success criteria met with pre-registered endpoints.
- Replication by an external run (independent seed/compute) within ±0.03 AUC and ±0.02 ECE.
- Robustness across at least two additional masking variants and one closed-loop task.

Minimal implementation roadmap
- Add report and confidence heads; log trial-wise reports and confidence.
- Implement BI, participation coefficient, k-core ignition, PCI-like perturbation tool.
- Create analysis scripts that output CAI per trial and compute AUC/ECE across benchmarks.
- Add matched-accuracy pipelines and confound regressions.
- Add preregistration doc and a blinded evaluation script writing signed artifacts only.

Quick usage (prototype CAI)
- Save per-trial JSON with key "module_activations" (and optional "baseline_activations") shaped [T, M].
- Compute CAI on a file or a folder of JSONs:

```powershell
python scripts\compute_cai.py --infile .\path\to\trial.json
# or a directory of JSONs
python scripts\compute_cai.py --infile .\path\to\trials_dir --outfile .\runs\cai_summary.json
```

Blinding & preregistration
- Freeze analysis and endpoints before final runs; store SHA of code.
- Use blinded file names; unblind after metrics are finalized.

Caveats
- CAI validates conscious access in this model; it is not a claim of human/animal consciousness.
- Avoid over-interpretation of PCI/complexity without careful normalization.
