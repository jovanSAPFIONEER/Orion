# Global Workspace Model - API Reference

## Table of Contents
1. [Core Classes](#core-classes)
2. [Main Functions](#main-functions)
3. [Utility Functions](#utility-functions)
4. [Type Definitions](#type-definitions)
5. [Constants](#constants)

## Core Classes

### GlobalWorkspace

**Location**: `overnight_full_run.py`

The main neural network model implementing Global Workspace Theory dynamics.

```python
class GlobalWorkspace:
    def __init__(self, n_nodes: int = 50, k_nearest: int = 6, tau: float = 10.0, 
                 noise_std: float = 0.1, dt: float = 1.0):
```

#### Parameters
- `n_nodes` (int): Number of nodes in the network (default: 50)
- `k_nearest` (int): Number of nearest neighbors in initial ring lattice (default: 6)
- `tau` (float): Time constant for neural dynamics (default: 10.0)
- `noise_std` (float): Standard deviation of neural noise (default: 0.1)
- `dt` (float): Integration time step in milliseconds (default: 1.0)

#### Attributes
- `n_nodes` (int): Number of network nodes
- `W` (NDArrayF): Connectivity matrix (n_nodes × n_nodes)
- `h` (NDArrayF): Current node activations
- `history` (List[NDArrayF]): Activation history over time

#### Methods

##### `set_connectivity(beta: float) -> None`
Sets the network connectivity using small-world topology.

**Parameters:**
- `beta` (float): Small-world rewiring probability [0, 1]

**Example:**
```python
gw = GlobalWorkspace()
gw.set_connectivity(beta=0.35)  # Set to optimal small-world connectivity
```

##### `reset() -> None`
Resets network to initial state with zero activations.

##### `step(inputs: NDArrayF) -> NDArrayF`
Performs one integration step of network dynamics.

**Parameters:**
- `inputs` (NDArrayF): External input to each node

**Returns:**
- NDArrayF: Current node activations after integration step

##### `detect_ignition(threshold: float = 0.6, min_duration: int = 50) -> bool`
Detects global ignition based on network-wide activity.

**Parameters:**
- `threshold` (float): Minimum participation fraction for ignition (default: 0.6)
- `min_duration` (int): Minimum duration in milliseconds (default: 50)

**Returns:**
- bool: True if global ignition detected

**Example:**
```python
gw = GlobalWorkspace()
gw.set_connectivity(0.35)
inputs = np.random.randn(50) * 0.5
gw.step(inputs)
is_ignited = gw.detect_ignition()
```

##### `get_participation() -> float`
Calculates participation coefficient for the current state.

**Returns:**
- float: Network participation coefficient [0, 1]

##### `get_ignition_latency() -> Optional[int]`
Gets the latency to global ignition in current trial.

**Returns:**
- Optional[int]: Latency in milliseconds, or None if no ignition

## Main Functions

### `run_trial_fast(gw: GlobalWorkspace, target_onset: int, target_strength: float, mask_onset: Optional[int] = None, mask_strength: float = 0.0, trial_duration: int = 500) -> TrialResult`

**Location**: `overnight_full_run.py`

Runs a single trial of visual processing with optional masking.

#### Parameters
- `gw` (GlobalWorkspace): The network model
- `target_onset` (int): Target stimulus onset time (ms)
- `target_strength` (float): Target stimulus strength
- `mask_onset` (Optional[int]): Mask onset time (ms), None for no mask
- `mask_strength` (float): Mask stimulus strength (default: 0.0)
- `trial_duration` (int): Total trial duration (ms) (default: 500)

#### Returns
- `TrialResult`: TypedDict containing trial outcomes

#### Example
```python
gw = GlobalWorkspace()
gw.set_connectivity(0.35)
result = run_trial_fast(gw, target_onset=100, target_strength=1.0, 
                       mask_onset=150, mask_strength=1.5)
print(f"Report accuracy: {result['report_acc']}")
```

### `run_masking_curve(gw: GlobalWorkspace, beta: float, soas: List[int], n_trials: int = 100) -> MaskingResult`

**Location**: `overnight_full_run.py`

Runs complete visual masking experiment across multiple SOAs.

#### Parameters
- `gw` (GlobalWorkspace): The network model
- `beta` (float): Small-world connectivity parameter
- `soas` (List[int]): List of stimulus-onset asynchronies (ms)
- `n_trials` (int): Number of trials per SOA (default: 100)

#### Returns
- `MaskingResult`: TypedDict containing aggregated results

### `run_attentional_blink(gw: GlobalWorkspace, beta: float, lags: List[int], n_trials: int = 100) -> BlinkResult`

**Location**: `overnight_full_run.py`

Runs attentional blink experiment with dual-target RSVP.

#### Parameters
- `gw` (GlobalWorkspace): The network model
- `beta` (float): Small-world connectivity parameter  
- `lags` (List[int]): List of T1-T2 lags (ms)
- `n_trials` (int): Number of trials per lag (default: 100)

#### Returns
- `BlinkResult`: TypedDict containing T1 and T2 accuracy measures

### `run_change_blindness(gw: GlobalWorkspace, beta: float, periods: List[int], n_trials: int = 100) -> ChangeBlindResult`

**Location**: `overnight_full_run.py`

Runs change blindness experiment with varying change periods.

#### Parameters
- `gw` (GlobalWorkspace): The network model
- `beta` (float): Small-world connectivity parameter
- `periods` (List[int]): List of change periods (ms)
- `n_trials` (int): Number of trials per period (default: 100)

#### Returns
- `ChangeBlindResult`: TypedDict containing detection metrics

### `run_dual_task(gw: GlobalWorkspace, beta: float, n_trials: int = 100) -> DualTaskResult`

**Location**: `overnight_full_run.py`

Runs dual-task interference experiment.

#### Parameters
- `gw` (GlobalWorkspace): The network model
- `beta` (float): Small-world connectivity parameter
- `n_trials` (int): Number of trials per condition (default: 100)

#### Returns
- `DualTaskResult`: TypedDict containing single and dual-task performance

## Utility Functions

### Network Generation

#### `make_small_world_W(n_nodes: int, k_nearest: int, beta: float) -> NDArrayF`

**Location**: `overnight_full_run.py`

Generates small-world connectivity matrix using Watts-Strogatz algorithm.

**Parameters:**
- `n_nodes` (int): Number of nodes
- `k_nearest` (int): Number of nearest neighbors in ring lattice
- `beta` (float): Rewiring probability [0, 1]

**Returns:**
- NDArrayF: Connectivity matrix (n_nodes × n_nodes)

**Example:**
```python
W = make_small_world_W(n_nodes=50, k_nearest=6, beta=0.35)
print(f"Network shape: {W.shape}")
print(f"Connection density: {np.mean(W):.3f}")
```

### Statistical Functions

#### `bootstrap_ci(data: NDArrayF, alpha: float = 0.05, n_boot: int = 1000) -> CIBounds`

**Location**: `overnight_full_run.py`

Computes bootstrap confidence intervals for data.

**Parameters:**
- `data` (NDArrayF): Input data array
- `alpha` (float): Significance level (default: 0.05 for 95% CI)
- `n_boot` (int): Number of bootstrap samples (default: 1000)

**Returns:**
- `CIBounds`: TypedDict with 'lo' and 'hi' confidence bounds

#### `compute_effect_size(group1: NDArrayF, group2: NDArrayF) -> float`

**Location**: `overnight_full_run.py`

Computes Cohen's d effect size between two groups.

**Parameters:**
- `group1` (NDArrayF): First group data
- `group2` (NDArrayF): Second group data

**Returns:**
- float: Cohen's d effect size

### Information Flow Functions

#### `compute_transfer_entropy(source: NDArrayF, target: NDArrayF, lag: int = 1) -> float`

**Location**: `overnight_full_run.py`

Computes Transfer Entropy from source to target time series.

**Parameters:**
- `source` (NDArrayF): Source time series
- `target` (NDArrayF): Target time series  
- `lag` (int): Time lag for analysis (default: 1)

**Returns:**
- float: Transfer entropy in bits

#### `compute_granger_causality(data: NDArrayF, max_lag: int = 5) -> NDArrayF`

**Location**: `overnight_full_run.py`

Computes Granger causality matrix for multivariate time series.

**Parameters:**
- `data` (NDArrayF): Time series data (time × nodes)
- `max_lag` (int): Maximum lag for autoregression (default: 5)

**Returns:**
- NDArrayF: Granger causality matrix (nodes × nodes)

#### `compute_participation_coefficient(W: NDArrayF, community_assignment: NDArrayF) -> NDArrayF`

**Location**: `overnight_full_run.py`

Computes participation coefficient for each node.

**Parameters:**
- `W` (NDArrayF): Connectivity matrix
- `community_assignment` (NDArrayF): Community membership for each node

**Returns:**
- NDArrayF: Participation coefficient for each node

### Conversion Utilities

#### `to_array(x: Union[NDArrayF, SeriesF, List, float]) -> NDArrayF`

**Location**: `gw_typing.py`

Converts various input types to numpy array.

**Parameters:**
- `x`: Input data of various types

**Returns:**
- NDArrayF: Numpy array representation

#### `to_series(x: Union[NDArrayF, SeriesF, List]) -> SeriesF`

**Location**: `gw_typing.py`

Converts various input types to pandas Series.

**Parameters:**
- `x`: Input data of various types

**Returns:**
- SeriesF: Pandas Series representation

## Type Definitions

### Core Types

**Location**: `gw_typing.py`

```python
# NumPy array type with float64 elements
NDArrayF = np.ndarray[Any, np.dtype[np.floating[Any]]]

# Pandas Series with float64 elements  
SeriesF = pd.Series[float]

# Pandas DataFrame with float64 elements
DataFrameF = pd.DataFrame
```

### Result Types

#### `TrialResult`
```python
class TrialResult(TypedDict):
    report_acc: float          # Report accuracy (0.0 or 1.0)
    ignition_latency: float    # Latency to ignition (ms)
    ignition_occurred: bool    # Whether ignition occurred
    max_activation: float      # Peak network activation
    participation: float       # Participation coefficient
```

#### `CIBounds`
```python
class CIBounds(TypedDict):
    lo: float                  # Lower confidence bound
    hi: float                  # Upper confidence bound
```

#### `MaskingResult`
```python
class MaskingResult(TypedDict):
    soas: List[int]           # Stimulus onset asynchronies
    report_acc: List[float]   # Report accuracy per SOA
    ignition_lat: List[float] # Ignition latency per SOA
    ci_bounds: List[CIBounds] # Confidence intervals
```

#### `BlinkResult`
```python
class BlinkResult(TypedDict):
    lags: List[int]           # T1-T2 lags (ms)
    t1_acc: List[float]       # T1 accuracy per lag
    t2_acc: List[float]       # T2 accuracy per lag  
    t2_given_t1: List[float]  # T2|T1 conditional accuracy
```

#### `BreakpointResult`
```python
class BreakpointResult(TypedDict):
    breakpoint: float         # Optimal breakpoint location
    aic_linear: float         # AIC for linear model
    aic_breakpoint: float     # AIC for breakpoint model
    evidence_ratio: float     # Evidence ratio favoring breakpoint
```

### Parameter Types

#### `ModelParams`
```python
class ModelParams(TypedDict):
    n_nodes: int              # Number of network nodes
    k_nearest: int            # Nearest neighbors in lattice
    tau: float                # Neural time constant
    noise_std: float          # Neural noise standard deviation
    dt: float                 # Integration time step
```

#### `ExperimentParams`
```python
class ExperimentParams(TypedDict):
    n_trials: int             # Trials per condition
    n_bootstrap: int          # Bootstrap iterations
    alpha: float              # Significance level
    seed: Optional[int]       # Random seed
```

## Constants

### Default Parameters

**Location**: `overnight_full_run.py`

```python
# Network parameters
DEFAULT_N_NODES = 50
DEFAULT_K_NEAREST = 6
DEFAULT_TAU = 10.0
DEFAULT_NOISE_STD = 0.1
DEFAULT_DT = 1.0

# Experimental parameters
DEFAULT_N_TRIALS = 100
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_ALPHA = 0.05

# Paradigm-specific parameters
MASKING_SOAS = [17, 33, 50, 67, 83, 100, 117, 133, 150]  # ms
BLINK_LAGS = [100, 200, 300, 400, 500, 600, 700, 800]   # ms
CHANGE_PERIODS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]  # ms

# Small-world beta sweep
DEFAULT_BETAS = np.linspace(0.0, 1.0, 21)

# Ignition detection parameters
IGNITION_THRESHOLD = 0.6     # Participation threshold
MIN_IGNITION_DURATION = 50   # Minimum duration (ms)

# Visualization parameters
DPI = 300                    # Figure resolution
FIGSIZE = (10, 8)           # Default figure size
```

### File Naming Conventions

```python
# Output file patterns
MASKING_CURVE_FILE = "masking_curve_ci.csv"
BLINK_CURVE_FILE = "blink_curve_ci.csv"
CHANGE_BLIND_FILE = "change_blind_curve_ci.csv"
DUAL_TASK_FILE = "dualtask_ci.csv"
TRIAL_LEVEL_FILE = "masking_trial_level.csv"
REGRESSION_FILE = "masking_trial_level_regression.csv"
INFOFLOW_FILE = "infoflow_pci_sweep.csv"
REPORT_FILE = "full_infoflow_report.pdf"
```

## Usage Examples

### Basic Model Setup
```python
from overnight_full_run import GlobalWorkspace, run_trial_fast
import numpy as np

# Create model
gw = GlobalWorkspace(n_nodes=50, tau=10.0, noise_std=0.1)

# Set connectivity  
gw.set_connectivity(beta=0.35)

# Run single trial
result = run_trial_fast(gw, target_onset=100, target_strength=1.0)
print(f"Trial result: {result}")
```

### Complete Experiment
```python
from overnight_full_run import run_masking_curve, MASKING_SOAS
import pandas as pd

# Setup
gw = GlobalWorkspace()
beta = 0.35

# Run masking experiment
results = run_masking_curve(gw, beta, MASKING_SOAS, n_trials=100)

# Convert to DataFrame
df = pd.DataFrame({
    'SOA': results['soas'],
    'ReportAcc': results['report_acc'],
    'IgnLat': results['ignition_lat']
})

print(df.head())
```

### Information Flow Analysis
```python
from overnight_full_run import compute_transfer_entropy
import numpy as np

# Generate sample time series
n_time, n_nodes = 1000, 50
data = np.random.randn(n_time, n_nodes)

# Compute transfer entropy between first two nodes
te = compute_transfer_entropy(data[:, 0], data[:, 1])
print(f"Transfer entropy: {te:.4f} bits")
```

### Statistical Analysis
```python
from overnight_full_run import bootstrap_ci, compute_effect_size
import numpy as np

# Sample data
group1 = np.random.normal(0.6, 0.1, 100)
group2 = np.random.normal(0.8, 0.1, 100)

# Compute confidence intervals
ci1 = bootstrap_ci(group1)
ci2 = bootstrap_ci(group2)

# Compute effect size
effect_size = compute_effect_size(group1, group2)

print(f"Group 1 CI: [{ci1['lo']:.3f}, {ci1['hi']:.3f}]")
print(f"Group 2 CI: [{ci2['lo']:.3f}, {ci2['hi']:.3f}]")
print(f"Effect size: {effect_size:.3f}")
```

This API reference provides comprehensive documentation for all major components of the Global Workspace consciousness model, enabling researchers to effectively use and extend the codebase for their own investigations.
