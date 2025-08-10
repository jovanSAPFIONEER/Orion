"""Shared type aliases for the GW-Awareness project."""
from typing import TypedDict, Sequence, Tuple, Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

# Convenient aliases for common array/dataframe types
NDArrayF = np.ndarray          # convenient alias for float arrays
SeriesF = pd.Series           # pandas Series (can hold floats)
DataFrame = pd.DataFrame      # pandas DataFrame
ArrayLike = Union[NDArrayF, SeriesF, List[float], Tuple[float, ...]]  # flexible array-like type

class CIBounds(TypedDict):
    """Low / high confidence-interval bounds."""
    lo: float
    hi: float

class BreakpointResult(TypedDict):
    """Results from breakpoint analysis."""
    tau_mean: float
    tau_lo: float
    tau_hi: float
    dAIC_mean: float
    dAIC_lo: float
    dAIC_hi: float

class TrialData(TypedDict):
    """Single trial data structure."""
    SOA: int
    trial: int
    ReportHit: int
    ConfMax: float
    IgnLatency: Optional[float]
    Granger_into_GW: Optional[float]
    TE_into_GW: Optional[float]

class ModelParams(TypedDict):
    """Model configuration parameters."""
    N: int
    g_scale: float
    b_gain: float
    degree_frac: float
    rewire_p: float
    seed: int

class ExperimentalResults(TypedDict):
    """Results from a single experimental paradigm."""
    SOA: List[int]
    ReportAcc: List[float]
    ReportAcc_lo: List[float] 
    ReportAcc_hi: List[float]
    ConfMean: List[float]
    ConfMean_lo: List[float]
    ConfMean_hi: List[float]

class InfoFlowMetrics(TypedDict):
    """Information flow and consciousness metrics."""
    g_scale: float
    ReportAcc: float
    MeanIgnition: float
    Participation: float
    TE_bits: float
    Granger_logVR: float
    PCI_like: float

# Utility functions for type conversion
def to_array(data: ArrayLike) -> NDArrayF:
    """Convert array-like data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values)
    return np.asarray(data)

def to_series(data: ArrayLike, name: Optional[str] = None) -> SeriesF:
    """Convert array-like data to pandas Series."""
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, name=name)
