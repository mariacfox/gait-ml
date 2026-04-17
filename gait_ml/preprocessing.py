"""
preprocessing.py — Signal filtering, gap-filling, and time normalization.

All parameters confirmed from MATLAB scripts:
  filterMarkerData.m  → kinematics: 4th-order LP Butterworth, fc=8 Hz, filtfilt
  filtForceCOP.m      → forces: fc=8 Hz; COP: fc=15 Hz; moments: not filtered
  gaitCycleNormalization.m → 101-point pchip interpolation

All functions are pure (no side effects, no global state).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt

from gait_ml.config import DEFAULT_LAB_CONFIG as _CFG


def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = _CFG.filter.order,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Equivalent to MATLAB ``butter`` + ``filtfilt``.

    Parameters
    ----------
    data : np.ndarray
        Input signal, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
    cutoff_hz : float
        Cutoff frequency in Hz.
    sample_rate_hz : float
        Sampling rate in Hz.
    order : int
        Filter order. Default 4 matches MATLAB pipeline.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input.
    """
    nyq = 0.5 * sample_rate_hz
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def detect_missing_markers(data: np.ndarray) -> np.ndarray:
    """Return a boolean mask of frames where any marker is missing.

    A frame is considered missing if all marker columns are zero (Qualisys
    occlusion convention before NaN replacement) or any column is NaN.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape ``(n_frames, n_channels)``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_frames,)``. True where the frame has
        all-zero columns or at least one NaN column.
    """
    all_zero = (data == 0.0).all(axis=1)
    any_nan = np.isnan(data).any(axis=1)
    return all_zero | any_nan


def fill_marker_gaps(data: np.ndarray) -> np.ndarray:
    """Fill missing marker data using pchip interpolation.

    Replaces NaN values column-wise with pchip-interpolated values,
    matching MATLAB ``fillmissing(data, 'pchip')`` in ``filterMarkerData.m``.
    Frames at the start or end of the trial with no surrounding valid data
    are left as NaN.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape ``(n_frames, n_channels)``. NaN entries are
        treated as missing (call after replacing Qualisys zeros with NaN).

    Returns
    -------
    np.ndarray
        Gap-filled array, same shape as input.
    """
    out = data.copy()
    x_all = np.arange(len(data))
    for col in range(data.shape[1]):
        y = data[:, col]
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue  # not enough points to interpolate
        interp = PchipInterpolator(x_all[valid], y[valid], extrapolate=False)
        missing = np.isnan(y)
        out[missing, col] = interp(x_all[missing])
    return out


def preprocess_markers(
    df: pd.DataFrame,
    sample_rate_hz: float = _CFG.acquisition.kinematic_sample_rate_hz,
    cutoff_hz: float = _CFG.filter.kinematic_lowpass_hz,
) -> pd.DataFrame:
    """Gap-fill and filter all marker columns in a kinematics DataFrame.

    Implements the full MATLAB ``filterMarkerData.m`` pipeline:
    zeros → NaN (already done in ``load_marker_tsv``), pchip gap-fill,
    then 4th-order zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    df : pd.DataFrame
        Marker DataFrame from ``load_marker_tsv``. Must have Frame and Time
        as the first two columns; all remaining columns are marker axes.
    sample_rate_hz : float
        Kinematic sampling rate in Hz. Default 160.
    cutoff_hz : float
        Low-pass cutoff frequency in Hz. Default 8 (confirmed from MATLAB).

    Returns
    -------
    pd.DataFrame
        DataFrame with the same columns; marker columns are gap-filled and
        filtered. Frame and Time columns are unchanged.
    """
    result = df.copy()
    marker_cols = list(df.columns[2:])
    arr = df[marker_cols].to_numpy(dtype=float)

    arr = fill_marker_gaps(arr)
    arr = butterworth_lowpass(arr, cutoff_hz, sample_rate_hz)

    result[marker_cols] = arr
    return result


def preprocess_forces(
    df: pd.DataFrame,
    sample_rate_hz: float = _CFG.acquisition.grf_sample_rate_hz,
    force_cutoff_hz: float = _CFG.filter.grf_lowpass_hz,
    cop_cutoff_hz: float = _CFG.filter.cop_lowpass_hz,
) -> pd.DataFrame:
    """Filter force and COP columns in a force plate DataFrame.

    Implements ``filtForceCOP.m``:

    - Force_X/Y/Z: 4th-order LP Butterworth at ``force_cutoff_hz`` (8 Hz)
    - COP_X/Y: 4th-order LP Butterworth at ``cop_cutoff_hz`` (15 Hz)
    - Moment_X/Y/Z: **not filtered** (passed through as-is)

    Parameters
    ----------
    df : pd.DataFrame
        Force DataFrame from ``load_force_tsv``.
    sample_rate_hz : float
        Force plate sampling rate in Hz. Default 1120.
    force_cutoff_hz : float
        Low-pass cutoff for Force_X/Y/Z in Hz. Default 8.
    cop_cutoff_hz : float
        Low-pass cutoff for COP_X/Y in Hz. Default 15.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same columns; forces and COP filtered,
        moments unchanged.
    """
    result = df.copy()

    for col in _CFG.qualisys.force_columns:
        if col in result.columns:
            result[col] = butterworth_lowpass(
                result[col].to_numpy(dtype=float), force_cutoff_hz, sample_rate_hz
            )

    for col in _CFG.qualisys.cop_columns:
        if col in result.columns:
            result[col] = butterworth_lowpass(
                result[col].to_numpy(dtype=float), cop_cutoff_hz, sample_rate_hz
            )

    return result


def normalize_gait_cycle(
    data: np.ndarray,
    n_points: int = _CFG.normalization.n_points,
) -> np.ndarray:
    """Time-normalize a single gait cycle to ``n_points`` using pchip.

    Matches MATLAB ``gaitCycleNormalization.m``:
    ``interp1(x, v, 0:1:100, 'pchip')``.

    Parameters
    ----------
    data : np.ndarray
        Single gait cycle, shape ``(n_frames,)`` or
        ``(n_frames, n_channels)``.
    n_points : int
        Number of output time points. Default 101 (0–100% inclusive).

    Returns
    -------
    np.ndarray
        Time-normalized data, shape ``(n_points,)`` or
        ``(n_points, n_channels)``.
    """
    x_old = np.linspace(0, 100, len(data))
    x_new = np.arange(n_points, dtype=float)  # 0, 1, ..., 100
    interp = PchipInterpolator(x_old, data, axis=0)
    return interp(x_new)


def normalize_all_cycles(
    data: np.ndarray,
    heel_strikes: np.ndarray,
    n_points: int = _CFG.normalization.n_points,
) -> np.ndarray:
    """Time-normalize all gait cycles identified by heel strike indices.

    Parameters
    ----------
    data : np.ndarray
        Continuous signal, shape ``(n_frames,)`` or
        ``(n_frames, n_channels)``.
    heel_strikes : np.ndarray
        1-D array of heel strike sample indices (from ``detect_gait_events``).
        Each consecutive pair defines one gait cycle.
    n_points : int
        Number of output time points per cycle. Default 101.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_cycles, n_points)`` or
        ``(n_cycles, n_points, n_channels)``, one row per complete cycle.
    """
    cycles = []
    for i in range(len(heel_strikes) - 1):
        cycle = data[heel_strikes[i] : heel_strikes[i + 1] + 1]
        cycles.append(normalize_gait_cycle(cycle, n_points))
    return np.array(cycles)
