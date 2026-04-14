"""
preprocessing.py Ñ Signal filtering and time normalization.

All functions are pure (no side effects). Filtering matches MATLAB filtfilt behavior.
Validate outputs against MATLAB scripts using notebooks/02_preprocessing_validation.ipynb.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


# Field standard: 101 points = 0-100% of gait cycle
GAIT_CYCLE_POINTS: int = 101

# Default filter cutoffs Ñ confirm against original MATLAB scripts
KINEMATIC_LOWPASS_HZ: float = 6.0
GRF_LOWPASS_HZ: float = 50.0


def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Equivalent to MATLAB's filtfilt with a Butterworth design.

    Parameters
    ----------
    data : np.ndarray
        Input signal, shape (n_frames,) or (n_frames, n_channels).
    cutoff_hz : float
        Filter cutoff frequency in Hz.
    sample_rate_hz : float
        Signal sampling rate in Hz.
    order : int
        Filter order. Default 4 matches common biomechanics convention.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input.
    """
    nyq = 0.5 * sample_rate_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def normalize_gait_cycle(
    data: np.ndarray,
    n_points: int = GAIT_CYCLE_POINTS,
    kind: str = "cubic",
) -> np.ndarray:
    """Time-normalize a single gait cycle to n_points.

    Parameters
    ----------
    data : np.ndarray
        Single gait cycle data, shape (n_frames,) or (n_frames, n_channels).
    n_points : int
        Number of output points. Default 101 (0-100% of cycle).
    kind : str
        Interpolation method passed to scipy.interpolate.interp1d.

    Returns
    -------
    np.ndarray
        Time-normalized data, shape (n_points,) or (n_points, n_channels).
    """
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, n_points)
    f = interp1d(x_old, data, axis=0, kind=kind)
    return f(x_new)


def filter_markers(
    marker_data: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float = KINEMATIC_LOWPASS_HZ,
) -> np.ndarray:
    """Filter marker trajectory data with a low-pass Butterworth filter.

    Parameters
    ----------
    marker_data : np.ndarray
        Marker positions, shape (n_frames, n_markers * 3).
    sample_rate_hz : float
        Kinematic sampling rate in Hz.
    cutoff_hz : float
        Low-pass cutoff frequency. Confirm against MATLAB scripts.

    Returns
    -------
    np.ndarray
        Filtered marker data, same shape as input.
    """
    return butterworth_lowpass(marker_data, cutoff_hz, sample_rate_hz)


def filter_grf(
    grf_data: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float = GRF_LOWPASS_HZ,
) -> np.ndarray:
    """Filter ground reaction force data with a low-pass Butterworth filter.

    Parameters
    ----------
    grf_data : np.ndarray
        GRF time series, shape (n_frames,) or (n_frames, 3).
    sample_rate_hz : float
        Force plate sampling rate in Hz.
    cutoff_hz : float
        Low-pass cutoff frequency. Typically 50Hz for GRF.

    Returns
    -------
    np.ndarray
        Filtered GRF data, same shape as input.
    """
    return butterworth_lowpass(grf_data, cutoff_hz, sample_rate_hz)


def detect_missing_markers(
    marker_data: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Detect frames with missing (zeroed or NaN) marker data.

    Parameters
    ----------
    marker_data : np.ndarray
        Marker positions, shape (n_frames, n_channels).
    threshold : float
        Values at or below this are considered missing. Default 0.0.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_frames,), True where data is missing.
    """
    is_zero = np.all(np.abs(marker_data) <= threshold, axis=1)
    is_nan = np.any(np.isnan(marker_data), axis=1)
    return is_zero | is_nan
