"""
segmentation.py Ñ Gait event detection and cycle segmentation.

Primary method: GRF threshold crossing (requires force plate data).
Heel strike = vertical GRF crosses threshold going up.
Toe off = vertical GRF crosses threshold going down.
"""

import numpy as np


DEFAULT_GRF_THRESHOLD_N: float = 20.0  # Newtons Ñ standard lab threshold


def detect_gait_events_grf(
    vertical_grf: np.ndarray,
    sample_rate_hz: float,
    threshold_n: float = DEFAULT_GRF_THRESHOLD_N,
) -> dict[str, np.ndarray]:
    """Detect heel strike and toe-off events from vertical GRF.

    Parameters
    ----------
    vertical_grf : np.ndarray
        Vertical ground reaction force time series in Newtons, shape (n_frames,).
    sample_rate_hz : float
        Force plate sampling rate in Hz.
    threshold_n : float
        Force threshold in Newtons. Default 20N is standard for lab data.

    Returns
    -------
    dict
        Keys 'heel_strike' and 'toe_off', values are arrays of frame indices.
    """
    above = vertical_grf > threshold_n
    above_int = above.astype(int)
    diff = np.diff(above_int)

    heel_strikes = np.where(diff == 1)[0] + 1   # rising edge
    toe_offs = np.where(diff == -1)[0] + 1       # falling edge

    return {"heel_strike": heel_strikes, "toe_off": toe_offs}


def extract_gait_cycles(
    data: np.ndarray,
    heel_strikes: np.ndarray,
    min_frames: int = 20,
) -> list[np.ndarray]:
    """Extract individual gait cycles from a continuous time series.

    Parameters
    ----------
    data : np.ndarray
        Continuous data, shape (n_frames,) or (n_frames, n_channels).
    heel_strikes : np.ndarray
        Frame indices of heel strike events.
    min_frames : int
        Minimum frames for a valid cycle (rejects very short segments).

    Returns
    -------
    list[np.ndarray]
        List of gait cycle arrays, each shape (cycle_frames,) or
        (cycle_frames, n_channels).
    """
    cycles = []
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        if (end - start) >= min_frames:
            cycles.append(data[start:end])
    return cycles


def get_stance_phase(
    data: np.ndarray,
    heel_strike: int,
    toe_off: int,
) -> np.ndarray:
    """Extract stance phase data between heel strike and toe off.

    Parameters
    ----------
    data : np.ndarray
        Continuous data, shape (n_frames,) or (n_frames, n_channels).
    heel_strike : int
        Heel strike frame index.
    toe_off : int
        Toe off frame index.

    Returns
    -------
    np.ndarray
        Stance phase data.
    """
    return data[heel_strike:toe_off]
