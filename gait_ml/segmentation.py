"""
segmentation.py — Gait event detection and cycle segmentation.

Primary method: GRF threshold crossing (requires force plate data).
Heel strike = vertical GRF crosses threshold going up.
Toe off = vertical GRF crosses threshold going down.
"""

import numpy as np

from gait_ml.config import DEFAULT_LAB_CONFIG as _CFG


def detect_gait_events_grf(
    vertical_grf: np.ndarray,
    sample_rate_hz: float,
    threshold_n: float = _CFG.gait_events.threshold_n,
    min_peak_n: float = _CFG.gait_events.min_stance_peak_n,
) -> dict[str, np.ndarray]:
    """Detect heel strike and toe-off events from vertical GRF.

    Implements ``gaitEventDetection.m``: threshold crossings identify candidate
    stances; stances whose peak force is below ``min_peak_n`` are discarded as
    foot crossover or treadmill artifact.

    Parameters
    ----------
    vertical_grf : np.ndarray
        Vertical ground reaction force time series in Newtons, shape (n_frames,).
    sample_rate_hz : float
        Force plate sampling rate in Hz.
    threshold_n : float
        Force threshold in Newtons for stance detection. Confirmed 15 N from MATLAB.
    min_peak_n : float
        Minimum peak force in Newtons for a valid stance. Confirmed 150 N from MATLAB.
        Stances below this are discarded (foot crossover / artifact filter).

    Returns
    -------
    dict
        Keys ``'heel_strike'`` and ``'toe_off'``, values are arrays of frame
        indices for valid stances only.
    """
    above = vertical_grf > threshold_n
    above_int = above.astype(int)
    diff = np.diff(above_int)

    hs_candidates = np.where(diff == 1)[0] + 1   # rising edge
    to_candidates = np.where(diff == -1)[0] + 1  # falling edge

    # Pair each heel strike with the next toe-off, then apply peak filter
    heel_strikes = []
    toe_offs = []
    for hs in hs_candidates:
        subsequent = to_candidates[to_candidates > hs]
        if len(subsequent) == 0:
            break
        to = subsequent[0]
        if np.max(vertical_grf[hs:to]) >= min_peak_n:
            heel_strikes.append(hs)
            toe_offs.append(to)

    return {
        "heel_strike": np.array(heel_strikes, dtype=int),
        "toe_off": np.array(toe_offs, dtype=int),
    }


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
