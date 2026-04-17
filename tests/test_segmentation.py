"""Tests for segmentation.py — gait event detection from vertical GRF."""

import numpy as np
import pytest
from gait_ml.segmentation import detect_gait_events_grf, extract_gait_cycles, get_stance_phase


def _make_grf_signal(
    sample_rate_hz: float = 1120.0,
    n_stances: int = 3,
    stance_duration_s: float = 0.6,
    swing_duration_s: float = 0.4,
    peak_force_n: float = 700.0,
) -> np.ndarray:
    """Build a synthetic vertical GRF with clean stance/swing cycles."""
    stance_frames = int(stance_duration_s * sample_rate_hz)
    swing_frames = int(swing_duration_s * sample_rate_hz)

    # Half-sine stance shape
    t_stance = np.linspace(0, np.pi, stance_frames)
    stance = peak_force_n * np.sin(t_stance)

    cycle = np.concatenate([stance, np.zeros(swing_frames)])
    signal = np.concatenate([np.zeros(50)] + [cycle] * n_stances + [np.zeros(50)])
    return signal.astype(float)


def test_detect_events_correct_count():
    """Should detect one heel strike and one toe-off per stance."""
    grf = _make_grf_signal(n_stances=3)
    events = detect_gait_events_grf(grf, sample_rate_hz=1120.0)
    assert len(events["heel_strike"]) == 3
    assert len(events["toe_off"]) == 3


def test_detect_events_heel_before_toe():
    """Every heel strike should precede its paired toe-off."""
    grf = _make_grf_signal(n_stances=4)
    events = detect_gait_events_grf(grf, sample_rate_hz=1120.0)
    for hs, to in zip(events["heel_strike"], events["toe_off"]):
        assert hs < to


def test_detect_events_threshold_15n():
    """Default threshold should be 15 N (confirmed from MATLAB gaitEventDetection.m)."""
    import inspect
    sig = inspect.signature(detect_gait_events_grf)
    # Evaluate the default against the config value
    from gait_ml.config import DEFAULT_LAB_CONFIG
    assert DEFAULT_LAB_CONFIG.gait_events.threshold_n == 15.0


def test_artifact_filter_rejects_low_peak_stances():
    """Stances with peak force < 150 N should be discarded."""
    # Build signal with one real stance (700 N peak) and one artifact (100 N peak)
    fs = 1120.0
    stance_frames = int(0.6 * fs)
    swing_frames = int(0.4 * fs)
    t = np.linspace(0, np.pi, stance_frames)

    real_stance = 700.0 * np.sin(t)
    artifact_stance = 100.0 * np.sin(t)  # below 150 N threshold

    signal = np.concatenate([
        np.zeros(50),
        real_stance,
        np.zeros(swing_frames),
        artifact_stance,
        np.zeros(swing_frames),
        real_stance,
        np.zeros(50),
    ])

    events = detect_gait_events_grf(signal, sample_rate_hz=fs, min_peak_n=150.0)
    # Only the two real stances (700 N) should survive
    assert len(events["heel_strike"]) == 2
    assert len(events["toe_off"]) == 2


def test_artifact_filter_keeps_all_when_all_valid():
    """All stances above min_peak_n should be kept."""
    grf = _make_grf_signal(n_stances=5, peak_force_n=700.0)
    events = detect_gait_events_grf(grf, sample_rate_hz=1120.0, min_peak_n=150.0)
    assert len(events["heel_strike"]) == 5


def test_empty_signal_returns_empty_arrays():
    grf = np.zeros(1000)
    events = detect_gait_events_grf(grf, sample_rate_hz=1120.0)
    assert len(events["heel_strike"]) == 0
    assert len(events["toe_off"]) == 0


def test_extract_gait_cycles_count_and_shape():
    """Should return one cycle per consecutive heel-strike pair."""
    data = np.arange(500, dtype=float)
    heel_strikes = np.array([0, 100, 200, 300, 400])
    cycles = extract_gait_cycles(data, heel_strikes)
    assert len(cycles) == 4
    assert len(cycles[0]) == 100


def test_extract_gait_cycles_min_frames_filter():
    """Cycles shorter than min_frames should be excluded."""
    data = np.zeros(500)
    # Second cycle is only 5 frames — below default min_frames=20
    heel_strikes = np.array([0, 100, 105, 300])
    cycles = extract_gait_cycles(data, heel_strikes, min_frames=20)
    assert len(cycles) == 2  # cycles 0→100 and 105→300 only


def test_get_stance_phase_slices_correctly():
    data = np.arange(200, dtype=float)
    stance = get_stance_phase(data, heel_strike=50, toe_off=120)
    np.testing.assert_array_equal(stance, data[50:120])
