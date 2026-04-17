"""Tests for preprocessing.py — validate against expected MATLAB behavior."""

import numpy as np
import pytest
from gait_ml.preprocessing import (
    butterworth_lowpass,
    normalize_gait_cycle,
    detect_missing_markers,
)


def test_butterworth_preserves_shape_1d():
    data = np.random.randn(200)
    result = butterworth_lowpass(data, cutoff_hz=6.0, sample_rate_hz=100.0)
    assert result.shape == data.shape


def test_butterworth_preserves_shape_2d():
    data = np.random.randn(200, 6)
    result = butterworth_lowpass(data, cutoff_hz=6.0, sample_rate_hz=100.0)
    assert result.shape == data.shape


def test_butterworth_attenuates_high_frequency():
    """Filter should reduce amplitude of a high-frequency sine wave."""
    t = np.linspace(0, 2, 200)
    high_freq = np.sin(2 * np.pi * 30 * t)  # 30 Hz — well above 6 Hz cutoff
    filtered = butterworth_lowpass(high_freq, cutoff_hz=6.0, sample_rate_hz=100.0)
    assert np.std(filtered) < 0.1 * np.std(high_freq)


def test_normalize_gait_cycle_output_length():
    data = np.random.randn(85)  # arbitrary cycle length
    result = normalize_gait_cycle(data, n_points=101)
    assert len(result) == 101


def test_normalize_gait_cycle_2d():
    data = np.random.randn(85, 3)
    result = normalize_gait_cycle(data, n_points=101)
    assert result.shape == (101, 3)


def test_detect_missing_markers_zeros():
    data = np.ones((100, 6))
    data[10:15, :] = 0.0
    missing = detect_missing_markers(data)
    assert missing[10:15].all()
    assert not missing[0:10].any()


def test_detect_missing_markers_nan():
    data = np.ones((100, 6))
    data[20, 2] = np.nan
    missing = detect_missing_markers(data)
    assert missing[20]
