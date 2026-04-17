"""Tests for grf.py — GRF feature extraction from stance-phase force data."""

import numpy as np
import pytest
from gait_ml.grf import extract_cop_features, extract_grf_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stance(
    n_frames: int = 672,  # ~0.6 s at 1120 Hz
    peak_n: float = 700.0,
    body_weight_n: float = 700.0,
    shape: str = "half_sine",
) -> tuple[np.ndarray, float, float]:
    """Return (vertical_grf_stance, sample_rate_hz, body_weight_n)."""
    t = np.linspace(0, np.pi, n_frames)
    if shape == "half_sine":
        grf = peak_n * np.sin(t)
    elif shape == "flat":
        grf = np.full(n_frames, peak_n)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return grf, 1120.0, body_weight_n


# ---------------------------------------------------------------------------
# extract_grf_features — return structure
# ---------------------------------------------------------------------------


def test_extract_grf_features_returns_all_keys():
    grf, fs, bw = _make_stance()
    feats = extract_grf_features(grf, fs, bw)
    for key in ("peak_vgrf_bw", "loading_rate_bw_s", "impulse_bw_s",
                 "contact_time_s", "impact_peak_bw"):
        assert key in feats, f"Missing key: {key}"


def test_extract_grf_features_all_finite_for_valid_input():
    grf, fs, bw = _make_stance()
    feats = extract_grf_features(grf, fs, bw)
    # peak, loading_rate, impulse, contact_time must be finite
    for key in ("peak_vgrf_bw", "loading_rate_bw_s", "impulse_bw_s", "contact_time_s"):
        assert np.isfinite(feats[key]), f"{key} is not finite: {feats[key]}"


# ---------------------------------------------------------------------------
# peak_vgrf_bw
# ---------------------------------------------------------------------------


def test_peak_vgrf_normalized_correctly():
    """Peak GRF of 700 N / 700 N body weight → 1.0 BW."""
    grf, fs, bw = _make_stance(peak_n=700.0, body_weight_n=700.0)
    feats = extract_grf_features(grf, fs, bw)
    assert abs(feats["peak_vgrf_bw"] - 1.0) < 1e-6


def test_peak_vgrf_scales_with_body_weight():
    """Halving body weight should double the normalized peak."""
    grf, fs, bw = _make_stance(peak_n=700.0, body_weight_n=700.0)
    feats_full = extract_grf_features(grf, fs, bw)
    feats_half = extract_grf_features(grf, fs, bw / 2)
    assert abs(feats_half["peak_vgrf_bw"] - 2 * feats_full["peak_vgrf_bw"]) < 1e-6


# ---------------------------------------------------------------------------
# contact_time_s
# ---------------------------------------------------------------------------


def test_contact_time_correct():
    """Contact time = n_frames / sample_rate_hz."""
    n = 672
    grf, fs, bw = _make_stance(n_frames=n)
    feats = extract_grf_features(grf, fs, bw)
    expected = n / fs
    assert abs(feats["contact_time_s"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# impulse_bw_s
# ---------------------------------------------------------------------------


def test_impulse_positive_for_positive_grf():
    """Impulse of a positive GRF signal must be positive."""
    grf, fs, bw = _make_stance()
    feats = extract_grf_features(grf, fs, bw)
    assert feats["impulse_bw_s"] > 0


def test_impulse_scales_with_force():
    """Doubling the GRF force should double the impulse."""
    grf, fs, bw = _make_stance(peak_n=700.0, body_weight_n=700.0)
    feats_1x = extract_grf_features(grf, fs, bw)
    feats_2x = extract_grf_features(grf * 2, fs, bw)
    assert abs(feats_2x["impulse_bw_s"] / feats_1x["impulse_bw_s"] - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# loading_rate_bw_s
# ---------------------------------------------------------------------------


def test_loading_rate_positive_for_rising_onset():
    """Loading rate should be positive when force rises at start of stance."""
    grf, fs, bw = _make_stance()
    feats = extract_grf_features(grf, fs, bw)
    assert feats["loading_rate_bw_s"] > 0


# ---------------------------------------------------------------------------
# impact_peak_bw — presence / absence
# ---------------------------------------------------------------------------


def test_impact_peak_nan_for_half_sine():
    """Smooth half-sine has no impact transient → impact_peak_bw should be NaN."""
    grf, fs, bw = _make_stance(shape="half_sine")
    feats = extract_grf_features(grf, fs, bw)
    # Half-sine rises monotonically in first 30% → no local maximum → NaN
    assert np.isnan(feats["impact_peak_bw"])


def test_impact_peak_detected_when_present():
    """Signal with a clear impact transient should have a finite impact_peak_bw."""
    # Construct: sharp rise to 1.2 BW, dip to 0.9 BW, then peak at 1.1 BW
    n = 672
    bw = 700.0
    t = np.linspace(0, np.pi, n)
    grf = bw * np.sin(t)  # base half-sine

    # Inject impact transient in first 15% of stance
    impact_end = int(n * 0.15)
    # Triangular impact spike
    spike = np.zeros(n)
    half = impact_end // 2
    spike[:half] = np.linspace(0, 1.3 * bw, half)
    spike[half:impact_end] = np.linspace(1.3 * bw, 0.85 * bw, impact_end - half)
    grf[:impact_end] = spike[:impact_end]

    feats = extract_grf_features(grf, 1120.0, bw)
    assert np.isfinite(feats["impact_peak_bw"])
    assert feats["impact_peak_bw"] > 0


# ---------------------------------------------------------------------------
# extract_cop_features
# ---------------------------------------------------------------------------


def test_extract_cop_features_returns_all_keys():
    cop_x = np.random.default_rng(0).normal(0, 5, 5000)
    cop_y = np.random.default_rng(1).normal(0, 8, 5000)
    feats = extract_cop_features(cop_x, cop_y, sample_rate_hz=1120.0)
    for key in ("cop_range_ml_mm", "cop_range_ap_mm",
                 "cop_sway_velocity_mm_s", "cop_area_mm2"):
        assert key in feats


def test_cop_range_ml_correct():
    """ML range should equal max(cop_x) - min(cop_x)."""
    cop_x = np.array([-5.0, 0.0, 10.0, 3.0])
    cop_y = np.zeros(4)
    feats = extract_cop_features(cop_x, cop_y, sample_rate_hz=1120.0)
    assert abs(feats["cop_range_ml_mm"] - 15.0) < 1e-9


def test_cop_range_ap_correct():
    """AP range should equal max(cop_y) - min(cop_y)."""
    cop_x = np.zeros(4)
    cop_y = np.array([1.0, -3.0, 5.0, 0.0])
    feats = extract_cop_features(cop_x, cop_y, sample_rate_hz=1120.0)
    assert abs(feats["cop_range_ap_mm"] - 8.0) < 1e-9


def test_cop_sway_velocity_positive():
    """Sway velocity should be non-negative for any non-constant signal."""
    cop_x = np.sin(np.linspace(0, 4 * np.pi, 1000)) * 5
    cop_y = np.cos(np.linspace(0, 4 * np.pi, 1000)) * 8
    feats = extract_cop_features(cop_x, cop_y, sample_rate_hz=1120.0)
    assert feats["cop_sway_velocity_mm_s"] >= 0


def test_cop_area_non_negative():
    """Sway area must be non-negative."""
    rng = np.random.default_rng(99)
    cop_x = rng.normal(0, 5, 2000)
    cop_y = rng.normal(0, 8, 2000)
    feats = extract_cop_features(cop_x, cop_y, sample_rate_hz=1120.0)
    assert feats["cop_area_mm2"] >= 0


def test_cop_area_larger_for_wider_sway():
    """Wider sway (larger std) should produce a larger CoP area."""
    rng = np.random.default_rng(7)
    small = extract_cop_features(rng.normal(0, 2, 2000), rng.normal(0, 2, 2000), 1120.0)
    rng2 = np.random.default_rng(7)
    large = extract_cop_features(rng2.normal(0, 20, 2000), rng2.normal(0, 20, 2000), 1120.0)
    assert large["cop_area_mm2"] > small["cop_area_mm2"]
