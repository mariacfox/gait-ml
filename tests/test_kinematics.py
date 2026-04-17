"""Tests for kinematics.py — validate joint angle computation geometry."""

import numpy as np
import pytest
from gait_ml.kinematics import (
    ankle_dorsiflexion,
    get_marker,
    hip_flexion,
    knee_flexion,
    segment_angle_from_down,
    static_ankle_reference,
    static_hip_reference,
    static_knee_reference,
    three_point_sagittal_angle,
)


# ---------------------------------------------------------------------------
# three_point_sagittal_angle
# ---------------------------------------------------------------------------


def test_three_point_straight_leg_is_zero():
    """Collinear markers (full extension) should give 0°."""
    # TRO above KNEE above ANK, all on Y axis
    prox = np.array([[0.0, 400.0, 0.0]] * 10)
    joint = np.array([[0.0, 200.0, 0.0]] * 10)
    dist = np.array([[0.0, 0.0, 0.0]] * 10)
    angles = three_point_sagittal_angle(prox, joint, dist)
    np.testing.assert_allclose(angles, 0.0, atol=1e-8)


def test_three_point_right_angle_is_90():
    """Segments at 90° should return 90°."""
    prox = np.array([[0.0, 1.0, 0.0]] * 5)   # above joint
    joint = np.zeros((5, 3))
    dist = np.array([[1.0, 0.0, 0.0]] * 5)   # beside joint (forward)
    # v_prox = [0,1,0], v_dist = [1,0,0] → cos = 0 → arccos = 90° → supplement = 90°
    angles = three_point_sagittal_angle(prox, joint, dist)
    np.testing.assert_allclose(angles, 90.0, atol=1e-8)


def test_three_point_no_wraparound():
    """Values must stay in [0°, 180°] — no ±360° artefacts."""
    # Simulate a downward-pointing limb at various flexion angles
    rng = np.random.default_rng(42)
    n = 100
    prox = np.column_stack([rng.uniform(-50, 50, n),
                            rng.uniform(300, 500, n),
                            np.zeros(n)])
    joint = np.zeros((n, 3))
    dist = np.column_stack([rng.uniform(-50, 50, n),
                            rng.uniform(-400, -100, n),
                            np.zeros(n)])
    angles = three_point_sagittal_angle(prox, joint, dist)
    assert np.all(angles >= 0.0)
    assert np.all(angles <= 180.0)


def test_three_point_mediolateral_ignored():
    """Z displacement should not affect the sagittal angle."""
    prox = np.array([[0.0, 1.0, 0.0]] * 5)
    joint = np.zeros((5, 3))
    dist_no_z = np.array([[0.0, -1.0, 0.0]] * 5)
    dist_with_z = np.array([[0.0, -1.0, 100.0]] * 5)
    np.testing.assert_allclose(
        three_point_sagittal_angle(prox, joint, dist_no_z),
        three_point_sagittal_angle(prox, joint, dist_with_z),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# segment_angle_from_down
# ---------------------------------------------------------------------------


def test_segment_angle_from_down_vertical():
    """Segment pointing straight down should be 0°."""
    prox = np.array([[0.0, 100.0, 0.0]] * 10)
    dist = np.array([[0.0, 0.0, 0.0]] * 10)  # directly below
    np.testing.assert_allclose(segment_angle_from_down(prox, dist), 0.0, atol=1e-10)


def test_segment_angle_from_down_forward():
    """Segment pointing forward (along X) should be +90°."""
    prox = np.zeros((5, 3))
    dist = np.array([[1.0, 0.0, 0.0]] * 5)
    np.testing.assert_allclose(segment_angle_from_down(prox, dist), 90.0, atol=1e-10)


def test_segment_angle_from_down_forward_lean():
    """Slightly forward-leaning segment should give small positive angle."""
    prox = np.array([[0.0, 100.0, 0.0]] * 5)
    dist = np.array([[10.0, 0.0, 0.0]] * 5)  # 10 mm forward, 100 mm down
    angles = segment_angle_from_down(prox, dist)
    assert np.all(angles > 0)
    assert np.all(angles < 10)  # small positive angle


# ---------------------------------------------------------------------------
# get_marker
# ---------------------------------------------------------------------------


def test_get_marker_correct_columns():
    data = np.arange(30).reshape(2, 15).astype(float)
    marker_names = ["A", "B", "C", "D", "E"]
    result = get_marker(data, marker_names, "B")
    np.testing.assert_array_equal(result, data[:, 3:6])


def test_get_marker_unknown_raises():
    data = np.zeros((10, 9))
    marker_names = ["A", "B", "C"]
    with pytest.raises(ValueError):
        get_marker(data, marker_names, "Z")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_marker_data(
    positions: dict[str, np.ndarray], n_frames: int = 10
) -> tuple[np.ndarray, list[str]]:
    names = list(positions.keys())
    data = np.zeros((n_frames, len(names) * 3))
    for i, name in enumerate(names):
        pos = np.asarray(positions[name], dtype=float)
        data[:, i * 3 : i * 3 + 3] = pos[np.newaxis, :] if pos.ndim == 1 else pos
    return data, names


# ---------------------------------------------------------------------------
# knee_flexion
# ---------------------------------------------------------------------------


def test_knee_flexion_zero_at_extension():
    """Collinear TRO-KNEE-ANK (straight leg) → 0°."""
    positions = {
        "LTRO": [0.0, 400.0, 0.0],
        "LKNEE": [0.0, 200.0, 0.0],
        "LANK": [0.0, 0.0, 0.0],
    }
    data, names = _make_marker_data(positions)
    angles = knee_flexion(data, names, side="L")
    np.testing.assert_allclose(angles, 0.0, atol=1e-8)


def test_knee_flexion_positive_when_flexed():
    """Shank angled posteriorly relative to thigh → positive flexion angle."""
    positions = {
        "LTRO": [0.0, 400.0, 0.0],
        "LKNEE": [10.0, 200.0, 0.0],   # thigh tilts slightly forward
        "LANK": [-60.0, 0.0, 0.0],     # shank tilts backward (flexion)
    }
    data, names = _make_marker_data(positions)
    angles = knee_flexion(data, names, side="L")
    assert np.all(angles > 0)


def test_knee_flexion_right_side():
    """Right side geometry should mirror left."""
    positions = {
        "RTRO": [0.0, 400.0, 100.0],
        "RKNEE": [0.0, 200.0, 100.0],
        "RANK": [0.0, 0.0, 100.0],
    }
    data, names = _make_marker_data(positions)
    angles = knee_flexion(data, names, side="R")
    np.testing.assert_allclose(angles, 0.0, atol=1e-8)


def test_knee_flexion_static_ref_subtracted():
    """static_ref should shift all output angles by that amount."""
    positions = {
        "LTRO": [0.0, 400.0, 0.0],
        "LKNEE": [0.0, 200.0, 0.0],
        "LANK": [0.0, 0.0, 0.0],
    }
    data, names = _make_marker_data(positions)
    raw = knee_flexion(data, names, side="L", static_ref=0.0)
    corrected = knee_flexion(data, names, side="L", static_ref=5.0)
    np.testing.assert_allclose(corrected, raw - 5.0, atol=1e-10)


def test_static_knee_reference_returns_mean_angle():
    """static_knee_reference should equal mean of knee_flexion on same data."""
    positions = {
        "LTRO": [10.0, 400.0, 0.0],   # slight AP offset → nonzero reference
        "LKNEE": [0.0, 200.0, 0.0],
        "LANK": [0.0, 0.0, 0.0],
    }
    data, names = _make_marker_data(positions, n_frames=20)
    ref = static_knee_reference(data, names, side="L")
    raw_angles = knee_flexion(data, names, side="L", static_ref=0.0)
    np.testing.assert_allclose(ref, np.nanmean(raw_angles), atol=1e-10)


def test_knee_flexion_no_wraparound():
    """Output must be in [0°, 180°] — no ±360° artefacts from subtraction."""
    # Simulate a realistic flexing knee over time
    n = 50
    tro = np.column_stack([np.zeros(n), np.full(n, 400.0), np.zeros(n)])
    knee = np.zeros((n, 3))
    knee[:, 1] = 200.0
    # Ankle oscillates from directly below (extension) to posteriorly offset (flexion)
    ankle_x = np.linspace(0, -100, n)
    ankle = np.column_stack([ankle_x, np.zeros(n), np.zeros(n)])
    data = np.hstack([tro, knee, ankle])
    names = ["LTRO", "LKNEE", "LANK"]
    angles = knee_flexion(data, names, side="L")
    assert np.all(angles >= 0.0)
    assert np.all(angles <= 180.0)


# ---------------------------------------------------------------------------
# hip_flexion
# ---------------------------------------------------------------------------


def test_hip_flexion_returns_finite():
    """hip_flexion should return finite values for valid marker positions."""
    positions = {
        "LASIS": [100.0, 600.0, -100.0],
        "RASIS": [100.0, 600.0, 100.0],
        "LPSIS": [-100.0, 600.0, -100.0],
        "RPSIS": [-100.0, 600.0, 100.0],
        "LTRO": [0.0, 500.0, -100.0],
        "LKNEE": [0.0, 300.0, -100.0],
    }
    data, names = _make_marker_data(positions)
    angles = hip_flexion(data, names, side="L", static_ref=0.0)
    assert np.all(np.isfinite(angles))


def test_static_hip_reference_returns_mean():
    """static_hip_reference should equal mean of hip_flexion(static_ref=0) on same data."""
    positions = {
        "LASIS": [100.0, 600.0, -100.0],
        "RASIS": [100.0, 600.0, 100.0],
        "LPSIS": [-100.0, 600.0, -100.0],
        "RPSIS": [-100.0, 600.0, 100.0],
        "LTRO": [0.0, 500.0, -100.0],
        "LKNEE": [0.0, 300.0, -100.0],
    }
    data, names = _make_marker_data(positions, n_frames=20)
    ref = static_hip_reference(data, names, side="L")
    raw = hip_flexion(data, names, side="L", static_ref=0.0)
    np.testing.assert_allclose(ref, np.nanmean(raw), atol=1e-10)


def test_hip_flexion_symmetric():
    """Symmetric pelvis and symmetric thighs → equal L and R hip angles."""
    positions = {
        "LASIS": [100.0, 600.0, -100.0],
        "RASIS": [100.0, 600.0, 100.0],
        "LPSIS": [-100.0, 600.0, -100.0],
        "RPSIS": [-100.0, 600.0, 100.0],
        "LTRO": [0.0, 500.0, -100.0],
        "LKNEE": [0.0, 300.0, -100.0],
        "RTRO": [0.0, 500.0, 100.0],
        "RKNEE": [0.0, 300.0, 100.0],
    }
    data, names = _make_marker_data(positions)
    np.testing.assert_allclose(
        hip_flexion(data, names, side="L"),
        hip_flexion(data, names, side="R"),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# ankle_dorsiflexion
# ---------------------------------------------------------------------------


def test_ankle_dorsiflexion_zero_at_neutral_with_static_ref():
    """Shank vertical, foot horizontal → 0° when static_ref = standing angle."""
    positions = {
        "LKNEE": [0.0, 300.0, 0.0],
        "LANK": [0.0, 100.0, 0.0],
        "LTOE": [100.0, 100.0, 0.0],  # foot horizontal → three_point gives 90°
    }
    data, names = _make_marker_data(positions)
    # Compute the standing reference from this same "static" posture
    ref = static_ankle_reference(data, names, side="L")
    angles = ankle_dorsiflexion(data, names, side="L", static_ref=ref)
    np.testing.assert_allclose(angles, 0.0, atol=1e-8)


def test_ankle_dorsiflexion_positive_when_dorsiflexed():
    """Foot angled upward relative to shank → positive DF angle."""
    positions = {
        "LKNEE": [0.0, 300.0, 0.0],
        "LANK": [0.0, 100.0, 0.0],
        "LTOE": [80.0, 130.0, 0.0],  # toe higher than ankle → DF
    }
    data, names = _make_marker_data(positions)
    angles = ankle_dorsiflexion(data, names, side="L")
    assert np.all(angles > 0)


def test_ankle_dorsiflexion_right_side():
    """Right side should return finite values."""
    positions = {
        "RKNEE": [0.0, 300.0, 100.0],
        "RANK": [0.0, 100.0, 100.0],
        "RTOE": [100.0, 100.0, 100.0],
    }
    data, names = _make_marker_data(positions)
    angles = ankle_dorsiflexion(data, names, side="R")
    assert np.all(np.isfinite(angles))
