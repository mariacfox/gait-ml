"""Tests for kinematics.py Ñ validate joint angle computation geometry."""

import numpy as np
import pytest
from gait_ml.kinematics import segment_angle_sagittal, joint_angle, get_marker


def test_segment_angle_vertical():
    """Vertical segment (straight up) should return 0 degrees."""
    proximal = np.array([[0.0, 0.0, 0.0]] * 10)
    distal = np.array([[0.0, 1.0, 0.0]] * 10)  # directly above
    angles = segment_angle_sagittal(proximal, distal)
    np.testing.assert_allclose(angles, 90.0, atol=0.01)


def test_joint_angle_zero_when_equal():
    """Joint angle should be zero when both segments are parallel."""
    seg1 = np.zeros(50)
    seg2 = np.zeros(50)
    result = joint_angle(seg1, seg2)
    np.testing.assert_array_equal(result, 0.0)


def test_get_marker_correct_columns():
    """get_marker should return columns idx*3 : idx*3+3."""
    data = np.arange(30).reshape(2, 15).astype(float)
    marker_names = ["A", "B", "C", "D", "E"]
    result = get_marker(data, marker_names, "B")
    expected = data[:, 3:6]
    np.testing.assert_array_equal(result, expected)


def test_get_marker_unknown_raises():
    data = np.zeros((10, 9))
    marker_names = ["A", "B", "C"]
    with pytest.raises(ValueError):
        get_marker(data, marker_names, "Z")
