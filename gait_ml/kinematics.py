"""
kinematics.py Ñ Joint angle computation from 3D marker trajectories.

All angles returned in degrees. Sagittal plane (flexion/extension) is the
priority; frontal and transverse plane angles can be added once marker set
is confirmed.

Coordinate system assumption (verify against QualityAssist output):
  X = mediolateral (right positive)
  Y = vertical (up positive)
  Z = anteroposterior (forward negative in many lab setups Ñ confirm)

TODO: Confirm marker names and segment definitions against your marker set
before using these functions. Cross-validate output against MATLAB scripts.
"""

import numpy as np


def get_marker(
    data: np.ndarray,
    marker_names: list[str],
    marker: str,
) -> np.ndarray:
    """Extract XYZ trajectory for a single marker.

    Parameters
    ----------
    data : np.ndarray
        Marker data array, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered list of marker names corresponding to data columns.
    marker : str
        Name of the marker to extract.

    Returns
    -------
    np.ndarray
        Shape (n_frames, 3) array of XYZ positions.
    """
    idx = marker_names.index(marker)
    return data[:, idx * 3 : idx * 3 + 3]


def segment_angle_sagittal(
    proximal: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    """Compute sagittal plane segment angle from two marker positions.

    Angle is measured from vertical in the sagittal (XY) plane.

    Parameters
    ----------
    proximal : np.ndarray
        Proximal marker positions, shape (n_frames, 3).
    distal : np.ndarray
        Distal marker positions, shape (n_frames, 3).

    Returns
    -------
    np.ndarray
        Segment angle in degrees, shape (n_frames,).
    """
    vec = distal - proximal
    # Sagittal plane: use Y (vertical) and Z (AP) components
    angle = np.degrees(np.arctan2(vec[:, 2], vec[:, 1]))
    return angle


def joint_angle(
    proximal_segment_angle: np.ndarray,
    distal_segment_angle: np.ndarray,
) -> np.ndarray:
    """Compute joint angle as relative angle between two segments.

    Parameters
    ----------
    proximal_segment_angle : np.ndarray
        Proximal segment angle in degrees, shape (n_frames,).
    distal_segment_angle : np.ndarray
        Distal segment angle in degrees, shape (n_frames,).

    Returns
    -------
    np.ndarray
        Joint angle in degrees, shape (n_frames,).
        Positive = flexion by convention (confirm against MATLAB).
    """
    return proximal_segment_angle - distal_segment_angle


def knee_flexion(
    data: np.ndarray,
    marker_names: list[str],
    proximal_marker: str = "LKNE",   # TODO: confirm marker names for your set
    distal_marker: str = "LANK",
    hip_marker: str = "LASI",
) -> np.ndarray:
    """Compute knee flexion angle in the sagittal plane.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered marker names.
    proximal_marker : str
        Knee joint center marker name. TODO: confirm for your marker set.
    distal_marker : str
        Ankle joint center marker name. TODO: confirm for your marker set.
    hip_marker : str
        Hip joint center or ASIS marker name. TODO: confirm for your marker set.

    Returns
    -------
    np.ndarray
        Knee flexion angle in degrees, shape (n_frames,).

    Notes
    -----
    Marker names are placeholders. Update to match your actual QualityAssist
    marker labels before use.
    """
    hip = get_marker(data, marker_names, hip_marker)
    knee = get_marker(data, marker_names, proximal_marker)
    ankle = get_marker(data, marker_names, distal_marker)

    thigh_angle = segment_angle_sagittal(hip, knee)
    shank_angle = segment_angle_sagittal(knee, ankle)
    return joint_angle(thigh_angle, shank_angle)


def hip_flexion(
    data: np.ndarray,
    marker_names: list[str],
    pelvis_marker: str = "SACR",    # TODO: confirm
    hip_marker: str = "LASI",       # TODO: confirm
    knee_marker: str = "LKNE",      # TODO: confirm
) -> np.ndarray:
    """Compute hip flexion angle in the sagittal plane.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered marker names.
    pelvis_marker, hip_marker, knee_marker : str
        Marker names. TODO: confirm for your marker set.

    Returns
    -------
    np.ndarray
        Hip flexion angle in degrees, shape (n_frames,).
    """
    pelvis = get_marker(data, marker_names, pelvis_marker)
    hip = get_marker(data, marker_names, hip_marker)
    knee = get_marker(data, marker_names, knee_marker)

    pelvis_angle = segment_angle_sagittal(pelvis, hip)
    thigh_angle = segment_angle_sagittal(hip, knee)
    return joint_angle(pelvis_angle, thigh_angle)


def ankle_dorsiflexion(
    data: np.ndarray,
    marker_names: list[str],
    knee_marker: str = "LKNE",    # TODO: confirm
    ankle_marker: str = "LANK",   # TODO: confirm
    toe_marker: str = "LTOE",     # TODO: confirm
) -> np.ndarray:
    """Compute ankle dorsiflexion angle in the sagittal plane.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered marker names.
    knee_marker, ankle_marker, toe_marker : str
        Marker names. TODO: confirm for your marker set.

    Returns
    -------
    np.ndarray
        Ankle dorsiflexion angle in degrees, shape (n_frames,).
        Positive = dorsiflexion by convention (confirm against MATLAB).
    """
    knee = get_marker(data, marker_names, knee_marker)
    ankle = get_marker(data, marker_names, ankle_marker)
    toe = get_marker(data, marker_names, toe_marker)

    shank_angle = segment_angle_sagittal(knee, ankle)
    foot_angle = segment_angle_sagittal(ankle, toe)
    return joint_angle(shank_angle, foot_angle)
