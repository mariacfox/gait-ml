"""
kinematics.py — Joint angle computation from 3D marker trajectories.

All angles returned in degrees. Sagittal plane (flexion/extension) is the
priority per Phase 1 requirements.

Coordinate system (confirmed from CLAUDE.md and data inspection):
  X = anteroposterior (forward along treadmill, positive = anterior)
  Y = vertical (up positive)
  Z = mediolateral

## Implementation notes

The naive approach of computing arctan2(X, Y) for each segment and
subtracting (proximal - distal) fails for leg segments because they point
generally *downward* (negative Y), placing arctan2 outputs near ±180°.
Subtraction of two values near ±180° wraps to ~±360°, producing nonsensical
joint angles.

Two strategies are used here to avoid wraparound:

1. **three_point_sagittal_angle** (knee, ankle): computes the supplement of
   the angle between the proximal and distal segment vectors at the joint
   using a dot-product. Always 0° at full extension, increases with flexion.
   No wrapping possible because arccos ∈ [0°, 180°].

2. **segment_angle_from_down** (hip): measures each segment's angle from the
   *downward* vertical (arctan2(X, -Y)) instead of the upward vertical.
   Leg segments pointing mostly downward stay near 0°, well clear of ±180°.

Sign conventions (flexion positive) must be confirmed against MATLAB output —
see notebooks/02_preprocessing_validation.ipynb.
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
        Shape (n_frames, 3) array of XYZ positions in mm.

    Raises
    ------
    ValueError
        If ``marker`` is not found in ``marker_names``.
    """
    idx = marker_names.index(marker)  # raises ValueError if not found
    return data[:, idx * 3 : idx * 3 + 3]


def three_point_sagittal_angle(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    """Flexion angle at a joint in the sagittal (XY) plane.

    Computes the supplement of the angle between the proximal and distal
    segment vectors meeting at ``joint``. Returns 0° when the segments are
    collinear (full extension) and increases with flexion.

    Uses a dot-product formulation (arccos), so the output is always in
    [0°, 180°] with no wraparound.

    Parameters
    ----------
    proximal : np.ndarray
        Proximal marker positions, shape (n_frames, 3). X = AP, Y = vertical.
    joint : np.ndarray
        Joint centre marker positions, shape (n_frames, 3).
    distal : np.ndarray
        Distal marker positions, shape (n_frames, 3).

    Returns
    -------
    np.ndarray
        Flexion angle in degrees, shape (n_frames,). 0° = full extension.
    """
    # Vectors pointing outward from the joint centre
    v_prox = proximal - joint   # toward proximal segment (generally upward)
    v_dist = distal - joint     # toward distal segment (generally downward)

    # Sagittal plane: X (AP, index 0) and Y (vertical, index 1) only
    vp = v_prox[:, :2]
    vd = v_dist[:, :2]

    dot = np.sum(vp * vd, axis=1)
    mag = np.linalg.norm(vp, axis=1) * np.linalg.norm(vd, axis=1)
    # Guard against zero-length vectors (occluded / missing markers)
    safe_mag = np.where(mag > 0, mag, 1.0)
    cos_a = np.clip(dot / safe_mag, -1.0, 1.0)

    # Supplement: 0° when segments are antiparallel (extension)
    return np.degrees(np.pi - np.arccos(cos_a))


def segment_angle_from_down(
    proximal: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    """Angle of a distally-pointing segment from the downward vertical.

    Uses arctan2(X, -Y) so that segments pointing straight down return 0°,
    forward lean returns positive angles, and backward lean returns negative
    angles. Stays well clear of ±180° for normally-oriented limb segments,
    avoiding the wraparound problem that affects arctan2(X, Y) for downward-
    pointing vectors.

    Parameters
    ----------
    proximal : np.ndarray
        Proximal end marker, shape (n_frames, 3).
    distal : np.ndarray
        Distal end marker, shape (n_frames, 3).

    Returns
    -------
    np.ndarray
        Angle in degrees, shape (n_frames,).
        0° = straight down, positive = anterior lean, negative = posterior lean.
    """
    vec = distal - proximal  # points distally (generally downward)
    return np.degrees(np.arctan2(vec[:, 0], -vec[:, 1]))


def static_knee_reference(
    static_data: np.ndarray,
    static_marker_names: list[str],
    side: str = "L",
) -> float:
    """Compute the mean knee angle from a static standing trial (Tcap).

    This reference angle is subtracted from dynamic knee angles to correct
    for marker placement offset — the same correction MATLAB applies via
    ``refknee``. Without this, the greater trochanter's AP offset from the
    femoral axis introduces a systematic ~5–10° bias.

    Parameters
    ----------
    static_data : np.ndarray
        Marker data from the Tcap (T-pose) trial, shape (n_frames, n_markers * 3).
    static_marker_names : list[str]
        Ordered marker names for the static trial (54 markers including medial
        calibration markers — only the lateral markers used here are required).
    side : str
        ``'L'`` for left or ``'R'`` for right.

    Returns
    -------
    float
        Mean knee angle in degrees during the static trial. Subtract from
        dynamic ``knee_flexion`` output to obtain corrected angles.
    """
    s = side.upper()
    tro = get_marker(static_data, static_marker_names, f"{s}TRO")
    knee = get_marker(static_data, static_marker_names, f"{s}KNEE")
    ankle = get_marker(static_data, static_marker_names, f"{s}ANK")
    angles = three_point_sagittal_angle(tro, knee, ankle)
    return float(np.nanmean(angles))


def knee_flexion(
    data: np.ndarray,
    marker_names: list[str],
    side: str = "L",
    static_ref: float = 0.0,
) -> np.ndarray:
    """Compute knee flexion angle in the sagittal plane.

    Uses ``three_point_sagittal_angle`` with greater trochanter (TRO),
    knee (KNEE), and ankle (ANK). Returns 0° at full extension, increasing
    with flexion. No wraparound artefacts.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered marker names (e.g. from ``io.load_marker_tsv``).
    side : str
        ``'L'`` for left or ``'R'`` for right.
    static_ref : float
        Static standing reference angle in degrees, computed from the Tcap
        trial via ``static_knee_reference()``. Subtracted from all frames to
        correct for marker placement offset. Default 0.0 (no correction).

    Returns
    -------
    np.ndarray
        Knee flexion angle in degrees, shape (n_frames,).
        0° = full extension (after static correction). Positive = flexion.
    """
    s = side.upper()
    tro = get_marker(data, marker_names, f"{s}TRO")
    knee = get_marker(data, marker_names, f"{s}KNEE")
    ankle = get_marker(data, marker_names, f"{s}ANK")
    return three_point_sagittal_angle(tro, knee, ankle) - static_ref


def static_hip_reference(
    static_data: np.ndarray,
    static_marker_names: list[str],
    side: str = "L",
) -> float:
    """Compute the mean hip angle from a static standing trial (Tcap).

    Parameters
    ----------
    static_data : np.ndarray
        Marker data from the Tcap trial, shape (n_frames, n_markers * 3).
    static_marker_names : list[str]
        Ordered marker names for the static trial.
    side : str
        ``'L'`` for left or ``'R'`` for right.

    Returns
    -------
    float
        Mean hip angle in degrees during the static trial. Subtract from
        dynamic ``hip_flexion`` output to obtain corrected angles.
    """
    angles = hip_flexion(static_data, static_marker_names, side=side, static_ref=0.0)
    return float(np.nanmean(angles))


def hip_flexion(
    data: np.ndarray,
    marker_names: list[str],
    side: str = "L",
    static_ref: float = 0.0,
) -> np.ndarray:
    """Compute hip flexion angle in the sagittal plane.

    Measures the thigh segment (TRO → KNEE) angle from the downward vertical
    using ``segment_angle_from_down``. Positive = thigh swings forward
    (flexion); negative = thigh swings backward (extension).

    The pelvis segment is used to correct for pelvis tilt: the reported angle
    is thigh-from-down minus pelvis-from-down, so pelvic anterior tilt does
    not inflate the hip flexion reading.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered marker names.
    side : str
        ``'L'`` for left or ``'R'`` for right. Controls which TRO and KNEE
        markers are used; pelvis markers are always bilateral.
    static_ref : float
        Static standing hip angle in degrees from ``static_hip_reference()``.
        Subtracted from all frames. Default 0.0 (no correction).

    Returns
    -------
    np.ndarray
        Hip flexion angle in degrees, shape (n_frames,).
        0° = standing neutral (after static correction). Positive = flexion.

    Notes
    -----
    ``segment_angle_from_down`` is used rather than ``three_point_sagittal_angle``
    because hip flexion is a global (segment-to-vertical) angle, not a joint
    angle between two adjacent segments.
    """
    s = side.upper()
    lasis = get_marker(data, marker_names, "LASIS")
    rasis = get_marker(data, marker_names, "RASIS")
    lpsis = get_marker(data, marker_names, "LPSIS")
    rpsis = get_marker(data, marker_names, "RPSIS")

    pelvis_post = (lpsis + rpsis) / 2.0
    pelvis_ant = (lasis + rasis) / 2.0

    tro = get_marker(data, marker_names, f"{s}TRO")
    knee = get_marker(data, marker_names, f"{s}KNEE")

    # Pelvis "downward" reference: use a vertical dropped from pelvis midpoint.
    # The pelvis mid → pelvis mid + [0, -1, 0] vector gives the downward
    # direction at the pelvis level; pelvis_from_down captures anterior tilt.
    pelvis_mid = (pelvis_ant + pelvis_post) / 2.0
    pelvis_below = pelvis_mid.copy()
    pelvis_below[:, 1] -= 1.0   # one unit directly below
    pelvis_from_down = segment_angle_from_down(pelvis_mid, pelvis_ant)

    thigh_from_down = segment_angle_from_down(tro, knee)
    return thigh_from_down - pelvis_from_down - static_ref


def static_ankle_reference(
    static_data: np.ndarray,
    static_marker_names: list[str],
    side: str = "L",
) -> float:
    """Compute the mean ankle angle from a static standing trial (Tcap).

    Replaces the hardcoded ``- 90.0`` in ``ankle_dorsiflexion`` with a
    subject-specific standing neutral, correcting for toe and shank marker
    placement offsets.

    Parameters
    ----------
    static_data : np.ndarray
        Marker data from the Tcap trial, shape (n_frames, n_markers * 3).
    static_marker_names : list[str]
        Ordered marker names for the static trial.
    side : str
        ``'L'`` for left or ``'R'`` for right.

    Returns
    -------
    float
        Mean ankle angle in degrees during the static trial. Subtract from
        dynamic ``ankle_dorsiflexion`` output to obtain corrected angles
        (0° = standing neutral, positive = dorsiflexion).
    """
    angles = ankle_dorsiflexion(static_data, static_marker_names, side=side, static_ref=0.0)
    return float(np.nanmean(angles))


def ankle_dorsiflexion(
    data: np.ndarray,
    marker_names: list[str],
    side: str = "L",
    static_ref: float = 0.0,
) -> np.ndarray:
    """Compute ankle dorsiflexion angle in the sagittal plane.

    Uses ``three_point_sagittal_angle`` with knee (KNEE), ankle (ANK), and
    toe (TOE). ``static_ref`` (from ``static_ankle_reference()``) is subtracted
    to normalize to the subject's own standing neutral, correcting for toe and
    shank marker placement offsets.

    Pass ``static_ref=0.0`` (default) to get the raw three-point angle — useful
    for computing the static reference itself via ``static_ankle_reference()``.

    Parameters
    ----------
    data : np.ndarray
        Marker data, shape (n_frames, n_markers * 3).
    marker_names : list[str]
        Ordered marker names.
    side : str
        ``'L'`` for left or ``'R'`` for right.
    static_ref : float
        Static standing ankle angle in degrees from ``static_ankle_reference()``.
        Subtracted from all frames. Default 0.0 (no correction).

    Returns
    -------
    np.ndarray
        Ankle dorsiflexion angle in degrees, shape (n_frames,).
        0° = standing neutral (after static correction).
        Positive = dorsiflexion, negative = plantarflexion.
    """
    s = side.upper()
    knee = get_marker(data, marker_names, f"{s}KNEE")
    ankle = get_marker(data, marker_names, f"{s}ANK")
    toe = get_marker(data, marker_names, f"{s}TOE")
    return three_point_sagittal_angle(knee, ankle, toe) - static_ref
