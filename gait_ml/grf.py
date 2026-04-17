"""
grf.py — Ground reaction force feature extraction.

All peak and impulse features are normalized to body weight (BW) by default.
Body weight in Newtons must be provided (pull from demographics via io.py).
"""

import numpy as np


def extract_grf_features(
    vertical_grf_stance: np.ndarray,
    sample_rate_hz: float,
    body_weight_n: float,
) -> dict[str, float]:
    """Extract standard biomechanical features from vertical GRF during stance.

    Parameters
    ----------
    vertical_grf_stance : np.ndarray
        Vertical GRF time series for a single stance phase, in Newtons.
        Shape (n_frames,). Must already be cropped to heel strike ? toe off.
    sample_rate_hz : float
        Force plate sampling rate in Hz.
    body_weight_n : float
        Subject body weight in Newtons for normalization.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys:
        - peak_vgrf_bw: peak vertical GRF normalized to body weight
        - loading_rate_bw_s: average loading rate (BW/s) over first 20% of stance
        - impulse_bw_s: GRF impulse normalized to body weight (BW·s)
        - contact_time_s: stance duration in seconds
        - impact_peak_bw: first local maximum (impact transient), if present
    """
    normalized = vertical_grf_stance / body_weight_n
    n_frames = len(normalized)
    dt = 1.0 / sample_rate_hz

    # Loading phase = first 20% of stance
    loading_end = max(1, int(n_frames * 0.20))
    loading_rate = (normalized[loading_end] - normalized[0]) / (loading_end * dt)

    features: dict[str, float] = {
        "peak_vgrf_bw": float(np.max(normalized)),
        "loading_rate_bw_s": float(loading_rate),
        "impulse_bw_s": float(np.trapz(normalized, dx=dt)),
        "contact_time_s": float(n_frames * dt),
    }

    # Impact peak: first local max in first 30% of stance (not always present in walking)
    loading_window = normalized[: int(n_frames * 0.30)]
    if len(loading_window) > 2:
        local_maxima = np.where(
            (np.diff(np.sign(np.diff(loading_window))) < 0)
        )[0] + 1
        if len(local_maxima) > 0:
            features["impact_peak_bw"] = float(loading_window[local_maxima[0]])
        else:
            features["impact_peak_bw"] = float("nan")
    else:
        features["impact_peak_bw"] = float("nan")

    return features


def extract_cop_features(
    cop_x: np.ndarray,
    cop_y: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float]:
    """Extract center of pressure features from standing CoP data.

    Parameters
    ----------
    cop_x : np.ndarray
        Mediolateral CoP displacement in mm, shape (n_frames,).
    cop_y : np.ndarray
        Anteroposterior CoP displacement in mm, shape (n_frames,).
    sample_rate_hz : float
        Force plate sampling rate in Hz.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys:
        - cop_range_ml_mm: mediolateral CoP range
        - cop_range_ap_mm: anteroposterior CoP range
        - cop_sway_velocity_mm_s: mean CoP sway velocity
        - cop_area_mm2: 95% confidence ellipse area (approximate)
    """
    dt = 1.0 / sample_rate_hz
    cop_displacement = np.sqrt(np.diff(cop_x) ** 2 + np.diff(cop_y) ** 2)
    mean_velocity = float(np.mean(cop_displacement) / dt)

    # Approximate sway area via covariance ellipse
    cov = np.cov(cop_x, cop_y)
    eigenvalues = np.linalg.eigvalsh(cov)
    # 95% CI ellipse area: pi * F_95 * sqrt(lambda1 * lambda2)
    # F_95 for 2 df ≈ 5.991
    area = np.pi * 5.991 * np.sqrt(np.maximum(eigenvalues[0], 0) * np.maximum(eigenvalues[1], 0))

    return {
        "cop_range_ml_mm": float(np.max(cop_x) - np.min(cop_x)),
        "cop_range_ap_mm": float(np.max(cop_y) - np.min(cop_y)),
        "cop_sway_velocity_mm_s": mean_velocity,
        "cop_area_mm2": float(area),
    }
