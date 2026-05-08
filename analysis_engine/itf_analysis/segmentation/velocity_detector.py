"""Layer 3 velocity signal: angle-based motion velocity and valley detection.

Computes frame-to-frame angular velocity from joint angle time series,
applies Gaussian smoothing, and finds local minima (valleys) that correspond
to movement completion moments where the body is momentarily still.

Typical usage:
    velocity = compute_motion_velocity(all_frame_angles, fps)
    smoothed  = smooth_velocity(velocity, fps, sigma_seconds=0.1)
    valleys   = find_velocity_valleys(smoothed, fps, min_gap_sec=0.5)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin
from typing import Dict, List, Tuple


def _circular_abs_diff(a: float, b: float, key: str) -> float:
    """Absolute angle difference with circular arithmetic for line angles."""
    diff = b - a
    if "line_angle" in key:
        diff = (diff + 180.0) % 360.0 - 180.0
    return abs(diff)


def compute_motion_velocity(
    all_frame_angles: List[Tuple[int, Dict[str, float]]],
    fps: float,
) -> np.ndarray:
    """Compute frame-to-frame angular velocity from joint angle time series.

    Velocity at position i = sum of absolute angle differences across all
    common angle keys between consecutive frames i and i+1. Line angles
    (shoulder_line_angle, hip_line_angle) use circular arithmetic to avoid
    180-degree wrap-around artefacts.

    Args:
        all_frame_angles: List of (frame_index, angles_dict), ordered by frame_index.
        fps: Frames per second. Reserved for future per-second normalization;
             not used in the raw computation.

    Returns:
        np.ndarray of shape (N-1,) where N = len(all_frame_angles).
        Element i represents motion between frame i and frame i+1.
        Returns empty array if fewer than 2 frames provided.
    """
    if len(all_frame_angles) < 2:
        return np.array([], dtype=float)

    velocity: List[float] = []
    for i in range(len(all_frame_angles) - 1):
        _, a1 = all_frame_angles[i]
        _, a2 = all_frame_angles[i + 1]
        common = set(a1.keys()) & set(a2.keys())
        total = sum(_circular_abs_diff(a1[k], a2[k], k) for k in common)
        velocity.append(total)

    return np.array(velocity, dtype=float)


def smooth_velocity(
    velocity: np.ndarray,
    fps: float,
    sigma_seconds: float = 0.1,
) -> np.ndarray:
    """Apply Gaussian smoothing to the raw angular velocity signal.

    Gaussian smoothing suppresses per-frame noise while preserving the
    broad velocity valleys that mark movement completions.

    Args:
        velocity: Raw velocity array from compute_motion_velocity.
        fps: Frames per second, used to convert sigma_seconds to frames.
        sigma_seconds: Gaussian standard deviation in seconds.
            0.1 s at 30 fps → sigma ≈ 3 frames → removes single-frame jitter
            while keeping ~200 ms valleys visible.

    Returns:
        Smoothed velocity array of the same length.
    """
    if velocity.size == 0:
        return velocity.copy()
    sigma_frames = sigma_seconds * fps
    if sigma_frames < 0.1:
        return velocity.copy()
    return gaussian_filter1d(velocity.astype(float), sigma=sigma_frames)


def find_velocity_valleys(
    smoothed_velocity: np.ndarray,
    fps: float,
    min_gap_sec: float = 0.5,
) -> List[int]:
    """Find local minima (valleys) in the smoothed velocity signal.

    Uses scipy.signal.argrelmin: a valley at position i must be strictly
    less than all neighbours within `order` positions on each side.
    Valleys correspond to moments where body motion is locally minimal —
    the natural pause at the completion of each Taekwondo technique.

    Args:
        smoothed_velocity: Smoothed velocity array from smooth_velocity.
        fps: Frames per second.
        min_gap_sec: Minimum gap between consecutive valleys in seconds.
            Converted to `order` for argrelmin: at 30 fps with 0.5 s,
            order = 15 frames → no two valleys within 15 frames of each other.

    Returns:
        Sorted list of valley positions (indices into smoothed_velocity).
        Use frame_indices[v] to convert to actual video frame numbers.
    """
    if smoothed_velocity.size < 3:
        return []
    order = max(1, int(min_gap_sec * fps))
    (valley_indices,) = argrelmin(smoothed_velocity, order=order)
    return sorted(valley_indices.tolist())


# ---------------------------------------------------------------------------
# __main__: run velocity detection on the master video
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    poses_path = str(
        project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_poses.json"
    )

    from itf_analysis.segmentation.segmentor import load_poses_from_json
    from itf_analysis.normalization.normalizer import normalize_pose, extract_joint_angles

    frames = load_poses_from_json(poses_path)
    print(f"Frames loaded: {len(frames)}")

    fps = 1000.0 / (frames[1].timestamp_ms - frames[0].timestamp_ms) if len(frames) > 1 else 30.0
    print(f"Estimated fps: {fps:.1f}")

    all_frame_angles = []
    for frame in frames:
        norm = normalize_pose(frame.landmarks)
        if norm is None:
            continue
        all_frame_angles.append((frame.frame_index, extract_joint_angles(norm)))

    print(f"Angles extracted: {len(all_frame_angles)} frames")

    velocity = compute_motion_velocity(all_frame_angles, fps)
    smoothed = smooth_velocity(velocity, fps)
    valleys = find_velocity_valleys(smoothed, fps)

    print(f"Velocity array length: {len(velocity)}")
    print(f"Valleys detected: {len(valleys)}  (need >= 19)")
    print(f"Valley positions: {valleys[:25]}")
