"""Layer 3 movement segmenter: combines velocity valley and keypose signals.

Runs both detection signals over per-movement search windows and returns 19
MovementBoundary objects with confidence labels.

    "high"   — velocity valley AND keypose match agree within tolerance_frames
    "medium" — exactly one signal found
    "low"    — neither signal found; falls back to the expected frame

Usage:
    from itf_analysis.segmentation.keypose_matcher import load_master_keyposes
    keyposes = load_master_keyposes("master_data/chon_ji/keyposes_angles.json")
    boundaries = segment_movements(all_frame_angles, keyposes, fps)
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from itf_analysis.segmentation.keypose_matcher import keypose_distance
from itf_analysis.segmentation.velocity_detector import (
    compute_motion_velocity,
    find_velocity_valleys,
    smooth_velocity,
)

Confidence = Literal["high", "medium", "low"]


@dataclass
class MovementBoundary:
    """Detected completion boundary for one numbered Chon-Ji movement.

    Attributes:
        movement_number: 1-based movement index (1–19).
        frame: Completion frame number (actual video frame index).
        timestamp_ms: Completion time in milliseconds (frame / fps * 1000).
        confidence: Signal quality — "high", "medium", or "low".
        matched_signals: Which signals contributed, e.g. ["velocity", "keypose"].
    """

    movement_number: int
    frame: int
    timestamp_ms: float
    confidence: Confidence
    matched_signals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_best_valley_in_window(
    valleys: List[int],
    frame_indices: List[int],
    smoothed: np.ndarray,
    lo: int,
    hi: int,
    angles_lookup: Optional[Dict[int, Dict[str, float]]] = None,
    kp_angles: Optional[Dict[str, float]] = None,
    keypose_threshold: float = 25.0,
) -> Optional[int]:
    """Return the frame of the best valley in [lo, hi].

    When angles_lookup and kp_angles are provided, selects the valley that
    minimises a combined score of normalised velocity depth and normalised
    keypose distance (equal weight). This improves accuracy for movements
    where velocity does not drop cleanly to zero.

    Falls back to deepest-valley selection when keypose data is absent.
    """
    in_window = [v for v in valleys if lo <= frame_indices[v] <= hi]
    if not in_window:
        return None

    if angles_lookup is not None and kp_angles:
        v_max = max(smoothed[v] for v in in_window) or 1.0

        def _score(v: int) -> float:
            fa = angles_lookup.get(frame_indices[v])
            norm_vel = smoothed[v] / v_max
            if fa:
                norm_kp = min(keypose_distance(fa, kp_angles) / keypose_threshold, 1.0)
                return 0.5 * norm_vel + 0.5 * norm_kp
            return norm_vel

        best = min(in_window, key=_score)
    else:
        best = min(in_window, key=lambda v: smoothed[v])

    return frame_indices[best]


def _find_best_keypose_in_window(
    all_frame_angles: List[Tuple[int, Dict]],
    kp_angles: Dict[str, float],
    lo: int,
    hi: int,
) -> Tuple[Optional[int], float]:
    """Return (frame_number, distance) of the best keypose match in [lo, hi].

    Args:
        all_frame_angles: Full sequence of (frame_index, angles_dict).
        kp_angles: Stored angle dict for this movement's completion pose.
        lo: Window lower bound (inclusive).
        hi: Window upper bound (inclusive).

    Returns:
        Tuple of (best_frame_number, best_distance). Returns (None, inf) if
        no frames fall within the window.
    """
    best_frame: Optional[int] = None
    best_dist = float("inf")
    for fi, fa in all_frame_angles:
        if lo <= fi <= hi:
            d = keypose_distance(fa, kp_angles)
            if d < best_dist:
                best_dist, best_frame = d, fi
    return best_frame, best_dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_movements(
    all_frame_angles: List[Tuple[int, Dict[str, float]]],
    master_keyposes: List[Dict],
    fps: float,
    tolerance_frames: int = 5,
    num_movements: int = 19,
    window_ratio: float = 0.45,
    keypose_threshold: float = 20.0,
    sigma_seconds: float = 0.1,
    min_gap_sec: float = 0.5,
    end_frame: Optional[int] = None,
) -> List[MovementBoundary]:
    """Segment an angle sequence into num_movements movement completions.

    For each of the num_movements movements, a windowed search region is
    computed around the linearly-expected completion frame. Within each window:
    - velocity signal: deepest velocity valley from velocity_detector
    - keypose signal: lowest-distance keypose match from keypose_matcher

    Confidence rules:
    - "high"   : both signals found AND within tolerance_frames of each other
    - "medium" : exactly one signal found (or both found but disagreeing)
    - "low"    : neither signal found → fallback to expected frame + warning

    Args:
        all_frame_angles: Pre-extracted, normalised angle sequence.
            List of (frame_index, angles_dict) ordered by frame_index.
        master_keyposes: Keyposes from load_master_keyposes(). Pass [] if not
            yet populated — runs velocity-only with all "medium" confidence.
        fps: Frames per second of the source video.
        tolerance_frames: Max frame gap between velocity and keypose signals
            for a "high" confidence classification.
        num_movements: Number of movements to detect (19 for Chon-Ji).
        window_ratio: Half-window as fraction of average movement length.
            0.45 → half_window ≈ 27 frames at 60 frames/movement.
        keypose_threshold: Max RMS angle distance (degrees) for a keypose
            match to count as a valid signal.
        sigma_seconds: Gaussian sigma for velocity smoothing.
        min_gap_sec: Minimum gap between velocity valleys in seconds.
        end_frame: Last frame of actual poomsae content. If provided, excludes
            post-poomsae tail (e.g., return-to-ready stance) from timing
            calculations. Defaults to the last frame in all_frame_angles.

    Returns:
        List of num_movements MovementBoundary objects in movement order.
        An empty list is returned if all_frame_angles has fewer than 2 entries.
    """
    if len(all_frame_angles) < 2:
        return []

    frame_indices = [fi for fi, _ in all_frame_angles]
    effective_end = (
        min(end_frame, frame_indices[-1]) if end_frame is not None else frame_indices[-1]
    )
    total_range = effective_end - frame_indices[0]
    avg_mov = total_range / num_movements
    half_window = max(1, int(avg_mov * window_ratio))

    # Use stored source_frames as expected positions when available — far more
    # accurate than linear spacing because real movements are not evenly timed.
    kp_source: Dict[int, int] = {
        kp["movement_index"]: kp["source_frame"]
        for kp in master_keyposes
        if "source_frame" in kp
    }
    if len(kp_source) == num_movements:
        expected = [kp_source[i + 1] for i in range(num_movements)]
    else:
        # Rolling fallback: each expected position is avg_mov after the previous.
        # More adaptive than fixed linear spacing when source_frames unavailable.
        expected = []
        rolling = frame_indices[0]
        for _ in range(num_movements):
            rolling = round(rolling + avg_mov)
            expected.append(rolling)

    # Build keypose lookup: movement_index → angle dict
    kp_map: Dict[int, Dict[str, float]] = {
        kp["movement_index"]: kp.get("angles", {}) for kp in master_keyposes
    }

    # Frame → angles lookup used for combined valley scoring
    angles_lookup: Dict[int, Dict[str, float]] = {fi: fa for fi, fa in all_frame_angles}

    # Velocity pipeline
    velocity = compute_motion_velocity(all_frame_angles, fps)
    smoothed = smooth_velocity(velocity, fps, sigma_seconds=sigma_seconds)
    valleys = find_velocity_valleys(smoothed, fps, min_gap_sec=min_gap_sec)

    boundaries: List[MovementBoundary] = []
    prev_frame = frame_indices[0] - 1

    for i in range(num_movements):
        mov = i + 1
        exp = expected[i]

        lo = max(frame_indices[0], prev_frame + 1, exp - half_window)
        hi = min(effective_end, exp + half_window)
        # Last movement: search entire tail so we never clip a late valley
        if hi >= effective_end:
            lo = max(frame_indices[0], prev_frame + 1)
        if lo > hi:
            lo = hi = exp

        # --- keypose signal ---
        kp_angles = kp_map.get(mov)

        # --- velocity signal (combined scoring when keypose angles available) ---
        v_frame = _find_best_valley_in_window(
            valleys, frame_indices, smoothed, lo, hi,
            angles_lookup=angles_lookup,
            kp_angles=kp_angles,
            keypose_threshold=keypose_threshold,
        )
        kp_frame: Optional[int] = None
        kp_found = False
        if kp_angles:
            kp_frame, kp_dist = _find_best_keypose_in_window(
                all_frame_angles, kp_angles, lo, hi
            )
            kp_found = kp_dist <= keypose_threshold

        # --- combine ---
        matched: List[str] = []
        if v_frame is not None:
            matched.append("velocity")
        if kp_found:
            matched.append("keypose")

        if "velocity" in matched and "keypose" in matched:
            if abs(v_frame - kp_frame) <= tolerance_frames:  # type: ignore[operator]
                confidence: Confidence = "high"
                frame = v_frame
            else:
                confidence = "medium"
                frame = v_frame  # velocity preferred when signals disagree
        elif "velocity" in matched:
            confidence = "medium"
            frame = v_frame
        elif "keypose" in matched:
            confidence = "medium"
            frame = kp_frame
        else:
            confidence = "low"
            frame = exp
            warnings.warn(
                f"Movement {mov}: no signal in [{lo}, {hi}] — using expected frame {exp}",
                stacklevel=2,
            )

        # Clamp to window and enforce strict monotonicity
        frame = max(lo, min(hi, frame))  # type: ignore[arg-type]
        if frame <= prev_frame:
            frame = prev_frame + 1

        boundaries.append(
            MovementBoundary(
                movement_number=mov,
                frame=frame,
                timestamp_ms=frame / fps * 1000.0,
                confidence=confidence,
                matched_signals=matched,
            )
        )
        prev_frame = frame

    return boundaries


# ---------------------------------------------------------------------------
# Student-video segmentation
# ---------------------------------------------------------------------------

def segment_student_video(
    student_frame_angles: List[Tuple[int, Dict[str, float]]],
    master_keyposes: List[Dict],
    fps: float,
    threshold: float = 25.0,
    sigma_seconds: float = 0.1,
    min_gap_sec: float = 0.5,
) -> List[MovementBoundary]:
    """Match student velocity valleys to master keyposes.

    Detects velocity valleys in the student sequence, then greedily assigns
    each valley to the closest unmatched master keypose (ordered: a valley
    cannot match an earlier movement than the previously matched one).  Valleys
    with no match below *threshold* are silently discarded.

    Confidence labelling:
        "high"   — RMS distance < threshold * 0.5
        "medium" — RMS distance < threshold

    Args:
        student_frame_angles: List of (frame_index, angles_dict) ordered by
            frame_index.  Produced by extract_joint_angles() on normalised frames.
        master_keyposes: List of keypose dicts from load_master_keyposes().
            Pass [] to get an empty result without raising.
        fps: Frames per second of the student video.
        threshold: Maximum RMS angle distance (degrees) for a valid keypose match.
        sigma_seconds: Gaussian sigma for velocity smoothing.
        min_gap_sec: Minimum gap between consecutive valley candidates.

    Returns:
        List of MovementBoundary sorted by frame number.  May have fewer than
        19 entries if movements were skipped or no valley was close enough.
    """
    if len(student_frame_angles) < 2 or not master_keyposes:
        return []

    frame_indices = [fi for fi, _ in student_frame_angles]
    angles_lookup: Dict[int, Dict[str, float]] = {
        fi: fa for fi, fa in student_frame_angles
    }

    velocity = compute_motion_velocity(student_frame_angles, fps)
    smoothed = smooth_velocity(velocity, fps, sigma_seconds=sigma_seconds)
    valleys = find_velocity_valleys(smoothed, fps, min_gap_sec=min_gap_sec)

    kp_list = sorted(master_keyposes, key=lambda k: k["movement_index"])
    matched_indices: set = set()
    next_kp_pos: int = 0  # ordering constraint index into kp_list

    boundaries: List[MovementBoundary] = []

    for v in valleys:
        fi = frame_indices[v]
        fa = angles_lookup.get(fi)
        if fa is None:
            continue

        best_kp: Optional[Dict] = None
        best_dist = float("inf")
        best_pos = -1

        for kp_pos in range(next_kp_pos, len(kp_list)):
            kp = kp_list[kp_pos]
            if kp["movement_index"] in matched_indices:
                continue
            kp_angles = kp.get("angles", {})
            if not kp_angles:
                continue
            d = keypose_distance(fa, kp_angles)
            if d < best_dist:
                best_dist = d
                best_kp = kp
                best_pos = kp_pos

        if best_kp is None or best_dist >= threshold:
            continue

        matched_indices.add(best_kp["movement_index"])
        next_kp_pos = best_pos + 1

        conf: Confidence = "high" if best_dist < threshold * 0.5 else "medium"
        boundaries.append(
            MovementBoundary(
                movement_number=best_kp["movement_index"],
                frame=fi,
                timestamp_ms=fi / fps * 1000.0,
                confidence=conf,
                matched_signals=["velocity", "keypose"],
            )
        )

    return sorted(boundaries, key=lambda b: b.frame)


# ---------------------------------------------------------------------------
# __main__: self-test on master video
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys
    from collections import Counter

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    poses_path = str(
        project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_poses.json"
    )
    keyposes_path = str(
        project_root / "itf_analysis" / "master_data" / "chon_ji" / "keyposes_angles.json"
    )

    from itf_analysis.normalization.normalizer import extract_joint_angles, normalize_pose
    from itf_analysis.segmentation.keypose_matcher import load_master_keyposes
    from itf_analysis.segmentation.segmentor import load_poses_from_json

    frames = load_poses_from_json(poses_path)
    fps = 1000.0 / (frames[1].timestamp_ms - frames[0].timestamp_ms)

    all_frame_angles = []
    for frame in frames:
        norm = normalize_pose(frame.landmarks)
        if norm is None:
            continue
        all_frame_angles.append((frame.frame_index, extract_joint_angles(norm)))

    keyposes = load_master_keyposes(keyposes_path)
    print(f"Keyposes loaded: {len(keyposes)} / 19")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boundaries = segment_movements(all_frame_angles, keyposes, fps)

    print(f"\n{'Mov':>4} {'frame':>6} {'time_ms':>9} {'conf':>8}  signals")
    print("-" * 55)
    for b in boundaries:
        print(
            f"{b.movement_number:>4} {b.frame:>6} {b.timestamp_ms:>9.0f} "
            f"{b.confidence:>8}  {b.matched_signals}"
        )

    dist = Counter(b.confidence for b in boundaries)
    print(f"\nConfidence: high={dist['high']}  medium={dist['medium']}  low={dist['low']}")
    if caught:
        print(f"\n{len(caught)} warning(s):")
        for w in caught:
            print(f"  {w.message}")
