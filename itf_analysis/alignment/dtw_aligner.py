"""Layer 4 DTW: temporal alignment of matched movement segments.

For each matched master-student movement pair, runs fastdtw over the per-frame
angle sequences to produce a warping path. The path maps each master frame to
the most temporally similar student frame.

Typical usage:
    from itf_analysis.alignment.dtw_aligner import (
        build_movement_segments, align_movement, align_all_movements
    )

    master_segs = build_movement_segments(master_angles, master_boundaries)
    student_segs = build_movement_segments(student_angles, student_boundaries)
    results = align_all_movements(master_segs, student_segs)
    # results[1] → MovementAlignment for movement 1
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastdtw import fastdtw

from itf_analysis.segmentation.segmenter import MovementBoundary

# Angles that wrap at ±180° and require circular difference arithmetic.
_CIRCULAR_KEYS = frozenset({"shoulder_line_angle", "hip_line_angle"})


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _circular_diff(a: float, b: float) -> float:
    """Smallest angular difference in degrees for a circular angle in [-180, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def rms_angle_distance(
    a: Dict[str, float],
    b: Dict[str, float],
) -> float:
    """RMS angle distance between two angle dicts.

    Only keys present in both dicts contribute. Uses circular arithmetic for
    shoulder_line_angle and hip_line_angle. Returns 0.0 when there are no
    common keys.
    """
    common = set(a) & set(b)
    if not common:
        return 0.0
    sq = 0.0
    for k in common:
        d = _circular_diff(a[k], b[k]) if k in _CIRCULAR_KEYS else abs(a[k] - b[k])
        sq += d * d
    return math.sqrt(sq / len(common))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FramePair:
    """One matched (master, student) frame pair from a DTW warping path.

    Attributes:
        master_frame: Actual frame index in the master video.
        student_frame: Actual frame index in the student video.
        angle_diffs: Per-angle absolute difference in degrees.
        rms_diff: RMS across all angle_diffs (0.0 when no common keys).
    """

    master_frame: int
    student_frame: int
    angle_diffs: Dict[str, float] = field(default_factory=dict)
    rms_diff: float = 0.0


@dataclass
class MovementAlignment:
    """DTW alignment result for one movement.

    Attributes:
        movement_number: 1-based movement index (1–19).
        path: Raw warping path as (master_seg_idx, student_seg_idx) pairs.
        frame_pairs: FramePair for each step in the path.
        mean_rms: Mean RMS angle difference across all pairs.
        max_rms: Peak RMS angle difference across all pairs.
    """

    movement_number: int
    path: List[Tuple[int, int]] = field(default_factory=list)
    frame_pairs: List[FramePair] = field(default_factory=list)
    mean_rms: float = 0.0
    max_rms: float = 0.0


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------

def build_movement_segments(
    all_frame_angles: List[Tuple[int, Dict[str, float]]],
    boundaries: List[MovementBoundary],
) -> Dict[int, List[Tuple[int, Dict[str, float]]]]:
    """Split a full frame-angle sequence into per-movement sub-sequences.

    Movement N spans from the frame immediately after movement N-1's boundary
    frame (or the very first frame for movement 1) up to and including
    movement N's boundary frame.

    Args:
        all_frame_angles: Full (frame_index, angles_dict) sequence ordered by
            frame_index. Produced by running extract_joint_angles() on every
            normalised frame.
        boundaries: MovementBoundary list from segment_movements() or
            segment_student_video(). May be in any order — sorted internally.

    Returns:
        Dict mapping movement_number to its frame-angle sub-sequence.
        Empty segments (no frames fall in the range) are included as [].
    """
    if not boundaries or not all_frame_angles:
        return {}

    angle_dict: Dict[int, Dict[str, float]] = {fi: fa for fi, fa in all_frame_angles}
    frame_list = sorted(angle_dict)
    sorted_bounds = sorted(boundaries, key=lambda b: b.movement_number)

    segments: Dict[int, List[Tuple[int, Dict[str, float]]]] = {}
    prev_end = frame_list[0] - 1

    for b in sorted_bounds:
        seg = [
            (fi, angle_dict[fi])
            for fi in frame_list
            if prev_end < fi <= b.frame
        ]
        segments[b.movement_number] = seg
        prev_end = b.frame

    return segments


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _build_angle_matrix(
    segment: List[Tuple[int, Dict[str, float]]],
    key_order: List[str],
) -> np.ndarray:
    """Convert a list of angle dicts to a float matrix (n_frames × n_keys).

    Missing keys are filled by interpolating from neighbouring frames; if an
    entire column is absent, it is set to 0.
    """
    n, k = len(segment), len(key_order)
    mat = np.full((n, k), np.nan, dtype=float)
    for i, (_, fa) in enumerate(segment):
        for j, key in enumerate(key_order):
            if key in fa:
                mat[i, j] = fa[key]

    for j in range(k):
        col = mat[:, j]
        valid_idx = np.where(~np.isnan(col))[0]
        if len(valid_idx) == 0:
            mat[:, j] = 0.0
        elif len(valid_idx) < n:
            mat[:, j] = np.interp(np.arange(n), valid_idx, col[valid_idx])

    return mat


def align_movement(
    master_segment: List[Tuple[int, Dict[str, float]]],
    student_segment: List[Tuple[int, Dict[str, float]]],
    movement_number: int = 0,
) -> MovementAlignment:
    """Align one master movement segment to the corresponding student segment.

    Converts angle dicts to fixed-order numpy arrays then runs fastdtw with an
    RMS distance function that applies circular arithmetic to line angles.  The
    warping path is translated from segment-local indices to actual video frame
    indices in the returned FramePair list.

    Args:
        master_segment: Master frames for this movement as (frame_idx, angles).
        student_segment: Student frames for this movement as (frame_idx, angles).
        movement_number: 1-based movement index (for labelling only).

    Returns:
        MovementAlignment. Returns an empty alignment when either segment is
        empty.
    """
    if not master_segment or not student_segment:
        return MovementAlignment(movement_number=movement_number)

    m_frames = [fi for fi, _ in master_segment]
    s_frames = [fi for fi, _ in student_segment]
    m_angle_dicts = [fa for _, fa in master_segment]
    s_angle_dicts = [fa for _, fa in student_segment]

    # Fixed key order — union of all keys so neither segment loses information.
    all_keys: List[str] = sorted(
        set().union(*m_angle_dicts, *s_angle_dicts)
    )
    circular_mask = np.array([k in _CIRCULAR_KEYS for k in all_keys])

    m_mat = _build_angle_matrix(master_segment, all_keys)
    s_mat = _build_angle_matrix(student_segment, all_keys)

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        diff[circular_mask] = ((diff[circular_mask] + 180.0) % 360.0) - 180.0
        return float(np.sqrt(np.mean(diff ** 2)))

    _, raw_path = fastdtw(m_mat, s_mat, dist=_dist)

    frame_pairs: List[FramePair] = []
    for mi, si in raw_path:
        ma, sa = m_angle_dicts[mi], s_angle_dicts[si]
        common = set(ma) & set(sa)
        diffs: Dict[str, float] = {}
        for k in common:
            diffs[k] = (
                _circular_diff(ma[k], sa[k]) if k in _CIRCULAR_KEYS else abs(ma[k] - sa[k])
            )
        rms = math.sqrt(sum(d * d for d in diffs.values()) / len(diffs)) if diffs else 0.0
        frame_pairs.append(
            FramePair(
                master_frame=m_frames[mi],
                student_frame=s_frames[si],
                angle_diffs=diffs,
                rms_diff=rms,
            )
        )

    rms_values = [fp.rms_diff for fp in frame_pairs]
    return MovementAlignment(
        movement_number=movement_number,
        path=list(raw_path),
        frame_pairs=frame_pairs,
        mean_rms=sum(rms_values) / len(rms_values),
        max_rms=max(rms_values),
    )


def align_all_movements(
    master_segments: Dict[int, List[Tuple[int, Dict[str, float]]]],
    student_segments: Dict[int, List[Tuple[int, Dict[str, float]]]],
) -> Dict[int, MovementAlignment]:
    """Align all movements present in both master and student segment dicts.

    Movements absent from student_segments (skipped movements) are silently
    omitted from the result. The caller should check which movement numbers
    are missing.

    Args:
        master_segments: Output of build_movement_segments() for the master.
        student_segments: Output of build_movement_segments() for the student.

    Returns:
        Dict mapping movement_number to MovementAlignment for each movement
        present in both inputs.
    """
    common = sorted(set(master_segments) & set(student_segments))
    return {
        mov: align_movement(master_segments[mov], student_segments[mov], mov)
        for mov in common
    }


# ---------------------------------------------------------------------------
# __main__: self-test on master video (self-match → near-zero RMS expected)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib
    import sys
    import warnings

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    poses_path = str(
        project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_poses.json"
    )
    keyposes_path = str(
        project_root / "itf_analysis" / "master_data" / "chon_ji" / "keyposes_angles.json"
    )

    from itf_analysis.normalization.normalizer import extract_joint_angles, normalize_pose
    from itf_analysis.pose_extraction.extractor import Landmark
    from itf_analysis.segmentation.keypose_matcher import load_master_keyposes
    from itf_analysis.segmentation.segmentor import load_poses_from_json
    from itf_analysis.segmentation.segmenter import segment_movements

    frames = load_poses_from_json(poses_path)
    fps = 1000.0 / (frames[1].timestamp_ms - frames[0].timestamp_ms)

    all_frame_angles = []
    for frame in frames:
        norm = normalize_pose(frame.landmarks)
        if norm is None:
            continue
        all_frame_angles.append((frame.frame_index, extract_joint_angles(norm)))

    keyposes = load_master_keyposes(keyposes_path)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        boundaries = segment_movements(all_frame_angles, keyposes, fps)

    segments = build_movement_segments(all_frame_angles, boundaries)

    # Self-match: align master against itself — warping path should be diagonal,
    # RMS differences should be near zero.
    results = align_all_movements(segments, segments)

    print(f"\n{'Mov':>4} {'frames':>7} {'mean_rms':>10} {'max_rms':>10}")
    print("-" * 36)
    for mov in sorted(results):
        r = results[mov]
        n = len(segments.get(mov, []))
        print(f"{mov:>4} {n:>7} {r.mean_rms:>10.2f} {r.max_rms:>10.2f}")

    all_means = [r.mean_rms for r in results.values()]
    print(f"\nOverall mean RMS: {sum(all_means)/len(all_means):.2f}° (expect ~0 for self-match)")
