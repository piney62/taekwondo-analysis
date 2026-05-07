"""Layer 5 difference measurement: extract per-joint angle issues from DTW alignment.

For each matched movement, iterates all DTW-aligned frame pairs and identifies
joints whose mean absolute deviation from the master reference exceeds the
configured threshold. Returns a ranked list of the worst offenders.

Typical usage:
    from itf_analysis.feedback.difference_analyzer import (
        load_thresholds, summarize_movement_issues
    )

    thresholds = load_thresholds("master_data/chon_ji/thresholds.json")
    summary = summarize_movement_issues(alignment_result, thresholds, top_n=3)
    for issue in summary.issues:
        print(f"{issue.joint}: {issue.diff:.1f}° ({issue.direction})")
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from itf_analysis.alignment.dtw_aligner import MovementAlignment

_CIRCULAR_KEYS = frozenset({"shoulder_line_angle", "hip_line_angle"})


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def load_thresholds(path: str) -> Dict:
    """Load thresholds JSON from *path*.

    Returns the raw dict with 'defaults' and 'movements' keys.
    Raises FileNotFoundError or json.JSONDecodeError on bad input.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_thresholds(thresholds: Dict, movement_number: int) -> Dict[str, float]:
    """Return per-joint threshold dict for *movement_number*.

    Movement-specific entries override the global default for that joint.
    """
    defaults: Dict[str, float] = thresholds.get("defaults", {})
    overrides: Dict[str, float] = (
        thresholds.get("movements", {}).get(str(movement_number), {})
    )
    resolved = {k: float(v) for k, v in defaults.items()}
    resolved.update({k: float(v) for k, v in overrides.items()})
    return resolved


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class JointIssue:
    """One joint that deviates beyond threshold from the master reference.

    Attributes:
        joint: Angle key, e.g. "left_elbow".
        master_angle: Mean master reference angle in degrees across matched frames.
        student_angle: Mean student angle in degrees across matched frames.
        diff: Mean absolute deviation |student − master| in degrees.
        direction: "too_large" when student angle > master; "too_small" otherwise.
        severity: diff / threshold — values > 1.0 exceed the threshold.
    """

    joint: str
    master_angle: float
    student_angle: float
    diff: float
    direction: str
    severity: float


@dataclass
class MovementIssueSummary:
    """Aggregated issues for one matched movement.

    Attributes:
        movement_number: 1-based movement index.
        issues: Top-N most severe joints, one entry per joint, sorted by severity.
        mean_rms: Mean RMS angle difference from MovementAlignment.
        max_rms: Peak RMS angle difference from MovementAlignment.
        frame_count: Total DTW-aligned frame pairs analyzed.
    """

    movement_number: int
    issues: List[JointIssue] = field(default_factory=list)
    mean_rms: float = 0.0
    max_rms: float = 0.0
    frame_count: int = 0


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def measure_frame_differences(
    master_angles: Dict[str, float],
    student_angles: Dict[str, float],
    joint_thresholds: Dict[str, float],
) -> List[JointIssue]:
    """Return issues for angles exceeding threshold in a single frame pair.

    Args:
        master_angles: Angle dict for the master reference frame.
        student_angles: Angle dict for the student frame aligned to it.
        joint_thresholds: Pre-resolved {joint: threshold_degrees} dict for
            this movement. Use _resolve_thresholds() to build it.

    Returns:
        List of JointIssue for joints whose absolute deviation exceeds the
        per-joint threshold. Empty when no deviation found.
    """
    issues: List[JointIssue] = []
    for joint in set(master_angles) & set(student_angles):
        m = master_angles[joint]
        s = student_angles[joint]
        if joint in _CIRCULAR_KEYS:
            delta = (s - m + 180.0) % 360.0 - 180.0
        else:
            delta = s - m
        diff = abs(delta)
        threshold = joint_thresholds.get(joint, 20.0)
        if diff > threshold:
            issues.append(
                JointIssue(
                    joint=joint,
                    master_angle=round(m, 1),
                    student_angle=round(s, 1),
                    diff=round(diff, 1),
                    direction="too_large" if delta > 0 else "too_small",
                    severity=round(diff / threshold, 2),
                )
            )
    return issues


def summarize_movement_issues(
    alignment: MovementAlignment,
    thresholds: Dict,
    top_n: int = 3,
) -> MovementIssueSummary:
    """Aggregate per-frame deviations into the top-N most severe joint issues.

    For each joint, collects signed deltas (student − master) across all DTW-
    aligned frame pairs, then checks whether the *mean* absolute deviation
    exceeds the threshold. Joints appearing across many frames are summarised
    by their mean angles and mean diff rather than by a single worst frame.
    This suppresses one-frame noise and surfaces persistent technique errors.

    Args:
        alignment: MovementAlignment from align_movement().
        thresholds: Full thresholds dict from load_thresholds().
        top_n: Maximum number of joint issues to return (default 3).

    Returns:
        MovementIssueSummary with up to top_n JointIssue entries ranked by
        severity descending. Returns an empty summary when alignment has no
        frame pairs.
    """
    if not alignment.frame_pairs:
        return MovementIssueSummary(
            movement_number=alignment.movement_number,
            mean_rms=alignment.mean_rms,
            max_rms=alignment.max_rms,
            frame_count=0,
        )

    mov = alignment.movement_number
    joint_thresholds = _resolve_thresholds(thresholds, mov)

    # Accumulate (master, student, signed_delta) per joint across all frame pairs.
    accum: Dict[str, List] = defaultdict(list)
    for fp in alignment.frame_pairs:
        common = set(fp.master_angles) & set(fp.student_angles)
        for joint in common:
            m = fp.master_angles[joint]
            s = fp.student_angles[joint]
            if joint in _CIRCULAR_KEYS:
                delta = (s - m + 180.0) % 360.0 - 180.0
            else:
                delta = s - m
            accum[joint].append((m, s, delta))

    issues: List[JointIssue] = []
    for joint, obs in accum.items():
        threshold = joint_thresholds.get(joint, 20.0)
        mean_master = sum(o[0] for o in obs) / len(obs)
        mean_student = sum(o[1] for o in obs) / len(obs)
        mean_delta = sum(o[2] for o in obs) / len(obs)
        mean_diff = sum(abs(o[2]) for o in obs) / len(obs)
        if mean_diff > threshold:
            issues.append(
                JointIssue(
                    joint=joint,
                    master_angle=round(mean_master, 1),
                    student_angle=round(mean_student, 1),
                    diff=round(mean_diff, 1),
                    direction="too_large" if mean_delta > 0 else "too_small",
                    severity=round(mean_diff / threshold, 2),
                )
            )

    issues.sort(key=lambda x: x.severity, reverse=True)
    return MovementIssueSummary(
        movement_number=mov,
        issues=issues[:top_n],
        mean_rms=alignment.mean_rms,
        max_rms=alignment.max_rms,
        frame_count=len(alignment.frame_pairs),
    )


# ---------------------------------------------------------------------------
# __main__: self-test — run on master-vs-master (expect no issues near zero)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
    thresholds_path = str(
        project_root / "itf_analysis" / "master_data" / "chon_ji" / "thresholds.json"
    )

    from itf_analysis.normalization.normalizer import extract_joint_angles, normalize_pose
    from itf_analysis.segmentation.keypose_matcher import load_master_keyposes
    from itf_analysis.segmentation.segmentor import load_poses_from_json
    from itf_analysis.segmentation.segmenter import segment_movements
    from itf_analysis.alignment.dtw_aligner import (
        build_movement_segments,
        align_all_movements,
    )

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
    results = align_all_movements(segments, segments)

    thresholds = load_thresholds(thresholds_path)

    print(f"\n{'Mov':>4}  {'mean_rms':>8}  {'issues'}  (self-match — expect 0 issues)")
    print("-" * 55)
    for mov in sorted(results):
        summary = summarize_movement_issues(results[mov], thresholds, top_n=3)
        issue_str = ", ".join(f"{i.joint} {i.diff:.1f}°" for i in summary.issues) or "—"
        print(f"{mov:>4}  {summary.mean_rms:>8.2f}  {issue_str}")
