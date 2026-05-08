"""Tests for segment_student_video().

Synthetic test data uses sinusoidal angle interpolation so that velocity
valleys naturally appear at movement completion frames.  Each movement's
angles differ by 50° per key from adjacent movements, so keypose distances
are either ~0 (exact match) or ≥50 (wrong movement), well clear of the
default threshold=25.

A neutral 'return block' (movement index 0) is appended to the sequence so
that the last real movement always produces a velocity valley before the
end of the array (which argrelmin cannot detect).
"""

import math
from typing import Dict, List, Tuple

import pytest

from itf_analysis.segmentation.keypose_matcher import ANGLE_KEYS
from itf_analysis.segmentation.segmenter import segment_student_video


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_angles(movement_idx: int) -> Dict[str, float]:
    """Return angles uniquely identifying movement_idx.

    Adjacent movements are 50° apart on every key, so cross-movement RMS
    distance = 50° >> default threshold of 25°.
    """
    return {k: float(movement_idx * 50) for k in ANGLE_KEYS}


def _make_synthetic_student(
    present: List[int],
    fps: float = 30.0,
    fpk: int = 60,
) -> Tuple[List[Tuple[int, Dict[str, float]]], List[Dict]]:
    """Build synthetic (all_frame_angles, master_keyposes).

    Args:
        present: Ordered list of movement indices (1–19 subset) to include.
        fps: Frames per second (used only as metadata; shapes velocity).
        fpk: Frames per movement block.  Must be > 30 for argrelmin(order=15)
             to detect inter-block valleys reliably.

    Returns:
        all_frame_angles: List of (frame_index, angles_dict).
        master_keyposes: All 19 master keypose dicts.

    Notes:
        Within each block the angles follow a sin²(πt/2) interpolation
        (0 at block start → 1 at block end), giving a smooth bell-shaped
        velocity profile with minima at block boundaries.

        A neutral return block (movement 0, angles all 0°) is appended so
        movement 19 gets a velocity valley after it.
    """
    master_keyposes = [
        {"movement_index": m, "angles": _make_angles(m)} for m in range(1, 20)
    ]

    sequence = present + [0]  # 0 = neutral return-to-ready
    all_frame_angles: List[Tuple[int, Dict[str, float]]] = []
    frame_idx = 0

    for i, m in enumerate(sequence):
        prev_m = sequence[i - 1] if i > 0 else 0
        start_a = _make_angles(prev_m)
        end_a = _make_angles(m)
        for f in range(fpk):
            t = f / (fpk - 1)
            pos = math.sin(math.pi * t / 2) ** 2
            angles = {k: start_a[k] + pos * (end_a[k] - start_a[k]) for k in ANGLE_KEYS}
            all_frame_angles.append((frame_idx, angles))
            frame_idx += 1

    return all_frame_angles, master_keyposes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSegmentStudentVideo:

    def test_perfect_sequence_all_19_matched(self):
        """All 19 movements present → all 19 detected with high confidence."""
        all_fa, keyposes = _make_synthetic_student(list(range(1, 20)))
        boundaries = segment_student_video(all_fa, keyposes, fps=30.0)

        found = {b.movement_number for b in boundaries}
        assert found == set(range(1, 20)), (
            f"Expected all 19 movements; got {sorted(found)}"
        )
        for b in boundaries:
            assert b.confidence == "high", (
                f"Movement {b.movement_number}: expected high, got {b.confidence}"
            )
        for b in boundaries:
            assert "keypose" in b.matched_signals
            assert "velocity" in b.matched_signals

    def test_movement_5_missing(self):
        """Student skips movement 5 → 18 boundaries returned, movement 5 absent."""
        present = list(range(1, 5)) + list(range(6, 20))  # all except 5
        all_fa, keyposes = _make_synthetic_student(present)
        boundaries = segment_student_video(all_fa, keyposes, fps=30.0)

        found = {b.movement_number for b in boundaries}
        assert 5 not in found, "Movement 5 should be absent"
        assert found == set(range(1, 20)) - {5}, (
            f"Expected movements 1-4 and 6-19; got {sorted(found)}"
        )

    def test_empty_frame_angles_returns_empty(self):
        _, keyposes = _make_synthetic_student([1, 2])
        result = segment_student_video([], keyposes, fps=30.0)
        assert result == []

    def test_single_frame_returns_empty(self):
        _, keyposes = _make_synthetic_student([1, 2])
        result = segment_student_video([(0, _make_angles(1))], keyposes, fps=30.0)
        assert result == []

    def test_empty_keyposes_returns_empty(self):
        all_fa, _ = _make_synthetic_student([1, 2, 3])
        result = segment_student_video(all_fa, [], fps=30.0)
        assert result == []

    def test_output_sorted_by_frame(self):
        """Return list must be ordered by frame number."""
        all_fa, keyposes = _make_synthetic_student(list(range(1, 20)))
        boundaries = segment_student_video(all_fa, keyposes, fps=30.0)
        frames = [b.frame for b in boundaries]
        assert frames == sorted(frames)

    def test_timestamp_consistent_with_frame(self):
        """timestamp_ms must equal frame / fps * 1000."""
        fps = 30.0
        all_fa, keyposes = _make_synthetic_student(list(range(1, 10)))
        boundaries = segment_student_video(all_fa, keyposes, fps=fps)
        for b in boundaries:
            expected_ms = b.frame / fps * 1000.0
            assert b.timestamp_ms == pytest.approx(expected_ms)
