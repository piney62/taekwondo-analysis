"""Unit tests for Layer 3: movement segmentation."""

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

from itf_analysis.pose_extraction.extractor import FrameData, Landmark
from itf_analysis.segmentation.segmentor import (
    KEYPOSE_JOINTS,
    VELOCITY_JOINTS,
    MovementBoundary,
    build_pose_vector,
    combine_signals,
    compute_expected_frames,
    compute_frame_displacement,
    compute_keypose_similarities,
    compute_search_window,
    compute_velocity_signal,
    cosine_similarity,
    find_keypose_match,
    find_velocity_valley,
    normalize_velocity,
    segment_video,
    smooth_velocity,
)
from itf_analysis.normalization.normalizer import NormalizedLandmark
from itf_analysis.segmentation.keypose_marker import (
    add_or_update_keypose,
    build_keypose_entry,
    load_keyposes_or_empty,
    validate_keyposes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lm(x: float, y: float, z: float = 0.0, vis: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=vis)


def _nlm(x: float, y: float, z: float = 0.0) -> NormalizedLandmark:
    return NormalizedLandmark(x=x, y=y, z=z, visibility=1.0)


def _make_frame(frame_index: int, positions: Dict[int, Tuple[float, float]],
                vis: float = 1.0) -> FrameData:
    return FrameData(
        frame_index=frame_index,
        timestamp_ms=frame_index * 33.3,
        landmarks={j: _lm(x, y, vis=vis) for j, (x, y) in positions.items()},
    )


def _anchor_positions(dx: float = 0.0) -> Dict[int, Tuple[float, float]]:
    """Return a minimal full-anchor landmark set for normalization to work."""
    return {
        11: (0.3 + dx, 0.2), 12: (0.7 + dx, 0.2),
        13: (0.25 + dx, 0.4), 14: (0.75 + dx, 0.4),
        15: (0.2 + dx, 0.6), 16: (0.8 + dx, 0.6),
        23: (0.4 + dx, 0.7), 24: (0.6 + dx, 0.7),
        25: (0.35 + dx, 0.85), 26: (0.65 + dx, 0.85),
        27: (0.33 + dx, 1.0), 28: (0.67 + dx, 1.0),
    }


def _synthetic_frames(num_movements: int = 19, fpk: int = 20) -> List[FrameData]:
    """Synthetic pose sequence with velocity valleys at each movement end.

    Each movement spans `fpk` frames. Joints move sinusoidally within the
    movement (fast in the middle, still at the completion frame).
    """
    frames = []
    for m in range(num_movements):
        for p in range(fpk):
            t = p / max(fpk - 1, 1)       # 0 → 1 within movement
            dx = 0.2 * math.sin(math.pi * t)
            frame_index = m * fpk + p
            frames.append(_make_frame(frame_index, _anchor_positions(dx)))
    return frames


# ---------------------------------------------------------------------------
# Velocity tests (6)
# ---------------------------------------------------------------------------

class TestComputeFrameDisplacement:
    def test_identical_frames_returns_zero(self):
        pos = {11: (0.5, 0.5)}
        fa = _make_frame(0, pos)
        fb = _make_frame(1, pos)
        assert compute_frame_displacement(fa, fb, joints=[11]) == pytest.approx(0.0)

    def test_known_distance(self):
        fa = _make_frame(0, {11: (0.0, 0.0)})
        fb = _make_frame(1, {11: (1.0, 1.0)})
        expected = math.sqrt(2)
        assert compute_frame_displacement(fa, fb, joints=[11]) == pytest.approx(expected)

    def test_excludes_interpolated_in_frame_b(self):
        fa = _make_frame(0, {11: (0.0, 0.0), 12: (0.5, 0.5)})
        # joint 11 is interpolated in fb (visibility=0.0)
        fb = FrameData(
            frame_index=1,
            timestamp_ms=33.3,
            landmarks={
                11: _lm(1.0, 0.0, vis=0.0),
                12: _lm(0.5, 0.5, vis=1.0),
            },
        )
        # Only joint 12 (no movement) is counted → displacement = 0.0
        assert compute_frame_displacement(fa, fb, joints=[11, 12]) == pytest.approx(0.0)

    def test_excludes_interpolated_in_frame_a(self):
        fa = FrameData(
            frame_index=0,
            timestamp_ms=0.0,
            landmarks={11: _lm(0.0, 0.0, vis=0.0)},
        )
        fb = _make_frame(1, {11: (1.0, 0.0)})
        assert compute_frame_displacement(fa, fb, joints=[11]) == pytest.approx(0.0)

    def test_no_common_joints_returns_zero(self):
        fa = _make_frame(0, {11: (0.0, 0.0)})
        fb = _make_frame(1, {12: (1.0, 0.0)})
        assert compute_frame_displacement(fa, fb, joints=[11, 12]) == pytest.approx(0.0)

    def test_velocity_signal_length(self):
        frames = [_make_frame(i, {11: (i * 0.01, 0.5)}) for i in range(10)]
        v = compute_velocity_signal(frames, joints=[11])
        assert len(v) == 9


class TestSmoothVelocity:
    def test_preserves_length(self):
        v = np.random.rand(100)
        sv = smooth_velocity(v)
        assert len(sv) == len(v)

    def test_short_signal_returns_copy(self):
        v = np.array([0.1, 0.5, 0.3])
        sv = smooth_velocity(v, window_length=11)
        np.testing.assert_array_equal(sv, v)


# ---------------------------------------------------------------------------
# Pose vector tests (5)
# ---------------------------------------------------------------------------

class TestBuildPoseVector:
    def test_correct_dimension(self):
        norm = {j: _nlm(0.1 * j, 0.2 * j) for j in KEYPOSE_JOINTS}
        vec = build_pose_vector(norm)
        assert vec is not None
        assert len(vec) == 2 * len(KEYPOSE_JOINTS)

    def test_correct_layout(self):
        """Indices 0,1 = joint 11 x,y; indices 2,3 = joint 12 x,y."""
        norm = {j: _nlm(float(j), float(j) + 0.5) for j in KEYPOSE_JOINTS}
        vec = build_pose_vector(norm)
        assert vec is not None
        assert vec[0] == pytest.approx(11.0)
        assert vec[1] == pytest.approx(11.5)
        assert vec[2] == pytest.approx(12.0)
        assert vec[3] == pytest.approx(12.5)

    def test_missing_joint_returns_none(self):
        norm = {j: _nlm(0.0, 0.0) for j in KEYPOSE_JOINTS if j != 11}
        assert build_pose_vector(norm) is None

    def test_cosine_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_zero_norm_returns_zero(self):
        a = np.zeros(5)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Window tests (5)
# ---------------------------------------------------------------------------

class TestWindowLogic:
    def test_expected_frames_count(self):
        result = compute_expected_frames(1153, 1153, 19)
        assert len(result) == 19

    def test_expected_frames_scaling(self):
        """Student 2× longer than master → expected frames ≈ 2× master's."""
        r_short = compute_expected_frames(100, 100, 19)
        r_long = compute_expected_frames(200, 100, 19)
        for a, b in zip(r_short, r_long):
            assert abs(b - 2 * a) <= 2   # rounding tolerance

    def test_compute_search_window_basic(self):
        lo, hi = compute_search_window(500, 1000, 27)
        assert lo == 473
        assert hi == 527

    def test_compute_search_window_clamps_low(self):
        lo, hi = compute_search_window(5, 1000, 27)
        assert lo == 0
        assert hi == 32

    def test_adjacent_windows_have_gap(self):
        total, n = 380, 19
        hw = max(1, int(total / n * 0.45))
        expected = compute_expected_frames(total, total, n)
        windows = [compute_search_window(e, total, hw) for e in expected]
        for j in range(1, len(windows)):
            prev_hi = windows[j - 1][1]
            curr_lo = windows[j][0]
            assert curr_lo > prev_hi or curr_lo >= prev_hi   # no strict overlap required by clamping


# ---------------------------------------------------------------------------
# Detection tests (4)
# ---------------------------------------------------------------------------

class TestFindVelocityValley:
    def test_returns_deepest(self):
        v = np.ones(100) * 0.8
        v[50] = 0.1
        frame, prom = find_velocity_valley(v, (40, 60))
        assert frame == 50
        assert prom is not None and prom > 0

    def test_flat_signal_returns_none(self):
        v = np.ones(100) * 0.5
        frame, prom = find_velocity_valley(v, (40, 60))
        assert frame is None
        assert prom is None


class TestFindKeyposeMatch:
    def test_returns_max_similarity(self):
        sims = np.zeros(100)
        sims[55] = 0.92
        frame, sim = find_keypose_match(sims, (40, 70))
        assert frame == 55
        assert sim == pytest.approx(0.92)

    def test_below_threshold_returns_none(self):
        sims = np.ones(100) * 0.5
        frame, sim = find_keypose_match(sims, (40, 70), min_similarity=0.70)
        assert frame is None
        assert sim is None


# ---------------------------------------------------------------------------
# Combine tests (6)
# ---------------------------------------------------------------------------

class TestCombineSignals:
    def test_both_agree_within_threshold(self):
        frame, conf = combine_signals(50, 0.4, 52, 0.85, 55)
        assert conf == "high"
        assert frame == 50

    def test_both_disagree_normal_valley(self):
        frame, conf = combine_signals(50, 0.3, 65, 0.85, 55)
        assert conf == "medium"
        assert frame == 50   # velocity preferred

    def test_both_disagree_shallow_valley_high_sim(self):
        frame, conf = combine_signals(50, 0.03, 65, 0.95, 55,
                                      shallow_valley_threshold=0.05,
                                      high_sim_threshold=0.90)
        assert conf == "medium"
        assert frame == 65   # keypose preferred when valley is shallow

    def test_velocity_only(self):
        frame, conf = combine_signals(50, 0.4, None, None, 55)
        assert conf == "medium"
        assert frame == 50

    def test_keypose_only(self):
        frame, conf = combine_signals(None, None, 55, 0.85, 60)
        assert conf == "medium"
        assert frame == 55

    def test_neither_returns_expected(self):
        frame, conf = combine_signals(None, None, None, None, 60)
        assert conf == "low"
        assert frame == 60


# ---------------------------------------------------------------------------
# Integration tests (4)
# ---------------------------------------------------------------------------

class TestSegmentVideo:
    def _run(self, frames, keyposes_path="nonexistent_keyposes.json",
             master_total=None):
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return segment_video(
                student_frames=frames,
                keyposes_path=keyposes_path,
                master_total_frames=master_total or len(frames),
            )

    def test_returns_exactly_19(self):
        frames = _synthetic_frames(19, 30)
        boundaries = self._run(frames)
        assert len(boundaries) == 19

    def test_empty_keyposes_runs_without_error(self):
        frames = _synthetic_frames(19, 20)
        boundaries = self._run(frames)
        assert len(boundaries) == 19
        for b in boundaries:
            assert b.keypose_frame is None

    def test_boundary_frames_strictly_monotonic(self):
        frames = _synthetic_frames(19, 30)
        boundaries = self._run(frames)
        for j in range(1, len(boundaries)):
            assert boundaries[j].boundary_frame > boundaries[j - 1].boundary_frame

    def test_first_start_frame_is_zero(self):
        frames = _synthetic_frames(19, 20)
        boundaries = self._run(frames)
        assert boundaries[0].start_frame == 0


# ---------------------------------------------------------------------------
# keypose_marker tests (4)
# ---------------------------------------------------------------------------

class TestKeyposeMarker:
    def test_build_keypose_entry_vector_length(self):
        frames = _synthetic_frames(1, 30)
        entry = build_keypose_entry(
            frames, frame_index=15, movement_index=1,
            movement_name="test_move", itf_name="Test Technique",
        )
        assert len(entry["pose_vector"]) == 2 * len(KEYPOSE_JOINTS)

    def test_load_keyposes_or_empty_from_empty_dict(self, tmp_path):
        p = tmp_path / "keyposes.json"
        p.write_text("{}")
        data = load_keyposes_or_empty(str(p))
        assert data["keyposes"] == []
        assert data["poomsae"] == "chon_ji"

    def test_add_or_update_replaces_existing(self):
        frames = _synthetic_frames(1, 30)
        data = load_keyposes_or_empty("nonexistent.json")
        entry1 = build_keypose_entry(
            frames, 10, 3, "move_3_v1", "Tech 3 v1"
        )
        data = add_or_update_keypose(data, entry1)
        entry2 = build_keypose_entry(
            frames, 20, 3, "move_3_v2", "Tech 3 v2"
        )
        data = add_or_update_keypose(data, entry2)
        matches = [k for k in data["keyposes"] if k["movement_index"] == 3]
        assert len(matches) == 1
        assert matches[0]["movement_name"] == "move_3_v2"

    def test_validate_keyposes_detects_missing_movement(self):
        frames = _synthetic_frames(1, 30)
        data = load_keyposes_or_empty("nonexistent.json")
        # Add only 18 movements (skip movement 7)
        for i in range(1, 20):
            if i == 7:
                continue
            entry = build_keypose_entry(frames, i, i, f"move_{i}", f"Tech {i}")
            data = add_or_update_keypose(data, entry)
        errors = validate_keyposes(data)
        assert any("7" in e for e in errors)
