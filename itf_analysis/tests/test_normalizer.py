"""Unit tests for Layer 2: normalization and joint angle extraction."""
import math

import pytest

from itf_analysis.normalization.normalizer import (
    NormalizedLandmark,
    extract_joint_angles,
    normalize_pose,
)
from itf_analysis.pose_extraction.extractor import Landmark


def _lm(x: float, y: float, z: float = 0.0, vis: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=vis)


def _nlm(x: float, y: float, z: float = 0.0) -> NormalizedLandmark:
    return NormalizedLandmark(x=x, y=y, z=z, visibility=1.0)


def _base_landmarks() -> dict:
    """Minimal landmark set with known geometry.

    Shoulders: 11=(0.3,0.2,0)  12=(0.7,0.2,0) → width=0.4
    Hips:      23=(0.4,0.6,0)  24=(0.6,0.6,0) → center=(0.5,0.6,0)
    """
    return {
        11: _lm(0.3, 0.2),
        12: _lm(0.7, 0.2),
        23: _lm(0.4, 0.6),
        24: _lm(0.6, 0.6),
    }


# ---------------------------------------------------------------------------
# normalize_pose
# ---------------------------------------------------------------------------

class TestNormalizePose:
    def test_hip_center_becomes_origin(self):
        result = normalize_pose(_base_landmarks())
        assert result is not None
        hip_cx = (result[23].x + result[24].x) / 2
        hip_cy = (result[23].y + result[24].y) / 2
        assert abs(hip_cx) < 1e-9
        assert abs(hip_cy) < 1e-9

    def test_shoulder_width_becomes_one(self):
        result = normalize_pose(_base_landmarks())
        assert result is not None
        sw = math.sqrt(
            (result[11].x - result[12].x) ** 2
            + (result[11].y - result[12].y) ** 2
            + (result[11].z - result[12].z) ** 2
        )
        assert abs(sw - 1.0) < 1e-9

    def test_returns_none_when_anchor_missing(self):
        lms = _base_landmarks()
        del lms[23]
        assert normalize_pose(lms) is None

    def test_returns_none_when_shoulder_width_zero(self):
        lms = _base_landmarks()
        lms[12] = lms[11]   # same position → zero width
        assert normalize_pose(lms) is None

    def test_visibility_preserved(self):
        lms = _base_landmarks()
        lms[11] = _lm(0.3, 0.2, vis=0.75)
        result = normalize_pose(lms)
        assert result is not None
        assert abs(result[11].visibility - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# extract_joint_angles
# ---------------------------------------------------------------------------

class TestExtractJointAngles:
    def test_right_angle_90_degrees(self):
        """A-B-C where A=(1,0), B=(0,0), C=(0,1) → angle at B = 90°."""
        lms = {
            24: _nlm(1.0, 0.0),   # a
            26: _nlm(0.0, 0.0),   # b  (vertex = right knee)
            28: _nlm(0.0, 1.0),   # c
        }
        angles = extract_joint_angles(lms)
        assert abs(angles["right_knee"] - 90.0) < 1e-6

    def test_straight_line_180_degrees(self):
        """Collinear A-B-C → angle at B = 180°."""
        lms = {
            24: _nlm(1.0,  0.0),
            26: _nlm(0.0,  0.0),
            28: _nlm(-1.0, 0.0),
        }
        angles = extract_joint_angles(lms)
        assert abs(angles["right_knee"] - 180.0) < 1e-6

    def test_45_degree_angle(self):
        """A=(1,0), B=(0,0), C=(1,1) normalized → angle ≈ 45°."""
        lms = {
            24: _nlm(1.0, 0.0),
            26: _nlm(0.0, 0.0),
            28: _nlm(1.0, 1.0),
        }
        angles = extract_joint_angles(lms)
        assert abs(angles["right_knee"] - 45.0) < 1e-6

    def test_missing_joint_skips_angle(self):
        lms = {
            24: _nlm(1.0, 0.0),
            26: _nlm(0.0, 0.0),
            # 28 absent
        }
        angles = extract_joint_angles(lms)
        assert "right_knee" not in angles

    def test_shoulder_line_horizontal(self):
        """11 and 12 on the same y → shoulder_line_angle ≈ 0°."""
        lms = {11: _nlm(0.0, 0.0), 12: _nlm(1.0, 0.0)}
        angles = extract_joint_angles(lms)
        assert abs(angles["shoulder_line_angle"]) < 1e-6

    def test_hip_line_horizontal(self):
        lms = {23: _nlm(0.0, 0.0), 24: _nlm(1.0, 0.0)}
        angles = extract_joint_angles(lms)
        assert abs(angles["hip_line_angle"]) < 1e-6

    def test_all_eight_angles_returned_when_complete(self):
        """All 8 named angles present when every required joint is provided."""
        lms = {
            11: _nlm(-1.0, -1.0), 12: _nlm(1.0, -1.0),
            13: _nlm(-0.5, -0.5), 14: _nlm(0.5, -0.5),
            15: _nlm(-0.5,  0.0), 16: _nlm(0.5,  0.0),
            23: _nlm(-0.3,  0.5), 24: _nlm(0.3,  0.5),
            25: _nlm(-0.3,  1.0), 26: _nlm(0.3,  1.0),
            27: _nlm(-0.3,  1.5), 28: _nlm(0.3,  1.5),
        }
        angles = extract_joint_angles(lms)
        expected = {
            "right_knee", "left_knee",
            "right_elbow", "left_elbow",
            "right_hip", "left_hip",
            "shoulder_line_angle", "hip_line_angle",
        }
        assert expected == set(angles.keys())
