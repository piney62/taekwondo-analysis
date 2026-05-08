"""Layer 2: Coordinate normalization and joint angle extraction."""
import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

from itf_analysis.pose_extraction.extractor import Landmark


@dataclass
class NormalizedLandmark:
    """Landmark in hip-centered, shoulder-width-scaled coordinate space."""
    x: float
    y: float
    z: float
    visibility: float


def normalize_pose(
    landmarks: Dict[int, Landmark],
) -> Optional[Dict[int, NormalizedLandmark]]:
    """Normalize one frame's landmarks to hip-centered, shoulder-width-scaled space.

    Origin is set to the midpoint of hips (23, 24).
    Scale is set so that the shoulder-to-shoulder distance (11→12) equals 1.0.

    Args:
        landmarks: Raw per-joint landmark dict from Layer 1.

    Returns:
        Normalized landmark dict, or None if any anchor joint (11, 12, 23, 24)
        is missing or shoulder width is below 1e-6.
    """
    for idx in (11, 12, 23, 24):
        if idx not in landmarks:
            return None

    hip_cx = (landmarks[23].x + landmarks[24].x) / 2
    hip_cy = (landmarks[23].y + landmarks[24].y) / 2
    hip_cz = (landmarks[23].z + landmarks[24].z) / 2

    shoulder_width = math.sqrt(
        (landmarks[11].x - landmarks[12].x) ** 2
        + (landmarks[11].y - landmarks[12].y) ** 2
        + (landmarks[11].z - landmarks[12].z) ** 2
    )

    if shoulder_width < 1e-6:
        return None

    return {
        idx: NormalizedLandmark(
            x=(lm.x - hip_cx) / shoulder_width,
            y=(lm.y - hip_cy) / shoulder_width,
            z=(lm.z - hip_cz) / shoulder_width,
            visibility=lm.visibility,
        )
        for idx, lm in landmarks.items()
    }


def _vec3(a, b) -> tuple:
    """Return vector from b to a as (dx, dy, dz)."""
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def _angle_at_vertex(a, b, c) -> float:
    """Angle in degrees at vertex b in the a-b-c triplet (3D).

    Args:
        a, b, c: Objects with .x .y .z attributes. b is the vertex.

    Returns:
        Angle in [0, 180] degrees, or 0.0 if any arm has near-zero length.
    """
    ba = _vec3(a, b)
    bc = _vec3(c, b)

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0

    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))))


def extract_joint_angles(
    landmarks: Dict[int, Union[Landmark, NormalizedLandmark]],
) -> Dict[str, float]:
    """Compute named joint angles from a landmark dict.

    Three-point angles use 3D coordinates (x, y, z).
    Line angles (shoulder tilt, hip tilt) use 2D (x, y) only.
    A key is absent from the result if any required joint is missing.

    Args:
        landmarks: Per-joint dict; values must have .x .y .z attributes.
            Works with both raw Landmark and NormalizedLandmark.

    Returns:
        Dict mapping angle name to degrees.
    """
    angles: Dict[str, float] = {}

    _three_point = {
        "right_knee":  (24, 26, 28),
        "left_knee":   (23, 25, 27),
        "right_elbow": (12, 14, 16),
        "left_elbow":  (11, 13, 15),
        "right_hip":   (12, 24, 26),
        "left_hip":    (11, 23, 25),
    }

    for name, (ai, bi, ci) in _three_point.items():
        if ai in landmarks and bi in landmarks and ci in landmarks:
            angles[name] = _angle_at_vertex(
                landmarks[ai], landmarks[bi], landmarks[ci]
            )

    # Line angles: atan2(dy, dx) relative to positive-x axis
    if 11 in landmarks and 12 in landmarks:
        dx = landmarks[12].x - landmarks[11].x
        dy = landmarks[12].y - landmarks[11].y
        angles["shoulder_line_angle"] = math.degrees(math.atan2(dy, dx))

    if 23 in landmarks and 24 in landmarks:
        dx = landmarks[24].x - landmarks[23].x
        dy = landmarks[24].y - landmarks[23].y
        angles["hip_line_angle"] = math.degrees(math.atan2(dy, dx))

    return angles


if __name__ == "__main__":
    import pathlib
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    json_path = (
        project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_poses.json"
    )

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ok = skip = 0
    last_angles: Dict[str, float] = {}

    for frame in data:
        raw = {
            int(k): Landmark(**v) for k, v in frame["landmarks"].items()
        }
        normalized = normalize_pose(raw)
        if normalized is None:
            skip += 1
            continue
        ok += 1
        last_angles = extract_joint_angles(normalized)

    print(f"\n=== normalization check ===")
    print(f"frames normalized: {ok}  |  skipped: {skip}")
    print("\nAngles from last frame:")
    for name, deg in last_angles.items():
        print(f"  {name:25s}: {deg:7.2f} deg")
