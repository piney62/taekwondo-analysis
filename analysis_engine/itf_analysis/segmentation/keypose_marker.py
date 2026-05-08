"""Instructor tool to build master_data/chon_ji/keyposes.json.

This module is a one-time utility — run it once per poomsae to mark the
19 completion frames in the master video.  It has no role in student
video segmentation and is never imported by segmentor.py at runtime.

Typical workflow (via notebooks/03_keypose_marking.ipynb):
    1. Load master poses: frames = load_poses_from_json(...)
    2. Scrub overlay video to each movement's completion frame.
    3. For each of 19 movements:
           entry = build_keypose_entry(frames, frame_index=..., movement_index=i, ...)
           data  = add_or_update_keypose(data, entry)
    4. validate_keyposes(data)   # confirm 19 entries, all 20-float vectors
    5. save_keyposes(data, path)
"""

import json
import os
from typing import List, Optional

import numpy as np

from itf_analysis.normalization.normalizer import normalize_pose
from itf_analysis.pose_extraction.extractor import FrameData
from itf_analysis.segmentation.segmentor import (
    KEYPOSE_JOINTS,
    KeyposeEntry,
    KeyposesData,
    build_pose_vector,
    load_poses_from_json,
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_keypose_vector(
    frames: List[FrameData],
    frame_index: int,
    joints: List[int] = KEYPOSE_JOINTS,
) -> Optional[np.ndarray]:
    """Normalize the frame at frame_index and extract a pose vector.

    Args:
        frames: Full pose sequence from Layer 1.
        frame_index: Target frame (0-based).
        joints: Joints to include in the vector.

    Returns:
        Float64 array of length 2*len(joints), or None if normalization fails.
    """
    if frame_index < 0 or frame_index >= len(frames):
        return None
    norm = normalize_pose(frames[frame_index].landmarks)
    if norm is None:
        return None
    return build_pose_vector(norm, joints)


def build_keypose_entry(
    frames: List[FrameData],
    frame_index: int,
    movement_index: int,
    movement_name: str,
    itf_name: str,
    joints: List[int] = KEYPOSE_JOINTS,
    mirror_of: Optional[int] = None,
    notes: str = "",
) -> KeyposeEntry:
    """Build a KeyposeEntry from a specific master video frame.

    Args:
        frames: Full pose sequence from Layer 1.
        frame_index: Completion frame of this movement (0-based).
        movement_index: 1-based ITF movement number (1–19).
        movement_name: Snake_case identifier, e.g. "left_walking_high_block".
        itf_name: Full technique name, e.g. "Ap Kubi Olgul Makgi (Left)".
        joints: Must match KEYPOSE_JOINTS used by segmentor.
        mirror_of: movement_index of the L/R counterpart, or None.
        notes: Optional instructor annotation.

    Returns:
        Populated KeyposeEntry.

    Raises:
        ValueError: If pose vector extraction fails for frame_index.
    """
    vec = extract_keypose_vector(frames, frame_index, joints)
    if vec is None:
        raise ValueError(
            f"Cannot extract pose vector for frame {frame_index} "
            f"(normalization failed — anchor joints missing?)."
        )
    return KeyposeEntry(
        movement_index=movement_index,
        movement_name=movement_name,
        itf_name=itf_name,
        pose_vector=vec.tolist(),
        source_frame=frame_index,
        source_timestamp_ms=frames[frame_index].timestamp_ms,
        mirror_of=mirror_of,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------

def load_keyposes_or_empty(path: str) -> KeyposesData:
    """Load keyposes.json; return a valid empty structure if absent or {}.

    Never raises. Safe for interactive instructor workflow.

    Args:
        path: Path to keyposes.json.

    Returns:
        KeyposesData with an empty keyposes list if the file is absent or empty.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data and data.get("keyposes") is not None:
            return data  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return KeyposesData(
        schema_version="1.0",
        poomsae="chon_ji",
        pose_joints=list(KEYPOSE_JOINTS),
        coord_order="x_then_y_interleaved",
        normalization="hip_centered_shoulder_width_1",
        keyposes=[],
    )


def add_or_update_keypose(
    data: KeyposesData,
    entry: KeyposeEntry,
) -> KeyposesData:
    """Add or replace the entry for entry['movement_index'].

    Returns a new KeyposesData sorted by movement_index (does not mutate).

    Args:
        data: Existing KeyposesData.
        entry: New or updated entry.

    Returns:
        Updated KeyposesData.
    """
    idx = entry["movement_index"]
    keyposes = [k for k in data["keyposes"] if k["movement_index"] != idx]
    keyposes.append(entry)
    keyposes.sort(key=lambda k: k["movement_index"])
    return KeyposesData(
        schema_version=data["schema_version"],
        poomsae=data["poomsae"],
        pose_joints=data["pose_joints"],
        coord_order=data["coord_order"],
        normalization=data["normalization"],
        keyposes=keyposes,
    )


def save_keyposes(data: KeyposesData, output_path: str) -> None:
    """Write KeyposesData to output_path as pretty-printed JSON.

    Args:
        data: KeyposesData to serialize.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"keyposes.json saved: {output_path} ({len(data['keyposes'])} entries)")


def validate_keyposes(
    data: KeyposesData,
    expected_count: int = 19,
) -> List[str]:
    """Validate a KeyposesData structure and return a list of error strings.

    Args:
        data: KeyposesData to validate.
        expected_count: Expected number of movement entries.

    Returns:
        Empty list if valid; otherwise a list of human-readable error messages.
    """
    errors: List[str] = []
    keyposes = data.get("keyposes", [])

    seen: dict = {}
    for entry in keyposes:
        idx = entry.get("movement_index")
        if idx in seen:
            errors.append(f"Duplicate movement_index: {idx}")
        seen[idx] = entry

        vec = entry.get("pose_vector", [])
        if len(vec) != 2 * len(data.get("pose_joints", KEYPOSE_JOINTS)):
            errors.append(
                f"Movement {idx}: pose_vector has {len(vec)} floats, "
                f"expected {2 * len(data.get('pose_joints', KEYPOSE_JOINTS))}"
            )

    expected_indices = set(range(1, expected_count + 1))
    missing = expected_indices - set(seen.keys())
    for idx in sorted(missing):
        errors.append(f"Movement {idx}: missing from keyposes")

    if len(keyposes) != expected_count and not missing:
        errors.append(
            f"Expected {expected_count} keyposes, found {len(keyposes)}"
        )

    return errors


# ---------------------------------------------------------------------------
# __main__: show marking status for master video
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    poses_path = str(
        project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_poses.json"
    )
    keyposes_path = str(
        project_root / "itf_analysis" / "master_data" / "chon_ji" / "keyposes.json"
    )

    frames = load_poses_from_json(poses_path)
    data = load_keyposes_or_empty(keyposes_path)

    print(f"Master video: {len(frames)} frames")
    print(f"Keyposes marked: {len(data['keyposes'])} / 19")

    errors = validate_keyposes(data)
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nAll 19 keyposes valid — ready for segmentation.")
