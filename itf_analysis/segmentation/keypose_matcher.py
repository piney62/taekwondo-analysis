"""Layer 3 keypose matching: angle-based distance and candidate search.

Loads stored master keypose angle profiles and finds the best matching frame
in a pose sequence. Used as one signal in the dual-signal segmentation pipeline
alongside velocity valley detection.

Typical usage:
    keyposes = load_master_keyposes(path)
    candidates = find_keypose_candidates(all_frame_angles, keyposes, threshold=25.0)
"""

import json
import math
from typing import Dict, List, Optional, Tuple

ANGLE_KEYS: List[str] = [
    "right_knee",
    "left_knee",
    "right_elbow",
    "left_elbow",
    "right_hip",
    "left_hip",
    "shoulder_line_angle",
    "hip_line_angle",
]


def load_master_keyposes(json_path: str) -> List[Dict]:
    """Load master keypose angle profiles from a JSON file.

    Args:
        json_path: Path to a keyposes_angles.json file with schema:
            { "poomsae": "chon_ji",
              "keyposes": [
                { "movement_index": 1, "source_frame": 87,
                  "angles": { "right_knee": 120.5, ... } },
                ...
              ] }

    Returns:
        List of keypose dicts sorted by movement_index.
        Returns empty list if the file is absent or has no keyposes key.

    Raises:
        ValueError: If the file exists but cannot be parsed as JSON.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse {json_path}: {exc}") from exc

    keyposes = data.get("keyposes", [])
    keyposes.sort(key=lambda k: k["movement_index"])
    return keyposes


def keypose_distance(
    frame_angles: Dict[str, float],
    keypose_angles: Dict[str, float],
    angle_keys: List[str] = ANGLE_KEYS,
) -> float:
    """Compute RMS angular distance between a frame and a stored keypose.

    Only angle keys present in both dicts are compared. Line angles
    (atan2-based, range -180..180) use circular arithmetic to avoid
    wrap-around errors.

    Args:
        frame_angles: Angle dict for one frame (from extract_joint_angles).
        keypose_angles: Stored angles for a movement completion pose.
        angle_keys: Which angle keys to compare.

    Returns:
        RMS distance in degrees. Returns float('inf') if no common keys.
    """
    squared_diffs: List[float] = []
    for key in angle_keys:
        if key not in frame_angles or key not in keypose_angles:
            continue
        diff = frame_angles[key] - keypose_angles[key]
        if "line_angle" in key:
            diff = (diff + 180.0) % 360.0 - 180.0
        squared_diffs.append(diff * diff)

    if not squared_diffs:
        return float("inf")
    return math.sqrt(sum(squared_diffs) / len(squared_diffs))


def find_keypose_candidates(
    all_frame_angles: List[Tuple[int, Dict[str, float]]],
    keyposes: List[Dict],
    threshold: float = 25.0,
) -> List[Dict]:
    """Find the best matching frame for each keypose across the full sequence.

    For each keypose, scans all frames and returns the one with minimum
    RMS angular distance. A match is declared when distance <= threshold.

    Args:
        all_frame_angles: List of (frame_index, angles_dict) for every frame.
        keyposes: Keypose list from load_master_keyposes (or in-memory list).
        threshold: RMS distance (degrees) below which a frame counts as matched.

    Returns:
        List of result dicts sorted by movement_index, each with:
        - movement_index (int)
        - source_frame (Optional[int])  — from the keypose entry if present
        - best_frame (Optional[int])    — frame with minimum distance
        - best_distance (float)
        - matched (bool)                — best_distance <= threshold
    """
    results: List[Dict] = []

    for kp in keyposes:
        movement_index: int = kp["movement_index"]
        kp_angles: Dict[str, float] = kp.get("angles", {})
        source_frame: Optional[int] = kp.get("source_frame")

        best_frame: Optional[int] = None
        best_dist = float("inf")

        for frame_idx, frame_angles in all_frame_angles:
            dist = keypose_distance(frame_angles, kp_angles)
            if dist < best_dist:
                best_dist = dist
                best_frame = frame_idx

        results.append(
            {
                "movement_index": movement_index,
                "source_frame": source_frame,
                "best_frame": best_frame,
                "best_distance": best_dist,
                "matched": best_dist <= threshold,
            }
        )

    results.sort(key=lambda r: r["movement_index"])
    return results


# ---------------------------------------------------------------------------
# __main__: status report using keyposes_angles.json if available
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    keyposes_path = str(
        project_root / "itf_analysis" / "master_data" / "chon_ji" / "keyposes_angles.json"
    )

    keyposes = load_master_keyposes(keyposes_path)
    if not keyposes:
        print("keyposes_angles.json not found or empty.")
        print("Run notebook 03a_keypose_matching.ipynb to bootstrap and save keyposes.")
    else:
        print(f"Keyposes loaded: {len(keyposes)} / 19")
        for kp in keyposes:
            print(
                f"  Movement {kp['movement_index']:2d}: "
                f"source_frame={kp.get('source_frame', '?'):>5}  "
                f"angles={len(kp.get('angles', {}))} keys"
            )
