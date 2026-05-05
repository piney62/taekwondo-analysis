"""Instructor tool to build master_data/chon_ji/keyposes_angles.json.

Extracts joint angles at instructor-marked completion frames and saves them
for use by keypose_matcher.py. Run once per poomsae, from the marking notebook.

Typical workflow (via notebooks/03_keypose_marking.ipynb):
    1. Run velocity detection → get 19 candidate frames
    2. Watch chon_ji_master_overlay.mp4, adjust KEYPOSE_FRAMES dict as needed
    3. Call build_angle_entry() for each of 19 movements
    4. Call validate_keyposes_angles() → confirm 0 errors
    5. Call save_keyposes_angles() → writes keyposes_angles.json
"""

import json
import os
from typing import Dict, List, Optional


def build_angle_entry(
    angles_by_frame: Dict[int, Dict[str, float]],
    frame_index: int,
    movement_index: int,
    movement_name: str = "",
    itf_name: str = "",
) -> Optional[Dict]:
    """Build a single keypose entry from the angles at frame_index.

    Args:
        angles_by_frame: Dict mapping frame_index → angles_dict
            (built from all_frame_angles in the notebook).
        frame_index: The completion frame of this movement.
        movement_index: 1-based movement number (1–19).
        movement_name: Short snake_case identifier, e.g. "left_high_block".
        itf_name: Full ITF technique name in English.

    Returns:
        Keypose entry dict, or None if frame_index is absent from angles_by_frame.
    """
    if frame_index not in angles_by_frame:
        return None
    return {
        "movement_index": movement_index,
        "movement_name": movement_name or f"movement_{movement_index:02d}",
        "itf_name": itf_name,
        "source_frame": frame_index,
        "angles": angles_by_frame[frame_index],
    }


def save_keyposes_angles(
    keyposes: List[Dict],
    output_path: str,
    poomsae: str = "chon_ji",
) -> None:
    """Serialize and write keypose entries to output_path as JSON.

    Args:
        keyposes: List of entries from build_angle_entry().
        output_path: Destination file path (created if absent).
        poomsae: Poomsae identifier stored in the file header.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = {
        "poomsae": poomsae,
        "keyposes": sorted(keyposes, key=lambda k: k["movement_index"]),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}  ({len(keyposes)} entries)")


def load_keyposes_angles_or_empty(
    path: str,
    poomsae: str = "chon_ji",
) -> Dict:
    """Load keyposes_angles.json; return an empty structure if absent or empty.

    Args:
        path: Path to keyposes_angles.json.
        poomsae: Default poomsae value for empty structure.

    Returns:
        Dict with "poomsae" and "keyposes" keys. Never raises.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data and data.get("keyposes"):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"poomsae": poomsae, "keyposes": []}


def validate_keyposes_angles(
    data: Dict,
    expected_count: int = 19,
) -> List[str]:
    """Validate a keyposes_angles structure.

    Args:
        data: Dict from load_keyposes_angles_or_empty or built in notebook.
        expected_count: Expected number of movement entries.

    Returns:
        Empty list if valid; list of human-readable error strings otherwise.
    """
    errors: List[str] = []
    keyposes = data.get("keyposes", [])

    seen: Dict[int, Dict] = {}
    for kp in keyposes:
        idx = kp.get("movement_index")
        if idx in seen:
            errors.append(f"Duplicate movement_index: {idx}")
        seen[idx] = kp
        if not kp.get("angles"):
            errors.append(f"Movement {idx}: angles dict is empty")
        elif len(kp["angles"]) < 6:
            errors.append(
                f"Movement {idx}: only {len(kp['angles'])} angle keys "
                f"(expected ≥ 6)"
            )

    missing = set(range(1, expected_count + 1)) - set(seen.keys())
    for idx in sorted(missing):
        errors.append(f"Movement {idx}: missing from keyposes")

    return errors


# ---------------------------------------------------------------------------
# __main__: show current marking status
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    keyposes_path = str(
        project_root / "itf_analysis" / "master_data" / "chon_ji" / "keyposes_angles.json"
    )

    data = load_keyposes_angles_or_empty(keyposes_path)
    keyposes = data.get("keyposes", [])
    print(f"Marked: {len(keyposes)} / 19")

    errors = validate_keyposes_angles(data)
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All 19 keyposes valid.")

    if keyposes:
        print(f"\n{'Mov':>4} {'frame':>7} {'angles':>8}")
        print("-" * 24)
        for kp in keyposes:
            print(
                f"{kp['movement_index']:>4} {kp['source_frame']:>7} "
                f"{len(kp.get('angles', {})):>8} keys"
            )
