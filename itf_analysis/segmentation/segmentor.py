"""Layer 3: Movement segmentation via velocity-valley and keypose matching."""

import json
import os
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
from scipy.signal import savgol_filter

from itf_analysis.normalization.normalizer import NormalizedLandmark, normalize_pose
from itf_analysis.pose_extraction.extractor import FrameData, Landmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VELOCITY_JOINTS: List[int] = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
KEYPOSE_JOINTS: List[int] = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28]

SAVGOL_WINDOW: int = 11       # 367 ms at 30 fps
SAVGOL_POLY: int = 2
WINDOW_RATIO: float = 0.45    # half_window = ratio × avg_frames_per_movement
AGREE_THRESHOLD: int = 5      # frames within which both signals "agree"
SHALLOW_VALLEY: float = 0.05  # normalized velocity below this = smooth practitioner
HIGH_SIM_THRESH: float = 0.90 # cosine similarity above this = reliable keypose match
MIN_PROMINENCE: float = 0.02  # valley must rise ≥ 2 % of normalized range
MIN_SIMILARITY: float = 0.70  # minimum cosine similarity to accept a keypose match

Confidence = Literal["high", "medium", "low"]

# ---------------------------------------------------------------------------
# Shared types  (imported by keypose_marker.py — never import back from there)
# ---------------------------------------------------------------------------

class KeyposeEntry(TypedDict):
    movement_index: int
    movement_name: str
    itf_name: str
    pose_vector: List[float]          # exactly 20 floats
    source_frame: int
    source_timestamp_ms: float
    mirror_of: Optional[int]          # movement_index of L/R pair, or None
    notes: str


class KeyposesData(TypedDict):
    schema_version: str
    poomsae: str
    pose_joints: List[int]
    coord_order: str
    normalization: str
    keyposes: List[KeyposeEntry]      # 19 entries for Chon-Ji


@dataclass
class MovementBoundary:
    """Detected boundary for one poomsae movement."""
    movement_index: int               # 1-based
    start_frame: int                  # inclusive
    end_frame: int                    # inclusive (same as boundary_frame)
    boundary_frame: int               # detected completion frame
    confidence: Confidence
    velocity_frame: Optional[int]
    velocity_prominence: Optional[float]   # 0–1 normalized
    keypose_frame: Optional[int]
    keypose_similarity: Optional[float]    # cosine similarity 0–1
    expected_frame: int               # from linear time scaling


class SegmentationWarning(UserWarning):
    pass


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_poses_from_json(path: str) -> List[FrameData]:
    """Deserialize Layer 1 JSON output into a list of FrameData.

    Args:
        path: Path to poses JSON file produced by save_poses_to_json.

    Returns:
        List of FrameData with int landmark keys and Landmark values.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for d in data:
        landmarks = {
            int(k): Landmark(**v)
            for k, v in d["landmarks"].items()
        }
        frames.append(FrameData(
            frame_index=d["frame_index"],
            timestamp_ms=d["timestamp_ms"],
            landmarks=landmarks,
        ))
    return frames


def load_keyposes(path: str) -> Optional[KeyposesData]:
    """Load keyposes.json; return None if absent, unreadable, or empty.

    Args:
        path: Path to keyposes.json.

    Returns:
        KeyposesData or None.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data and data.get("keyposes"):
            return data  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def save_boundaries(boundaries: List[MovementBoundary], output_path: str) -> None:
    """Serialize MovementBoundary list to JSON.

    Args:
        boundaries: Output of segment_video.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in boundaries], f, indent=2)
    print(f"Boundaries saved: {output_path} ({len(boundaries)} movements)")


# ---------------------------------------------------------------------------
# Velocity signal pipeline
# ---------------------------------------------------------------------------

def compute_frame_displacement(
    frame_a: FrameData,
    frame_b: FrameData,
    joints: List[int] = VELOCITY_JOINTS,
) -> float:
    """Mean L2 displacement of active joints between two consecutive frames.

    Only joints present in BOTH frames with visibility > 0.0 are included.
    Interpolated joints (visibility == 0.0) are excluded from the mean.

    Args:
        frame_a: Earlier frame.
        frame_b: Later frame.
        joints: Joint indices to consider.

    Returns:
        Mean displacement in normalized image coordinates, or 0.0 if no
        qualifying joint pair exists.
    """
    displacements = []
    for j in joints:
        lm_a = frame_a.landmarks.get(j)
        lm_b = frame_b.landmarks.get(j)
        if lm_a is None or lm_b is None:
            continue
        if lm_a.visibility <= 0.0 or lm_b.visibility <= 0.0:
            continue
        dx = lm_b.x - lm_a.x
        dy = lm_b.y - lm_a.y
        dz = lm_b.z - lm_a.z
        displacements.append(np.sqrt(dx * dx + dy * dy + dz * dz))
    return float(np.mean(displacements)) if displacements else 0.0


def compute_velocity_signal(
    frames: List[FrameData],
    joints: List[int] = VELOCITY_JOINTS,
) -> np.ndarray:
    """Compute frame-to-frame mean joint displacement for all consecutive pairs.

    Args:
        frames: Sequence of FrameData from Layer 1.
        joints: Joint indices to use for displacement calculation.

    Returns:
        1-D array of length len(frames) - 1.
    """
    return np.array([
        compute_frame_displacement(frames[i], frames[i + 1], joints)
        for i in range(len(frames) - 1)
    ])


def smooth_velocity(
    velocity: np.ndarray,
    window_length: int = SAVGOL_WINDOW,
    polyorder: int = SAVGOL_POLY,
) -> np.ndarray:
    """Apply Savitzky-Golay filter to the raw velocity signal.

    Args:
        velocity: Raw velocity array.
        window_length: Must be odd; 11 ≈ 367 ms at 30 fps.
        polyorder: Polynomial order for the filter.

    Returns:
        Smoothed array of the same length.
    """
    if len(velocity) < window_length:
        return velocity.copy()
    return savgol_filter(velocity, window_length=window_length, polyorder=polyorder)


def normalize_velocity(velocity: np.ndarray) -> np.ndarray:
    """Scale velocity to [0, 1] using the 95th percentile as the ceiling.

    Using p95 (not max) makes the scale robust to brief occlusion spikes.

    Args:
        velocity: Raw or smoothed velocity array.

    Returns:
        Array clipped to [0, 1].
    """
    p95 = float(np.percentile(velocity, 95))
    if p95 < 1e-9:
        return np.zeros_like(velocity)
    return np.clip(velocity / p95, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Keypose signal pipeline
# ---------------------------------------------------------------------------

def build_pose_vector(
    normalized: Dict[int, NormalizedLandmark],
    joints: List[int] = KEYPOSE_JOINTS,
) -> Optional[np.ndarray]:
    """Extract a flat pose vector from normalized landmarks.

    Vector layout: [x11, y11, x12, y12, ..., x28, y28] (20 floats for
    the default KEYPOSE_JOINTS set).

    Args:
        normalized: Output of normalize_pose for one frame.
        joints: Joint indices to include.

    Returns:
        Float64 array of length 2*len(joints), or None if any joint absent.
    """
    for j in joints:
        if j not in normalized:
            return None
    vec: List[float] = []
    for j in joints:
        vec.append(normalized[j].x)
        vec.append(normalized[j].y)
    return np.array(vec, dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Args:
        a, b: 1-D float arrays of equal length.

    Returns:
        Value in [-1, 1]; returns 0.0 if either vector has near-zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_keypose_similarities(
    frames: List[FrameData],
    keypose_entry: KeyposeEntry,
    joints: List[int] = KEYPOSE_JOINTS,
) -> np.ndarray:
    """Compute cosine similarity of each frame's pose vector against a keypose.

    Frames where normalization or pose vector extraction fails are assigned
    similarity 0.0.

    Args:
        frames: Full sequence of FrameData.
        keypose_entry: Reference keypose from keyposes.json.
        joints: Must match the joints used when the keypose was created.

    Returns:
        Float64 array of length len(frames).
    """
    ref_vec = np.array(keypose_entry["pose_vector"], dtype=np.float64)
    sims = np.zeros(len(frames), dtype=np.float64)
    for i, frame in enumerate(frames):
        norm = normalize_pose(frame.landmarks)
        if norm is None:
            continue
        vec = build_pose_vector(norm, joints)
        if vec is None:
            continue
        sims[i] = cosine_similarity(vec, ref_vec)
    return sims


# ---------------------------------------------------------------------------
# Window and search
# ---------------------------------------------------------------------------

def compute_expected_frames(
    student_total: int,
    master_total: int,
    num_movements: int = 19,
) -> List[int]:
    """Compute expected boundary frames for the student video by uniform scaling.

    Expected positions are placed at 1/N, 2/N, ..., N/N of the video length,
    clamped to [0, student_total - 1].

    Args:
        student_total: Total number of frames in the student video.
        master_total: Total frames in the master video (reserved for future
            use when master keypose timing is incorporated).
        num_movements: Number of movements in the poomsae.

    Returns:
        List of num_movements frame indices, 0-based, monotonically increasing.
    """
    return [
        min(round((i + 1) * student_total / num_movements), student_total - 1)
        for i in range(num_movements)
    ]


def compute_search_window(
    expected: int,
    total_frames: int,
    half_width: int,
) -> Tuple[int, int]:
    """Return an inclusive search window [lo, hi] clamped to valid frame range.

    Args:
        expected: Centre of the window.
        total_frames: Total frames in the video.
        half_width: Half-width in frames.

    Returns:
        (lo, hi) inclusive, both within [0, total_frames - 1].
    """
    lo = max(0, expected - half_width)
    hi = min(total_frames - 1, expected + half_width)
    return lo, hi


def find_velocity_valley(
    smoothed_v: np.ndarray,
    window: Tuple[int, int],
    min_prominence: float = MIN_PROMINENCE,
) -> Tuple[Optional[int], Optional[float]]:
    """Find the deepest velocity valley within a search window.

    Prominence is defined as (window_max - window_min) / window_max, measuring
    how much the signal drops relative to the window's range.

    Args:
        smoothed_v: Normalized, smoothed velocity array (length N).
        window: (lo, hi) inclusive frame indices.
        min_prominence: Minimum required prominence to accept the valley.

    Returns:
        (frame_index, prominence) or (None, None) if no qualifying valley.
    """
    lo, hi = window
    window_slice = smoothed_v[lo:hi + 1]
    if len(window_slice) == 0:
        return None, None

    min_val = float(window_slice.min())
    max_val = float(window_slice.max())
    denom = max(max_val, 1e-9)
    prominence = (max_val - min_val) / denom

    if prominence < min_prominence:
        return None, None

    local_idx = int(np.argmin(window_slice))
    return lo + local_idx, float(prominence)


def find_keypose_match(
    similarities: Optional[np.ndarray],
    window: Tuple[int, int],
    min_similarity: float = MIN_SIMILARITY,
) -> Tuple[Optional[int], Optional[float]]:
    """Find the frame with peak cosine similarity within a search window.

    Args:
        similarities: Per-frame cosine similarity array, or None if unavailable.
        window: (lo, hi) inclusive frame indices.
        min_similarity: Minimum similarity required to accept the match.

    Returns:
        (frame_index, similarity) or (None, None) if unavailable or below threshold.
    """
    if similarities is None:
        return None, None
    lo, hi = window
    window_slice = similarities[lo:hi + 1]
    if len(window_slice) == 0:
        return None, None
    max_idx = int(np.argmax(window_slice))
    max_sim = float(window_slice[max_idx])
    if max_sim < min_similarity:
        return None, None
    return lo + max_idx, max_sim


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def combine_signals(
    velocity_frame: Optional[int],
    velocity_prominence: Optional[float],
    keypose_frame: Optional[int],
    keypose_similarity: Optional[float],
    expected_frame: int,
    agree_threshold: int = AGREE_THRESHOLD,
    shallow_valley_threshold: float = SHALLOW_VALLEY,
    high_sim_threshold: float = HIGH_SIM_THRESH,
) -> Tuple[int, Confidence]:
    """Merge velocity and keypose signals into a single boundary and confidence.

    Rules:
    - Both found, agree within agree_threshold → high confidence, velocity frame
    - Both found, disagree, shallow valley + high similarity → keypose frame, medium
    - Both found, disagree otherwise → velocity frame, medium
    - Only velocity found → velocity frame, medium
    - Only keypose found → keypose frame, medium
    - Neither found → expected_frame, low

    Args:
        velocity_frame: Detected valley frame, or None.
        velocity_prominence: Valley prominence in [0, 1], or None.
        keypose_frame: Best keypose match frame, or None.
        keypose_similarity: Cosine similarity of match, or None.
        expected_frame: Fallback frame from uniform time scaling.
        agree_threshold: Max frame distance to count as agreement.
        shallow_valley_threshold: Prominence below which the valley is "shallow".
        high_sim_threshold: Similarity above which the keypose match is "strong".

    Returns:
        (boundary_frame, confidence).
    """
    have_v = velocity_frame is not None
    have_k = keypose_frame is not None

    if have_v and have_k:
        distance = abs(velocity_frame - keypose_frame)  # type: ignore[operator]
        if distance <= agree_threshold:
            return velocity_frame, "high"  # type: ignore[return-value]
        shallow = (velocity_prominence or 0.0) < shallow_valley_threshold
        strong = (keypose_similarity or 0.0) >= high_sim_threshold
        if shallow and strong:
            return keypose_frame, "medium"  # type: ignore[return-value]
        return velocity_frame, "medium"  # type: ignore[return-value]

    if have_v:
        return velocity_frame, "medium"  # type: ignore[return-value]
    if have_k:
        return keypose_frame, "medium"  # type: ignore[return-value]
    return expected_frame, "low"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment_video(
    student_frames: List[FrameData],
    keyposes_path: str,
    master_total_frames: int,
    num_movements: int = 19,
) -> List[MovementBoundary]:
    """Segment a student video into num_movements movement boundaries.

    Combines velocity-valley detection and keypose cosine-similarity matching.
    When keyposes.json is empty or absent the algorithm falls back to velocity
    only (all confidence = "medium" or "low").

    Args:
        student_frames: Layer 1 output for the student video.
        keyposes_path: Path to keyposes.json (may be empty / absent).
        master_total_frames: Frame count of the master reference video.
        num_movements: Number of movements in the poomsae.

    Returns:
        List of num_movements MovementBoundary objects in movement order,
        with strictly increasing boundary_frame values.
    """
    N = len(student_frames)

    # --- Load keypose references ---
    keyposes_data = load_keyposes(keyposes_path)
    keypose_map: Dict[int, KeyposeEntry] = {}
    if keyposes_data is not None:
        for entry in keyposes_data["keyposes"]:
            keypose_map[entry["movement_index"]] = entry

    # --- Velocity pipeline ---
    raw_v = compute_velocity_signal(student_frames)
    smoothed_v = smooth_velocity(raw_v)
    norm_v = normalize_velocity(smoothed_v)
    # Pad to length N so frame indices align 1:1
    norm_v = np.append(norm_v, norm_v[-1] if len(norm_v) > 0 else 0.0)

    # --- Pre-compute expected positions and window size ---
    expected_frames = compute_expected_frames(N, master_total_frames, num_movements)
    half_window = max(1, int(N / num_movements * WINDOW_RATIO))

    # --- Pre-compute keypose similarity arrays (skip if no keyposes) ---
    similarity_arrays: Dict[int, np.ndarray] = {}
    for mov_idx, entry in keypose_map.items():
        similarity_arrays[mov_idx] = compute_keypose_similarities(
            student_frames, entry
        )

    # --- Per-movement detection ---
    boundaries: List[MovementBoundary] = []
    prev_end = -1

    for i in range(num_movements):
        movement_index = i + 1
        expected = expected_frames[i]

        lo, hi = compute_search_window(expected, N, half_window)
        lo = max(lo, prev_end + 1)
        if lo > hi:
            lo = hi = min(expected, N - 1)

        valley_frame, valley_prom = find_velocity_valley(norm_v, (lo, hi))
        sims = similarity_arrays.get(movement_index)
        kp_frame, kp_sim = find_keypose_match(sims, (lo, hi))

        boundary_frame, confidence = combine_signals(
            valley_frame, valley_prom, kp_frame, kp_sim, expected
        )

        if confidence == "low":
            warnings.warn(
                f"Movement {movement_index}: no signal found in window "
                f"[{lo}, {hi}]. Using expected frame {expected}.",
                SegmentationWarning,
                stacklevel=2,
            )

        boundaries.append(MovementBoundary(
            movement_index=movement_index,
            start_frame=prev_end + 1,
            end_frame=boundary_frame,
            boundary_frame=boundary_frame,
            confidence=confidence,
            velocity_frame=valley_frame,
            velocity_prominence=valley_prom,
            keypose_frame=kp_frame,
            keypose_similarity=kp_sim,
            expected_frame=expected,
        ))
        prev_end = boundary_frame

    # --- Monotonicity safety pass + start_frame fix ---
    for j in range(1, len(boundaries)):
        prev = boundaries[j - 1]
        curr = boundaries[j]
        if curr.boundary_frame <= prev.boundary_frame:
            fixed = prev.boundary_frame + 1
            warnings.warn(
                f"Movement {curr.movement_index}: boundary adjusted "
                f"from {curr.boundary_frame} to {fixed} for monotonicity.",
                SegmentationWarning,
                stacklevel=2,
            )
            curr = replace(curr, boundary_frame=fixed, end_frame=fixed,
                           confidence="low")
        curr = replace(curr, start_frame=prev.boundary_frame + 1)
        boundaries[j] = curr

    return boundaries


# ---------------------------------------------------------------------------
# __main__: validation on master video (velocity-only, no keyposes)
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

    print("Loading master poses...")
    frames = load_poses_from_json(poses_path)
    print(f"  {len(frames)} frames loaded")

    print(f"\nSegmenting (master-on-master, keyposes={'present' if load_keyposes(keyposes_path) else 'empty'})...")
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        boundaries = segment_video(
            student_frames=frames,
            keyposes_path=keyposes_path,
            master_total_frames=len(frames),
        )

    print(f"\n{'Mov':>4} {'start':>6} {'end':>6} {'expected':>9} {'conf':>8}  "
          f"{'v_frame':>8} {'v_prom':>7} {'k_frame':>8}")
    print("-" * 70)
    for b in boundaries:
        print(
            f"{b.movement_index:>4} {b.start_frame:>6} {b.end_frame:>6} "
            f"{b.expected_frame:>9} {b.confidence:>8}  "
            f"{str(b.velocity_frame):>8} "
            f"{f'{b.velocity_prominence:.3f}' if b.velocity_prominence else '   —':>7} "
            f"{str(b.keypose_frame):>8}"
        )

    conf_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for b in boundaries:
        conf_counts[b.confidence] += 1
    print(f"\nConfidence — high: {conf_counts['high']}  "
          f"medium: {conf_counts['medium']}  low: {conf_counts['low']}")

    if caught:
        print(f"\nWarnings ({len(caught)}):")
        for w in caught:
            print(f"  {w.message}")
