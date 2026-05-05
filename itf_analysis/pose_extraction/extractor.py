"""Layer 1: MediaPipe-based pose extraction from video."""
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

JOINT_COUNT = 33


@dataclass
class Landmark:
    """Single joint coordinate from MediaPipe Pose."""
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FrameData:
    """Pose data for one video frame."""
    frame_index: int
    timestamp_ms: float
    landmarks: Dict[int, Landmark]


def extract_poses_from_video(
    video_path: str,
    visibility_threshold: float = 0.5,
) -> List[FrameData]:
    """Extract 33-joint pose landmarks from every frame of a video.

    Landmarks whose visibility score is below *visibility_threshold* are
    dropped from the frame's landmark dict so downstream code never sees
    low-confidence joints.  Call interpolate_missing_landmarks() afterward
    to fill those gaps.

    Args:
        video_path: Absolute or relative path to the input video file.
        visibility_threshold: Minimum visibility score to keep a landmark.
            Landmarks below this value are treated as undetected.

    Returns:
        List of FrameData, one per frame. Frames where MediaPipe fails to
        detect a pose are included with an empty landmarks dict so that
        frame indices stay contiguous for downstream DTW alignment.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames: List[FrameData] = []
    frame_index = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = {
                    idx: Landmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility,
                    )
                    for idx, lm in enumerate(results.pose_landmarks.landmark)
                    if lm.visibility >= visibility_threshold
                }
            else:
                landmarks = {}

            frames.append(FrameData(frame_index, timestamp_ms, landmarks))

            if frame_index % 100 == 0:
                print(f"  processing: {frame_index}/{total_frames} frames")

            frame_index += 1
    finally:
        cap.release()
        pose.close()

    detected = sum(1 for f in frames if f.landmarks)
    print(
        f"\ndone: {len(frames)} frames total, pose detected in {detected} frames "
        f"({detected / len(frames) * 100:.1f}%)"
    )
    return frames


def interpolate_missing_landmarks(frames: List[FrameData]) -> List[FrameData]:
    """Fill gaps where a joint drops below the visibility threshold.

    For each of the 33 joints, finds runs of frames where the joint is absent
    and linearly interpolates x/y/z from the nearest valid frames on either
    side.  Edge gaps (no valid frame before or after) are left empty.
    Interpolated landmarks are assigned visibility=0.0 to flag their origin.

    Args:
        frames: Output of extract_poses_from_video.  Modified in-place.

    Returns:
        The same list with gaps filled.
    """
    n = len(frames)

    for joint_idx in range(JOINT_COUNT):
        # Collect (frame_index, Landmark) for frames that have this joint
        valid: List[tuple[int, Landmark]] = [
            (i, frames[i].landmarks[joint_idx])
            for i in range(n)
            if joint_idx in frames[i].landmarks
        ]

        if len(valid) < 2:
            continue

        valid_indices = [v[0] for v in valid]

        for gap_start in range(n):
            if joint_idx in frames[gap_start].landmarks:
                continue

            # Find nearest valid frames before and after this gap frame
            before: Optional[tuple[int, Landmark]] = None
            after: Optional[tuple[int, Landmark]] = None

            for vi, lm in reversed(valid):
                if vi < gap_start:
                    before = (vi, lm)
                    break
            for vi, lm in valid:
                if vi > gap_start:
                    after = (vi, lm)
                    break

            if before is None or after is None:
                continue

            bi, blm = before
            ai, alm = after
            t = (gap_start - bi) / (ai - bi)

            frames[gap_start].landmarks[joint_idx] = Landmark(
                x=blm.x + t * (alm.x - blm.x),
                y=blm.y + t * (alm.y - blm.y),
                z=blm.z + t * (alm.z - blm.z),
                visibility=0.0,
            )

    filled = sum(
        1
        for f in frames
        for lm in f.landmarks.values()
        if lm.visibility == 0.0
    )
    print(f"Interpolation complete: {filled} landmark-frames filled.")
    return frames


def compute_visibility_stats(frames: List[FrameData]) -> Dict[int, List[float]]:
    """Collect per-joint visibility scores across all frames.

    Args:
        frames: Output of extract_poses_from_video (before or after interpolation).

    Returns:
        Dict mapping joint index to a list of visibility scores (one per frame
        where that joint was detected by MediaPipe, i.e. visibility > 0).
    """
    stats: Dict[int, List[float]] = {i: [] for i in range(JOINT_COUNT)}
    for f in frames:
        for idx, lm in f.landmarks.items():
            if lm.visibility > 0.0:
                stats[idx].append(lm.visibility)
    return stats


def save_poses_to_json(frames: List[FrameData], output_path: str) -> None:
    """Serialize extracted pose data to a JSON file.

    Args:
        frames: Output of extract_poses_from_video.
        output_path: Destination file path (created if absent).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = []
    for f in frames:
        d = asdict(f)
        # asdict converts Dict[int, Landmark] keys to int — JSON needs str keys
        d["landmarks"] = {str(k): v for k, v in d["landmarks"].items()}
        data.append(d)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    print(f"JSON saved: {output_path} ({len(data)} frames)")


def visualize_pose_overlay(
    video_path: str,
    frames: List[FrameData],
    output_path: str,
) -> None:
    """Render skeleton overlay on the original video for visual QA.

    Args:
        video_path: Path to the original input video.
        frames: Output of extract_poses_from_video (must be same video).
        output_path: Destination path for the annotated output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    try:
        for frame_data in frames:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_data.landmarks:
                lm_proto = landmark_pb2.NormalizedLandmarkList()
                for idx in range(JOINT_COUNT):
                    lm = frame_data.landmarks.get(idx)
                    if lm:
                        entry = lm_proto.landmark.add()
                        entry.x, entry.y, entry.z, entry.visibility = (
                            lm.x, lm.y, lm.z, lm.visibility
                        )
                    else:
                        lm_proto.landmark.add()

                drawing.draw_landmarks(
                    bgr,
                    lm_proto,
                    pose_connections,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                )

            cv2.putText(
                bgr,
                f"frame={frame_data.frame_index}  t={frame_data.timestamp_ms:.0f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        cap.release()
        writer.release()

    print(f"Overlay video saved: {output_path}")


if __name__ == "__main__":
    import pathlib
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    project_root = pathlib.Path(__file__).parent.parent.parent
    video = str(project_root / "itf_analysis" / "sample_videos" / "chon_ji_master.mp4")
    json_out = str(project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_poses.json")
    overlay_out = str(project_root / "itf_analysis" / "sample_videos" / "chon_ji_master_overlay.mp4")

    print("=== pose extraction (visibility_threshold=0.5) ===")
    frames = extract_poses_from_video(video, visibility_threshold=0.5)

    print("\n=== interpolate missing landmarks ===")
    frames = interpolate_missing_landmarks(frames)

    print("\n=== save JSON ===")
    save_poses_to_json(frames, json_out)

    print("\n=== generate overlay video ===")
    visualize_pose_overlay(video, frames, overlay_out)
