"""Layer 5 relay server: receives alignment data from Flutter, returns LLM feedback.

Start with:
    uvicorn server.relay:app --reload --port 8000

Flutter sends POST /feedback with serialized MovementAlignment data.
The server runs difference analysis and returns natural-language feedback.
"""

from __future__ import annotations

import sys
import pathlib

# Make the project root importable when run from within server/ or project root.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from itf_analysis.feedback.difference_analyzer import (
    MovementIssueSummary,
    JointIssue,
    _resolve_thresholds,
    _DEFAULT_THRESHOLDS,
)
from itf_analysis.feedback.llm_client import get_llm_client
from itf_analysis.feedback.llm_feedback import generate_full_feedback

app = FastAPI(title="Taekwondo Feedback Relay", version="1.0.0")

_VIDEOS_DIR = pathlib.Path(__file__).parent.parent / "itf_analysis" / "sample_videos"
if _VIDEOS_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(_VIDEOS_DIR)), name="videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models (mirror Flutter's serialized MovementAlignment)
# ---------------------------------------------------------------------------

class FramePairPayload(BaseModel):
    master_frame: int
    student_frame: int
    angle_diffs: Dict[str, float]
    angle_delta: Dict[str, float]
    rms_diff: float


class MovementAlignmentPayload(BaseModel):
    movement_number: int
    frame_pairs: List[FramePairPayload]
    mean_rms: float
    max_rms: float


class KeyposePayload(BaseModel):
    movement_index: int
    movement_name: str = ""
    itf_name: str = ""


class FeedbackRequest(BaseModel):
    poomsae: str = "chon_ji"
    alignments: List[MovementAlignmentPayload]
    keyposes: Optional[List[KeyposePayload]] = None


class FeedbackResponse(BaseModel):
    feedback: Dict[int, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CIRCULAR_KEYS = frozenset({"shoulder_line_angle", "hip_line_angle"})


def _summarize_from_payload(
    payload: MovementAlignmentPayload,
    thresholds: Dict,
    top_n: int = 3,
) -> MovementIssueSummary:
    """Reproduce summarize_movement_issues() from serialized frame pairs."""
    resolved = _resolve_thresholds(thresholds, payload.movement_number)
    joint_diffs: Dict[str, list] = {}

    for fp in payload.frame_pairs:
        for joint, diff in fp.angle_diffs.items():
            joint_diffs.setdefault(joint, []).append(diff)

    issues: List[JointIssue] = []
    for joint, diffs in joint_diffs.items():
        mean_diff = sum(diffs) / len(diffs)
        threshold = resolved.get(joint, 20.0)
        if mean_diff <= threshold:
            continue

        # Direction from angle_delta
        deltas = [
            fp.angle_delta.get(joint, 0.0)
            for fp in payload.frame_pairs
            if joint in fp.angle_delta
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        direction = "too_large" if mean_delta > 0 else "too_small"
        severity = mean_diff / threshold

        issues.append(
            JointIssue(
                joint=joint,
                master_angle=0.0,
                student_angle=0.0,
                diff=mean_diff,
                direction=direction,
                severity=severity,
            )
        )

    issues.sort(key=lambda x: x.severity, reverse=True)
    rms_values = [fp.rms_diff for fp in payload.frame_pairs]
    mean_rms = sum(rms_values) / len(rms_values) if rms_values else 0.0

    return MovementIssueSummary(
        movement_number=payload.movement_number,
        frame_count=len(payload.frame_pairs),
        mean_rms=mean_rms,
        issues=issues[:top_n],
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/feedback", response_model=FeedbackResponse)
def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Receive Flutter alignment data, run LLM, return per-movement feedback."""
    try:
        client = get_llm_client()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM client init failed: {exc}")

    thresholds = {k: v for k, v in _DEFAULT_THRESHOLDS.items()}

    summaries = {
        payload.movement_number: _summarize_from_payload(payload, thresholds)
        for payload in request.alignments
    }

    kp_list = None
    if request.keyposes:
        kp_list = [
            {
                "movement_index": kp.movement_index,
                "movement_name":  kp.movement_name,
                "itf_name":       kp.itf_name,
            }
            for kp in request.keyposes
        ]

    feedback = generate_full_feedback(
        summaries,
        client,
        keyposes=kp_list,
        poomsae_name=request.poomsae.replace("_", "-").title(),
    )

    return FeedbackResponse(feedback=feedback)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.relay:app", host="0.0.0.0", port=8000, reload=True)
