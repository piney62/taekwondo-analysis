"""Layer 4 alignment: match student boundaries to master movements with gap handling.

Maps each of the 19 master movements to the closest student boundary (by
movement_number), labels unmatched master movements as 'skipped', and labels
student boundaries that were not claimed by any master movement as 'extra'.

Also provides validate_alignment() to produce user-facing warnings.

Typical usage:
    from itf_analysis.alignment.gap_aligner import (
        align_with_gaps, validate_alignment, AlignmentPair, ValidationResult
    )
    pairs  = align_with_gaps(master_keyposes, student_boundaries)
    result = validate_alignment(pairs)
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from itf_analysis.segmentation.segmenter import MovementBoundary

MatchType = Literal["matched", "skipped", "extra"]


@dataclass
class AlignmentPair:
    """Pairing between one master movement and one student boundary.

    Attributes:
        master_movement: 1-based master movement index (1–19).
            None for 'extra' student boundaries.
        student_boundary: Matched student MovementBoundary.
            None for 'skipped' master movements.
        match_type: 'matched', 'skipped', or 'extra'.
    """

    master_movement: Optional[int]
    student_boundary: Optional[MovementBoundary]
    match_type: MatchType


def align_with_gaps(
    master_keyposes: List[Dict],
    student_boundaries: List[MovementBoundary],
) -> List[AlignmentPair]:
    """Align student boundaries to master movements, inserting gap markers.

    Matching is done by movement_number: a student boundary matches master
    movement N if boundary.movement_number == N (as assigned by
    segment_student_video).  Each student boundary can be claimed at most once.

    Args:
        master_keyposes: List of master keypose dicts (from load_master_keyposes).
            Used only to enumerate the 19 expected movement indices.
        student_boundaries: Output of segment_student_video().  May have
            fewer than 19 entries.

    Returns:
        List of AlignmentPair sorted by master_movement (None-last for extras).
        Contains exactly one entry per master movement (matched or skipped)
        plus one entry per extra student boundary.
    """
    master_indices = sorted(kp["movement_index"] for kp in master_keyposes)

    # Build lookup: movement_number → student boundary
    student_map: Dict[int, MovementBoundary] = {
        b.movement_number: b for b in student_boundaries
    }

    pairs: List[AlignmentPair] = []
    claimed: set = set()

    # One entry for each of the 19 master movements
    for mov in master_indices:
        if mov in student_map:
            claimed.add(mov)
            pairs.append(
                AlignmentPair(
                    master_movement=mov,
                    student_boundary=student_map[mov],
                    match_type="matched",
                )
            )
        else:
            pairs.append(
                AlignmentPair(
                    master_movement=mov,
                    student_boundary=None,
                    match_type="skipped",
                )
            )

    # Remaining student boundaries not claimed by any master movement
    for b in student_boundaries:
        if b.movement_number not in claimed:
            pairs.append(
                AlignmentPair(
                    master_movement=None,
                    student_boundary=b,
                    match_type="extra",
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validate_alignment().

    Attributes:
        valid: False if the alignment is too poor for reliable feedback.
        message: English message shown to the user (empty string when clean).
        skipped_movements: Movement numbers that were not detected.
        extra_count: Number of extra (unmatched) student boundaries.
    """

    valid: bool
    message: str  # English message shown to the user (empty string when clean).
    skipped_movements: List[int]
    extra_count: int


def validate_alignment(alignment: List[AlignmentPair]) -> ValidationResult:
    """Validate an alignment and produce a user-facing English message.

    Rules (checked in priority order):
    1. skipped > 3  → valid=False, hard failure
    2. extra  > 2   → valid=False, hard failure
    3. low confidence > 5 → valid=True, soft warning
    4. otherwise    → valid=True, empty message

    Args:
        alignment: Output of align_with_gaps().

    Returns:
        ValidationResult with an English message for the student.
    """
    skipped = [
        p.master_movement
        for p in alignment
        if p.match_type == "skipped" and p.master_movement is not None
    ]
    extra_count = sum(1 for p in alignment if p.match_type == "extra")
    low_conf = sum(
        1
        for p in alignment
        if p.student_boundary is not None and p.student_boundary.confidence == "low"
    )

    if len(skipped) > 3:
        return ValidationResult(
            valid=False,
            message=(
                f"{len(skipped)} movements were not detected. "
                "Please make your movements larger and clearer."
            ),
            skipped_movements=sorted(skipped),
            extra_count=extra_count,
        )

    if extra_count > 2:
        return ValidationResult(
            valid=False,
            message="Duplicate movements detected. Please perform each movement exactly once.",
            skipped_movements=sorted(skipped),
            extra_count=extra_count,
        )

    if low_conf > 5:
        return ValidationResult(
            valid=True,
            message=(
                f"{low_conf} movements are unclear. "
                "Results may not be accurate."
            ),
            skipped_movements=sorted(skipped),
            extra_count=extra_count,
        )

    return ValidationResult(
        valid=True,
        message="",
        skipped_movements=sorted(skipped),
        extra_count=extra_count,
    )


# ---------------------------------------------------------------------------
# __main__: demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    # Simulate 19 master keyposes and 18 student boundaries (movement 5 skipped)
    fake_keyposes = [{"movement_index": m} for m in range(1, 20)]
    fake_boundaries = [
        MovementBoundary(
            movement_number=m,
            frame=m * 60,
            timestamp_ms=m * 2000.0,
            confidence="high" if m % 3 != 0 else "medium",
            matched_signals=["velocity", "keypose"],
        )
        for m in range(1, 20)
        if m != 5  # simulate skipped movement 5
    ]

    pairs = align_with_gaps(fake_keyposes, fake_boundaries)

    print(f"{'Master':>7} {'Student':>8} {'Type':>10} {'Conf':>8}")
    print("-" * 40)
    for p in pairs:
        master_str = str(p.master_movement) if p.master_movement else "-"
        student_str = (
            str(p.student_boundary.movement_number) if p.student_boundary else "-"
        )
        conf_str = p.student_boundary.confidence if p.student_boundary else "-"
        print(
            f"{master_str:>7} {student_str:>8} {p.match_type:>10} {conf_str:>8}"
        )

    matched = sum(1 for p in pairs if p.match_type == "matched")
    skipped = sum(1 for p in pairs if p.match_type == "skipped")
    extra = sum(1 for p in pairs if p.match_type == "extra")
    print(f"\nmatched={matched}  skipped={skipped}  extra={extra}")
