"""Tests for validate_alignment()."""

from itf_analysis.alignment.gap_aligner import (
    AlignmentPair,
    ValidationResult,
    align_with_gaps,
    validate_alignment,
)
from itf_analysis.segmentation.segmenter import MovementBoundary


def _kp(m: int) -> dict:
    return {"movement_index": m}


def _boundary(m: int, conf: str = "high") -> MovementBoundary:
    return MovementBoundary(
        movement_number=m,
        frame=m * 60,
        timestamp_ms=m * 2000.0,
        confidence=conf,
        matched_signals=["velocity", "keypose"],
    )


def _pairs_with_skipped(skipped: set, extra: int = 0) -> list:
    """Build alignment pairs: all movements present except those in skipped."""
    keyposes = [_kp(m) for m in range(1, 20)]
    boundaries = [_boundary(m) for m in range(1, 20) if m not in skipped]
    pairs = align_with_gaps(keyposes, boundaries)
    for i in range(extra):
        pairs.append(
            AlignmentPair(master_movement=None, student_boundary=_boundary(20 + i), match_type="extra")
        )
    return pairs


def _pairs_with_low_conf(low_count: int) -> list:
    """Build alignment pairs where low_count boundaries have 'low' confidence."""
    keyposes = [_kp(m) for m in range(1, 20)]
    boundaries = [
        _boundary(m, conf="low" if m <= low_count else "high")
        for m in range(1, 20)
    ]
    return align_with_gaps(keyposes, boundaries)


class TestValidateAlignment:

    def test_clean_sequence_valid_no_message(self):
        """All matched, no skipped, no extras → valid=True, message empty."""
        pairs = _pairs_with_skipped(skipped=set())
        result = validate_alignment(pairs)
        assert result.valid is True
        assert result.message == ""
        assert result.skipped_movements == []
        assert result.extra_count == 0

    def test_3_skipped_still_valid(self):
        """Exactly 3 skipped (≤3) → still valid."""
        pairs = _pairs_with_skipped({2, 8, 15})
        result = validate_alignment(pairs)
        assert result.valid is True
        assert sorted(result.skipped_movements) == [2, 8, 15]

    def test_4_skipped_invalid(self):
        """4 skipped > 3 → valid=False with English message."""
        pairs = _pairs_with_skipped({1, 5, 10, 15})
        result = validate_alignment(pairs)
        assert result.valid is False
        assert "4 movements" in result.message
        assert "not detected" in result.message
        assert sorted(result.skipped_movements) == [1, 5, 10, 15]

    def test_2_extra_still_valid(self):
        """Exactly 2 extras (≤2) → still valid."""
        pairs = _pairs_with_skipped(skipped=set(), extra=2)
        result = validate_alignment(pairs)
        assert result.valid is True
        assert result.extra_count == 2

    def test_3_extra_invalid(self):
        """3 extras > 2 → valid=False with English message."""
        pairs = _pairs_with_skipped(skipped=set(), extra=3)
        result = validate_alignment(pairs)
        assert result.valid is False
        assert "Duplicate movements" in result.message
        assert result.extra_count == 3

    def test_skipped_takes_priority_over_extra(self):
        """4 skipped + 3 extra → skipped rule fires first (valid=False, skipped message)."""
        pairs = _pairs_with_skipped({1, 2, 3, 4}, extra=3)
        result = validate_alignment(pairs)
        assert result.valid is False
        assert "not detected" in result.message

    def test_6_low_confidence_soft_warning(self):
        """6 low-confidence (>5) → valid=True with soft warning message."""
        pairs = _pairs_with_low_conf(6)
        result = validate_alignment(pairs)
        assert result.valid is True
        assert "6 movements" in result.message
        assert "unclear" in result.message

    def test_5_low_confidence_no_warning(self):
        """Exactly 5 low-confidence (≤5) → no warning message."""
        pairs = _pairs_with_low_conf(5)
        result = validate_alignment(pairs)
        assert result.valid is True
        assert result.message == ""

    def test_returns_validation_result_type(self):
        pairs = _pairs_with_skipped(set())
        result = validate_alignment(pairs)
        assert isinstance(result, ValidationResult)
