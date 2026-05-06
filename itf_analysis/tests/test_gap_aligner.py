"""Tests for alignment/gap_aligner.py."""

from itf_analysis.alignment.gap_aligner import AlignmentPair, align_with_gaps
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


class TestAlignWithGaps:

    def test_perfect_match_all_matched(self):
        """All 19 movements present → 19 'matched' pairs, no skipped or extra."""
        keyposes = [_kp(m) for m in range(1, 20)]
        boundaries = [_boundary(m) for m in range(1, 20)]
        pairs = align_with_gaps(keyposes, boundaries)

        assert len(pairs) == 19
        assert all(p.match_type == "matched" for p in pairs)
        assert all(p.student_boundary is not None for p in pairs)
        assert all(p.master_movement is not None for p in pairs)

    def test_movement_5_skipped(self):
        """Student missing movement 5 → pair for master 5 is 'skipped'."""
        keyposes = [_kp(m) for m in range(1, 20)]
        boundaries = [_boundary(m) for m in range(1, 20) if m != 5]
        pairs = align_with_gaps(keyposes, boundaries)

        # Still exactly 19 entries (no extras)
        assert len(pairs) == 19

        skipped = [p for p in pairs if p.match_type == "skipped"]
        assert len(skipped) == 1
        assert skipped[0].master_movement == 5
        assert skipped[0].student_boundary is None

        matched = [p for p in pairs if p.match_type == "matched"]
        assert len(matched) == 18

    def test_extra_boundary(self):
        """Student boundary with movement_number not in master → 'extra' pair."""
        keyposes = [_kp(m) for m in range(1, 20)]
        boundaries = [_boundary(m) for m in range(1, 20)]
        # Add a boundary for movement 20 (not in master)
        boundaries.append(_boundary(20))
        pairs = align_with_gaps(keyposes, boundaries)

        assert len(pairs) == 20  # 19 master + 1 extra
        extra = [p for p in pairs if p.match_type == "extra"]
        assert len(extra) == 1
        assert extra[0].master_movement is None
        assert extra[0].student_boundary.movement_number == 20

    def test_multiple_skipped(self):
        """Three movements skipped → three 'skipped' entries."""
        keyposes = [_kp(m) for m in range(1, 20)]
        skipped_set = {3, 10, 17}
        boundaries = [_boundary(m) for m in range(1, 20) if m not in skipped_set]
        pairs = align_with_gaps(keyposes, boundaries)

        assert len(pairs) == 19
        skipped = [p for p in pairs if p.match_type == "skipped"]
        assert {p.master_movement for p in skipped} == skipped_set

    def test_output_order_master_movements_first(self):
        """Matched/skipped entries come before extra entries."""
        keyposes = [_kp(m) for m in range(1, 20)]
        boundaries = [_boundary(m) for m in range(1, 20)]
        boundaries.append(_boundary(99))  # extra
        pairs = align_with_gaps(keyposes, boundaries)

        extra_seen = False
        for p in pairs:
            if p.match_type == "extra":
                extra_seen = True
            else:
                assert not extra_seen, "Non-extra pair found after extra pair"

    def test_empty_student_boundaries_all_skipped(self):
        """No student boundaries → all 19 master movements are skipped."""
        keyposes = [_kp(m) for m in range(1, 20)]
        pairs = align_with_gaps(keyposes, [])

        assert len(pairs) == 19
        assert all(p.match_type == "skipped" for p in pairs)
        assert all(p.student_boundary is None for p in pairs)

    def test_empty_master_keyposes_all_extra(self):
        """No master keyposes → all student boundaries are extra."""
        boundaries = [_boundary(m) for m in range(1, 5)]
        pairs = align_with_gaps([], boundaries)

        assert len(pairs) == 4
        assert all(p.match_type == "extra" for p in pairs)
