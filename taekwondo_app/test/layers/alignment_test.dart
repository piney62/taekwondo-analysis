import 'package:flutter_test/flutter_test.dart';
import 'package:taekwondo_app/layers/segmentation/segmenter.dart';
import 'package:taekwondo_app/layers/alignment/gap_aligner.dart';
import 'package:taekwondo_app/layers/alignment/dtw_aligner.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

MovementBoundary _boundary(int mov, {Confidence conf = Confidence.high}) =>
    MovementBoundary(
      movementNumber: mov,
      frame: mov * 60,
      timestampMs: mov * 2000.0,
      confidence: conf,
      matchedSignals: const ['velocity', 'keypose'],
    );

List<(int, Map<String, double>)> _simpleSegment(int startFrame, int len) =>
    List.generate(
      len,
      (i) => (startFrame + i, {'right_knee': 120.0 + i, 'left_knee': 90.0}),
    );

// ---------------------------------------------------------------------------
// gap_aligner
// ---------------------------------------------------------------------------

void main() {
  group('alignWithGaps', () {
    test('all movements matched when all boundaries present', () {
      final master = List.generate(5, (i) => i + 1);
      final student = master.map((m) => _boundary(m)).toList();
      final pairs = alignWithGaps(master, student);
      expect(pairs.where((p) => p.matchType == MatchType.matched).length, 5);
      expect(pairs.where((p) => p.matchType == MatchType.skipped).length, 0);
    });

    test('skipped when student boundary missing', () {
      final master = [1, 2, 3, 4, 5];
      final student = [_boundary(1), _boundary(2), _boundary(4), _boundary(5)];
      final pairs = alignWithGaps(master, student);
      final skipped = pairs.where((p) => p.matchType == MatchType.skipped);
      expect(skipped.length, 1);
      expect(skipped.first.masterMovement, 3);
    });

    test('extra when student has unmatched boundary', () {
      final master = [1, 2, 3];
      final student = [_boundary(1), _boundary(2), _boundary(3), _boundary(7)];
      final pairs = alignWithGaps(master, student);
      final extra = pairs.where((p) => p.matchType == MatchType.extra);
      expect(extra.length, 1);
      expect(extra.first.studentBoundary!.movementNumber, 7);
    });
  });

  group('validateAlignment', () {
    test('valid with empty message when all matched and confident', () {
      final master = List.generate(5, (i) => i + 1);
      final student = master.map((m) => _boundary(m)).toList();
      final result = validateAlignment(alignWithGaps(master, student));
      expect(result.valid, isTrue);
      expect(result.message, isEmpty);
    });

    test('invalid when skipped > 3', () {
      final master = List.generate(19, (i) => i + 1);
      // Only 14 boundaries — 5 skipped
      final student = [for (int m = 1; m <= 14; m++) _boundary(m)];
      final result = validateAlignment(alignWithGaps(master, student));
      expect(result.valid, isFalse);
      expect(result.message, contains('were not detected'));
    });

    test('invalid when extra > 2', () {
      final master = [1, 2, 3];
      final student = [
        _boundary(1), _boundary(2), _boundary(3),
        _boundary(10), _boundary(11), _boundary(12),
      ];
      final result = validateAlignment(alignWithGaps(master, student));
      expect(result.valid, isFalse);
      expect(result.message, contains('Duplicate'));
    });

    test('soft warning when low confidence > 5', () {
      final master = List.generate(19, (i) => i + 1);
      final student = [
        for (int m = 1; m <= 19; m++)
          _boundary(m, conf: m <= 6 ? Confidence.low : Confidence.high),
      ];
      final result = validateAlignment(alignWithGaps(master, student));
      expect(result.valid, isTrue);
      expect(result.message, contains('unclear'));
    });
  });

  // ---------------------------------------------------------------------------
  // dtw_aligner
  // ---------------------------------------------------------------------------

  group('buildMovementSegments', () {
    test('returns empty map when boundaries are empty', () {
      final angles = _simpleSegment(0, 10);
      expect(buildMovementSegments(angles, []), isEmpty);
    });

    test('segments cover correct frame ranges', () {
      final angles = _simpleSegment(0, 100);
      final bounds = [_boundary(1), _boundary(2)];
      // boundary 1 frame=60, boundary 2 frame=120 (clamped to 99 by segment)
      final segs = buildMovementSegments(angles, bounds);
      expect(segs.containsKey(1), isTrue);
      expect(segs[1]!.every((e) => e.$1 <= 60), isTrue);
    });
  });

  group('alignMovement', () {
    test('returns empty alignment when segment is empty', () {
      final result = alignMovement([], [], 1);
      expect(result.framePairs, isEmpty);
      expect(result.meanRms, closeTo(0.0, 1e-9));
    });

    test('self-alignment gives near-zero RMS', () {
      final seg = _simpleSegment(0, 20);
      final result = alignMovement(seg, seg, 1);
      expect(result.meanRms, closeTo(0.0, 1e-6));
    });

    test('frame_pairs length equals path length', () {
      final seg = _simpleSegment(0, 15);
      final result = alignMovement(seg, seg, 1);
      expect(result.framePairs.length, equals(result.path.length));
    });

    test('path endpoints match (0,0) and (m-1, s-1)', () {
      final mSeg = _simpleSegment(0, 10);
      final sSeg = _simpleSegment(100, 8);
      final result = alignMovement(mSeg, sSeg, 1);
      expect(result.path.first, equals((0, 0)));
      expect(result.path.last, equals((9, 7)));
    });

    test('different segments give non-zero RMS', () {
      final mSeg = _simpleSegment(0, 10);
      // Student has different angles
      final sSeg = List.generate(
        10,
        (i) => (i + 100, {'right_knee': 90.0, 'left_knee': 60.0}),
      );
      final result = alignMovement(mSeg, sSeg, 1);
      expect(result.meanRms, greaterThan(1.0));
    });
  });

  group('alignAllMovements', () {
    test('only aligns movements present in both maps', () {
      final master = {1: _simpleSegment(0, 10), 2: _simpleSegment(60, 10)};
      final student = {1: _simpleSegment(0, 8)};
      final result = alignAllMovements(master, student);
      expect(result.containsKey(1), isTrue);
      expect(result.containsKey(2), isFalse);
    });
  });
}
