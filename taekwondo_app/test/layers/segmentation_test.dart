import 'package:flutter_test/flutter_test.dart';
import 'package:taekwondo_app/layers/segmentation/velocity_detector.dart';
import 'package:taekwondo_app/layers/segmentation/keypose_matcher.dart';
import 'package:taekwondo_app/layers/segmentation/segmenter.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a flat (constant velocity) frame angle sequence.
List<(int, Map<String, double>)> _flatSequence(int n) => List.generate(
      n,
      (i) => (i, {'right_knee': 120.0, 'left_knee': 120.0}),
    );

// ---------------------------------------------------------------------------
// velocity_detector
// ---------------------------------------------------------------------------

void main() {
  group('computeMotionVelocity', () {
    test('returns empty for fewer than 2 frames', () {
      expect(computeMotionVelocity([], 30), isEmpty);
      expect(computeMotionVelocity([_flatSequence(1).first], 30), isEmpty);
    });

    test('identical consecutive frames give velocity 0', () {
      final seq = _flatSequence(5);
      final v = computeMotionVelocity(seq, 30);
      expect(v.length, 4);
      for (final val in v) {
        expect(val, closeTo(0.0, 1e-9));
      }
    });

    test('output length is N-1', () {
      final seq = _flatSequence(10);
      expect(computeMotionVelocity(seq, 30).length, 9);
    });

    test('changing angles give non-zero velocity', () {
      final seq = [
        (0, {'right_knee': 90.0}),
        (1, {'right_knee': 130.0}),
      ];
      final v = computeMotionVelocity(seq, 30);
      expect(v.first, closeTo(40.0, 1e-6));
    });

    test('line_angle uses circular arithmetic (wraps at 180)', () {
      // Diff across 180 boundary: 175 → -175 should give |diff|=10, not 350.
      final seq = [
        (0, {'shoulder_line_angle': 175.0}),
        (1, {'shoulder_line_angle': -175.0}),
      ];
      final v = computeMotionVelocity(seq, 30);
      expect(v.first, closeTo(10.0, 1e-6));
    });

    test('no common keys gives velocity 0', () {
      final seq = [
        (0, {'right_knee': 90.0}),
        (1, {'left_knee': 130.0}),
      ];
      expect(computeMotionVelocity(seq, 30).first, closeTo(0.0, 1e-9));
    });
  });

  group('smoothVelocity', () {
    test('empty input returns empty', () {
      expect(smoothVelocity([], 30), isEmpty);
    });

    test('output length equals input length', () {
      final v = List<double>.generate(20, (i) => i.toDouble());
      expect(smoothVelocity(v, 30).length, 20);
    });

    test('constant signal is unchanged after smoothing', () {
      final v = List<double>.filled(30, 5.0);
      final s = smoothVelocity(v, 30);
      for (final val in s) {
        expect(val, closeTo(5.0, 1e-6));
      }
    });
  });

  group('findVelocityValleys', () {
    test('returns empty for short signal', () {
      expect(findVelocityValleys([1.0, 0.5], 30), isEmpty);
    });

    test('finds valley at known position', () {
      // 0: high, 1: high, 2: low (valley), 3: high, 4: high ... repeated.
      final signal = [5.0, 4.0, 1.0, 4.0, 5.0, 4.0, 1.0, 4.0, 5.0];
      final valleys = findVelocityValleys(signal, 30, minGapSec: 0.03);
      expect(valleys, contains(2));
      expect(valleys, contains(6));
    });

    test('flat signal has no valleys', () {
      final signal = List<double>.filled(20, 3.0);
      expect(findVelocityValleys(signal, 30), isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // keypose_matcher
  // ---------------------------------------------------------------------------

  group('keyposeDistance', () {
    test('identical angles give distance 0', () {
      final a = {'right_knee': 120.0, 'left_knee': 90.0};
      expect(keyposeDistance(a, a), closeTo(0.0, 1e-9));
    });

    test('no common keys gives infinity', () {
      expect(
        keyposeDistance({'right_knee': 90.0}, {'left_knee': 90.0}),
        equals(double.infinity),
      );
    });

    test('line_angle uses circular arithmetic', () {
      final a = {'shoulder_line_angle': 175.0};
      final b = {'shoulder_line_angle': -175.0};
      // Circular diff = 10°, RMS = 10.
      expect(keyposeDistance(a, b), closeTo(10.0, 1e-4));
    });
  });

  group('loadMasterKeyposes', () {
    test('returns empty list for invalid JSON', () {
      expect(loadMasterKeyposes('not json'), isEmpty);
    });

    test('parses valid JSON and sorts by movement_index', () {
      const json = '''
      {
        "poomsae": "chon_ji",
        "keyposes": [
          {"movement_index": 2, "movement_name": "b", "itf_name": "B",
           "source_frame": 200, "angles": {"right_knee": 110.0}},
          {"movement_index": 1, "movement_name": "a", "itf_name": "A",
           "source_frame": 100, "angles": {"right_knee": 120.0}}
        ]
      }
      ''';
      final kps = loadMasterKeyposes(json);
      expect(kps.length, 2);
      expect(kps[0].movementIndex, 1);
      expect(kps[1].movementIndex, 2);
      expect(kps[0].angles['right_knee'], closeTo(120.0, 1e-9));
    });
  });

  // ---------------------------------------------------------------------------
  // segmenter
  // ---------------------------------------------------------------------------

  group('segmentMovements', () {
    test('returns empty for fewer than 2 frames', () {
      expect(segmentMovements([], [], 30), isEmpty);
    });

    test('returns exactly numMovements boundaries', () {
      final seq = _flatSequence(200);
      final result = segmentMovements(seq, [], 30, numMovements: 5);
      expect(result.length, 5);
    });

    test('boundary frames are strictly monotonic', () {
      final seq = _flatSequence(200);
      final result = segmentMovements(seq, [], 30, numMovements: 10);
      for (int i = 1; i < result.length; i++) {
        expect(result[i].frame, greaterThan(result[i - 1].frame));
      }
    });

    test('all frames fall within valid range', () {
      final seq = _flatSequence(300);
      final result = segmentMovements(seq, [], 30, numMovements: 10);
      for (final b in result) {
        expect(b.frame, greaterThanOrEqualTo(0));
        expect(b.frame, lessThan(300));
      }
    });

    test('timestampMs = frame / fps * 1000', () {
      final seq = _flatSequence(200);
      final result = segmentMovements(seq, [], 30.0, numMovements: 5);
      for (final b in result) {
        expect(b.timestampMs, closeTo(b.frame / 30.0 * 1000.0, 1e-6));
      }
    });
  });

  group('segmentStudentVideo', () {
    test('returns empty when masterKeyposes is empty', () {
      expect(segmentStudentVideo(_flatSequence(100), [], 30), isEmpty);
    });

    test('returns empty for fewer than 2 frames', () {
      expect(segmentStudentVideo([], [], 30), isEmpty);
    });
  });
}
