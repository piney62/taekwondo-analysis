import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:taekwondo_app/layers/normalization/normalizer.dart';

/// Build a minimal landmark map with all 4 anchor joints + extras.
Map<int, Landmark> _makeFrame({
  double shoulderWidth = 1.0,
  Map<int, Landmark> extra = const {},
}) {
  // Hips at origin, shoulders 1 unit apart horizontally.
  final base = <int, Landmark>{
    11: Landmark(x: -shoulderWidth / 2, y: -1.0, z: 0, visibility: 1),
    12: Landmark(x:  shoulderWidth / 2, y: -1.0, z: 0, visibility: 1),
    23: Landmark(x: -0.2, y: 0.0, z: 0, visibility: 1),
    24: Landmark(x:  0.2, y: 0.0, z: 0, visibility: 1),
  };
  base.addAll(extra);
  return base;
}

void main() {
  group('normalizePose', () {
    test('returns null when anchor joint is missing', () {
      final lm = _makeFrame()..remove(23);
      expect(normalizePose(lm), isNull);
    });

    test('returns null when shoulder width is zero', () {
      final lm = _makeFrame(shoulderWidth: 0.0);
      expect(normalizePose(lm), isNull);
    });

    test('hip midpoint becomes origin (0, 0)', () {
      final lm = _makeFrame();
      final norm = normalizePose(lm)!;
      // After centering, (hip23 + hip24) / 2 should be (0, 0)
      final hipX = (norm[23]!.x + norm[24]!.x) / 2;
      final hipY = (norm[23]!.y + norm[24]!.y) / 2;
      expect(hipX, closeTo(0.0, 1e-9));
      expect(hipY, closeTo(0.0, 1e-9));
    });

    test('shoulder width after normalization equals 1.0', () {
      final lm = _makeFrame(shoulderWidth: 2.5);
      final norm = normalizePose(lm)!;
      final sw = sqrt(
        pow(norm[11]!.x - norm[12]!.x, 2) +
        pow(norm[11]!.y - norm[12]!.y, 2) +
        pow(norm[11]!.z - norm[12]!.z, 2),
      );
      expect(sw, closeTo(1.0, 1e-9));
    });

    test('visibility is preserved unchanged', () {
      final lm = _makeFrame()
        ..[11] = Landmark(x: -0.5, y: -1, z: 0, visibility: 0.75);
      final norm = normalizePose(lm)!;
      expect(norm[11]!.visibility, closeTo(0.75, 1e-9));
    });

    test('extra joints are also normalized', () {
      final lm = _makeFrame(
        extra: {13: Landmark(x: 0.3, y: -0.5, z: 0, visibility: 1)},
      );
      final norm = normalizePose(lm)!;
      expect(norm.containsKey(13), isTrue);
    });
  });

  group('extractJointAngles', () {
    /// Straight-arm: shoulder–elbow–wrist collinear → 180°.
    test('straight arm gives ~180 degrees for right_elbow', () {
      final lm = {
        12: NormalizedLandmark(x: 1.0, y: 0.0, z: 0, visibility: 1),
        14: NormalizedLandmark(x: 2.0, y: 0.0, z: 0, visibility: 1),
        16: NormalizedLandmark(x: 3.0, y: 0.0, z: 0, visibility: 1),
      };
      final angles = extractJointAngles(lm);
      expect(angles['right_elbow'], closeTo(180.0, 1e-4));
    });

    test('right-angle gives ~90 degrees for left_elbow', () {
      final lm = {
        11: NormalizedLandmark(x: 0.0, y: 0.0, z: 0, visibility: 1),
        13: NormalizedLandmark(x: 1.0, y: 0.0, z: 0, visibility: 1),
        15: NormalizedLandmark(x: 1.0, y: 1.0, z: 0, visibility: 1),
      };
      final angles = extractJointAngles(lm);
      expect(angles['left_elbow'], closeTo(90.0, 1e-4));
    });

    test('shoulder_line_angle horizontal right gives 0°', () {
      final lm = {
        11: NormalizedLandmark(x: 0.0, y: 0.0, z: 0, visibility: 1),
        12: NormalizedLandmark(x: 1.0, y: 0.0, z: 0, visibility: 1),
      };
      final angles = extractJointAngles(lm);
      expect(angles['shoulder_line_angle'], closeTo(0.0, 1e-9));
    });

    test('hip_line_angle tilted 45° gives ~45°', () {
      final lm = {
        23: NormalizedLandmark(x: 0.0, y: 0.0, z: 0, visibility: 1),
        24: NormalizedLandmark(x: 1.0, y: 1.0, z: 0, visibility: 1),
      };
      final angles = extractJointAngles(lm);
      expect(angles['hip_line_angle'], closeTo(45.0, 1e-4));
    });

    test('missing joint omits that angle key', () {
      final lm = {
        12: NormalizedLandmark(x: 1.0, y: 0.0, z: 0, visibility: 1),
        // joint 14 and 16 missing → right_elbow absent
      };
      final angles = extractJointAngles(lm);
      expect(angles.containsKey('right_elbow'), isFalse);
    });

    test('returns all 8 keys when all joints present', () {
      // Minimal landmark map covering joints 11-16 and 23-28.
      final lm = <int, NormalizedLandmark>{
        for (int i = 11; i <= 16; i++)
          i: NormalizedLandmark(x: i * 0.1, y: i * 0.05, z: 0, visibility: 1),
        for (int i = 23; i <= 28; i++)
          i: NormalizedLandmark(x: i * 0.1, y: i * 0.05, z: 0, visibility: 1),
      };
      final angles = extractJointAngles(lm);
      expect(angles.length, 8);
      for (final key in [
        'right_knee', 'left_knee', 'right_elbow', 'left_elbow',
        'right_hip', 'left_hip', 'shoulder_line_angle', 'hip_line_angle',
      ]) {
        expect(angles.containsKey(key), isTrue, reason: '$key missing');
      }
    });
  });
}
