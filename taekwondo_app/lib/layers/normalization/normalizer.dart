/// Layer 2: Coordinate normalization and joint angle extraction.
///
/// Mirrors itf_analysis/normalization/normalizer.py exactly.
library;

import 'dart:math';

/// Raw landmark from MediaPipe (or any pose detector).
class Landmark {
  final double x;
  final double y;
  final double z;
  final double visibility;

  const Landmark({
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });
}

/// Landmark in hip-centered, shoulder-width-scaled coordinate space.
class NormalizedLandmark {
  final double x;
  final double y;
  final double z;
  final double visibility;

  const NormalizedLandmark({
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });
}

/// Normalize one frame's landmarks to hip-centered, shoulder-width-scaled space.
///
/// Origin = midpoint of hips (joints 23, 24).
/// Scale  = shoulder-to-shoulder distance (joint 11 → 12) == 1.0.
///
/// Returns null if any anchor joint (11, 12, 23, 24) is missing or if
/// shoulder width is below 1e-6.
Map<int, NormalizedLandmark>? normalizePose(Map<int, Landmark> landmarks) {
  for (final idx in [11, 12, 23, 24]) {
    if (!landmarks.containsKey(idx)) return null;
  }

  final hip23 = landmarks[23]!;
  final hip24 = landmarks[24]!;
  final sh11  = landmarks[11]!;
  final sh12  = landmarks[12]!;

  final hipCx = (hip23.x + hip24.x) / 2;
  final hipCy = (hip23.y + hip24.y) / 2;
  final hipCz = (hip23.z + hip24.z) / 2;

  final sw = sqrt(
    pow(sh11.x - sh12.x, 2) +
    pow(sh11.y - sh12.y, 2) +
    pow(sh11.z - sh12.z, 2),
  );

  if (sw < 1e-6) return null;

  return landmarks.map(
    (idx, lm) => MapEntry(
      idx,
      NormalizedLandmark(
        x:          (lm.x - hipCx) / sw,
        y:          (lm.y - hipCy) / sw,
        z:          (lm.z - hipCz) / sw,
        visibility: lm.visibility,
      ),
    ),
  );
}

/// 3-D vector from b to a.
(double, double, double) _vec3(NormalizedLandmark a, NormalizedLandmark b) =>
    (a.x - b.x, a.y - b.y, a.z - b.z);

/// Angle in degrees at vertex b in the a–b–c triplet (3-D).
///
/// Returns 0.0 if either arm has near-zero length.
double _angleAtVertex(
  NormalizedLandmark a,
  NormalizedLandmark b,
  NormalizedLandmark c,
) {
  final ba = _vec3(a, b);
  final bc = _vec3(c, b);

  final dot    = ba.$1 * bc.$1 + ba.$2 * bc.$2 + ba.$3 * bc.$3;
  final magBa  = sqrt(ba.$1 * ba.$1 + ba.$2 * ba.$2 + ba.$3 * ba.$3);
  final magBc  = sqrt(bc.$1 * bc.$1 + bc.$2 * bc.$2 + bc.$3 * bc.$3);

  if (magBa < 1e-9 || magBc < 1e-9) return 0.0;

  return degrees(acos(dot / (magBa * magBc).clamp(-1.0, 1.0)));
}

/// Convert radians to degrees.
double degrees(double rad) => rad * 180.0 / pi;

/// Joint index triplets for three-point angles: (a, vertex, c).
const _threePoint = {
  'right_knee':  (24, 26, 28),
  'left_knee':   (23, 25, 27),
  'right_elbow': (12, 14, 16),
  'left_elbow':  (11, 13, 15),
  'right_hip':   (12, 24, 26),
  'left_hip':    (11, 23, 25),
};

/// Compute named joint angles from a normalized landmark map.
///
/// Three-point angles use 3-D (x, y, z).
/// Line angles (shoulder tilt, hip tilt) use 2-D (x, y) only.
/// A key is absent if any required joint is missing.
Map<String, double> extractJointAngles(Map<int, NormalizedLandmark> lm) {
  final angles = <String, double>{};

  for (final entry in _threePoint.entries) {
    final (ai, bi, ci) = entry.value;
    if (lm.containsKey(ai) && lm.containsKey(bi) && lm.containsKey(ci)) {
      final a = _angleAtVertex(lm[ai]!, lm[bi]!, lm[ci]!);
      if (!a.isNaN) angles[entry.key] = a;
    }
  }

  if (lm.containsKey(11) && lm.containsKey(12)) {
    final dx = lm[12]!.x - lm[11]!.x;
    final dy = lm[12]!.y - lm[11]!.y;
    final a = degrees(atan2(dy, dx));
    if (!a.isNaN) angles['shoulder_line_angle'] = a;
  }

  if (lm.containsKey(23) && lm.containsKey(24)) {
    final dx = lm[24]!.x - lm[23]!.x;
    final dy = lm[24]!.y - lm[23]!.y;
    final a = degrees(atan2(dy, dx));
    if (!a.isNaN) angles['hip_line_angle'] = a;
  }

  return angles;
}
