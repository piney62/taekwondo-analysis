/// Layer 3 keypose matching: angle-based distance and candidate search.
///
/// Mirrors itf_analysis/segmentation/keypose_matcher.py exactly.
library;

import 'dart:convert';
import 'dart:math';

const List<String> angleKeys = [
  'right_knee',
  'left_knee',
  'right_elbow',
  'left_elbow',
  'right_hip',
  'left_hip',
  'shoulder_line_angle',
  'hip_line_angle',
];

/// One master keypose entry loaded from keyposes_angles.json.
class KeyposeEntry {
  final int movementIndex;
  final String movementName;
  final String itfName;
  final int? sourceFrame;
  final Map<String, double> angles;

  const KeyposeEntry({
    required this.movementIndex,
    required this.movementName,
    required this.itfName,
    this.sourceFrame,
    required this.angles,
  });

  factory KeyposeEntry.fromJson(Map<String, dynamic> json) {
    final rawAngles = (json['angles'] as Map<String, dynamic>?) ?? {};
    return KeyposeEntry(
      movementIndex: json['movement_index'] as int,
      movementName:  json['movement_name']  as String? ?? '',
      itfName:       json['itf_name']        as String? ?? '',
      sourceFrame:   json['source_frame']    as int?,
      angles: rawAngles.map((k, v) => MapEntry(k, (v as num).toDouble())),
    );
  }
}

/// Load master keyposes from JSON string (read from Flutter assets).
///
/// Returns list sorted by movementIndex, or empty list if parsing fails.
List<KeyposeEntry> loadMasterKeyposes(String jsonString) {
  try {
    final data = jsonDecode(jsonString) as Map<String, dynamic>;
    final raw  = (data['keyposes'] as List<dynamic>?) ?? [];
    final list = raw
        .map((e) => KeyposeEntry.fromJson(e as Map<String, dynamic>))
        .toList();
    list.sort((a, b) => a.movementIndex.compareTo(b.movementIndex));
    return list;
  } catch (_) {
    return [];
  }
}

/// RMS angular distance between a frame and a stored keypose.
///
/// Only keys present in both maps are compared.
/// Line angles use circular arithmetic (wrap ±180°).
/// Returns double.infinity if no common keys.
double keyposeDistance(
  Map<String, double> frameAngles,
  Map<String, double> keyposeAngles, {
  List<String> keys = angleKeys,
}) {
  final squaredDiffs = <double>[];

  for (final key in keys) {
    final fa = frameAngles[key];
    final ka = keyposeAngles[key];
    if (fa == null || ka == null) continue;

    double diff = fa - ka;
    if (key.contains('line_angle')) {
      diff = (diff + 180.0) % 360.0 - 180.0;
    }
    squaredDiffs.add(diff * diff);
  }

  if (squaredDiffs.isEmpty) return double.infinity;
  return sqrt(squaredDiffs.fold(0.0, (s, v) => s + v) / squaredDiffs.length);
}
