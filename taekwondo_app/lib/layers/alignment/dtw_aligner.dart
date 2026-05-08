/// Layer 4 DTW: temporal alignment of matched movement segments.
///
/// Mirrors itf_analysis/alignment/dtw_aligner.py.
/// DTW is implemented from scratch with a Sakoe-Chiba band constraint
/// for mobile performance (no fastdtw dependency needed).
library;

import 'dart:math';

import '../segmentation/segmenter.dart';

typedef FrameAngles = List<(int, Map<String, double>)>;

const _circularKeys = {'shoulder_line_angle', 'hip_line_angle'};

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

double _circularDiff(double a, double b) =>
    ((a - b + 180.0) % 360.0 - 180.0).abs();

/// RMS angle distance between two angle dicts.
double rmsAngleDistance(Map<String, double> a, Map<String, double> b) {
  final common = a.keys.toSet().intersection(b.keys.toSet());
  if (common.isEmpty) return 0.0;
  double sq = 0.0;
  int count = 0;
  for (final k in common) {
    final av = a[k]!, bv = b[k]!;
    if (av.isNaN || bv.isNaN) continue;
    final d = _circularKeys.contains(k)
        ? _circularDiff(av, bv)
        : (av - bv).abs();
    sq += d * d;
    count++;
  }
  return count == 0 ? 0.0 : sqrt(sq / count);
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

class FramePair {
  final int masterFrame;
  final int studentFrame;
  final Map<String, double> masterAngles;
  final Map<String, double> studentAngles;
  final Map<String, double> angleDiffs;
  final Map<String, double> angleDelta;
  final double rmsDiff;

  const FramePair({
    required this.masterFrame,
    required this.studentFrame,
    required this.masterAngles,
    required this.studentAngles,
    required this.angleDiffs,
    required this.angleDelta,
    required this.rmsDiff,
  });
}

class MovementAlignment {
  final int movementNumber;
  final List<(int, int)> path;
  final List<FramePair> framePairs;
  final double meanRms;
  final double maxRms;

  const MovementAlignment({
    required this.movementNumber,
    this.path = const [],
    this.framePairs = const [],
    this.meanRms = 0.0,
    this.maxRms = 0.0,
  });
}

// ---------------------------------------------------------------------------
// Segment builder
// ---------------------------------------------------------------------------

/// Split a full frame-angle sequence into per-movement sub-sequences.
Map<int, FrameAngles> buildMovementSegments(
  FrameAngles allFrameAngles,
  List<MovementBoundary> boundaries,
) {
  if (boundaries.isEmpty || allFrameAngles.isEmpty) return {};

  final angleDict = {for (final (fi, fa) in allFrameAngles) fi: fa};
  final frameList = angleDict.keys.toList()..sort();
  final sortedBounds = List<MovementBoundary>.from(boundaries)
    ..sort((a, b) => a.movementNumber.compareTo(b.movementNumber));

  final segments = <int, FrameAngles>{};
  int prevEnd = frameList.first - 1;

  for (final b in sortedBounds) {
    final seg = frameList
        .where((fi) => fi > prevEnd && fi <= b.frame)
        .map((fi) => (fi, angleDict[fi]!))
        .toList();
    segments[b.movementNumber] = seg;
    prevEnd = b.frame;
  }

  return segments;
}

// ---------------------------------------------------------------------------
// DTW
// ---------------------------------------------------------------------------

/// DTW warping path with Sakoe-Chiba band (radius = bandRadius frames).
///
/// Returns list of (masterIdx, studentIdx) pairs.
List<(int, int)> _dtw(
  List<Map<String, double>> mAngles,
  List<Map<String, double>> sAngles, {
  int bandRadius = 10,
}) {
  final m = mAngles.length;
  final s = sAngles.length;
  const inf = double.infinity;

  // Cost matrix (m × s), initialised to infinity.
  final cost = List.generate(m, (_) => List<double>.filled(s, inf));

  cost[0][0] = rmsAngleDistance(mAngles[0], sAngles[0]);

  for (int i = 0; i < m; i++) {
    // Sakoe-Chiba: |i - j| <= bandRadius
    final jLo = max(0, i - bandRadius);
    final jHi = min(s - 1, i + bandRadius);
    for (int j = jLo; j <= jHi; j++) {
      final d = rmsAngleDistance(mAngles[i], sAngles[j]);
      if (i == 0 && j == 0) {
        cost[i][j] = d;
        continue;
      }
      double best = inf;
      if (i > 0 && cost[i - 1][j] < best) best = cost[i - 1][j];
      if (j > 0 && cost[i][j - 1] < best) best = cost[i][j - 1];
      if (i > 0 && j > 0 && cost[i - 1][j - 1] < best) {
        best = cost[i - 1][j - 1];
      }
      cost[i][j] = d + best;
    }
  }

  // Traceback from (m-1, s-1) to (0, 0).
  final path = <(int, int)>[];
  int i = m - 1, j = s - 1;
  while (i > 0 || j > 0) {
    path.add((i, j));
    if (i == 0) {
      j--;
    } else if (j == 0) {
      i--;
    } else {
      final diag = cost[i - 1][j - 1];
      final left = cost[i][j - 1];
      final up   = cost[i - 1][j];
      if (diag <= left && diag <= up) {
        i--;
        j--;
      } else if (left < up) {
        j--;
      } else {
        i--;
      }
    }
  }
  path.add((0, 0));
  return path.reversed.toList();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Align one master movement segment to the corresponding student segment.
MovementAlignment alignMovement(
  FrameAngles masterSegment,
  FrameAngles studentSegment,
  int movementNumber, {
  int bandRadius = 10,
}) {
  if (masterSegment.isEmpty || studentSegment.isEmpty) {
    return MovementAlignment(movementNumber: movementNumber);
  }

  final mFrames  = masterSegment.map((e) => e.$1).toList();
  final sFrames  = studentSegment.map((e) => e.$1).toList();
  final mAngles  = masterSegment.map((e) => e.$2).toList();
  final sAngles  = studentSegment.map((e) => e.$2).toList();

  final rawPath = _dtw(mAngles, sAngles, bandRadius: bandRadius);

  final framePairs = <FramePair>[];
  for (final (mi, si) in rawPath) {
    final ma = mAngles[mi];
    final sa = sAngles[si];
    final common = ma.keys.toSet().intersection(sa.keys.toSet());
    final diffs  = <String, double>{};
    final deltas = <String, double>{};

    for (final k in common) {
      final mv = ma[k]!, sv = sa[k]!;
      if (mv.isNaN || sv.isNaN) continue;
      final delta = _circularKeys.contains(k)
          ? (sv - mv + 180.0) % 360.0 - 180.0
          : sv - mv;
      diffs[k]  = delta.abs();
      deltas[k] = delta;
    }

    final rms = diffs.isEmpty
        ? 0.0
        : sqrt(diffs.values.fold(0.0, (s, v) => s + v * v) / diffs.length);

    framePairs.add(FramePair(
      masterFrame:   mFrames[mi],
      studentFrame:  sFrames[si],
      masterAngles:  {for (final k in common) k: ma[k]!},
      studentAngles: {for (final k in common) k: sa[k]!},
      angleDiffs:    diffs,
      angleDelta:    deltas,
      rmsDiff:       rms,
    ));
  }

  final rmsValues = framePairs.map((fp) => fp.rmsDiff)
      .where((v) => !v.isNaN && !v.isInfinite)
      .toList();
  final meanRms = rmsValues.isEmpty
      ? 0.0
      : rmsValues.fold(0.0, (s, v) => s + v) / rmsValues.length;
  final maxRms = rmsValues.isEmpty ? 0.0 : rmsValues.reduce(max);

  return MovementAlignment(
    movementNumber: movementNumber,
    path:           rawPath,
    framePairs:     framePairs,
    meanRms:        meanRms,
    maxRms:         maxRms,
  );
}

/// Align all movements present in both master and student segment maps.
Map<int, MovementAlignment> alignAllMovements(
  Map<int, FrameAngles> masterSegments,
  Map<int, FrameAngles> studentSegments, {
  int bandRadius = 10,
}) {
  final common = masterSegments.keys
      .toSet()
      .intersection(studentSegments.keys.toSet())
    ..toList();

  return {
    for (final mov in common)
      mov: alignMovement(
        masterSegments[mov]!,
        studentSegments[mov]!,
        mov,
        bandRadius: bandRadius,
      ),
  };
}
