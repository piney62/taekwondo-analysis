/// Layer 3: Movement segmenter — combines velocity valley and keypose signals.
///
/// Mirrors itf_analysis/segmentation/segmenter.py exactly.
library;

import 'dart:math';

import 'keypose_matcher.dart';
import 'velocity_detector.dart';

typedef FrameAngles = List<(int, Map<String, double>)>;

/// Confidence level for a detected movement boundary.
enum Confidence { high, medium, low }

/// Detected completion boundary for one numbered movement.
class MovementBoundary {
  final int movementNumber;
  final int frame;
  final double timestampMs;
  final Confidence confidence;
  final List<String> matchedSignals;

  const MovementBoundary({
    required this.movementNumber,
    required this.frame,
    required this.timestampMs,
    required this.confidence,
    required this.matchedSignals,
  });

  @override
  String toString() =>
      'Movement $movementNumber  frame=$frame  '
      '${confidence.name}  $matchedSignals';
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

int? _findBestValleyInWindow(
  List<int> valleys,
  List<int> frameIndices,
  List<double> smoothed,
  int lo,
  int hi, {
  Map<int, Map<String, double>>? anglesLookup,
  Map<String, double>? kpAngles,
  double keyposeThreshold = 25.0,
}) {
  final inWindow = valleys.where((v) {
    final fi = frameIndices[v];
    return fi >= lo && fi <= hi;
  }).toList();

  if (inWindow.isEmpty) return null;

  if (anglesLookup != null && kpAngles != null && kpAngles.isNotEmpty) {
    final vMax = inWindow.map((v) => smoothed[v]).reduce(max);
    final effectiveMax = vMax == 0 ? 1.0 : vMax;

    double score(int v) {
      final fa = anglesLookup[frameIndices[v]];
      final normVel = smoothed[v] / effectiveMax;
      if (fa != null) {
        final normKp =
            (keyposeDistance(fa, kpAngles) / keyposeThreshold).clamp(0.0, 1.0);
        return 0.5 * normVel + 0.5 * normKp;
      }
      return normVel;
    }

    final best = inWindow.reduce((a, b) => score(a) < score(b) ? a : b);
    return frameIndices[best];
  }

  final best = inWindow.reduce(
    (a, b) => smoothed[a] < smoothed[b] ? a : b,
  );
  return frameIndices[best];
}

(int?, double) _findBestKeyposeInWindow(
  FrameAngles frameAngles,
  Map<String, double> kpAngles,
  int lo,
  int hi,
) {
  int? bestFrame;
  double bestDist = double.infinity;

  for (final (fi, fa) in frameAngles) {
    if (fi < lo || fi > hi) continue;
    final d = keyposeDistance(fa, kpAngles);
    if (d < bestDist) {
      bestDist = d;
      bestFrame = fi;
    }
  }
  return (bestFrame, bestDist);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Segment an angle sequence into [numMovements] movement completions.
///
/// Mirrors segment_movements() in segmenter.py.
List<MovementBoundary> segmentMovements(
  FrameAngles frameAngles,
  List<KeyposeEntry> masterKeyposes,
  double fps, {
  int toleranceFrames = 5,
  int numMovements = 19,
  double windowRatio = 0.45,
  double keyposeThreshold = 20.0,
  double sigmaSeconds = 0.1,
  double minGapSec = 0.5,
  int? endFrame,
  int? startFrame,
}) {
  if (frameAngles.length < 2) return [];

  final frameIndices = frameAngles.map((e) => e.$1).toList();

  final effectiveEnd = endFrame != null
      ? min(endFrame, frameIndices.last)
      : frameIndices.last;
  final effectiveStart = startFrame != null
      ? max(startFrame, frameIndices.first)
      : frameIndices.first;

  final totalRange = effectiveEnd - effectiveStart;
  final avgMov = totalRange / numMovements;
  final halfWindow = max(1, (avgMov * windowRatio).toInt());

  // source_frames from keyposes when all movements have them.
  final kpSource = <int, int>{};
  for (final kp in masterKeyposes) {
    if (kp.sourceFrame != null) kpSource[kp.movementIndex] = kp.sourceFrame!;
  }
  final useSourceFrames = kpSource.length == numMovements;
  final sourceExpected = useSourceFrames
      ? List<int>.generate(numMovements, (i) => kpSource[i + 1]!)
      : <int>[];

  final kpMap = {for (final kp in masterKeyposes) kp.movementIndex: kp.angles};
  final anglesLookup = {for (final (fi, fa) in frameAngles) fi: fa};

  final velocity = computeMotionVelocity(frameAngles, fps);
  final smoothed  = smoothVelocity(velocity, fps, sigmaSeconds: sigmaSeconds);
  final valleys   = findVelocityValleys(smoothed, fps, minGapSec: minGapSec);

  final boundaries = <MovementBoundary>[];
  int prevFrame = effectiveStart - 1;

  for (int i = 0; i < numMovements; i++) {
    final mov = i + 1;
    int exp = useSourceFrames
        ? sourceExpected[i]
        : (prevFrame + avgMov).round();

    int lo = [frameIndices.first, prevFrame + 1, exp - halfWindow].reduce(max);
    int hi = min(effectiveEnd, exp + halfWindow);
    if (hi >= effectiveEnd) lo = max(frameIndices.first, prevFrame + 1);
    if (lo > hi) lo = hi = exp;

    final kpAngles = kpMap[mov];

    final vFrame = _findBestValleyInWindow(
      valleys, frameIndices, smoothed, lo, hi,
      anglesLookup: anglesLookup,
      kpAngles: kpAngles,
      keyposeThreshold: keyposeThreshold,
    );

    int? kpFrame;
    bool kpFound = false;
    if (kpAngles != null && kpAngles.isNotEmpty) {
      final (f, dist) = _findBestKeyposeInWindow(frameAngles, kpAngles, lo, hi);
      kpFrame = f;
      kpFound = dist <= keyposeThreshold;
    }

    final matched = <String>[];
    if (vFrame != null) matched.add('velocity');
    if (kpFound) matched.add('keypose');

    Confidence confidence;
    int frame;

    if (matched.contains('velocity') && matched.contains('keypose')) {
      if ((vFrame! - kpFrame!).abs() <= toleranceFrames) {
        confidence = Confidence.high;
        frame = vFrame;
      } else {
        confidence = Confidence.medium;
        frame = vFrame;
      }
    } else if (matched.contains('velocity')) {
      confidence = Confidence.medium;
      frame = vFrame!;
    } else if (matched.contains('keypose')) {
      confidence = Confidence.medium;
      frame = kpFrame!;
    } else {
      confidence = Confidence.low;
      frame = exp;
    }

    frame = frame.clamp(lo, hi);
    if (frame <= prevFrame) frame = prevFrame + 1;

    boundaries.add(MovementBoundary(
      movementNumber: mov,
      frame: frame,
      timestampMs: frame / fps * 1000.0,
      confidence: confidence,
      matchedSignals: matched,
    ));
    prevFrame = frame;
  }

  return boundaries;
}

/// Match student velocity valleys to master keyposes.
///
/// Mirrors segment_student_video() in segmenter.py.
List<MovementBoundary> segmentStudentVideo(
  FrameAngles studentFrameAngles,
  List<KeyposeEntry> masterKeyposes,
  double fps, {
  double threshold = 25.0,
  double sigmaSeconds = 0.1,
  double minGapSec = 0.5,
}) {
  if (studentFrameAngles.length < 2 || masterKeyposes.isEmpty) return [];

  final frameIndices = studentFrameAngles.map((e) => e.$1).toList();
  final anglesLookup = {for (final (fi, fa) in studentFrameAngles) fi: fa};

  final velocity = computeMotionVelocity(studentFrameAngles, fps);
  final smoothed  = smoothVelocity(velocity, fps, sigmaSeconds: sigmaSeconds);
  final valleys   = findVelocityValleys(smoothed, fps, minGapSec: minGapSec);

  final kpList = List<KeyposeEntry>.from(masterKeyposes)
    ..sort((a, b) => a.movementIndex.compareTo(b.movementIndex));

  final matchedIndices = <int>{};
  int nextKpPos = 0;
  final boundaries = <MovementBoundary>[];

  for (final v in valleys) {
    final fi = frameIndices[v];
    final fa = anglesLookup[fi];
    if (fa == null) continue;

    KeyposeEntry? bestKp;
    double bestDist = double.infinity;
    int bestPos = -1;

    for (int kpPos = nextKpPos; kpPos < kpList.length; kpPos++) {
      final kp = kpList[kpPos];
      if (matchedIndices.contains(kp.movementIndex)) continue;
      if (kp.angles.isEmpty) continue;
      final d = keyposeDistance(fa, kp.angles);
      if (d < bestDist) {
        bestDist = d;
        bestKp = kp;
        bestPos = kpPos;
      }
    }

    if (bestKp == null || bestDist >= threshold) continue;

    matchedIndices.add(bestKp.movementIndex);
    nextKpPos = bestPos + 1;

    final conf = bestDist < threshold * 0.5
        ? Confidence.high
        : Confidence.medium;

    boundaries.add(MovementBoundary(
      movementNumber: bestKp.movementIndex,
      frame: fi,
      timestampMs: fi / fps * 1000.0,
      confidence: conf,
      matchedSignals: const ['velocity', 'keypose'],
    ));
  }

  boundaries.sort((a, b) => a.frame.compareTo(b.frame));
  return boundaries;
}
