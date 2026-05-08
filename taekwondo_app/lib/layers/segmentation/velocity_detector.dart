/// Layer 3 velocity signal: angle-based motion velocity and valley detection.
///
/// Mirrors itf_analysis/segmentation/velocity_detector.py exactly.
library;

import 'dart:math';

/// Absolute angle difference with circular wrap for line angles.
double _circularAbsDiff(double a, double b, String key) {
  double diff = b - a;
  if (key.contains('line_angle')) {
    diff = (diff + 180.0) % 360.0 - 180.0;
  }
  return diff.abs();
}

/// Compute frame-to-frame angular velocity from joint angle time series.
///
/// Returns a list of length N-1 where N = frameAngles.length.
/// Element i = sum of absolute angle differences across all common keys
/// between frame i and frame i+1.
List<double> computeMotionVelocity(
  List<(int, Map<String, double>)> frameAngles,
  double fps,
) {
  if (frameAngles.length < 2) return [];

  final velocity = <double>[];
  for (int i = 0; i < frameAngles.length - 1; i++) {
    final a1 = frameAngles[i].$2;
    final a2 = frameAngles[i + 1].$2;
    final common = a1.keys.toSet().intersection(a2.keys.toSet());
    if (common.isEmpty) {
      velocity.add(0.0);
      continue;
    }
    double total = 0.0;
    int count = 0;
    for (final k in common) {
      final v1 = a1[k]!, v2 = a2[k]!;
      if (v1.isNaN || v2.isNaN) continue;
      total += _circularAbsDiff(v1, v2, k);
      count++;
    }
    // count==0 means all shared angles were NaN → mark as unreliable
    velocity.add(count > 0 ? total : double.nan);
  }
  return velocity;
}

/// Apply Gaussian smoothing to a velocity signal.
///
/// [sigmaSeconds]: Gaussian standard deviation in seconds (default 0.1 s).
/// Returns a list of the same length as [velocity].
List<double> smoothVelocity(
  List<double> velocity,
  double fps, {
  double sigmaSeconds = 0.1,
}) {
  if (velocity.isEmpty) return [];
  final sigmaFrames = sigmaSeconds * fps;
  if (sigmaFrames < 0.1) return List.of(velocity);
  return _gaussianFilter1d(velocity, sigmaFrames);
}

/// 1-D Gaussian filter (mirrors scipy.ndimage.gaussian_filter1d, reflect mode).
List<double> _gaussianFilter1d(List<double> signal, double sigma) {
  // Kernel half-width: 3σ rounded up, minimum 1.
  final halfW = max(1, (3 * sigma).ceil());
  final kernelSize = 2 * halfW + 1;

  // Build normalised Gaussian kernel.
  final kernel = List<double>.generate(kernelSize, (i) {
    final x = (i - halfW).toDouble();
    return exp(-0.5 * x * x / (sigma * sigma));
  });
  final kSum = kernel.fold(0.0, (s, v) => s + v);
  for (int i = 0; i < kernel.length; i++) {
    kernel[i] /= kSum;
  }

  final n = signal.length;
  final out = List<double>.filled(n, 0.0);

  for (int i = 0; i < n; i++) {
    double acc = 0.0;
    for (int j = 0; j < kernelSize; j++) {
      // Reflect padding: mirror at boundaries.
      int si = i + j - halfW;
      if (si < 0) si = -si - 1;
      if (si >= n) si = 2 * n - si - 1;
      si = si.clamp(0, n - 1);
      acc += signal[si] * kernel[j];
    }
    out[i] = acc;
  }
  return out;
}

/// Find local minima (valleys) in the smoothed velocity signal.
///
/// A valley at position i is strictly less than all neighbours within
/// [order] positions on each side (mirrors scipy.signal.argrelmin).
///
/// [minGapSec]: minimum gap between consecutive valleys in seconds →
/// converted to order = round(minGapSec * fps).
///
/// Returns sorted list of indices into [smoothed].
List<int> findVelocityValleys(
  List<double> smoothed,
  double fps, {
  double minGapSec = 0.5,
}) {
  if (smoothed.length < 3) return [];
  final order = max(1, (minGapSec * fps).round());
  final n = smoothed.length;
  final valleys = <int>[];

  for (int i = order; i < n - order; i++) {
    final v = smoothed[i];
    if (v.isNaN || v.isInfinite) continue;
    bool isMin = true;
    for (int k = 1; k <= order; k++) {
      final prev = smoothed[i - k];
      final next = smoothed[i + k];
      if (prev.isNaN || next.isNaN) { isMin = false; break; }
      if (v >= prev || v >= next) { isMin = false; break; }
    }
    if (isMin) valleys.add(i);
  }
  return valleys;
}
