/// Layer 4: Match student boundaries to master movements with gap handling.
///
/// Mirrors itf_analysis/alignment/gap_aligner.py exactly.
library;

import '../segmentation/segmenter.dart';

enum MatchType { matched, skipped, extra }

/// Pairing between one master movement and one student boundary.
class AlignmentPair {
  final int? masterMovement;
  final MovementBoundary? studentBoundary;
  final MatchType matchType;

  const AlignmentPair({
    required this.masterMovement,
    required this.studentBoundary,
    required this.matchType,
  });
}

/// Alignment validation result.
class ValidationResult {
  final bool valid;
  final String message;
  final List<int> skippedMovements;
  final int extraCount;

  const ValidationResult({
    required this.valid,
    required this.message,
    required this.skippedMovements,
    required this.extraCount,
  });
}

/// Align student boundaries to master movements, inserting gap markers.
///
/// Matching by movement_number: student boundary N matches master movement N.
List<AlignmentPair> alignWithGaps(
  List<int> masterMovementIndices,
  List<MovementBoundary> studentBoundaries,
) {
  final sorted = List<int>.from(masterMovementIndices)..sort();
  final studentMap = {for (final b in studentBoundaries) b.movementNumber: b};

  final pairs = <AlignmentPair>[];
  final claimed = <int>{};

  for (final mov in sorted) {
    if (studentMap.containsKey(mov)) {
      claimed.add(mov);
      pairs.add(AlignmentPair(
        masterMovement: mov,
        studentBoundary: studentMap[mov],
        matchType: MatchType.matched,
      ));
    } else {
      pairs.add(AlignmentPair(
        masterMovement: mov,
        studentBoundary: null,
        matchType: MatchType.skipped,
      ));
    }
  }

  for (final b in studentBoundaries) {
    if (!claimed.contains(b.movementNumber)) {
      pairs.add(AlignmentPair(
        masterMovement: null,
        studentBoundary: b,
        matchType: MatchType.extra,
      ));
    }
  }

  return pairs;
}

/// Validate an alignment and produce a user-facing message.
///
/// Rules (priority order):
/// 1. skipped > 3 → valid=false, hard failure
/// 2. extra  > 2 → valid=false, hard failure
/// 3. low confidence > 5 → valid=true, soft warning
/// 4. otherwise → valid=true, empty message
ValidationResult validateAlignment(List<AlignmentPair> alignment) {
  final skipped = alignment
      .where((p) =>
          p.matchType == MatchType.skipped && p.masterMovement != null)
      .map((p) => p.masterMovement!)
      .toList()
    ..sort();

  final extraCount =
      alignment.where((p) => p.matchType == MatchType.extra).length;

  final lowConf = alignment
      .where((p) =>
          p.studentBoundary != null &&
          p.studentBoundary!.confidence == Confidence.low)
      .length;

  if (skipped.length > 3) {
    return ValidationResult(
      valid: false,
      message: '${skipped.length} movements were not detected. '
          'Please make your movements larger and clearer.',
      skippedMovements: skipped,
      extraCount: extraCount,
    );
  }

  if (extraCount > 2) {
    return ValidationResult(
      valid: false,
      message:
          'Duplicate movements detected. Please perform each movement exactly once.',
      skippedMovements: skipped,
      extraCount: extraCount,
    );
  }

  if (lowConf > 5) {
    return ValidationResult(
      valid: true,
      message: '$lowConf movements are unclear. Results may not be accurate.',
      skippedMovements: skipped,
      extraCount: extraCount,
    );
  }

  return ValidationResult(
    valid: true,
    message: '',
    skippedMovements: skipped,
    extraCount: extraCount,
  );
}
