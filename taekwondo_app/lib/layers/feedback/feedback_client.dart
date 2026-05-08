/// Layer 5: HTTP client that sends alignment data to the relay server
/// and returns per-movement natural-language feedback.
library;

import 'dart:convert';
import 'package:http/http.dart' as http;

import '../alignment/dtw_aligner.dart';
import '../segmentation/keypose_matcher.dart';

/// Base URL of the relay server.
/// Override at app startup via [FeedbackClient.relayUrl].
String _relayUrl = 'http://10.0.2.2:8000'; // Android emulator → localhost

class FeedbackClient {
  /// Set the relay URL once at app startup.
  static set relayUrl(String url) => _relayUrl = url.trimRight().replaceAll(RegExp(r'/$'), '');

  /// Send alignment results to the relay and receive per-movement feedback.
  ///
  /// Returns a map of movement_number → feedback string.
  /// Throws [FeedbackClientError] on network or server errors.
  static Future<Map<int, String>> getFeedback({
    required Map<int, MovementAlignment> alignments,
    required List<KeyposeEntry> keyposes,
    String poomsae = 'chon_ji',
  }) async {
    final body = jsonEncode({
      'poomsae': poomsae,
      'alignments': alignments.values.map(_serializeAlignment).toList(),
      'keyposes': keyposes.map(_serializeKeypose).toList(),
    });

    http.Response response;
    try {
      response = await http
          .post(
            Uri.parse('$_relayUrl/feedback'),
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 30));
    } catch (e) {
      throw FeedbackClientError('Network error: $e');
    }

    if (response.statusCode != 200) {
      throw FeedbackClientError(
        'Relay returned ${response.statusCode}: ${response.body}',
      );
    }

    try {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      final raw  = json['feedback'] as Map<String, dynamic>;
      return raw.map((k, v) => MapEntry(int.parse(k), v as String));
    } catch (e) {
      throw FeedbackClientError('Failed to parse relay response: $e');
    }
  }

  static Map<String, dynamic> _serializeAlignment(MovementAlignment a) => {
        'movement_number': a.movementNumber,
        'mean_rms': _safe(a.meanRms),
        'max_rms':  _safe(a.maxRms),
        'frame_pairs': a.framePairs.map(_serializeFramePair).toList(),
      };

  static Map<String, dynamic> _serializeFramePair(FramePair fp) => {
        'master_frame':  fp.masterFrame,
        'student_frame': fp.studentFrame,
        'angle_diffs':   fp.angleDiffs.map((k, v) => MapEntry(k, _safe(v))),
        'angle_delta':   fp.angleDelta.map((k, v) => MapEntry(k, _safe(v))),
        'rms_diff':      _safe(fp.rmsDiff),
      };

  static double _safe(double v) => (v.isNaN || v.isInfinite) ? 0.0 : v;

  static Map<String, dynamic> _serializeKeypose(KeyposeEntry kp) => {
        'movement_index': kp.movementIndex,
        'movement_name':  kp.movementName,
        'itf_name':       kp.itfName,
      };
}

class FeedbackClientError implements Exception {
  final String message;
  const FeedbackClientError(this.message);
  @override
  String toString() => 'FeedbackClientError: $message';
}
