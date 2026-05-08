/// Chon-Ji analysis screen: loads pre-extracted poses, runs Layers 2–5,
/// shows master + student video alongside per-movement results.
library;

import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';

import '../layers/normalization/normalizer.dart';
import '../layers/segmentation/keypose_matcher.dart';
import '../layers/segmentation/segmenter.dart';
import '../layers/segmentation/velocity_detector.dart';
import '../layers/alignment/dtw_aligner.dart' hide FrameAngles;
import '../layers/alignment/gap_aligner.dart';
import '../layers/feedback/feedback_client.dart';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

String get _relayBase =>
    kIsWeb ? 'http://localhost:8000' : 'http://10.0.2.2:8000';

const _poomsae = 'chon_ji';
const _fps = 30.0;

// ---------------------------------------------------------------------------
// Isolate-safe top-level helpers
// ---------------------------------------------------------------------------

typedef _SegArgs = ({FrameAngles frames, List<KeyposeEntry> keyposes, double fps, int n});
typedef _StudentSegArgs = ({FrameAngles frames, List<KeyposeEntry> keyposes, double fps, double threshold});
typedef _DtwArgs = ({Map<int, FrameAngles> master, Map<int, FrameAngles> student});
typedef _DiagArgs = ({FrameAngles frames, List<KeyposeEntry> keyposes, double fps, double threshold});

List<MovementBoundary> _runSegmentMovements(_SegArgs a) =>
    segmentMovements(a.frames, a.keyposes, a.fps, numMovements: a.n);

List<MovementBoundary> _runSegmentStudent(_StudentSegArgs a) =>
    segmentStudentVideo(a.frames, a.keyposes, a.fps, threshold: a.threshold);

Map<int, MovementAlignment> _runDtw(_DtwArgs a) =>
    alignAllMovements(a.master, a.student);

/// Returns a diagnostic string: frame count, valley count, best keypose
/// distance found across all valley frames, and how many (valley, keypose)
/// pairs fall below the threshold.
String _runDiagnostic(_DiagArgs a) {
  final vel      = computeMotionVelocity(a.frames, a.fps);
  final smoothed = smoothVelocity(vel, a.fps, sigmaSeconds: 0.1);
  final valleys  = findVelocityValleys(smoothed, a.fps, minGapSec: 0.5);

  final frameIndices  = a.frames.map((e) => e.$1).toList();
  final anglesLookup  = {for (final (fi, fa) in a.frames) fi: fa};

  double minDist   = double.infinity;
  int    hitsBelow = 0;

  for (final v in valleys) {
    final fi = frameIndices[v];
    final fa = anglesLookup[fi];
    if (fa == null) continue;
    for (final kp in a.keyposes) {
      if (kp.angles.isEmpty) continue;
      final d = keyposeDistance(fa, kp.angles);
      if (d < minDist) minDist = d;
      if (d < a.threshold) hitsBelow++;
    }
  }

  final minDistStr = minDist.isInfinite ? '∞' : minDist.toStringAsFixed(1);
  return '${a.frames.length} frames · ${valleys.length} valleys · '
      'best ${minDistStr}° · ${hitsBelow} hits<${a.threshold.round()}°';
}

// ---------------------------------------------------------------------------
// Poses JSON → frame angles
// ---------------------------------------------------------------------------

List<(int, Map<String, double>)> _parsePoses(String jsonStr) {
  final frames = jsonDecode(jsonStr) as List<dynamic>;
  final result = <(int, Map<String, double>)>[];

  for (final frame in frames) {
    final fi = frame['frame_index'] as int;
    final raw = frame['landmarks'] as Map<String, dynamic>;

    final landmarks = <int, Landmark>{};
    for (final e in raw.entries) {
      final lm = e.value as Map<String, dynamic>;
      landmarks[int.parse(e.key)] = Landmark(
        x: (lm['x'] as num).toDouble(),
        y: (lm['y'] as num).toDouble(),
        z: (lm['z'] as num).toDouble(),
        visibility: (lm['visibility'] as num).toDouble(),
      );
    }

    final norm = normalizePose(landmarks);
    if (norm == null) continue;
    final angles = extractJointAngles(norm);
    if (angles.isEmpty) continue;
    result.add((fi, angles));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Video player widget
// ---------------------------------------------------------------------------

class _VideoBox extends StatefulWidget {
  final String url;
  final String label;

  const _VideoBox({super.key, required this.url, required this.label});

  @override
  State<_VideoBox> createState() => _VideoBoxState();
}

class _VideoBoxState extends State<_VideoBox> {
  late VideoPlayerController _ctrl;
  bool _ready = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _ctrl = VideoPlayerController.networkUrl(Uri.parse(widget.url))
      ..initialize().then((_) {
        if (mounted) setState(() => _ready = true);
      }).catchError((e) {
        if (mounted) setState(() => _error = e.toString());
      });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(widget.label,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
        const SizedBox(height: 4),
        AspectRatio(
          aspectRatio: 9 / 16,
          child: _error != null
              ? _placeholder('Video unavailable\n(start relay server)')
              : _ready
                  ? VideoPlayer(_ctrl)
                  : _placeholder('Loading…'),
        ),
        if (_ready)
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                icon: Icon(
                    _ctrl.value.isPlaying ? Icons.pause : Icons.play_arrow),
                onPressed: () => setState(() {
                  _ctrl.value.isPlaying ? _ctrl.pause() : _ctrl.play();
                }),
              ),
              IconButton(
                icon: const Icon(Icons.replay),
                onPressed: () => _ctrl.seekTo(Duration.zero),
              ),
            ],
          ),
      ],
    );
  }

  Widget _placeholder(String msg) => Container(
        color: Colors.black87,
        child: Center(
          child: Text(msg,
              textAlign: TextAlign.center,
              style:
                  const TextStyle(color: Colors.white70, fontSize: 12)),
        ),
      );
}

// ---------------------------------------------------------------------------
// Analysis screen
// ---------------------------------------------------------------------------

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

// Available student pose files — add new entries here as you extract more poses.
const _studentOptions = [
  _StudentOption('student',  'student_poses.json',  'student.mp4'),
  _StudentOption('student3', 'student3_poses.json', 'student3.mp4'),
];

class _StudentOption {
  final String label;
  final String posesAsset;
  final String videoFile;
  const _StudentOption(this.label, this.posesAsset, this.videoFile);
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  // State
  String _status = 'Press Analyze to run the pipeline.';
  bool _busy = false;
  List<_ResultRow> _rows = [];
  Map<int, String> _feedback = {};
  bool _feedbackLoading = false;
  _StudentOption _selectedStudent = _studentOptions.first;
  double _threshold = 40.0; // keypose match threshold (degrees)

  // Computed
  List<KeyposeEntry> _keyposes = [];
  Map<int, MovementAlignment> _alignments = {};

  @override
  void initState() {
    super.initState();
    FeedbackClient.relayUrl = _relayBase;
  }

  // ---- pipeline ----

  Future<void> _analyze() async {
    setState(() {
      _busy = true;
      _status = 'Loading poses…';
      _rows = [];
      _feedback = {};
    });

    try {
      // Load assets
      final kpJson = await rootBundle
          .loadString('assets/master_data/$_poomsae/keyposes_angles.json');
      final masterJson = await rootBundle
          .loadString('assets/test_data/$_poomsae/master_poses.json');
      final studentJson = await rootBundle
          .loadString('assets/test_data/$_poomsae/${_selectedStudent.posesAsset}');

      setState(() => _status = 'Parsing frames…');

      // Parse in isolates to keep UI responsive
      final keyposes = await compute(loadMasterKeyposes, kpJson);
      final masterAngles = await compute(_parsePoses, masterJson);
      final studentAngles = await compute(_parsePoses, studentJson);

      // Diagnostic: count valleys and check keypose distances before segmenting
      final diagMsg = await compute(
        _runDiagnostic,
        (frames: studentAngles, keyposes: keyposes, fps: _fps, threshold: _threshold),
      );
      setState(() => _status = 'Student: $diagMsg');

      // Give the user a moment to read the diagnostic before it changes.
      await Future.delayed(const Duration(milliseconds: 1500));
      setState(() => _status =
          'Segmenting (master ${masterAngles.length} frames, '
          'student ${studentAngles.length} frames)…');

      // Layer 3
      final masterBounds = await compute(
        _runSegmentMovements,
        (frames: masterAngles, keyposes: keyposes, fps: _fps, n: keyposes.length),
      );
      final studentBounds = await compute(
        _runSegmentStudent,
        (frames: studentAngles, keyposes: keyposes, fps: _fps, threshold: _threshold),
      );

      setState(() => _status = 'Running DTW alignment…');

      // Layer 4
      final masterSegs = buildMovementSegments(masterAngles, masterBounds);
      final studentSegs = buildMovementSegments(studentAngles, studentBounds);
      final alignments = await compute(
        _runDtw,
        (master: masterSegs, student: studentSegs),
      );

      // Gap alignment
      final masterNums = keyposes.map((k) => k.movementIndex).toList();
      final pairs = alignWithGaps(masterNums, studentBounds);
      final validation = validateAlignment(pairs);

      // Build result rows
      final rows = masterBounds.map((b) {
        final a = alignments[b.movementNumber];
        return _ResultRow(
          num: b.movementNumber,
          masterFrame: b.frame,
          studentFrame: studentBounds
              .where((s) => s.movementNumber == b.movementNumber)
              .firstOrNull
              ?.frame,
          confidence: b.confidence,
          rms: a?.meanRms,
        );
      }).toList();

      setState(() {
        _keyposes = keyposes;
        _alignments = alignments;
        _rows = rows;
        _status = validation.valid
            ? 'Done. ${alignments.length} movements aligned.'
                '${validation.message.isNotEmpty ? "  ⚠ ${validation.message}" : ""}'
            : 'Validation failed: ${validation.message}';
        _busy = false;
      });
    } catch (e, st) {
      setState(() {
        _status = 'Error: $e';
        _rows = [_ResultRow.error('$st')];
        _busy = false;
      });
    }
  }

  Future<void> _getFeedback() async {
    if (_alignments.isEmpty) return;
    setState(() => _feedbackLoading = true);
    try {
      final fb = await FeedbackClient.getFeedback(
        alignments: _alignments,
        keyposes: _keyposes,
        poomsae: _poomsae,
      );
      setState(() => _feedback = fb);
    } catch (e) {
      setState(() => _feedback = {0: 'Error: $e'});
    } finally {
      setState(() => _feedbackLoading = false);
    }
  }

  // ---- UI ----

  @override
  Widget build(BuildContext context) {
    final videoBase = '$_relayBase/videos/$_poomsae';
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Chon-Ji: Master vs Student'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Student selector
            Row(
              children: [
                const Text('Student video:',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
                const SizedBox(width: 8),
                DropdownButton<_StudentOption>(
                  value: _selectedStudent,
                  items: _studentOptions
                      .map((o) => DropdownMenuItem(
                            value: o,
                            child: Text(o.label,
                                style: const TextStyle(fontSize: 13)),
                          ))
                      .toList(),
                  onChanged: _busy
                      ? null
                      : (v) => setState(() {
                            _selectedStudent = v!;
                            _rows = [];
                            _feedback = {};
                            _status = 'Press Analyze to run the pipeline.';
                          }),
                ),
              ],
            ),
            // Threshold slider
            Row(
              children: [
                const Text('Match threshold:',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
                const SizedBox(width: 8),
                Expanded(
                  child: Slider(
                    value: _threshold,
                    min: 20,
                    max: 120,
                    divisions: 20,
                    label: '${_threshold.round()}°',
                    onChanged: _busy
                        ? null
                        : (v) => setState(() {
                              _threshold = v;
                              _rows = [];
                              _feedback = {};
                            }),
                  ),
                ),
                Text('${_threshold.round()}°',
                    style: const TextStyle(fontSize: 13)),
              ],
            ),
            const SizedBox(height: 4),

            // Videos
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                    child: _VideoBox(
                        url: '$videoBase/master.mp4', label: 'Master')),
                const SizedBox(width: 8),
                Expanded(
                    child: _VideoBox(
                        key: ValueKey(_selectedStudent.videoFile),
                        url: '$videoBase/${_selectedStudent.videoFile}',
                        label: _selectedStudent.label)),
              ],
            ),
            const SizedBox(height: 16),

            // Analyze button + status
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: _busy ? null : _analyze,
                  icon: const Icon(Icons.analytics),
                  label: const Text('Analyze (Layers 2–4)'),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(_status,
                      style: const TextStyle(fontSize: 12),
                      softWrap: true),
                ),
              ],
            ),
            if (_busy) const LinearProgressIndicator(),
            const SizedBox(height: 12),

            // Results table
            if (_rows.isNotEmpty) ...[
              _ResultsTable(rows: _rows),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                onPressed:
                    (_feedbackLoading || _alignments.isEmpty) ? null : _getFeedback,
                icon: _feedbackLoading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.chat),
                label: const Text('Get LLM Feedback (Layer 5)'),
              ),
            ],

            // Feedback
            if (_feedback.isNotEmpty) ...[
              const SizedBox(height: 12),
              const Text('LLM Feedback',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
              const SizedBox(height: 6),
              for (final e in (_feedback.entries
                      .where((e) => e.key > 0)
                      .toList()
                    ..sort((a, b) => a.key.compareTo(b.key))))
                Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Text('Mov ${e.key}: ${e.value}',
                      style: const TextStyle(fontSize: 12)),
                ),
              if (_feedback.containsKey(0))
                Text(_feedback[0]!,
                    style: const TextStyle(color: Colors.red, fontSize: 12)),
            ],

          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Result row model + table widget
// ---------------------------------------------------------------------------

class _ResultRow {
  final int? num;
  final int? masterFrame;
  final int? studentFrame;
  final Confidence? confidence;
  final double? rms;
  final String? errorMsg;

  const _ResultRow({
    required this.num,
    required this.masterFrame,
    required this.studentFrame,
    required this.confidence,
    required this.rms,
  }) : errorMsg = null;

  _ResultRow.error(this.errorMsg)
      : num = null,
        masterFrame = null,
        studentFrame = null,
        confidence = null,
        rms = null;
}

class _ResultsTable extends StatelessWidget {
  final List<_ResultRow> rows;
  const _ResultsTable({required this.rows});

  Color _confColor(Confidence? c) {
    return switch (c) {
      Confidence.high   => Colors.green,
      Confidence.medium => Colors.orange,
      Confidence.low    => Colors.red,
      null              => Colors.grey,
    };
  }

  @override
  Widget build(BuildContext context) {
    return Table(
      columnWidths: const {
        0: FixedColumnWidth(36),
        1: FixedColumnWidth(70),
        2: FixedColumnWidth(70),
        3: FixedColumnWidth(70),
        4: FlexColumnWidth(),
      },
      children: [
        TableRow(
          decoration: BoxDecoration(color: Colors.grey.shade200),
          children: const [
            _TH('Mov'),
            _TH('M-frame'),
            _TH('S-frame'),
            _TH('Conf'),
            _TH('RMS°'),
          ],
        ),
        for (final r in rows)
          r.errorMsg != null
              ? TableRow(children: [
                  _TD('!', color: Colors.red),
                  TableCell(
                      child: Padding(
                          padding: const EdgeInsets.all(4),
                          child: Text(r.errorMsg!,
                              style: const TextStyle(
                                  fontSize: 10, color: Colors.red)))),
                  const _TD(''),
                  const _TD(''),
                  const _TD(''),
                ])
              : TableRow(children: [
                  _TD('${r.num}'),
                  _TD('${r.masterFrame ?? "—"}'),
                  _TD('${r.studentFrame ?? "—"}'),
                  _TD(r.confidence?.name ?? '—',
                      color: _confColor(r.confidence)),
                  _TD(r.rms != null && !r.rms!.isNaN ? r.rms!.toStringAsFixed(1) : '—'),
                ]),
      ],
    );
  }
}

class _TH extends StatelessWidget {
  final String text;
  const _TH(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 6),
        child: Text(text,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
      );
}

class _TD extends StatelessWidget {
  final String text;
  final Color? color;
  const _TD(this.text, {this.color});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
        child: Text(text,
            style: TextStyle(
                fontSize: 12,
                color: color,
                fontFamily: 'monospace')),
      );
}
