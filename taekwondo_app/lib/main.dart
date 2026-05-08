import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'layers/segmentation/keypose_matcher.dart';
import 'layers/segmentation/segmenter.dart';
import 'layers/alignment/dtw_aligner.dart';
import 'layers/alignment/gap_aligner.dart';
import 'ui/analysis_screen.dart';

void main() {
  runApp(const TaekwondoApp());
}

class TaekwondoApp extends StatelessWidget {
  const TaekwondoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ITF Taekwondo Analysis',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Home — choose demo mode
// ---------------------------------------------------------------------------

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('ITF Taekwondo Analysis'),
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Select a mode',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              icon: const Icon(Icons.compare_arrows),
              label: const Text('Chon-Ji: Master vs Student'),
              style: ElevatedButton.styleFrom(
                  minimumSize: const Size(260, 52)),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (_) => const AnalysisScreen()),
              ),
            ),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              icon: const Icon(Icons.science),
              label: const Text('Pipeline Self-test (synthetic data)'),
              style: OutlinedButton.styleFrom(
                  minimumSize: const Size(260, 52)),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (_) => const PipelineDemoPage()),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Original self-test demo (unchanged)
// ---------------------------------------------------------------------------

class PipelineDemoPage extends StatefulWidget {
  const PipelineDemoPage({super.key});

  @override
  State<PipelineDemoPage> createState() => _PipelineDemoPageState();
}

class _PipelineDemoPageState extends State<PipelineDemoPage> {
  String _status = 'Press Run to test the pipeline.';
  List<String> _results = [];
  bool _running = false;

  Future<void> _runPipeline() async {
    setState(() {
      _running = true;
      _status = 'Loading master keyposes...';
      _results = [];
    });

    try {
      final jsonStr = await rootBundle.loadString(
        'assets/master_data/chon_ji/keyposes_angles.json',
      );
      final keyposes = loadMasterKeyposes(jsonStr);

      setState(() => _status = 'Keyposes loaded: ${keyposes.length} movements');

      final frameAngles = <(int, Map<String, double>)>[];
      for (int i = 0; i < keyposes.length; i++) {
        final kp = keyposes[i];
        for (int f = 0; f < 60; f++) {
          final angles = kp.angles.map(
            (k, v) => MapEntry(k, v + (f % 5 - 2) * 0.5),
          );
          frameAngles.add((i * 60 + f, angles));
        }
      }

      setState(() => _status = 'Running segmentation...');

      final boundaries = segmentMovements(
        frameAngles,
        keyposes,
        30.0,
        numMovements: keyposes.length,
      );

      setState(() => _status = 'Running DTW alignment (self-match)...');

      final masterSegs = buildMovementSegments(frameAngles, boundaries);
      final alignments = alignAllMovements(masterSegs, masterSegs);

      final masterIndices = keyposes.map((k) => k.movementIndex).toList();
      final pairs = alignWithGaps(masterIndices, boundaries);
      final validation = validateAlignment(pairs);

      final rows = <String>[];
      for (final b in boundaries) {
        final a = alignments[b.movementNumber];
        final rms = a != null ? a.meanRms.toStringAsFixed(1) : '—';
        final conf = b.confidence.name;
        rows.add(
          'Mov ${b.movementNumber.toString().padLeft(2)} '
          '| frame ${b.frame.toString().padLeft(4)} '
          '| $conf '
          '| RMS $rms°',
        );
      }

      setState(() {
        _status = validation.valid
            ? 'Pipeline OK  ${validation.message.isEmpty ? "(all clean)" : "(${validation.message})"}'
            : 'Validation failed: ${validation.message}';
        _results = rows;
        _running = false;
      });
    } catch (e, st) {
      setState(() {
        _status = 'Error: $e';
        _results = ['$st'];
        _running = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Pipeline Self-test'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_status, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (_running) const LinearProgressIndicator(),
            const SizedBox(height: 16),
            Expanded(
              child: _results.isEmpty
                  ? const Center(child: Text('No results yet.'))
                  : ListView.builder(
                      itemCount: _results.length,
                      itemBuilder: (_, i) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2),
                        child: Text(
                          _results[i],
                          style: const TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 13,
                          ),
                        ),
                      ),
                    ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _running ? null : _runPipeline,
        icon: const Icon(Icons.play_arrow),
        label: const Text('Run Pipeline'),
      ),
    );
  }
}
