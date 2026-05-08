# CLAUDE.md

## Project Overview
ITF Taekwondo poomsae video comparison analysis system. Compares a master's reference video against a student's video and provides natural-language feedback on per-movement differences.

- **Phase 1 target**: Chon-Ji poomsae (19 movements); Do-San data also collected
- **Evaluation philosophy**: No numeric scores. Only point out what's wrong.
- **Privacy**: Student videos are never stored on servers. Only extracted joint coordinates are sent to the LLM relay.

## Repository Layout

```
E:\Projects\Saas\Taekwon\
├── CLAUDE.md
├── .git / .gitignore
├── analysis_engine/          # Python pipeline (Layers 1–5) + FastAPI relay
│   ├── itf_analysis/         # Pipeline source code
│   ├── server/relay.py       # FastAPI relay — proxies Groq API calls
│   ├── .env                  # LLM_PROVIDER, API keys (gitignored)
│   ├── .env.example
│   ├── requirements.txt
│   └── verify_setup.py
└── taekwondo_app/            # Flutter mobile app (Layers 1–5 on-device)
    ├── lib/
    │   ├── main.dart
    │   └── layers/           # Dart ports of pipeline layers
    ├── assets/master_data/   # keyposes_angles.json bundled as Flutter assets
    └── test/layers/          # Dart unit tests
```

## Tech Stack

### Python pipeline (`analysis_engine/`)
- Python 3.11+
- MediaPipe Pose (joint extraction, on-device)
- NumPy, SciPy (numerical computation)
- fastdtw (temporal alignment)
- FastAPI + uvicorn (relay server)
- Groq Python SDK / OpenAI-compatible (Llama 3.3 70B — Layer 5 active)
- Anthropic Python SDK, Google Gemini SDK, DeepSeek (alternate LLM clients)
- Jupyter (validation notebooks)

### Flutter app (`taekwondo_app/`)
- Flutter / Dart (Android + iOS target; web build supported for Layers 2–5)
- `google_mlkit_pose_detection` — Layer 1, native-only (Android + iOS)
- `http: ^1.2.2` — Layer 5 relay calls
- `collection: ^1.19.1`
- DTW implemented from scratch in Dart (no fastdtw package available)
- Gaussian smoothing and local-minimum finder re-implemented in Dart (replace scipy)

## Pipeline Architecture (5 layers)

1. **Pose extraction**: MediaPipe extracts 33 joint coordinates per frame
2. **Normalization**: Hip center as origin, shoulder width as unit scale; extract 8 joint angles
3. **Segmentation**: Velocity valley detection + keypose angle matching → 19 MovementBoundary objects
4. **Alignment**: Map student boundaries to master movements; handle skipped/extra movements
5. **Difference measurement + LLM feedback**: Extract joints exceeding thresholds, convert to natural language

Each layer's output is the next layer's input. Layers must be independently testable.

## Implementation Status

### Python pipeline
| Layer | Status | Gate |
|-------|--------|------|
| 1 Pose extraction | ✅ Complete | — |
| 2 Normalization | ✅ Complete | — |
| 3 Segmentation | ✅ Complete | 18/19 high on master-on-master ✓ |
| 4 Alignment | ✅ Complete | 19/19 movements aligned; overall mean RMS 25.6° on test student ✓ |
| 5 Feedback | ✅ Complete | Groq (Llama 3.3 70B) active; feedback verified in notebook 06 ✓ |

### Flutter app
| Layer | Status | Notes |
|-------|--------|-------|
| 1 Pose extraction | ⏳ Pending | `google_mlkit_pose_detection` integration (native only) |
| 2 Normalization | ✅ Complete | `normalizer.dart` — `normalizePose()`, `extractJointAngles()` |
| 3 Segmentation | ✅ Complete | `velocity_detector.dart`, `keypose_matcher.dart`, `segmenter.dart` |
| 4 Alignment | ✅ Complete | `gap_aligner.dart`, `dtw_aligner.dart` (DTW from scratch, Sakoe-Chiba band) |
| 5 Feedback | ✅ Complete | `feedback_client.dart` — HTTP POST to relay server |
| Relay server | ✅ Complete | `analysis_engine/server/relay.py` — FastAPI, wraps `llm_feedback.py` |
| UI screens | ⏳ Pending | `record_screen.dart`, `results_screen.dart` |

**Next immediate step**: Layer 1 Flutter — `google_mlkit_pose_detection` camera integration (Android/iOS).

## Running the Project

### Start relay server
```bash
cd analysis_engine
.venv\Scripts\activate          # Windows
uvicorn server.relay:app --reload --port 8000
```

### Run Flutter app (web — Layers 2–5 only)
```bash
cd taekwondo_app
flutter run -d chrome
```

### Run Flutter app (Android/iOS — all layers)
```bash
cd taekwondo_app
flutter run                     # connects to attached device
```

### Run Python tests
```bash
cd analysis_engine
python -m pytest itf_analysis/tests/
```

### Run Flutter tests
```bash
cd taekwondo_app
flutter test
```

## MediaPipe Pose Landmarks (frequently used)
- 11 / 12: left / right shoulder
- 13 / 14: left / right elbow
- 15 / 16: left / right wrist
- 23 / 24: left / right hip
- 25 / 26: left / right knee
- 27 / 28: left / right ankle

Full reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

## Folder Structure

### `analysis_engine/itf_analysis/`

```
itf_analysis/
├── pose_extraction/
│   └── extractor.py                  # Layer 1: MediaPipe → FrameData / Landmark
├── normalization/
│   └── normalizer.py                 # Layer 2: normalize_pose(), extract_joint_angles()
├── segmentation/
│   ├── keypose_matcher.py            # load_master_keyposes(), keypose_distance()
│   ├── velocity_detector.py          # compute_motion_velocity(), smooth_velocity(), find_velocity_valleys()
│   ├── segmenter.py                  # segment_movements(), segment_student_video()  ← main runtime
│   ├── keypose_angle_marker.py       # instructor tool: build_angle_entry(), save_keyposes_angles()
│   ├── segmentor.py                  # [LEGACY — do not use]
│   └── keypose_marker.py             # [LEGACY — do not use]
├── alignment/
│   ├── gap_aligner.py                # align_with_gaps(), validate_alignment()
│   └── dtw_aligner.py                # build_movement_segments(), align_movement(), align_all_movements()
├── feedback/
│   ├── llm_client.py                 # LLMClient ABC + GroqClient / GeminiClient / ClaudeClient / DeepSeekClient
│   ├── llm_feedback.py               # generate_full_feedback() — formats prompts, calls LLMClient
│   └── difference_analyzer.py        # MovementIssueSummary — extracts per-angle differences above threshold
├── master_data/
│   ├── chon_ji/
│   │   └── keyposes_angles.json      # 8 joint angles × 19 movements
│   └── do_san/
│       └── keyposes_angles.json      # Do-San reference keyposes
├── tests/
│   ├── test_normalizer.py
│   ├── test_segment_student.py       # 7 tests — segment_student_video()
│   ├── test_gap_aligner.py           # 7 tests — align_with_gaps()
│   ├── test_validate_alignment.py    # 9 tests — validate_alignment()
│   └── test_segmentor.py             # [LEGACY]
└── notebooks/
    ├── 01_pose_extraction.ipynb
    ├── 02_normalization.ipynb
    ├── 03_keypose_marking.ipynb      # instructor workflow: auto-detect + color-coded verification
    ├── 03a_keypose_matching.ipynb
    ├── 03b_velocity_valleys.ipynb
    ├── 03c_full_segmentation.ipynb
    ├── 04_student_segmentation.ipynb
    ├── 05_dtw_alignment.ipynb        # Layer 4 DTW alignment + heatmap
    ├── 05_difference_analysis.ipynb
    └── 06_llm_feedback.ipynb         # Layer 5 end-to-end feedback verification
```

### `taekwondo_app/`

```
taekwondo_app/
├── lib/
│   ├── main.dart                     # Pipeline demo: loads keyposes, runs Layers 2–4, shows results table
│   └── layers/
│       ├── pose_extraction/          # Layer 1 — placeholder (google_mlkit_pose_detection, pending)
│       ├── normalization/
│       │   └── normalizer.dart       # normalizePose(), extractJointAngles()
│       ├── segmentation/
│       │   ├── velocity_detector.dart  # computeMotionVelocity(), smoothVelocity(), findVelocityValleys()
│       │   ├── keypose_matcher.dart    # KeyposeEntry, loadMasterKeyposes(), keyposeDistance()
│       │   └── segmenter.dart          # segmentMovements(), segmentStudentVideo(), MovementBoundary
│       ├── alignment/
│       │   ├── gap_aligner.dart        # alignWithGaps(), validateAlignment()
│       │   └── dtw_aligner.dart        # DTW + Sakoe-Chiba band, alignMovement(), alignAllMovements()
│       └── feedback/
│           └── feedback_client.dart    # HTTP POST to relay, FeedbackClient, FeedbackClientError
├── assets/
│   └── master_data/
│       ├── chon_ji/keyposes_angles.json
│       └── do_san/keyposes_angles.json
├── test/
│   └── layers/
│       ├── normalizer_test.dart
│       ├── segmentation_test.dart
│       └── alignment_test.dart
└── pubspec.yaml
```

## Flutter Architecture

```
Flutter App (on-device)
  ├── Layer 1  Pose extraction     ← google_mlkit_pose_detection (native; pending)
  ├── Layer 2  Normalization       ← pure Dart math
  ├── Layer 3  Segmentation        ← velocity valleys + keypose matching
  ├── Layer 4  DTW alignment       ← DTW from scratch, Sakoe-Chiba band radius=10
  └── Layer 5  HTTP POST ──────────────────────────────────────────────────┐
                                                                            ▼
                                                           analysis_engine/server/relay.py
                                                             └── Groq API (Llama 3.3 70B)
```

- **Master keyposes** (`keyposes_angles.json`) bundled as Flutter assets — no server fetch needed.
- **No video ever leaves the device.**
- **API key** (`GROQ_API_KEY`) lives only in `analysis_engine/.env` — never in Flutter app.
- Default relay URL: `http://10.0.2.2:8000` (Android emulator → host localhost).

## Layer 3: Segmentation Algorithm

### Angle features
`extract_joint_angles()` / `extractJointAngles()` produces 8 keys per frame:
`right_knee`, `left_knee`, `right_elbow`, `left_elbow`, `right_hip`, `left_hip`,
`shoulder_line_angle`, `hip_line_angle`

`shoulder_line_angle` and `hip_line_angle` are atan2-based and use circular arithmetic in distance calculations.

### Keypose distance
RMS Euclidean distance across all common angle keys between a frame and a stored keypose.
Default threshold: 25°.

### Master segmentation — `segment_movements()`

**Expected frame calculation (priority order):**
1. `source_frames` from `keyposes_angles.json` — used when all 19 are present (most accurate)
2. Rolling fallback — when json is absent: `expected[i] = expected[i-1] + avg_mov`, where
   `avg_mov = total_range / num_movements`.

`half_window ≈ avg_movement_frames × 0.45`. `end_frame` clips the search upper bound to
exclude post-poomsae content. `start_frame` clips the lower bound to exclude pre-poomsae lead-in.
Both affect `total_range`, `avg_mov`, and `prev_frame` initialization.

**Valley selection — combined scoring:**
```
score = 0.5 × (velocity / max_velocity_in_window)
      + 0.5 × min(keypose_distance / threshold, 1.0)
```
Falls back to deepest-velocity selection when no keypose angles are available.

**Confidence rules:**
- `"high"` — both signals found AND within `tolerance_frames=5` of each other
- `"medium"` — exactly one signal found (or both found but disagreeing)
- `"low"` — neither signal → fallback to expected frame + warning

### Student segmentation — `segment_student_video()`
Greedy valley-to-keypose matching (no fixed windows). Naturally handles skipped movements.

### Keypose marking workflow — `notebooks/03_keypose_marking.ipynb`
Auto-detects candidate frames, shows a color-coded diagnostic table (OK / VERIFY / CHECK MANUALLY / PAST END_FRAME) and color-coded chart overlay. Instructor manually overrides any wrong frames, then saves `keyposes_angles.json`.

**IMPORTANT**: Commit `keyposes_angles.json` immediately after a successful marking run.

## Layer 4: Alignment

### `align_with_gaps()` — `alignment/gap_aligner.py` / `gap_aligner.dart`
Maps each master movement to the student boundary with the same `movement_number`.

### `validate_alignment()` — `alignment/gap_aligner.py` / `gap_aligner.dart`

| Condition | valid | Message |
|-----------|-------|---------|
| skipped > 3 | False | "N movements were not detected. Please make your movements larger and clearer." |
| extra > 2 | False | "Duplicate movements detected. Please perform each movement exactly once." |
| low conf > 5 | True | "N movements are unclear. Results may not be accurate." |
| otherwise | True | "" |

Skipped rule fires before extra rule when both are triggered.

### `build_movement_segments()` — `alignment/dtw_aligner.py` / `dtw_aligner.dart`
Movement N = frames from (boundary[N-1].frame + 1) to boundary[N].frame inclusive.

### `align_movement()` — `alignment/dtw_aligner.py` / `dtw_aligner.dart`
fastdtw (Python) / custom DTW with Sakoe-Chiba band radius=10 (Dart).
- Circular arithmetic for `shoulder_line_angle` / `hip_line_angle`
- Returns `MovementAlignment` with `frame_pairs`, per-angle diffs, and summary stats

## Layer 5: LLM Feedback

### Python relay (`analysis_engine/server/relay.py`)
- `POST /feedback` — accepts Flutter payload (serialized alignments + keyposes), reconstructs
  `MovementIssueSummary` objects, calls `generate_full_feedback()`, returns JSON
- `GET /health`

### Flutter client (`taekwondo_app/lib/layers/feedback/feedback_client.dart`)
```dart
POST {relayUrl}/feedback
body: { "movement_issues": [...], "poomsae": "chon_ji" }
response: { "feedback": ["Movement 1: ...", "Movement 2: ..."] }
```

### LLM Client Architecture (Python)

All LLM calls MUST go through the abstraction layer in `feedback/llm_client.py`.

```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        """Returns generated text. Raises LLMClientError on failure."""
```

Active provider is controlled by `LLM_PROVIDER` in `analysis_engine/.env`.

| Provider | Class | LLM_PROVIDER value | Notes |
|---|---|---|---|
| Groq (Llama 3.3 70B) | `GroqClient` | `groq` | **Active default** |
| Google Gemini | `GeminiClient` | `gemini` | Free tier dev option |
| Anthropic Claude | `ClaudeClient` | `claude` | Production option |
| DeepSeek | `DeepSeekClient` | `deepseek` | Budget option |

### `.env` keys (`analysis_engine/.env`)
```
LLM_PROVIDER=groq
GROQ_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Code Style
- Small functions, single responsibility
- Type hints required on all Python function signatures; type annotations on all Dart functions
- Every Python module includes a validation block under `if __name__ == "__main__":`
- Docstrings in English, Google style (Python); `///` doc comments (Dart)
- All strings (code, comments, prompts, error messages, commit messages) in English only

## Validation Principles
- After each new module, validate visually in a notebook before integrating
- Master-on-master self-matching must pass (high ≥ 15/19) before testing student videos
- End-to-end pipeline must work on one master video before adding more poomsae
- Thresholds must be tuned with a real instructor in the loop, not by the developer alone

## Domain Notes
- A poomsae is a choreographed sequence of stances, blocks, strikes, and kicks
- Chon-Ji ("Heaven and Earth") is the first ITF poomsae, 19 movements total
- "Walking stance" (ap-kubi) front knee angle is typically 100°–120°
- "L-stance" (dwi-kubi) back knee bears ~70% of weight
- A "keypose" in this project = the completion posture of one numbered movement

## Non-Goals (Phase 1)
- No numeric scoring (intentional — see project overview)
- No real-time analysis (post-recording only)
- No support for poomsae other than Chon-Ji / Do-San (data collected, not integrated)
- No deep-learning-based segmentation (insufficient training data)
- No dojang/instructor management features
