# CLAUDE.md

## Project Overview
ITF Taekwondo poomsae video comparison analysis system. Compares a master's reference video against a student's video and provides natural-language feedback on per-movement differences.

- **Phase 1 target**: Chon-Ji poomsae (19 movements)
- **Evaluation philosophy**: No numeric scores. Only point out what's wrong.
- **Privacy**: Student videos are never stored on servers. Only extracted joint coordinates are sent to Claude.

## Tech Stack
- Python 3.11+
- MediaPipe Pose (joint extraction, on-device)
- NumPy, SciPy (numerical computation)
- fastdtw (temporal alignment)
- Anthropic Python SDK (Claude Haiku for feedback generation)
- FastAPI (server wrapper, added later)
- Jupyter (validation notebooks)

## Pipeline Architecture (5 layers)
1. **Pose extraction**: MediaPipe extracts 33 joint coordinates per frame
2. **Normalization**: Hip center as origin, shoulder width as unit scale; extract 8 joint angles
3. **Segmentation**: Velocity valley detection + keypose angle matching → 19 MovementBoundary objects
4. **Alignment**: Map student boundaries to master movements; handle skipped/extra movements
5. **Difference measurement + Claude feedback**: Extract joints exceeding thresholds, convert to natural language

Each layer's output is the next layer's input. Layers must be independently testable.

## Implementation Status

| Layer | Status | Gate |
|-------|--------|------|
| 1 Pose extraction | ✅ Complete | — |
| 2 Normalization | ✅ Complete | — |
| 3 Segmentation | ✅ Complete | 18/19 high on master-on-master ✓ |
| 4 Alignment | ✅ Complete | 19/19 movements aligned; overall mean RMS 25.6° on test student ✓ |
| 5 Feedback | ❌ Not started | — |

**Next immediate step**: Layer 5 — per-movement angle difference → natural-language feedback via LLM.

## MediaPipe Pose Landmarks (frequently used)
- 11 / 12: left / right shoulder
- 13 / 14: left / right elbow
- 15 / 16: left / right wrist
- 23 / 24: left / right hip
- 25 / 26: left / right knee
- 27 / 28: left / right ankle

Full reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

## Folder Structure

```
itf_analysis/
├── pose_extraction/
│   └── extractor.py              # Layer 1: MediaPipe → FrameData / Landmark
├── normalization/
│   └── normalizer.py             # Layer 2: normalize_pose(), extract_joint_angles()
├── segmentation/
│   ├── keypose_matcher.py        # load_master_keyposes(), keypose_distance()
│   ├── velocity_detector.py      # compute_motion_velocity(), smooth_velocity(), find_velocity_valleys()
│   ├── segmenter.py              # segment_movements(), segment_student_video()  ← main runtime
│   ├── keypose_angle_marker.py   # instructor tool: build_angle_entry(), save_keyposes_angles()
│   ├── segmentor.py              # [LEGACY — do not use]
│   └── keypose_marker.py         # [LEGACY — do not use]
├── alignment/
│   ├── gap_aligner.py            # align_with_gaps(), validate_alignment()
│   └── dtw_aligner.py            # build_movement_segments(), align_movement(), align_all_movements()
├── feedback/                     # Layer 5 — not yet implemented
│   └── llm_client.py             # LLMClient ABC + GeminiClient / ClaudeClient / DeepSeekClient
├── master_data/
│   └── chon_ji/
│       └── keyposes_angles.json  # 8 joint angles × 19 movements (written by keypose_marking workflow)
├── tests/
│   ├── test_normalizer.py
│   ├── test_segment_student.py   # 7 tests — segment_student_video()
│   ├── test_gap_aligner.py       # 7 tests — align_with_gaps()
│   ├── test_validate_alignment.py # 9 tests — validate_alignment()
│   └── test_segmentor.py         # [LEGACY — references old segmentor.py]
├── notebooks/
│   ├── 01_pose_extraction.ipynb
│   ├── 02_normalization.ipynb
│   ├── 03_keypose_marking.ipynb  # instructor workflow to populate keyposes_angles.json
│   ├── 03a_keypose_matching.ipynb
│   ├── 03b_velocity_valleys.ipynb
│   └── 03c_full_segmentation.ipynb
│   ├── 04_student_segmentation.ipynb  # student video workflow (config at top)
│   └── 05_dtw_alignment.ipynb         # Layer 4 DTW alignment + heatmap
└── sample_videos/                # local-only, gitignored
```

## Layer 3: Segmentation Algorithm

### Angle features
`extract_joint_angles()` produces 8 keys per frame:
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
   `avg_mov = total_range / num_movements`. More adaptive than fixed linear spacing.

`half_window ≈ avg_movement_frames × 0.45`. `end_frame` clips the search upper bound to
exclude post-poomsae content (e.g., return-to-ready stance). `start_frame` is the symmetric
counterpart — clips the lower bound to exclude pre-poomsae lead-in (e.g., ready stance).
Both affect `total_range`, `avg_mov`, and `prev_frame` initialization.

**Valley selection — combined scoring:**
Within each search window, valleys are scored as:
```
score = 0.5 × (velocity / max_velocity_in_window)
      + 0.5 × min(keypose_distance / threshold, 1.0)
```
The valley with the lowest combined score is selected. Falls back to deepest-velocity
selection when no keypose angles are available.

**Keypose signal:** frame with minimum RMS angle distance to stored keypose angles.

**Confidence rules:**
- `"high"` — both signals found AND within `tolerance_frames=5` of each other
- `"medium"` — exactly one signal found (or both found but disagreeing)
- `"low"` — neither signal → fallback to expected frame + warning

### Student segmentation — `segment_student_video()`
Does **not** use fixed search windows (student timing is unknown). Instead:
1. Detect all velocity valleys in the student sequence
2. For each valley (in order), find the closest unmatched master keypose with an ordering
   constraint (cannot match a movement earlier than the last matched one)
3. If RMS distance < threshold → create boundary (`high` if < threshold×0.5, else `medium`)
4. Valleys with no close match are silently discarded → undetected movements are simply absent

This naturally handles skipped movements: no valley near that keypose → it stays unmatched.

### Keypose marking workflow (one-time per poomsae)
Run `notebooks/03_keypose_marking.ipynb`:
1. **Auto-detect** candidate frames (cell 6):
   - `START_FRAME` = first frame of actual poomsae (after ready stance)
   - `END_FRAME` = last frame of poomsae content (before return-to-ready)
   - Both clip `total_range` so `avg_mov` is based only on the real poomsae duration
   - **Rolling expected**: each window centered at `last_detected + avg_mov`
   - **Deepest valley** selected within each window
2. **Verify** `KEYPOSE_FRAMES` dict (cell 8) — watch overlay video, override any wrong frames
3. **Build** angle entries and save `keyposes_angles.json` (cells 12–16)
4. **Re-run** `segment_movements()` → confirm high ≥ 15/19 (cell 18)

**IMPORTANT**: `keyposes_angles.json` is untracked by default. Commit it immediately after
a successful marking run (`git add` + `git commit`) to prevent data loss.

## Layer 4: Alignment

### `align_with_gaps()` — `alignment/gap_aligner.py`
Maps each of 19 master movements to the student boundary with the same `movement_number`.
```python
@dataclass
class AlignmentPair:
    master_movement: Optional[int]           # 1–19, or None for extras
    student_boundary: Optional[MovementBoundary]
    match_type: Literal["matched", "skipped", "extra"]
```

### `validate_alignment()` — `alignment/gap_aligner.py`
Returns a `ValidationResult` with an English message for the student.

| Condition | valid | Message |
|-----------|-------|---------|
| skipped > 3 | False | "N movements were not detected. Please make your movements larger and clearer." |
| extra > 2 | False | "Duplicate movements detected. Please perform each movement exactly once." |
| low conf > 5 | True | "N movements are unclear. Results may not be accurate." |
| otherwise | True | "" |

Skipped rule fires before extra rule when both are triggered.

### `build_movement_segments()` — `alignment/dtw_aligner.py`
Splits the full frame-angle sequence into per-movement sub-sequences using boundary frames.
Movement N = frames from (boundary[N-1].frame + 1) to boundary[N].frame inclusive.

### `align_movement()` — `alignment/dtw_aligner.py`
Runs fastdtw on one master–student movement pair.
- Angle dicts are converted to fixed-order numpy arrays (sorted key union of both segments)
- Missing keys are forward-filled via linear interpolation; empty columns set to 0
- Distance function: RMS across common keys; circular arithmetic for `shoulder_line_angle` / `hip_line_angle`
- Returns `MovementAlignment` with `frame_pairs` (actual frame indices + per-angle diffs) and summary stats

### `align_all_movements()` — `alignment/dtw_aligner.py`
Calls `align_movement` for each movement present in both segment dicts.
Skipped movements (absent from student) are simply not included in the result.

## Code Style
- Small functions, single responsibility
- Type hints required on all function signatures
- Every module includes a validation block under `if __name__ == "__main__":`
- Docstrings in English, Google style
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

## LLM Client Architecture

All LLM calls MUST go through the abstraction layer in `feedback/llm_client.py`.
Never call any LLM SDK directly from pipeline code.

### Rules
- All LLM interaction goes through `LLMClient` abstract base class
- Active provider is controlled by `LLM_PROVIDER` in `.env`
- Adding a new provider = add a new subclass only, zero changes to pipeline code
- Every client must handle: timeout, empty response, API error — and raise `LLMClientError`

### Interface
```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        """Returns generated text. Raises LLMClientError on failure."""
        pass
```

### Supported Providers

| Provider | Class | LLM_PROVIDER value | Notes |
|---|---|---|---|
| Google Gemini | `GeminiClient` | `gemini` | Default. Free tier during dev. |
| Anthropic Claude | `ClaudeClient` | `claude` | Fallback / production option |
| DeepSeek | `DeepSeekClient` | `deepseek` | Budget option, OpenAI-compatible |

### .env keys
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

### Switching providers
Changing `LLM_PROVIDER` in `.env` is the only change needed to switch providers.
No pipeline code should ever need to change when switching providers.

### Usage in pipeline
```python
# feedback/llm_feedback.py — correct usage
from feedback.llm_client import get_llm_client

client = get_llm_client()  # reads LLM_PROVIDER from env
feedback = client.generate(system_prompt=..., user_prompt=...)
```

## Non-Goals (Phase 1)
- No numeric scoring (intentional — see project overview)
- No real-time analysis (post-recording only)
- No support for poomsae other than Chon-Ji
- No deep-learning-based segmentation (insufficient training data)
- No dojang/instructor management features
