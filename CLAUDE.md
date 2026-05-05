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
2. **Normalization**: Hip center as origin, shoulder width as unit scale
3. **Segmentation**: Combine keypose matching + velocity valley detection to split into 19 movements
4. **DTW alignment**: Per-movement temporal alignment between master and student frames
5. **Difference measurement + Claude feedback**: Extract joints exceeding thresholds, convert to natural language

Each layer's output is the next layer's input. Layers must be independently testable.

## MediaPipe Pose Landmarks (frequently used)
- 11 / 12: left / right shoulder
- 13 / 14: left / right elbow
- 15 / 16: left / right wrist
- 23 / 24: left / right hip
- 25 / 26: left / right knee
- 27 / 28: left / right ankle

Full reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

## Code Style
- Small functions, single responsibility
- Type hints required on all function signatures
- Every module includes a small validation block under `if __name__ == "__main__":`
- Docstrings in English, formatted as Google style
- All strings (code, comments, prompts, error messages, commit messages) in English only

## Folder Structure
```
itf_analysis/
├── pose_extraction/      # Layer 1
├── normalization/        # Layer 2
├── segmentation/         # Layer 3
├── alignment/            # Layer 4
├── feedback/             # Layer 5
├── master_data/          # Master reference data (JSON)
│   └── chon_ji/
│       ├── poses.json
│       ├── keyposes.json
│       └── thresholds.json
├── tests/                # Unit tests (pytest)
├── notebooks/            # Validation notebooks (numbered by layer)
└── sample_videos/        # Local-only, gitignored
```

## Validation Principles
- After each new module, validate visually in a notebook before integrating
- Master-on-master self-matching must succeed before testing student videos
  (e.g., running master video through segmentation should detect all 19 movements)
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

### Factory function
`get_llm_client()` in `llm_client.py` reads `LLM_PROVIDER` from env and returns the correct client instance.

## Non-Goals (Phase 1)
- No numeric scoring (intentional — see project overview)
- No real-time analysis (post-recording only)
- No support for poomsae other than Chon-Ji
- No deep-learning-based segmentation (insufficient training data)
- No dojang/instructor management features
