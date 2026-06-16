# VoiceTransfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.3 CPU](https://img.shields.io/badge/PyTorch-2.3%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Backend: kNN-VC](https://img.shields.io/badge/backend-kNN--VC-4CAF50)](https://github.com/bshall/knn-vc)
[![Device: CPU only](https://img.shields.io/badge/device-CPU%20only-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CPU-only, zero-shot voice conversion. Takes a **content** audio clip (e.g. speech extracted
from a video) and a **target speaker** reference clip, and outputs the content speech restyled
to sound like the target speaker. No GPU. No text/ASR. No alignment. No manual downloads.

---

## Setup

Both scripts are **idempotent** — safe to re-run. Each step prints `[skip]` if it
detects that the work is already done (`.venv` exists, PyTorch importable, weights cached).

Both use **[uv](https://github.com/astral-sh/uv)** for fast dependency installation.
Install uv first if you don't have it:

```bash
pip install uv
# or (Windows):  winget install astral-sh.uv
# or (macOS/Linux): curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
# If script execution is blocked, run once in an elevated prompt:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1` does, in order (skipping completed steps):

| Step | Command used |
|------|-------------|
| Create `.venv` | `uv venv .venv` |
| Activate | `. .\.venv\Scripts\Activate.ps1` |
| Install requirements | `uv pip install -r requirements.txt` |
| Download model weights | `python download_models.py` |

### macOS / Linux (Bash)

```bash
bash setup.sh
# or, after chmod +x:
./setup.sh
```

`setup.sh` does the same four steps using `source .venv/bin/activate`.

---

### Manual steps (if you prefer not to use the scripts)

> Each step below can be skipped if already done.

```bash
# 1. Create venv (skip if .venv exists)
uv venv .venv

# 2. Activate (skip if prompt already shows (.venv))
source .venv/bin/activate          # macOS / Linux
.\.venv\Scripts\Activate.ps1      # Windows PowerShell

# 3. Install requirements (skip if PyTorch already importable)
uv pip install -r requirements.txt

# 4. Pre-download weights (skip if models/hub/checkpoints/ is populated)
python download_models.py
```

---

## Running the app

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload your audio files, adjust settings in the sidebar,
and click **Convert**. Model weights download automatically on first run (~650 MB).

### CLI

```bash
# Place audio files, then:
python run.py --config config.yaml

# Or pass paths directly (overrides config.yaml):
python run.py --content speech.wav --target voice.wav --output out.wav
```

### Pre-download model weights only

```bash
python download_models.py
```

Fetches all weights into `paths.models_dir` without running a conversion.
Useful to warm the cache before going offline.

---

## Project structure

```
app.py                         Streamlit UI entry point
run.py                         CLI entry point
download_models.py             Pre-fetch weights without converting
config.yaml                    All tunable settings
requirements.txt               Pinned CPU-only dependencies

voicetransfer/
  __init__.py
  config.py                    AppConfig dataclass + load_config + validate_for_conversion
  audio.py                     load / resample / normalize_loudness / save
  download.py                  ensure_models() — TORCH_HOME redirect + hub.load
  pipeline.py                  8-step pipeline + _length_align (pure function)
  mux.py                       ffmpeg remux via imageio-ffmpeg
  converters/
    base.py                    BaseConverter ABC
    knn_vc.py                  Default kNN-VC backend

tests/
  test_length.py               Length-alignment unit + integration tests
```

---

## Where model weights are downloaded

All weights land in `paths.models_dir` (default: `./models`).
`TORCH_HOME` and `HF_HOME` are both redirected there before any hub call.

```
models/
  hub/
    checkpoints/
      wavlm_large_finetune.pt      (~600 MB) — WavLM-Large encoder
      prematch_g_02500000           (~55 MB)  — prematched HiFiGAN vocoder
```

Weights are downloaded once and reused on every subsequent run.

---

## Config reference (`config.yaml`)

```yaml
paths:
  models_dir: "./models"          # Cache for ALL downloaded weights (TORCH_HOME)
  content_audio: "./input/content.wav"
  target_refs:
    - "./input/target.wav"        # One or more target speaker clips
  output_audio: "./output/converted.wav"
  input_video: ""                 # Optional: source video for audio remuxing
  output_video: "./output/converted.mp4"

device:
  type: "cpu"                     # Only "cpu" is supported
  num_threads: 4                  # torch.set_num_threads() — match your core count

backend:
  name: "knn_vc"                  # Selects voicetransfer/converters/<name>.py

knn_vc:
  prematched: true                # Prematched HiFiGAN (recommended for zero-shot)
  topk: 4                         # k nearest neighbours — higher = smoother, slower
  wavlm_layer: 6                  # WavLM-Large transformer layer index (0–23)

audio:
  output_sample_rate: 0           # 0 = match content audio's native sample rate
  normalize_loudness: true        # EBU R128 integrated loudness normalization
  target_lufs: -23.0              # Target loudness; -23 is the broadcast standard

length:
  enforce_exact: true             # Guarantee output == round(content_duration × out_sr)
  pad_mode: "silence"             # Padding strategy when backend returns short audio
  warn_if_drift_ms: 50            # Log WARNING if pre-correction drift exceeds this

mux:
  enabled: false                  # Set true to remux audio onto input_video
  copy_video_codec: true          # -c:v copy — never re-encode the video stream
  ffmpeg_path: "auto"             # "auto" uses imageio-ffmpeg's bundled binary

logging:
  level: "INFO"                   # DEBUG / INFO / WARNING / ERROR
```

---

## How to swap backends

1. **Create** `voicetransfer/converters/<name>.py` and implement `BaseConverter`:

```python
from voicetransfer.converters.base import BaseConverter
import torch

class MyConverter(BaseConverter):
    @property
    def input_sample_rate(self) -> int:
        return 22050  # whatever rate your model expects

    def convert(
        self,
        content_wav: torch.Tensor,
        target_refs: list[torch.Tensor],
    ) -> torch.Tensor:
        # ... inference code ...
        return converted_wav   # float32 mono at input_sample_rate
```

2. **Register** it in `voicetransfer/pipeline.py` → `_build_converter()`:

```python
elif name == "my_backend":
    from voicetransfer.converters.my_backend import MyConverter
    return MyConverter(model, ...)
```

3. **Switch** in `config.yaml`:

```yaml
backend:
  name: "my_backend"
```

The pipeline handles all resampling, length alignment, and loudness normalization —
your backend only needs to return a waveform at `input_sample_rate`.

---

## CPU / quality / reference-length tradeoffs

| Factor | Effect |
|---|---|
| **`num_threads`** | More threads → faster WavLM feature extraction. Set to physical core count. On a 4-core laptop start with `4`; on a workstation try `8`–`16`. |
| **WavLM-Large CPU cost** | The dominant bottleneck. Expect 3–10× realtime (a 10 s clip takes 30–100 s on typical hardware). No smaller CPU-friendly WavLM option exists in kNN-VC. |
| **`topk`** | Higher k (8–16) produces smoother pitch but increases memory and time linearly with k. Default `4` balances quality and speed well. |
| **Amount of target audio** | More reference clips → richer matching set → better voice quality. A 5–10 s clip is the minimum; 30–60 s is ideal. Pass multiple paths in `target_refs`. |
| **`wavlm_layer`** | Layer 6 (default) captures speaker-discriminative features. Layers near 24 encode more semantic content; layers near 0 are more acoustic/phonetic. |
| **`output_sample_rate`** | kNN-VC outputs 16 kHz; setting `0` resamples back to the source rate (e.g. 44100 Hz), adding a small upsampling cost with no quality gain above 8 kHz. |

---

## Length alignment guarantee

The pipeline enforces `output_samples == round(content_duration_sec × out_sr)` exactly,
by trimming or zero-padding the backend's output before saving. This is a hard requirement
when remuxing onto video — any drift causes audio/video desync.

The corrected drift is printed in the one-line summary at the end of each CLI run and
shown in the Streamlit status panel.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use only synthetic audio (sine waves + random noise) and a mocked model backend,
so they run in seconds with no network access or GPU needed.

---

## License

MIT
