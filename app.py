#!/usr/bin/env python3
"""Streamlit UI for VoiceTransfer.

Launch with:
    streamlit run app.py

All pipeline logic is shared with the CLI (run.py) — no duplication.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VoiceTransfer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model cache ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_model(models_dir: str, prematched: bool, num_threads: int):
    """Load kNN-VC once and cache it for the lifetime of the Streamlit server.

    Changing any of the three arguments invalidates the cache and forces a reload.
    """
    import torch

    abs_dir = str(Path(models_dir).resolve())
    os.environ["TORCH_HOME"] = abs_dir
    os.environ["HF_HOME"] = abs_dir
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(num_threads)

    return torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=prematched,
        trust_repo=True,
        pretrained=True,
        device="cpu",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_upload(uploaded_file) -> str:
    """Write a Streamlit UploadedFile to a named temp file; return its path."""
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def _wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Encode a float32 numpy array to WAV bytes (for st.audio / st.download_button)."""
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _build_config(
    content_path: str,
    target_paths: list[str],
    output_path: str,
    models_dir: str,
    prematched: bool,
    topk: int,
    wavlm_layer: int,
    target_vad_level: int,
    pitch_enabled: bool,
    pitch_max_shift_semitones: float,
    bandwidth_enabled: bool,
    bandwidth_cutoff_hz: float,
    bandwidth_blend_gain: float,
    normalize_loudness: bool,
    target_lufs: float,
    output_sample_rate: int,
    num_threads: int,
):
    """Build an AppConfig from UI values — same dataclass used by the CLI."""
    from voicetransfer.config import (
        AppConfig, AudioConfig, BackendConfig, BandwidthConfig, DeviceConfig,
        KnnVcConfig, LengthConfig, LoggingConfig, MuxConfig, PathsConfig, PitchConfig,
    )
    return AppConfig(
        paths=PathsConfig(
            models_dir=models_dir,
            content_audio=content_path,
            target_refs=target_paths,
            output_audio=output_path,
        ),
        device=DeviceConfig(type="cpu", num_threads=num_threads),
        backend=BackendConfig(name="knn_vc"),
        knn_vc=KnnVcConfig(
            prematched=prematched, topk=topk, wavlm_layer=wavlm_layer,
            target_vad_level=target_vad_level,
        ),
        pitch=PitchConfig(
            enabled=pitch_enabled,
            max_shift_semitones=pitch_max_shift_semitones,
        ),
        bandwidth=BandwidthConfig(
            enabled=bandwidth_enabled,
            cutoff_hz=bandwidth_cutoff_hz,
            blend_gain=bandwidth_blend_gain,
        ),
        audio=AudioConfig(
            output_sample_rate=output_sample_rate,
            normalize_loudness=normalize_loudness,
            target_lufs=target_lufs,
        ),
        length=LengthConfig(enforce_exact=True, pad_mode="silence", warn_if_drift_ms=50.0),
        mux=MuxConfig(enabled=False),
        logging=LoggingConfig(level="WARNING"),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> dict:
    """Render sidebar controls and return their values as a dict."""
    with st.sidebar:
        st.header("⚙️ Settings")

        models_dir = st.text_input(
            "Models directory",
            value="./models",
            help="Downloaded weights are stored here (sets TORCH_HOME). "
                 "Changing this invalidates the cached model.",
        )

        st.subheader("kNN-VC")
        prematched = st.checkbox(
            "Prematched HiFiGAN",
            value=True,
            help="Recommended for zero-shot use. Uncheck to use the standard vocoder.",
        )
        topk = st.slider(
            "Top-k neighbours",
            min_value=1, max_value=16, value=4,
            help="Higher k → smoother pitch, slower inference. "
                 "Increase to 8-16 when target reference is short (<30 s).",
        )
        wavlm_layer = st.slider(
            "WavLM layer",
            min_value=0, max_value=23, value=6,
            help="WavLM-Large transformer layer used for speaker features. "
                 "Layer 6 generalises well; layers closer to 24 are more semantic.",
        )
        target_vad_level = st.slider(
            "Target VAD aggressiveness",
            min_value=0, max_value=7, value=3,
            help="Strips silence from the target reference before building the kNN pool. "
                 "0 = keep all frames (risks matching speech to silence frames); "
                 "3 = recommended (removes silence, keeps all voiced speech); "
                 "7 = aggressive (only for very noisy references).",
        )

        st.subheader("🎵 Pitch alignment")
        pitch_enabled = st.checkbox(
            "Auto pitch shift",
            value=True,
            help="Estimate median F0 of content and target, then shift content pitch "
                 "to match before kNN conversion.  Critical when speakers differ in "
                 "gender or natural pitch range.",
        )
        pitch_max_shift_semitones = st.slider(
            "Max shift (semitones)",
            min_value=1.0, max_value=24.0, value=12.0, step=0.5,
            disabled=not pitch_enabled,
            help="Clamp the computed shift to this range.  ±12 st = ±1 octave. "
                 "Larger values risk phase-vocoder artifacts.",
        )

        st.subheader("📡 Bandwidth extension")
        bandwidth_enabled = st.checkbox(
            "Blend high frequencies",
            value=True,
            help="kNN-VC/HiFiGAN is band-limited to 8 kHz.  This blends the "
                 ">7.5 kHz content from the original audio back in, restoring "
                 "sibilance and air.  Only active when output SR > 16 kHz.",
        )
        bandwidth_blend_gain = st.slider(
            "HF blend gain",
            min_value=0.0, max_value=1.0, value=0.8, step=0.05,
            disabled=not bandwidth_enabled,
            help="How much of the high-frequency content to mix in. "
                 "0 = off, 1 = full level of original HF.",
        )
        bandwidth_cutoff_hz = st.number_input(
            "HF cutoff (Hz)",
            min_value=4000, max_value=12000, value=7500, step=500,
            disabled=not bandwidth_enabled,
            help="Frequencies above this are taken from the original content audio.",
        )

        st.subheader("Audio output")
        normalize_loudness = st.checkbox(
            "Normalize loudness (EBU R128)",
            value=True,
        )
        target_lufs = st.number_input(
            "Target LUFS",
            min_value=-40.0, max_value=-6.0, value=-23.0, step=0.5,
            disabled=not normalize_loudness,
            help="-23 LUFS is the EBU R128 broadcast standard.",
        )
        output_sample_rate = st.number_input(
            "Output sample rate (0 = match source)",
            min_value=0, max_value=48000, value=0, step=100,
            help="kNN-VC outputs 16 kHz; the pipeline resamples to this rate. "
                 "0 matches the content file's native rate.",
        )

        st.subheader("Performance")
        num_threads = st.slider(
            "CPU threads",
            min_value=1, max_value=16, value=4,
            help="Passed to torch.set_num_threads(). "
                 "Set to the number of physical cores for best throughput.",
        )

    return dict(
        models_dir=models_dir,
        prematched=prematched,
        topk=topk,
        wavlm_layer=wavlm_layer,
        target_vad_level=int(target_vad_level),
        pitch_enabled=pitch_enabled,
        pitch_max_shift_semitones=float(pitch_max_shift_semitones),
        bandwidth_enabled=bandwidth_enabled,
        bandwidth_cutoff_hz=float(bandwidth_cutoff_hz),
        bandwidth_blend_gain=float(bandwidth_blend_gain),
        normalize_loudness=normalize_loudness,
        target_lufs=float(target_lufs),
        output_sample_rate=int(output_sample_rate),
        num_threads=num_threads,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings = _sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎙️ VoiceTransfer")
    st.caption(
        "CPU-only zero-shot voice conversion via "
        "[kNN-VC](https://github.com/bshall/knn-vc). "
        "No GPU • No text/ASR • No alignment."
    )
    st.divider()

    # ── File uploaders ────────────────────────────────────────────────────────
    col_content, col_target = st.columns(2)

    with col_content:
        st.subheader("Content audio")
        st.caption("Speech you want to convert.")
        content_upload = st.file_uploader(
            "content_audio",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
            key="content_upload",
        )
        if content_upload:
            st.audio(content_upload)

    with col_target:
        st.subheader("Target speaker reference(s)")
        st.caption("Voice to convert into. More clips → better kNN match pool.")
        target_uploads = st.file_uploader(
            "target_refs",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="target_uploads",
        )
        if target_uploads:
            for uf in target_uploads:
                st.audio(uf, format=f"audio/{Path(uf.name).suffix.lstrip('.')}")

    st.divider()

    # ── Convert button ────────────────────────────────────────────────────────
    can_convert = bool(content_upload and target_uploads)
    if not can_convert:
        st.info(
            "Upload a **content** audio file and at least one **target speaker** "
            "reference to enable conversion."
        )

    convert_clicked = st.button(
        "🔄  Convert",
        type="primary",
        disabled=not can_convert,
        use_container_width=True,
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if convert_clicked and can_convert:
        # Rewind Streamlit UploadedFile buffers (they are seekable)
        content_upload.seek(0)
        for uf in target_uploads:
            uf.seek(0)

        temp_files: list[str] = []
        try:
            with st.status("Converting…", expanded=True) as status:
                st.write("📥 Saving uploaded files to temporary storage…")
                content_tmp = _save_upload(content_upload)
                temp_files.append(content_tmp)

                target_tmps: list[str] = []
                for uf in target_uploads:
                    p = _save_upload(uf)
                    target_tmps.append(p)
                    temp_files.append(p)

                out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                out_tmp.close()
                output_tmp = out_tmp.name
                temp_files.append(output_tmp)

                st.write(
                    "🤖 Loading model "
                    "(cached after first load — first run downloads ~650 MB)…"
                )
                model = _load_model(
                    settings["models_dir"],
                    settings["prematched"],
                    settings["num_threads"],
                )

                st.write("🔄 Running voice conversion pipeline…")
                cfg = _build_config(
                    content_path=content_tmp,
                    target_paths=target_tmps,
                    output_path=output_tmp,
                    **settings,
                )
                from voicetransfer.pipeline import run_pipeline
                stats = run_pipeline(cfg, model)

                status.update(label="✅ Conversion complete!", state="complete")

            # ── Results ───────────────────────────────────────────────────────
            out_wav, out_sr = sf.read(output_tmp, dtype="float32")
            out_bytes = _wav_to_bytes(out_wav, out_sr)

            st.subheader("🔊 Converted audio")
            st.audio(out_bytes, format="audio/wav")

            info_col, dl_col = st.columns([3, 1])
            with info_col:
                dur = len(out_wav) / out_sr
                st.caption(
                    f"Duration: **{dur:.3f} s**  ·  "
                    f"Sample rate: **{out_sr} Hz**  ·  "
                    f"Samples: **{len(out_wav):,}**"
                )
            with dl_col:
                st.download_button(
                    "⬇️ Download WAV",
                    data=out_bytes,
                    file_name="converted.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )

            # ── Performance stats ──────────────────────────────────────────────
            st.subheader("📊 Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total time",      f"{stats.total_s:.1f} s")
            m2.metric("Realtime factor", f"{stats.realtime_factor:.1f}×",
                      help="Seconds of compute per second of audio. Lower is faster.")
            m3.metric("Peak RAM",        f"{stats.peak_ram_mb:.0f} MB",
                      delta=f"+{stats.net_ram_mb:.0f} MB vs baseline",
                      delta_color="off")
            m4.metric("Output size",     f"{stats.output_size_mb:.1f} MB")

            with st.expander("Step-by-step breakdown"):
                step_data = {
                    "Step": [s.name for s in stats.steps],
                    "Time (s)": [f"{s.duration_s:.2f}" for s in stats.steps],
                    "RAM at end (MB)": [f"{s.ram_mb:.0f}" for s in stats.steps],
                }
                st.table(step_data)

        except Exception as exc:
            st.error(f"**Conversion failed:** {exc}")
            raise

        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
