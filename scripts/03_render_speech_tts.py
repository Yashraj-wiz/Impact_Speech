"""
Stage 3a — Render TTS speech for all Task-1 items (condition C1: fixed voice).
Plan reference: Implementation Plan §2.3

Synthesises each item's prompt text using Kokoro-TTS with the fixed C1 voice (af_heart),
then loudness-normalises the output to SPEECH_LUFS.

Outputs: acousticscene-bias/data/speech_tts/task1/{item_id}.wav

Usage:
  python scripts/03_render_speech_tts.py [--smoke-test]

  --smoke-test   Render only the first 3 items.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_items(path: Path) -> List[Dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _loudness_norm(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        measured = meter.integrated_loudness(audio)
        if np.isinf(measured) or np.isnan(measured):
            return audio
        return pyln.normalize.loudness(audio, measured, target_lufs)
    except Exception:
        return audio


def render_items(items: List[Dict], out_dir: Path) -> None:
    try:
        from kokoro import KPipeline
    except ImportError:
        log.error("kokoro is not installed. Run: pip install kokoro")
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code=config.TTS_LANG)

    for item in tqdm(items, desc="TTS rendering"):
        item_id = item["id"]
        out_path = out_dir / f"{item_id}.wav"
        if out_path.exists():
            log.debug("  %s already exists, skipping.", item_id)
            continue

        text = item["prompt"]
        try:
            # Kokoro returns a generator of (audio_chunk, sample_rate, phonemes)
            audio_chunks = []
            sr_out = None
            for audio_chunk, sample_rate, _ in pipeline(
                text,
                voice=config.TTS_VOICE_C1,
                speed=1.0,
                split_pattern=r"\n+",
            ):
                audio_chunks.append(audio_chunk)
                sr_out = sample_rate

            if not audio_chunks or sr_out is None:
                log.warning("  %s: empty TTS output.", item_id)
                continue

            audio = np.concatenate(audio_chunks).astype(np.float32)

            # Resample to target sample rate if needed
            if sr_out != config.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr_out, target_sr=config.SAMPLE_RATE)

            audio = _loudness_norm(audio, config.SAMPLE_RATE, config.SPEECH_LUFS)
            sf.write(str(out_path), audio, config.SAMPLE_RATE)
            log.debug("  %s -> %.1f s", item_id, len(audio) / config.SAMPLE_RATE)

        except Exception as exc:
            log.error("  %s: TTS failed: %s", item_id, exc)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Render TTS speech for Task-1 items.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Render only the first 3 items.")
    args = parser.parse_args()

    items = _load_items(config.ITEMBANKS / "task1_items.jsonl")

    if args.smoke_test:
        items = items[: config.SMOKE_ITEMS]
        log.info("SMOKE TEST — rendering %d items.", len(items))

    out_dir = config.SPEECH_TTS / "task1"
    render_items(items, out_dir)

    rendered = sorted(out_dir.glob("*.wav"))
    log.info("TTS complete: %d / %d files in %s", len(rendered), len(items), out_dir)

    if args.smoke_test:
        for p in rendered:
            import soundfile as sf
            data, sr = sf.read(str(p))
            log.info("  %s  %.2f s  sr=%d", p.name, len(data) / sr, sr)


if __name__ == "__main__":
    main()
