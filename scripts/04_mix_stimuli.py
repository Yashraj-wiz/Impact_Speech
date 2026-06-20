"""
Stage 3b — Build stimuli by mixing TTS speech over curated scene clips at target SNRs.
Plan reference: Implementation Plan §3.1

For each (item, scene, SNR, condition) tuple, mixes foreground speech over a randomly
selected scene exemplar clip using SNR-controlled placement.  Also produces:
  - clean baseline  : speech only (no background)
  - scrambled scene : phase-scrambled scene at same SNR (control)

Outputs: acousticscene-bias/data/stimuli/task1/{item_id}/{scene}/{snr}dB/{condition}.wav
  conditions: c1 (fixed voice), clean, scrambled

Usage:
  python scripts/04_mix_stimuli.py [--smoke-test]

  --smoke-test   3 items × 2 scenes × 1 SNR only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

random.seed(42)


# ── Phase scramble (semantic-content control) ─────────────────────────────────

def _phase_scramble(audio: np.ndarray) -> np.ndarray:
    fft       = np.fft.rfft(audio)
    magnitudes = np.abs(fft)
    phases     = np.random.uniform(0, 2 * np.pi, len(magnitudes))
    scrambled  = magnitudes * np.exp(1j * phases)
    return np.fft.irfft(scrambled, n=len(audio)).astype(np.float32)


# ── Loudness helpers ──────────────────────────────────────────────────────────

def _lufs(audio: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    try:
        val = meter.integrated_loudness(audio)
        return float(val) if not (np.isinf(val) or np.isnan(val)) else -70.0
    except Exception:
        return -70.0


def _norm_to(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    measured = _lufs(audio, sr)
    if measured <= -69.0:
        return audio
    return pyln.normalize.loudness(audio, measured, target_lufs).astype(np.float32)


# ── SNR-controlled mix ────────────────────────────────────────────────────────

def _mix_at_snr(
    speech: np.ndarray,
    scene: np.ndarray,
    snr_db: float,
    sr: int,
    bg_lufs: float = config.BG_LUFS,
    speech_lufs: float = config.SPEECH_LUFS,
) -> np.ndarray:
    """
    Mix speech over scene at the requested SNR.

    Protocol (plan §3.1):
      1. Normalise background to bg_lufs.
      2. Set speech level = bg_lufs + snr_db (so SNR = speech_level - bg_level = snr_db).
      3. Sum and clip.

    We implement this without Scaper to avoid the sox dependency on Windows.
    The result is functionally equivalent to the Scaper pseudocode in the plan.
    """
    # Ensure lengths match: loop background, trim to speech length
    n = len(speech)
    if len(scene) < n:
        reps = int(np.ceil(n / len(scene)))
        scene = np.tile(scene, reps)
    scene = scene[:n]

    # Normalise background to reference level
    bg = _norm_to(scene, sr, bg_lufs)

    # Derive required speech level from SNR
    required_speech_lufs = bg_lufs + snr_db
    fg = _norm_to(speech, sr, required_speech_lufs)

    mixed = fg + bg
    peak  = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


# ── Load item bank ────────────────────────────────────────────────────────────

def _load_items(path: Path) -> List[Dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── Mix all stimuli ───────────────────────────────────────────────────────────

def build_stimuli(
    items: List[Dict],
    scenes: List[str],
    snrs: List[int],
    speech_dir: Path,
    scene_dir: Path,
    out_dir: Path,
) -> List[Dict]:
    """
    Returns a list of manifest rows (dicts) for each generated stimulus.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for item in tqdm(items, desc="Items"):
        item_id = item["id"]
        speech_path = speech_dir / f"{item_id}.wav"
        if not speech_path.exists():
            log.warning("  %s: TTS file missing, skipping.", item_id)
            continue

        speech, sr = sf.read(str(speech_path), always_2d=False)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        speech = speech.astype(np.float32)

        # ── Clean baseline ────────────────────────────────────────────────────
        clean_dir = out_dir / item_id / "clean" / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_path = clean_dir / "c1.wav"
        if not clean_path.exists():
            fg_clean = _norm_to(speech, config.SAMPLE_RATE, config.SPEECH_LUFS)
            sf.write(str(clean_path), fg_clean, config.SAMPLE_RATE)

        manifest_rows.append({
            "stimulus_id": f"{item_id}__clean__clean__c1",
            "task": "task1",
            "item_id": item_id,
            "scene": "clean",
            "snr": "clean",
            "condition": "c1",
            "fg_path": str(speech_path),
            "bg_path": "",
            "out_path": str(clean_path),
        })

        # ── Scenes × SNRs ─────────────────────────────────────────────────────
        for scene in scenes:
            clips = sorted((scene_dir / scene).glob("*.wav"))
            if not clips:
                log.warning("  No clips for scene '%s', skipping.", scene)
                continue

            for snr_db in snrs:
                # Pick a random scene exemplar
                bg_path = random.choice(clips)
                bg, bg_sr = sf.read(str(bg_path), always_2d=False)
                if bg.ndim > 1:
                    bg = bg.mean(axis=1)
                if bg_sr != config.SAMPLE_RATE:
                    bg = librosa.resample(bg, orig_sr=bg_sr, target_sr=config.SAMPLE_RATE)
                bg = bg.astype(np.float32)

                snr_tag = f"{snr_db}dB"

                # C1 mixed
                mix_dir = out_dir / item_id / scene / snr_tag
                mix_dir.mkdir(parents=True, exist_ok=True)
                mix_path = mix_dir / "c1.wav"
                if not mix_path.exists():
                    mixed = _mix_at_snr(speech, bg, snr_db, config.SAMPLE_RATE)
                    sf.write(str(mix_path), mixed, config.SAMPLE_RATE)

                manifest_rows.append({
                    "stimulus_id": f"{item_id}__{scene}__{snr_tag}__c1",
                    "task": "task1",
                    "item_id": item_id,
                    "scene": scene,
                    "snr": snr_tag,
                    "condition": "c1",
                    "fg_path": str(speech_path),
                    "bg_path": str(bg_path),
                    "out_path": str(mix_path),
                })

                # Scrambled-scene control
                scr_path = mix_dir / "scrambled.wav"
                if not scr_path.exists():
                    bg_scr = _phase_scramble(bg)
                    mixed_scr = _mix_at_snr(speech, bg_scr, snr_db, config.SAMPLE_RATE)
                    sf.write(str(scr_path), mixed_scr, config.SAMPLE_RATE)

                manifest_rows.append({
                    "stimulus_id": f"{item_id}__{scene}__{snr_tag}__scrambled",
                    "task": "task1",
                    "item_id": item_id,
                    "scene": scene,
                    "snr": snr_tag,
                    "condition": "scrambled",
                    "fg_path": str(speech_path),
                    "bg_path": str(bg_path),
                    "out_path": str(scr_path),
                })

    return manifest_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mix Task-1 stimuli.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="3 items × 2 scenes × 1 SNR.")
    args = parser.parse_args()

    items = _load_items(config.ITEMBANKS / "task1_items.jsonl")
    speech_dir = config.SPEECH_TTS / "task1"
    scene_dir  = config.SCENES_CURATED
    out_dir    = config.STIMULI / "task1"

    if args.smoke_test:
        items  = items[: config.SMOKE_ITEMS]
        scenes = config.SMOKE_SCENES
        snrs   = config.SMOKE_SNRS
        log.info("SMOKE TEST — %d items × %d scenes × %d SNRs.", len(items), len(scenes), len(snrs))
    else:
        scenes = config.SCENES
        snrs   = config.SNR_LEVELS

    rows = build_stimuli(items, scenes, snrs, speech_dir, scene_dir, out_dir)

    log.info("Mixing complete: %d stimuli generated.", len(rows))

    # Spot-check a few
    sample = rows[:3] if len(rows) >= 3 else rows
    for r in sample:
        p = Path(r["out_path"])
        if p.exists():
            data, sr = sf.read(str(p))
            log.info("  %s  %.2f s  SNR=%s", p.name, len(data) / sr, r["snr"])
        else:
            log.warning("  Missing: %s", r["out_path"])


if __name__ == "__main__":
    main()
