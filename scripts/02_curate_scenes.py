"""
Stage 2 — Curate scene clips: select, trim, high-pass filter, loudness-normalise.
Plan reference: Implementation Plan §2.2

For each of the 8 scenes, selects up to CLIPS_PER_SCENE distinct exemplar clips from the
raw dataset pool, trims them to a common duration, applies a 60 Hz high-pass filter to
remove DC/low-frequency noise, and loudness-normalises to BG_LUFS using pyloudnorm.

Outputs: acousticscene-bias/data/scenes_curated/{scene}/{scene}_{idx:03d}.wav

Usage:
  python scripts/02_curate_scenes.py [--smoke-test] [--scene hospital office ...]

  --smoke-test   Process only 2 clips per scene for the 2 smoke-test scenes.
  --scene X ...  Curate only the named scene(s).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy import signal
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

# ── Target clip duration: will be updated once TTS speech is rendered. ────────
# We use a safe default here; 04_mix_stimuli.py pads/trims to the actual speech length.
DEFAULT_DURATION_S = 8.0

# ── ESC-50 category folder names (from the ESC-50 audio/ESC-50 structure) ─────
# ESC-50 audio files are in audio/ with filenames like NNN-NNNN-N-NN.wav
# and the CSV has: filename, fold, target, category, esc10, src_file, take
ESC50_AUDIO_DIR  = config.SCENES_RAW / "esc50" / "audio"
ESC50_META_CSV   = config.SCENES_RAW / "esc50" / "meta" / "esc50.csv"

TAU_AUDIO_DIR    = config.SCENES_RAW / "tau_urban" / "TAU-urban-acoustic-scenes-2019-development" / "audio"
MUSAN_NOISE_DIR  = config.SCENES_RAW / "musan" / "noise"
FSD50K_DEV_DIR   = config.SCENES_RAW / "fsd50k" / "dev_audio" / "FSD50K.dev_audio"
FSD50K_META_CSV  = config.SCENES_RAW / "fsd50k" / "ground_truth" / "FSD50K.ground_truth" / "dev.csv"


# ── Helper: collect candidate files from ESC-50 ───────────────────────────────

def _esc50_candidates(categories: List[str]) -> List[Path]:
    if not ESC50_META_CSV.exists():
        log.warning("ESC-50 metadata not found at %s. Run 01_download_datasets.py first.", ESC50_META_CSV)
        return []
    candidates = []
    with open(ESC50_META_CSV, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            fname, _, _, cat = parts[0], parts[1], parts[2], parts[3]
            if cat in categories:
                audio_path = ESC50_AUDIO_DIR / fname
                if audio_path.exists():
                    candidates.append(audio_path)
    return candidates


# ── Helper: collect candidate files from TAU Urban ────────────────────────────

def _tau_candidates(scene_labels: List[str]) -> List[Path]:
    if not TAU_AUDIO_DIR.exists():
        log.warning("TAU audio not found at %s. Run 01_download_datasets.py first.", TAU_AUDIO_DIR)
        return []
    candidates = []
    for p in TAU_AUDIO_DIR.glob("*.wav"):
        # TAU filenames: {scene_label}-{city}-{location}-{segment}.wav
        label = p.stem.split("-")[0]
        if label in scene_labels:
            candidates.append(p)
    return candidates


# ── Helper: collect candidate files from FSD50K ───────────────────────────────

def _fsd50k_candidates(label_keywords: List[str]) -> List[Path]:
    if not FSD50K_META_CSV.exists() or not FSD50K_DEV_DIR.exists():
        log.warning("FSD50K metadata/audio not found. Run 01_download_datasets.py first.")
        return []
    candidates = []
    with open(FSD50K_META_CSV, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            fname, labels_col = parts[0], parts[2]
            for kw in label_keywords:
                if kw.lower() in labels_col.lower():
                    audio_path = FSD50K_DEV_DIR / f"{fname}.wav"
                    if audio_path.exists():
                        candidates.append(audio_path)
                        break
    return candidates


# ── Build candidate list per scene ───────────────────────────────────────────

def _collect_candidates(scene: str) -> List[Path]:
    esc50_cats    = config.ESC50_SCENE_MAP.get(scene, [])
    tau_labels    = config.TAU_SCENE_MAP.get(scene, [])

    # FSD50K keyword hints per scene (matching document §1.2 source notes)
    fsd50k_map = {
        "hospital":         ["Siren", "Alarm"],
        "office":           ["Keyboard", "Typing", "Computer"],
        "factory":          ["Engine", "Buzz", "Crushing"],
        "construction":     ["Drill", "Power_tool", "Sawing", "Hammer"],
        "kitchen":          ["Cooking", "Frying", "Dishes_and_pots_and_pans", "Water_tap_and_faucet", "Sink_(filling_or_washing)"],
        "classroom":        ["Writing"],
        "public_transport": ["Bus", "Subway_and_metro_and_underground", "Train", "Rail_transport"],
        "infant_cry":       ["Baby", "Crying", "Infant"],
    }
    fsd50k_kws = fsd50k_map.get(scene, [])

    candidates: List[Path] = []
    candidates.extend(_esc50_candidates(esc50_cats))
    candidates.extend(_tau_candidates(tau_labels))
    candidates.extend(_fsd50k_candidates(fsd50k_kws))

    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ── Audio processing ──────────────────────────────────────────────────────────

def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 60.0) -> np.ndarray:
    sos = signal.butter(4, cutoff_hz / (sr / 2), btype="high", output="sos")
    return signal.sosfilt(sos, audio)


def _loudness_norm(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        measured = meter.integrated_loudness(audio)
        if np.isinf(measured) or np.isnan(measured):
            return audio
        return pyln.normalize.loudness(audio, measured, target_lufs)
    except Exception:
        return audio


def _trim_pad(audio: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    target_len = int(duration_s * sr)
    if len(audio) >= target_len:
        return audio[:target_len]
    # pad with silence
    return np.pad(audio, (0, target_len - len(audio)))


def process_clip(src: Path, duration_s: float) -> Optional[np.ndarray]:
    try:
        audio, sr = sf.read(str(src), always_2d=False)
    except Exception as exc:
        log.debug("Cannot read %s: %s", src, exc)
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

    audio = _highpass(audio, config.SAMPLE_RATE)
    audio = _trim_pad(audio, config.SAMPLE_RATE, duration_s)
    audio = _loudness_norm(audio, config.SAMPLE_RATE, config.BG_LUFS)

    if np.max(np.abs(audio)) < 1e-6:
        return None
    return audio.astype(np.float32)


# ── Per-scene curation ────────────────────────────────────────────────────────

def curate_scene(
    scene: str,
    n_clips: int = config.CLIPS_PER_SCENE,
    duration_s: float = DEFAULT_DURATION_S,
) -> int:
    out_dir = config.SCENES_CURATED / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.wav"))
    if len(existing) >= n_clips:
        log.info("  %s: already has %d clips, skipping.", scene, len(existing))
        return len(existing)

    candidates = _collect_candidates(scene)
    if not candidates:
        log.warning("  %s: no candidate clips found. Check that raw datasets are downloaded.", scene)
        return 0

    random.shuffle(candidates)
    saved = 0
    for src in tqdm(candidates, desc=f"  {scene}", leave=False):
        if saved >= n_clips:
            break
        audio = process_clip(src, duration_s)
        if audio is None:
            continue
        out_path = out_dir / f"{scene}_{saved:03d}.wav"
        sf.write(str(out_path), audio, config.SAMPLE_RATE)
        saved += 1

    log.info("  %s: saved %d / %d clips.", scene, saved, n_clips)
    return saved


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Curate scene exemplar clips.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use only 2 clips per scene for 2 smoke-test scenes.")
    parser.add_argument("--scene", nargs="+", choices=config.SCENES, metavar="SCENE",
                        help="Curate only the named scene(s). Default: all 8.")
    args = parser.parse_args()

    config.SCENES_CURATED.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        scenes    = config.SMOKE_SCENES
        n_clips   = 2
        duration  = 5.0
        log.info("SMOKE TEST — processing %d clips per scene for: %s", n_clips, scenes)
    else:
        scenes   = args.scene or config.SCENES
        n_clips  = config.CLIPS_PER_SCENE
        duration = DEFAULT_DURATION_S

    for scene in scenes:
        n = curate_scene(scene, n_clips=n_clips, duration_s=duration)
        if n == 0:
            log.warning("Scene '%s' has 0 clips — manipulation checks will fail.", scene)

    log.info("Scene curation complete.")


if __name__ == "__main__":
    main()
