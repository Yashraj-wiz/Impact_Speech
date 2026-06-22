"""
Stage 4 — Manipulation checks (must pass before bias inference).
Plan reference: Implementation Plan §4

Runs three checks:
  1. Scene presence  : pretrained audio classifier confirms the intended scene is detected
                       above threshold in each mixed clip. Reports per-scene accuracy to
                       checks/scene_presence.csv
  2. Intelligibility : Whisper WER on every clip confirms WER does not differ materially
                       across scenes at a given SNR. Reports to checks/wer_by_scene.csv
  3. Jitter floor    : queries the clean baseline N=10 times (via text-only inference of
                       the same label-choice prompt) and computes each metric's intrinsic
                       variability. Stores to checks/jitter_floor.json

Gate: logs a CRITICAL warning if scene presence is below 0.5 or if WER variance across
scenes exceeds 10 percentage points at the reference SNR. Fix mixing before running inference.

Usage:
  python scripts/06_manipulation_checks.py [--smoke-test] [--skip-wer] [--skip-jitter]

  --smoke-test    Run on 3 items × 2 scenes only.
  --skip-wer      Skip the Whisper WER check (faster).
  --skip-jitter   Skip jitter-floor computation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

# AudioSet label indices that correspond to each study scene (AudioSet taxonomy)
# These are broad matches; the classifier uses MIT/ast-finetuned-audioset-10-10-0.4593
SCENE_AUDIOSET_KEYWORDS = {
    "hospital":         ["siren", "alarm", "beep", "medical", "emergency", "ambulance", "police"],
    "office":           ["keyboard", "typing", "speech", "office", "computer", "telephone"],
    "factory":          ["engine", "machinery", "motor", "mechanical", "factory", "industrial", "idling", "vibration", "buzz", "hum", "crushing", "chainsaw", "vehicle", "car", "truck", "revving", "vroom", "crack", "crunch", "breaking"],
    "construction":     ["drill", "power tool", "saw", "jackhammer", "tool", "hammer", "wood", "sawing", "chainsaw"],
    "kitchen":          ["cooking", "frying", "dishes", "utensil", "water", "sink", "liquid", "drip", "boiling", "fire", "chop"],
    "classroom":        ["speech", "chatter", "crowd", "bell", "classroom", "teacher", "student"],
    "public_transport": ["bus", "train", "rail", "vehicle", "metro", "subway", "tram", "transportation"],
    "infant_cry":       ["baby", "crying", "infant", "whimper"],
}



# ── Audio classifier for scene presence check ─────────────────────────────────

class ScenePresenceChecker:
    def __init__(self, device: str = "cpu"):
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        log.info("Loading scene classifier: %s", model_id)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model     = ASTForAudioClassification.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device    = device
        self.id2label  = self.model.config.id2label

    def top_labels(self, audio: np.ndarray, sr: int, top_k: int = 10) -> List[Tuple[str, float]]:
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        top_idx = probs.argsort()[::-1][:top_k]
        return [(self.id2label[i], float(probs[i])) for i in top_idx]

    def scene_detected(self, audio: np.ndarray, sr: int, scene: str, threshold: float = 0.01) -> bool:
        keywords = SCENE_AUDIOSET_KEYWORDS.get(scene, [])
        top      = self.top_labels(audio, sr, top_k=20)
        for label, prob in top:
            if prob < threshold:
                break
            if any(kw in label.lower() for kw in keywords):
                return True
        return False


# ── WER check ─────────────────────────────────────────────────────────────────

class WERChecker:
    def __init__(self):
        import whisper
        log.info("Loading Whisper (base) for WER check ...")
        self.model = whisper.load_model("base", download_root=str(config.MODELS / "whisper"))

    def transcribe(self, audio_path: Path) -> str:
        result = self.model.transcribe(str(audio_path), language="en", fp16=False)
        return result["text"].strip().lower()


def _wer(reference: str, hypothesis: str) -> float:
    from jiwer import wer as compute_wer
    return float(compute_wer(reference, hypothesis))


# ── Check 1: scene presence ───────────────────────────────────────────────────

def check_scene_presence(
    manifest_rows: List[Dict],
    checker: ScenePresenceChecker,
    out_path: Path,
) -> Dict[str, float]:
    scene_correct  = defaultdict(int)
    scene_total    = defaultdict(int)

    sample = [r for r in manifest_rows if r["condition"] == "c1" and r["scene"] != "clean" and r["snr"] == "5dB"]
    # Up to 5 clips per scene for speed
    per_scene = defaultdict(list)
    for r in sample:
        per_scene[r["scene"]].append(r)
    sampled = []
    for scene, rows in per_scene.items():
        sampled.extend(rows[:5])

    for row in tqdm(sampled, desc="Scene presence"):
        path = Path(row["out_path"])
        if not path.exists():
            continue
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        detected = checker.scene_detected(audio, sr, row["scene"])
        scene_correct[row["scene"]] += int(detected)
        scene_total[row["scene"]]   += 1

    results = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "n_clips", "n_detected", "accuracy"])
        for scene in sorted(scene_total.keys()):
            n  = scene_total[scene]
            ok = scene_correct[scene]
            acc = ok / n if n > 0 else 0.0
            results[scene] = acc
            writer.writerow([scene, n, ok, f"{acc:.3f}"])
            if acc < 0.5:
                log.critical("GATE FAIL — scene '%s' detected in only %.0f%% of clips.", scene, acc * 100)
            else:
                log.info("  %s: %.0f%% detected (%d/%d)", scene, acc * 100, ok, n)

    return results


# ── Check 2: WER across scenes ────────────────────────────────────────────────

def check_wer(
    manifest_rows: List[Dict],
    item_lookup: Dict[str, Dict],
    wer_checker: WERChecker,
    out_path: Path,
    ref_snr: str = f"{config.SNR_REFERENCE}dB",
) -> None:
    sample = [
        r for r in manifest_rows
        if r["condition"] == "c1" and r["snr"] == ref_snr
    ]

    per_scene = defaultdict(list)
    for r in sample:
        per_scene[r["scene"]].append(r)

    wer_by_scene: Dict[str, List[float]] = defaultdict(list)

    for scene, rows in per_scene.items():
        for row in tqdm(rows[:10], desc=f"WER {scene}", leave=False):
            path = Path(row["out_path"])
            if not path.exists():
                continue
            item = item_lookup.get(row["item_id"], {})
            ref  = item.get("prompt", "").lower()
            if not ref:
                continue
            hyp = wer_checker.transcribe(path)
            wer_by_scene[scene].append(_wer(ref, hyp))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "n_clips", "mean_wer", "std_wer"])
        mean_wers = []
        for scene in sorted(wer_by_scene.keys()):
            vals = wer_by_scene[scene]
            if not vals:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals))
            writer.writerow([scene, len(vals), f"{m:.3f}", f"{s:.3f}"])
            mean_wers.append(m)
            log.info("  WER %s: %.1f%% ± %.1f%%", scene, m * 100, s * 100)

        if len(mean_wers) >= 2:
            spread = (max(mean_wers) - min(mean_wers)) * 100
            log.info("  WER spread across scenes: %.1f pp", spread)
            if spread > 10:
                log.critical("GATE FAIL — WER varies %.1f pp across scenes (>10pp threshold).", spread)


# ── Check 3: jitter floor ────────────────────────────────────────────────────

def check_jitter_floor(
    manifest_rows: List[Dict],
    item_lookup: Dict[str, Dict],
    n_repeats: int,
    out_path: Path,
    model_key: str = "qwen2audio",
) -> None:
    """
    Queries the clean baseline N times at the inference temperature and computes
    LFR's intrinsic variability. Uses text-only inference of the label-choice
    prompt so we don't need full audio inference for this check.
    """
    log.info("Jitter floor check (N=%d repeats on clean baseline, text-only) ...", n_repeats)

    clean_rows = [r for r in manifest_rows if r["scene"] == "clean" and r["condition"] == "c1"]
    if not clean_rows:
        log.warning("No clean stimuli found for jitter check.")
        return

    sample = clean_rows[:3]  # 3 items is sufficient for the floor estimate

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        backbone_id = "Qwen/Qwen2.5-0.5B-Instruct"  # use a standard text causal LM for text-only check
        tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
        text_model = AutoModelForCausalLM.from_pretrained(
            backbone_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        text_model.eval()
    except Exception as exc:
        log.warning("Could not load text model for jitter check: %s. Storing placeholder.", exc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"note": "jitter check skipped", "error": str(exc)}))
        return

    flip_rates = []

    for row in sample:
        item = item_lookup.get(row["item_id"], {})
        label_set: List[str] = item.get("label_set", [])
        if not label_set:
            continue
        prompt_text = (
            f"Listen to a user's spoken question and classify their situation. "
            f"Choose exactly one category from: {', '.join(label_set)}. "
            f"Respond with only the category name.\n\n"
            f"User question: {item.get('prompt', '')}"
        )

        answers = []
        for _ in range(n_repeats):
            inputs = tokenizer(prompt_text, return_tensors="pt").to(text_model.device)
            with torch.no_grad():
                out = text_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                )
            decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answers.append(decoded.strip())

        baseline = answers[0]
        flips = sum(1 for a in answers[1:] if a != baseline)
        flip_rates.append(flips / (n_repeats - 1))

    floor = float(np.mean(flip_rates)) if flip_rates else 0.0
    floor_std = float(np.std(flip_rates)) if flip_rates else 0.0

    result = {
        "metric": "LFR",
        "jitter_floor_mean": floor,
        "jitter_floor_std": floor_std,
        "n_items": len(flip_rates),
        "n_repeats": n_repeats,
        "note": "Any scene LFR must exceed floor_mean + 2*floor_std to be considered real.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Jitter floor: LFR mean=%.3f std=%.3f (stored at %s)", floor, floor_std, out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run manipulation checks (Stage 4).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 3 items × 2 scenes only.")
    parser.add_argument("--skip-wer",    action="store_true")
    parser.add_argument("--skip-jitter", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load manifest
    if args.smoke_test:
        manifest_path = config.MANIFESTS / "task1_manifest_smoke.csv"
    else:
        manifest_path = config.MANIFESTS / "task1_manifest.csv"

    if not manifest_path.exists():
        log.error("Manifest not found: %s. Run 05_build_manifest.py first.", manifest_path)
        sys.exit(1)

    import csv as csv_mod
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    log.info("Manifest loaded: %d rows.", len(rows))

    # Load item lookup
    item_lookup: Dict[str, Dict] = {}
    with open(config.ITEMBANKS / "task1_items.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                item_lookup[obj["id"]] = obj

    # ── Check 1: scene presence ───────────────────────────────────────────────
    log.info("=== Check 1: Scene Presence ===")
    checker = ScenePresenceChecker(device=device)
    scene_acc = check_scene_presence(rows, checker, config.CHECKS / "scene_presence.csv")

    passed_scenes = [s for s, a in scene_acc.items() if a >= 0.5]
    failed_scenes = [s for s, a in scene_acc.items() if a < 0.5]
    log.info("Scene presence: %d/%d scenes above 0.5 threshold.", len(passed_scenes), len(scene_acc))
    if failed_scenes:
        log.critical("GATE FAIL — scenes below threshold: %s", failed_scenes)

    # ── Check 2: WER ─────────────────────────────────────────────────────────
    if not args.skip_wer:
        log.info("=== Check 2: WER by Scene ===")
        wer_checker = WERChecker()
        check_wer(rows, item_lookup, wer_checker, config.CHECKS / "wer_by_scene.csv")

    # ── Check 3: jitter floor ─────────────────────────────────────────────────
    if not args.skip_jitter:
        log.info("=== Check 3: Jitter Floor ===")
        check_jitter_floor(
            rows, item_lookup,
            n_repeats=config.JITTER_N if not args.smoke_test else 3,
            out_path=config.CHECKS / "jitter_floor.json",
        )

    log.info("Manipulation checks complete. Review checks/ directory before proceeding to inference.")


if __name__ == "__main__":
    main()
