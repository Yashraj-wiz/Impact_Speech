"""
Stage 5 — Model inference for Task 1 (Classification).
Plan reference: Implementation Plan §5.1, §5.2, §5.3

Reads the master manifest and runs each model over every Task-1 stimulus.  For Task 1
the prompt constrains the model to pick one label from the item's label_set (regex +
sequence-match parse per plan §5.2).  Parse failures are logged, not silently dropped.

Outputs:
  acousticscene-bias/inference/{model_key}/task1.jsonl
  Each row: {stimulus_id, item_id, scene, snr, condition, raw_text, parsed_label, parse_ok}

Usage:
  python scripts/07_inference_task1.py --model qwen2audio [--smoke-test] [--full-precision]

  --model KEY       Model key (qwen25omni | qwen2audio | kimiaudio | glm4voice | phi4 |
                    audioflamingo3). Required.
  --smoke-test      Run on 3 items × 2 scenes × 1 SNR only.
  --full-precision  Load model in float16 without 8-bit quantization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Task-1 prompt builder ─────────────────────────────────────────────────────

def build_prompt(label_set: List[str]) -> str:
    options = ", ".join(f'"{l}"' for l in label_set)
    return (
        "You will hear a spoken question from a user. "
        "Your task is to classify the user's situation by choosing exactly ONE category "
        f"from the following options: {options}. "
        "Respond with only the exact category name and nothing else."
    )


# ── Label parser ──────────────────────────────────────────────────────────────

def parse_label(raw: str, label_set: List[str]) -> Optional[str]:
    """
    Extracts the predicted label from the model's raw text output.

    Strategy (plan §5.2 'parse defensively'):
      1. Exact match (case-insensitive, stripped).
      2. Prefix match — label appears at the start of the response.
      3. Substring match — first label found anywhere in the response.
    Returns None on failure (caller logs the failure).
    """
    text = raw.strip().lower()

    # 1. Exact match
    for label in label_set:
        if text == label.lower():
            return label

    # 2. Prefix match
    for label in label_set:
        if text.startswith(label.lower()):
            return label

    # 3. Substring match (pick the one whose cleaned form appears earliest)
    best_pos = len(text) + 1
    best_label = None
    for label in label_set:
        pos = text.find(label.lower())
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_label = label

    return best_label


# ── Load manifest + items ────────────────────────────────────────────────────

def load_manifest(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_items(path: Path) -> Dict[str, Dict]:
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                items[obj["id"]] = obj
    return items


# ── Filter manifest for smoke test ────────────────────────────────────────────

def filter_smoke(rows: List[Dict]) -> List[Dict]:
    allowed_items  = {f"t1_{i:03d}" for i in range(1, config.SMOKE_ITEMS + 1)}
    allowed_scenes = set(config.SMOKE_SCENES) | {"clean"}
    allowed_snrs   = {f"{s}dB" for s in config.SMOKE_SNRS} | {"clean"}
    return [
        r for r in rows
        if r["item_id"] in allowed_items
        and r["scene"] in allowed_scenes
        and r["snr"] in allowed_snrs
    ]


# ── Run inference ─────────────────────────────────────────────────────────────

def run_inference(
    model_key: str,
    rows: List[Dict],
    item_lookup: Dict[str, Dict],
    out_path: Path,
    load_in_8bit: bool = True,
) -> None:
    # Import adapters (deferred to avoid importing torch at module level)
    sys.path.insert(0, str(config.AB))
    from inference.adapters import load_adapter

    model = load_adapter(model_key, load_in_8bit=load_in_8bit)
    log.info("Model loaded: %s", model)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from last completed row if output already exists
    done_ids: set = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line).get("stimulus_id", ""))
        log.info("Resuming — %d rows already done.", len(done_ids))

    parse_failures = 0
    total = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for row in rows:
            stim_id = row["stimulus_id"]
            if stim_id in done_ids:
                continue

            item = item_lookup.get(row["item_id"], {})
            label_set: List[str] = item.get("label_set", [])
            if not label_set:
                log.warning("No label_set for item %s, skipping.", row["item_id"])
                continue

            audio_path = Path(row["out_path"])
            if not audio_path.exists():
                log.warning("Audio file missing: %s", audio_path)
                continue

            system_prompt = build_prompt(label_set)

            try:
                raw_text = model.generate(
                    audio_path=audio_path,
                    system_prompt=system_prompt,
                    max_new_tokens=32,
                )
            except Exception as exc:
                log.error("Inference failed for %s: %s", stim_id, exc)
                raw_text = ""

            parsed = parse_label(raw_text, label_set)
            parse_ok = parsed is not None
            if not parse_ok:
                parse_failures += 1
                log.warning("Parse failure — stim_id=%s  raw=%r", stim_id, raw_text[:80])

            record = {
                "stimulus_id":  stim_id,
                "item_id":      row["item_id"],
                "scene":        row["scene"],
                "snr":          row["snr"],
                "condition":    row["condition"],
                "model":        model_key,
                "raw_text":     raw_text,
                "parsed_label": parsed,
                "parse_ok":     parse_ok,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            total += 1

    log.info("Inference complete: %d rows, %d parse failures (%.1f%%)",
             total, parse_failures, 100 * parse_failures / max(total, 1))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task-1 inference for one model.")
    parser.add_argument("--model", required=True,
                        choices=list(config.MODEL_IDS.keys()),
                        help="Model key to run.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="3 items × 2 scenes × 1 SNR only.")
    parser.add_argument("--full-precision", action="store_true",
                        help="Load in float16 without 8-bit quantization.")
    args = parser.parse_args()

    manifest_path = (
        config.MANIFESTS / "task1_manifest_smoke.csv"
        if args.smoke_test
        else config.MANIFESTS / "task1_manifest.csv"
    )
    if not manifest_path.exists():
        log.error("Manifest not found: %s. Run 05_build_manifest.py first.", manifest_path)
        sys.exit(1)

    rows = load_manifest(manifest_path)
    item_lookup = load_items(config.ITEMBANKS / "task1_items.jsonl")

    if args.smoke_test:
        rows = filter_smoke(rows)
        log.info("SMOKE TEST — %d rows after filter.", len(rows))

    out_path = config.INFERENCE / args.model / "task1.jsonl"

    log.info("Running %s on %d stimuli -> %s", args.model, len(rows), out_path)
    run_inference(
        model_key=args.model,
        rows=rows,
        item_lookup=item_lookup,
        out_path=out_path,
        load_in_8bit=not args.full_precision,
    )


if __name__ == "__main__":
    main()
