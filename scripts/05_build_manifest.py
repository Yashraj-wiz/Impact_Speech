"""
Stage 3c — Build the master manifest CSV from the stimuli directory.
Plan reference: Implementation Plan §3.2

Scans acousticscene-bias/data/stimuli/task1/ and writes one row per stimulus to
acousticscene-bias/manifests/task1_manifest.csv

Columns: stimulus_id, task, item_id, scene, snr, condition, fg_path, bg_path, out_path

Usage:
  python scripts/05_build_manifest.py [--smoke-test]

  --smoke-test   Only scan smoke-test scenes/items.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COLUMNS = [
    "stimulus_id", "task", "item_id",
    "scene", "snr", "condition",
    "fg_path", "bg_path", "out_path",
]


def _load_items(path: Path) -> Dict[str, Dict]:
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                items[obj["id"]] = obj
    return items


def build_manifest(
    stimuli_dir: Path,
    item_lookup: Dict[str, Dict],
    scenes_filter: List[str],
) -> List[Dict]:
    rows = []
    task_dir = stimuli_dir

    if not task_dir.exists():
        log.warning("Stimuli directory not found: %s. Run 04_mix_stimuli.py first.", task_dir)
        return rows

    for item_dir in sorted(task_dir.iterdir()):
        if not item_dir.is_dir():
            continue
        item_id = item_dir.name
        if item_id not in item_lookup:
            continue

        for scene_dir in sorted(item_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name

            # Apply filter
            if scene not in scenes_filter and scene != "clean":
                continue

            for snr_dir in sorted(scene_dir.iterdir()):
                if not snr_dir.is_dir():
                    continue
                snr = snr_dir.name

                for wav in sorted(snr_dir.glob("*.wav")):
                    condition = wav.stem  # c1, scrambled, etc.
                    stim_id   = f"{item_id}__{scene}__{snr}__{condition}"
                    rows.append({
                        "stimulus_id": stim_id,
                        "task":        "task1",
                        "item_id":     item_id,
                        "scene":       scene,
                        "snr":         snr,
                        "condition":   condition,
                        "fg_path":     "",   # reconstructed from item_id
                        "bg_path":     "",   # not tracked post-mixing
                        "out_path":    str(wav),
                    })

    return rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build master manifest for Task 1.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Only scan smoke-test scenes.")
    args = parser.parse_args()

    items = _load_items(config.ITEMBANKS / "task1_items.jsonl")
    stimuli_dir = config.STIMULI / "task1"

    if args.smoke_test:
        scenes_filter = config.SMOKE_SCENES
        out_path = config.MANIFESTS / "task1_manifest_smoke.csv"
        log.info("SMOKE TEST — scanning scenes: %s", scenes_filter)
    else:
        scenes_filter = config.SCENES
        out_path = config.MANIFESTS / "task1_manifest.csv"

    rows = build_manifest(stimuli_dir, items, scenes_filter)
    write_csv(rows, out_path)

    log.info("Manifest written: %s  (%d rows)", out_path, len(rows))

    # Sanity check: count per scene
    from collections import Counter
    counts = Counter(r["scene"] for r in rows if r["condition"] == "c1")
    for scene, n in sorted(counts.items()):
        log.info("  %s: %d stimuli", scene, n)

    missing = [r for r in rows if not Path(r["out_path"]).exists()]
    if missing:
        log.warning("%d stimulus files referenced in manifest do not exist on disk.", len(missing))


if __name__ == "__main__":
    main()
