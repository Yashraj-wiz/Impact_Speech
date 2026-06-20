"""
Stage 6 — Score Task-1 inference outputs.
Plan reference: Implementation Plan §6, §7, §8

Reads inference/{model}/task1.jsonl for all models, computes:
  - LFR per scene × model × SNR
  - Cramér's V + χ² for scene→label association
  - Paired permutation test (Bonferroni-corrected across scenes)
  - Scrambled-scene comparison
  - Dose-response (LFR vs SNR)
  - Significance gate (plan §6.4)

Outputs:
  results/task1/lfr_table.csv          Headline table (plan §8)
  results/task1/cramers_v_table.csv    Cramér's V per model
  results/task1/dose_response.csv      LFR vs SNR per model × scene
  results/task1/significance.csv       Which (model, scene) effects pass the gate
  results/task1/figures/*.png          LFR heatmap, dose-response plot

Usage:
  python scripts/08_score_task1.py [--smoke-test] [--model qwen2audio ...]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Load inference outputs ────────────────────────────────────────────────────

def load_inference(model_key: str) -> pd.DataFrame:
    path = config.INFERENCE / model_key / "task1.jsonl"
    if not path.exists():
        log.warning("Inference file not found: %s", path)
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    log.info("Loaded %d rows for model %s", len(df), model_key)
    return df


# ── Load jitter floor ─────────────────────────────────────────────────────────

def load_jitter_floor() -> Dict:
    path = config.CHECKS / "jitter_floor.json"
    if not path.exists():
        log.warning("Jitter floor not found at %s. Defaulting to 0.", path)
        return {"jitter_floor_mean": 0.0, "jitter_floor_std": 0.0}
    return json.loads(path.read_text(encoding="utf-8"))


# ── Score one model ───────────────────────────────────────────────────────────

def score_model(
    model_key: str,
    df: pd.DataFrame,
    scenes: List[str],
    snr_ref: str,
    snr_levels: List[str],
    jitter_floor: Dict,
) -> Dict:
    sys.path.insert(0, str(config.AB))
    from scoring.metrics_task1 import (
        compute_lfr,
        compute_cramers_v,
        permutation_test_lfr,
        apply_bonferroni,
        compute_dose_response,
        compare_scrambled,
        significance_gate,
    )

    results = {
        "model":         model_key,
        "lfr":           {},
        "permutation":   {},
        "cramers":       {},
        "dose_response": {},
        "scrambled":     {},
        "gate":          {},
    }

    # ── LFR + permutation test per scene ─────────────────────────────────────
    p_values = {}
    for scene in scenes:
        lfr_res  = compute_lfr(df, scene, snr=snr_ref)
        perm_res = permutation_test_lfr(df, scene, snr=snr_ref)
        results["lfr"][scene]          = lfr_res
        results["permutation"][scene]  = perm_res
        p_values[scene]                = perm_res.get("p_value", 1.0) or 1.0

    bonferroni = apply_bonferroni(p_values)

    # ── Cramér's V ────────────────────────────────────────────────────────────
    results["cramers"] = compute_cramers_v(df, snr=snr_ref)

    # ── Dose-response ─────────────────────────────────────────────────────────
    for scene in scenes:
        results["dose_response"][scene] = compute_dose_response(df, scene, snrs=snr_levels)

    # ── Scrambled comparison ──────────────────────────────────────────────────
    for scene in scenes:
        results["scrambled"][scene] = compare_scrambled(df, scene, snr=snr_ref)

    # ── Significance gate ─────────────────────────────────────────────────────
    for scene in scenes:
        lfr  = results["lfr"][scene].get("lfr", np.nan)
        p    = p_values[scene]
        scr  = results["scrambled"][scene]
        bon  = bonferroni.get(scene, False)
        results["gate"][scene] = significance_gate(lfr, p, jitter_floor, scr, bon)

    return results


# ── Write output tables ───────────────────────────────────────────────────────

def write_lfr_table(all_results: List[Dict], scenes: List[str], out_path: Path) -> None:
    rows = []
    for res in all_results:
        model = res["model"]
        for scene in scenes:
            lfr_res  = res["lfr"].get(scene, {})
            perm_res = res["permutation"].get(scene, {})
            gate_res = res["gate"].get(scene, {})
            rows.append({
                "model":            model,
                "scene":            scene,
                "lfr":              round(lfr_res.get("lfr", np.nan), 4),
                "n_items":          lfr_res.get("n_items", 0),
                "n_flipped":        lfr_res.get("n_flipped", 0),
                "p_value":          round(perm_res.get("p_value", np.nan), 4),
                "declared_real":    gate_res.get("declared_real", False),
                "cond_a_jitter":    gate_res.get("cond_a_exceeds_jitter_floor", False),
                "cond_b_sig":       gate_res.get("cond_b_significant_corrected", False),
                "cond_c_scrambled": gate_res.get("cond_c_not_in_scrambled", False),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("LFR table: %s", out_path)


def write_cramers_table(all_results: List[Dict], out_path: Path) -> None:
    rows = []
    for res in all_results:
        c = res["cramers"]
        rows.append({
            "model":     res["model"],
            "cramers_v": round(c.get("cramers_v", np.nan), 4),
            "chi2":      round(c.get("chi2",      np.nan), 4),
            "p_value":   round(c.get("p_value",   np.nan), 4),
            "dof":       c.get("dof", np.nan),
            "n":         c.get("n",   0),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Cramér's V table: %s", out_path)


def write_dose_response(all_results: List[Dict], scenes: List[str], out_path: Path) -> None:
    rows = []
    for res in all_results:
        model = res["model"]
        for scene in scenes:
            for entry in res["dose_response"].get(scene, []):
                rows.append({
                    "model":  model,
                    "scene":  scene,
                    "snr":    entry["snr"],
                    "lfr":    round(entry["lfr"] if not np.isnan(entry.get("lfr", np.nan)) else np.nan, 4),
                    "n_items": entry["n_items"],
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Dose-response table: %s", out_path)


def write_significance(all_results: List[Dict], scenes: List[str], out_path: Path) -> None:
    rows = []
    for res in all_results:
        for scene in scenes:
            g = res["gate"].get(scene, {})
            s = res["scrambled"].get(scene, {})
            rows.append({
                "model":          res["model"],
                "scene":          scene,
                "lfr_c1":         round(s.get("lfr_c1", np.nan), 4),
                "lfr_scrambled":  round(s.get("lfr_scrambled", np.nan), 4),
                "effect_ratio":   round(s.get("effect_ratio", np.nan), 4),
                "declared_real":  g.get("declared_real", False),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Significance table: %s", out_path)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_lfr_heatmap(lfr_csv: Path, fig_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        log.warning("matplotlib/seaborn not available — skipping figures.")
        return

    df = pd.read_csv(lfr_csv)
    if df.empty:
        return

    pivot = df.pivot(index="scene", columns="model", values="lfr")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title(f"Task-1 LFR at {config.SNR_REFERENCE} dB SNR (vs clean)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Scene")
    plt.tight_layout()
    out = fig_dir / "lfr_heatmap.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("LFR heatmap saved: %s", out)


def plot_dose_response(dose_csv: Path, fig_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.read_csv(dose_csv)
    if df.empty:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        for scene in sub["scene"].unique():
            s = sub[sub["scene"] == scene].sort_values("snr")
            ax.plot(s["snr"], s["lfr"], marker="o", label=scene)
        ax.set_title(f"Dose-Response: LFR vs SNR — {model}")
        ax.set_xlabel("SNR level")
        ax.set_ylabel("LFR")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        out = fig_dir / f"dose_response_{model}.png"
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        log.info("Dose-response plot: %s", out)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Score Task-1 inference outputs.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Score smoke-test outputs only.")
    parser.add_argument("--model", nargs="+", choices=list(config.MODEL_IDS.keys()),
                        metavar="KEY", help="Score only these models. Default: all with output.")
    args = parser.parse_args()

    # Determine which models have inference output
    if args.model:
        model_keys = args.model
    else:
        model_keys = [
            k for k in config.MODEL_IDS
            if (config.INFERENCE / k / "task1.jsonl").exists()
        ]

    if not model_keys:
        log.error("No inference output found. Run 07_inference_task1.py first.")
        sys.exit(1)

    log.info("Scoring models: %s", model_keys)

    jitter_floor = load_jitter_floor()
    snr_ref    = f"{config.SNR_REFERENCE}dB"
    snr_levels = [f"{s}dB" for s in config.SNR_LEVELS]
    scenes     = config.SMOKE_SCENES if args.smoke_test else config.SCENES

    all_results = []
    for model_key in model_keys:
        df = load_inference(model_key)
        if df.empty:
            log.warning("No data for %s, skipping.", model_key)
            continue

        # Parse failures in inference output
        if "parse_ok" in df.columns:
            fail_rate = (~df["parse_ok"]).mean() * 100
            log.info("  %s parse failure rate: %.1f%%", model_key, fail_rate)

        try:
            res = score_model(model_key, df, scenes, snr_ref, snr_levels, jitter_floor)
            all_results.append(res)
        except Exception as exc:
            log.error("Scoring failed for %s: %s", model_key, exc)

    if not all_results:
        log.error("No results produced.")
        sys.exit(1)

    out_dir = config.RESULTS / "task1"
    fig_dir = out_dir / "figures"

    write_lfr_table(all_results, scenes, out_dir / "lfr_table.csv")
    write_cramers_table(all_results, out_dir / "cramers_v_table.csv")
    write_dose_response(all_results, scenes, out_dir / "dose_response.csv")
    write_significance(all_results, scenes, out_dir / "significance.csv")

    plot_lfr_heatmap(out_dir / "lfr_table.csv",    fig_dir)
    plot_dose_response(out_dir / "dose_response.csv", fig_dir)

    # Print headline summary
    lfr_df = pd.read_csv(out_dir / "lfr_table.csv")
    real   = lfr_df[lfr_df["declared_real"] == True]
    log.info("=== Headline Summary ===")
    log.info("Declared real effects: %d / %d (model × scene) pairs", len(real), len(lfr_df))
    if not real.empty:
        log.info("\n%s", real[["model", "scene", "lfr", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
