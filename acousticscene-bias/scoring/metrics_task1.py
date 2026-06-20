"""
Task-1 metric implementations.
Plan reference: Implementation Plan §6.1, §6.3, §6.4

Metrics:
  LFR      Label-Flip Rate per scene (plan §6.3, §5.5)
  cramers_v  Cramér's V for scene→label association strength
  chi2_test  χ² test of independence with p-value
  permutation_test  Paired permutation test for LFR significance
  dose_response     LFR across SNR levels (plan §7.2)

All functions are pure: inputs are DataFrames or dicts, outputs are scalars or dicts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


# ── LFR — Label-Flip Rate ─────────────────────────────────────────────────────

def compute_lfr(
    df: pd.DataFrame,
    scene: str,
    snr: str = "10dB",
    condition: str = "c1",
) -> Dict:
    """
    LFR(scene) = mean_i [ label_i(scene) != label_i(clean) ]    (plan §5.5, §6.3)

    Args:
        df:        DataFrame with columns [item_id, scene, snr, condition, parsed_label]
        scene:     Target scene to compare against clean baseline.
        snr:       SNR level to use (default = reference SNR).
        condition: Stimulus condition (default 'c1').

    Returns dict with:
        lfr         scalar in [0, 1]
        n_items     number of paired items
        n_flipped   count of label changes
        pairs       list of (item_id, label_clean, label_scene) for audit
    """
    clean = df[(df["scene"] == "clean") & (df["condition"] == condition)].copy()
    scn   = df[
        (df["scene"] == scene) & (df["snr"] == snr) & (df["condition"] == condition)
    ].copy()

    clean = clean.set_index("item_id")["parsed_label"].dropna()
    scn   = scn.set_index("item_id")["parsed_label"].dropna()

    shared = clean.index.intersection(scn.index)
    if len(shared) == 0:
        log.warning("LFR: no paired items for scene=%s snr=%s", scene, snr)
        return {"lfr": np.nan, "n_items": 0, "n_flipped": 0, "pairs": []}

    pairs = []
    flipped = 0
    for item_id in shared:
        lc = clean[item_id]
        ls = scn[item_id]
        is_flip = lc != ls
        flipped += int(is_flip)
        pairs.append({"item_id": item_id, "label_clean": lc, "label_scene": ls, "flipped": is_flip})

    return {
        "lfr":       flipped / len(shared),
        "n_items":   len(shared),
        "n_flipped": flipped,
        "pairs":     pairs,
    }


# ── Cramér's V ────────────────────────────────────────────────────────────────

def compute_cramers_v(
    df: pd.DataFrame,
    snr: str = "10dB",
    condition: str = "c1",
    include_clean: bool = True,
) -> Dict:
    """
    Cramér's V for the association between scene and predicted label.   (plan §6.3)

    V = sqrt( chi2 / (n * min(r-1, c-1)) )

    Returns dict with V, chi2, p_value, dof, n.
    """
    sub = df[(df["snr"].isin([snr, "clean"])) & (df["condition"] == condition)].dropna(subset=["parsed_label"])
    if not include_clean:
        sub = sub[sub["scene"] != "clean"]

    if len(sub) < 5:
        log.warning("Cramér's V: too few rows (%d).", len(sub))
        return {"cramers_v": np.nan, "chi2": np.nan, "p_value": np.nan, "dof": np.nan, "n": len(sub)}

    contingency = pd.crosstab(sub["scene"], sub["parsed_label"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency.values)
    n = contingency.values.sum()
    r, c = contingency.shape
    v = float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else np.nan

    return {"cramers_v": v, "chi2": float(chi2), "p_value": float(p), "dof": int(dof), "n": int(n)}


# ── Paired permutation test for LFR ──────────────────────────────────────────

def permutation_test_lfr(
    df: pd.DataFrame,
    scene: str,
    snr: str = "10dB",
    condition: str = "c1",
    n_permutations: int = 10_000,
    rng_seed: int = 42,
) -> Dict:
    """
    Paired permutation test: under H0, the scene label and clean label for each
    item are exchangeable.  We permute the scene/clean assignment per item and
    recompute LFR.

    Returns dict with observed_lfr, p_value, n_permutations.
    """
    rng = np.random.default_rng(rng_seed)

    clean = df[(df["scene"] == "clean") & (df["condition"] == condition)].set_index("item_id")["parsed_label"].dropna()
    scn   = df[(df["scene"] == scene) & (df["snr"] == snr) & (df["condition"] == condition)].set_index("item_id")["parsed_label"].dropna()

    shared = clean.index.intersection(scn.index)
    if len(shared) < 3:
        return {"observed_lfr": np.nan, "p_value": np.nan, "n_permutations": n_permutations}

    pairs_clean = np.array([clean[i] for i in shared])
    pairs_scene = np.array([scn[i]   for i in shared])

    flips = (pairs_clean != pairs_scene).astype(float)
    observed_lfr = float(flips.mean())

    null_lfrs = np.empty(n_permutations)
    for k in range(n_permutations):
        mask = rng.integers(0, 2, size=len(shared)).astype(bool)
        swapped = np.where(mask, pairs_scene, pairs_clean)
        base    = np.where(mask, pairs_clean, pairs_scene)
        null_lfrs[k] = (swapped != base).mean()

    p_value = float((null_lfrs >= observed_lfr).mean())
    return {
        "observed_lfr": observed_lfr,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ── Bonferroni-corrected significance ─────────────────────────────────────────

def apply_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """
    Bonferroni correction across scenes × tasks (plan §6.4).
    Returns dict of scene -> is_significant.
    """
    n = len(p_values)
    corrected_alpha = alpha / n if n > 0 else alpha
    return {scene: p <= corrected_alpha for scene, p in p_values.items()}


# ── Dose-response: LFR vs SNR ─────────────────────────────────────────────────

def compute_dose_response(
    df: pd.DataFrame,
    scene: str,
    snrs: List[str],
    condition: str = "c1",
) -> List[Dict]:
    """
    Computes LFR at each SNR level for one scene.  A monotone increase as SNR
    decreases (scene louder) supports a perception-mediated reading (plan §7.2).
    """
    results = []
    for snr in snrs:
        lfr_result = compute_lfr(df, scene, snr=snr, condition=condition)
        results.append({"snr": snr, "lfr": lfr_result["lfr"], "n_items": lfr_result["n_items"]})
    return results


# ── Scrambled-scene comparison ────────────────────────────────────────────────

def compare_scrambled(
    df: pd.DataFrame,
    scene: str,
    snr: str = "10dB",
) -> Dict:
    """
    Checks whether the scene effect is reproduced by the scrambled-scene control.
    A real scene effect should NOT appear in the scrambled condition (plan §6.4, §7.3).

    Returns:
        lfr_c1         LFR for real scene (condition=c1)
        lfr_scrambled  LFR for phase-scrambled scene (condition=scrambled)
        effect_ratio   lfr_scrambled / lfr_c1  (should be << 1 for a real effect)
    """
    c1_res  = compute_lfr(df, scene, snr=snr, condition="c1")
    scr_res = compute_lfr(df, scene, snr=snr, condition="scrambled")

    lfr_c1  = c1_res["lfr"]
    lfr_scr = scr_res["lfr"]
    ratio   = (lfr_scr / lfr_c1) if (lfr_c1 and not np.isnan(lfr_c1) and lfr_c1 > 0) else np.nan

    return {
        "lfr_c1":         lfr_c1,
        "lfr_scrambled":  lfr_scr,
        "effect_ratio":   ratio,
        "passes_control": bool(np.isnan(ratio) or ratio < 0.5),
    }


# ── Significance gate (plan §6.4) ────────────────────────────────────────────

def significance_gate(
    lfr: float,
    p_value: float,
    jitter_floor: Dict,
    scrambled_result: Dict,
    bonferroni_significant: bool,
    alpha: float = 0.05,
) -> Dict:
    """
    An effect is declared real iff all three conditions hold (plan §6.4):
      (a) exceeds jitter floor
      (b) significant under corrected permutation test
      (c) NOT reproduced by scrambled-scene control
    """
    floor_mean = jitter_floor.get("jitter_floor_mean", 0.0)
    floor_std  = jitter_floor.get("jitter_floor_std",  0.0)
    threshold  = floor_mean + 2 * floor_std

    cond_a = (not np.isnan(lfr)) and (lfr > threshold)
    cond_b = bonferroni_significant and (not np.isnan(p_value)) and (p_value < alpha)
    cond_c = scrambled_result.get("passes_control", False)

    return {
        "declared_real": bool(cond_a and cond_b and cond_c),
        "cond_a_exceeds_jitter_floor": cond_a,
        "cond_b_significant_corrected": cond_b,
        "cond_c_not_in_scrambled": cond_c,
        "jitter_threshold": threshold,
    }
