"""
Stage 1 — Download scene audio datasets and license log.
Plan reference: Implementation Plan §2.1

Datasets downloaded:
  - ESC-50       : GitHub archive (~600 MB)
  - MUSAN        : OpenSLR resource 17 (~11 GB)
  - TAU Urban Acoustic Scenes 2019 : Zenodo 2589280 (split, ~22 GB total)
  - FSD50K       : Zenodo 4060432 (split, ~24 GB total)
  - WHAM! noise  : direct link (~17 GB)

Usage:
  python scripts/01_download_datasets.py [--smoke-test] [--dataset esc50|musan|tau|fsd50k|wham]

  --smoke-test   Download only ESC-50 (smallest dataset, ~600 MB) for pipeline verification.
  --dataset X    Download only the named dataset (can repeat).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import logging
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Dataset manifests ─────────────────────────────────────────────────────────

DATASETS = {
    "esc50": {
        "url":     "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        "dest":    config.SCENES_RAW / "esc50",
        "archive": config.SCENES_RAW / "esc50.zip",
        "extract_root": "ESC-50-master",
        "license": "CC BY-NC 3.0",
        "citation": "Piczak (2015). ESC: Dataset for Environmental Sound Classification.",
        "size_gb":  0.6,
    },
    "musan": {
        "url":     "https://openslr.org/resources/17/musan.tar.gz",
        "dest":    config.SCENES_RAW / "musan",
        "archive": config.SCENES_RAW / "musan.tar.gz",
        "extract_root": "musan",
        "license": "CC BY 4.0",
        "citation": "Snyder et al. (2015). MUSAN: A Music, Speech, and Noise Corpus. arXiv:1510.08484.",
        "size_gb":  11.0,
    },
    "tau": {
        "urls": [
            # TAU Urban Acoustic Scenes 2019 — Development set parts (Zenodo 2589280)
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.1.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.2.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.3.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.4.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.5.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.6.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.7.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.8.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.9.zip",
            "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip",
        ],
        "dest":    config.SCENES_RAW / "tau_urban",
        "license": "CC BY 4.0",
        "citation": "Mesaros et al. TAU/TUT Urban Acoustic Scenes (DCASE Challenge).",
        "size_gb":  22.0,
    },
    "fsd50k": {
        "urls": [
            # FSD50K — Zenodo 4060432
            "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip",
            "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01",
            "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02",
            "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03",
            "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04",
            "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05",
            "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip",
            "https://zenodo.org/record/4060432/files/FSD50K.metadata.zip",
        ],
        "dest":    config.SCENES_RAW / "fsd50k",
        "license": "CC BY 4.0",
        "citation": "Fonseca et al. (2022). FSD50K: an Open Dataset of Human-Labeled Sound Events. IEEE/ACM TASLP.",
        "size_gb":  24.0,
    },
    "wham": {
        "url":     "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip",
        "dest":    config.SCENES_RAW / "wham",
        "archive": config.SCENES_RAW / "wham_noise.zip",
        "extract_root": "wham_noise",
        "license": "CC BY 4.0",
        "citation": "Wichern et al. (2019). WHAM!: Extending Speech Separation to Noisy Environments.",
        "size_gb":  17.0,
    },
}


def _download_file(url: str, dest: Path, chunk_mb: int = 8) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume_pos = 0
    if dest.exists():
        try:
            r_head = requests.head(url, timeout=30, allow_redirects=True)
            expected = int(r_head.headers.get("content-length", 0))
        except Exception:
            expected = 0
        current = dest.stat().st_size
        if expected and current == expected:
            log.info("  already exists (size verified), skipping: %s", dest.name)
            return
        elif expected and current < expected:
            log.info("  partial file (%d/%d bytes), resuming: %s", current, expected, dest.name)
            resume_pos = current
        else:
            log.info("  size mismatch, re-downloading: %s", dest.name)
            dest.unlink()
    log.info("  downloading %s -> %s", url, dest.name)
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    r = requests.get(url, stream=True, timeout=120, headers=headers)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    mode = "ab" if resume_pos else "wb"
    with open(dest, mode) as f, tqdm(
        total=(total + resume_pos) if total else None,
        unit="B", unit_scale=True, desc=dest.name, initial=resume_pos
    ) as bar:
        for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def _extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log.info("  extracting %s -> %s", archive.name, dest)
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(dest)


def _extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log.info("  extracting %s -> %s", archive.name, dest)
    with tarfile.open(archive, "r:gz") as t:
        t.extractall(dest)


def download_esc50() -> None:
    meta = DATASETS["esc50"]
    dest: Path = meta["dest"]
    if dest.exists() and any(dest.iterdir()):
        log.info("ESC-50 already present, skipping.")
        return
    log.info("=== ESC-50 (%.1f GB) ===", meta["size_gb"])
    _download_file(meta["url"], meta["archive"])
    _extract_zip(meta["archive"], dest.parent)
    extracted = dest.parent / meta["extract_root"]
    if extracted != dest:
        extracted.rename(dest)
    meta["archive"].unlink(missing_ok=True)
    log.info("ESC-50 done: %s", dest)


def download_musan() -> None:
    meta = DATASETS["musan"]
    dest: Path = meta["dest"]
    if dest.exists() and any(dest.iterdir()):
        log.info("MUSAN already present, skipping.")
        return
    log.info("=== MUSAN (%.1f GB) — this may take a while ===", meta["size_gb"])
    _download_file(meta["url"], meta["archive"])
    _extract_tar(meta["archive"], dest.parent)
    extracted = dest.parent / meta["extract_root"]
    if extracted != dest:
        extracted.rename(dest)
    meta["archive"].unlink(missing_ok=True)
    log.info("MUSAN done: %s", dest)


def download_tau() -> None:
    meta = DATASETS["tau"]
    dest: Path = meta["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    log.info("=== TAU Urban Acoustic Scenes 2019 (%.1f GB total) ===", meta["size_gb"])
    for url in meta["urls"]:
        fname = url.split("/")[-1]
        archive = dest / fname
        sentinel = dest / (fname + ".done")
        if sentinel.exists():
            log.info("  already extracted, skipping: %s", fname)
            continue
        _download_file(url, archive)
        if fname.endswith(".zip"):
            _extract_zip(archive, dest)
            archive.unlink(missing_ok=True)
            sentinel.touch()
    log.info("TAU done: %s", dest)


def download_fsd50k() -> None:
    meta = DATASETS["fsd50k"]
    dest: Path = meta["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    log.info("=== FSD50K (%.1f GB total) ===", meta["size_gb"])
    for url in meta["urls"]:
        fname = url.split("/")[-1]
        archive = dest / fname
        _download_file(url, archive)
    # The .zip + .z01-.z05 form a multi-part zip — extract the main part
    main_zip = dest / "FSD50K.dev_audio.zip"
    if main_zip.exists():
        log.info("  Extracting multi-part FSD50K archive (requires compatible extractor) ...")
        try:
            _extract_zip(main_zip, dest / "dev_audio")
        except Exception as exc:
            log.warning("  Zip extraction failed (%s). Try extracting manually with 7-Zip.", exc)
    gt_zip = dest / "FSD50K.ground_truth.zip"
    if gt_zip.exists():
        _extract_zip(gt_zip, dest / "ground_truth")
        gt_zip.unlink(missing_ok=True)
    meta_zip = dest / "FSD50K.metadata.zip"
    if meta_zip.exists():
        _extract_zip(meta_zip, dest / "metadata")
        meta_zip.unlink(missing_ok=True)
    log.info("FSD50K done: %s", dest)


def download_wham() -> None:
    meta = DATASETS["wham"]
    dest: Path = meta["dest"]
    if dest.exists() and any(dest.iterdir()):
        log.info("WHAM! already present, skipping.")
        return
    log.info("=== WHAM! noise (%.1f GB) ===", meta["size_gb"])
    _download_file(meta["url"], meta["archive"])
    _extract_zip(meta["archive"], dest.parent)
    extracted = dest.parent / meta["extract_root"]
    if extracted.exists() and extracted != dest:
        extracted.rename(dest)
    meta["archive"].unlink(missing_ok=True)
    log.info("WHAM! done: %s", dest)


def write_license_log() -> None:
    log_path = config.CHECKS / "licenses.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dataset,license,citation,size_gb\n"]
    for name, meta in DATASETS.items():
        lines.append(
            f'{name},{meta["license"]},"{meta["citation"]}",{meta["size_gb"]}\n'
        )
    log_path.write_text("".join(lines), encoding="utf-8")
    log.info("License log written: %s", log_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download scene audio datasets.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Download only ESC-50 (~600 MB) to verify the pipeline end-to-end.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASETS.keys()),
        metavar="NAME",
        help="Download only the named dataset. Repeat for multiple. Default: all.",
    )
    args = parser.parse_args()

    config.SCENES_RAW.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        log.info("SMOKE TEST — downloading ESC-50 only.")
        download_esc50()
        write_license_log()
        log.info("Smoke test complete. ESC-50 available at %s", DATASETS["esc50"]["dest"])
        return

    targets = args.dataset or list(DATASETS.keys())
    dispatch = {
        "esc50":  download_esc50,
        "musan":  download_musan,
        "tau":    download_tau,
        "fsd50k": download_fsd50k,
        "wham":   download_wham,
    }
    for name in targets:
        dispatch[name]()

    write_license_log()
    log.info("All requested datasets downloaded.")


if __name__ == "__main__":
    main()
