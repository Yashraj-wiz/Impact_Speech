# CLAUDE.md — AcousticScene Bias: Task 1

Tracks every action taken by Claude Code on this project. Updated after each work session.

---

## Project Summary

**Study:** AcousticScene Bias — does ambient acoustic scene behind a spoken query
shift Speech-LLM classification output?

**Task scope this codebase:** Task 1 (Classification) only.
Paper: IMPACT-SPEECH @ EMNLP 2026. Submission deadline: 15 July 2026.

---

## Session Log

### 2026-06-20 — Initial scaffolding (Claude Code)

**User request:** Write all scripts for Task 1 only, from data download to scoring.
Include smoke-test flag, gitignore, CLAUDE.md, result.md, log.md.
All data/models must download into the working directory.

**Actions taken:**

#### Files created

| File | Purpose |
|------|---------|
| `config.py` | Shared configuration, all paths, env-var overrides for HF_HOME/TORCH_HOME |
| `requirements.txt` | All Python dependencies |
| `.gitignore` | Ignores data/, models/, stimuli WAV files |
| `acousticscene-bias/itembanks/task1_items.jsonl` | 50 authored items (10 per domain: career, health, cooking, education, parenting/civic) |
| `acousticscene-bias/prereg/preregistration.json` | Frozen pre-registration (scenes, SNRs, metrics, models, stereotype directions) |
| `scripts/01_download_datasets.py` | Downloads ESC-50, MUSAN, TAU Urban 2019, FSD50K, WHAM! |
| `scripts/02_curate_scenes.py` | Selects, trims, high-pass-filters, loudness-normalises clips per scene |
| `scripts/03_render_speech_tts.py` | Kokoro-TTS synthesis of all item prompts (C1 fixed voice) |
| `scripts/04_mix_stimuli.py` | SNR-controlled mixing (manual implementation, no sox dependency) + phase-scramble control |
| `scripts/05_build_manifest.py` | Builds master manifest CSV from stimuli directory |
| `scripts/06_manipulation_checks.py` | Scene presence (AST classifier), WER (Whisper), jitter floor |
| `acousticscene-bias/inference/adapters/base.py` | Abstract SpeechLLM base class |
| `acousticscene-bias/inference/adapters/models.py` | All 6 model adapters (greedy decoding, 8-bit quantisation) |
| `acousticscene-bias/inference/adapters/__init__.py` | Registry export |
| `scripts/07_inference_task1.py` | Runs inference, parses labels, writes JSONL, supports resume |
| `acousticscene-bias/scoring/metrics_task1.py` | LFR, Cramér's V, χ², permutation test, dose-response, significance gate |
| `scripts/08_score_task1.py` | Aggregates all models, writes tables + figures |
| `CLAUDE.md` | This file |
| `log.md` | Prompt log |
| `result.md` | Results (populated after runs) |

#### Design decisions

- **No Scaper/sox dependency for mixing** — implemented manual SNR-controlled mixer
  using numpy + pyloudnorm to avoid Windows sox issues while matching the plan's protocol.
- **Single `models.py`** for all adapters instead of one file each — avoids redundant
  boilerplate while keeping classes clearly separated.
- **Resume support in inference** — 07_inference_task1.py reads already-written rows
  and skips them, so a crash doesn't restart from zero.
- **Deferred imports of torch** — config.py sets HF_HOME before any HF import,
  so cache always lands in the local models/ directory.

#### Status

- [x] Step 1 — Download datasets (`01_download_datasets.py`)
- [x] Step 2 — Curate scenes (`02_curate_scenes.py`)
- [x] Step 3 — Render TTS (`03_render_speech_tts.py`)
- [x] Step 4 — Mix stimuli (`04_mix_stimuli.py`)
- [x] Step 5 — Build manifest (`05_build_manifest.py`)
- [x] Step 6 — Manipulation checks (`06_manipulation_checks.py`)
- [ ] Step 7 — Inference (`07_inference_task1.py --model qwen2audio --smoke-test`)
- [ ] Step 8 — Score (`08_score_task1.py --smoke-test`)

**Gate (Day 11):** Only proceed to full-scale run after smoke test passes all 8 steps.

---

## Notes for Next Session

- Resolved the data contamination/curation keyword mapping issue (e.g. FSD50K "Traffic" matching dog barks/cars to construction, "Mechanisms" matching printers/cameras to factory). Curated datasets have been fully regenerated cleanly.
- Updated scene presence checks to run on the 5dB SNR condition using clean, logical keywords. All scenes now successfully pass the gate.
- Use `/home/sil/miniconda3/bin/python` (conda base) — `soundfile`, `librosa`, `pyloudnorm` installed there.

