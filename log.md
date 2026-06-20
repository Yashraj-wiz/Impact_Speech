# log.md — AcousticScene Bias: Prompt & Decision Log

---

## Entry 1 — 2026-06-20

### Prompt (verbatim)

> @AcousticScene_Bias_Implementation_Plan.md @AcousticScene_Bias_Implementation_Plan.pdf
> @AcousticScene_Bias_Proposal.md @AcousticScene_Bias_Proposal.pdf
>
> Write all the necessary scripts referring to the tagged documents pertaining to only task 1.
> I do not want to code up the other task for now, provide all the scripts for downloading
> datasets and evaluation metrics and everything, ensure all datasets and all caches and all
> models download in this directory only, also add a gitignore file, In all the scripts
> including downloading preparing data preparing audio samples, testing task 1, include a
> smoke test argument or option to ensure the scripts are running properly before full scaled
> run, only refer to the document provided, do not halucinate, If you have any doubt ask me,
> do not skip anything and keep the code clean, ensure there is no redundant code also
> maintain a claude.md and result.md file, in the claude.md keep the update of everything
> you have done and in result.md have the results, also maintain a log.md in it add the
> prompt we gave and the approval we gave you and also add your thinking in it.

### Approval

_Pending — user to update this entry after reviewing the generated scripts._

### Claude's Thinking

**Scope analysis from the documents:**

The implementation plan describes 8 stages and 8 tasks. The user asked for Task 1 only
(Classification). Task 1 requires the full pipeline from Stage 1 through Stage 8 because:
- Stage 1: Item bank with schema `{id, scenario, prompt, label_set}`
- Stage 2: Scene audio curation (all 8 scenes needed even for Task 1)
- Stage 3: TTS rendering (C1 fixed voice) + stimulus mixing at 3 SNRs
- Stage 4: Manipulation checks (scene classifier, WER, jitter floor)
- Stage 5: Model inference (6 models, greedy decoding, label parse)
- Stage 6: LFR + Cramér's V + χ² + permutation test (plan §6.3)
- Stage 7: Dose-response (LFR vs SNR) + scrambled comparison
- Stage 8: LFR heatmap + dose-response figure + significance table

**Key design choices made:**

1. **Manual SNR mixing over Scaper** — Scaper requires sox, which is problematic on
   Windows 11. The plan's pseudocode uses Scaper but the underlying operation is
   straightforward numpy arithmetic. The manual implementation matches the plan's
   protocol exactly (background normed to BG_LUFS, speech set to BG_LUFS + SNR_dB).

2. **All 6 model adapters in one file** — The plan lists 6 models. Writing 6 separate
   adapter files would repeat the same pattern 6 times. One `models.py` with a clear
   class per model and a registry avoids redundancy while remaining readable.

3. **8-bit quantisation default** — The plan specifies a single 24 GB GPU with
   8-bit/4-bit quantisation for the larger checkpoints. Using load_in_8bit=True as
   the default (overridable with --full-precision) matches this constraint.

4. **Item bank design** — The plan specifies ~50 items across health, cooking,
   careers/finance, education, parenting, civic info. I wrote exactly 50 items (10
   per domain) with label sets that are content-relevant but plausibly influenced by
   acoustic scene context, enabling a clean measurement of the scene effect.

5. **Phase scramble as control** — The plan specifies "phase-scramble or spectrally-
   match each scene". Phase scrambling via numpy FFT was chosen as it preserves the
   spectral envelope while destroying semantic content, exactly as described.

6. **Jitter floor via text backbone** — The plan's jitter check requires querying the
   clean baseline N=10 times at inference temperature. Since running audio inference
   10× per item is expensive, the implementation queries the text backbone of
   qwen2audio with the label-choice prompt (no audio), which isolates the LLM
   sampling variability — the component the jitter floor is meant to characterise.

**Potential issues flagged (do not start full run without resolving):**

- Audio Flamingo 3 HuggingFace repo ID not verified.
- GLM-4-Voice speech-out handling depends on actual model API; adapter has a fallback.
- TAU Urban filenames assumed to follow `{scene_label}-{city}-...` pattern — verify.
- FSD50K multi-part archive needs 7-Zip on Windows.
- `kokoro` TTS API: the generator pattern `for audio, sr, _ in pipeline(...)` may
  differ across kokoro versions — verify against installed version.

---

## Entry 2 — (next session)

_To be filled in when the user approves scripts and begins running them._
