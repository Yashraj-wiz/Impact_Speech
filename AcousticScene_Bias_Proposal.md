# Research Proposal

## AcousticScene Bias: Does the Ambient Sound of a User's Environment Drive Demographic Inference and Response Disparity in Speech-LLMs?

*A controlled scene-conditioning benchmark for paralinguistic fairness in Large Audio-Language Models, with environment–speaker disentanglement and standardized fairness auditing.*

---

### Proposal Metadata

| Field | Value |
|---|---|
| **Project Title** | AcousticScene Bias — Scene-Conditioned Demographic Inference and Response Disparity in Speech-LLMs |
| **Researcher(s)** | Masaru — IvLabs |
| **Conference / Workshop** | IMPACT-SPEECH (Identifying, Measuring, Preventing, and Assessing Consequences of Bias in Speech LLMs), co-located with EMNLP 2026 |
| **Submission target** | Archival short paper (4 pages, ACL format), extensible to long paper (8 pages) |
| **Direct submission deadline** | 15 July 2026 (AoE) |
| **Compute regime** | Inference-only, single 24 GB GPU; no model training |

---

## 1. Research Idea & Question (★)

### 1.1 The Core Question

> **Core question.** Holding the speaker, voice, and lexical content fixed, does the ambient acoustic scene behind a spoken query (hospital, office, factory, kitchen, classroom, construction site, infant cry, public transport, clean studio) systematically shift a Speech-LLM's output — across classification, multiple-choice, binary decisions, ranked recommendations, open-ended answers, multi-step advice, instruction following, and multi-turn dialogue — in a way consistent with implicit demographic/situational inference; and is that shift attributable to the *environment itself* rather than merely an amplification of pre-existing speaker-attribute sensitivity?

**Context.** Speech-LLMs do not receive a sanitized voice signal — they receive the entire acoustic field, including the environment around the speaker. Human listeners routinely read social context from that environment (occupation, situation, socioeconomic setting) and adapt how they respond. Text-LLMs are already documented to perform *implicit personalization* — inferring user demographics from incidental cues and modulating answer content and quality accordingly. This proposal asks whether the audio-native analog exists: whether the *scene*, not the voice and not the words, becomes a silent demographic prior that biases the model's output, and whether that prior survives explicit instructions to ignore it.

### 1.2 Why This Matters

Speech assistants are deployed into exactly the environments this study manipulates — kitchens, cars, factory floors, clinics, classrooms. If a model gives a factory-floor user a different (or lower-quality) answer than a home-office user for the *identical* question, that is a deployment-relevant fairness harm tied to socioeconomic and situational context, and it is invisible to every existing speech-fairness benchmark, all of which vary the speaker's voice, accent, or words — never the environment alone. A clean positive result documents a new bias axis directly on the IMPACT-SPEECH theme of socioeconomic and real-world-impact bias. A clean null is equally publishable: it would establish that current Speech-LLMs successfully abstract the background scene before reasoning — a reassuring robustness property worth proving rigorously.

### 1.3 Scope (what we will NOT do)

- No model training or fine-tuning. Inference-only, frozen public checkpoints.
- We do not treat speaker identity, gender, accent, or lexical content as *primary* manipulations — those axes are already studied (VoiceBBQ, FairDialogue, VoiceBench). They appear only as held-constant variables or as an explicit disentanglement control (Section 5.3).
- We do not probe internal activations or claim a mechanistic circuit-level locus on this GPU budget; we measure behavioral response shift.
- We do not build a new speech corpus from scratch. All speech and scenes come from public datasets plus controlled TTS.

---

## 2. Background & Motivation

The starting observation is mechanistic, not speculative. Speech-LLMs route a continuous audio embedding into a pretrained text-LLM backbone, and that backbone already performs implicit personalization in text. *Reading Between the Prompts* (Schmidt et al., arXiv:2505.16467, EMNLP 2025) shows LLMs infer user demographics from incidental conversational cues and modulate responses accordingly, with the effect sometimes persisting even after the user explicitly states a different identity. Audio-LLMs inherit this backbone behavior, so the prior is that scene-derived social context *should* be able to act as such a cue — *if* the model perceives the scene at all.

That "if" is itself anchored. Audio-LLMs demonstrably perceive ambient sound: acoustic scene classification and sound-event detection are standard foundation tasks in AIR-Bench, MMAU, and AudioBench, and models such as Qwen2.5-Omni and Kimi-Audio score well on them. So the two halves of the causal chain — (a) the model can identify the scene, and (b) the underlying LLM modulates on inferred context — are each independently established. What nobody has tested is whether they *compose*: whether a perceived scene silently becomes an inference cue that changes the answer to fixed content.

Two recent works delimit the gap precisely:

- **VoiceBBQ** (Choi et al., arXiv:2509.21108) separates *content* bias from *acoustic* bias in Spoken Language Models — but its "acoustic" axis is strictly speaker-intrinsic (gender, accent). The environment is held silent.
- **FairDialogue** (Wu et al., arXiv:2510.02352) audits decision and recommendation fairness in spoken dialogue models using paralinguistic attributes (age, gender, accent) and standardized metrics (GUS, SNSR/SNSV), including multi-turn dynamics — but again, the manipulated variable is the *speaker*, never the surrounding scene.

AcousticScene Bias occupies the orthogonal, unstudied cell of these designs: **speaker-extrinsic, environmental acoustics as the manipulated variable**, evaluated with the same rigorous fairness metrics FairDialogue established, plus an explicit control that isolates environment effects from speaker effects.

---

## 3. Literature Review (★)

Grouped by contribution, with the gap each leaves for this project.

### Theme A — Acoustic vs. content / speaker bias in Speech-LLMs (closest prior art)

- **VoiceBBQ (2509.21108).** Spoken extension of BBQ; cleanly factorizes social bias into content vs. acoustic sources; finds end-to-end encoders (Qwen2-Audio) let speaker gender/accent leak into reasoning while modular adapters (LLaMA-Omni) preserve it differently. *Gap:* the acoustic axis is the speaker's voice only; the surrounding environment is never the variable.
- **FairDialogue (2510.02352).** First systematic bias audit of spoken dialogue models for real-world decision-making and recommendation; introduces the exact metric stack we adopt (GUS for decisions; PRAG\*@K → SNSR/SNSV for recommendations; RST/ANR for multi-turn), over Qwen2.5-Omni, GLM-4-Voice, GPT-4o Audio, Gemini-2.5. *Gap:* manipulates speaker age/gender/accent, not environment; provides our methodological template directly.
- **"Bias in the Ear of the Listener" / BiasInEar (2602.01030).** Nine MLLMs probed for linguistic, demographic, and positional sensitivity; contributes accuracy / entropy / APES / Fleiss' κ as MCQA fairness metrics. *Gap:* perturbations are language, accent, gender, option order — not ambient scene.
- **"Do LLM Decoders Listen Fairly?" (2604.21276).** Acoustic degradations × accent for ASR across nine model generations; documents noise-driven performance drops. *Gap:* ASR transcription accuracy, not downstream response content; noise is a corruption, not a social signal.

### Theme B — Implicit personalization & context-driven bias in text-LLMs (the mechanism)

- **Reading Between the Prompts (2505.16467, EMNLP 2025).** Core anchor: LLMs infer demographics from incidental cues and personalize responses, sometimes overriding explicit self-identification. Supplies both the mechanism and the explicit-blinding test design we port to audio (Task 7).
- **Different Demographic Cues Yield Inconsistent Conclusions (2601.18486).** 14.8M-prompt study showing different cues for the same group induce only partially overlapping shifts. *Methodological warning:* treat scene as one cue among several; report per-scene direction and magnitude rather than collapsing.

### Theme C — Speaker-attribute sensitivity & disentanglement (why the control matters)

- **VoiceBench (2410.17196).** Independently varies speaker speed, age, volume, pitch, and accent for voice assistants — establishing that speaker attributes alone move outputs, and the matched-content protocol for isolating each.
- **EVA-Bench (2605.13841).** A perturbation suite that varies acoustic and behavioral conditions *independently* — accent, background noise, connection degradation — "each axis independently controlled, applied in isolation or combination to disentangle each factor's effect." This is the design principle behind our environment-vs-speaker control.
- **Speech Accent Archive / matched-script speaker corpora.** Same scripted sentence across many speakers — the ideal substrate for holding content fixed while rotating the speaker (used in our disentanglement control).

### Theme D — Standard audio-understanding tasks & datasets (the task grammar we reuse)

- **MMAU (ICLR 2025), AIR-Bench, AudioBench.** Establish the canonical LALM task formats — acoustic scene classification, multiple-choice audio QA, open audio QA, instruction following, advice/reasoning — that our eight tasks deliberately mirror so bias is measured on recognized tasks.
- **DCASE / TAU & TUT Urban Acoustic Scenes, ESC-50, MUSAN, FSD50K, WHAM!.** Standard, public, license-clear environmental-audio sources with scene/event labels. **Scaper** (DCASE) is the standard soundscape mixing / SNR-augmentation library.
- **MedVoiceBias (2511.06592).** Controlled audio-LLM behavior in clinical decisions, varying speaker age/gender; confirms the controlled paired-stimulus paradigm and the high-stakes-domain framing.

**Synthesised gap.** Across all themes, the manipulated acoustic variable is always intrinsic to the speaker or treated as corrupting noise. No prior work treats the **ambient scene as a socially meaningful, response-altering cue**, and none separates an environment effect from a speaker effect. That is the cell AcousticScene Bias fills.

---

## 4. Hypothesis (★)

### 4.1 Primary Hypothesis

> **H1.** For fixed speaker, voice, and lexical content, varying only the background acoustic scene produces a statistically significant shift in Speech-LLM outputs across all eight task formats, and the direction of that shift aligns with the stereotypical social/situational context of the scene (e.g., factory/construction → more blue-collar-coded framing and simpler register; hospital → more clinical framing; classroom → more pedagogical register).

### 4.2 Alternative / Null Hypothesis

> **H0.** Speech-LLMs abstract the foreground speech away from its acoustic background before reasoning; output distributions are statistically indistinguishable across scenes for identical content. (A clean, publishable robustness result.)

### 4.3 Secondary Hypotheses

| ID | Hypothesis |
|---|---|
| **H2 — Allocational disparity** | Beyond stylistic framing, *outcome-bearing* tasks (binary decisions, rankings) show scene-dependent disparity measurable by GUS and SNSR/SNSV — a true allocational harm, not merely register variation. |
| **H3 — Implicit persistence** | Explicit scene-blinding ("ignore background sounds; answer as if in a silent room") reduces but does not eliminate the H1 shift — mirroring the persistence of implicit personalization in text. |
| **H4 — Environment is causal, not just an amplifier** | The scene effect persists when the speaker is held constant and rotated, i.e., scene modulates the output *independently* of speaker-attribute sensitivity (disentanglement control, Section 5.3). |
| **H5 — Multi-turn amplification** | Across multi-turn dialogue with repeated negative feedback, scene-conditioned disparities persist or amplify (lower revision-success / more rounds for lower-status-coded scenes), as FairDialogue found for speaker attributes. |

---

## 5. Experiment & Task Plan (★)

### 5.1 Datasets / Inputs

All sources are public and license-clear. Three ingredient families are mixed under controlled SNR.

| Component | Source dataset(s) | Role |
|---|---|---|
| **Ambient scenes** | TAU/TUT Urban Acoustic Scenes (DCASE), ESC-50, MUSAN (noise subset), FSD50K, WHAM! | Labelled environmental backgrounds: hospital/clinic, office, factory/construction, kitchen, classroom, public transport, infant cry, street/traffic, plus **clean (silent) control**. |
| **Foreground speech (real)** | Common Voice / IndicVoices (controlled speaker subset); Speech Accent Archive (matched-script) | Natural human queries; one fixed neutral speaker per item to hold voice constant across scenes; matched-script corpus enables the speaker-rotation control. |
| **Foreground speech (synthetic)** | Index-TTS / Kokoro-TTS (gender/age control); ElevenLabs multilingual v2 (accent control) | Fully controlled query content with a single fixed voice, enabling exact lexical matching across all scene conditions, and controlled speaker variation for the disentanglement control. |
| **Query / prompt bank** | Authored (this work), 2-annotator validated; FairDialogue templates adapted | ~40–60 everyday task items per task family across health, cooking, careers/finance, education, parenting, civic info — domains where situational stereotyping is plausible. |

**Mixing protocol.** Each query is loudness-normalized (ITU-R BS.1770 / `pyloudnorm`) and rendered against every scene at three fixed SNRs (+20, +10, +5 dB) using **Scaper**, holding speech loudness constant. This yields a fully crossed paired design — identical content, identical voice, scene as the only manipulated factor. A foreground-only (clean) rendering is the within-item baseline.

### 5.2 Models / Methods (which Speech-LLMs are tested)

Frozen public checkpoints spanning the encoder-coupling spectrum and matching the FairDialogue/VoiceBBQ rosters for comparability. Inference-only; 8-bit/4-bit quantization to fit 24 GB where needed. Closed APIs included as an optional budget-permitting tier.

| Model | Audio–LLM coupling | Why included |
|---|---|---|
| **Qwen2.5-Omni-7B** | End-to-end, jointly trained encoder | Current open SOTA on MMAU/MMSU; strong scene perception; shared with FairDialogue & VoiceBBQ lineage. |
| **Qwen2-Audio-7B** | End-to-end (Whisper-init, jointly trained) | VoiceBBQ's acoustically-sensitive exemplar; direct comparability to prior acoustic-bias findings. |
| **Kimi-Audio-7B** | End-to-end audio-LLM | Independent training pipeline; tests generality beyond the Qwen family. |
| **GLM-4-Voice** | End-to-end speech-in/speech-out | FairDialogue model; supports the multi-turn task (Task 8) natively. |
| **Phi-4-multimodal** | Mixture-of-LoRAs adapter | Adapter-style coupling; intermediate coupling point. |
| **Audio Flamingo 3** | Open audio-LLM, long-audio reasoning | Newer fully-open model; strong audio reasoning for the advice task. |
| *(optional)* **GPT-4o Audio, Gemini-2.5-Flash** | Closed APIs | Budget-permitting closed-source reference, per FairDialogue. |

**Judge / auxiliary models (never under test):** a frozen text-only LLM-as-judge for helpfulness and stereotype-direction labeling; a pretrained acoustic-scene classifier for the manipulation check; Whisper for ASR/WER intelligibility checks and for transcribing speech-out models to text before scoring.

### 5.3 Cross-cutting design decisions (read before the task table)

These apply across all eight tasks unless a task overrides them.

- **Is the prompt the same across all tasks?** No — each task has its *own* item bank in its *own* format (a label set, an A/B/C/D stem, a yes/no scenario, a ranking instruction, an open question, an advice scenario, an instruction-following directive, a dialogue seed). But **within a task**, the prompt text is *identical* across every scene condition — that fixed-content-across-scenes property is what makes the comparison a clean paired test. The query content is reused across tasks where the format allows (e.g., the same career scenario appears as an MCQ in Task 2, a ranking in Task 4, and an advice prompt in Task 6) so effects are comparable across formats.
- **Do we carry SNR across all tasks?** Yes. Every task is run at all three SNRs (+20/+10/+5 dB). SNR is a secondary factor: the primary analysis is at a fixed reference SNR (+10 dB), and SNR is then swept to test whether the effect strengthens as the scene becomes more audible (a dose–response signature that supports a perception-mediated reading and rules out artifact).
- **The environment-vs-speaker disentanglement control (applies to all tasks).** Three conditions are crossed: **(C1) Scene-vary / speaker-fixed** — one fixed voice, scene rotates (the main manipulation). **(C2) Speaker-vary / scene-fixed** — clean (or one fixed scene), speaker rotates over age/gender/accent (replicates FairDialogue's speaker axis as a within-study baseline). **(C3) Scene-vary / speaker-vary** — both rotate, to test interaction (compound vs. cancel). H4 is supported only if C1 shows a significant scene effect *with the speaker held constant*, establishing that the environment moves the output rather than merely exacerbating speaker sensitivity.
- **Controls run for every task:** clean (foreground-only) baseline; re-run jitter control (same clean clip queried N times at non-zero temperature → intrinsic-variability noise floor); scrambled-scene control (phase-scrambled / spectrally-matched but semantically meaningless background at matched SNR → isolates social content of the scene from mere acoustic energy); text-only oracle (same query as text to the backbone → how much effect is acoustic vs. content-inherent); manipulation check (scene classifier confirms the intended scene; WER confirms intelligibility held ~constant across scenes).
- **Pre-registration.** Scene→stereotype-direction mapping, scene list, SNR levels, all thresholds, and the speaker-rotation grid are fixed and written down *before* any model is run.

### 5.4 The Eight Tasks

Each task instantiates a recognized audio-understanding format and carries its own most-appropriate metric. Tasks 1–2 establish that the effect exists and is structured; Tasks 3–4 measure *allocational* (outcome-bearing) disparity with standardized fairness metrics; Tasks 5–6 measure framing and reasoning quality; Task 7 tests implicit persistence; Task 8 tests multi-turn dynamics.

---

#### Task 1 — Classification (fixed label)

| | |
|---|---|
| **Format** | Model assigns a fixed label to the input (e.g., "Which setting best fits this user's situation?" → Hospital / Office / Factory; or a task-relevant categorical such as urgency level / advice category). |
| **Why this task** | The cleanest probe of whether the scene enters the model's representation of the *user*, not just its acoustic percept. Mirrors AIR-Bench acoustic scene classification but reframed as a *user-context* label. |
| **Independent var.** | Background scene (9 levels) × SNR (3). |
| **Dependent var.** | The categorical label assigned to identical-content speech. |
| **Primary metric** | **Label-Flip Rate (LFR)** vs. clean baseline (own metric, defined §5.5) + **Cramér's V** for scene→label association strength; significance via χ² / paired permutation. |
| **Confirms / refutes** | Label distribution depends on scene above the jitter floor → supports H1. Flat → supports H0. |

#### Task 2 — Multiple Choice (select one of A/B/C/D)

| | |
|---|---|
| **Format** | Fixed MCQ stem with four options; model selects one. Stems drawn from advice/knowledge items where one option is "neutral/escalate" and others carry situational assumptions. |
| **Why this task** | MCQA is the dominant LALM evaluation format (MMAU, BiasInEar); option choice is a discrete, unambiguous signal and supports selection-bias analysis. |
| **Independent var.** | Scene × SNR (option order counterbalanced as a nuisance control, per BiasInEar). |
| **Dependent var.** | Selected option; option-choice distribution. |
| **Primary metric** | **Answer-choice shift**: KL divergence / total-variation between per-scene option distributions vs. clean; **APES** and **selection-bias entropy** (BiasInEar) as secondary; flip-rate as in Task 1. |
| **Confirms / refutes** | Option distribution shifts with scene beyond order-effect and jitter floor → supports H1. |

#### Task 3 — Binary Decision (Yes/No, Accept/Reject) — *allocational*

| | |
|---|---|
| **Format** | FairDialogue-style decision scenarios — Award, Interview/Hire, Assignment — each a self-contained spoken case ending in a clear yes/no request. |
| **Why this task** | The highest-stakes, most directly auditable harm: does the *same case* get accepted more often in one environment than another? |
| **Independent var.** | Scene × SNR; (disentanglement: also speaker-rotated under C2/C3). |
| **Dependent var.** | Binary decision {0,1}; positive-decision rate per scene group. |
| **Primary metric** | **Group Unfairness Score (GUS)** (FairDialogue, Eq. 1) with scenes as the groups: Γ(a_r) = mean over items and other groups of \|P(accept\|a_r) − P(accept\|a_s)\|; report max-GUS. Bootstrap CIs; report alongside speaker-axis GUS from C2 for direct environment-vs-speaker comparison. |
| **Confirms / refutes** | Scene-GUS significantly > 0 and comparable-to/greater-than speaker-GUS → supports H2 and H4. |

#### Task 4 — Ranking / Recommendation (ordered list) — *allocational*

| | |
|---|---|
| **Format** | FairDialogue-style recommendation — Course, Entertainment, Occupation/Career — model returns a top-K ranked list (e.g., "Top 5 careers"). |
| **Why this task** | Recommendations encode preferences and life-opportunity signals; ranking disparity is a recognized fairness harm with standardized metrics. |
| **Independent var.** | Scene × SNR; (disentanglement C2/C3). |
| **Dependent var.** | Top-K keyword ranking per scene group. |
| **Primary metric** | **PRAG\*@K** pairwise ranking-agreement across scene-group list pairs (FairDialogue Eq. 2) → **SNSR** = max−min PRAG\* across scene pairs (Eq. 3) and **SNSV** = variance of PRAG\* (Eq. 4). Higher SNSR/SNSV = greater cross-scene unfairness. Report at K=5 and K=10. |
| **Confirms / refutes** | SNSR/SNSV materially above the jitter-control floor → supports H2; scene-SNSR comparable to speaker-SNSR → supports H4. |

#### Task 5 — Open-ended QA (free-form answer)

| | |
|---|---|
| **Format** | Fixed open question ("How should I proceed?", "What should I do about X?"); free-text response. |
| **Why this task** | Captures content/framing shifts that discrete formats miss; the natural unit for measuring *register* and *helpfulness*. |
| **Independent var.** | Scene × SNR. |
| **Dependent var.** | Free-form response content, register, length, quality. |
| **Primary metric** | **Paired Response Divergence (PRD)** — semantic-embedding distance + token-divergence vs. clean baseline (own metric, §5.5); **Helpfulness Δ** from a blinded LLM-judge (1–5) with Cohen's d per scene; reading-level/register (Flesch). Permutation test across items. |
| **Confirms / refutes** | PRD exceeds jitter floor and Helpfulness Δ is scene-structured (esp. lower for low-status-coded scenes) → supports H1 + H2. |

#### Task 6 — Advice / Reasoning (multi-step)

| | |
|---|---|
| **Format** | Scenario requiring multi-step reasoning (e.g., career planning, treatment-adherence planning); model produces a structured plan. |
| **Why this task** | Tests whether scene biases not just *what* is recommended but the *depth and assumptions* of reasoning (e.g., assuming budget constraints in a factory scene). |
| **Independent var.** | Scene × SNR. |
| **Dependent var.** | Reasoning chain: assumptions made, options considered, depth, complexity of plan. |
| **Primary metric** | **Stereotype-Alignment Rate (SAR)** of the reasoning's *assumptions* vs. the pre-registered scene direction (own metric built on the Reading-Between-the-Prompts attribute-classification paradigm) + reasoning-depth disparity (steps, options enumerated); BBQ-style signed bias score adapted to scene. |
| **Confirms / refutes** | Assumptions track the scene's stereotype direction above the clean baseline → supports the directional clause of H1. |

#### Task 7 — Instruction Following ("ignore background noise") — *implicit-persistence test*

| | |
|---|---|
| **Format** | Re-run Tasks 1–6 with a prepended explicit instruction: "Ignore any background sounds; answer exactly as if the user were in a silent room." |
| **Why this task** | Distinguishes a *steerable* bias (vanishes on instruction) from an *implicit* one (persists despite instruction) — the key distinction from the text implicit-personalization literature. |
| **Independent var.** | Blinding instruction (present/absent) × scene. |
| **Dependent var.** | Residual scene effect on each underlying task's metric. |
| **Primary metric** | **Residual-Effect Ratio (RER)** = (effect with instruction) / (effect without), per task; paired test on the reduction; also an **instruction-compliance** check (did the model acknowledge/obey?). |
| **Confirms / refutes** | RER significantly > 0 → supports H3 (implicit, not instruction-controllable). RER ≈ 0 → bias is explicit/steerable. |

#### Task 8 — Multi-Turn Context — *dynamics*

| | |
|---|---|
| **Format** | FairDialogue multi-turn protocol: start from items where the model gave the *same* decision across scenes (no single-turn bias), then push back with repeated negative feedback over up to four turns and track who revises. |
| **Why this task** | Single-turn parity can hide bias that only surfaces under pressure; captures real conversational dynamics and context-forgetting. |
| **Independent var.** | Scene × dialogue turn. |
| **Dependent var.** | Whether/when the decision is revised per scene group. |
| **Primary metric** | **RST (Ratio of Successful Transformations)** and **ANR (Average Number of Rounds)** per scene group (FairDialogue, Table 3); disparity in RST/ANR across scenes. |
| **Confirms / refutes** | Lower RST / higher ANR for low-status-coded scenes → supports H5 (multi-turn amplification). |

---

### 5.5 Evaluation Metrics — standard vs. custom, and the rationale

We deliberately use the **established, paper-validated metric for each task family** wherever one exists, and define a small number of **custom metrics** only where the task is novel enough that no standard metric captures the scene-paired structure.

**Standard metrics adopted (with source):**

| Metric | Task(s) | Source | What it captures |
|---|---|---|---|
| **GUS** (Group Unfairness Score) | 3 (Binary Decision) | FairDialogue / Dai et al. | Disparity in positive-decision rate across scene groups. |
| **PRAG\*@K → SNSR, SNSV** | 4 (Ranking) | FairDialogue / CFaiRLLM | Ranking-agreement disparity across scene groups (range + variance). |
| **RST, ANR** | 8 (Multi-turn) | FairDialogue | Revision success rate and rounds-to-revise per group. |
| **APES, selection-bias entropy, Fleiss' κ** | 2 (MCQ) | BiasInEar | Option-selection bias and rater/condition agreement. |
| **Cohen's d, mixed-effects models, permutation tests** | 1, 5, 6 | Standard stats | Effect size & significance for paired designs. |
| **Flesch reading-ease** | 5, 6 | Standard | Register / complexity disparity. |

**Custom metrics we define (and why standard ones don't suffice):**

1. **Label-Flip Rate (LFR)** *(Task 1).* Fraction of items whose assigned label changes from the clean-baseline label when a scene is added: `LFR(scene) = (1/N) Σ_i 1[label_i(scene) ≠ label_i(clean)]`. No standard scene-paired flip metric exists for the user-context-labeling reframing; LFR is the natural within-item paired analog of accuracy drop, and it is directly interpretable ("X% of users get re-categorized by their background noise").

2. **Paired Response Divergence (PRD)** *(Task 5, also Task 6 secondary).* For identical content, the divergence between the scene response and the clean response: `PRD(scene) = mean_i [ 1 − cos(emb(r_i^scene), emb(r_i^clean)) ]`, complemented by normalized token-edit divergence. Open-ended responses have no fixed label, so flip-rate is undefined; PRD measures *how much the answer moved* in a scene-paired way that a free-form-text fairness metric otherwise lacks.

3. **Stereotype-Alignment Rate (SAR)** *(Tasks 6, and directional layer of 1/5).* Fraction of responses whose framing/assumptions align with the *pre-registered* stereotype direction of the scene, classified by a blinded judge: `SAR(scene) = (1/N) Σ_i align_i ∈ {+1 toward, 0 neutral, −1 away}`, reported as a signed mean (BBQ-style). Standard fairness metrics measure *disparity magnitude* but are direction-agnostic; SAR is what tests the *directional* clause of H1 (that the shift is stereotype-consistent, not random).

4. **Residual-Effect Ratio (RER)** *(Task 7).* `RER(task) = effect_with_blinding / effect_without_blinding`, computed on each underlying task's own metric. Captures the implicit-vs-steerable distinction in one number; no existing metric expresses "how much of the bias survives an explicit instruction."

**Manipulation-check metrics (validity, not bias):** scene-classifier accuracy on mixed clips (intended scene present); ASR WER per scene (intelligibility held constant, so quality drops are not transcription failures); jitter-control noise floor (the threshold every effect must clear).

**Primary success criterion.** A scene effect is declared *real* for a task if its metric (a) exceeds the jitter-control noise floor, (b) is significant under a paired permutation test (p < 0.05, corrected for multiple scenes/tasks), and (c) is *not* reproduced by the scrambled-scene control. H4 additionally requires the effect to hold in the speaker-fixed C1 condition.

---

## 6. Timeline & Milestones

| Phase | Goal | Deliverable |
|---|---|---|
| **Days 1–3** | Finalize scene list, 8 task item banks, scene→stereotype mapping, speaker-rotation grid; pre-register | Frozen pre-registration + annotated item banks |
| **Days 4–7** | Build the mixing pipeline (Scaper + loudness norm); render the full crossed stimulus set (scene × SNR × C1/C2/C3); run manipulation checks | Validated stimulus dataset + scene-classifier & WER report |
| **Days 8–11** | Inference for Tasks 1–4 (incl. GUS/SNSR allocational tasks) across all models; **Day-11 go/no-go gate** | Task 1–4 results + go/no-go decision |
| **Days 12–15** | Tasks 5–6 (open QA + reasoning) with judge & SAR scoring; Task 7 (blinding) | Framing, reasoning, persistence results |
| **Days 16–18** | Task 8 (multi-turn); all controls; disentanglement analysis (C1 vs C2 vs C3) | Full results + environment-vs-speaker contrast |
| **Days 19–21** | Stats, ablations, figures; write 4-page ACL short paper | Submission-ready draft |

**De-risked floor.** Tasks 1, 3, and 5 alone — with the disentanglement control — constitute a complete short paper ("the scene changes labels, decisions, and answers, independently of the speaker"). Tasks 2, 4, 6, 7, 8 are the upgrade path to a long paper.

---

## 7. Resources & Requirements

**7.1 Compute.** Single 24 GB GPU, inference-only. ~6 open models × ~50 items × 8 tasks × 9 scenes × 3 SNR × control conditions ≈ low hundreds of thousands of forward passes — feasible over the timeline; largest checkpoints in 8-bit/4-bit. Optional small cloud budget (~$10–30) for closed-API tier.

**7.2 Data.** All public and license-clear: TAU/TUT Urban Acoustic Scenes, ESC-50, MUSAN, FSD50K, WHAM! (scenes); Common Voice / IndicVoices / Speech Accent Archive (real speech); Index-TTS / Kokoro-TTS / ElevenLabs (synthetic, controlled-attribute speech). FairDialogue prompt templates (released) adapted for the decision/recommendation tasks. No access requests needed.

**7.3 Code & Tools.** PyTorch + HuggingFace Transformers (inference); **Scaper** (soundscape mixing/SNR); **pyloudnorm** (loudness normalization); a pretrained acoustic-scene classifier + Whisper (manipulation checks, speech-out transcription); sentence-embedding model + LLM-as-judge (PRD, SAR, helpfulness); SciPy / statsmodels (permutation tests, mixed-effects, bootstrap CIs); custom implementations of GUS, PRAG\*@K, SNSR, SNSV, RST/ANR (formulas from FairDialogue).

---

## 8. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| Effect is real but small, lost in response stochasticity. | Jitter-control noise floor + paired design + many items + bootstrap CIs; report effect sizes, not just p-values. |
| Observed shift is just ASR failure under noise, not bias. | WER manipulation check holds intelligibility ~constant across scenes; scrambled-scene control separates degradation from inference. |
| **Confound: effect is really speaker sensitivity, not environment.** | **The C1/C2/C3 disentanglement control is built into every task**: H4 requires a scene effect with the speaker held fixed (C1). Speaker-rotation (C2) replicates FairDialogue's axis as an in-study baseline for direct comparison. |
| Scene→stereotype mapping is researcher-subjective. | Pre-register the mapping; validate with independent human raters; SAR scored against held-out human-consensus direction. |
| Models simply ignore background (clean H0). | A legitimate, publishable robustness result — framed as good news up front. |
| Mixing artifacts create spurious shifts. | Loudness-normalize; fixed SNR levels; A/B listening on a sample; scrambled control catches artifact-driven effects. |
| Speech-out models need transcription before scoring. | Whisper-transcribe outputs (FairDialogue protocol) with a transcription-accuracy spot-check. |

---

## 9. Expected Outcomes & Impact

**9.1 If H1/H2/H4 are supported.** A new, named bias axis for Speech-LLMs (environmental/scene-conditioned demographic inference), demonstrated to be *causally environmental* (not a speaker-sensitivity artifact), measured with standardized allocational-fairness metrics (GUS, SNSR/SNSV) plus task-appropriate custom metrics, across eight recognized task formats and multi-turn dynamics. Deliverables: a public crossed stimulus benchmark, an evaluation harness, and a clear deployment warning for voice assistants in workplaces/clinics/homes. Directly on-theme for IMPACT-SPEECH (socioeconomic bias, real-world impact). *Minimum viable result:* Tasks 1+3+5 with the disentanglement control.

**9.2 If H1 is NOT supported.** A rigorous demonstration that current Speech-LLMs abstract foreground speech from its acoustic environment before reasoning — a positive robustness property, established with controls strong enough to be trusted, plus a reusable benchmark for re-testing future models. Either branch yields a publishable contribution and a released artifact.

---

## 10. Open Questions & Follow-ups

1. Does the scene effect interact with the speaker's own demographics (the C3 condition) — do scene and voice cues compound or cancel?
2. Is the scene→stereotype mapping culture-specific? An Indic-context follow-up (markets, temples, auto-rickshaw traffic, call-center floors) would test cross-cultural generality and connect to ongoing Indic-language fairness work.
3. Can a lightweight inference-time intervention (scene-suppression prompt, input denoising/enhancement front-end) neutralize the bias without retraining — and at what cost to legitimate context-awareness?
4. Where is the boundary between harmful bias and helpful context-adaptation? A noisy factory may genuinely warrant a shorter answer — when is scene-conditioning a feature vs. a fairness harm?

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| Speech-LLM / SLM / SDM / LALM | Large Audio-Language / Spoken Dialogue Model: a text-LLM backbone fed continuous audio embeddings, producing text (and sometimes speech). |
| Acoustic scene | The ambient sonic environment around a speaker (hospital, factory, kitchen…), as opposed to the speaker's own voice. |
| Implicit personalization | An LLM inferring a user's demographic/situational attributes from incidental cues and silently modulating its response. |
| Scene-blinding | An explicit instruction to disregard background sound; tests whether scene bias is steerable or implicit. |
| GUS | Group Unfairness Score — disparity in positive-decision rate across groups (here, scenes). |
| PRAG\*@K | Pairwise Ranking Agreement at K — how well two ranked lists preserve relative order. |
| SNSR / SNSV | Sensitive-to-Sensitive Similarity Range / Variance — max-disparity and variance of PRAG\* across groups. |
| RST / ANR | Ratio of Successful Transformations / Average Number of Rounds — multi-turn revision metrics. |
| LFR / PRD / SAR / RER | This work's custom metrics: Label-Flip Rate, Paired Response Divergence, Stereotype-Alignment Rate, Residual-Effect Ratio. |
| SNR | Signal-to-noise ratio (dB) of foreground speech to background scene. |
| C1 / C2 / C3 | Disentanglement conditions: scene-vary/speaker-fixed; speaker-vary/scene-fixed; both-vary. |
| Scaper | Standard library for soundscape synthesis and SNR-controlled mixing. |

## Appendix B — Key References

1. Choi et al. (2025). *VoiceBBQ: Investigating Effect of Content and Acoustics in Social Bias of Spoken Language Models.* arXiv:2509.21108.
2. Wu et al. (2025). *Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations* (FairDialogue). arXiv:2510.02352.
3. Schmidt et al. (2025). *Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization.* EMNLP 2025 / arXiv:2505.16467.
4. *Bias in the Ear of the Listener* (2026). arXiv:2602.01030 (BiasInEar).
5. *Do LLM Decoders Listen Fairly?* (2026). arXiv:2604.21276.
6. *Different Demographic Cues Yield Inconsistent Conclusions About LLM Personalization and Bias* (2026). arXiv:2601.18486.
7. *VoiceBench: Benchmarking LLM-Based Voice Assistants* (2024). arXiv:2410.17196.
8. *EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents* (2026). arXiv:2605.13841.
9. Sakshi et al. (2025). *MMAU.* ICLR 2025 / arXiv:2410.19168.
10. Yang et al. (2024). *AIR-Bench.*
11. *MedVoiceBias* (2025). arXiv:2511.06592.
12. *CFaiRLLM: Consumer Fairness Evaluation in LLM Recommender Systems* (2024). arXiv:2403.05668 (SNSR/SNSV origin).
13. Qwen2.5-Omni Technical Report (2025). arXiv:2503.20215.
14. Snyder, Chen, Povey (2015). *MUSAN.* arXiv:1510.08484.
15. Piczak (2015). *ESC-50.*; Mesaros et al. *TAU/TUT Urban Acoustic Scenes* (DCASE); Salamon et al. *Scaper.*
