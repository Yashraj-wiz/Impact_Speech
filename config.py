"""
Shared configuration for AcousticScene Bias — Task 1.
All paths resolve relative to this file. All caches are routed to ./models/.
Import this module before any HuggingFace/torch import to ensure env vars land first.
"""

from pathlib import Path
import os

ROOT = Path(__file__).parent.resolve()
AB = ROOT / "acousticscene-bias"

# ── Data directories (matches plan §0) ────────────────────────────────────────
DATA           = AB / "data"
SCENES_RAW     = DATA / "scenes_raw"
SCENES_CURATED = DATA / "scenes_curated"
SPEECH_TTS     = DATA / "speech_tts"
SPEECH_REAL    = DATA / "speech_real"
STIMULI        = DATA / "stimuli"

ITEMBANKS  = AB / "itembanks"
PREREG     = AB / "prereg"
MANIFESTS  = AB / "manifests"
INFERENCE  = AB / "inference"
RESULTS    = AB / "results"
CHECKS     = AB / "checks"

MODELS   = ROOT / "models"
HF_CACHE = MODELS / "hf_cache"

# ── Redirect ALL caches to local directory ────────────────────────────────────
os.environ["HF_HOME"]             = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"]  = str(HF_CACHE / "transformers")
os.environ["HF_DATASETS_CACHE"]   = str(HF_CACHE / "datasets")
os.environ["TORCH_HOME"]          = str(MODELS / "torch")

# ── Scene configuration (plan §1.2) ──────────────────────────────────────────
SCENES = [
    "hospital",
    "office",
    "factory",
    "construction",
    "kitchen",
    "classroom",
    "public_transport",
    "infant_cry",
]

# Pre-registered stereotype direction — frozen before inference (plan §1.2)
SCENE_STEREOTYPE = {
    "hospital":         "clinical/medical framing",
    "office":           "white-collar/professional",
    "factory":          "blue-collar, simpler register",
    "construction":     "blue-collar, time-pressured",
    "kitchen":          "domestic/caregiving",
    "classroom":        "pedagogical/younger",
    "public_transport": "commuter/transient",
    "infant_cry":       "parenting/caregiving, urgency",
}

# ── SNR levels (plan §1.4, §5.3) ─────────────────────────────────────────────
SNR_LEVELS    = [20, 10, 5]   # dB
SNR_REFERENCE = 10             # primary-analysis SNR

# ── Audio constants ───────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
BG_LUFS      = -30.0   # background loudness reference (LUFS)
SPEECH_LUFS  = -20.0   # foreground speech loudness reference (LUFS)
CLIPS_PER_SCENE = 15   # number of exemplar clips to curate per scene

# ── Jitter control (plan §4) ──────────────────────────────────────────────────
JITTER_N = 10

# ── TTS voice for C1 (plan §1.3, §2.3) ───────────────────────────────────────
TTS_VOICE_C1 = "af_heart"   # Kokoro fixed neutral voice
TTS_LANG     = "a"          # American English

# ── HuggingFace model IDs (plan §5.2) — verify each before first run ─────────
MODEL_IDS = {
    "qwen25omni":    "Qwen/Qwen2.5-Omni-7B-Instruct",
    "qwen2audio":    "Qwen/Qwen2-Audio-7B-Instruct",
    "kimiaudio":     "moonshotai/Kimi-Audio-7B-Instruct",
    "glm4voice":     "THUDM/glm-4-voice-9b",
    "phi4":          "microsoft/Phi-4-multimodal-instruct",
    "audioflamingo3":"AudioFlamingo/audio-flamingo-3",  # verify HF repo before use
}

# ── ESC-50 category → scene mapping (plan §2.2) ───────────────────────────────
ESC50_SCENE_MAP = {
    "hospital":         ["siren"],
    "office":           ["keyboard_typing", "mouse_click"],
    "factory":          ["engine", "chainsaw"],
    "construction":     ["hand_saw", "chainsaw"],
    "kitchen":          ["washing_dishes", "crackling_fire"],
    "classroom":        ["clock_tick", "clock_alarm"],
    "public_transport": ["train"],
    "infant_cry":       ["crying_baby"],
}

# ── TAU Urban scene label → scene mapping ─────────────────────────────────────
TAU_SCENE_MAP = {
    "office":           ["office"],
    "public_transport": ["bus", "metro", "tram"],
}

# ── Smoke-test defaults ───────────────────────────────────────────────────────
SMOKE_ITEMS   = 3
SMOKE_SCENES  = ["hospital", "office"]
SMOKE_SNRS    = [10]
SMOKE_MODELS  = ["qwen2audio"]
