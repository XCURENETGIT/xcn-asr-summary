from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "xcn-asr-summary"
APP_VERSION = "1.0.2"

BASE_DIR = Path(os.getenv("BASE_DIR", "/app"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
TRAINING_CLIP_DIR = Path(os.getenv("TRAINING_CLIP_DIR", str(DATA_DIR / "training-clips")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs")))
VOICE_DIR = Path(os.getenv("VOICE_DIR", str(DATA_DIR / "voice")))
VOICE_FINISH_DIR = Path(os.getenv("VOICE_FINISH_DIR", str(DATA_DIR / "voice_finish")))
VOICE_FAILED_DIR = Path(os.getenv("VOICE_FAILED_DIR", str(DATA_DIR / "voice_failed")))
TRANSLATE_DIR = Path(os.getenv("TRANSLATE_DIR", str(DATA_DIR / "translate")))
VOICE_WATCH_ENABLED = os.getenv("VOICE_WATCH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
VOICE_WATCH_INTERVAL_SEC = float(os.getenv("VOICE_WATCH_INTERVAL_SEC", "30"))
VOICE_WATCH_BATCH_LIMIT = int(os.getenv("VOICE_WATCH_BATCH_LIMIT", "1"))
VOICE_BATCH_EXTENSIONS = tuple(
    value.strip().lower()
    for value in os.getenv("VOICE_BATCH_EXTENSIONS", ".wav").split(",")
    if value.strip()
)

API_KEY = os.getenv("API_KEY", "").strip()
API_KEY_NAME = "X-API-Key"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


HF_HUB_OFFLINE = _env_bool("HF_HUB_OFFLINE")
TRANSFORMERS_OFFLINE = _env_bool("TRANSFORMERS_OFFLINE")
MODEL_LOCAL_FILES_ONLY = _env_bool(
    "MODEL_LOCAL_FILES_ONLY",
    "true" if HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE else "false",
)

DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "telsummary")
DB_USER = os.getenv("DB_USER", "teluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "telpass")
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ko")

DIARIZATION_ENABLED = _env_bool("DIARIZATION_ENABLED")
STEREO_CHANNEL_SPEAKERS_ENABLED = _env_bool("STEREO_CHANNEL_SPEAKERS_ENABLED", "true")
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()
PYANNOTE_DEVICE = os.getenv("PYANNOTE_DEVICE", WHISPER_DEVICE).strip()
PYANNOTE_NUM_SPEAKERS = int(os.getenv("PYANNOTE_NUM_SPEAKERS", "0"))
PYANNOTE_MIN_SPEAKERS = int(os.getenv("PYANNOTE_MIN_SPEAKERS", "0"))
PYANNOTE_MAX_SPEAKERS = int(os.getenv("PYANNOTE_MAX_SPEAKERS", "0"))

SUMMARY_BACKEND = os.getenv("SUMMARY_BACKEND", "stt_only").strip().lower()
SLLM_PROVIDER = os.getenv("SLLM_PROVIDER", "llamacpp").strip().lower()
SLLM_BASE_URL = os.getenv("SLLM_BASE_URL", "http://sllm-llamacpp:8080").rstrip("/")
SLLM_MODEL = os.getenv("SLLM_MODEL", "mykor/A.X-4.0-Light-gguf:Q4_K_M").strip()
SLLM_API_KEY = os.getenv("SLLM_API_KEY", "").strip() or "xcn-local"
SLLM_TIMEOUT_SEC = int(os.getenv("SLLM_TIMEOUT_SEC", "120"))
SLLM_STARTUP_WAIT_SEC = int(os.getenv("SLLM_STARTUP_WAIT_SEC", "600"))
SLLM_CONNECT_RETRIES = int(os.getenv("SLLM_CONNECT_RETRIES", "24"))
SLLM_CONNECT_RETRY_DELAY_SEC = float(os.getenv("SLLM_CONNECT_RETRY_DELAY_SEC", "5"))
SLLM_MAX_TOKENS = int(os.getenv("SLLM_MAX_TOKENS", "192"))
SLLM_MAX_PROMPT_CHARS = int(os.getenv("SLLM_MAX_PROMPT_CHARS", "10000"))
SLLM_TEMPERATURE = float(os.getenv("SLLM_TEMPERATURE", "0.2"))
SLLM_TOP_P = float(os.getenv("SLLM_TOP_P", "0.9"))
SLLM_REQUEST_PATH = os.getenv("SLLM_REQUEST_PATH", "/v1/chat/completions").strip()
SLLM_USE_CHAT_ENDPOINT = os.getenv("SLLM_USE_CHAT_ENDPOINT", "true").lower() == "true"
SPEAKER_PAUSE_THRESHOLD_SEC = float(os.getenv("SPEAKER_PAUSE_THRESHOLD_SEC", "0.8"))
SPEAKER_MAX_TURN_MERGE_SEC = float(os.getenv("SPEAKER_MAX_TURN_MERGE_SEC", "0.45"))
SPEAKER_ACK_MAX_CHARS = int(os.getenv("SPEAKER_ACK_MAX_CHARS", "10"))
SPEAKER_SHORT_TURN_MAX_SEC = float(os.getenv("SPEAKER_SHORT_TURN_MAX_SEC", "1.2"))
SPEAKER_SHORT_TURN_MAX_CHARS = int(os.getenv("SPEAKER_SHORT_TURN_MAX_CHARS", "12"))
SPEAKER_SHORT_TURN_CONTEXT_GAP_SEC = float(os.getenv("SPEAKER_SHORT_TURN_CONTEXT_GAP_SEC", "0.7"))
SPEAKER_ROLE_REFINE_MIN_CHARS = int(os.getenv("SPEAKER_ROLE_REFINE_MIN_CHARS", "8"))
SPEAKER_ROLE_REFINE_MIN_SEC = float(os.getenv("SPEAKER_ROLE_REFINE_MIN_SEC", "0.8"))
SPEAKER_ROLE_REFINE_SCORE_MARGIN = int(os.getenv("SPEAKER_ROLE_REFINE_SCORE_MARGIN", "3"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
SAVE_UPLOADS = os.getenv("SAVE_UPLOADS", "true").lower() == "true"
SAVE_TRAINING_CLIPS = os.getenv("SAVE_TRAINING_CLIPS", "true").lower() == "true"
TRAINING_CLIP_FORMAT = os.getenv("TRAINING_CLIP_FORMAT", "wav").strip().lower() or "wav"
TRAINING_DOCKER_IMAGE = os.getenv("TRAINING_DOCKER_IMAGE", "xcn-asr-summary/whisper-train:latest").strip()
TRAINING_GPU_DEVICE = os.getenv("TRAINING_GPU_DEVICE", "0").strip()
TRAINING_HOST_PROJECT_DIR = os.getenv("TRAINING_HOST_PROJECT_DIR", "/data01/xcn-asr-summary").strip()
TRAINING_DEFAULT_MAX_STEPS = int(os.getenv("TRAINING_DEFAULT_MAX_STEPS", "40"))
XCN_CRYPTO_KEY_FILE = os.getenv("XCN_CRYPTO_KEY_FILE", "/models/enckey").strip()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
