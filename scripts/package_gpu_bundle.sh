#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="$(tr -d '[:space:]' < VERSION)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/dist"
PACKAGE_NAME=""
SOURCE_MODE="binary"
INCLUDE_VLLM_IMAGE=false
INCLUDE_LLAMACPP_IMAGE=false
INCLUDE_MODEL_CACHE=false
WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-models/hf/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo}"
WHISPER_PACKAGE_MODEL="large-v3-turbo"
HF_HUB_OFFLINE_VALUE=0
TRANSFORMERS_OFFLINE_VALUE=0
MODEL_LOCAL_FILES_ONLY_VALUE=0

usage() {
  cat <<'EOF'
Usage: scripts/package_gpu_bundle.sh [options]

Options:
  --version VERSION       Image/package version. Default: VERSION file.
  --output-dir DIR        Output directory. Default: ./dist
  --name NAME             Package base name. Default: xcn-asr-summary-gpu-package-<version>-<timestamp>
  --source-mode MODE      binary or source. Default: binary
  --include-vllm-image    Include vllm image tar. This can be large.
  --include-llamacpp-image Include llama.cpp CUDA server image tar. This can be large.
  --include-model-cache   Include ./models cache. This can be very large.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      PACKAGE_NAME="$2"
      shift 2
      ;;
    --source-mode)
      SOURCE_MODE="$2"
      shift 2
      ;;
    --include-vllm-image)
      INCLUDE_VLLM_IMAGE=true
      shift
      ;;
    --include-llamacpp-image)
      INCLUDE_LLAMACPP_IMAGE=true
      shift
      ;;
    --include-model-cache)
      INCLUDE_MODEL_CACHE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$SOURCE_MODE" != "binary" && "$SOURCE_MODE" != "source" ]]; then
  echo "--source-mode must be binary or source" >&2
  exit 1
fi

if [[ "$INCLUDE_MODEL_CACHE" == "true" ]]; then
  if [[ ! -d "$WHISPER_CACHE_DIR" ]]; then
    echo "Whisper cache directory not found: $WHISPER_CACHE_DIR" >&2
    echo "Start the API once online so faster-whisper large-v3-turbo is downloaded before packaging." >&2
    exit 1
  fi
  if ! find -L "$WHISPER_CACHE_DIR" -type f -name 'model.bin' | grep -q .; then
    echo "Whisper CTranslate2 model.bin not found under: $WHISPER_CACHE_DIR" >&2
    exit 1
  fi
  WHISPER_MODEL_BIN="$(find -L "$WHISPER_CACHE_DIR" -type f -name 'model.bin' | head -n 1)"
  WHISPER_SNAPSHOT_DIR="$(dirname "$WHISPER_MODEL_BIN")"
  WHISPER_PACKAGE_MODEL="/models/hf/hub/$(basename "$WHISPER_CACHE_DIR")/snapshots/$(basename "$WHISPER_SNAPSHOT_DIR")"
  HF_HUB_OFFLINE_VALUE=1
  TRANSFORMERS_OFFLINE_VALUE=1
  MODEL_LOCAL_FILES_ONLY_VALUE=1
else
  echo "[package] model cache is not included; packaged .env allows online model download on first start"
fi

IMAGE_REPO="${ASR_SUMMARY_IMAGE_REPO:-${TEL_SUMMARY_IMAGE_REPO:-xcn-asr-summary}}"
IMAGE_TAG="${ASR_SUMMARY_IMAGE_TAG:-${TEL_SUMMARY_IMAGE_TAG:-$VERSION}}"
API_IMAGE="${IMAGE_REPO}/api-gpu:${IMAGE_TAG}"
DOCKERFILE="Dockerfile"
if [[ "$SOURCE_MODE" == "binary" ]]; then
  DOCKERFILE="Dockerfile.binary"
fi

if [[ -z "$PACKAGE_NAME" ]]; then
  PACKAGE_NAME="xcn-asr-summary-gpu-package-${VERSION}-${TIMESTAMP}"
fi

STAGING="$OUTPUT_DIR/$PACKAGE_NAME"
ARCHIVE="$OUTPUT_DIR/${PACKAGE_NAME}.tar.gz"

echo "[package] version=$VERSION"
echo "[package] source_mode=$SOURCE_MODE"
echo "[package] api_image=$API_IMAGE"

rm -rf "$STAGING"
mkdir -p "$STAGING"/{images,scripts,db/init}

docker build -f "$DOCKERFILE" -t "$API_IMAGE" .
docker save "$API_IMAGE" -o "$STAGING/images/xcn-asr-summary-api-gpu-${IMAGE_TAG}.tar"

if [[ "$INCLUDE_VLLM_IMAGE" == "true" ]]; then
  VLLM_IMAGE="${ASR_SUMMARY_VLLM_IMAGE:-${TEL_SUMMARY_VLLM_IMAGE:-vllm/vllm-openai:latest}}"
  docker pull "$VLLM_IMAGE"
  docker save "$VLLM_IMAGE" -o "$STAGING/images/xcn-asr-summary-vllm.tar"
fi

if [[ "$INCLUDE_LLAMACPP_IMAGE" == "true" ]]; then
  LLAMACPP_IMAGE="${ASR_SUMMARY_LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
  docker pull "$LLAMACPP_IMAGE"
  docker save "$LLAMACPP_IMAGE" -o "$STAGING/images/xcn-asr-summary-llamacpp.tar"
fi

cp VERSION docker-compose.yml README.md .env.example .env.llamacpp-gguf .env.vllm "$STAGING/"
cp scripts/start.sh scripts/stop.sh scripts/reset-db.sh scripts/process_voice_batch.sh scripts/build_binary_app.sh "$STAGING/scripts/"
cp -R db/init/. "$STAGING/db/init/"

cat > "$STAGING/.env.package" <<EOF
ASR_SUMMARY_IMAGE_REPO=$IMAGE_REPO
ASR_SUMMARY_IMAGE_TAG=$IMAGE_TAG
ASR_SUMMARY_DOCKERFILE=$DOCKERFILE
HF_HUB_DISABLE_XET=1

DB_NAME=telsummary
DB_USER=teluser
DB_PASSWORD=telpass
DB_ROOT_PASSWORD=rootpass

WHISPER_MODEL=$WHISPER_PACKAGE_MODEL
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float32
WHISPER_LANGUAGE=ko
HF_HUB_OFFLINE=$HF_HUB_OFFLINE_VALUE
TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE_VALUE
MODEL_LOCAL_FILES_ONLY=$MODEL_LOCAL_FILES_ONLY_VALUE

SUMMARY_BACKEND=stt_only
VOICE_WATCH_ENABLED=true
VOICE_WATCH_INTERVAL_SEC=30
VOICE_WATCH_BATCH_LIMIT=1
VOICE_FAILED_DIR=/app/data/voice_failed
TRANSLATE_DIR=/app/data/translate
SLLM_API_KEY=xcn-local
SLLM_STARTUP_WAIT_SEC=600
SLLM_CONNECT_RETRIES=24
SLLM_CONNECT_RETRY_DELAY_SEC=5
EOF

cat > "$STAGING/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for image_tar in images/*.tar; do
  [[ -f "$image_tar" ]] || continue
  echo "[install] docker load $image_tar"
  docker load -i "$image_tar"
done

if [[ ! -f .env ]]; then
  cp .env.package .env
  echo "[install] created .env from .env.package"
fi

mkdir -p data/{uploads,training-clips,voice,voice_finish,voice_failed,translate,db} models
chmod +x scripts/*.sh
echo "[install] run: ./scripts/start.sh"
EOF
chmod +x "$STAGING/install.sh" "$STAGING"/scripts/*.sh

if [[ "$INCLUDE_MODEL_CACHE" == "true" ]]; then
  mkdir -p "$STAGING/models"
  cp -a models/. "$STAGING/models/"
fi

mkdir -p "$OUTPUT_DIR"
tar -C "$OUTPUT_DIR" -czf "$ARCHIVE" "$PACKAGE_NAME"
echo "[package] created $ARCHIVE"
