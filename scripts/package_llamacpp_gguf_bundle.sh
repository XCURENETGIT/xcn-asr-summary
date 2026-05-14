#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="$(tr -d '[:space:]' < VERSION)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/dist"
PACKAGE_NAME=""
PACKAGE_ROOT_NAME="xcn-asr-summary"
SOURCE_MODE="binary"
BUILD_IMAGE=true

usage() {
  cat <<'EOF'
Usage: scripts/package_llamacpp_gguf_bundle.sh [options]

Options:
  --version VERSION     Image/package version. Default: VERSION file.
  --output-dir DIR      Output directory. Default: ./dist
  --name NAME           Package base name. Default: xcn-asr-summary-llamacpp-gguf-package-<version>-<timestamp>
  --root-name NAME      Root directory inside tar.gz. Default: xcn-asr-summary
  --source-mode MODE    binary or source. Default: binary
  --skip-build          Use an existing API image instead of building it.
  -h, --help

This package includes:
  - xcn-asr-summary API image
  - llama.cpp CUDA server image
  - MariaDB image
  - A.X-4.0-Light GGUF Hugging Face cache
  - faster-whisper large-v3-turbo Hugging Face cache
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
    --root-name)
      PACKAGE_ROOT_NAME="$2"
      shift 2
      ;;
    --source-mode)
      SOURCE_MODE="$2"
      shift 2
      ;;
    --skip-build)
      BUILD_IMAGE=false
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

IMAGE_REPO="${ASR_SUMMARY_IMAGE_REPO:-xcn-asr-summary}"
IMAGE_TAG="${ASR_SUMMARY_IMAGE_TAG:-$VERSION}"
API_IMAGE="${IMAGE_REPO}/api-gpu:${IMAGE_TAG}"
LLAMACPP_IMAGE="${ASR_SUMMARY_LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
MARIADB_IMAGE="${ASR_SUMMARY_MARIADB_IMAGE:-mariadb:11.4}"
LLAMACPP_HF_MODEL="${LLAMACPP_HF_MODEL:-mykor/A.X-4.0-Light-gguf:Q4_K_M}"
GGUF_CACHE_DIR="${GGUF_CACHE_DIR:-models/hf-cache/hub/models--mykor--A.X-4.0-Light-gguf}"
WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-models/hf/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo}"

DOCKERFILE="Dockerfile"
if [[ "$SOURCE_MODE" == "binary" ]]; then
  DOCKERFILE="Dockerfile.binary"
fi

if [[ ! -d "$GGUF_CACHE_DIR" ]]; then
  echo "GGUF cache directory not found: $GGUF_CACHE_DIR" >&2
  echo "Start sllm-llamacpp once so the GGUF model is downloaded before packaging." >&2
  exit 1
fi

if ! find -L "$GGUF_CACHE_DIR" -type f -name '*.gguf' | grep -q .; then
  echo "GGUF model file not found under: $GGUF_CACHE_DIR" >&2
  exit 1
fi

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
WHISPER_PACKAGE_MODEL_PATH="/models/hf/hub/$(basename "$WHISPER_CACHE_DIR")/snapshots/$(basename "$WHISPER_SNAPSHOT_DIR")"

if [[ -z "$PACKAGE_NAME" ]]; then
  PACKAGE_NAME="xcn-asr-summary-llamacpp-gguf-package-${VERSION}-${TIMESTAMP}"
fi

STAGING="$OUTPUT_DIR/$PACKAGE_NAME"
PACKAGE_ROOT="$STAGING/$PACKAGE_ROOT_NAME"
ARCHIVE="$OUTPUT_DIR/${PACKAGE_NAME}.tar.gz"

echo "[package] version=$VERSION"
echo "[package] source_mode=$SOURCE_MODE"
echo "[package] api_image=$API_IMAGE"
echo "[package] llamacpp_image=$LLAMACPP_IMAGE"
echo "[package] mariadb_image=$MARIADB_IMAGE"
echo "[package] gguf_cache=$GGUF_CACHE_DIR"
echo "[package] whisper_cache=$WHISPER_CACHE_DIR"

rm -rf "$STAGING"
mkdir -p "$PACKAGE_ROOT"/{images,scripts,db/init,models/hf-cache/hub,models/hf/hub}

if [[ "$BUILD_IMAGE" == "true" ]]; then
  docker build -f "$DOCKERFILE" -t "$API_IMAGE" .
fi
docker image inspect "$API_IMAGE" >/dev/null
docker image inspect "$LLAMACPP_IMAGE" >/dev/null || docker pull "$LLAMACPP_IMAGE"
docker image inspect "$MARIADB_IMAGE" >/dev/null || docker pull "$MARIADB_IMAGE"

docker save "$API_IMAGE" -o "$PACKAGE_ROOT/images/xcn-asr-summary-api-gpu-${IMAGE_TAG}.tar"
docker save "$LLAMACPP_IMAGE" -o "$PACKAGE_ROOT/images/xcn-asr-summary-llamacpp-server-cuda.tar"
docker save "$MARIADB_IMAGE" -o "$PACKAGE_ROOT/images/xcn-asr-summary-mariadb-11.4.tar"

cp -a "$GGUF_CACHE_DIR" "$PACKAGE_ROOT/models/hf-cache/hub/"
cp -a "$WHISPER_CACHE_DIR" "$PACKAGE_ROOT/models/hf/hub/"

cp VERSION docker-compose.yml README.md .env.example .env.llamacpp-gguf .env.vllm "$PACKAGE_ROOT/"
cp Dockerfile Dockerfile.binary entrypoint.sh requirements.txt "$PACKAGE_ROOT/"
cp scripts/start.sh scripts/stop.sh scripts/reset-db.sh scripts/process_voice_batch.sh \
  scripts/build_binary_app.sh scripts/package_llamacpp_gguf_bundle.sh "$PACKAGE_ROOT/scripts/"
cp -R db/init/. "$PACKAGE_ROOT/db/init/"

cat > "$PACKAGE_ROOT/.env.package" <<EOF
ASR_SUMMARY_IMAGE_REPO=$IMAGE_REPO
ASR_SUMMARY_IMAGE_TAG=$IMAGE_TAG
ASR_SUMMARY_DOCKERFILE=$DOCKERFILE
HF_HUB_DISABLE_XET=1

DB_NAME=telsummary
DB_USER=teluser
DB_PASSWORD=telpass
DB_ROOT_PASSWORD=rootpass

WHISPER_MODEL=$WHISPER_PACKAGE_MODEL_PATH
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float32
WHISPER_LANGUAGE=ko
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

VOICE_WATCH_ENABLED=true
VOICE_WATCH_INTERVAL_SEC=30
VOICE_WATCH_BATCH_LIMIT=1
SLLM_API_KEY=xcn-local
SLLM_STARTUP_WAIT_SEC=600
SLLM_CONNECT_RETRIES=24
SLLM_CONNECT_RETRY_DELAY_SEC=5
EOF

cat > "$PACKAGE_ROOT/.env.llamacpp-gguf" <<EOF
ASR_SUMMARY_LLAMACPP_IMAGE=$LLAMACPP_IMAGE

SLLM_PROVIDER=llamacpp
SLLM_BASE_URL=http://sllm-llamacpp:8080
SLLM_MODEL=mykor/A.X-4.0-Light-gguf:Q4_K_M
SLLM_TIMEOUT_SEC=120
SLLM_MAX_TOKENS=192
SLLM_MAX_PROMPT_CHARS=10000
SLLM_TEMPERATURE=0.2
SLLM_TOP_P=0.9
SLLM_REQUEST_PATH=/v1/chat/completions
SLLM_USE_CHAT_ENDPOINT=true

LLAMACPP_HF_MODEL=$LLAMACPP_HF_MODEL
LLAMACPP_MODEL_PATH=/models/hf-cache/hub/models--mykor--A.X-4.0-Light-gguf/snapshots/b1cd7c8eee44c52ce4683545b7140658659e92cd/A.X-4.0-Light-Q4_K_M.gguf
LLAMACPP_HOST_MODELS=./models
LLAMACPP_GPU_DEVICE=0
LLAMACPP_CUDA_VISIBLE_DEVICES=0
LLAMACPP_CTX_SIZE=4096
LLAMACPP_N_GPU_LAYERS=24
LLAMACPP_PARALLEL=1
LLAMACPP_THREADS=8
LLAMACPP_BATCH_SIZE=512
LLAMACPP_UBATCH_SIZE=128
EOF

cat > "$PACKAGE_ROOT/install.sh" <<'EOF'
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

mkdir -p data/{uploads,training-clips,voice,voice_finish,translate,db} models
chmod +x scripts/*.sh entrypoint.sh
echo "[install] run: ./scripts/start.sh --sllm"
EOF
chmod +x "$PACKAGE_ROOT/install.sh" "$PACKAGE_ROOT"/scripts/*.sh "$PACKAGE_ROOT/entrypoint.sh"

cat > "$PACKAGE_ROOT/MANIFEST.txt" <<EOF
package=$PACKAGE_NAME
app_version=$VERSION
mode=llamacpp-gguf
source_mode=$SOURCE_MODE
images:
  - $API_IMAGE
  - $LLAMACPP_IMAGE
  - $MARIADB_IMAGE
model_cache:
  - $LLAMACPP_HF_MODEL
  - mobiuslabsgmbh/faster-whisper-large-v3-turbo
EOF

mkdir -p "$OUTPUT_DIR"
tar -C "$STAGING" --warning=no-file-changed --ignore-failed-read -czf "$ARCHIVE" "$PACKAGE_ROOT_NAME"

echo "[package] created $ARCHIVE"
du -h "$ARCHIVE"
