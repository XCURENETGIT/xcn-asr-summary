#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.llamacpp-gguf)

SERVICES=(mariadb api)

usage() {
  cat <<'EOF'
Usage: scripts/start.sh [--sllm] [--vllm] [--build] [--binary]

Options:
  default   Start STT-only API and MariaDB.
  --sllm    Start llama.cpp/GGUF SLLM using .env.llamacpp-gguf.
  --vllm    Start vLLM profile using .env.vllm.
  --build   Build images before starting.
  --binary  Build api image from Dockerfile.binary.
  -h, --help
EOF
}

BUILD=false
BINARY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sllm)
      COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.llamacpp-gguf --profile sllm)
      SERVICES=(mariadb sllm-llamacpp api)
      shift
      ;;
    --vllm)
      COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.vllm --profile sllm-vllm)
      SERVICES=(mariadb sllm-vllm api)
      shift
      ;;
    --build)
      BUILD=true
      shift
      ;;
    --binary)
      BINARY=true
      export ASR_SUMMARY_DOCKERFILE=Dockerfile.binary
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

compose() {
  docker compose "${COMPOSE_ARGS[@]}" "$@"
}

if [[ ! -f docker-compose.yml ]]; then
  echo "required file not found: docker-compose.yml" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "required file not found: .env" >&2
  exit 1
fi
for env_file in "${COMPOSE_ARGS[@]}"; do
  if [[ "$env_file" == .env.* && ! -f "$env_file" ]]; then
    echo "required file not found: $env_file" >&2
    exit 1
  fi
done

if [[ "$BUILD" == "true" ]]; then
  if [[ "$BINARY" == "true" ]]; then
    echo "[start] building binary api-gpu image"
  else
    echo "[start] building api-gpu image"
  fi
  compose build api
fi

echo "[start] starting ${SERVICES[*]}"
compose up -d "${SERVICES[@]}"

echo "[start] status"
compose ps
