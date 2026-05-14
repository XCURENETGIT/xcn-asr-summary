#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.llamacpp-gguf)

REMOVE_VOLUMES=false

usage() {
  cat <<'EOF'
Usage: scripts/stop.sh [--sllm] [--vllm] [--volumes]

Options:
  --sllm     Stop llama.cpp/GGUF profile using .env.llamacpp-gguf.
  --vllm     Stop vLLM profile using .env.vllm.
  --volumes  Stop containers and remove compose-managed volumes.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sllm)
      COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.llamacpp-gguf)
      shift
      ;;
    --vllm)
      COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.vllm --profile sllm-vllm)
      shift
      ;;
    --volumes)
      REMOVE_VOLUMES=true
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

if [[ "$REMOVE_VOLUMES" == "true" ]]; then
  echo "[stop] stopping and removing containers with volumes"
  compose down -v
else
  echo "[stop] stopping and removing containers"
  compose down
fi

echo "[stop] done"
