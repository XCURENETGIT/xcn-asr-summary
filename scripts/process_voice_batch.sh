#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE_ARGS=(-f docker-compose.yml --env-file .env --env-file .env.llamacpp-gguf)
docker compose "${COMPOSE_ARGS[@]}" up -d mariadb

if docker compose "${COMPOSE_ARGS[@]}" ps --status running --services | grep -qx api; then
  docker compose "${COMPOSE_ARGS[@]}" exec -T api python -m app.voice_batch "$@"
else
  docker compose "${COMPOSE_ARGS[@]}" run --rm api python -m app.voice_batch "$@"
fi
