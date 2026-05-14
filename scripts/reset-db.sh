#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml)

compose() {
  docker compose "${COMPOSE_ARGS[@]}" "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file not found: $1" >&2
    exit 1
  fi
}

require_file docker-compose.yml
require_file db/init/001_schema.sql

echo "[reset-db] starting mariadb"
compose up -d mariadb

echo "[reset-db] waiting for mariadb health"
for _ in $(seq 1 60); do
  status="$(docker inspect xcn-asr-summary-mariadb --format '{{.State.Health.Status}}' 2>/dev/null || true)"
  if [[ "$status" == "healthy" ]]; then
    break
  fi
  sleep 2
done

status="$(docker inspect xcn-asr-summary-mariadb --format '{{.State.Health.Status}}' 2>/dev/null || true)"
if [[ "$status" != "healthy" ]]; then
  echo "mariadb is not healthy: ${status:-unknown}" >&2
  exit 1
fi

echo "[reset-db] stopping api"
compose stop api >/dev/null 2>&1 || true

echo "[reset-db] dropping and recreating database"
compose exec -T mariadb sh -lc '
  set -eu
  db="${MARIADB_DATABASE:-telsummary}"
  mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" -e "DROP DATABASE IF EXISTS \`$db\`; CREATE DATABASE \`$db\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
'

echo "[reset-db] applying schema"
for sql in db/init/*.sql; do
  [[ -f "$sql" ]] || continue
  echo "[reset-db] applying $sql"
  compose exec -T mariadb sh -lc '
    set -eu
    db="${MARIADB_DATABASE:-telsummary}"
    mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" "$db"
  ' < "$sql"
done

echo "[reset-db] starting api"
compose up -d api

echo "[reset-db] done"
