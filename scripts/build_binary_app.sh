#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="$(tr -d '[:space:]' < VERSION)"
OUTPUT_DIR="$ROOT_DIR/dist/binary-app-${VERSION}"
BUILDER_IMAGE="xcn-asr-summary/binary-builder:${VERSION}"

usage() {
  cat <<'EOF'
Usage: scripts/build_binary_app.sh [--version VERSION] [--output-dir DIR]

Build app Python modules as Cython .so files and export the compiled app
directory for inspection or separate delivery.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      BUILDER_IMAGE="xcn-asr-summary/binary-builder:${VERSION}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
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

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

docker build --target binary-builder -f Dockerfile.binary -t "$BUILDER_IMAGE" .
container_id="$(docker create "$BUILDER_IMAGE")"
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT
docker cp "$container_id:/build/app" "$OUTPUT_DIR/app"

echo "[binary] exported $OUTPUT_DIR/app"
