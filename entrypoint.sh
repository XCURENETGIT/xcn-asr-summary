#!/usr/bin/env bash
set -euo pipefail

if python - <<'PY' >/tmp/xcn_tel_summary_ld_path 2>/dev/null
import os
from pathlib import Path

paths = []
current = os.getenv("LD_LIBRARY_PATH", "")
if current:
    paths.extend([item for item in current.split(":") if item])

try:
    import nvidia.cublas.lib
    paths.append(str(Path(nvidia.cublas.lib.__file__).resolve().parent))
except Exception:
    pass

try:
    import nvidia.cudnn.lib
    paths.append(str(Path(nvidia.cudnn.lib.__file__).resolve().parent))
except Exception:
    pass

seen = set()
ordered = []
for item in paths:
    if item and item not in seen:
        ordered.append(item)
        seen.add(item)

print(":".join(ordered))
PY
then
  export LD_LIBRARY_PATH="$(cat /tmp/xcn_tel_summary_ld_path)"
fi

exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
