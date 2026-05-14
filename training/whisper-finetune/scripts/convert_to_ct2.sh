#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/whisper-large-v3-turbo-ko.yaml}"

HF_OUTPUT_DIR="$(python - <<'PY'
import os
import yaml

with open(os.environ.get("CONFIG_PATH", "configs/whisper-large-v3-turbo-ko.yaml"), "r", encoding="utf-8") as file:
    print(yaml.safe_load(file)["output_dir"])
PY
)"

CT2_OUTPUT_DIR="$(python - <<'PY'
import os
import yaml

with open(os.environ.get("CONFIG_PATH", "configs/whisper-large-v3-turbo-ko.yaml"), "r", encoding="utf-8") as file:
    print(yaml.safe_load(file)["ct2_output_dir"])
PY
)"

ct2-transformers-converter \
  --model "${HF_OUTPUT_DIR}" \
  --output_dir "${CT2_OUTPUT_DIR}" \
  --copy_files tokenizer.json preprocessor_config.json config.json generation_config.json \
  --quantization float16

echo "CTranslate2 model written to ${CT2_OUTPUT_DIR}"
