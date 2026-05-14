#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/whisper-large-v3-turbo-ko.yaml}"

python train.py --config "${CONFIG_PATH}"
