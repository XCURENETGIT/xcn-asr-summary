from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Whisper fine-tuning JSONL manifests from a CSV file.")
    parser.add_argument("--csv", required=True, help="CSV path with audio and text columns.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-prefix", default="")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_prefix = args.audio_prefix.rstrip("/")

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as csv_file, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            audio = str(row.get(args.audio_column) or "").strip()
            text = str(row.get(args.text_column) or "").strip()
            if not audio or not text:
                continue
            if audio_prefix and not audio.startswith("/"):
                audio = f"{audio_prefix}/{audio}"
            output_file.write(json.dumps({"audio": audio, "text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
