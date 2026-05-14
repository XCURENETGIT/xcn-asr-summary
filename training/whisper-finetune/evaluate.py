from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import evaluate
import torch
import yaml
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whisper-large-v3-turbo-ko.yaml")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config["output_dir"]
    audio_column = config.get("audio_column", "audio")
    text_column = config.get("text_column", "text")

    dataset = load_dataset("json", data_files={"validation": config["validation_manifest"]})
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=config.get("sampling_rate", 16000)))

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).cuda()
    model.eval()

    metric = evaluate.load("cer")
    predictions: list[str] = []
    references: list[str] = []

    for row in dataset["validation"]:
        audio = row[audio_column]
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features.cuda()
        with torch.no_grad():
            generated_ids = model.generate(inputs, max_length=int(config.get("generation_max_length", 225)))
        prediction = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(prediction.strip())
        references.append(str(row[text_column]).strip())

    cer = metric.compute(predictions=predictions, references=references)
    print(f"model_path={Path(model_path)}")
    print(f"validation_items={len(predictions)}")
    print(f"cer={cer:.6f}")


if __name__ == "__main__":
    main()
