from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and torch.all(labels[:, 0] == self.processor.tokenizer.bos_token_id):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_manifest_dataset(config: dict[str, Any]) -> DatasetDict:
    data_files = {
        "train": config["train_manifest"],
        "validation": config["validation_manifest"],
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset.cast_column(config.get("audio_column", "audio"), Audio(sampling_rate=config.get("sampling_rate", 16000)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whisper-large-v3-turbo-ko.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model_name_or_path"]
    language = config.get("language", "Korean")
    task = config.get("task", "transcribe")
    audio_column = config.get("audio_column", "audio")
    text_column = config.get("text_column", "text")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    dataset = load_manifest_dataset(config)
    max_duration = float(config.get("max_duration_seconds", 30.0))

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[audio_column]
        duration = len(audio["array"]) / audio["sampling_rate"]
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = tokenizer(str(batch[text_column]).strip()).input_ids
        batch["duration_seconds"] = duration
        return batch

    dataset = dataset.map(
        prepare_batch,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    dataset = dataset.filter(lambda row: 0.0 < row["duration_seconds"] <= max_duration)

    predict_with_generate = bool(config.get("predict_with_generate", True))
    metric = evaluate.load("cer") if predict_with_generate else None

    def compute_metrics(pred: Any) -> dict[str, float]:
        if metric is None:
            return {}
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        warmup_steps=int(config.get("warmup_steps", 50)),
        max_steps=int(config.get("max_steps", 1000)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        fp16=bool(config.get("fp16", True)),
        eval_strategy="steps",
        eval_steps=int(config.get("eval_steps", 100)),
        save_steps=int(config.get("save_steps", 100)),
        logging_steps=int(config.get("logging_steps", 10)),
        predict_with_generate=predict_with_generate,
        generation_max_length=int(config.get("generation_max_length", 225)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="cer" if predict_with_generate else "eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics if predict_with_generate else None,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
