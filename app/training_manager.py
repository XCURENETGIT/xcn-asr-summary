from __future__ import annotations

import glob
import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from . import config, db


JOB_ROOT = config.DATA_DIR / "training-jobs"
HF_FINETUNED_ROOT = Path("/models/hf-finetuned")
DEFAULT_BASE_MODEL = "openai/whisper-large-v3-turbo"
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _edit_distance(left: list[str], right: list[str]) -> int:
    dp = list(range(len(right) + 1))
    for i, item in enumerate(left, start=1):
        ndp = [i] + [0] * len(right)
        for j, other in enumerate(right, start=1):
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + (item != other))
        dp = ndp
    return dp[-1]


def _cer(prediction: str, reference: str) -> float:
    pred_chars = list(_norm(prediction))
    ref_chars = list(_norm(reference))
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return _edit_distance(pred_chars, ref_chars) / len(ref_chars)


def _wer(prediction: str, reference: str) -> float:
    pred_words = _norm(prediction).split()
    ref_words = _norm(reference).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return _edit_distance(pred_words, ref_words) / len(ref_words)


def _job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def _job_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_job(job: dict[str, Any]) -> None:
    path = _job_path(str(job["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def list_jobs() -> list[dict[str, Any]]:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [_read_json(path, {}) for path in JOB_ROOT.glob("*/job.json")]
    jobs = [job for job in jobs if job.get("id")]
    return sorted(jobs, key=lambda item: str(item.get("started_at") or ""), reverse=True)


def get_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    job = _read_json(path, {})
    return job if job.get("id") else None


def get_job_log(job_id: str) -> str:
    log_path = _job_dir(job_id) / "job.log"
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def list_models() -> list[dict[str, Any]]:
    root = Path("/models/whisper")
    hf_root = HF_FINETUNED_ROOT
    active = str(config.WHISPER_MODEL)
    models: list[dict[str, Any]] = []
    if not root.exists():
        return models
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "model.bin").exists():
            continue
        hf_path = hf_root / path.name
        has_hf_checkpoint = (hf_path / "config.json").exists()
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        stat = path.stat()
        metrics = None
        for job in list_jobs():
            if job.get("model_path") == str(path):
                metrics = job.get("metrics")
                break
        models.append(
            {
                "name": path.name,
                "path": str(path),
                "is_active": active == str(path) or active == path.name,
                "has_hf_checkpoint": has_hf_checkpoint,
                "hf_model_path": str(hf_path) if has_hf_checkpoint else None,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "size_bytes": size,
                "metrics": metrics,
            }
        )
    return models


def list_base_models() -> list[dict[str, Any]]:
    active = str(config.WHISPER_MODEL)
    models = [
        {
            "name": "openai/whisper-large-v3-turbo",
            "value": DEFAULT_BASE_MODEL,
            "path": None,
            "source": "base",
            "is_active": active == DEFAULT_BASE_MODEL,
        }
    ]
    if HF_FINETUNED_ROOT.exists():
        for path in sorted(HF_FINETUNED_ROOT.iterdir()):
            if not path.is_dir() or not (path / "config.json").exists():
                continue
            models.append(
                {
                    "name": path.name,
                    "value": path.name,
                    "path": str(path),
                    "source": "fine-tuned",
                    "is_active": active == str(Path("/models/whisper") / path.name) or active == path.name,
                }
            )
    return models


def _validate_model_name(model_name: str) -> str:
    value = (model_name or "").strip()
    if not MODEL_NAME_RE.match(value):
        raise ValueError("model_name must use letters, numbers, dot, dash, or underscore")
    return value


def _resolve_base_model(base_model: str | None) -> tuple[str, str]:
    value = (base_model or DEFAULT_BASE_MODEL).strip()
    if not value or value == DEFAULT_BASE_MODEL:
        return DEFAULT_BASE_MODEL, DEFAULT_BASE_MODEL
    if value.startswith("openai/"):
        return value, value
    if value.startswith("/models/hf-finetuned/"):
        path = Path(value)
        if not (path / "config.json").exists():
            raise ValueError(f"base model checkpoint not found: {value}")
        return str(path), str(path)
    name = _validate_model_name(value)
    path = HF_FINETUNED_ROOT / name
    if not (path / "config.json").exists():
        raise ValueError(f"base model checkpoint not found: {name}")
    return name, str(path)


def _docker_client():
    try:
        import docker
    except ImportError as exc:
        raise RuntimeError("docker Python SDK is not installed") from exc
    try:
        return docker.from_env()
    except Exception as exc:
        raise RuntimeError(f"Docker socket is not available: {exc}") from exc


def _run_container(client, *, command: str, volumes: dict[str, dict[str, str]], log_path: Path, gpu_device: str) -> None:
    try:
        import docker

        device_requests = [docker.types.DeviceRequest(device_ids=[gpu_device], capabilities=[["gpu"]])]
    except Exception:
        device_requests = None
    with log_path.open("a", encoding="utf-8") as log:
        container = client.containers.run(
            config.TRAINING_DOCKER_IMAGE,
            command=["bash", "-lc", command],
            detach=True,
            remove=True,
            volumes=volumes,
            environment={
                "CUDA_VISIBLE_DEVICES": gpu_device,
                "NVIDIA_VISIBLE_DEVICES": gpu_device,
            },
            device_requests=device_requests,
        )
        for chunk in container.logs(stream=True, follow=True):
            log.write(chunk.decode("utf-8", errors="replace"))
            log.flush()
        result = container.wait()
        status_code = int(result.get("StatusCode", 1))
        if status_code != 0:
            raise RuntimeError(f"container exited with status {status_code}")


def _copy_tokenizer_if_missing(output_dir: Path) -> None:
    target = output_dir / "tokenizer.json"
    if target.exists():
        return
    candidates = glob.glob("/models/hf/hub/models--openai--whisper-large-v3-turbo/snapshots/*/tokenizer.json")
    if not candidates:
        return
    shutil.copyfile(candidates[0], target)


def _evaluate_model(model_path: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    model = WhisperModel(model_path, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
    rows = []
    total_cer = 0.0
    total_wer = 0.0
    for sample in samples:
        segments, _info = model.transcribe(str(sample["audio_path"]), language=config.WHISPER_LANGUAGE or "ko", vad_filter=True)
        prediction = _norm(" ".join(_norm(segment.text) for segment in segments))
        reference = _norm(str(sample["corrected_text"] or ""))
        cer_value = _cer(prediction, reference)
        wer_value = _wer(prediction, reference)
        total_cer += cer_value
        total_wer += wer_value
        rows.append(
            {
                "id": sample["id"],
                "prediction": prediction,
                "reference": reference,
                "cer": cer_value,
                "wer": wer_value,
            }
        )
    count = len(samples) or 1
    return {
        "avg_cer": total_cer / count,
        "avg_wer": total_wer / count,
        "samples": rows,
    }


def _build_config(*, model_name: str, base_model_path: str, run_dir: Path, max_steps: int, learning_rate: float) -> str:
    return f"""model_name_or_path: {base_model_path}
language: Korean
task: transcribe
train_manifest: /workspace/data/manifest/train.jsonl
validation_manifest: /workspace/data/manifest/validation.jsonl
output_dir: /workspace/outputs/hf
ct2_output_dir: /models/whisper/{model_name}
audio_column: audio
text_column: text
sampling_rate: 16000
max_duration_seconds: 30.0
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: {learning_rate}
warmup_steps: 0
max_steps: {max_steps}
eval_steps: {max(1, max_steps // 2)}
save_steps: {max(1, max_steps // 2)}
logging_steps: 5
generation_max_length: 225
fp16: true
gradient_checkpointing: true
predict_with_generate: false
save_total_limit: 2
"""


def start_training_job(
    *,
    status: str,
    model_name: str | None,
    base_model: str | None,
    include_used_samples: bool,
    max_steps: int,
    learning_rate: float,
    gpu_device: str | None,
) -> dict[str, Any]:
    base_model_value, base_model_path = _resolve_base_model(base_model)
    where_statuses = [status]
    if include_used_samples and status != "used":
        where_statuses.append("used")
    placeholders = ", ".join(["%s"] * len(where_statuses))
    samples = db.fetch_all(
        f"""
        SELECT id, audio_path, corrected_text, status
        FROM stt_training_samples
        WHERE status IN ({placeholders}) AND audio_path IS NOT NULL
        ORDER BY id ASC
        """,
        tuple(where_statuses),
    )
    if not samples:
        raise ValueError(f"no training samples found for status={','.join(where_statuses)}")
    primary_count = sum(1 for sample in samples if sample.get("status") == status)
    if primary_count == 0:
        raise ValueError(f"no training samples found for status={status}")

    job_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    model_name_v = _validate_model_name(model_name or f"whisper-{job_id}-ct2")
    queued_count = primary_count
    reused_count = sum(1 for sample in samples if sample.get("status") == "used")
    run_dir = _job_dir(job_id)
    manifest_dir = run_dir / "manifest"
    output_dir = run_dir / "outputs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl"):
        with (manifest_dir / name).open("w", encoding="utf-8") as file:
            for sample in samples:
                audio_name = Path(str(sample["audio_path"])).name
                payload = {"audio": f"/workspace/data/raw/training-clips/{audio_name}", "text": sample["corrected_text"]}
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (run_dir / "config.yaml").write_text(
        _build_config(
            model_name=model_name_v,
            base_model_path=base_model_path,
            run_dir=run_dir,
            max_steps=max_steps,
            learning_rate=learning_rate,
        ),
        encoding="utf-8",
    )

    job = {
        "id": job_id,
        "status": "running",
        "model_name": model_name_v,
        "base_model": base_model_value,
        "sample_count": len(samples),
        "queued_sample_count": queued_count,
        "reused_sample_count": reused_count,
        "sample_ids": [sample["id"] for sample in samples],
        "max_steps": max_steps,
        "started_at": _now(),
        "model_path": f"/models/whisper/{model_name_v}",
        "hf_model_path": f"/models/hf-finetuned/{model_name_v}",
        "error_message": None,
        "metrics": None,
    }
    _write_job(job)

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, samples, gpu_device or config.TRAINING_GPU_DEVICE),
        daemon=True,
    )
    thread.start()
    return job


def _run_training_job(job_id: str, samples: list[dict[str, Any]], gpu_device: str) -> None:
    job = get_job(job_id) or {"id": job_id}
    run_dir = _job_dir(job_id)
    log_path = run_dir / "job.log"
    host_root = Path(config.TRAINING_HOST_PROJECT_DIR)
    host_run = host_root / "data" / "training-jobs" / job_id
    client = None
    try:
        client = _docker_client()
        volumes = {
            str(host_run / "manifest"): {"bind": "/workspace/data/manifest", "mode": "rw"},
            str(host_root / "data" / "training-clips"): {"bind": "/workspace/data/raw/training-clips", "mode": "ro"},
            str(host_run / "outputs"): {"bind": "/workspace/outputs", "mode": "rw"},
            str(host_root / "models"): {"bind": "/models", "mode": "rw"},
            str(host_run / "config.yaml"): {"bind": "/workspace/configs/job.yaml", "mode": "ro"},
            str(host_root / "training" / "whisper-finetune" / "train.py"): {"bind": "/workspace/train.py", "mode": "ro"},
        }
        _run_container(
            client,
            command="python train.py --config configs/job.yaml",
            volumes=volumes,
            log_path=log_path,
            gpu_device=gpu_device,
        )
        _copy_tokenizer_if_missing(run_dir / "outputs" / "hf")
        target_model = Path(str(job["model_path"]))
        if target_model.exists():
            shutil.rmtree(target_model)
        _run_container(
            client,
            command=(
                "mkdir -p /models/hf-finetuned && "
                "rm -rf /models/hf-finetuned/{name} && "
                "cp -a /workspace/outputs/hf /models/hf-finetuned/{name} && "
                "rm -rf /models/whisper/{name} && "
                "ct2-transformers-converter --model /workspace/outputs/hf "
                "--output_dir /models/whisper/{name} "
                "--copy_files tokenizer.json preprocessor_config.json "
                "--quantization float16"
            ).format(name=job["model_name"]),
            volumes={
                str(host_run / "outputs"): {"bind": "/workspace/outputs", "mode": "rw"},
                str(host_root / "models"): {"bind": "/models", "mode": "rw"},
            },
            log_path=log_path,
            gpu_device=gpu_device,
        )
        metrics = {
            "active_model": _evaluate_model(config.WHISPER_MODEL, samples),
            "trained_model": _evaluate_model(str(job["model_path"]), samples),
        }
        for sample_id in job.get("sample_ids") or []:
            db.execute("UPDATE stt_training_samples SET status = 'used' WHERE id = %s AND status = 'queued'", (sample_id,))
        job.update({"status": "completed", "finished_at": _now(), "metrics": metrics})
    except Exception as exc:
        job.update({"status": "failed", "finished_at": _now(), "error_message": str(exc)})
    finally:
        _write_job(job)
        if client is not None:
            client.close()
