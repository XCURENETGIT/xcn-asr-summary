from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, db
from .filename_metadata import parse_collected_audio_filename
from .logging_utils import configure_logging
from .pipeline import transcribe_and_summarize

logger = logging.getLogger("xcn-asr-summary.voice-batch")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_output_path(filename: str) -> Path:
    stem = Path(filename).stem
    output = config.TRANSLATE_DIR / f"{stem}.txt"
    if not output.exists():
        return output
    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    return config.TRANSLATE_DIR / f"{stem}.{suffix}.txt"


def _safe_finish_path(source: Path) -> Path:
    target = config.VOICE_FINISH_DIR / source.name
    if not target.exists():
        return target
    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    return config.VOICE_FINISH_DIR / f"{source.stem}.{suffix}{source.suffix}"


def _safe_failed_path(source: Path) -> Path:
    target = config.VOICE_FAILED_DIR / source.name
    if not target.exists():
        return target
    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    return config.VOICE_FAILED_DIR / f"{source.stem}.{suffix}{source.suffix}"


def _acquire_lock(source: Path) -> Path | None:
    lock_path = source.with_name(f"{source.name}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
        lock_file.write(f"{os.getpid()} {time.time()}\n")
    return lock_path


def _release_lock(lock_path: Path | None) -> None:
    if lock_path and lock_path.exists():
        try:
            lock_path.unlink()
        except OSError as exc:
            logger.warning("failed to remove lock %s: %s", lock_path, exc)


def _iter_voice_files() -> list[Path]:
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in config.VOICE_BATCH_EXTENSIONS}
    files = [
        path
        for path in config.VOICE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
        and not path.name.endswith(".lock")
        and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda item: item.name)


def _format_speaker_summaries(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for item in items:
        speaker = str(item.get("speaker") or "").strip()
        summary = str(item.get("summary_text") or "").strip()
        if speaker and summary:
            summaries.append({"speaker_name": speaker, "speaker_summary": summary})
    return summaries


def _format_datetime(value) -> str | None:
    return value.isoformat() if value else None


def _result_payload(source: Path, result, metadata=None) -> dict[str, Any]:
    return {
        "processing_id": str(uuid.uuid4()),
        "processing_status": "completed",
        "input_type": "voice_file",
        "audio_file_name": source.name,
        "caller": metadata.caller if metadata else None,
        "extension_number": metadata.extension_number if metadata else None,
        "callee": metadata.extension_number if metadata else None,
        "call_started_at": _format_datetime(metadata.call_started_at) if metadata else None,
        "call_ended_at": _format_datetime(metadata.call_ended_at) if metadata else None,
        "result_created_at": _utc_now(),
        "speech_recognition_model": config.WHISPER_MODEL,
        "summary_model_backend": result.summary_backend,
        "summary_generation_model": result.summary_model,
        "detected_language": result.detected_language,
        "audio_duration_seconds": result.duration_seconds,
        "processing_time_ms": result.processing_ms,
        "full_transcript": result.transcript_text,
        "structured_call_summary": result.summary_text,
        "plain_call_summary": result.conversational_summary_text,
        "speaker_summary_list": _format_speaker_summaries(result.speaker_summaries),
    }


def _insert_completed_summary(source: Path, finish_path: Path, payload: dict[str, Any], result, metadata=None) -> int:
    return db.insert_summary(
        processing_id=str(payload["processing_id"]),
        input_type="voice_file",
        audio_file_name=source.name,
        audio_content_type="audio/wav" if source.suffix.lower() == ".wav" else None,
        stored_audio_path=str(finish_path),
        caller=metadata.caller if metadata else None,
        extension_number=metadata.extension_number if metadata else None,
        callee=metadata.extension_number if metadata else None,
        call_started_at=metadata.call_started_at if metadata else None,
        call_ended_at=metadata.call_ended_at if metadata else None,
        audio_duration_seconds=result.duration_seconds,
        detected_language=result.detected_language,
        speech_recognition_model=config.WHISPER_MODEL,
        summary_generation_model=result.summary_model,
        summary_model_backend=result.summary_backend,
        full_transcript=result.transcript_text,
        structured_call_summary=result.summary_text,
        plain_call_summary=result.conversational_summary_text,
        speaker_summary_list=result.speaker_summaries,
        speaker_segment_list=result.speaker_segments,
        processing_time_ms=result.processing_ms,
        processing_status="completed",
        error_message=None,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text((text or "").strip() + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _move_failed_voice_file(source: Path, exc: Exception) -> Path | None:
    if not source.exists():
        logger.warning("failed voice file no longer exists, skip quarantine: %s", source)
        return None
    config.VOICE_FAILED_DIR.mkdir(parents=True, exist_ok=True)
    failed_path = _safe_failed_path(source)
    shutil.move(str(source), str(failed_path))
    logger.warning("moved failed voice file: %s -> %s error=%s", source.name, failed_path, exc)
    return failed_path


def process_voice_file(source: Path) -> dict[str, Any]:
    lock_path = _acquire_lock(source)
    if lock_path is None:
        return {"status": "skipped", "filename": source.name, "reason": "locked"}

    try:
        output_path = _safe_output_path(source.name)
        finish_path = _safe_finish_path(source)
        metadata = parse_collected_audio_filename(source.name)
        logger.info("processing voice file: %s", source)
        result = transcribe_and_summarize(source)
        payload = _result_payload(source, result, metadata)
        _write_text_atomic(output_path, result.transcript_text)
        shutil.move(str(source), str(finish_path))
        summary_id = _insert_completed_summary(source, finish_path, payload, result, metadata)
        logger.info("completed voice file: %s -> %s", source.name, output_path.name)
        return {
            "status": "completed",
            "id": summary_id,
            "processing_id": payload["processing_id"],
            "input_type": "voice_file",
            "audio_file_name": source.name,
            "output_file": output_path.name,
            "finished_file": finish_path.name,
        }
    finally:
        _release_lock(lock_path)


def process_voice_batch(limit: int | None = None, *, wait_for_dependencies: bool = True) -> list[dict[str, Any]]:
    configure_logging(config.APP_NAME, config.LOG_DIR, config.LOG_LEVEL)
    config.VOICE_DIR.mkdir(parents=True, exist_ok=True)
    config.VOICE_FINISH_DIR.mkdir(parents=True, exist_ok=True)
    config.VOICE_FAILED_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSLATE_DIR.mkdir(parents=True, exist_ok=True)
    if wait_for_dependencies:
        db.wait_for_db()
        db.ensure_schema()

    files = _iter_voice_files()
    if limit is not None:
        files = files[: max(0, limit)]

    results: list[dict[str, Any]] = []
    for source in files:
        try:
            results.append(process_voice_file(source))
        except Exception as exc:
            logger.exception("failed to process voice file: %s", source)
            failed_path = _move_failed_voice_file(source, exc)
            results.append(
                {
                    "status": "failed",
                    "filename": source.name,
                    "error": str(exc),
                    "failed_file": failed_path.name if failed_path else None,
                }
            )
    return results


def run_voice_watch_loop(
    *,
    interval_sec: float | None = None,
    limit: int | None = None,
    stop_event=None,
) -> None:
    configure_logging(config.APP_NAME, config.LOG_DIR, config.LOG_LEVEL)
    interval = max(1.0, interval_sec if interval_sec is not None else config.VOICE_WATCH_INTERVAL_SEC)
    batch_limit = config.VOICE_WATCH_BATCH_LIMIT if limit is None else limit
    logger.info(
        "voice watch loop started dir=%s interval_sec=%s batch_limit=%s",
        config.VOICE_DIR,
        interval,
        batch_limit,
    )
    while stop_event is None or not stop_event.is_set():
        try:
            results = process_voice_batch(limit=batch_limit, wait_for_dependencies=False)
            completed = sum(1 for item in results if item.get("status") == "completed")
            failed = sum(1 for item in results if item.get("status") == "failed")
            if completed or failed:
                logger.info("voice watch cycle completed=%s failed=%s", completed, failed)
        except Exception as exc:
            logger.exception("voice watch cycle failed: %s", exc)
        if stop_event is not None:
            stop_event.wait(interval)
        else:
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process wav files from data/voice into summary JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process.")
    parser.add_argument("--watch", action="store_true", help="Continuously watch VOICE_DIR and process files.")
    parser.add_argument("--interval-sec", type=float, default=None, help="Watch interval seconds.")
    args = parser.parse_args()

    if args.watch:
        db.wait_for_db()
        db.ensure_schema()
        run_voice_watch_loop(interval_sec=args.interval_sec, limit=args.limit)
    else:
        results = process_voice_batch(limit=args.limit)
        print(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
