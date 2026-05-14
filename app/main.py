from __future__ import annotations

import json
import subprocess
import threading
import uuid
from datetime import datetime
from typing import Any
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Security, UploadFile, status
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from . import config, db
from .filename_metadata import parse_collected_audio_filename
from .logging_utils import configure_logging
from .pipeline import get_whisper_model, transcribe_and_summarize
from .schemas import (
    CallProcessResponse,
    CallSummaryItem,
    HealthResponse,
    DeleteResult,
    SpeakerSegment,
    SpeakerSummary,
    TrainingSampleCreateRequest,
    TrainingSampleItem,
    TrainingBaseModelItem,
    TrainingStartRequest,
    TrainingJobItem,
    TrainingModelItem,
    TranscriptUpdateRequest,
)
from .sllm_client import is_sllm_configured, wait_for_sllm
from . import training_manager
from .xcn_crypto import XcnCryptoError, decrypt_file, is_encrypted_bytes
from .voice_batch import run_voice_watch_loop

logger = configure_logging(config.APP_NAME, config.LOG_DIR, config.LOG_LEVEL)
api_key_header = APIKeyHeader(name=config.API_KEY_NAME, auto_error=False)
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.TRAINING_CLIP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)
voice_watch_stop_event = threading.Event()
voice_watch_thread: threading.Thread | None = None
ADMIN_DIR = Path(__file__).resolve().parent / "admin"
if ADMIN_DIR.exists():
    app.mount("/admin/static", StaticFiles(directory=ADMIN_DIR / "static"), name="admin-static")


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    return _verify_api_key_value(api_key)


def _verify_api_key_value(api_key: str | None) -> str:
    if not config.API_KEY:
        return ""
    if not api_key or api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")
    return api_key


def enforce_file_limit(size_bytes: int) -> None:
    if size_bytes <= 0:
        raise HTTPException(status_code=400, detail="empty file")
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large: max {config.MAX_UPLOAD_MB}MB")


def _normalize_speaker_summaries(items: list[dict[str, Any]]) -> list[SpeakerSummary]:
    return [
        SpeakerSummary(
            speaker_name=str(item.get("speaker_name") or item.get("speaker") or "").strip(),
            speaker_summary=str(item.get("speaker_summary") or item.get("summary_text") or "").strip(),
        )
        for item in items
        if (item.get("speaker_name") or item.get("speaker")) and (item.get("speaker_summary") or item.get("summary_text"))
    ]


def _normalize_speaker_segments(items: list[dict[str, Any]]) -> list[SpeakerSegment]:
    normalized: list[SpeakerSegment] = []
    for item in items:
        try:
            normalized.append(
                SpeakerSegment(
                    speaker=str(item.get("speaker") or ""),
                    start_seconds=float(item.get("start_seconds") or 0.0),
                    end_seconds=float(item.get("end_seconds") or 0.0),
                    text=str(item.get("text") or "").strip(),
                )
            )
        except (TypeError, ValueError):
            continue
    return sorted(normalized, key=lambda item: (item.start_seconds, item.end_seconds, item.speaker))


def normalize_row(row: dict) -> CallSummaryItem:
    speaker_summaries = _normalize_speaker_summaries(db.decode_json_field(row.get("speaker_summary_list_json")))
    return CallSummaryItem(
        id=int(row["id"]),
        processing_id=row["processing_id"],
        input_type=row.get("input_type") or "api_request",
        audio_file_name=row["audio_file_name"],
        caller=row.get("caller"),
        extension_number=row.get("extension_number"),
        callee=row.get("callee"),
        call_started_at=row.get("call_started_at"),
        call_ended_at=row.get("call_ended_at"),
        detected_language=row.get("detected_language"),
        audio_duration_seconds=float(row["audio_duration_seconds"]) if row.get("audio_duration_seconds") is not None else None,
        speech_recognition_model=row["speech_recognition_model"],
        summary_generation_model=row["summary_generation_model"],
        summary_model_backend=row.get("summary_model_backend"),
        structured_call_summary=row.get("structured_call_summary"),
        plain_call_summary=row.get("plain_call_summary"),
        speaker_summary_list=speaker_summaries,
        processing_status=row["processing_status"],
        error_message=row.get("error_message"),
        processing_time_ms=row.get("processing_time_ms"),
        result_created_at=row["result_created_at"],
    )


def normalize_training_sample(row: dict) -> TrainingSampleItem:
    return TrainingSampleItem(
        id=int(row["id"]),
        call_summary_id=int(row["call_summary_id"]),
        request_id=row["request_id"],
        filename=row["filename"],
        audio_path=row.get("audio_path"),
        segment_index=row.get("segment_index"),
        speaker=row.get("speaker"),
        start_seconds=float(row["start_seconds"]) if row.get("start_seconds") is not None else None,
        end_seconds=float(row["end_seconds"]) if row.get("end_seconds") is not None else None,
        original_text=row.get("original_text"),
        corrected_text=row["corrected_text"],
        note=row.get("note"),
        status=row["status"],
        created_at=row["created_at"],
    )


def _get_call_row_or_404(call_id: int) -> dict[str, Any]:
    row = db.fetch_one("SELECT * FROM call_summaries WHERE id = %s", (call_id,))
    if not row:
        raise HTTPException(status_code=404, detail="call summary not found")
    return row


def _create_training_clip(
    *,
    source_path: str | None,
    request_id: str,
    segment_index: int | None,
    start_seconds: float | None,
    end_seconds: float | None,
) -> str | None:
    if not config.SAVE_TRAINING_CLIPS:
        return source_path
    if not source_path or start_seconds is None or end_seconds is None:
        return source_path
    if end_seconds <= start_seconds:
        raise HTTPException(status_code=400, detail="end_seconds must be greater than start_seconds")
    source = Path(source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail="source audio file not found")
    config.TRAINING_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    safe_index = "full" if segment_index is None else str(segment_index)
    output_path = config.TRAINING_CLIP_DIR / f"{request_id}_seg{safe_index}.{config.TRAINING_CLIP_FORMAT}"
    duration = max(0.01, float(end_seconds) - float(start_seconds))
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(start_seconds):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=500, detail=f"failed to create training clip: {message[:500]}") from exc
    return str(output_path)


def _delete_training_clip_if_safe(audio_path: str | None) -> None:
    if not audio_path:
        return
    path = Path(audio_path)
    try:
        clip_root = config.TRAINING_CLIP_DIR.resolve()
        target = path.resolve()
    except OSError:
        return
    if clip_root not in target.parents:
        return
    if target.exists() and target.is_file():
        try:
            target.unlink()
        except OSError as exc:
            logger.warning("failed to delete training clip %s: %s", target, exc)


def _audio_media_type(path: Path, fallback: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    if suffix == ".ogg":
        return "audio/ogg"
    return fallback or "application/octet-stream"


def _delete_upload_file_if_safe(file_path: str | None) -> None:
    if not file_path:
        return
    path = Path(file_path)
    try:
        allowed_roots = (config.UPLOAD_DIR.resolve(), config.VOICE_FINISH_DIR.resolve())
        target = path.resolve()
    except OSError:
        return
    if not any(root == target or root in target.parents for root in allowed_roots):
        return
    if target.exists() and target.is_file():
        try:
            target.unlink()
        except OSError as exc:
            logger.warning("failed to delete call upload %s: %s", target, exc)


def _delete_call_uploads_if_safe(row: dict[str, Any]) -> None:
    _delete_upload_file_if_safe(row.get("stored_audio_path"))
    processing_id = str(row.get("processing_id") or "").strip()
    if not processing_id:
        return
    try:
        upload_root = config.UPLOAD_DIR.resolve()
    except OSError:
        return
    for path in upload_root.glob(f"{processing_id}.*"):
        _delete_upload_file_if_safe(str(path))


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    index_path = ADMIN_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="admin ui not found")
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


@app.on_event("startup")
def startup() -> None:
    global voice_watch_thread
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.VOICE_DIR.mkdir(parents=True, exist_ok=True)
    config.VOICE_FINISH_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSLATE_DIR.mkdir(parents=True, exist_ok=True)
    db.wait_for_db()
    db.ensure_schema()
    wait_for_sllm()
    _ = get_whisper_model()
    if config.VOICE_WATCH_ENABLED:
        voice_watch_thread = threading.Thread(
            target=run_voice_watch_loop,
            kwargs={
                "interval_sec": config.VOICE_WATCH_INTERVAL_SEC,
                "limit": config.VOICE_WATCH_BATCH_LIMIT,
                "stop_event": voice_watch_stop_event,
            },
            name="voice-watch",
            daemon=True,
        )
        voice_watch_thread.start()
    logger.info(
        "models loaded whisper=%s summary_backend=%s summary_model=%s sllm_configured=%s voice_watch_enabled=%s",
        config.WHISPER_MODEL,
        config.SUMMARY_BACKEND,
        config.SLLM_MODEL,
        is_sllm_configured(),
        config.VOICE_WATCH_ENABLED,
    )


@app.on_event("shutdown")
def shutdown() -> None:
    voice_watch_stop_event.set()
    if voice_watch_thread and voice_watch_thread.is_alive():
        voice_watch_thread.join(timeout=5)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        db_ready = db.fetch_one("SELECT 1 AS ok") is not None
    except Exception as exc:
        logger.warning("health db check failed: %s", exc)
        db_ready = False
    return HealthResponse(
        status="healthy" if db_ready else "degraded",
        db_ready=db_ready,
        whisper_model=config.WHISPER_MODEL,
        summary_backend=config.SUMMARY_BACKEND,
        summary_model=config.SLLM_MODEL,
        sllm_configured=is_sllm_configured(),
    )


@app.post("/calls/process", response_model=CallProcessResponse)
async def process_call(
    file: UploadFile = File(...),
    caller: str | None = Form(default=None),
    extension_number: str | None = Form(default=None),
    callee: str | None = Form(default=None),
    call_started_at: datetime | None = Form(default=None),
    call_ended_at: datetime | None = Form(default=None),
    api_key: str = Security(verify_api_key),
) -> CallProcessResponse:
    processing_id = str(uuid.uuid4())
    content = await file.read()
    enforce_file_limit(len(content))

    suffix = Path(file.filename or "call.bin").suffix or ".bin"
    filename_metadata = parse_collected_audio_filename(file.filename or "")
    if filename_metadata:
        caller = caller or filename_metadata.caller
        extension_number = extension_number or filename_metadata.extension_number
        callee = callee or filename_metadata.extension_number
        call_started_at = call_started_at or filename_metadata.call_started_at
        call_ended_at = call_ended_at or filename_metadata.call_ended_at
    stored_path = config.UPLOAD_DIR / f"{processing_id}{suffix}"
    audio_path = stored_path
    decrypted_path: Path | None = None
    if config.SAVE_UPLOADS:
        stored_path.write_bytes(content)
    else:
        stored_path.write_bytes(content)

    if is_encrypted_bytes(content[:16]):
        decrypted_path = config.UPLOAD_DIR / f"{processing_id}.wav"
        try:
            decrypt_file(stored_path, decrypted_path)
        except XcnCryptoError as exc:
            raise HTTPException(status_code=400, detail=f"failed to decrypt XCN encrypted file: {exc}") from exc
        audio_path = decrypted_path

    try:
        result = transcribe_and_summarize(audio_path)
        summary_id = db.insert_summary(
            processing_id=processing_id,
            input_type="api_request",
            audio_file_name=file.filename or stored_path.name,
            audio_content_type=file.content_type,
            stored_audio_path=str(audio_path) if config.SAVE_UPLOADS else None,
            caller=caller,
            extension_number=extension_number,
            callee=callee,
            call_started_at=call_started_at,
            call_ended_at=call_ended_at,
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
    except Exception as exc:
        summary_id = db.insert_summary(
            processing_id=processing_id,
            input_type="api_request",
            audio_file_name=file.filename or stored_path.name,
            audio_content_type=file.content_type,
            stored_audio_path=str(audio_path) if config.SAVE_UPLOADS else None,
            caller=caller,
            extension_number=extension_number,
            callee=callee,
            call_started_at=call_started_at,
            call_ended_at=call_ended_at,
            audio_duration_seconds=None,
            detected_language=None,
            speech_recognition_model=config.WHISPER_MODEL,
            summary_generation_model=config.SLLM_MODEL,
            summary_model_backend=config.SUMMARY_BACKEND,
            full_transcript=None,
            structured_call_summary=None,
            plain_call_summary=None,
            speaker_summary_list=[],
            speaker_segment_list=[],
            processing_time_ms=None,
            processing_status="failed",
            error_message=str(exc)[:1000],
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not config.SAVE_UPLOADS and stored_path.exists():
            stored_path.unlink()
        if not config.SAVE_UPLOADS and decrypted_path and decrypted_path.exists():
            decrypted_path.unlink()

    return CallProcessResponse(
        id=summary_id,
        processing_id=processing_id,
        input_type="api_request",
        audio_file_name=file.filename or stored_path.name,
        caller=caller,
        extension_number=extension_number,
        callee=callee,
        call_started_at=call_started_at,
        call_ended_at=call_ended_at,
        processing_status="completed",
        detected_language=result.detected_language,
        audio_duration_seconds=result.duration_seconds,
        full_transcript=result.transcript_text,
        summary_model_backend=result.summary_backend,
        summary_generation_model=result.summary_model,
        structured_call_summary=result.summary_text,
        plain_call_summary=result.conversational_summary_text,
        speaker_summary_list=_normalize_speaker_summaries(result.speaker_summaries),
        processing_time_ms=result.processing_ms,
    )


@app.get("/calls/{call_id}", response_model=CallSummaryItem)
def get_call(call_id: int, api_key: str = Security(verify_api_key)) -> CallSummaryItem:
    return normalize_row(_get_call_row_or_404(call_id))


@app.delete("/calls/{call_id}", response_model=DeleteResult)
def delete_call(
    call_id: int,
    delete_audio: bool = Query(default=True),
    delete_training_clips: bool = Query(default=True),
    api_key: str = Security(verify_api_key),
) -> DeleteResult:
    deleted = db.delete_call_summary(call_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="call summary not found")
    row, samples = deleted
    if delete_audio:
        _delete_call_uploads_if_safe(row)
    if delete_training_clips:
        for sample in samples:
            _delete_training_clip_if_safe(sample.get("audio_path"))
    return DeleteResult(deleted_count=1)


@app.get("/calls", response_model=list[CallSummaryItem])
def list_calls(
    limit: int = Query(default=20, ge=1, le=200),
    q: str | None = Query(default=None),
    processing_status: str | None = Query(default=None),
    input_type: str | None = Query(default=None),
    caller: str | None = Query(default=None),
    extension_number: str | None = Query(default=None),
    callee: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    api_key: str = Security(verify_api_key),
) -> list[CallSummaryItem]:
    where: list[str] = []
    params: list[Any] = []
    if q:
        keyword = f"%{q.strip()}%"
        where.append(
            "(audio_file_name LIKE %s OR processing_id LIKE %s OR full_transcript LIKE %s "
            "OR structured_call_summary LIKE %s OR plain_call_summary LIKE %s)"
        )
        params.extend([keyword, keyword, keyword, keyword, keyword])
    if processing_status:
        where.append("processing_status = %s")
        params.append(processing_status)
    if input_type:
        where.append("input_type = %s")
        params.append(input_type)
    if caller:
        where.append("caller LIKE %s")
        params.append(f"%{caller.strip()}%")
    if extension_number:
        where.append("extension_number LIKE %s")
        params.append(f"%{extension_number.strip()}%")
    if callee:
        where.append("(callee LIKE %s OR extension_number LIKE %s)")
        keyword = f"%{callee.strip()}%"
        params.extend([keyword, keyword])
    if date_from:
        where.append("result_created_at >= %s")
        params.append(date_from)
    if date_to:
        where.append("result_created_at <= %s")
        params.append(date_to)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.fetch_all(f"SELECT * FROM call_summaries {where_sql} ORDER BY id DESC LIMIT %s", (*params, limit))
    return [normalize_row(row) for row in rows]


@app.get("/calls/{call_id}/audio")
def get_call_audio(
    call_id: int,
    api_key_query: str | None = Query(default=None, alias="api_key"),
    api_key_header_value: str | None = Security(api_key_header),
) -> FileResponse:
    _verify_api_key_value(api_key_header_value or api_key_query)
    row = _get_call_row_or_404(call_id)
    file_path = row.get("stored_audio_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="audio file was not saved")
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file not found")
    return FileResponse(path, media_type=row.get("audio_content_type") or "application/octet-stream", filename=row["audio_file_name"])


@app.get("/calls/{call_id}/segments", response_model=list[SpeakerSegment])
def get_call_segments(call_id: int, api_key: str = Security(verify_api_key)) -> list[SpeakerSegment]:
    row = _get_call_row_or_404(call_id)
    return _normalize_speaker_segments(db.decode_json_field(row.get("speaker_segment_list_json")))


@app.patch("/calls/{call_id}/transcript", response_model=CallSummaryItem)
def update_call_transcript(
    call_id: int,
    payload: TranscriptUpdateRequest,
    api_key: str = Security(verify_api_key),
) -> CallSummaryItem:
    text = payload.full_transcript.strip()
    if not text:
        raise HTTPException(status_code=400, detail="full_transcript is required")
    affected = db.execute(
        """
        UPDATE call_summaries
        SET full_transcript = %s
        WHERE id = %s
        """,
        (text, call_id),
    )
    if affected < 1:
        raise HTTPException(status_code=404, detail="call summary not found")
    return normalize_row(_get_call_row_or_404(call_id))


@app.post("/calls/{call_id}/training-samples", response_model=TrainingSampleItem)
def create_training_sample(
    call_id: int,
    payload: TrainingSampleCreateRequest,
    api_key: str = Security(verify_api_key),
) -> TrainingSampleItem:
    corrected_text = payload.corrected_text.strip()
    if not corrected_text:
        raise HTTPException(status_code=400, detail="corrected_text is required")
    row = _get_call_row_or_404(call_id)
    training_audio_path = _create_training_clip(
        source_path=row.get("stored_audio_path"),
        request_id=row["processing_id"],
        segment_index=payload.segment_index,
        start_seconds=payload.start_seconds,
        end_seconds=payload.end_seconds,
    )
    sample_id = db.insert_training_sample(
        call_summary_id=call_id,
        request_id=row["processing_id"],
        filename=row["audio_file_name"],
        audio_path=training_audio_path,
        segment_index=payload.segment_index,
        speaker=payload.speaker,
        start_seconds=payload.start_seconds,
        end_seconds=payload.end_seconds,
        original_text=payload.original_text if payload.original_text is not None else row.get("full_transcript"),
        corrected_text=corrected_text,
        note=payload.note,
    )
    sample = db.fetch_one("SELECT * FROM stt_training_samples WHERE id = %s", (sample_id,))
    if not sample:
        raise HTTPException(status_code=500, detail="failed to create training sample")
    return normalize_training_sample(sample)


@app.get("/training-samples", response_model=list[TrainingSampleItem])
def list_training_samples(
    limit: int = Query(default=50, ge=1, le=200),
    status_filter: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None),
    api_key: str = Security(verify_api_key),
) -> list[TrainingSampleItem]:
    where: list[str] = []
    params: list[Any] = []
    if status_filter:
        where.append("status = %s")
        params.append(status_filter)
    if q:
        keyword = f"%{q.strip()}%"
        where.append("(filename LIKE %s OR request_id LIKE %s OR corrected_text LIKE %s)")
        params.extend([keyword, keyword, keyword])
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.fetch_all(f"SELECT * FROM stt_training_samples {where_sql} ORDER BY id DESC LIMIT %s", (*params, limit))
    return [normalize_training_sample(row) for row in rows]


@app.get("/training-samples/{sample_id}/audio")
def get_training_sample_audio(
    sample_id: int,
    api_key_query: str | None = Query(default=None, alias="api_key"),
    api_key_header_value: str | None = Security(api_key_header),
) -> FileResponse:
    _verify_api_key_value(api_key_header_value or api_key_query)
    row = db.fetch_one("SELECT * FROM stt_training_samples WHERE id = %s", (sample_id,))
    if not row:
        raise HTTPException(status_code=404, detail="training sample not found")
    audio_path = row.get("audio_path")
    if not audio_path:
        raise HTTPException(status_code=404, detail="training sample audio was not saved")
    path = Path(str(audio_path))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="training sample audio not found")
    return FileResponse(path, media_type=_audio_media_type(path), filename=path.name)


@app.delete("/training-samples/{sample_id}", response_model=DeleteResult)
def delete_training_sample(
    sample_id: int,
    delete_clip: bool = Query(default=True),
    api_key: str = Security(verify_api_key),
) -> DeleteResult:
    row = db.delete_training_sample(sample_id)
    if not row:
        raise HTTPException(status_code=404, detail="training sample not found")
    if delete_clip:
        _delete_training_clip_if_safe(row.get("audio_path"))
    return DeleteResult(deleted_count=1)


@app.delete("/training-samples", response_model=DeleteResult)
def delete_training_samples(
    status_filter: str = Query(default="queued", alias="status"),
    delete_clip: bool = Query(default=True),
    api_key: str = Security(verify_api_key),
) -> DeleteResult:
    if status_filter not in {"queued", "used", "rejected"}:
        raise HTTPException(status_code=400, detail="status must be one of queued, used, rejected")
    rows = db.delete_training_samples_by_status(status_filter)
    if delete_clip:
        for row in rows:
            _delete_training_clip_if_safe(row.get("audio_path"))
    return DeleteResult(deleted_count=len(rows))


@app.get("/training-samples/manifest")
def export_training_manifest(
    status_filter: str | None = Query(default="queued", alias="status"),
    audio_prefix: str = Query(default="/workspace/data/raw/training-clips"),
    api_key_query: str | None = Query(default=None, alias="api_key"),
    api_key_header_value: str | None = Security(api_key_header),
) -> Response:
    _verify_api_key_value(api_key_header_value or api_key_query)
    where: list[str] = []
    params: list[Any] = []
    if status_filter:
        where.append("status = %s")
        params.append(status_filter)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.fetch_all(f"SELECT * FROM stt_training_samples {where_sql} ORDER BY id ASC", params)
    lines: list[str] = []
    for row in rows:
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        audio_name = Path(str(audio_path)).name
        payload = {
            "audio": f"{audio_prefix.rstrip('/')}/{audio_name}",
            "text": row["corrected_text"],
        }
        if row.get("start_seconds") is not None and row.get("end_seconds") is not None:
            payload["start_seconds"] = float(row["start_seconds"])
            payload["end_seconds"] = float(row["end_seconds"])
            payload["segment_index"] = row.get("segment_index")
            payload["speaker"] = row.get("speaker")
        lines.append(json.dumps(payload, ensure_ascii=False))
    content = "\n".join(lines)
    if content:
        content += "\n"
    return Response(
        content=content,
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=stt-training-manifest.jsonl"},
    )


@app.post("/training/jobs", response_model=TrainingJobItem)
def start_training_job(
    payload: TrainingStartRequest,
    api_key: str = Security(verify_api_key),
) -> TrainingJobItem:
    try:
        job = training_manager.start_training_job(
            status=payload.status,
            model_name=payload.model_name,
            base_model=payload.base_model,
            include_used_samples=payload.include_used_samples,
            max_steps=payload.max_steps,
            learning_rate=payload.learning_rate,
            gpu_device=payload.gpu_device,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TrainingJobItem(**job)


@app.get("/training/jobs", response_model=list[TrainingJobItem])
def list_training_jobs(api_key: str = Security(verify_api_key)) -> list[TrainingJobItem]:
    return [TrainingJobItem(**job) for job in training_manager.list_jobs()]


@app.get("/training/jobs/{job_id}", response_model=TrainingJobItem)
def get_training_job(job_id: str, api_key: str = Security(verify_api_key)) -> TrainingJobItem:
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="training job not found")
    return TrainingJobItem(**job)


@app.get("/training/jobs/{job_id}/log")
def get_training_job_log(job_id: str, api_key: str = Security(verify_api_key)) -> Response:
    if not training_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="training job not found")
    return Response(training_manager.get_job_log(job_id), media_type="text/plain; charset=utf-8")


@app.get("/training/models", response_model=list[TrainingModelItem])
def list_training_models(api_key: str = Security(verify_api_key)) -> list[TrainingModelItem]:
    return [TrainingModelItem(**model) for model in training_manager.list_models()]


@app.get("/training/base-models", response_model=list[TrainingBaseModelItem])
def list_training_base_models(api_key: str = Security(verify_api_key)) -> list[TrainingBaseModelItem]:
    return [TrainingBaseModelItem(**model) for model in training_manager.list_base_models()]
