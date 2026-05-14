from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterable

import pymysql
from pymysql.cursors import DictCursor

from . import config


def connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME,
        connect_timeout=config.DB_CONNECT_TIMEOUT,
        cursorclass=DictCursor,
        autocommit=False,
        charset="utf8mb4",
    )


@contextmanager
def get_conn():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def wait_for_db(timeout_sec: int = 120) -> None:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    cur.fetchone()
                conn.commit()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"MariaDB not ready: {last_error}")


def ensure_schema() -> None:
    alter_statements = [
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS input_type VARCHAR(32) NOT NULL DEFAULT 'api_request' AFTER processing_id",
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS extension_number VARCHAR(64) NULL AFTER caller",
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS call_ended_at DATETIME NULL AFTER call_started_at",
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS summary_model_backend VARCHAR(64) NULL AFTER summary_generation_model",
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS plain_call_summary LONGTEXT NULL AFTER structured_call_summary",
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS speaker_summary_list_json LONGTEXT NULL AFTER plain_call_summary",
        "ALTER TABLE call_summaries ADD COLUMN IF NOT EXISTS speaker_segment_list_json LONGTEXT NULL AFTER speaker_summary_list_json",
        "ALTER TABLE call_summaries ADD INDEX IF NOT EXISTS idx_call_summaries_input_type (input_type)",
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            for statement in alter_statements:
                cur.execute(statement)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stt_training_samples (
                    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    call_summary_id BIGINT NOT NULL,
                    request_id VARCHAR(64) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    audio_path VARCHAR(1024) NULL,
                    segment_index INT NULL,
                    speaker VARCHAR(64) NULL,
                    start_seconds DECIMAL(10,3) NULL,
                    end_seconds DECIMAL(10,3) NULL,
                    original_text LONGTEXT NULL,
                    corrected_text LONGTEXT NOT NULL,
                    note VARCHAR(1000) NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'queued',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    KEY idx_stt_training_samples_call_summary_id (call_summary_id),
                    KEY idx_stt_training_samples_segment (call_summary_id, segment_index),
                    KEY idx_stt_training_samples_status (status),
                    CONSTRAINT fk_stt_training_samples_call_summary_id
                        FOREIGN KEY (call_summary_id) REFERENCES call_summaries(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            training_alters = [
                "ALTER TABLE stt_training_samples ADD COLUMN IF NOT EXISTS segment_index INT NULL AFTER audio_path",
                "ALTER TABLE stt_training_samples ADD COLUMN IF NOT EXISTS speaker VARCHAR(64) NULL AFTER segment_index",
                "ALTER TABLE stt_training_samples ADD COLUMN IF NOT EXISTS start_seconds DECIMAL(10,3) NULL AFTER speaker",
                "ALTER TABLE stt_training_samples ADD COLUMN IF NOT EXISTS end_seconds DECIMAL(10,3) NULL AFTER start_seconds",
                "ALTER TABLE stt_training_samples ADD INDEX IF NOT EXISTS idx_stt_training_samples_segment (call_summary_id, segment_index)",
            ]
            for statement in training_alters:
                cur.execute(statement)
        conn.commit()


def fetch_one(query: str, params: Iterable[Any] | None = None) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            row = cur.fetchone()
        conn.commit()
        return row


def fetch_all(query: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            rows = cur.fetchall()
        conn.commit()
        return list(rows)


def execute(query: str, params: Iterable[Any] | None = None) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            affected = cur.execute(query, params or ())
        conn.commit()
        return int(affected)


def decode_json_field(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def _normalize_speaker_summary_list(items: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in items or []:
        speaker_name = str(item.get("speaker_name") or item.get("speaker") or "").strip()
        speaker_summary = str(item.get("speaker_summary") or item.get("summary_text") or "").strip()
        if speaker_name and speaker_summary:
            normalized.append({"speaker_name": speaker_name, "speaker_summary": speaker_summary})
    return normalized


def insert_summary(
    *,
    processing_id: str,
    input_type: str,
    audio_file_name: str,
    audio_content_type: str | None,
    stored_audio_path: str | None,
    caller: str | None,
    extension_number: str | None,
    callee: str | None,
    call_started_at,
    call_ended_at,
    audio_duration_seconds: float | None,
    detected_language: str | None,
    speech_recognition_model: str,
    summary_generation_model: str,
    summary_model_backend: str | None,
    full_transcript: str | None,
    structured_call_summary: str | None,
    plain_call_summary: str | None,
    speaker_summary_list: list[dict[str, Any]] | None,
    speaker_segment_list: list[dict[str, Any]] | None,
    processing_time_ms: int | None,
    processing_status: str,
    error_message: str | None,
) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO call_summaries (
                    processing_id, input_type, audio_file_name, audio_content_type, stored_audio_path,
                    caller, extension_number, callee, call_started_at, call_ended_at,
                    audio_duration_seconds, detected_language, speech_recognition_model, summary_generation_model, summary_model_backend,
                    full_transcript, structured_call_summary, plain_call_summary,
                    speaker_summary_list_json, speaker_segment_list_json,
                    processing_time_ms, processing_status, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    processing_id,
                    input_type,
                    audio_file_name,
                    audio_content_type,
                    stored_audio_path,
                    caller,
                    extension_number,
                    callee,
                    call_started_at,
                    call_ended_at,
                    audio_duration_seconds,
                    detected_language,
                    speech_recognition_model,
                    summary_generation_model,
                    summary_model_backend,
                    full_transcript,
                    structured_call_summary,
                    plain_call_summary,
                    json.dumps(_normalize_speaker_summary_list(speaker_summary_list), ensure_ascii=False),
                    json.dumps(speaker_segment_list or [], ensure_ascii=False),
                    processing_time_ms,
                    processing_status,
                    error_message,
                ),
            )
            summary_id = cur.lastrowid
        conn.commit()
        return int(summary_id)


def insert_training_sample(
    *,
    call_summary_id: int,
    request_id: str,
    filename: str,
    audio_path: str | None,
    segment_index: int | None,
    speaker: str | None,
    start_seconds: float | None,
    end_seconds: float | None,
    original_text: str | None,
    corrected_text: str,
    note: str | None,
) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if segment_index is not None:
                cur.execute(
                    """
                    SELECT id FROM stt_training_samples
                    WHERE call_summary_id = %s AND segment_index = %s AND status = 'queued'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (call_summary_id, segment_index),
                )
                existing = cur.fetchone()
                if existing:
                    cur.execute(
                        """
                        UPDATE stt_training_samples
                        SET audio_path = %s,
                            speaker = %s,
                            start_seconds = %s,
                            end_seconds = %s,
                            original_text = %s,
                            corrected_text = %s,
                            note = %s
                        WHERE id = %s
                        """,
                        (
                            audio_path,
                            speaker,
                            start_seconds,
                            end_seconds,
                            original_text,
                            corrected_text,
                            note,
                            existing["id"],
                        ),
                    )
                    conn.commit()
                    return int(existing["id"])
            cur.execute(
                """
                INSERT INTO stt_training_samples (
                    call_summary_id, request_id, filename, audio_path,
                    segment_index, speaker, start_seconds, end_seconds,
                    original_text, corrected_text, note, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'queued')
                """,
                (
                    call_summary_id,
                    request_id,
                    filename,
                    audio_path,
                    segment_index,
                    speaker,
                    start_seconds,
                    end_seconds,
                    original_text,
                    corrected_text,
                    note,
                ),
            )
            sample_id = cur.lastrowid
        conn.commit()
        return int(sample_id)


def delete_training_sample(sample_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM stt_training_samples WHERE id = %s", (sample_id,))
            row = cur.fetchone()
            if not row:
                conn.commit()
                return None
            cur.execute("DELETE FROM stt_training_samples WHERE id = %s", (sample_id,))
        conn.commit()
        return row


def delete_training_samples_by_status(status: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM stt_training_samples WHERE status = %s", (status,))
            rows = list(cur.fetchall())
            cur.execute("DELETE FROM stt_training_samples WHERE status = %s", (status,))
        conn.commit()
        return rows


def delete_call_summary(call_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM call_summaries WHERE id = %s", (call_id,))
            row = cur.fetchone()
            if not row:
                conn.commit()
                return None
            cur.execute("SELECT * FROM stt_training_samples WHERE call_summary_id = %s", (call_id,))
            samples = list(cur.fetchall())
            cur.execute("DELETE FROM stt_training_samples WHERE call_summary_id = %s", (call_id,))
            cur.execute("DELETE FROM call_summaries WHERE id = %s", (call_id,))
        conn.commit()
        return row, samples
