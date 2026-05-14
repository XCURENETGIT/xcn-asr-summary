from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SpeakerSegment(BaseModel):
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


class SpeakerSummary(BaseModel):
    speaker_name: str
    speaker_summary: str


class CallProcessResponse(BaseModel):
    id: int
    processing_id: str
    input_type: str = "api_request"
    audio_file_name: str
    caller: str | None = None
    extension_number: str | None = None
    callee: str | None = None
    call_started_at: datetime | None = None
    call_ended_at: datetime | None = None
    processing_status: str
    detected_language: str | None = None
    audio_duration_seconds: float | None = None
    full_transcript: str | None = None
    summary_model_backend: str | None = None
    summary_generation_model: str | None = None
    structured_call_summary: str | None = None
    plain_call_summary: str | None = None
    speaker_summary_list: list[SpeakerSummary] = Field(default_factory=list)
    processing_time_ms: int | None = None


class CallSummaryItem(BaseModel):
    id: int
    processing_id: str
    input_type: str = "api_request"
    audio_file_name: str
    caller: str | None = None
    extension_number: str | None = None
    callee: str | None = None
    call_started_at: datetime | None = None
    call_ended_at: datetime | None = None
    detected_language: str | None = None
    audio_duration_seconds: float | None = None
    speech_recognition_model: str
    summary_generation_model: str
    summary_model_backend: str | None = None
    structured_call_summary: str | None = None
    plain_call_summary: str | None = None
    speaker_summary_list: list[SpeakerSummary] = Field(default_factory=list)
    processing_status: str
    error_message: str | None = None
    processing_time_ms: int | None = None
    result_created_at: datetime


class HealthResponse(BaseModel):
    status: str
    db_ready: bool
    whisper_model: str
    summary_backend: str
    summary_model: str
    sllm_configured: bool


class TranscriptUpdateRequest(BaseModel):
    full_transcript: str


class TrainingSampleCreateRequest(BaseModel):
    corrected_text: str
    note: str | None = None
    segment_index: int | None = None
    speaker: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    original_text: str | None = None


class TrainingSampleItem(BaseModel):
    id: int
    call_summary_id: int
    request_id: str
    filename: str
    audio_path: str | None = None
    segment_index: int | None = None
    speaker: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    original_text: str | None = None
    corrected_text: str
    note: str | None = None
    status: str
    created_at: datetime


class TrainingStartRequest(BaseModel):
    status: str = "queued"
    model_name: str | None = None
    base_model: str | None = None
    include_used_samples: bool = True
    max_steps: int = 40
    learning_rate: float = 0.00002
    gpu_device: str | None = None


class TrainingJobItem(BaseModel):
    id: str
    status: str
    model_name: str
    sample_count: int = 0
    queued_sample_count: int | None = None
    reused_sample_count: int | None = None
    base_model: str | None = None
    hf_model_path: str | None = None
    max_steps: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    model_path: str | None = None
    metrics: dict | None = None


class TrainingModelItem(BaseModel):
    name: str
    path: str
    is_active: bool = False
    has_hf_checkpoint: bool = False
    hf_model_path: str | None = None
    created_at: str | None = None
    size_bytes: int | None = None
    metrics: dict | None = None


class TrainingBaseModelItem(BaseModel):
    name: str
    value: str
    path: str | None = None
    source: str
    is_active: bool = False


class DeleteResult(BaseModel):
    deleted_count: int
