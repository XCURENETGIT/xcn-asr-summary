from __future__ import annotations

import logging
import re
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from . import config
from .sllm_client import request_summary

logger = logging.getLogger("xcn-asr-summary")

STEREO_CHANNEL_COUNT = 2


@dataclass
class PipelineResult:
    transcript_text: str
    summary_backend: str
    summary_model: str
    summary_text: str
    conversational_summary_text: str
    speaker_summaries: list[dict[str, str]]
    speaker_segments: list[dict[str, str | float]]
    detected_language: str | None
    duration_seconds: float | None
    processing_ms: int


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class TranscriptWord:
    start_seconds: float
    end_seconds: float
    word: str


@dataclass
class SpeakerTurn:
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class DiarizationSegment:
    speaker: str
    start_seconds: float
    end_seconds: float


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        local_files_only=config.MODEL_LOCAL_FILES_ONLY,
    )


@lru_cache(maxsize=1)
def get_diarization_pipeline():
    if not config.DIARIZATION_ENABLED:
        return None
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is not installed") from exc

    token = config.HF_TOKEN or None
    try:
        pipeline = Pipeline.from_pretrained(config.PYANNOTE_MODEL, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(config.PYANNOTE_MODEL, use_auth_token=token)
    if pipeline is None:
        logger.warning("pyannote pipeline is not available: model=%s token_set=%s", config.PYANNOTE_MODEL, bool(token))
        return None

    device_name = config.PYANNOTE_DEVICE or config.WHISPER_DEVICE
    if device_name:
        try:
            import torch

            device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
            pipeline.to(device)
        except Exception as exc:
            logger.warning("failed to move pyannote pipeline to %s: %s", device_name, exc)
    return pipeline


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"([.!?])(?=[^\s])", r"\1 ", text)
    return text


def _summary_prompt(text: str, *, mode: str, speaker: str | None = None) -> str:
    if mode == "speaker" and speaker:
        return (
            f"다음은 {speaker}의 통화 발화 내용이다.\n"
            "화자의 핵심 발언, 요청, 안내, 결론만 한국어로 요약하라.\n"
            "가능하면 문의내용 또는 안내내용이 바로 드러나게 1~2문장으로 정리하라.\n"
            "군더더기 표현, 인사말, 반복 문구는 제외하라.\n"
            "주어진 발화에 없는 사실은 추가하지 말고, 모호한 내용은 '확인 필요'로 표기하라.\n\n"
            f"{text}"
        )
    return (
        "다음은 전화 통화 전사 내용이다.\n"
        "통화의 핵심 내용, 처리 결과, 중요한 요청사항만 한국어로 요약하라.\n"
        "문의내용, 안내내용, 처리결과, 후속조치가 드러나게 2~4문장으로 정리하라.\n"
        "자동 안내 문구, 불필요한 인사말, 반복 표현은 제외하라.\n"
        "주어진 통화 내용에 없는 사실은 추가하지 말고, 모호한 내용은 '확인 필요'로 표기하라.\n\n"
        f"{text}"
    )


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _compact_speaker_name(speaker: str) -> str:
    if speaker.startswith("speaker_"):
        suffix = speaker.split("_")[-1]
        if suffix.isdigit():
            return f"S{suffix}"
    return speaker


def _format_turn_transcript(turns: list[SpeakerTurn]) -> str:
    lines: list[str] = []
    for turn in turns:
        start = _format_timestamp(turn.start_seconds)
        text = _normalize_text(turn.text)
        if not text:
            continue
        lines.append(f"[{start}] {_compact_speaker_name(turn.speaker)}: {text}")
    return "\n".join(lines).strip()


def _summary_prompt_from_turns(turn_transcript: str) -> str:
    return (
        "다음은 전화 통화의 원본 발화 기록이다.\n"
        "각 줄은 [시각] 고객/상담사: 발화 형식이다.\n"
        "고객의 요구사항과 상담사의 조치를 명확히 구분해서 작성하라.\n"
        "통화 내용에 없는 사실은 절대 추가하지 말라.\n"
        "특히 금액, 일정, 약속 사항, 발송 예정, 방문 예정, 처리 완료 여부는 원문에 있을 때만 작성하라.\n"
        "원문에 없는 후속 조치나 리스크를 일반론으로 만들지 말라.\n"
        "불확실한 내용은 '확인 필요'로 표시하라.\n"
        "고객 감정과 한 줄 요약 항목은 작성하지 말라.\n"
        "항목별 1~2줄 이내로 핵심만 작성하라.\n"
        "답변은 반드시 아래 라벨을 그대로 사용하라.\n"
        "통화 목적:\n"
        "핵심 이슈:\n"
        "상담사 안내:\n"
        "처리 결과:\n"
        "리스크/특이사항:\n"
        "후속 조치:\n\n"
        f"{turn_transcript}"
    )


def _fit_turn_transcript_to_budget(turns: list[SpeakerTurn], max_chars: int) -> str:
    full_text = _format_turn_transcript(turns)
    if len(full_text) <= max_chars:
        return full_text

    lines = [line for line in full_text.splitlines() if line.strip()]
    if not lines:
        return full_text[:max_chars].strip()

    omission = "[...중간 발화 생략...]"
    reserved = len(omission) + 2
    budget = max(200, max_chars - reserved)
    head_budget = max(80, int(budget * 0.4))
    tail_budget = max(80, budget - head_budget)

    head: list[str] = []
    used = 0
    head_end = -1
    for idx, line in enumerate(lines):
        line_cost = len(line) + 1
        if head and used + line_cost > head_budget:
            break
        head.append(line)
        used += line_cost
        head_end = idx

    tail: list[str] = []
    used = 0
    tail_start = len(lines)
    for idx in range(len(lines) - 1, head_end, -1):
        line = lines[idx]
        line_cost = len(line) + 1
        if tail and used + line_cost > tail_budget:
            break
        tail.append(line)
        used += line_cost
        tail_start = idx

    tail.reverse()
    if head_end + 1 >= tail_start:
        compact = head + [line for line in tail if line not in head]
        return "\n".join(compact)[:max_chars].strip()
    return "\n".join(head + [omission] + tail).strip()


SUMMARY_LABELS = (
    "통화 목적",
    "핵심 이슈",
    "상담사 안내",
    "처리 결과",
    "리스크/특이사항",
    "후속 조치",
)

IGNORED_SUMMARY_LABELS = (
    "고객 감정",
    "한 줄 요약",
)


def _display_role_for_speaker(speaker: str, speaker_roles: dict[str, str]) -> str:
    role = speaker_roles.get(speaker)
    if role == "customer":
        return "고객"
    if role == "agent":
        return "상담사"
    return _compact_speaker_name(speaker)


def _is_low_signal_turn(turn: SpeakerTurn) -> bool:
    text = _normalize_text(turn.text)
    if not text:
        return True
    if _is_repeated_low_signal_hallucination(text, turn.end_seconds - turn.start_seconds):
        return True
    if len(text) <= config.SPEAKER_ACK_MAX_CHARS and _is_acknowledgement(text):
        return True
    if len(text) <= 3 and text in {"네", "예", "아", "음", "어"}:
        return True
    return False


def _format_llm_turn_transcript(turns: list[SpeakerTurn]) -> str:
    speaker_roles = _infer_speaker_roles(turns)
    lines: list[str] = []
    for turn in turns:
        if _is_low_signal_turn(turn):
            continue
        start = _format_timestamp(turn.start_seconds)
        role = _display_role_for_speaker(turn.speaker, speaker_roles)
        text = _normalize_text(turn.text)
        if text:
            lines.append(f"[{start}] {role}: {text}")
    if lines:
        return "\n".join(lines).strip()
    return _format_turn_transcript(turns)


def _chunk_turns_by_chars(turns: list[SpeakerTurn], max_chars: int) -> list[list[SpeakerTurn]]:
    if not turns:
        return []
    chunks: list[list[SpeakerTurn]] = []
    current: list[SpeakerTurn] = []
    current_chars = 0
    budget = max(1200, max_chars)
    for turn in turns:
        line = _format_llm_turn_transcript([turn])
        line_chars = len(line) + 1
        if current and current_chars + line_chars > budget:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += line_chars
    if current:
        chunks.append(current)
    return chunks


def _parse_labeled_summary(text: str) -> dict[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return {}
    label_pattern = "|".join(re.escape(label) for label in (*SUMMARY_LABELS, *IGNORED_SUMMARY_LABELS))
    matches = list(
        re.finditer(
            rf"(?:^|\n|\s)\s*(?:[-*]\s*)?(?:\[)?({label_pattern})(?:\])?\s*[:：]\s*",
            cleaned,
        )
    )
    parsed: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1)
        if label in IGNORED_SUMMARY_LABELS:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        value = cleaned[start:end].strip()
        value = re.sub(r"\n\s*", " ", value).strip(" -")
        if value:
            normalized_value = _normalize_text(value)
            existing = parsed.get(label, "")
            if existing and ("확인 필요" in normalized_value or len(normalized_value) < len(existing)):
                continue
            parsed[label] = normalized_value
    return parsed


def _format_labeled_summary(parsed: dict[str, str]) -> str:
    return "\n".join(
        f"{label}: {parsed.get(label) or '확인 필요'}"
        for label in SUMMARY_LABELS
    )


def _split_summary_clauses(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s+-\s+", normalized)
    return [part.strip(" -") for part in parts if part.strip(" -")]


def _sanitize_structured_summary(parsed: dict[str, str], source_text: str) -> dict[str, str]:
    source = _normalize_text(source_text)
    sanitized = dict(parsed)

    sanitized.pop("고객 감정", None)
    sanitized.pop("한 줄 요약", None)

    action_text = sanitized.get("후속 조치", "")
    if action_text and action_text != "확인 필요":
        kept: list[str] = []
        for clause in _split_summary_clauses(action_text):
            if "문자" in clause and "주문 확인 문자" not in source and "문자 발송" not in source:
                continue
            if "이메일" in clause and "이메일" not in source:
                continue
            if "방문" in clause and "방문" not in source:
                continue
            if "재연락" in clause and "재연락" not in source and "다시 연락" not in source:
                continue
            if "안내 필요" in clause and clause not in source:
                continue
            kept.append(clause)
        sanitized["후속 조치"] = " - ".join(kept) if kept else "확인 필요"

    risk_text = sanitized.get("리스크/특이사항", "")
    if risk_text and risk_text != "확인 필요":
        kept = []
        for clause in _split_summary_clauses(risk_text):
            if "보안" in clause and "보안" not in source and "카드번호" not in source:
                continue
            if "리스크" in clause and clause not in source:
                continue
            kept.append(clause)
        sanitized["리스크/특이사항"] = " - ".join(kept) if kept else "확인 필요"

    return sanitized


def _summary_prompt_from_chunk(chunk_transcript: str, *, chunk_index: int, chunk_count: int) -> str:
    return (
        f"다음은 전화 통화의 {chunk_count}개 구간 중 {chunk_index}번째 구간이다.\n"
        "이 구간에서 확인되는 사실만 한국어로 구조화하라.\n"
        "고객의 요구사항과 상담사의 조치를 명확히 구분하라.\n"
        "전체 통화 결론을 추정하지 말고, 이 구간에 없는 내용은 '확인 필요'로 표기하라.\n"
        "금액, 일정, 약속 사항, 발송 예정, 방문 예정, 처리 완료 여부는 원문에 있을 때만 작성하라.\n"
        "문자 발송, 이메일 발송, 배송 일정 추가 안내처럼 통화에서 약속하지 않은 예정 조치는 만들지 말라.\n"
        "고객 감정과 한 줄 요약 항목은 작성하지 말라.\n"
        "답변은 반드시 아래 라벨을 그대로 사용하라.\n"
        "통화 목적:\n"
        "핵심 이슈:\n"
        "상담사 안내:\n"
        "처리 결과:\n"
        "리스크/특이사항:\n"
        "후속 조치:\n\n"
        f"{chunk_transcript}"
    )


def _summary_prompt_from_chunk_summaries(chunk_summaries: list[str]) -> str:
    joined = "\n\n".join(
        f"[구간 {idx} 요약]\n{summary}"
        for idx, summary in enumerate(chunk_summaries, start=1)
        if summary.strip()
    )
    return (
        "다음은 긴 전화 통화를 구간별로 요약한 내용이다.\n"
        "구간 요약에 명시된 사실만 통합하고, 없는 사실은 추가하지 말라.\n"
        "중복 표현을 제거하고 최종 통화 요약을 아래 라벨 형식으로 작성하라.\n"
        "고객의 요구사항과 상담사의 조치를 명확히 구분하라.\n"
        "금액, 일정, 약속 사항, 발송 예정, 방문 예정, 처리 완료 여부는 구간 요약에 있을 때만 작성하라.\n"
        "문자 발송, 이메일 발송, 배송 일정 추가 안내처럼 구간 요약에 없는 예정 조치는 만들지 말라.\n"
        "고객 감정과 한 줄 요약 항목은 작성하지 말라.\n"
        "모호한 내용은 '확인 필요'로 표기하라.\n"
        "통화 목적:\n"
        "핵심 이슈:\n"
        "상담사 안내:\n"
        "처리 결과:\n"
        "리스크/특이사항:\n"
        "후속 조치:\n\n"
        f"{joined}"
    )


def _conversational_summary_prompt(text: str) -> str:
    return (
        "다음은 전화 통화 내용이다.\n"
        "전체 흐름을 누군가에게 설명하듯 자연스러운 구어체 한국어로 요약하라.\n"
        "라벨이나 목록을 쓰지 말고 3~5문장의 문단으로 작성하라.\n"
        "고객이 무엇을 문의했고 상담사가 어떻게 안내했으며 통화가 어떻게 마무리됐는지 포함하라.\n"
        "원문에 없는 사실, 약속, 일정, 발송 예정은 추가하지 말라.\n"
        "확실하지 않은 내용은 단정하지 말고 확인이 필요하다고 표현하라.\n\n"
        f"{text}"
    )


def _build_conversational_summary(turns: list[SpeakerTurn], transcript_text: str) -> str:
    chunks = _chunk_turns_by_chars(turns, config.SLLM_MAX_PROMPT_CHARS)
    if not chunks:
        return _fallback_general_summary(transcript_text)

    if len(chunks) == 1:
        source = _format_turn_transcript(chunks[0]) or transcript_text
        return _summarize_prompt_with_sllm(_conversational_summary_prompt(source), fallback_text=transcript_text)

    chunk_summaries: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source = _format_turn_transcript(chunk)
        if not source:
            continue
        chunk_summaries.append(
            _summarize_prompt_with_sllm(
                _conversational_summary_prompt(
                    f"아래 내용은 전체 통화 중 {len(chunks)}개 구간 중 {idx}번째 구간이다.\n{source}"
                ),
                fallback_text=source,
            )
        )
    joined = "\n".join(f"- {summary}" for summary in chunk_summaries if summary.strip())
    if not joined:
        return _fallback_general_summary(transcript_text)
    return _summarize_prompt_with_sllm(
        _conversational_summary_prompt(
            "다음은 긴 전화 통화를 구간별로 자연스럽게 요약한 내용이다. "
            "중복을 제거하고 전체 통화 흐름이 이어지도록 하나의 구어체 문단으로 다시 정리하라.\n"
            f"{joined}"
        ),
        fallback_text=joined,
    )


def _summarize_structured_with_sllm(prompt: str) -> tuple[str, dict[str, str]]:
    summary_text = _dedupe_repeated_phrases(request_summary(prompt)).strip()
    parsed = _parse_labeled_summary(summary_text)
    if not parsed:
        return summary_text, {}
    return _format_labeled_summary(parsed), parsed


def _build_sllm_structured_summary(turns: list[SpeakerTurn], transcript_text: str) -> tuple[str, str, str, str, str]:
    chunks = _chunk_turns_by_chars(turns, config.SLLM_MAX_PROMPT_CHARS)
    if not chunks:
        fallback = _fallback_general_summary(transcript_text)
        return _format_structured_summary(fallback, "", "", ""), fallback, "", "", ""

    if len(chunks) == 1:
        summary_text, parsed = _summarize_structured_with_sllm(
            _summary_prompt_from_turns(_format_turn_transcript(chunks[0]))
        )
    else:
        chunk_summaries: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_summary, _ = _summarize_structured_with_sllm(
                _summary_prompt_from_chunk(
                    _format_turn_transcript(chunk),
                    chunk_index=idx,
                    chunk_count=len(chunks),
                )
            )
            chunk_summaries.append(chunk_summary)
        summary_text, parsed = _summarize_structured_with_sllm(
            _summary_prompt_from_chunk_summaries(chunk_summaries)
        )

    if not parsed:
        fallback = _fallback_general_summary(transcript_text)
        return _format_structured_summary(fallback, "", "", ""), fallback, "", "", ""

    parsed = _sanitize_structured_summary(parsed, transcript_text)
    summary_text = _format_labeled_summary(parsed)
    inquiry_summary = " ".join(
        item for item in (
            parsed.get("통화 목적", ""),
            parsed.get("핵심 이슈", ""),
        )
        if item and item != "확인 필요"
    ).strip()
    guidance_summary = parsed.get("상담사 안내", "")
    call_outcome = parsed.get("처리 결과", "")
    action_items = parsed.get("후속 조치", "")
    return summary_text, inquiry_summary, guidance_summary, action_items, call_outcome


def _summarize_prompt_with_sllm(prompt: str, *, fallback_text: str) -> str:
    summary_text = _trim_to_complete_sentence(_dedupe_repeated_phrases(request_summary(prompt)))
    if _summary_quality_is_bad(summary_text) or _looks_like_source_excerpt(summary_text, fallback_text):
        return _trim_to_complete_sentence(_dedupe_repeated_phrases(_fallback_general_summary(fallback_text)))
    return summary_text


def _contains_ivr_marker(text: str) -> bool:
    cleaned = _normalize_text(text)
    ivr_markers = [
        "보이스피싱",
        "피해 신고",
        "조회 1번",
        "조회 2번",
        "문의 4번",
        "문의 5번",
        "문의 6번",
        "문의 7번",
        "문의 8번",
        "자동 안내",
        "ARS",
        "금융 사기",
        "산업안전보건법",
        "폭언이나 욕설",
        "성희롱",
        "따뜻한 말 한마디",
        "상담 내용은 녹음",
        "상담이 제한",
        "고객님 곧 연결",
        "곧 연결하겠습니다",
        "고객님의 주민번호",
        "계좌 비밀번호 4자리",
        "원활한 상담을 위해",
        "소통과 배려로",
        "통화 연결 시",
        "통화 연결 후에는",
        "직원 연결은",
        "다시 듣기",
        "각종 서식 팩스",
        "민생회복소비",
    ]
    ivr_regexes = [
        r"\d+\s*번(?:을|은|으로)?",
        r"(?:은행|카드|보험|증권|캐피탈|상담센터|고객센터|콜센터)(?:입니다|입니다\.)",
        r"(?:앱|어플|홈페이지).*(?:다운로드|설치|이용).*(?:편리|가능)",
        r"(?:조회|증명서\s*발급|업무).*(?:이용|가능)",
        r"(?:상담|통화).*녹음",
        r"(?:주민번호|주민등록번호|계좌\s*비밀번호).*눌러",
    ]
    return any(marker in cleaned for marker in ivr_markers) or any(re.search(pattern, cleaned) for pattern in ivr_regexes)


def _contains_conversation_marker(text: str) -> bool:
    cleaned = _normalize_text(text)
    conversation_markers = ["여보세요", "문의드리려고", "상담사", "무엇을 도와", "문의드리는데"]
    return any(marker in cleaned for marker in conversation_markers)


def _strip_ivr_preamble(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return cleaned
    if not _contains_ivr_marker(cleaned):
        return cleaned
    conversation_markers = ["무엇을 도와", "여보세요", "문의드리려고", "상담사", "문의드리는데"]
    cut_index: int | None = None
    for marker in conversation_markers:
        idx = cleaned.find(marker)
        if idx != -1 and idx > 80:
            cut_index = idx
            break
    if cut_index is None:
        return cleaned
    return cleaned[cut_index:].strip()


def _dedupe_repeated_phrases(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return cleaned
    for size in range(6, 1, -1):
        pattern = re.compile(rf"((?:\S+\s+){{{size - 1}}}\S+)(?:\s+\1)+")
        cleaned = pattern.sub(r"\1", cleaned)
    return cleaned


def _is_repeated_low_signal_hallucination(text: str, duration_seconds: float | None = None) -> bool:
    cleaned = _normalize_text(text).strip()
    if not cleaned:
        return True

    low_signal_words = {
        "네", "예", "응", "음", "어", "아", "흠",
        "네네", "예예", "응응", "음음", "어어", "아아",
    }
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", cleaned)
    if not tokens:
        return True

    compact = re.sub(r"[\s,.;:!?~\-]+", "", cleaned)
    if re.fullmatch(r"(?:네|예|응|음|어|아|흠){6,}", compact):
        return True

    low_signal_count = sum(1 for token in tokens if token in low_signal_words)
    top_count = max(tokens.count(token) for token in set(tokens))
    token_count = len(tokens)
    low_signal_ratio = low_signal_count / token_count
    top_ratio = top_count / token_count
    duration = max(0.0, float(duration_seconds or 0.0))

    if token_count >= 8 and low_signal_ratio >= 0.85 and top_ratio >= 0.65:
        return True
    if duration and duration <= 2.5 and token_count >= 4 and low_signal_ratio >= 0.8:
        return True
    if duration and duration <= 2.0 and len(cleaned) >= 30 and top_ratio >= 0.7:
        return True

    sentence_parts = [
        re.sub(r"[\s,.;:!?~\-]+", "", item)
        for item in re.split(r"(?<=[.!?])\s+", cleaned)
        if item.strip()
    ]
    sentence_parts = [item for item in sentence_parts if item]
    if len(sentence_parts) >= 4:
        unique_parts = set(sentence_parts)
        if len(unique_parts) <= 2 and all(part in low_signal_words for part in unique_parts):
            return True

    return False


def _trim_to_complete_sentence(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return cleaned
    match = re.search(r"^.*[.!?다요]\b", cleaned)
    if match:
        return match.group(0).strip()
    sentence_match = re.search(r"^.*[.!?]", cleaned)
    if sentence_match:
        return sentence_match.group(0).strip()
    return cleaned


def _is_low_value_sentence(sentence: str) -> bool:
    cleaned = _normalize_text(sentence)
    if not cleaned:
        return True
    low_value_patterns = [
        r"^여보세요[.!?]?$",
        r"^네[.!?]?$",
        r"^예[.!?]?$",
        r"^(?:네|예)[,\s]*(?:과장님|부장님|팀장님|차장님|대표님|고객님)[.!?]?$",
        r"^감사합니다[.!?]?$",
        r"^건강하세요[.!?]?$",
        r"^다시 듣기",
        r"^직원 연결",
        r"^소통과 배려로",
        r"^통화 연결",
        r"^센터입니다[.!?]?$",
        r"^안녕하세요[.!?]?$",
        r"^고객님[.!?]?$",
        r"^고객님의 주민번호",
        r"^원활한 상담을 위해",
    ]
    return any(re.search(pattern, cleaned) for pattern in low_value_patterns)


def _split_meaningful_sentences(text: str) -> list[str]:
    cleaned = _normalize_text(text)
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", cleaned) if item.strip()]
    return [item for item in sentences if not _is_low_value_sentence(item)]


def _starts_like_raw_utterance(text: str) -> bool:
    cleaned = _normalize_text(text).strip(" \"'")
    return bool(
        re.match(
            r"^(?:네|예|아|어)[,\s]+(?:과장님|부장님|팀장님|차장님|대표님|고객님|그|저|제가|이거|그거)",
            cleaned,
        )
    )


def _looks_like_source_excerpt(summary_text: str, source_text: str) -> bool:
    summary = _normalize_text(summary_text)
    source = _normalize_text(source_text)
    if not summary or not source:
        return False
    if len(summary) >= 40 and summary in source:
        return True
    summary_compact = re.sub(r"\s+", "", summary)
    source_compact = re.sub(r"\s+", "", source)
    return len(summary_compact) >= 40 and summary_compact in source_compact


def _is_acknowledgement(text: str) -> bool:
    cleaned = _normalize_text(text).strip(" .!?")
    ack_words = {
        "네", "예", "네네", "아 네", "예예", "알겠습니다", "네 알겠습니다",
        "감사합니다", "고맙습니다", "잠시만요", "네 감사합니다",
    }
    return len(cleaned) <= config.SPEAKER_ACK_MAX_CHARS and cleaned in ack_words


def _split_dialogue_units(text: str) -> list[str]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return []

    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", cleaned) if item.strip()]
    if not sentences:
        sentences = [cleaned]

    cue_pattern = re.compile(
        r"(?=(?:네,\s*)?(?:고객님|확인(?:해서|해|해보|되)|도와드리|ARS\s*연결|"
        r"비밀번호|정보\s*보호|말씀\s*부탁|문자|경로|보내드릴|가능하세요|"
        r"필요하시고|제가\s+다운로드)|(?:제가|저는|아니,|그럼|그러면|그리고|"
        r"요즘|근데|차는|일단|배차가|알겠습니다|해지|인증서))"
    )

    units: list[str] = []
    for sentence in sentences:
        parts = [part.strip(" ,") for part in cue_pattern.split(sentence) if part.strip(" ,")]
        if len(parts) <= 1:
            units.append(sentence)
            continue
        units.extend(parts)
    return [
        unit for unit in units
        if unit
        and not _is_low_value_sentence(unit)
        and not (_contains_ivr_marker(unit) and not _contains_conversation_marker(unit))
    ]


def _has_customer_phrase(text: str) -> bool:
    cleaned = _normalize_text(text)
    customer_regexes = [
        r"(?:제가|저는|저희가)\s+.+(?:하려고|하고|문의|요청|신청|취소|변경)",
        r"(?:어떻게|왜|언제|얼마|어디|뭐|무엇).*(?:되나요|돼요|인가요|가능|해요|\?)",
        r"(?:안\s*되|못\s*하|찾을 수|모르겠|궁금|문의)",
        r"(?:해지|신청|취소|변경|등록|인증|로그인|결제|환불|배송|예약).*(?:하려고|하고 싶은|되나요|인가요|\?)",
        r"(?:알겠습니다|네|예).*(?:해주세요|부탁|궁금|문의|\?)",
    ]
    return any(re.search(pattern, cleaned) for pattern in customer_regexes)


def _has_agent_phrase(text: str) -> bool:
    cleaned = _normalize_text(text)
    agent_regexes = [
        r"고객님",
        r"(?:확인|조회|처리|접수|안내).*(?:하겠습니다|해드리|도와드리|드릴|됩니다|가능)",
        r"(?:말씀|입력|확인|인증).*(?:부탁|주시겠)",
        r"(?:가능|불가|어렵|됩니다|되세요|가능하세요|필요하세요|필요하시고)",
        r"(?:문자|경로|URL|서류|링크).*(?:보내|발송|안내)",
        r"(?:기다려|잠시만|연결해드리|도와드릴)",
        r"(?:무엇을|어떤).*도와",
        r"(?:로그인|인증|선택|접속).*(?:하시면|해주세요|가능)",
    ]
    return any(re.search(pattern, cleaned) for pattern in agent_regexes)


def _split_segments_for_speaker_assignment(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    split_segments: list[TranscriptSegment] = []
    for segment in segments:
        units = _split_dialogue_units(segment.text)
        if len(units) <= 1:
            split_segments.append(TranscriptSegment(**asdict(segment)))
            continue

        duration = max(0.0, segment.end_seconds - segment.start_seconds)
        total_chars = sum(max(1, len(unit)) for unit in units)
        cursor = segment.start_seconds
        for idx, unit in enumerate(units):
            if idx == len(units) - 1:
                end = segment.end_seconds
            else:
                ratio = max(1, len(unit)) / total_chars
                end = min(segment.end_seconds, cursor + duration * ratio)
            split_segments.append(
                TranscriptSegment(
                    start_seconds=cursor,
                    end_seconds=end,
                    text=unit,
                )
            )
            cursor = end
    return split_segments


def _run_diarization(audio_path: Path) -> list[DiarizationSegment]:
    if not config.DIARIZATION_ENABLED:
        return []
    try:
        pipeline = get_diarization_pipeline()
        if pipeline is None:
            return []

        kwargs: dict[str, int] = {}
        if config.PYANNOTE_NUM_SPEAKERS > 0:
            kwargs["num_speakers"] = config.PYANNOTE_NUM_SPEAKERS
        else:
            if config.PYANNOTE_MIN_SPEAKERS > 0:
                kwargs["min_speakers"] = config.PYANNOTE_MIN_SPEAKERS
            if config.PYANNOTE_MAX_SPEAKERS > 0:
                kwargs["max_speakers"] = config.PYANNOTE_MAX_SPEAKERS

        diarization = pipeline(str(audio_path), **kwargs)
        raw_segments = [
            DiarizationSegment(
                speaker=str(label),
                start_seconds=float(turn.start),
                end_seconds=float(turn.end),
            )
            for turn, _, label in diarization.itertracks(yield_label=True)
            if float(turn.end) > float(turn.start)
        ]
        return _canonicalize_diarization_segments(raw_segments)
    except Exception as exc:
        logger.warning("pyannote diarization failed; using heuristic speaker assignment: %s", exc)
        return []


def _canonicalize_diarization_segments(segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
    speaker_map: dict[str, str] = {}
    canonical: list[DiarizationSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start_seconds, item.end_seconds)):
        if segment.speaker not in speaker_map:
            speaker_map[segment.speaker] = f"speaker_{len(speaker_map) + 1}"
        canonical.append(
            DiarizationSegment(
                speaker=speaker_map[segment.speaker],
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
            )
        )
    return canonical


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _sort_speaker_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    return sorted(turns, key=lambda turn: (turn.start_seconds, turn.end_seconds, turn.speaker))


def _split_speaker_turns_by_dialogue_units(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    split_turns: list[SpeakerTurn] = []
    for turn in turns:
        units = _split_dialogue_units(turn.text)
        if len(units) <= 1:
            split_turns.append(turn)
            continue

        duration = max(0.0, turn.end_seconds - turn.start_seconds)
        total_chars = sum(max(1, len(unit)) for unit in units)
        cursor = turn.start_seconds
        for idx, unit in enumerate(units):
            if idx == len(units) - 1:
                end = turn.end_seconds
            else:
                ratio = max(1, len(unit)) / total_chars
                end = min(turn.end_seconds, cursor + duration * ratio)
            split_turns.append(
                SpeakerTurn(
                    speaker=turn.speaker,
                    start_seconds=cursor,
                    end_seconds=end,
                    text=unit,
                )
            )
            cursor = end
    return _sort_speaker_turns(split_turns)


def _merge_speaker_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    merged: list[SpeakerTurn] = []
    for turn in _sort_speaker_turns(turns):
        gap = max(0.0, turn.start_seconds - merged[-1].end_seconds) if merged else 0.0
        can_merge = (
            merged
            and merged[-1].speaker == turn.speaker
            and gap <= config.SPEAKER_MAX_TURN_MERGE_SEC
            and not str(merged[-1].text).strip().endswith((".", "?", "!"))
        )
        if can_merge:
            merged[-1].end_seconds = max(merged[-1].end_seconds, turn.end_seconds)
            merged[-1].text = _normalize_text(f"{merged[-1].text} {turn.text}")
            continue
        merged.append(turn)
    return merged


def _is_short_uncertain_turn(turn: SpeakerTurn) -> bool:
    text = _normalize_text(turn.text)
    if not text:
        return True
    if _is_acknowledgement(text):
        return True
    customer_score, agent_score = _speaker_role_scores(text)
    if customer_score or agent_score or text.endswith("?"):
        return False
    duration = max(0.0, turn.end_seconds - turn.start_seconds)
    if duration <= config.SPEAKER_SHORT_TURN_MAX_SEC and len(text) <= config.SPEAKER_SHORT_TURN_MAX_CHARS:
        return True
    return len(text) <= 3 and text in {"네", "예", "아", "음", "어"}


def _is_short_information_fragment(turn: SpeakerTurn) -> bool:
    text = _normalize_text(turn.text)
    if not text or text.endswith("?") or _is_acknowledgement(text):
        return False
    customer_score, agent_score = _speaker_role_scores(text)
    if customer_score or agent_score:
        return False
    duration = max(0.0, turn.end_seconds - turn.start_seconds)
    if duration > max(config.SPEAKER_SHORT_TURN_MAX_SEC, 1.2):
        return False
    compact = re.sub(r"[\s,.\-]", "", text)
    if len(compact) > config.SPEAKER_SHORT_TURN_MAX_CHARS:
        return False
    return bool(re.search(r"\d", compact) or re.search(r"(년|월|일|생|번|원|프로|퍼센트|%)", compact))


def _smooth_short_speaker_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    if len(turns) < 3:
        return turns

    smoothed = [
        SpeakerTurn(
            speaker=turn.speaker,
            start_seconds=turn.start_seconds,
            end_seconds=turn.end_seconds,
            text=turn.text,
        )
        for turn in turns
    ]
    for idx in range(1, len(smoothed) - 1):
        turn = smoothed[idx]
        previous_turn = smoothed[idx - 1]
        next_turn = smoothed[idx + 1]
        prev_gap = max(0.0, turn.start_seconds - previous_turn.end_seconds)
        next_gap = max(0.0, next_turn.start_seconds - turn.end_seconds)
        if (
            _is_acknowledgement(turn.text)
            and turn.speaker == previous_turn.speaker
            and previous_turn.text.endswith("?")
            and next_turn.speaker != turn.speaker
            and next_gap <= max(config.SPEAKER_SHORT_TURN_CONTEXT_GAP_SEC, 10.0)
        ):
            smoothed[idx] = SpeakerTurn(
                speaker=next_turn.speaker,
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                text=turn.text,
            )
            continue
        if (
            _is_short_information_fragment(turn)
            and turn.speaker != previous_turn.speaker
            and _is_short_information_fragment(previous_turn)
            and not previous_turn.text.endswith("?")
            and prev_gap <= max(config.SPEAKER_SHORT_TURN_CONTEXT_GAP_SEC, 1.2)
            and (next_turn.speaker != turn.speaker or next_gap > prev_gap)
        ):
            smoothed[idx] = SpeakerTurn(
                speaker=previous_turn.speaker,
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                text=turn.text,
            )
            continue
        if _is_short_information_fragment(turn):
            continue
        if _is_acknowledgement(turn.text) or previous_turn.text.endswith("?"):
            continue
        if not _is_short_uncertain_turn(turn):
            continue
        if previous_turn.speaker != next_turn.speaker or turn.speaker == previous_turn.speaker:
            continue
        if prev_gap > config.SPEAKER_SHORT_TURN_CONTEXT_GAP_SEC or next_gap > config.SPEAKER_SHORT_TURN_CONTEXT_GAP_SEC:
            continue
        smoothed[idx] = SpeakerTurn(
            speaker=previous_turn.speaker,
            start_seconds=turn.start_seconds,
            end_seconds=turn.end_seconds,
            text=turn.text,
        )
    return _merge_speaker_turns(smoothed)


def _extract_transcript_segments(segments) -> list[TranscriptSegment]:
    extracted: list[TranscriptSegment] = []
    for segment in segments:
        text = _normalize_text(getattr(segment, "text", ""))
        if not text:
            continue
        start_seconds = float(getattr(segment, "start", 0.0) or 0.0)
        end_seconds = float(getattr(segment, "end", 0.0) or 0.0)
        if _is_repeated_low_signal_hallucination(text, end_seconds - start_seconds):
            logger.info(
                "dropping repeated low-signal STT segment start=%.3f end=%.3f text=%s",
                start_seconds,
                end_seconds,
                text[:120],
            )
            continue
        extracted.append(
            TranscriptSegment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
            )
        )
    return extracted


def _extract_transcript_words(segments) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            text = _normalize_text(getattr(word, "word", ""))
            if not text:
                continue
            start = float(getattr(word, "start", 0.0) or 0.0)
            end = float(getattr(word, "end", 0.0) or 0.0)
            if end <= start:
                continue
            words.append(TranscriptWord(start_seconds=start, end_seconds=end, word=text))
    return words


def _wav_channel_count(audio_path: Path) -> int:
    try:
        with wave.open(str(audio_path), "rb") as wav:
            return wav.getnchannels()
    except (wave.Error, OSError):
        return 0


def _split_stereo_wav_channels(audio_path: Path, output_dir: Path) -> list[Path]:
    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frame_rate = source.getframerate()
        params = source.getparams()
        frames = source.readframes(source.getnframes())

    if channels != STEREO_CHANNEL_COUNT:
        return []
    if sample_width <= 0:
        return []

    frame_width = channels * sample_width
    output_paths: list[Path] = []
    for channel_index in range(channels):
        channel_bytes = bytearray()
        start = channel_index * sample_width
        for offset in range(start, len(frames), frame_width):
            channel_bytes.extend(frames[offset : offset + sample_width])

        target = output_dir / f"{audio_path.stem}.ch{channel_index + 1}.wav"
        with wave.open(str(target), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(sample_width)
            out.setframerate(frame_rate)
            out.setcomptype(params.comptype, params.compname)
            out.writeframes(bytes(channel_bytes))
        output_paths.append(target)
    return output_paths


def _transcribe_stereo_channels(whisper: WhisperModel, audio_path: Path) -> tuple[list[SpeakerTurn], str | None, float | None] | None:
    if not config.STEREO_CHANNEL_SPEAKERS_ENABLED:
        return None
    if _wav_channel_count(audio_path) != STEREO_CHANNEL_COUNT:
        return None

    speaker_turns: list[SpeakerTurn] = []
    detected_language: str | None = None
    duration_seconds: float | None = None
    with tempfile.TemporaryDirectory(prefix="xcn-stereo-") as temp_dir_name:
        channel_paths = _split_stereo_wav_channels(audio_path, Path(temp_dir_name))
        if len(channel_paths) != STEREO_CHANNEL_COUNT:
            return None

        for channel_index, channel_path in enumerate(channel_paths, start=1):
            segments, info = whisper.transcribe(
                str(channel_path),
                language=config.WHISPER_LANGUAGE or None,
                vad_filter=True,
                word_timestamps=False,
            )
            channel_segments = _extract_transcript_segments(list(segments))
            if not channel_segments:
                continue
            detected_language = detected_language or getattr(info, "language", None)
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            duration_seconds = max(duration_seconds or 0.0, duration)
            conversation_segments = _trim_system_segments(channel_segments)
            split_segments = _split_segments_for_speaker_assignment(conversation_segments)
            turns = _drop_system_turns(split_segments)
            for turn in turns:
                speaker_turns.append(
                    SpeakerTurn(
                        speaker=f"speaker_{channel_index}",
                        start_seconds=turn.start_seconds,
                        end_seconds=turn.end_seconds,
                        text=turn.text,
                    )
                )

    if not speaker_turns:
        return None
    speaker_turns.sort(key=lambda turn: (turn.start_seconds, turn.end_seconds, turn.speaker))
    logger.info("using stereo channel speaker assignment for %s", audio_path)
    return _smooth_short_speaker_turns(_merge_speaker_turns(speaker_turns)), detected_language, duration_seconds


def _trim_system_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []
    start_index = 0
    for idx, segment in enumerate(segments):
        if _contains_conversation_marker(segment.text):
            start_index = idx
            break
        if not _contains_ivr_marker(segment.text):
            start_index = idx
            break
    trimmed = segments[start_index:]
    return trimmed or segments


def _drop_system_turns(turns: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not turns:
        return []

    filtered: list[TranscriptSegment] = []
    conversation_started = False
    for turn in turns:
        text = _normalize_text(turn.text)
        if not text:
            continue
        if _is_repeated_low_signal_hallucination(text, turn.end_seconds - turn.start_seconds):
            continue
        if not conversation_started:
            if _is_low_value_sentence(text):
                continue
            if _contains_conversation_marker(text) and not _contains_ivr_marker(text):
                conversation_started = True
            elif _contains_ivr_marker(text):
                continue
        if _contains_ivr_marker(text) and not _contains_conversation_marker(text):
            continue
        filtered.append(turn)
    if filtered:
        return filtered
    if len(turns) >= 4 and all(
        _is_low_value_sentence(_normalize_text(turn.text))
        or _is_repeated_low_signal_hallucination(
            _normalize_text(turn.text),
            turn.end_seconds - turn.start_seconds,
        )
        for turn in turns
    ):
        return []
    return turns


def _build_turns(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []
    turns: list[TranscriptSegment] = [TranscriptSegment(**asdict(segments[0]))]
    for segment in segments[1:]:
        last = turns[-1]
        gap = max(0.0, segment.start_seconds - last.end_seconds)
        same_turn = (
            gap <= config.SPEAKER_MAX_TURN_MERGE_SEC
            and len(last.text) + len(segment.text) <= 140
            and not last.text.endswith("?")
            and not _is_acknowledgement(segment.text)
        )
        if same_turn:
            last.end_seconds = segment.end_seconds
            last.text = _normalize_text(f"{last.text} {segment.text}")
            continue
        turns.append(TranscriptSegment(**asdict(segment)))
    return turns


def _assign_speakers(turns: list[TranscriptSegment]) -> list[SpeakerTurn]:
    if not turns:
        return []
    assigned: list[SpeakerTurn] = []
    previous_turn: TranscriptSegment | None = None
    previous_speaker = "speaker_2"
    for turn in turns:
        customer_score, agent_score = _speaker_role_scores(turn.text)
        if agent_score >= customer_score + 1:
            speaker = "speaker_2"
        elif customer_score >= agent_score + 1:
            speaker = "speaker_1"
        elif previous_turn is not None:
            gap = max(0.0, turn.start_seconds - previous_turn.end_seconds)
            should_switch = (
                gap >= config.SPEAKER_PAUSE_THRESHOLD_SEC
                or previous_turn.text.endswith("?")
                or _is_acknowledgement(turn.text)
            )
            if should_switch:
                speaker = "speaker_1" if previous_speaker == "speaker_2" else "speaker_2"
            else:
                speaker = previous_speaker
        else:
            speaker = "speaker_2"

        if _is_acknowledgement(turn.text) and previous_turn is not None:
            speaker = "speaker_1" if previous_speaker == "speaker_2" else "speaker_2"

        assigned.append(
            SpeakerTurn(
                speaker=speaker,
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                text=turn.text,
            )
        )
        previous_turn = turn
        previous_speaker = speaker

    return _smooth_short_speaker_turns(_merge_speaker_turns(assigned))


def _assign_speakers_with_diarization(
    turns: list[TranscriptSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[SpeakerTurn]:
    if not turns or not diarization_segments:
        return []

    assigned: list[SpeakerTurn] = []
    previous_speaker = diarization_segments[0].speaker
    for turn in turns:
        overlap_by_speaker: dict[str, float] = {}
        for diar_segment in diarization_segments:
            if diar_segment.end_seconds < turn.start_seconds:
                continue
            if diar_segment.start_seconds > turn.end_seconds:
                break
            overlap = _overlap_seconds(
                turn.start_seconds,
                turn.end_seconds,
                diar_segment.start_seconds,
                diar_segment.end_seconds,
            )
            if overlap > 0:
                overlap_by_speaker[diar_segment.speaker] = overlap_by_speaker.get(diar_segment.speaker, 0.0) + overlap

        if overlap_by_speaker:
            speaker = max(overlap_by_speaker.items(), key=lambda item: item[1])[0]
        else:
            midpoint = (turn.start_seconds + turn.end_seconds) / 2
            nearest = min(
                diarization_segments,
                key=lambda item: min(abs(midpoint - item.start_seconds), abs(midpoint - item.end_seconds)),
            )
            speaker = nearest.speaker if min(abs(midpoint - nearest.start_seconds), abs(midpoint - nearest.end_seconds)) <= 1.0 else previous_speaker

        assigned.append(
            SpeakerTurn(
                speaker=speaker,
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                text=turn.text,
            )
        )
        previous_speaker = speaker

    return _smooth_short_speaker_turns(_merge_speaker_turns(assigned))


def _speaker_for_time_range(
    start_seconds: float,
    end_seconds: float,
    diarization_segments: list[DiarizationSegment],
    fallback_speaker: str,
) -> str:
    overlap_by_speaker: dict[str, float] = {}
    for diar_segment in diarization_segments:
        if diar_segment.end_seconds < start_seconds:
            continue
        if diar_segment.start_seconds > end_seconds:
            break
        overlap = _overlap_seconds(
            start_seconds,
            end_seconds,
            diar_segment.start_seconds,
            diar_segment.end_seconds,
        )
        if overlap > 0:
            overlap_by_speaker[diar_segment.speaker] = overlap_by_speaker.get(diar_segment.speaker, 0.0) + overlap
    if overlap_by_speaker:
        return max(overlap_by_speaker.items(), key=lambda item: item[1])[0]
    return fallback_speaker


def _assign_speakers_from_words(
    words: list[TranscriptWord],
    diarization_segments: list[DiarizationSegment],
) -> list[SpeakerTurn]:
    if not words or not diarization_segments:
        return []

    turns: list[SpeakerTurn] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    current_start = 0.0
    current_end = 0.0
    previous_speaker = diarization_segments[0].speaker

    def flush() -> None:
        nonlocal current_speaker, current_words, current_start, current_end
        text = _normalize_text(" ".join(current_words))
        if _is_repeated_low_signal_hallucination(text, current_end - current_start):
            current_speaker = None
            current_words = []
            current_start = 0.0
            current_end = 0.0
            return
        if current_speaker and text:
            turns.append(
                SpeakerTurn(
                    speaker=current_speaker,
                    start_seconds=current_start,
                    end_seconds=current_end,
                    text=text,
                )
            )
        current_speaker = None
        current_words = []
        current_start = 0.0
        current_end = 0.0

    for word in words:
        speaker = _speaker_for_time_range(
            word.start_seconds,
            word.end_seconds,
            diarization_segments,
            previous_speaker,
        )
        gap = max(0.0, word.start_seconds - current_end) if current_words else 0.0
        should_break = (
            current_speaker is not None
            and (
                speaker != current_speaker
                or gap >= config.SPEAKER_PAUSE_THRESHOLD_SEC
                or (current_words and current_words[-1].endswith((".", "?", "!")))
            )
        )
        if should_break:
            flush()

        if current_speaker is None:
            current_speaker = speaker
            current_start = word.start_seconds
        current_words.append(word.word)
        current_end = word.end_seconds
        previous_speaker = speaker

    flush()
    if not turns:
        return []

    transcript_segments = [
        TranscriptSegment(
            start_seconds=turn.start_seconds,
            end_seconds=turn.end_seconds,
            text=turn.text,
        )
        for turn in turns
    ]
    split_segments = _split_segments_for_speaker_assignment(transcript_segments)
    filtered_segments = _drop_system_turns(split_segments)
    reassigned = _assign_speakers_with_diarization(filtered_segments, diarization_segments)
    return _smooth_short_speaker_turns(_merge_speaker_turns(reassigned))


def _is_question_answer_call(sentences: list[str]) -> bool:
    if not sentences:
        return False
    question_hits = 0
    answer_hits = 0
    question_patterns = ["?", "문의", "어떻게", "가능한가", "되나요", "인가요", "할까요"]
    answer_patterns = ["가능", "불가", "안됩니다", "됩니다", "맞습니다", "아닙니다", "안내", "확인", "처리", "접수"]
    for sentence in sentences:
        if any(pattern in sentence for pattern in question_patterns):
            question_hits += 1
        if any(pattern in sentence for pattern in answer_patterns):
            answer_hits += 1
    return question_hits >= 1 and answer_hits >= 1


def _fallback_general_summary(text: str) -> str:
    sentences = _split_meaningful_sentences(text)
    if not sentences:
        return _normalize_text(text)[:240].strip()

    scored: list[tuple[int, str]] = []
    topical_patterns = [
        "주문", "배송", "교환", "환불", "결제", "상품", "사이즈", "색상", "재고",
        "주소", "문의", "요청", "가능", "안내", "확인", "처리", "예약", "취소",
        "서버", "워런티", "연장", "유지보수", "계약", "장비", "연구소", "구매",
        "사업", "진행", "고장", "교체", "트래픽", "제외", "추가", "결정",
    ]
    for sentence in sentences:
        score = 0
        if 12 <= len(sentence) <= 120:
            score += 2
        score += sum(2 for pattern in topical_patterns if pattern in sentence)
        if "?" in sentence:
            score += 1
        if sentence.endswith(("습니다.", "입니다.", "돼요.", "되나요?", "가능합니다.")):
            score += 1
        scored.append((score, sentence))

    scored.sort(key=lambda item: (-item[0], sentences.index(item[1])))
    selected = [sentence for _, sentence in scored[:2]]
    ordered = [sentence for sentence in sentences if sentence in selected]
    if not ordered:
        ordered = sentences[:2]
    return " ".join(ordered).strip()


def _summary_quality_is_bad(summary_text: str) -> bool:
    cleaned = _normalize_text(summary_text)
    if not cleaned:
        return True
    if any(marker in cleaned for marker in ["소통과 배려로", "통화 연결", "직원 연결", "민생회복소비"]):
        return True
    if _starts_like_raw_utterance(cleaned):
        return True
    if cleaned.endswith("여보세요") or cleaned.endswith("네") or cleaned.endswith("예"):
        return True
    if len(cleaned) > 280:
        return True
    duplicate_hits = len(re.findall(r"(\b\S+\b)(?:\s+\1){1,}", cleaned))
    if duplicate_hits > 0:
        return True
    return False


def _speaker_role_scores(text: str) -> tuple[int, int]:
    cleaned = _normalize_text(text)
    customer_patterns = [
        "제가", "저는", "저희가", "하려고", "하고 싶은데", "문의", "요청",
        "어떻게", "왜", "언제", "얼마", "되나요", "인가요", "상관없나요",
        "알겠습니다", "그래요", "그러면", "궁금", "모르겠", "아니,",
        "안되", "안 돼", "못 찾", "부탁",
    ]
    agent_patterns = [
        "고객님", "안내", "확인해", "확인해서", "확인 감사합니다", "조회", "처리",
        "접수", "도와드리", "말씀드리", "말씀 부탁", "가능합니다", "가능하세요",
        "불가", "문자", "서류", "ARS", "연결해드리", "입력 부탁", "정보 보호",
        "주민번호", "비밀번호", "기다려주셔서", "로그인하시면", "보내드릴",
        "보내드려볼까요", "무엇을 도와", "도와드릴까요", "본인 되십니까",
        "하시면", "가능하시고", "필요 없으세요", "보실 수", "경로 보내드릴",
        "연락 부탁", "선택하시면", "되세요", "제가 확인해",
        "잠시만 기다려", "어떤 등록", "아니에요.", "가능하십니다", "좋은 하루",
    ]
    customer_score = sum(1 for pattern in customer_patterns if pattern in cleaned)
    agent_score = sum(1 for pattern in agent_patterns if pattern in cleaned)
    customer_score += 2 if _has_customer_phrase(cleaned) else 0
    agent_score += 2 if _has_agent_phrase(cleaned) else 0
    return customer_score, agent_score


def _infer_speaker_roles(turns: list[SpeakerTurn]) -> dict[str, str]:
    scores: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for turn in turns:
        if _contains_ivr_marker(turn.text) and not _contains_conversation_marker(turn.text):
            continue
        order.append(turn.speaker) if turn.speaker not in order else None
        role_score = scores.setdefault(turn.speaker, {"customer": 0, "agent": 0})
        customer_score, agent_score = _speaker_role_scores(turn.text)
        role_score["customer"] += customer_score
        role_score["agent"] += agent_score
    roles: dict[str, str] = {}
    for idx, speaker in enumerate(order):
        score = scores.get(speaker, {"customer": 0, "agent": 0})
        if score["agent"] > score["customer"]:
            roles[speaker] = "agent"
        elif score["customer"] > score["agent"]:
            roles[speaker] = "customer"
        else:
            roles[speaker] = "customer" if idx == 0 else "agent"
    return roles


def _normalize_speaker_labels_by_role(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    if not turns:
        return []

    turns = [
        turn for turn in turns
        if not (_contains_ivr_marker(turn.text) and not _contains_conversation_marker(turn.text))
    ]
    if not turns:
        return []

    roles = _infer_speaker_roles(turns)
    mapping: dict[str, str] = {}
    next_index = 3
    for turn in turns:
        if turn.speaker in mapping:
            continue
        role = roles.get(turn.speaker)
        if role == "customer":
            mapping[turn.speaker] = "speaker_1"
        elif role == "agent":
            mapping[turn.speaker] = "speaker_2"
        else:
            mapping[turn.speaker] = f"speaker_{next_index}"
            next_index += 1

    normalized = [
        SpeakerTurn(
            speaker=mapping.get(turn.speaker, turn.speaker),
            start_seconds=turn.start_seconds,
            end_seconds=turn.end_seconds,
            text=turn.text,
        )
        for turn in turns
    ]
    return _smooth_short_speaker_turns(_merge_speaker_turns(normalized))


def _refine_speaker_labels_by_text_role(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    refined: list[SpeakerTurn] = []
    for turn in turns:
        units = _split_dialogue_units(turn.text)
        if len(units) <= 1:
            units = [turn.text]
        duration = max(0.0, turn.end_seconds - turn.start_seconds)
        total_chars = sum(max(1, len(unit)) for unit in units)
        cursor = turn.start_seconds
        for idx, unit in enumerate(units):
            if idx == len(units) - 1:
                end = turn.end_seconds
            else:
                end = min(turn.end_seconds, cursor + duration * (max(1, len(unit)) / total_chars))
            customer_score, agent_score = _speaker_role_scores(unit)
            speaker = turn.speaker
            unit_duration = max(0.0, end - cursor)
            can_refine = (
                len(_normalize_text(unit)) >= config.SPEAKER_ROLE_REFINE_MIN_CHARS
                and unit_duration >= config.SPEAKER_ROLE_REFINE_MIN_SEC
                and not _is_acknowledgement(unit)
            )
            if can_refine and agent_score >= customer_score + config.SPEAKER_ROLE_REFINE_SCORE_MARGIN:
                speaker = "speaker_2"
            elif can_refine and customer_score >= agent_score + config.SPEAKER_ROLE_REFINE_SCORE_MARGIN:
                speaker = "speaker_1"
            refined.append(
                SpeakerTurn(
                    speaker=speaker,
                    start_seconds=cursor,
                    end_seconds=end,
                    text=unit,
                )
            )
            cursor = end
    return _smooth_short_speaker_turns(_merge_speaker_turns(refined))


def _get_role_summary(
    speaker_summaries: list[dict[str, str]],
    speaker_roles: dict[str, str],
    role: str,
) -> str:
    parts = [
        str(item.get("summary_text") or "").strip()
        for item in speaker_summaries
        if speaker_roles.get(str(item.get("speaker") or "")) == role and item.get("summary_text")
    ]
    return " ".join(parts).strip()


def _pick_sentences(text: str, patterns: list[str], *, limit: int = 2) -> str:
    sentences = _split_meaningful_sentences(text)
    selected = [sentence for sentence in sentences if any(pattern in sentence for pattern in patterns)]
    if not selected:
        selected = sentences[:limit]
    return " ".join(selected[:limit]).strip()


def _derive_action_items(overall_summary: str, guidance_summary: str) -> str:
    action_patterns = ["확인", "접수", "안내", "문자", "서류", "재연락", "방문", "처리", "후", "추가", "진행"]
    action_text = _pick_sentences(f"{guidance_summary} {overall_summary}", action_patterns, limit=2)
    return action_text or "추가 조치 사항이 명확히 확인되지 않았다."


def _derive_call_outcome(overall_summary: str, guidance_summary: str, inquiry_summary: str) -> str:
    outcome_patterns = ["처리", "완료", "접수", "가능", "불가", "안내", "예정", "확인", "보류"]
    outcome_text = _pick_sentences(f"{overall_summary} {guidance_summary} {inquiry_summary}", outcome_patterns, limit=2)
    return outcome_text or "통화 결과가 명확히 확인되지 않았다."


def _format_structured_summary(
    inquiry_summary: str,
    guidance_summary: str,
    action_items: str,
    call_outcome: str,
) -> str:
    sections = [
        f"문의내용: {inquiry_summary or '문의 내용이 명확히 확인되지 않았다.'}",
        f"안내내용: {guidance_summary or '안내 내용이 명확히 확인되지 않았다.'}",
        f"처리결과: {call_outcome or '처리 결과가 명확히 확인되지 않았다.'}",
        f"후속조치: {action_items or '후속 조치가 명확히 확인되지 않았다.'}",
    ]
    return "\n".join(sections)


def _build_structured_call_summary(
    turns: list[SpeakerTurn],
    speaker_summaries: list[dict[str, str]],
    overall_narrative: str,
) -> tuple[str, str, str, str, str]:
    speaker_roles = _infer_speaker_roles(turns)
    inquiry_summary = _get_role_summary(speaker_summaries, speaker_roles, "customer")
    guidance_summary = _get_role_summary(speaker_summaries, speaker_roles, "agent")
    if not inquiry_summary and speaker_summaries:
        inquiry_summary = str(speaker_summaries[0].get("summary_text") or "").strip()
    if not guidance_summary and len(speaker_summaries) > 1:
        guidance_summary = " ".join(
            str(item.get("summary_text") or "").strip() for item in speaker_summaries[1:] if item.get("summary_text")
        ).strip()
    action_items = _derive_action_items(overall_narrative, guidance_summary)
    call_outcome = _derive_call_outcome(overall_narrative, guidance_summary, inquiry_summary)
    formatted = _format_structured_summary(inquiry_summary, guidance_summary, action_items, call_outcome)
    return formatted, inquiry_summary, guidance_summary, action_items, call_outcome


def _build_summary_bundle(
    turns: list[SpeakerTurn],
    transcript_text: str,
    *,
    backend: str,
    tokenizer=None,
    summary_model=None,
) -> dict[str, str]:
    speaker_summaries = _build_speaker_summaries(
        turns,
        backend=backend,
        tokenizer=tokenizer,
        summary_model=summary_model,
    )
    turn_transcript = _format_turn_transcript(turns)
    if backend == "sllm" and turn_transcript:
        (
            overall_summary_text,
            inquiry_summary_text,
            guidance_summary_text,
            action_items_text,
            call_outcome_text,
        ) = _build_sllm_structured_summary(turns, " ".join(turn.text for turn in turns).strip() or transcript_text)
        if not guidance_summary_text or not action_items_text or not call_outcome_text:
            (
                _fallback_overall,
                fallback_inquiry,
                fallback_guidance,
                fallback_action,
                fallback_outcome,
            ) = _build_structured_call_summary(turns, speaker_summaries, overall_summary_text)
            inquiry_summary_text = inquiry_summary_text or fallback_inquiry
            guidance_summary_text = guidance_summary_text or fallback_guidance
            action_items_text = action_items_text or fallback_action
            call_outcome_text = call_outcome_text or fallback_outcome
            parsed = _parse_labeled_summary(overall_summary_text)
            if not parsed.get("통화 목적"):
                parsed["통화 목적"] = inquiry_summary_text
            if not parsed.get("상담사 안내"):
                parsed["상담사 안내"] = guidance_summary_text
            if not parsed.get("처리 결과"):
                parsed["처리 결과"] = call_outcome_text
            if not parsed.get("후속 조치"):
                parsed["후속 조치"] = action_items_text
            parsed = _sanitize_structured_summary(parsed, " ".join(turn.text for turn in turns).strip() or transcript_text)
            action_items_text = parsed.get("후속 조치", action_items_text)
            overall_summary_text = _format_labeled_summary(parsed)
    else:
        overall_narrative = _build_overall_summary(
            " ".join(turn.text for turn in turns).strip() or transcript_text,
            speaker_summaries,
            backend=backend,
            tokenizer=tokenizer,
            summary_model=summary_model,
        )
        (
            overall_summary_text,
            inquiry_summary_text,
            guidance_summary_text,
            action_items_text,
            call_outcome_text,
        ) = _build_structured_call_summary(turns, speaker_summaries, overall_narrative)
    try:
        conversational_summary_text = _build_conversational_summary(
            turns,
            " ".join(turn.text for turn in turns).strip() or transcript_text,
        )
    except Exception as exc:
        logger.warning("conversational summary failed; using structured summary fallback: %s", exc)
        conversational_summary_text = _fallback_general_summary(overall_summary_text)
    return {
        "summary_text": overall_summary_text,
        "conversational_summary_text": conversational_summary_text,
        "_speaker_summaries": speaker_summaries,
    }


def _summarize_text_with_sllm(text: str, *, mode: str, speaker: str | None = None) -> str:
    source_text = _strip_ivr_preamble(text)
    prompt = _summary_prompt(source_text, mode=mode, speaker=speaker)
    return _summarize_prompt_with_sllm(prompt, fallback_text=source_text)


def _summarize_text_with_backend(text: str, *, backend: str, tokenizer=None, summary_model=None, mode: str = "overall", speaker: str | None = None) -> str:
    if backend == "sllm":
        return _summarize_text_with_sllm(text, mode=mode, speaker=speaker)
    raise RuntimeError(f"unsupported summary backend: {backend}")


def _build_speaker_summaries(turns: list[SpeakerTurn], *, backend: str, tokenizer=None, summary_model=None) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    grouped_turns: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        grouped.setdefault(turn.speaker, []).append(turn.text)
        grouped_turns.setdefault(turn.speaker, []).append(turn)

    summaries: list[dict[str, str]] = []
    for speaker in sorted(grouped.keys()):
        if backend == "sllm":
            speaker_text = _fit_turn_transcript_to_budget(grouped_turns.get(speaker, []), config.SLLM_MAX_PROMPT_CHARS)
        else:
            speaker_text = " ".join(grouped[speaker]).strip()
        if not speaker_text:
            continue
        summary_text = _summarize_text_with_backend(
            speaker_text,
            backend=backend,
            tokenizer=tokenizer,
            summary_model=summary_model,
            mode="speaker",
            speaker=speaker,
        )
        summaries.append({"speaker": speaker, "summary_text": summary_text})
    return summaries


def _build_overall_summary(
    transcript_text: str,
    speaker_summaries: list[dict[str, str]],
    *,
    backend: str,
    tokenizer=None,
    summary_model=None,
) -> str:
    if speaker_summaries:
        joined = " ".join(
            f"{item['speaker']} 발화 요약: {item['summary_text']}"
            for item in speaker_summaries
            if item.get("summary_text")
        ).strip()
        if joined:
            overall = _summarize_text_with_backend(
                joined,
                backend=backend,
                tokenizer=tokenizer,
                summary_model=summary_model,
                mode="overall",
            )
            if not _summary_quality_is_bad(overall):
                return overall
    return _summarize_text_with_backend(
        transcript_text,
        backend=backend,
        tokenizer=tokenizer,
        summary_model=summary_model,
        mode="overall",
    )


def transcribe_and_summarize(audio_path: Path) -> PipelineResult:
    start = time.perf_counter()
    whisper = get_whisper_model()

    stereo_result = _transcribe_stereo_channels(whisper, audio_path)
    if stereo_result:
        speaker_turns, detected_language, duration_seconds = stereo_result
        transcript_text = " ".join(turn.text for turn in speaker_turns).strip()
        info = None
    else:
        segments, info = whisper.transcribe(
            str(audio_path),
            language=config.WHISPER_LANGUAGE or None,
            vad_filter=True,
            word_timestamps=True,
        )
        whisper_segments = list(segments)
        transcript_segments = _extract_transcript_segments(whisper_segments)
        transcript_words = _extract_transcript_words(whisper_segments)
        transcript_parts = [segment.text for segment in transcript_segments]
        transcript_text = " ".join(transcript_parts).strip()
        if not transcript_text:
            raise ValueError("transcript is empty")

        conversation_segments = _trim_system_segments(transcript_segments)
        split_segments = _split_segments_for_speaker_assignment(conversation_segments)
        turns = _drop_system_turns(split_segments)
        diarization_segments = _run_diarization(audio_path)
        if diarization_segments and transcript_words:
            speaker_turns = _assign_speakers_from_words(transcript_words, diarization_segments)
            if not speaker_turns:
                speaker_turns = _assign_speakers_with_diarization(turns, diarization_segments)
        elif diarization_segments:
            speaker_turns = _assign_speakers_with_diarization(turns, diarization_segments)
        else:
            speaker_turns = _assign_speakers(turns)
        speaker_turns = _normalize_speaker_labels_by_role(speaker_turns)
        speaker_turns = _refine_speaker_labels_by_text_role(speaker_turns)
        detected_language = getattr(info, "language", None)
        duration_seconds = float(getattr(info, "duration", 0.0) or 0.0)
    summary_backend = "stt_only"
    summary_model_name = "none"
    speaker_summaries: list[dict[str, str]] = []
    summary_text = ""
    conversational_summary_text = ""
    speaker_turns = _split_speaker_turns_by_dialogue_units(_sort_speaker_turns(speaker_turns))
    speaker_segments = [asdict(turn) for turn in speaker_turns]
    processing_ms = int((time.perf_counter() - start) * 1000)
    return PipelineResult(
        transcript_text=transcript_text,
        summary_backend=summary_backend,
        summary_model=summary_model_name,
        summary_text=summary_text,
        conversational_summary_text=conversational_summary_text,
        speaker_summaries=speaker_summaries,
        speaker_segments=speaker_segments,
        detected_language=detected_language,
        duration_seconds=duration_seconds,
        processing_ms=processing_ms,
    )
