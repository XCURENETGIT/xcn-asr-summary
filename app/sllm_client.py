from __future__ import annotations

import logging
import time

import httpx

from . import config

logger = logging.getLogger(__name__)


def is_sllm_configured() -> bool:
    return bool(config.SLLM_BASE_URL and config.SLLM_MODEL)


def is_sllm_ready(timeout_sec: float = 3.0) -> bool:
    if not is_sllm_configured():
        return False
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.get(f"{config.SLLM_BASE_URL}/v1/models")
            return response.status_code < 500
    except httpx.HTTPError:
        return False


def wait_for_sllm(timeout_sec: int | None = None) -> None:
    if not is_sllm_configured():
        return
    timeout = config.SLLM_STARTUP_WAIT_SEC if timeout_sec is None else timeout_sec
    deadline = time.monotonic() + max(0, timeout)
    last_error = "not ready"
    while True:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{config.SLLM_BASE_URL}/v1/models")
                if response.status_code < 500:
                    logger.info("sllm ready: %s", config.SLLM_BASE_URL)
                    return
                last_error = f"status={response.status_code} body={response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = repr(exc)

        if time.monotonic() >= deadline:
            raise RuntimeError(f"SLLM is not ready: {config.SLLM_BASE_URL} ({last_error})")
        logger.info("waiting for sllm: %s (%s)", config.SLLM_BASE_URL, last_error)
        time.sleep(config.SLLM_CONNECT_RETRY_DELAY_SEC)


def request_summary(prompt: str) -> str:
    if not is_sllm_configured():
        raise RuntimeError("SLLM is not configured")

    headers = {
        "Authorization": f"Bearer {config.SLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{config.SLLM_BASE_URL}{config.SLLM_REQUEST_PATH}"
    payload = {
        "model": config.SLLM_MODEL,
        "temperature": config.SLLM_TEMPERATURE,
        "top_p": config.SLLM_TOP_P,
        "max_tokens": config.SLLM_MAX_TOKENS,
    }
    if config.SLLM_USE_CHAT_ENDPOINT:
        payload["messages"] = [
            {
                "role": "system",
                "content": "당신은 한국어 통화 요약 시스템이다. 핵심 사실만 간결하게 요약하고 군더더기 표현은 제거한다.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    else:
        payload["prompt"] = prompt

    retries = max(1, config.SLLM_CONNECT_RETRIES)
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=config.SLLM_TIMEOUT_SEC) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.is_error:
                    body = response.text[:1000]
                    logger.warning(
                        "sllm request failed status=%s prompt_chars=%s body=%s",
                        response.status_code,
                        len(prompt),
                        body,
                    )
                    response.raise_for_status()
                data = response.json()
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if attempt >= retries:
                raise
            logger.info(
                "sllm connection not ready attempt=%s/%s url=%s error=%s",
                attempt,
                retries,
                url,
                exc,
            )
            time.sleep(config.SLLM_CONNECT_RETRY_DELAY_SEC)

    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        logger.info(
            "sllm usage model=%s prompt_chars=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            config.SLLM_MODEL,
            len(prompt),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
    else:
        logger.info("sllm usage model=%s prompt_chars=%s usage=missing", config.SLLM_MODEL, len(prompt))

    if config.SLLM_USE_CHAT_ENDPOINT:
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise RuntimeError("SLLM response did not include chat content")
        return str(content).strip()

    choices = data.get("choices") or []
    text = choices[0].get("text") if choices else None
    if not text:
        raise RuntimeError("SLLM response did not include completion text")
    return str(text).strip()
