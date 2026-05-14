# xcn-asr-summary 외부 연동 API 설계서

## 1. 문서 목적

외부 시스템이 xcn-asr-summary 서비스와 연동하기 위해 필요한 API, 인증, 요청/응답 형식, 오류 처리 기준을 정의한다.

## 2. 기본 정보

- 서비스명: xcn-asr-summary
- 기본 URL: `http://{server-host}:18080`
- API 문서: `http://{server-host}:18080/docs`
- Content-Type
  - 파일 업로드: `multipart/form-data`
  - JSON 요청: `application/json`
- 인증
  - 환경변수 `API_KEY`가 비어 있으면 인증 없이 호출 가능
  - `API_KEY`가 설정된 경우 요청 헤더에 `X-API-Key: {api_key}` 필요

## 3. 주요 연동 API

### 3.1 상태 확인

- Method: `GET`
- Path: `/health`
- 설명: 서비스, DB, 모델 설정 상태를 확인한다.

응답 예시:

```json
{
  "status": "healthy",
  "db_ready": true,
  "whisper_model": "large-v3-turbo",
  "summary_backend": "llamacpp",
  "summary_model": "mykor/A.X-4.0-Light-gguf:Q4_K_M",
  "sllm_configured": true
}
```

### 3.2 음성 파일 처리 요청

- Method: `POST`
- Path: `/calls/process`
- 설명: 음성 파일을 업로드하면 STT 변환, 통화 요약, 화자별 요약을 수행하고 DB에 저장한다.

요청 Form 필드:

| 항목 | 필수 | 설명 |
| --- | --- | --- |
| `file` | Y | 처리할 음성 파일 |
| `caller` | N | 발신번호 |
| `extension_number` | N | 내선번호 |
| `callee` | N | 수신번호 |
| `call_started_at` | N | 통화 시작 시각. ISO-8601 형식 |
| `call_ended_at` | N | 통화 종료 시각. ISO-8601 형식 |

파일명이 `발신번호_내선번호_년_월_일_시_분_초_월_일_시_분_초.wav` 형식이면 `caller`, `extension_number`, `call_started_at`, `call_ended_at`을 자동 추출한다. 요청 Form 값이 있으면 Form 값을 우선 사용한다.

요청 예시:

```bash
curl -X POST "http://localhost:18080/calls/process" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@sample.wav" \
  -F "caller=0212345678" \
  -F "callee=0298765432" \
  -F "call_started_at=2026-05-12T10:00:00"
```

응답 예시:

```json
{
  "id": 1,
  "processing_id": "6b8db2f4-4a6c-4f86-8db5-7d6b92f6450f",
  "input_type": "api_request",
  "audio_file_name": "sample.wav",
  "caller": "0212345678",
  "extension_number": "1001",
  "callee": "1001",
  "call_started_at": "2026-05-12T10:00:00",
  "call_ended_at": "2026-05-12T10:05:30",
  "processing_status": "completed",
  "detected_language": "ko",
  "audio_duration_seconds": 63.42,
  "full_transcript": "전체 전사 내용",
  "summary_model_backend": "llamacpp",
  "summary_generation_model": "mykor/A.X-4.0-Light-gguf:Q4_K_M",
  "structured_call_summary": "통화 목적: ...\n핵심 이슈: ...",
  "plain_call_summary": "고객이 문의한 내용과 상담사의 안내 내용을 문장형으로 요약한 결과",
  "speaker_summary_list": [
    {
      "speaker_name": "고객",
      "speaker_summary": "고객 문의 요약"
    }
  ],
  "processing_time_ms": 12450
}
```

### 3.3 처리 결과 목록 조회

- Method: `GET`
- Path: `/calls`
- 설명: 처리된 통화 결과 목록을 조회한다.

Query 파라미터:

| 항목 | 필수 | 설명 |
| --- | --- | --- |
| `limit` | N | 조회 개수. 기본 20, 최대 200 |
| `q` | N | 파일명, 처리 ID, 전사/요약 내용 검색 |
| `processing_status` | N | `completed`, `failed` |
| `input_type` | N | `api_request`, `voice_file` |
| `caller` | N | 발신번호 검색 |
| `extension_number` | N | 내선번호 검색 |
| `callee` | N | 수신번호 검색 |
| `date_from` | N | 결과 생성 시작 시각 |
| `date_to` | N | 결과 생성 종료 시각 |

요청 예시:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls?limit=20&input_type=voice_file&processing_status=completed"
```

### 3.4 처리 결과 상세 조회

- Method: `GET`
- Path: `/calls/{call_id}`
- 설명: 단일 처리 결과를 조회한다.

요청 예시:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls/1"
```

### 3.5 원본 음성 파일 조회

- Method: `GET`
- Path: `/calls/{call_id}/audio`
- 설명: 저장된 원본 음성 파일을 내려받는다. `SAVE_UPLOADS=false` 또는 파일 삭제 시 404가 반환된다.

### 3.6 화자 발화 구간 조회

- Method: `GET`
- Path: `/calls/{call_id}/segments`
- 설명: 화자별 발화 구간과 텍스트를 조회한다.

응답 예시:

```json
[
  {
    "speaker": "고객",
    "start_seconds": 0.0,
    "end_seconds": 4.2,
    "text": "문의 발화 내용"
  }
]
```

### 3.7 전사문 수정

- Method: `PATCH`
- Path: `/calls/{call_id}/transcript`
- 설명: 저장된 전체 전사문을 수정한다.

요청 예시:

```json
{
  "full_transcript": "수정된 전체 전사문"
}
```

### 3.8 처리 결과 삭제

- Method: `DELETE`
- Path: `/calls/{call_id}`
- 설명: 처리 결과를 삭제한다.

Query 파라미터:

| 항목 | 기본값 | 설명 |
| --- | --- | --- |
| `delete_audio` | `true` | 저장된 원본 음성 파일 삭제 여부 |
| `delete_training_clips` | `true` | 연결된 학습용 segment clip 삭제 여부 |

## 4. 폴더 기반 배치 연동

외부 시스템은 API 업로드 대신 `data/voice` 폴더에 `.wav` 파일을 넣을 수 있다.

수집 파일명 규칙:

```text
발신번호_내선번호_년_월_일_시_분_초_월_일_시_분_초.wav
```

예시:

```text
01012345678_1001_2026_05_12_10_00_00_05_12_10_05_30.wav
```

파싱 결과:

- `caller`: `01012345678`
- `extension_number`: `1001`
- `callee`: `1001`
- `call_started_at`: `2026-05-12T10:00:00`
- `call_ended_at`: `2026-05-12T10:05:30`

처리 흐름:

1. API 컨테이너가 `data/voice` 폴더를 주기적으로 확인한다.
2. 처리 대상 파일 옆에 `.lock` 파일을 생성한다.
3. STT/요약 처리를 수행한다.
4. 결과를 `call_summaries` DB에 저장한다.
5. 결과 JSON을 `data/translate/<원본파일명>.json`에 저장한다.
6. 원본 파일을 `data/voice_finish`로 이동한다.

배치 결과 JSON 필드는 API 응답 필드와 동일한 의미 체계를 사용한다.

## 5. 오류 처리

| HTTP 상태 | 상황 |
| --- | --- |
| `400` | 빈 파일, 잘못된 요청값, 복호화 실패 |
| `401` | API Key 누락 또는 불일치 |
| `404` | 처리 결과, 음성 파일, 학습 데이터 없음 |
| `413` | 업로드 파일 크기 초과 |
| `500` | STT/요약 처리 또는 내부 저장 오류 |
