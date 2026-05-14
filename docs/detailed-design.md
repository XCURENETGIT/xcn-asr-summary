# xcn-asr-summary 상세설계서

## 1. 시스템 개요

xcn-asr-summary는 통화 음성 파일을 입력받아 STT 변환, 화자 구간 추정, 통화 요약, 화자별 요약을 수행하고 결과를 MariaDB와 JSON 파일에 저장하는 서비스다.

## 2. 구성 요소

| 구성 요소 | 역할 |
| --- | --- |
| `api` | FastAPI 기반 외부 API, Admin UI, voice watcher 실행 |
| `mariadb` | 통화 처리 결과와 학습 보정 데이터 저장 |
| `sllm-llamacpp` | llama.cpp 기반 A.X-4.0-Light GGUF 요약 모델 |
| `sllm-vllm` | vLLM 기반 고성능 요약 모델 옵션 |
| `data/voice` | 외부 시스템 파일 입력 폴더 |
| `data/translate` | 배치 처리 결과 JSON 저장 폴더 |
| `data/voice_finish` | 처리 완료 원본 파일 이동 폴더 |

## 3. 처리 흐름

### 3.1 API 업로드 처리

1. 외부 시스템이 `POST /calls/process`로 음성 파일을 업로드한다.
2. API Key가 설정된 경우 `X-API-Key`를 검증한다.
3. 파일명이 수집 파일명 규칙과 일치하면 발신번호, 내선번호, 통화 시작/종료 시각을 추출한다.
4. 업로드 크기와 파일 내용을 검증한다.
5. XCN 암호화 파일이면 복호화한다.
6. Whisper 모델로 STT 변환을 수행한다.
7. stereo 또는 segment 정보를 기반으로 화자 turn을 생성한다.
8. SLLM으로 구조화 요약과 문장형 요약을 생성한다.
9. 화자별 요약과 발화 segment를 생성한다.
10. `call_summaries` 테이블에 저장한다.
11. API 응답으로 처리 결과를 반환한다.

### 3.2 폴더 배치 처리

1. voice watcher가 `VOICE_WATCH_INTERVAL_SEC` 주기로 `VOICE_DIR`을 확인한다.
2. 확장자가 `VOICE_BATCH_EXTENSIONS`에 포함된 파일만 처리한다.
3. 수집 파일명 규칙과 일치하면 발신번호, 내선번호, 통화 시작/종료 시각을 추출한다.
4. 처리 중복 방지를 위해 `.lock` 파일을 생성한다.
5. `transcribe_and_summarize()` 파이프라인을 실행한다.
6. `data/translate`에 사용자용 JSON 결과를 저장한다.
7. 원본 파일을 `data/voice_finish`로 이동한다.
8. DB에는 `input_type='voice_file'`로 저장한다.

수집 파일명 규칙:

```text
발신번호_내선번호_년_월_일_시_분_초_월_일_시_분_초.wav
```

예시:

```text
01012345678_1001_2026_05_12_10_00_00_05_12_10_05_30.wav
```

## 4. 주요 모듈

| 파일 | 역할 |
| --- | --- |
| `app/main.py` | API 라우팅, 인증, 응답 변환, Admin UI 제공 |
| `app/pipeline.py` | STT, 화자 turn 처리, 요약 생성 핵심 파이프라인 |
| `app/db.py` | MariaDB 연결, 스키마 보정, 저장/조회 유틸 |
| `app/voice_batch.py` | `data/voice` 폴더 배치 처리 |
| `app/sllm_client.py` | OpenAI-compatible SLLM 호출 |
| `app/config.py` | 환경변수 기반 설정 |
| `app/admin/static/app.js` | Admin UI 클라이언트 로직 |

## 5. 데이터베이스 설계

### 5.1 `call_summaries`

통화 처리 결과를 저장한다.

| 컬럼 | 설명 |
| --- | --- |
| `id` | 내부 PK |
| `processing_id` | 처리 건 식별자 |
| `input_type` | 입력 방식. `api_request`, `voice_file` |
| `audio_file_name` | 원본 음성 파일명 |
| `audio_content_type` | 업로드 MIME 타입 |
| `stored_audio_path` | 저장된 음성 파일 경로 |
| `caller` | 발신번호 |
| `extension_number` | 내선번호 |
| `callee` | 수신번호 |
| `call_started_at` | 통화 시작 시각 |
| `call_ended_at` | 통화 종료 시각 |
| `audio_duration_seconds` | 음성 길이 |
| `detected_language` | 감지 언어 |
| `speech_recognition_model` | STT 모델명 |
| `summary_generation_model` | 요약 모델명 |
| `summary_model_backend` | 요약 백엔드 |
| `full_transcript` | 전체 전사 결과 |
| `structured_call_summary` | 항목형 통화 요약 |
| `plain_call_summary` | 일반 문장형 통화 요약 |
| `speaker_summary_list_json` | 화자별 요약 JSON |
| `speaker_segment_list_json` | 화자별 발화 segment JSON |
| `processing_time_ms` | 처리 소요 시간 |
| `processing_status` | 처리 상태 |
| `error_message` | 오류 메시지 |
| `result_created_at` | 결과 생성 시각 |

### 5.2 `stt_training_samples`

STT 보정 학습용 샘플을 저장한다.

| 컬럼 | 설명 |
| --- | --- |
| `call_summary_id` | 원본 통화 결과 ID |
| `request_id` | 처리 건 식별자. 현재 `processing_id` 값을 저장 |
| `filename` | 원본 음성 파일명 |
| `audio_path` | 학습용 clip 경로 |
| `segment_index` | 발화 segment 번호 |
| `speaker` | 화자명 |
| `start_seconds` | segment 시작 시각 |
| `end_seconds` | segment 종료 시각 |
| `original_text` | 원본 전사문 |
| `corrected_text` | 보정 전사문 |
| `status` | 학습 샘플 상태 |

## 6. API 응답 필드 기준

운영 연동 및 배치 JSON은 아래 명명 규칙을 사용한다.

| 필드 | 의미 |
| --- | --- |
| `processing_id` | 처리 건 식별자 |
| `processing_status` | 처리 상태 |
| `input_type` | 입력 방식 |
| `audio_file_name` | 원본 음성 파일명 |
| `caller` | 발신번호 |
| `extension_number` | 내선번호 |
| `callee` | 수신번호 |
| `call_started_at` | 통화 시작 시각 |
| `call_ended_at` | 통화 종료 시각 |
| `result_created_at` | 결과 생성 시각 |
| `speech_recognition_model` | STT 모델 |
| `summary_model_backend` | 요약 처리 방식 |
| `summary_generation_model` | 요약 모델 |
| `detected_language` | 감지 언어 |
| `audio_duration_seconds` | 음성 길이 |
| `processing_time_ms` | 처리 소요 시간 |
| `full_transcript` | 전체 전사 결과 |
| `structured_call_summary` | 항목형 통화 요약 |
| `plain_call_summary` | 문장형 통화 요약 |
| `speaker_summary_list` | 화자별 요약 목록 |

## 7. 주요 설정

| 환경변수 | 설명 |
| --- | --- |
| `WHISPER_MODEL` | faster-whisper 모델명 |
| `WHISPER_DEVICE` | Whisper 실행 장치 |
| `WHISPER_COMPUTE_TYPE` | Whisper compute type |
| `SLLM_BASE_URL` | OpenAI-compatible SLLM API 주소 |
| `SLLM_MODEL` | 요약 모델 served name |
| `SLLM_PROVIDER` | 요약 백엔드 라벨 |
| `VOICE_WATCH_ENABLED` | voice watcher 활성화 여부 |
| `VOICE_WATCH_INTERVAL_SEC` | voice 폴더 확인 주기 |
| `VOICE_WATCH_BATCH_LIMIT` | 주기당 처리 파일 수 |
| `SAVE_UPLOADS` | 업로드 원본 저장 여부 |
| `SAVE_TRAINING_CLIPS` | 학습용 segment clip 저장 여부 |

## 8. 배포 방식

- 기본 실행: `docker-compose.yml`
- llama.cpp/GGUF 기본 옵션: `.env.llamacpp-gguf`
- vLLM 옵션: `.env.vllm`
- API 바이너리 이미지: `Dockerfile.binary`
- GPU 배포 패키지: `scripts/package_gpu_bundle.sh`
- llama.cpp/GGUF 포함 배포 패키지: `scripts/package_llamacpp_gguf_bundle.sh`

## 9. 초기화 기준

현재 DB/API 필드명이 운영용으로 재정의되어 기존 DB와 호환되지 않는다. 기존 데이터가 필요 없으면 MariaDB 볼륨을 삭제하고 `db/init/001_schema.sql` 기준으로 새로 생성한다.
