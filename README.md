# xcn-asr-summary

유선 통화 음성 파일을 업로드하면 Whisper ASR로 전사하고, STT 변환 텍스트를 저장하는 Docker 기반 예제입니다. 현재 기본 파이프라인은 sLLM 요약 단계를 호출하지 않는 STT-only 구성입니다.

## 구성

- `POST /calls/process`
  - 음성 파일 업로드
  - Whisper 전사
  - 화자 turn 추정
  - MariaDB 저장
- `GET /calls/{call_id}`
  - 저장 결과 조회
- `GET /calls`
  - 최근 처리 목록 조회
- `GET /health`
  - DB/모델 상태 확인
- `GET /admin`
  - 관리/운용자 UI

## 실행

```bash
cp .env.example .env
docker compose up --build -d
```

접속:

- API docs: `http://localhost:18080/docs`
- Admin UI: `http://localhost:18080/admin`
- MariaDB: `localhost:13306`

## 요청 예시

```bash
curl -X POST "http://localhost:18080/calls/process" \
  -F "file=@sample.wav" \
  -F "caller=0212345678" \
  -F "callee=0298765432" \
  -F "call_started_at=2026-03-26T10:00:00"
```

## 폴더 배치 처리

외부 시스템이 `data/voice` 폴더에 `.wav` 파일을 넣으면 API 컨테이너가 주기적으로 확인해 순차 처리합니다. 처리 결과는 API 업로드와 동일하게 `call_summaries` DB에 저장되어 Admin UI에서 확인할 수 있으며, `input_type` 값으로 `api_request`와 `voice_file`을 구분합니다.

기본 스케줄링:

- `VOICE_WATCH_ENABLED=true`
- `VOICE_WATCH_INTERVAL_SEC=30`
- `VOICE_WATCH_BATCH_LIMIT=1`

수동으로 한 번만 실행할 때는 아래 명령을 사용할 수 있습니다.

```bash
./scripts/process_voice_batch.sh
```

처리 흐름:

- 입력: `data/voice/*.wav`
- 파일명 규칙이 `발신번호_내선번호_년_월_일_시_분_초_월_일_시_분_초.wav`와 일치하면 발신번호, 내선번호, 통화 시작/종료 시각을 자동 저장
- STT 전사 텍스트 저장: `data/translate/<원본파일명>.txt`
- DB 저장: `call_summaries.input_type='voice_file'`
- 성공한 원본 파일 이동: `data/voice_finish/<원본파일명>.wav`
- 처리 중복 방지: 원본 파일 옆에 `.lock` 파일을 생성해 동시 실행을 막음
- 실패한 파일은 `data/voice_failed`로 이동해 watcher가 같은 파일을 계속 재시도하지 않음

일부 파일만 테스트할 때:

```bash
./scripts/process_voice_batch.sh --limit 1
```

## 모델

- ASR: `faster-whisper` (`large-v3-turbo`, GPU 기본)
- 화자 분리: Whisper segment 기반 2인 통화 turn 추정
- SLLM 실행: 기본 파이프라인에서는 사용하지 않음. 기존 llama.cpp/vLLM 서비스 정의는 옵션으로 유지

첫 실행 시 모델을 다운로드하므로 초기 기동이 오래 걸릴 수 있습니다.

## 주요 환경변수

- `WHISPER_MODEL`
  - 기본 `large-v3-turbo`
- `WHISPER_DEVICE`
  - 기본 `cuda`
- `WHISPER_COMPUTE_TYPE`
  - 기본 `float16`
- `SLLM_BASE_URL`
  - OpenAI-compatible 서버 URL. 기본 `http://sllm-llamacpp:8080`
- `SLLM_MODEL`
  - API에서 호출할 served model name. 기본 `mykor/A.X-4.0-Light-gguf:Q4_K_M`
- `LLAMACPP_HF_MODEL`
  - llama.cpp가 로드할 GGUF 모델. 기본 `mykor/A.X-4.0-Light-gguf:Q4_K_M`
- `LLAMACPP_CTX_SIZE`
  - llama.cpp context size. 기본 `4096`
- `LLAMACPP_N_GPU_LAYERS`
  - GPU에 올릴 layer 수. 기본 `24`
- `SLLM_VLLM_MODEL`
  - vLLM 옵션 사용 시 로드할 Hugging Face 모델. 기본 `skt/A.X-4.0-Light`
- `SLLM_SERVED_MODEL_NAME`
  - vLLM served model name. 기본 `ax-4-light`
- `VLLM_MAX_MODEL_LEN`
  - A.X-4.0-Light 최대 컨텍스트에 맞춘 기본값 `16384`
- `SLLM_MAX_PROMPT_CHARS`
  - 통화 전사 프롬프트 최대 문자 수. 기본 `10000`
- `SLLM_PROVIDER`
  - `vllm`, `llamacpp`, `trtllm` 라벨 용도
- `SPEAKER_PAUSE_THRESHOLD_SEC`
  - 침묵 간격이 이 값 이상이면 화자 전환 후보로 판단
- `SPEAKER_MAX_TURN_MERGE_SEC`
  - 짧은 연속 segment를 같은 화자 turn으로 합칠 최대 간격
- `SPEAKER_ACK_MAX_CHARS`
  - `네`, `예`, `감사합니다` 같은 짧은 응답 판정 길이
- `HF_TOKEN`
  - private/gated 모델을 쓸 때만 설정
- `API_KEY`
  - 비워 두면 인증 없이 호출 가능
  - 값을 넣으면 `X-API-Key` 헤더가 필요
- `MAX_UPLOAD_MB`
  - 업로드 최대 크기
- `SAVE_UPLOADS`
  - 업로드 원본 파일 저장 여부
- `SAVE_TRAINING_CLIPS`
  - 학습 데이터 전달 시 segment 음성 clip 별도 저장 여부
  - 기본 `true`
- `VOICE_DIR`
  - 배치 처리 입력 폴더. 기본 `/app/data/voice`
- `VOICE_FINISH_DIR`
  - 처리 완료 원본 이동 폴더. 기본 `/app/data/voice_finish`
- `VOICE_FAILED_DIR`
  - STT/요약 실패 원본 이동 폴더. 기본 `/app/data/voice_failed`
- `TRANSLATE_DIR`
  - STT 전사 텍스트 저장 폴더. 기본 `/app/data/translate`
- `VOICE_BATCH_EXTENSIONS`
  - 배치 처리할 확장자 목록. 기본 `.wav`
- `VOICE_WATCH_ENABLED`
  - `data/voice` 폴더 자동 처리 활성화 여부. 기본 `true`
- `VOICE_WATCH_INTERVAL_SEC`
  - 자동 처리 확인 주기. 기본 `30`
- `VOICE_WATCH_BATCH_LIMIT`
  - 한 번의 주기에서 처리할 최대 파일 수. 기본 `1`
- `TRAINING_CLIP_FORMAT`
  - segment clip 저장 포맷. 기본 `wav`

## GPU 실행

- Whisper는 기본적으로 GPU에서 실행됩니다.
- `docker-compose.yml`은 구버전 Compose 호환을 위해 `runtime: nvidia`를 사용합니다.
- 서버에는 NVIDIA 드라이버와 Docker GPU runtime이 준비되어 있어야 합니다.
- 기본 실행은 STT-only이며 `sllm-llamacpp` 컨테이너를 시작하지 않습니다.
- 16GB GPU 환경을 고려해 llama.cpp 기본 context는 4096, GPU layer는 24로 제한합니다.
- 기존 llama.cpp/vLLM 방식은 옵션으로만 남아 있으며 각각 `./scripts/start.sh --sllm`, `./scripts/start.sh --vllm`으로 선택할 수 있습니다.

설정 파일은 역할별로 분리합니다.

- `.env`: DB, Whisper, voice watcher, 공통 API 설정
- `.env.llamacpp-gguf`: llama.cpp/GGUF 모델, context, GPU layer, 요약 생성값
- `.env.vllm`: vLLM 고성능 모델, served model name, GPU memory, max model length

컨테이너 이름은 `xcn-asr-summary-<역할>-gpu` 형태로 맞춥니다.

- `xcn-asr-summary-mariadb`
- `xcn-asr-summary-api-gpu`
- `xcn-asr-summary-llamacpp-gpu`
- `xcn-asr-summary-vllm-gpu`

API 이미지는 `xcn-asr-summary/api-gpu:<버전>` 형태로 태그를 지정합니다.

```bash
ASR_SUMMARY_IMAGE_TAG=1.0.0 ./scripts/start.sh --build
```

소스 노출을 줄여 배포하려면 Cython으로 `app/*.py`를 `.so`로 컴파일하는 바이너리 이미지를 사용합니다.

```bash
ASR_SUMMARY_IMAGE_TAG=1.0.0 ./scripts/start.sh --build --binary
```

컴파일된 `.so` 파일만 별도로 확인하거나 전달해야 할 때:

```bash
./scripts/build_binary_app.sh --version 1.0.0
```

## 배포 패키지 생성

GPU 배포 패키지는 기본적으로 바이너리 이미지(`Dockerfile.binary`)를 빌드한 뒤 이미지 tar, compose, 실행 스크립트, DB 초기화 파일을 묶습니다.

```bash
./scripts/package_gpu_bundle.sh --version 1.0.0
```

생성 파일 예:

```text
dist/xcn-asr-summary-gpu-package-1.0.0-YYYYMMDD-HHMMSS.tar.gz
```

llama.cpp/vLLM 이미지는 용량이 커서 기본 포함하지 않습니다. 오프라인 배포가 필요하면 `--include-llamacpp-image` 또는 `--include-vllm-image`를 추가합니다. 모델 캐시까지 포함해야 할 때만 `--include-model-cache`를 사용합니다.

llama.cpp + GGUF 모델까지 포함한 배포본은 전용 스크립트를 사용합니다. 이 패키지는 API 이미지, llama.cpp CUDA 이미지, `mykor/A.X-4.0-Light-gguf:Q4_K_M` 캐시만 포함합니다.

```bash
./scripts/package_llamacpp_gguf_bundle.sh --version 1.0.0
```

생성된 tar.gz를 풀면 최상위 폴더명은 기본 `xcn-asr-summary`입니다.

## 저장 테이블

- `call_summaries`
  - 업로드 메타데이터
  - 전사 텍스트
  - 전체 요약 텍스트
  - 화자별 요약 JSON
  - 화자 segment JSON
  - 사용 모델 정보
  - 처리 시간
  - 상태 / 에러 메시지

## 응답 예시

`POST /calls/process`, `GET /calls`, `GET /calls/{call_id}`, `data/translate/<원본파일명>.json` 결과는 사용자가 바로 의미를 알 수 있도록 아래 항목명으로 생성됩니다.

- `processing_id`: 처리 건 식별자
- `processing_status`: 처리 상태
- `input_type`: 입력 방식
- `audio_file_name`: 원본 음성 파일명
- `caller`: 발신번호
- `extension_number`: 내선번호
- `callee`: 수신번호 또는 내선번호
- `call_started_at`: 통화 시작 시각
- `call_ended_at`: 통화 종료 시각
- `result_created_at`: 결과 생성 시각
- `speech_recognition_model`: STT 모델
- `summary_model_backend`: 요약 처리 방식
- `summary_generation_model`: 요약 모델
- `detected_language`: 감지 언어
- `audio_duration_seconds`: 음성 길이
- `processing_time_ms`: 처리 소요 시간
- `full_transcript`: 전체 전사 결과
- `structured_call_summary`: 항목형 통화 요약
- `plain_call_summary`: 일반 문장형 통화 요약
- `speaker_summary_list`: 화자별 요약 목록

## Admin UI

`/admin`에서 관리/운용자용 화면을 제공합니다.

주요 기능:

- API Key 기반 로그인
- STT 변환 목록 조회
- 파일명, 처리 ID, 전사/요약 내용, 발신번호, 수신번호, 상태, 생성일 조건 검색
- 변환 상세 조회
- 요청 음성 파일 재생
- 재생 위치에 맞춘 STT segment 표시
- segment 단위 STT 텍스트 수정
- 수정된 segment 텍스트를 시작/종료 시간과 함께 Whisper fine-tuning용 학습 데이터 큐로 전달
- 학습 데이터 큐 조회
- 학습 데이터 JSONL manifest 다운로드
- 학습 데이터 큐 단건 삭제 및 queued 일괄 삭제

학습 데이터 큐는 `stt_training_samples` 테이블에 저장됩니다. segment 단위 샘플은 `segment_index`, `speaker`, `start_seconds`, `end_seconds`, `original_text`, `corrected_text`를 함께 저장하므로 음성 구간과 정답 텍스트를 매핑할 수 있습니다.

`SAVE_TRAINING_CLIPS=true`이면 학습 데이터 전달 시 `/app/data/training-clips`에 segment 음성만 별도 `wav`로 저장합니다. 호스트 기준 경로는 `data/training-clips`입니다. 이 방식이면 업로드 원본 파일을 장기 보관하지 않아도 학습 샘플은 유지됩니다.

manifest 기본 `audio` 경로는 fine-tuning 컨테이너 기준 `/workspace/data/raw/training-clips/{clip}.wav`입니다. 학습 실행 시 호스트의 `data/training-clips`를 해당 위치로 마운트하면 됩니다.

## SLLM 실행

기본 실행은 llama.cpp + GGUF Q4_K_M 모델입니다.

```bash
./scripts/start.sh --sllm
```

기본 모델:

```text
.env.llamacpp-gguf
LLAMACPP_HF_MODEL=mykor/A.X-4.0-Light-gguf:Q4_K_M
SLLM_BASE_URL=http://sllm-llamacpp:8080
SLLM_MODEL=mykor/A.X-4.0-Light-gguf:Q4_K_M
```

vLLM을 사용할 때:

```bash
./scripts/start.sh --vllm
```

기본 vLLM 모델은 `skt/A.X-4.0-Light`이며 API에서는 `ax-4-light` 이름으로 호출합니다.
vLLM 관련 튜닝값은 `.env.vllm`에서 관리합니다.

vLLM 모델 캐시는 현재 프로젝트의 `models` 디렉터리에 저장합니다.

```text
호스트 경로: /data01/xcn-asr-summary/models
컨테이너 경로: /model-store
HF_HOME: /model-store/hf-cache
```

별도 `docker-compose.server.yml` 오버라이드는 사용하지 않습니다. `scripts/start.sh`, `scripts/stop.sh`, `scripts/reset-db.sh`는 `docker-compose.yml`만 사용합니다.

## Whisper Fine-tuning

Whisper fine-tuning은 운영 API 이미지에 포함하지 않고 `training/whisper-finetune` 모듈에서 별도로 수행합니다.

기본 흐름:

```bash
docker build -f training/whisper-finetune/Dockerfile -t xcn-asr-summary/whisper-train:latest .
docker run --rm --gpus all \
  -v "$(pwd)/training/whisper-finetune/data:/workspace/data" \
  -v "$(pwd)/training/whisper-finetune/outputs:/workspace/outputs" \
  -v "$(pwd)/models:/models" \
  xcn-asr-summary/whisper-train:latest \
  bash scripts/train.sh
docker run --rm --gpus all \
  -v "$(pwd)/training/whisper-finetune/outputs:/workspace/outputs" \
  -v "$(pwd)/models:/models" \
  xcn-asr-summary/whisper-train:latest \
  bash scripts/convert_to_ct2.sh
```

변환 완료 후 `.env`의 `WHISPER_MODEL`을 CTranslate2 모델 경로로 변경합니다.

```env
WHISPER_MODEL=/models/whisper/large-v3-turbo-ko-custom-ct2
```

자세한 데이터 형식과 실행 방법은 `training/whisper-finetune/README.md`를 참고합니다.
