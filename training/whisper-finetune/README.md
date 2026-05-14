# Whisper Fine-tuning Module

이 모듈은 `xcn-asr-summary` 운영 API와 분리된 Whisper STT fine-tuning 영역입니다.

운영 API는 `faster-whisper`로 CTranslate2 모델을 로딩합니다. 따라서 학습은 Hugging Face Transformers 포맷으로 수행하고, 학습 결과를 CTranslate2 포맷으로 변환한 뒤 API의 `WHISPER_MODEL` 경로만 교체합니다.

## 데이터 형식

`data/manifest/train.jsonl`, `data/manifest/validation.jsonl` 파일을 준비합니다.

각 줄은 아래 JSON 형식입니다.

```json
{"audio": "/workspace/data/raw/call_001.wav", "text": "상담 통화 정답 전사 문장입니다."}
```

- `audio`: 컨테이너 내부에서 접근 가능한 오디오 파일 경로
- `text`: 정답 전사 텍스트
- 권장 포맷: `wav`, 16 kHz mono
- 원본이 `m4a`, `mp3`여도 `datasets`/`ffmpeg`가 읽을 수 있으면 처리 가능

## 학습 이미지 빌드

프로젝트 루트에서 실행합니다.

```bash
docker build -f training/whisper-finetune/Dockerfile -t xcn-asr-summary/whisper-train:latest .
```

## 학습 실행

```bash
docker run --rm --gpus all \
  -v "$(pwd)/training/whisper-finetune/data:/workspace/data" \
  -v "$(pwd)/training/whisper-finetune/outputs:/workspace/outputs" \
  -v "$(pwd)/models:/models" \
  -e HF_TOKEN="${HF_TOKEN}" \
  xcn-asr-summary/whisper-train:latest \
  bash scripts/train.sh
```

기본 설정은 `configs/whisper-large-v3-turbo-ko.yaml`을 사용합니다.

## 평가 실행

```bash
docker run --rm --gpus all \
  -v "$(pwd)/training/whisper-finetune/data:/workspace/data" \
  -v "$(pwd)/training/whisper-finetune/outputs:/workspace/outputs" \
  -v "$(pwd)/models:/models" \
  xcn-asr-summary/whisper-train:latest \
  python evaluate.py --config configs/whisper-large-v3-turbo-ko.yaml
```

## CTranslate2 변환

```bash
docker run --rm --gpus all \
  -v "$(pwd)/training/whisper-finetune/outputs:/workspace/outputs" \
  -v "$(pwd)/models:/models" \
  xcn-asr-summary/whisper-train:latest \
  bash scripts/convert_to_ct2.sh
```

기본 출력 경로:

```text
models/whisper/large-v3-turbo-ko-custom-ct2
```

## 운영 API 적용

`.env`에서 아래처럼 변경합니다.

```env
WHISPER_MODEL=/models/whisper/large-v3-turbo-ko-custom-ct2
```

API 재기동:

```bash
docker compose up -d --build api
```

확인:

```bash
curl -sS http://127.0.0.1:18080/health
```

`whisper_model` 값이 `/models/whisper/large-v3-turbo-ko-custom-ct2`로 표시되면 적용된 것입니다.
