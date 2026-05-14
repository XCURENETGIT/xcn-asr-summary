# xcn-asr-summary 명령어 모음

## 1. 기본 실행

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

## 2. 서비스 시작/중지

llama.cpp/GGUF 기본 구성:

```bash
./scripts/start.sh --sllm
./scripts/stop.sh --sllm
```

빌드 후 시작:

```bash
./scripts/start.sh --build --sllm
```

바이너리 API 이미지 빌드 후 시작:

```bash
./scripts/start.sh --build --binary --sllm
```

vLLM 구성:

```bash
./scripts/start.sh --vllm
./scripts/stop.sh --vllm
```

볼륨까지 삭제하며 중지:

```bash
./scripts/stop.sh --sllm --volumes
```

PowerShell 실행:

```powershell
.\scripts\start.ps1 -Sllm
.\scripts\stop.ps1 -Sllm
```

## 3. DB 초기화

```bash
./scripts/reset-db.sh
```

Docker Compose로 직접 초기화:

```bash
docker compose down -v
docker compose up --build -d
```

## 4. 상태 확인

```bash
curl http://localhost:18080/health
docker compose ps
docker compose logs -f api
docker compose logs -f sllm-llamacpp
docker compose logs -f mariadb
```

## 5. API 호출

음성 파일 처리:

```bash
curl -X POST "http://localhost:18080/calls/process" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@sample.wav" \
  -F "caller=0212345678" \
  -F "callee=0298765432" \
  -F "call_started_at=2026-05-12T10:00:00"
```

목록 조회:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls?limit=20"
```

상세 조회:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls/1"
```

화자 segment 조회:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls/1/segments"
```

원본 음성 다운로드:

```bash
curl -L -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls/1/audio" \
  -o call.wav
```

전사문 수정:

```bash
curl -X PATCH "http://localhost:18080/calls/1/transcript" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"full_transcript":"수정된 전체 전사문"}'
```

처리 결과 삭제:

```bash
curl -X DELETE -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/calls/1?delete_audio=true&delete_training_clips=true"
```

## 6. voice 폴더 배치 처리

수동 1회 처리:

```bash
./scripts/process_voice_batch.sh
```

1개 파일만 처리:

```bash
./scripts/process_voice_batch.sh --limit 1
```

watch 모드:

```bash
./scripts/process_voice_batch.sh --watch
```

결과 확인:

```bash
ls -al data/translate
ls -al data/voice_finish
```

## 7. 학습 데이터 관리

학습 샘플 조회:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/training-samples?status=queued"
```

학습 manifest 다운로드:

```bash
curl -L -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/training-samples/manifest?status=queued" \
  -o stt-training-manifest.jsonl
```

학습 작업 시작:

```bash
curl -X POST "http://localhost:18080/training/jobs" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"status":"queued","max_steps":40,"learning_rate":0.00002}'
```

학습 작업 목록:

```bash
curl -H "X-API-Key: ${API_KEY}" \
  "http://localhost:18080/training/jobs"
```

## 8. 바이너리 빌드와 패키지 생성

`.so` 컴파일 산출물 생성:

```bash
./scripts/build_binary_app.sh --version 1.0.0
```

GPU 패키지 생성:

```bash
./scripts/package_gpu_bundle.sh --version 1.0.0
```

llama.cpp/GGUF 포함 패키지 생성:

```bash
./scripts/package_llamacpp_gguf_bundle.sh --version 1.0.0
```

llama.cpp 이미지까지 포함:

```bash
./scripts/package_gpu_bundle.sh --version 1.0.0 --include-llamacpp-image
```

모델 캐시까지 포함:

```bash
./scripts/package_llamacpp_gguf_bundle.sh --version 1.0.0 --include-model-cache
```

## 9. 자주 쓰는 Docker 명령

```bash
docker compose config
docker compose ps
docker compose logs --tail=200 api
docker compose restart api
docker compose exec api python -m app.voice_batch --limit 1
docker compose exec mariadb mariadb -uroot -p${MARIADB_ROOT_PASSWORD}
```

