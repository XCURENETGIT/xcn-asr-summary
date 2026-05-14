import WaveSurfer from "/admin/static/vendor/wavesurfer.esm.js?v=20260415-1748";

const state = {
  apiKey: localStorage.getItem("xcnTelSummaryApiKey") || "",
  currentCall: null,
  segments: [],
  activeSegmentIndex: null,
  view: "calls",
  wavesurfer: null,
  trainingStatusFilter: "queued",
};

const $ = (id) => document.getElementById(id);

function headers(extra = {}) {
  const base = { ...extra };
  if (state.apiKey) base["X-API-Key"] = state.apiKey;
  return base;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: headers(options.headers || {}),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_error) {
      // Ignore non-JSON error responses.
    }
    throw new Error(detail);
  }
  return response.json();
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(node.timer);
  node.timer = setTimeout(() => {
    node.hidden = true;
  }, 3200);
}

function formatDate(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString("ko-KR");
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "-";
  const total = Math.round(seconds);
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

function formatSource(value) {
  return value === "voice_file" ? "voice 폴더" : "API";
}

function formatClock(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const hour = Math.floor(total / 3600);
  const min = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (hour > 0) return `${hour}:${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function sortSegments(segments) {
  return [...segments].sort((left, right) => {
    const startDiff = (left.start_seconds || 0) - (right.start_seconds || 0);
    if (startDiff !== 0) return startDiff;
    const endDiff = (left.end_seconds || 0) - (right.end_seconds || 0);
    if (endDiff !== 0) return endDiff;
    return String(left.speaker || "").localeCompare(String(right.speaker || ""));
  });
}

const SUMMARY_LABELS = [
  "통화 목적",
  "핵심 이슈",
  "상담사 안내",
  "처리 결과",
  "리스크/특이사항",
  "후속 조치",
];

function parseLabeledSummary(text) {
  const source = String(text || "").trim();
  if (!source) return {};
  const escapedLabels = SUMMARY_LABELS.map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const pattern = new RegExp(`(?:^|\\n|\\s)\\s*(?:[-*]\\s*)?(?:\\[)?(${escapedLabels})(?:\\])?\\s*[:：]\\s*`, "g");
  const matches = [...source.matchAll(pattern)];
  const parsed = {};
  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index];
    const label = match[1];
    const start = match.index + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index : source.length;
    const value = source.slice(start, end).replace(/\n\s*/g, " ").trim().replace(/^[-\s]+|[-\s]+$/g, "");
    if (value) parsed[label] = value;
  }
  return parsed;
}

function renderSummaryFields(call) {
  const labeled = parseLabeledSummary(call.structured_call_summary || "");
  $("conversationalSummary").textContent = call.plain_call_summary || "-";
  $("inquirySummary").textContent = labeled["통화 목적"] || "-";
  $("issueSummary").textContent = labeled["핵심 이슈"] || "-";
  $("guidanceSummary").textContent = labeled["상담사 안내"] || "-";
  $("outcomeSummary").textContent = labeled["처리 결과"] || "-";
  $("riskSummary").textContent = labeled["리스크/특이사항"] || "-";
  $("actionSummary").textContent = labeled["후속 조치"] || "-";
}

function bootApp() {
  $("loginShell").hidden = true;
  $("appShell").hidden = false;
  initWaveSurfer();
  loadHealth();
  loadCalls();
}

function initWaveSurfer() {
  if (state.wavesurfer) return;
  state.wavesurfer = WaveSurfer.create({
    container: "#waveform",
    media: $("audioPlayer"),
    height: 96,
    normalize: true,
    waveColor: "#9bb8b2",
    progressColor: "#0f766e",
    cursorColor: "#d97706",
    cursorWidth: 2,
    barWidth: 2,
    barGap: 2,
    barRadius: 2,
    mediaControls: false,
    dragToSeek: true,
  });
  $("wavePlayer").classList.add("wave-ready");
  state.wavesurfer.on("error", (error) => {
    console.error("WaveSurfer error", error);
    $("wavePlayer").classList.remove("wave-ready");
    $("waveDuration").textContent = "파형 로딩 실패";
  });
  state.wavesurfer.on("timeupdate", syncActiveSegment);
  state.wavesurfer.on("ready", () => {
    updateWaveDuration();
    $("wavePlayButton").disabled = false;
  });
  state.wavesurfer.on("play", () => {
    $("wavePlayButton").textContent = "일시정지";
  });
  state.wavesurfer.on("pause", () => {
    $("wavePlayButton").textContent = "재생";
  });
  state.wavesurfer.on("finish", () => {
    $("wavePlayButton").textContent = "재생";
  });
}

function setAudioSource(src) {
  const audio = $("audioPlayer");
  audio.src = src;
  $("wavePlayButton").disabled = true;
  $("wavePlayButton").textContent = "재생";
  $("waveDuration").textContent = "00:00 / 00:00";
  if (state.wavesurfer) {
    state.wavesurfer.load(src);
  }
}

function getCurrentTime() {
  if (state.wavesurfer) return state.wavesurfer.getCurrentTime();
  return $("audioPlayer").currentTime;
}

function playFrom(seconds) {
  if (state.wavesurfer) {
    state.wavesurfer.setTime(seconds);
    state.wavesurfer.play().catch(() => undefined);
    return;
  }
  $("audioPlayer").currentTime = seconds;
  $("audioPlayer").play().catch(() => undefined);
}

function updateWaveDuration() {
  const current = getCurrentTime();
  const duration = state.wavesurfer?.getDuration() || $("audioPlayer").duration || 0;
  $("waveDuration").textContent = `${formatClock(current)} / ${formatClock(duration)}`;
}

async function authenticate(apiKey) {
  state.apiKey = apiKey.trim();
  const previousMessage = $("loginMessage").textContent;
  $("loginMessage").textContent = "접속 확인 중입니다.";
  try {
    await api("/calls?limit=1");
    if (state.apiKey) {
      localStorage.setItem("xcnTelSummaryApiKey", state.apiKey);
    } else {
      localStorage.removeItem("xcnTelSummaryApiKey");
    }
    bootApp();
  } catch (error) {
    $("loginMessage").textContent = `접속 실패: ${error.message}`;
    state.apiKey = "";
    if (previousMessage) localStorage.removeItem("xcnTelSummaryApiKey");
  }
}

async function loadHealth() {
  try {
    const health = await api("/health");
    $("healthText").textContent = `${health.status} / ${health.whisper_model}`;
  } catch (error) {
    $("healthText").textContent = `상태 확인 실패: ${error.message}`;
  }
}

function buildQueryFromForm(form) {
  const params = new URLSearchParams();
  const formData = new FormData(form);
  for (const [key, value] of formData.entries()) {
    if (!value) continue;
    params.set(key, value);
  }
  params.set("limit", "80");
  return params.toString();
}

async function loadCalls() {
  const query = buildQueryFromForm($("searchForm"));
  const calls = await api(`/calls?${query}`);
  renderCalls(calls);
}

function renderCalls(calls) {
  const list = $("callList");
  list.innerHTML = "";
  if (!calls.length) {
    list.innerHTML = `<div class="empty-state"><strong>검색 결과가 없습니다.</strong><span>조건을 바꿔 다시 조회하세요.</span></div>`;
    return;
  }
  for (const item of calls) {
    const card = document.createElement("article");
    card.className = `call-item ${state.currentCall?.id === item.id ? "active" : ""}`;
    card.tabIndex = 0;
    card.innerHTML = `
      <div class="call-item-head">
        <strong>${escapeHtml(item.audio_file_name)}</strong>
        <button class="call-delete" type="button">삭제</button>
      </div>
      <div class="call-meta">
        <span class="badge">${escapeHtml(item.processing_status)}</span>
        <span class="badge source-${escapeHtml(item.input_type || "api_request")}">${escapeHtml(formatSource(item.input_type))}</span>
        <span>#${item.id}</span>
        <span>${formatDuration(item.audio_duration_seconds)}</span>
        <span>${formatDate(item.result_created_at)}</span>
      </div>
      <span class="muted">${escapeHtml(item.plain_call_summary || item.structured_call_summary || item.full_transcript || "내용 없음").slice(0, 120)}</span>
    `;
    card.addEventListener("click", () => selectCall(item.id));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectCall(item.id);
      }
    });
    card.querySelector(".call-delete").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteCall(item).catch((error) => toast(error.message));
    });
    list.appendChild(card);
  }
}

function clearDetail() {
  state.currentCall = null;
  state.segments = [];
  state.activeSegmentIndex = null;
  $("detailContent").hidden = true;
  $("emptyDetail").hidden = false;
  $("audioPlayer").removeAttribute("src");
  $("wavePlayButton").disabled = true;
  $("wavePlayButton").textContent = "재생";
  $("waveDuration").textContent = "00:00 / 00:00";
  if (state.wavesurfer && typeof state.wavesurfer.empty === "function") {
    state.wavesurfer.empty();
  }
}

async function deleteCall(item) {
  const label = item.audio_file_name ? `${item.audio_file_name} (#${item.id})` : `#${item.id}`;
  if (!confirm(`${label} STT 변환 항목을 삭제할까요? 저장된 음성 파일과 연결된 학습 segment clip도 함께 삭제됩니다.`)) return;
  await api(`/calls/${item.id}?delete_audio=true&delete_training_clips=true`, { method: "DELETE" });
  if (state.currentCall?.id === item.id) {
    clearDetail();
  }
  toast("STT 변환 항목을 삭제했습니다.");
  await loadCalls();
  if (state.view === "training") {
    await loadTrainingSamples();
  }
}

async function selectCall(id) {
  const call = await api(`/calls/${id}`);
  let segments = [];
  try {
    segments = await api(`/calls/${id}/segments`);
  } catch (_error) {
    segments = [];
  }
  state.currentCall = call;
  state.segments = sortSegments(segments);
  renderDetail();
  loadCalls();
}

function renderDetail() {
  const call = state.currentCall;
  $("emptyDetail").hidden = true;
  $("detailContent").hidden = false;
  $("detailStatus").textContent = `${call.processing_status} / ${call.summary_model_backend || "-"} / ${formatSource(call.input_type)}`;
  $("detailFilename").textContent = call.audio_file_name;
  const callTime = call.call_started_at
    ? ` · 통화 ${formatDate(call.call_started_at)}${call.call_ended_at ? ` ~ ${formatDate(call.call_ended_at)}` : ""}`
    : "";
  const callerText = call.caller ? ` · 발신 ${call.caller}` : "";
  const extensionText = call.extension_number ? ` · 내선 ${call.extension_number}` : "";
  $("detailMeta").textContent = `처리 ID ${call.processing_id} · 생성 ${formatDate(call.result_created_at)} · 길이 ${formatDuration(call.audio_duration_seconds)}${callTime}${callerText}${extensionText}`;
  const audioParams = new URLSearchParams();
  if (state.apiKey) audioParams.set("api_key", state.apiKey);
  audioParams.set("t", Date.now());
  setAudioSource(`/calls/${call.id}/audio?${audioParams.toString()}`);
  renderSummaryFields(call);
  renderSegments();
}

function renderSegments() {
  const list = $("segmentList");
  list.innerHTML = "";
  state.activeSegmentIndex = null;
  if (!state.segments.length) {
    list.innerHTML = `<p class="active-segment">저장된 segment가 없어 학습 데이터를 생성할 수 없습니다.</p>`;
    return;
  }
  for (let index = 0; index < state.segments.length; index += 1) {
    const segment = state.segments[index];
    const row = document.createElement("div");
    row.className = "segment-row";
    row.dataset.index = String(index);
    row.innerHTML = `
      <span class="segment-time">${formatClock(segment.start_seconds)}-${formatClock(segment.end_seconds)}</span>
      <span class="segment-speaker">${escapeHtml(segment.speaker)}</span>
      <p class="segment-text">${escapeHtml(segment.text)}</p>
      <button class="segment-edit" type="button">수정</button>
    `;
    row.addEventListener("click", (event) => {
      if (event.target.closest("button") || event.target.closest("textarea")) return;
      playFrom(segment.start_seconds);
    });
    row.querySelector(".segment-edit").addEventListener("click", () => openSegmentEditor(index));
    list.appendChild(row);
  }
}

function openSegmentEditor(index) {
  const row = document.querySelector(`.segment-row[data-index="${index}"]`);
  const segment = state.segments[index];
  if (!row || !segment) return;
  document.querySelectorAll(".segment-editor").forEach((item) => item.remove());
  const editor = document.createElement("div");
  editor.className = "segment-editor";
  editor.innerHTML = `
    <textarea>${escapeHtml(segment.corrected_text || segment.text || "")}</textarea>
    <input class="segment-note" placeholder="수정 메모 (선택)" />
    <div class="segment-editor-actions">
      <button class="cancel" type="button">취소</button>
      <button class="send" type="button">수정 저장 + 학습 전달</button>
    </div>
  `;
  editor.querySelector(".cancel").addEventListener("click", () => editor.remove());
  editor.querySelector(".send").addEventListener("click", () => {
    const corrected = editor.querySelector("textarea").value.trim();
    const note = editor.querySelector(".segment-note").value.trim();
    sendSegmentTraining(index, corrected, note)
      .then(() => {
        segment.corrected_text = corrected;
        segment.text = corrected;
        renderSegments();
        toast("segment 수정 내용을 학습 데이터 큐에 전달했습니다.");
      })
      .catch((error) => toast(error.message));
  });
  row.appendChild(editor);
  editor.querySelector("textarea").focus();
}

function scrollActiveSegmentIntoView(row) {
  const list = $("segmentList");
  if (!list || !row) return;
  const listRect = list.getBoundingClientRect();
  const rowRect = row.getBoundingClientRect();
  const padding = 12;
  if (rowRect.top >= listRect.top + padding && rowRect.bottom <= listRect.bottom - padding) {
    return;
  }
  const targetTop = list.scrollTop + (rowRect.top - listRect.top) - (list.clientHeight - row.offsetHeight) / 2;
  list.scrollTo({
    top: Math.max(0, targetTop),
    behavior: "smooth",
  });
}

function syncActiveSegment() {
  const current = getCurrentTime();
  $("activeTime").textContent = formatClock(current);
  updateWaveDuration();
  let index = -1;
  for (let candidateIndex = 0; candidateIndex < state.segments.length; candidateIndex += 1) {
    const item = state.segments[candidateIndex];
    if (current >= item.start_seconds && current <= item.end_seconds) {
      index = candidateIndex;
    }
    if (item.start_seconds > current) break;
  }
  if (index === state.activeSegmentIndex) return;
  state.activeSegmentIndex = index >= 0 ? index : null;
  document.querySelectorAll(".segment-row.active").forEach((item) => item.classList.remove("active"));
  if (index >= 0) {
    const row = document.querySelector(`.segment-row[data-index="${index}"]`);
    if (row) {
      row.classList.add("active");
      scrollActiveSegmentIntoView(row);
    }
  }
}

async function sendSegmentTraining(index, corrected, note) {
  if (!state.currentCall) return;
  const segment = state.segments[index];
  if (!segment) throw new Error("segment를 찾을 수 없습니다.");
  if (!corrected) throw new Error("수정 텍스트가 없습니다.");
  await api(`/calls/${state.currentCall.id}/training-samples`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segment_index: index,
      speaker: segment.speaker,
      start_seconds: segment.start_seconds,
      end_seconds: segment.end_seconds,
      original_text: segment.text,
      corrected_text: corrected,
      note: note || null,
    }),
  });
  loadTrainingSamples();
}

async function loadTrainingSamples() {
  const params = new URLSearchParams({ limit: "100" });
  if (state.trainingStatusFilter) params.set("status", state.trainingStatusFilter);
  const samples = await api(`/training-samples?${params.toString()}`);
  const list = $("trainingList");
  list.innerHTML = "";
  if (!samples.length) {
    const statusLabel = state.trainingStatusFilter || "전체";
    list.innerHTML = `<div class="empty-state"><strong>${escapeHtml(statusLabel)} 학습 데이터가 없습니다.</strong><span>상세 화면에서 수정 텍스트를 학습 데이터로 전달하세요.</span></div>`;
    return;
  }
  for (const item of samples) {
    const card = document.createElement("article");
    card.className = "training-item";
    card.innerHTML = `
      <strong>${escapeHtml(item.filename)}</strong>
      <div class="training-meta">
        <span class="badge">${escapeHtml(item.status)}</span>
        <span>call #${item.call_summary_id}</span>
        ${item.segment_index !== null && item.segment_index !== undefined ? `<span>segment #${item.segment_index}</span>` : ""}
        ${item.start_seconds !== null && item.end_seconds !== null ? `<span>${formatClock(item.start_seconds)}-${formatClock(item.end_seconds)}</span>` : ""}
        <span>${formatDate(item.created_at)}</span>
      </div>
      ${item.audio_path ? `<audio class="training-audio" controls preload="metadata" src="${trainingAudioUrl(item.id)}"></audio>` : `<p class="muted">저장된 음성 clip이 없습니다.</p>`}
      <div class="training-texts">
        <div>
          <span>수정 전</span>
          <p>${escapeHtml(item.original_text || "-")}</p>
        </div>
        <div>
          <span>수정 후</span>
          <p>${escapeHtml(item.corrected_text || "-")}</p>
        </div>
      </div>
      ${item.audio_path ? `<p class="muted">clip: ${escapeHtml(item.audio_path)}</p>` : ""}
      ${item.note ? `<p><strong>메모</strong> ${escapeHtml(item.note)}</p>` : ""}
      <div class="training-actions">
        <button class="training-delete" type="button">삭제</button>
      </div>
    `;
    card.querySelector(".training-delete").addEventListener("click", () => deleteTrainingSample(item.id));
    list.appendChild(card);
  }
}

async function deleteTrainingSample(id) {
  if (!confirm(`학습 데이터 #${id}를 삭제할까요? 연결된 segment clip도 함께 삭제됩니다.`)) return;
  await api(`/training-samples/${id}?delete_clip=true`, { method: "DELETE" });
  toast("학습 데이터를 삭제했습니다.");
  loadTrainingSamples();
}

async function deleteQueuedSamples() {
  if (!confirm("queued 상태 학습 데이터를 모두 삭제할까요? 연결된 segment clip도 함께 삭제됩니다.")) return;
  const result = await api("/training-samples?status=queued&delete_clip=true", { method: "DELETE" });
  toast(`queued 학습 데이터 ${result.deleted_count}건을 삭제했습니다.`);
  loadTrainingSamples();
}

async function loadTrainingOps() {
  const [jobs, models, baseModels] = await Promise.all([
    api("/training/jobs"),
    api("/training/models"),
    api("/training/base-models"),
  ]);
  renderTrainingBaseModels(baseModels);
  renderTrainingJobs(jobs);
  renderTrainingModels(models);
}

function renderTrainingBaseModels(models) {
  const select = $("trainingBaseModel");
  const current = select.value;
  select.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.value;
    option.textContent = model.source === "base" ? `${model.name} (base)` : `${model.name} (fine-tuned)`;
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

function renderTrainingJobs(jobs) {
  const list = $("trainingJobList");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = `<div class="empty-state"><strong>학습 Job이 없습니다.</strong><span>queued 학습 데이터로 새 학습을 시작하세요.</span></div>`;
    return;
  }
  for (const job of jobs) {
    const activeMetrics = job.metrics?.active_model;
    const trainedMetrics = job.metrics?.trained_model;
    const card = document.createElement("article");
    card.className = "training-item";
    card.innerHTML = `
      <strong>${escapeHtml(job.model_name)}</strong>
      <div class="training-meta">
        <span class="badge">${escapeHtml(job.status)}</span>
        <span>samples ${job.sample_count || 0}</span>
        <span>queued ${job.queued_sample_count ?? "-"}</span>
        <span>reused ${job.reused_sample_count ?? "-"}</span>
        <span>steps ${job.max_steps || "-"}</span>
        <span>${escapeHtml(job.started_at || "-")}</span>
      </div>
      ${job.base_model ? `<p class="muted">base: ${escapeHtml(job.base_model)}</p>` : ""}
      ${job.model_path ? `<p class="muted">${escapeHtml(job.model_path)}</p>` : ""}
      ${job.hf_model_path ? `<p class="muted">hf: ${escapeHtml(job.hf_model_path)}</p>` : ""}
      ${job.error_message ? `<p class="muted">오류: ${escapeHtml(job.error_message)}</p>` : ""}
      <div class="metric-row">
        <span>기존 CER ${formatPercent(activeMetrics?.avg_cer)}</span>
        <span>기존 WER ${formatPercent(activeMetrics?.avg_wer)}</span>
        <span>학습 CER ${formatPercent(trainedMetrics?.avg_cer)}</span>
        <span>학습 WER ${formatPercent(trainedMetrics?.avg_wer)}</span>
      </div>
      <div class="training-actions">
        <button class="training-log ghost-button" type="button">로그 보기</button>
      </div>
    `;
    card.querySelector(".training-log").addEventListener("click", () => toggleTrainingLog(card, job.id));
    list.appendChild(card);
  }
}

async function toggleTrainingLog(card, jobId) {
  const existing = card.querySelector(".job-log");
  if (existing) {
    existing.remove();
    return;
  }
  const response = await fetch(`/training/jobs/${jobId}/log`, { headers: headers() });
  const logText = response.ok ? await response.text() : `로그 조회 실패: ${response.status}`;
  const node = document.createElement("pre");
  node.className = "job-log";
  node.textContent = logText.slice(-12000) || "로그가 없습니다.";
  card.appendChild(node);
}

function renderTrainingModels(models) {
  const list = $("trainingModelList");
  list.innerHTML = "";
  if (!models.length) {
    list.innerHTML = `<div class="empty-state"><strong>생성 모델이 없습니다.</strong><span>학습 완료 후 모델이 표시됩니다.</span></div>`;
    return;
  }
  for (const model of models) {
    const metrics = model.metrics?.trained_model;
    const card = document.createElement("article");
    card.className = "training-item";
    card.innerHTML = `
      <strong>${escapeHtml(model.name)}</strong>
      <div class="training-meta">
        ${model.is_active ? `<span class="badge">active</span>` : ""}
        ${model.has_hf_checkpoint ? `<span class="badge">continued</span>` : ""}
        <span>${escapeHtml(model.created_at || "-")}</span>
        <span>${model.size_bytes ? `${Math.round(model.size_bytes / 1024 / 1024)} MB` : "-"}</span>
      </div>
      <p class="muted">${escapeHtml(model.path)}</p>
      ${model.hf_model_path ? `<p class="muted">hf: ${escapeHtml(model.hf_model_path)}</p>` : ""}
      <div class="metric-row">
        <span>CER ${formatPercent(metrics?.avg_cer)}</span>
        <span>WER ${formatPercent(metrics?.avg_wer)}</span>
      </div>
    `;
    list.appendChild(card);
  }
}

async function startTraining() {
  const payload = {
    status: "queued",
    model_name: $("trainingModelName").value.trim() || null,
    base_model: $("trainingBaseModel").value || null,
    include_used_samples: $("trainingIncludeUsed").checked,
    max_steps: Number($("trainingMaxSteps").value || 40),
    learning_rate: Number($("trainingLearningRate").value || 0.00002),
    gpu_device: $("trainingGpuDevice").value.trim() || null,
  };
  const job = await api("/training/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  toast(`학습 Job을 시작했습니다: ${job.id}`);
  loadTrainingOps();
}

function switchView(view) {
  state.view = view;
  $("callsView").hidden = view !== "calls";
  $("trainingView").hidden = view !== "training";
  $("trainingOpsView").hidden = view !== "trainingOps";
  $("pageTitle").textContent = view === "calls" ? "STT 변환 목록" : (view === "training" ? "학습 데이터 확인" : "학습 관리");
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === view);
  });
  if (view === "training") loadTrainingSamples();
  if (view === "trainingOps") loadTrainingOps();
}

function exportManifest() {
  const params = new URLSearchParams({ status: "queued" });
  if (state.apiKey) params.set("api_key", state.apiKey);
  window.open(`/training-samples/manifest?${params.toString()}`, "_blank");
}

function trainingAudioUrl(id) {
  const params = new URLSearchParams();
  if (state.apiKey) params.set("api_key", state.apiKey);
  const query = params.toString();
  return `/training-samples/${id}/audio${query ? `?${query}` : ""}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

$("loginButton").addEventListener("click", () => {
  authenticate($("apiKeyInput").value).catch((error) => {
    $("loginMessage").textContent = `접속 실패: ${error.message}`;
  });
});

$("guestButton").addEventListener("click", () => {
  authenticate("").catch((error) => {
    $("loginMessage").textContent = `접속 실패: ${error.message}`;
  });
});

$("apiKeyInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    $("loginButton").click();
  }
});

$("logoutButton").addEventListener("click", () => {
  localStorage.removeItem("xcnTelSummaryApiKey");
  location.reload();
});

$("searchForm").addEventListener("submit", (event) => {
  event.preventDefault();
  loadCalls().catch((error) => toast(error.message));
});

$("refreshButton").addEventListener("click", () => {
  const action = state.view === "training" ? loadTrainingSamples : (state.view === "trainingOps" ? loadTrainingOps : loadCalls);
  action().catch((error) => toast(error.message));
});

$("trainingStatusFilter").addEventListener("change", (event) => {
  state.trainingStatusFilter = event.target.value;
  loadTrainingSamples().catch((error) => toast(error.message));
});

$("exportManifestButton").addEventListener("click", exportManifest);

$("deleteQueuedButton").addEventListener("click", () => {
  deleteQueuedSamples().catch((error) => toast(error.message));
});

$("startTrainingButton").addEventListener("click", () => {
  startTraining().catch((error) => toast(error.message));
});

$("wavePlayButton").addEventListener("click", () => {
  if (!state.wavesurfer) return;
  state.wavesurfer.playPause();
});

$("audioPlayer").addEventListener("timeupdate", syncActiveSegment);

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => switchView(item.dataset.view));
});

if (state.apiKey) {
  $("apiKeyInput").value = state.apiKey;
  authenticate(state.apiKey).catch(() => undefined);
} else {
  authenticate("").catch(() => undefined);
}
