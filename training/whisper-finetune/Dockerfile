FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    XDG_CACHE_HOME=/models/cache

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

COPY training/whisper-finetune/requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /workspace/requirements.txt

COPY training/whisper-finetune /workspace

RUN mkdir -p /workspace/data/raw /workspace/data/manifest /workspace/outputs /models/whisper /models/hf /models/cache

CMD ["bash"]
