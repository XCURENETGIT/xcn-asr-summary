FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    XDG_CACHE_HOME=/models/cache \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

COPY app /app/app
COPY db /app/db
COPY README.md /app/README.md
COPY entrypoint.sh /app/entrypoint.sh

RUN gcc -shared -fPIC -O3 \
    -o /app/app/native/libxcnaria.so \
    /app/app/native/xcn_aria_native.c

RUN mkdir -p \
    /app/data/uploads \
    /app/data/training-clips \
    /app/data/voice \
    /app/data/voice_finish \
    /app/data/translate \
    /models/hf \
    /models/cache
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
