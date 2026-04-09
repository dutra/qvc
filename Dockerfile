FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    JAX_PLATFORM_NAME=cpu \
    NUM_CORES=4 \
    QVC_CODE_DIR=/opt/qvc \
    QVC_WORKDIR=/work/qvc-demo \
    QVC_DATA_DIR=/work/qvc-demo/data \
    QVC_RESULT_DIR=/work/qvc-demo/results \
    QVC_DUSTMAPS_DIR=/work/qvc-demo/.dustmaps

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    fonts-dejavu-core \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qvc

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -e .

RUN mkdir -p /work/qvc-demo && \
    chmod +x /opt/qvc/scripts/run_demo.sh /opt/qvc/scripts/docker_entrypoint.sh

WORKDIR /work/qvc-demo

VOLUME ["/work/qvc-demo"]

ENTRYPOINT ["/opt/qvc/scripts/docker_entrypoint.sh"]
CMD ["all"]
