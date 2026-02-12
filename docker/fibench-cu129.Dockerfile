FROM lmsysorg/sglang:dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-setuptools \
    gcc \
    zlib1g-dev \
    build-essential \
    cmake \
    libedit-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 -y && \
    pip install --no-cache-dir nvidia-cutlass-dsl && \
    git clone --recursive https://github.com/tile-ai/tilelang.git /sgl-workspace/tilelang && cd /sgl-workspace/tilelang && \
    sed -i 's/set(USE_GTEST AUTO)/set(USE_GTEST OFF)/g' 3rdparty/tvm/cmake/config.cmake && \
    pip install --no-cache-dir -e . -v
