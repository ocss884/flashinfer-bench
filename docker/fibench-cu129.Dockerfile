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
    
# install .vscode-server and extensions
ARG VSCODE_COMMIT=b6a47e94e326b5c209d118cf0f994d6065585705
ARG VSCODE_SERVER_DIR=/root/.vscode-server

RUN mkdir -p ${VSCODE_SERVER_DIR}/bin/${VSCODE_COMMIT} \
 && curl -fsSL \
    "https://update.code.visualstudio.com/commit:${VSCODE_COMMIT}/server-linux-x64/stable" \
    -o /tmp/vscode-server.tar.gz \
 && tar -xzf /tmp/vscode-server.tar.gz -C ${VSCODE_SERVER_DIR}/bin/${VSCODE_COMMIT} --strip-components=1 \
 && rm /tmp/vscode-server.tar.gz \
 && ${VSCODE_SERVER_DIR}/bin/${VSCODE_COMMIT}/bin/code-server \
    --force \
    --install-extension ms-python.python \
    --install-extension ms-python.vscode-pylance \
    --install-extension ms-python.debugpy \
    --install-extension ms-python.vscode-python-envs \
    --install-extension charliermarsh.ruff \
    --install-extension ms-toolsai.jupyter \
    --install-extension ms-toolsai.vscode-jupyter-cell-tags \
    --install-extension ms-toolsai.vscode-jupyter-slideshow \
    --install-extension ms-toolsai.jupyter-keymap \
    --install-extension ms-toolsai.jupyter-renderers \
    --install-extension ms-vscode.cpptools \
    --install-extension ms-vscode.makefile-tools \
    --install-extension ms-vscode.cmake-tools \
    --install-extension eamodio.gitlens \
    --install-extension github.vscode-pull-request-github \
    --install-extension yzhang.markdown-all-in-one \
    --install-extension tamasfe.even-better-toml \
    --install-extension tht13.rst-vscode \
    --install-extension drain99.perfetto-trace \
    --install-extension openai.chatgpt \
    --install-extension github.copilot-chat

RUN git clone https://github.com/ocss884/flashinfer-bench.git /sgl-workspace/flashinfer-bench && \
    git clone https://github.com/flashinfer-ai/flashinfer.git

# Set workspace directory
WORKDIR /sgl-workspace/flashinfer-bench
