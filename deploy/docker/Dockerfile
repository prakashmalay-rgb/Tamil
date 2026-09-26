# Multimodal serving image for RunPod (text/vision LLM + Whisper ASR)
#
# IMPORTANT: this stack targets NVIDIA Blackwell GPUs (RTX PRO 6000 Blackwell,
# compute capability sm_120 / CUDA 12.8+). torch 2.4.0 / vllm 0.6.1.post2
# (the "obvious" versions used by most RunPod pytorch base images) DO NOT
# support sm_120 and will hang or silently fall back to unsupported kernels.
# This image uses torch 2.7.0+cu128 and vllm 0.9.2, which were verified this
# session to detect sm_120 and run a real GPU matmul with no compatibility
# warnings. If you are deploying to older (Ampere/Hopper) GPUs, you can
# likely go back to a simpler/older stack, but this image is built and
# tested specifically for Blackwell.
#
# KNOWN UNRESOLVED ISSUE: with this stack, vllm's model loading (both
# tensor-parallel and single-GPU) was observed to hang indefinitely on a
# MIG-partitioned Blackwell pod, even though raw safetensors reads worked
# fine in isolation (~20s for a 3.5GB shard). This looks like a vllm 0.9.2
# multiprocess/IPC issue specific to MIG GPU slices, not a weights/driver
# problem. It was NOT resolved before the pod was terminated (out of
# credits). The entrypoint below defaults VLLM_USE_V1=0 (legacy engine) as
# the leading untested candidate fix -- if engines still hang on a MIG pod,
# that is the first thing to investigate further. Non-MIG (full) GPU
# allocations were not tested but are less likely to hit this.

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    VLLM_USE_V1=0

# --- System packages + Python 3.12 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl ca-certificates git ffmpeg libsm6 libxext6 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build

# --- torch 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0 (cu128, Blackwell-capable) ---
# Pulled from PyTorch's own CUDA wheel index -- these are NOT on default PyPI.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# --- vllm 0.9.2 and its dependency tree ---
# Installed with --no-deps first so pip's resolver can't silently upgrade
# torch/xformers/etc. from underneath the pinned versions above (this is
# what caused hours of version drift during manual debugging). All actual
# runtime deps are then pinned explicitly in the second step.
RUN pip install --no-cache-dir --no-deps vllm==0.9.2

RUN pip install --no-cache-dir \
        xformers==0.0.30 \
        transformers==4.53.2 \
        tokenizers==0.21.2 \
        "huggingface-hub[hf_xet]>=0.33.0,<1.0" \
        outlines==0.1.11 \
        lark==1.2.2 \
        gguf==0.13.0 \
        "setuptools<80,>=77.0.3" \
        cachetools \
        blake3 \
        "llguidance<0.8.0,>=0.7.11" \
        xgrammar==0.1.19 \
        opencv-python-headless \
        "compressed-tensors==0.10.2" \
        "depyf==0.18.0" \
        scipy \
        ninja \
        pybase64 \
        "mistral_common[opencv]>=1.6.2" \
        aiohttp regex cachetools psutil sentencepiece "numpy<2.0.0" \
        "requests>=2.26.0" tqdm py-cpuinfo protobuf \
        "fastapi[standard]>=0.115.0" "openai<=1.90.0,>=1.52.0" "pydantic>=2.10" \
        "prometheus_client>=0.18.0" pillow "prometheus-fastapi-instrumentator>=7.0.0" \
        "tiktoken>=0.6.0" "lm-format-enforcer<0.11,>=0.10.11" \
        "typing_extensions>=4.10" "filelock>=3.16.1" partial-json-parser \
        "pyzmq>=25.0.0" msgspec "importlib_metadata; python_version<'3.10'" \
        pyyaml einops cloudpickle watchfiles python-json-logger \
        "ray[cgraph]!=2.44.*,>=2.43.0"

# numba + llvmlite installed last, pinned and forced: numba 0.61.2 requires
# llvmlite>=0.44.0,<0.45, but pip's resolver (or a transitive dep from the
# packages above) tends to pull llvmlite 0.49+, which REMOVES the
# llvmlite.binding.initialize() API that this numba version still calls at
# import time -- breaking vllm's ngram_proposer module on startup. This bit
# us live; --force-reinstall --no-deps guarantees the correct pair sticks.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        numba==0.61.2 \
        llvmlite==0.44.0

# --- Whisper ASR + serving deps ---
RUN pip install --no-cache-dir \
        faster-whisper \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        requests \
        pillow \
        qwen-vl-utils

COPY start_services.sh /usr/local/bin/start_services.sh
COPY whisper_api.py /opt/services/whisper_api.py
RUN chmod +x /usr/local/bin/start_services.sh

WORKDIR /workspace

# Ports: 8000 text engine, 8002 whisper API, 8003 vision engine
EXPOSE 8000 8002 8003

ENTRYPOINT ["/usr/local/bin/start_services.sh"]
