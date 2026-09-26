#!/bin/bash
# Entrypoint: starts text engine (8000), vision engine (8003), whisper API (8002).
#
# Assumes model weights live on the attached RunPod network volume at
# /workspace/.cache/huggingface (HF_HOME) -- NOT baked into this image.
# Pre-download weights once with `huggingface-cli download <model>` into
# that path so containers start without needing to fetch 30-40GB each time.
set -e

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# Leading untested candidate fix for the MIG/model-loading hang described in
# the Dockerfile header. Override with `-e VLLM_USE_V1=1` at pod launch to
# test the new engine instead.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
VISION_MODEL="${VISION_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
TEXT_TP="${TEXT_TP:-2}"

mkdir -p /workspace/logs

echo "=== Starting text engine (${TEXT_MODEL}, tp=${TEXT_TP}) on :8000 ==="
# --enforce-eager: confirmed this session to avoid a CUDA-graph-capture
# stall on Blackwell/MIG. Drop it once that's root-caused for better
# throughput, but it is the known-safe starting point.
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${TEXT_MODEL}" \
    --tensor-parallel-size "${TEXT_TP}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.55 \
    --enforce-eager \
    --port 8000 \
    > /workspace/logs/vllm_text.log 2>&1 &

echo "=== Starting vision engine (${VISION_MODEL}) on :8003 ==="
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${VISION_MODEL}" \
    --port 8003 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.35 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    > /workspace/logs/vllm_vision.log 2>&1 &

echo "=== Starting Whisper ASR API on :8002 ==="
nohup python3 /opt/services/whisper_api.py \
    > /workspace/logs/whisper.log 2>&1 &

echo "All services launched. Tailing logs (ctrl-c to stop tailing, services keep running)."
tail -f /workspace/logs/vllm_text.log /workspace/logs/vllm_vision.log /workspace/logs/whisper.log
