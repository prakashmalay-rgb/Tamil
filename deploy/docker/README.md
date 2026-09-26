# Multimodal RunPod image (text + vision + Whisper)

Built from what was actually verified working in a live debugging session on
a RunPod pod with dual **NVIDIA RTX PRO 6000 Blackwell** GPUs (MIG-partitioned,
`2g.48gb` slices), compute capability **sm_120**.

## Why this stack, not the "obvious" one

The straightforward choice -- a RunPod `pytorch:2.4.0-cu121` base image with
`vllm==0.6.1.post2` -- **does not work on Blackwell**. torch 2.4.0 only
supports up to `sm_90`; it silently falls back to an unsupported-kernel
warning path instead of erroring, so install scripts "succeed" but the
engine hangs or never serves. This was the root cause of a multi-hour stall
in the original setup session.

This image instead uses:
- **torch 2.7.0+cu128**, **torchvision 0.22.0+cu128**, **torchaudio 2.7.0+cu128**
  (from PyTorch's own `cu128` wheel index) -- confirmed via
  `torch.cuda.get_arch_list()` to include `sm_120`, and confirmed via a real
  `4096x4096` GPU matmul with zero compatibility warnings.
- **vllm 0.9.2** (the version that pins `torch==2.7.0`), plus the full
  corrected dependency tree (see Dockerfile comments for specifics like
  `llvmlite==0.44.0`, which is required -- newer llvmlite removes an API
  `numba` 0.61.2 still calls, breaking vllm's speculative-decoding module
  at import time).
- vllm is installed with `--no-deps` first, then its actual runtime
  dependencies are pinned explicitly by hand, and `numba`/`llvmlite` are
  force-reinstalled last as an exact pair. This is deliberate: letting
  pip's resolver run free across this whole set is what caused hours of
  version drift (pip happily "upgrading" transformers, tokenizers,
  llvmlite, etc. to incompatible latest releases) during manual debugging
  on the original pod.

## Confirmed working: non-MIG single GPU (A100 80GB)

Deployed and verified end-to-end on a **1x A100 SXM 80GB** (non-MIG) pod:
`--enforce-eager` + `VLLM_USE_V1=0` + `TEXT_TP=1` loaded the text engine
successfully and served real completions via `/v1/chat/completions`. This
is the known-good configuration for a single full GPU.

## Known unresolved issue: MIG model-loading hang

On an earlier **MIG-partitioned** Blackwell pod (`2g.48gb` slices), vllm's
model loading hung indefinitely -- both at `--tensor-parallel-size 2` and
`--tensor-parallel-size 1`. Symptoms:

- Process sits in `do_poll` (blocked on an fd, not doing CPU or disk work).
- `Loading safetensors checkpoint shards: 0%` never advances.
- Raw `safetensors.torch.load_file()` on the same shard works fine in
  isolation (~20s for a 3.5GB file) -- so it's not a storage or file
  problem.

This was **not root-caused** and the MIG pod was never retested after the
fix (it was terminated for unrelated reasons -- out of RunPod credits).
Given the non-MIG pod above worked cleanly with the same flags, **MIG
GPU partitioning remains the prime suspect** (MIG slices lack full P2P
between GPUs, which the logs flagged: `Custom allreduce is disabled
because your platform lacks GPU P2P capability`).

**If engines hang on a MIG pod:**
1. Check `docker logs` / `/workspace/logs/vllm_text.log` for where it stops.
2. Confirm `--enforce-eager` and `VLLM_USE_V1=0` are in effect (both are
   defaults in `start_services.sh` now).
3. If it still hangs, request a **non-MIG (full GPU)** pod instead --
   this is the configuration actually confirmed working.

## Known unresolved issue: 3 engines don't fit on one GPU

Running the text engine (Qwen2.5-72B-AWQ) and vision engine (Qwen2-VL-7B)
simultaneously on a single 80GB GPU crashed the vision engine with
`ValueError: No available memory for the cache blocks`, even at
`--gpu-memory-utilization 0.20` for vision (after text had already claimed
0.65) and with `--mm-processor-kwargs '{"max_pixels": 1003520}'` capping
image resolution to reduce vision's worst-case token reservation. Each
vLLM process profiles/reserves GPU memory independently without knowing
about the other engine's reservation, and Qwen2-VL's own worst-case
multimodal-token profiling (up to 32,768 tokens per image) makes the
budget tighter than the raw weight sizes (39GB + 15.5GB) suggest.

**Current state**: `start_services.sh` starts text + whisper only. Vision
is not auto-started. To use vision, you need either a second GPU (so it
gets its own full memory budget) or a smaller vision model. If you retest
this, don't assume retuning `--gpu-memory-utilization` alone will fix it --
0.20, 0.35, and pixel-capping were all tried and failed identically.

## Build & push

A GitHub Actions workflow (`.github/workflows/docker-publish.yml`) builds and
pushes this image to GHCR automatically on every push to `main` that touches
`deploy/docker/**`. It publishes to:

```
ghcr.io/prakashmalay-rgb/tamil:latest
```

(GHCR lowercases the repo path -- `Tamil` becomes `tamil`.) No manual build
step needed; just push changes to this folder.

To build/push manually instead:

```bash
docker build -t ghcr.io/prakashmalay-rgb/tamil:latest .
docker push ghcr.io/prakashmalay-rgb/tamil:latest
```

The published package is private by default (inherits the repo's visibility).
If RunPod needs to pull it, either make the package public (GitHub -> your
profile -> Packages -> tamil -> Package settings -> Change visibility), or
give RunPod a registry credential (Personal Access Token with `read:packages`)
in the pod template's registry auth settings.

## Deploy on RunPod

1. RunPod dashboard -> Templates -> New Template.
2. Image: `ghcr.io/prakashmalay-rgb/tamil:latest`.
3. Attach your existing network volume at `/workspace` (this is where model
   weights and the HF cache must live -- they are **not** baked into the
   image; pre-download them once with `huggingface-cli download <model>
   --local-dir /workspace/.cache/huggingface/...` or let the first launch
   pull them, which will be slow).
4. Expose ports 8000 (text), 8002 (whisper), and 8003 (vision, only if
   you set `ENABLE_VISION=1` -- see below).
5. Deploy. The entrypoint starts the text engine and Whisper automatically
   and tails their logs. Vision does not start by default (see "Known
   unresolved issue: 3 engines don't fit on one GPU" above).

## Overriding models / GPU count

Set these as pod environment variables:
- `TEXT_MODEL` (default `Qwen/Qwen2.5-72B-Instruct-AWQ`)
- `VISION_MODEL` (default `Qwen/Qwen2-VL-7B-Instruct`)
- `TEXT_TP` (default `2`) -- tensor-parallel size for the text engine. Use
  `1` on a single-GPU pod.
- `VLLM_USE_V1` (default `0`) -- see "Known unresolved issue" above
- `ENABLE_VISION` (default `0`) -- set to `1` to also start the vision
  engine on port 8003. Only do this on a pod with a GPU dedicated to
  vision alone; it is not safe to co-locate with the text engine (see
  above).
