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

## Known unresolved issue: MIG model-loading hang

Even with the correct torch/vllm versions, **vllm's model loading hung
indefinitely** on this MIG-partitioned pod -- both at `--tensor-parallel-size 2`
and `--tensor-parallel-size 1` (single GPU). Symptoms:

- Process sits in `do_poll` (blocked on an fd, not doing CPU or disk work).
- `Loading safetensors checkpoint shards: 0%` never advances.
- Raw `safetensors.torch.load_file()` on the same shard works fine in
  isolation (~20s for a 3.5GB file) -- so it's not a storage or file
  problem.
- `--enforce-eager` (disabling CUDA graph capture / torch.compile) got
  further than without it, past NCCL/P2P init, suggesting graph capture was
  *part* of the problem -- but the hang recurred at the weight-loading step
  regardless.

This was **not root-caused** before the pod was terminated (ran out of
RunPod credits mid-investigation). The entrypoint defaults to
`VLLM_USE_V1=0` (forces vllm's older, more battle-tested engine instead of
the newer async V1 engine) as the leading untested hypothesis -- the V1
engine's multiprocess IPC is newer and more likely to have rough edges on
unusual hardware (MIG slices lack full P2P between GPUs, which the logs
flagged: `Custom allreduce is disabled because your platform lacks GPU P2P
capability`).

**If engines still hang when you deploy this image on a MIG pod:**
1. Check `docker logs` / `/workspace/logs/vllm_text.log` for where it stops.
2. Try `-e VLLM_USE_V1=1` to see if the new engine behaves differently (it
   was the one observed hanging, so this mainly confirms the symptom).
3. Try requesting a **non-MIG (full GPU)** pod instead -- MIG is the prime
   suspect and was not tested without it.
4. Try `TEXT_TP=1` (single GPU) to isolate multi-GPU sync issues, though
   note single-GPU also hung in testing.

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
4. Expose ports 8000 (text), 8002 (whisper), 8003 (vision).
5. Deploy. The entrypoint starts all three engines automatically and tails
   their logs.

## Overriding models / GPU count

Set these as pod environment variables:
- `TEXT_MODEL` (default `Qwen/Qwen2.5-72B-Instruct-AWQ`)
- `VISION_MODEL` (default `Qwen/Qwen2-VL-7B-Instruct`)
- `TEXT_TP` (default `2`) -- tensor-parallel size for the text engine
- `VLLM_USE_V1` (default `0`) -- see "Known unresolved issue" above
