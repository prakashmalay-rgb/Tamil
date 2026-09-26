# Tamil LLM & GPT Studio — architecture and rules for AI agents

This file orients any AI agent (Claude Code or otherwise) picking up work on
this repo. Read it before touching the backend, the frontend, or the Docker
image — several of the rules below exist because skipping them cost hours of
real debugging time.

## Architecture

Three independent systems, each with its own deploy path:

1. **RunPod GPU backend** (`deploy/docker/`) — a Docker image
   (`ghcr.io/prakashmalay-rgb/tamil:latest`) running vLLM OpenAI-compatible
   servers for a text model (Qwen2.5-72B-Instruct-AWQ, port 8000), a vision
   model (Qwen2-VL-7B-Instruct, port 8003, currently **not working** — see
   Known Issues), and Whisper ASR (faster-whisper large-v3, port 8002).
   Model weights and the HuggingFace cache live on a RunPod **persistent
   network volume** mounted at `/workspace`, not baked into the image.

2. **Vercel frontend + API** (`server.py` at repo root, `public/`/`frontend/`
   static assets) — serves the chat UI and a `/chat` + `/api/chat` endpoint
   that forwards to the RunPod text engine's `/v1/chat/completions`.

3. **GitHub Actions CI** (`.github/workflows/docker-publish.yml`) — builds
   and pushes the Docker image to GHCR automatically on every push to `main`
   that touches `deploy/docker/**`.

### Request flow (working, verified 2026-09-26)

```
Browser -> https://tamil-nu.vercel.app/chat
        -> server.py's /chat handler (Vercel serverless function)
        -> httpx POST to RUNPOD_TEXT_URL/v1/chat/completions
        -> vLLM on the RunPod pod -> Qwen2.5-72B-Instruct-AWQ
        -> response bubbles back up
```

`RUNPOD_TEXT_URL` is a Vercel **environment variable**
(`https://<pod-id>-8000.proxy.runpod.net`) — it changes every time the pod is
redeployed and must be updated in Vercel's project settings, not in code.

## Automatic vs. manual updates

- **Docker image**: auto-builds and pushes to GHCR on every push to `main`
  touching `deploy/docker/**`. No manual build step needed.
- **Vercel frontend**: auto-deploys on every push to `main`. No manual step.
- **RunPod pod**: does **not** auto-update. A running pod keeps using
  whatever image it pulled at creation — there is no live image swap.
  After any `deploy/docker/**` change lands and the new image finishes
  building, the human operator must manually stop/restart the pod (or
  deploy a fresh one) to pick it up. Always say so explicitly rather than
  assuming it happened.

## Hard-won gotchas (read before debugging any of these symptoms)

### 1. Blackwell GPUs need torch 2.7+, not the "obvious" 2.4.0
RunPod pods with **RTX PRO 6000 Blackwell** GPUs (compute capability
`sm_120`) are not supported by torch 2.4.0 (max `sm_90`), which is what most
RunPod PyTorch base images ship. Symptom: install "succeeds" but vLLM hangs
silently during model load with no error, or emits a `CUDA capability
sm_120 is not compatible` warning and then hangs anyway. Fix: torch
2.7.0+cu128 (from `download.pytorch.org/whl/cu128`, not default PyPI) +
vllm 0.9.2, with dependency versions pinned exactly as in
`deploy/docker/Dockerfile` (see its own comments for the specific
`llvmlite`/`numba` and `llguidance` version traps).

### 2. MIG-partitioned GPUs may hang vLLM even with correct torch/vllm
A Blackwell pod with MIG-partitioned GPU slices (`2g.48gb` style) hung
during model loading even after the torch fix, with zero GPU utilization
and zero log progress. Never fully root-caused; `--enforce-eager` and
`VLLM_USE_V1=0` (legacy engine) were the working mitigations found. A
**non-MIG, single full GPU (e.g. A100 80GB)** pod worked cleanly with the
same image and no special flags beyond `--enforce-eager`/`VLLM_USE_V1=0`,
which are now the defaults in `start_services.sh`. If you hit an
unexplained hang on a new pod, check whether it's MIG-partitioned first.

### 3. Multiple vLLM engines sharing one GPU fight over `gpu_memory_utilization`
Running text + vision engines simultaneously on a single 80GB GPU crashed
both with `ValueError: No available memory for the cache blocks` even when
their `--gpu-memory-utilization` fractions summed to under 1.0. Each
engine profiles/reserves memory independently without knowing about the
other's reservation. Qwen2-VL-7B's vision-token worst-case profiling
(reserves space for up to 32,768 tokens per image regardless of
`--limit-mm-per-prompt`) makes this worse. **Currently unresolved** — the
vision engine (port 8003) is disabled by not being auto-started; only text
+ whisper are confirmed stable together on a single GPU. Don't assume
retuning the fraction alone will fix it — it was tried at 0.20, 0.35, and
with `--mm-processor-kwargs max_pixels` capping, all failed the same way.

### 4. HuggingFace cache symlinks can silently corrupt into 0-byte files
After a pod crash/restart cycle, nearly every small metadata file
(`config.json`, `tokenizer.json`, etc.) under
`/workspace/.cache/huggingface/hub/*/snapshots/*/` became a 0-byte file,
while the actual weight data in `blobs/` stayed intact (symlinks pointing
to real blob hashes just went missing/empty). Symptom:
`OSError: ... is not a valid JSON file` or `RuntimeError: File model.bin is
incomplete`. This is **not** a re-download situation — `du -sh
.../blobs/` will show the real weights are still there. Fix: delete the
zero-byte files (`find ... -type f -size 0 -not -path '*/.no_exist/*'
-delete`) and re-run `huggingface_hub.snapshot_download(repo_id,
cache_dir=".../hub")` for the affected repo — it detects the existing
blobs by hash and only re-links the missing small files in seconds,
without re-downloading gigabytes of weights. If a large file (like
Whisper's `model.bin`) is corrupted mid-content rather than empty, delete
that specific blob file and re-run `snapshot_download` to force a real
re-fetch of just that blob.

### 5. Never trust a single frontend file — check what's actually being served
This repo has (or had) **three different frontend/backend entrypoint
candidates**: a root-level `index.html` + `js/api.js`, `public/index.html`
+ `public/js/api.js`, and `frontend/index.html` + `frontend/js/api.js`,
plus (until removed) both `server.py` (root) and `api/index.py` defining
the *same* `/chat` route. `vercel.json`'s `rewrites` pointed at
`api/index.py`, but `server.py` was the one actually running in
production — editing `api/index.py` alone had zero effect on the live
site for a while. **`api/index.py` has since been deleted**; `server.py`
is now the sole entrypoint. If similar duplication reappears, don't assume
which file Vercel is serving — fetch the live JS/HTML directly
(`curl https://<domain>/js/api.js`) and diff it against the repo before
editing anything.

### 6. The frontend's backend URL lives in browser localStorage, not code
`public/js/api.js` / `frontend/js/api.js` read the backend URL from
`localStorage.getItem("kaggle_api_url")`, defaulting to
`http://localhost:8000` if unset. This caused a long, confusing debugging
loop: a stale/reset localStorage value pointed the browser at
`localhost:8000`, which happened to have an **old local `python
server.py` process still running** on the user's own machine from a
previous session — so the browser got real (but wrong, outdated) error
responses that looked exactly like a live backend failure, while the
actual Vercel/RunPod pipeline was working correctly the whole time
(verified independently via `curl` to the production URL, which returned
correct results). **Rule: if a browser-reported error doesn't match what a
direct `curl` to the same endpoint produces, suspect a client-side
override (localStorage, cached JS, or a stray local process on the same
port) before re-debugging the server side.** Check
`Settings & API Keys -> Kaggle FastAPI Backend URL` in the UI, and check
`netstat`/`Get-NetTCPConnection` for local processes squatting on the
port the frontend defaults to.

## Rules for future work in this repo

1. **Verify, don't assume.** Every "it's fixed" claim in this session that
   turned out wrong was based on reading code or logs without directly
   testing the actual live path (a `curl` to the real URL, a `tail` of the
   real log, a check of what's really listening on a port). Before
   declaring something fixed, hit the actual endpoint and read the actual
   response or log line.
2. **Distinguish "slow" from "stalled" with evidence, not elapsed time
   alone.** Several multi-minute waits during vLLM startup or Docker builds
   looked like hangs but were genuinely progressing (confirmed via
   `/proc/<pid>/io` read_bytes deltas, or CPU time deltas across two
   `ps` samples). Check for forward progress before concluding something
   is broken and killing it.
3. **When editing `deploy/docker/**`, remember the image won't reach
   RunPod automatically.** Say so explicitly: "image rebuilt, restart the
   pod to pick it up" — don't imply it's already live on the running pod.
4. **Pin dependency versions explicitly and install with `--no-deps`
   first** when building the Docker image or fixing a broken Python
   environment. Letting pip's resolver run free across a large dependency
   set repeatedly caused version drift (pulling in incompatible "latest"
   releases of `transformers`, `llvmlite`, `huggingface-hub`, etc.) during
   manual debugging on the original pod.
5. **Treat RunPod's SSH proxy (`ssh.runpod.io`) as unreliable for
   long-running or interactive commands** — it drops mid-command
   unpredictably. Prefer the direct TCP SSH endpoint
   (`ssh root@<ip> -p <port>`) shown on the pod's Connect tab when
   available, and always re-verify state after a dropped connection rather
   than assuming a command didn't run.
