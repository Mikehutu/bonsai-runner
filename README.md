# 🪢 Bonsai 27B Runner

Run a 27B‑class LLM locally with **one command**.

| Variant | Footprint | Quality | Hardware |
| --- | --- | --- | --- |
| **Bonsai 2** (PTQ1_0) | **6.0 GB** | 98.2% of FP16, thinking model | CPU, or any GPU with ≥8 GB VRAM |
| **Bonsai 2** (PQ2_0) | **7.2 GB** | 98.2% of FP16, thinking model | CPU, or any GPU with ≥10 GB VRAM |
| **1-bit** (Q1_0) | **3.9 GB** | 89.5% of FP16 | CPU, or any GPU with ≥6 GB VRAM |
| **Ternary** (Q2_g64) | **7.2 GB** | 94.6% of FP16 | CPU, or any GPU with ≥10 GB VRAM |
| **1-bit + DSpark** | 5.7 GB | Lossless speedup (legacy drafter) | CUDA GPU (speculative decoding) |
| **Ternary + DSpark** | 9.1 GB | Lossless speedup (legacy drafter) | CUDA GPU (speculative decoding) |

> **Bonsai 2** is the new generation (Qwen3.8-27B ternary). It needs the updated
> PrismML llama.cpp fork (`prism-b10658+`) — `start.sh` updates/re-clones it
> automatically. The old `ternary` variant now uses the `Q2_g64` file (the
> legacy `Q2_0` file is refused by post-rebase fork binaries).
>
> ⚠ **Known upstream bug** ([PrismML-Eng/llama.cpp#180](https://github.com/PrismML-Eng/llama.cpp/issues/180)):
> repacking the F32 Hadamard helper tensors segfaults the loader for **PQ2_0**
> files (PTQ1_0 is unaffected). `start.sh` therefore auto-applies the verified
> `--no-repack` workaround for the `bonsai2-pq2` variant only (slightly slower).
> Set `BONSAI2_REPACK=1` to re-enable repacking once upstream lands the fix.

No cloud API. No subscription. Your data stays on your machine.

---

## Quickstart

```bash
git clone https://github.com/Mikehutu/bonsai-runner
cd bonsai-runner

# Default: 1-bit (3.9 GB, works on most GPUs)
bash start.sh

# Bonsai 2 — the latest generation (thinking model, 262K context):
bash start.sh bonsai2          # 6.0 GB PTQ1_0 (default pick)
bash start.sh bonsai2-pq2      # 7.2 GB PQ2_0 (fastest on modern NVIDIA GPUs)

# Or pick a variant:
bash start.sh ternary          # 7.2 GB, higher quality
bash start.sh 1bit+dspark      # with speculative decoding (legacy drafter)
bash start.sh ternary+dspark   # quality + speed
```

**Tip:** Run `bash start.sh` with no arguments to auto-detect your hardware
and get a model recommendation. On interactive terminals, you'll be prompted
to choose a variant **and** a context size. On SSH/CI, it defaults to `1bit`
with the model's default context (~32K).

The script will:

1. ✅ Check prerequisites (cmake, make, python3)
2. ✅ Install `huggingface-hub` for model downloads
3. ✅ Clone and build the PrismML llama.cpp fork with CUDA/Metal/CPU support
4. ✅ Download your chosen model from HuggingFace (one‑time)
5. ✅ Start an OpenAI‑compatible server on `http://0.0.0.0:8080`

**To stop the server:** press `Ctrl+C` or run `./stop.sh` in another terminal.

**Requirements:** `cmake`, `make`, `python3`, and a C++ compiler (`build-essential` on Linux, Xcode CLI on macOS).

```bash
# Debian/Ubuntu
sudo apt install cmake build-essential python3 python3-pip

# macOS (Xcode CLI)
xcode-select --install
brew install cmake
```

---

## Configuration

See `.env.example` for all available environment variables. Copy it to `.env` and adjust:

```bash
cp .env.example .env
```

| Env var | Default | Description |
| --- | --- | --- |
| `PORT` | `8080` | Server port (auto‑fallbacks to 8081, 8082 if busy) |
| `HOST` | `0.0.0.0` | Bind address |
| `NGL` | `99` | GPU layers (Metal/CUDA — ignored on CPU) |
| `CONTEXT_SIZE` | `0` | Context length in tokens. `0`=model default (~32K). See [Context Sizing](#context-sizing) below. |
| `PARALLEL` | `0` | Server parallelism: `0`=auto, `1`=single-request (benchmark-safe) |
| `HF_HUB_OFFLINE` | (unset) | Set to `1` for airgapped machines with pre-cached models |

Example:

```bash
PORT=18080 NGL=60 bash start.sh ternary
```

### Offline / Airgapped Setup

If you cannot reach HuggingFace (airgapped network, no internet), pre-download
the models on a connected machine:

```bash
pip install huggingface-hub
hf download prism-ml/Bonsai-27B-gguf --local-dir ~/.bonsai/models/1bit
hf download prism-ml/Ternary-Bonsai-27B-gguf --local-dir ~/.bonsai/models/ternary
```

Then set `HF_HUB_OFFLINE=1` in your shell profile. The `start.sh` script
automatically unsets this variable during download attempts, so it works
whether you're online or offline — if the model is already cached, no
download is attempted.

---

## Context Sizing

Bonsai-27B supports up to **262K tokens** of context — enough to process
entire codebases, long documents, or multi-hour conversations. The default
is ~32K (model default), but you can increase it if your machine has enough
RAM.

### How Context Size Affects RAM

The KV cache grows linearly with context length. Each additional token of
context costs roughly:

| Variant | KV cache per 1K tokens |
| --- | --- |
| **1-bit (Q1_0)** | ~6 MB |
| **Ternary (Q2_0)** | ~8 MB |

**Formula:** `total RAM needed ≈ model_size + (context_k × kv_per_1k) + 1 GB overhead`

| Context | 1-bit (3.9 GB model) | Ternary (7.2 GB model) |
| --- | --- | --- |
| **32K** (default) | ~5.1 GB | ~8.5 GB |
| **64K** | ~5.5 GB | ~8.9 GB |
| **128K** | ~6.3 GB | ~9.7 GB |
| **262K** (max) | ~6.5 GB | ~10.3 GB |

### How to Set Context Size

```bash
# 262K context (tested on CPU + 48 GB DDR5)
CONTEXT_SIZE=262144 bash start.sh 1bit

# 128K context (works on 32 GB machines)
CONTEXT_SIZE=131072 bash start.sh 1bit

# 64K context (works on 16 GB machines)
CONTEXT_SIZE=65536 bash start.sh 1bit

# Model default (~32K, lowest RAM)
bash start.sh 1bit
```

### Auto-Detection

When you run `bash start.sh` with no arguments, the script estimates the
maximum context your machine can handle based on available RAM and shows
it in the hardware detection output:

```
── Model Recommendations ──
   Context estimate: ~262K tokens (based on 48000 MB RAM)
   Set CONTEXT_SIZE=268288 for max context, or
   CONTEXT_SIZE=0 for model default (~32K)
```

### What Happens If Context Is Too Large?

If you set `CONTEXT_SIZE` higher than your machine can handle:

1. **llama-server will crash on startup** with an out-of-memory error
2. The error message will mention `llama_kv_cache_init` or `bad_alloc`
3. **Fix:** Reduce `CONTEXT_SIZE` to a lower value (see table above)

The script does not pre-validate context size against available RAM because
llama.cpp's actual memory usage depends on many factors (batch size, number
of layers offloaded to GPU, concurrent requests). The formula above is a
conservative estimate — start with a lower value and increase until you find
your machine's sweet spot.

---

## What You Get

### Web UI

After `start.sh` finishes, open **`http://localhost:8080`** in your browser.
llama.cpp includes a built-in chat interface — no Docker, no extra setup.

Features:
- Chat with the model
- Upload files (PDF, images, text, code)
- Multi-turn conversation
- Copy/paste responses

### Image Input (Vision)

Bonsai-27B is a **vision-language model** — it accepts images alongside text.
The `start.sh` script automatically downloads the multimodal projector
(`mmproj`) file and passes it to `llama-server` via the `-mm` flag.

**Web UI:** Click the `+` icon in the message box to upload a photo or screenshot.

**OpenAI-compatible API:** Send `image_url` content parts to `/v1/chat/completions`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"bonsai","messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64_DATA>"}}]}],"stream":true}'
```

**Vision settings:**

| Setting | Default | Description |
|---|---|---|
| mmproj file | `*mmproj-Q8_0.gguf` | Multimodal projector (vision tower), ~0.6 GB |
| `--image-max-tokens` | `1024` | Downscales images to 1024 vision tokens for speed |
| `--image-min-tokens` | model default | Minimum vision tokens (typically 256) |

**Image token pricing:**
- Each image is encoded into vision tokens (one token per ~32×32 pixel patch)
- Bonsai-27B accepts up to ~4096 vision tokens (about a 4.2 MP image)
- `--image-max-tokens 1024` is the sweet spot for everyday photos and screenshots
- For OCR-style tasks (small text, serial numbers), use `IMAGE_MAX_TOKENS=0` to disable capping

**Performance impact:**
- The mmproj file adds ~0.6 GB to the model footprint (Q8_0) or ~0.9 GB (BF16)
- Vision tokens are prefill — a large photo adds thousands of tokens before the first word
- The prompt cache means follow-up questions about the same image are near-instant
- On CPU, expect ~16 t/s prompt processing and ~9 t/s generation with images

**Concurrency:** llama-server auto-detects `n_parallel` based on CPU cores.
On a 24-thread machine, it defaults to 4. To override, set the `PARALLEL`
env var:

```bash
# Single-request mode (benchmark-safe, lowest memory)
PARALLEL=1 bash start.sh 1bit

# 4 concurrent requests
PARALLEL=4 bash start.sh ternary
```

Each slot gets its own KV cache, so memory scales linearly with `PARALLEL`.
For CPU-only, `PARALLEL=1` minimizes memory pressure.

### OpenAI‑compatible API

The same endpoint also serves an OpenAI‑compatible API at
`http://localhost:8080/v1/chat/completions`. Works with any tool or library
that speaks the OpenAI API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

---

## Models

| Model | HF repo | Size | Notes |
| --- | --- | --- | --- |
| **Ternary-Bonsai-2-27B-PTQ1_0** | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | 6.0 GB | Bonsai 2, ternary g128 — 98.2% of FP16, thinking model (default) |
| **Ternary-Bonsai-2-27B-PQ2_0** | same repo | 7.2 GB | Bonsai 2, ternary g128 — 98.2% of FP16, fastest on modern NVIDIA |
| **Bonsai-27B-Q1_0** | [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) | 3.9 GB | 1-bit, 89.5% of FP16 |
| **Ternary-Bonsai-27B-Q2_g64** | [`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) | 7.2 GB | Ternary, 94.6% of FP16 (official group-64 format) |

No DSpark drafter is published for Bonsai 2 — the `+dspark` variants apply only
to the v1 line, and their published `*dspark-Q4_1.gguf` drafters are legacy
packings that must be converted (`gguf-dspark-to-dflash`) for post-rebase fork
binaries; `start.sh` warns when they are used.

---

## Benchmark Results

| Variant | Full Suite (79 scenarios) | Finnish (10 FI) | Speed |
|---|---|---|---|---||
| **Ternary Q2_0** | **83/100** ⭐ | **90/100** | 2.5s/turn, ~30 t/s |
| **Q1_0** | **81/100** | **85 Quality** | 5.1s/turn, 44 t/s |
| **Ternary + DSpark** | **83** | **85 Quality** | 4.9s/turn (corrected) — DSpark hurts on DGX Spark |
| **Ternary + DSpark** (old) | **77** | **85 Quality** | 16.2s/turn (contaminated — vllm backend) |

Hardware: NVIDIA GB10 (128 GB unified VRAM). Full report at [Bonsai 27B Benchmarks](https://mikehutu.github.io/AI-reports/bonsai-27b-benchmarks/).

---

## CPU‑only Performance (Minisforum AI X1 Pro / WSL)

The 1‑bit (Q1_0) variant runs on **CPU‑only hardware** with no GPU at all:

| Metric | Value |
|---|---|
| **Hardware** | Minisforum AI X1 Pro (WSL2, 48 GB DDR5, AMD Ryzen AI 9 HX 370, 24 cores) |
| **Model** | Bonsai‑27B‑Q1_0 (3.6 GB GGUF) |
| **Backend** | llama.cpp CPU (no GPU — `-ngl` ignored) |
| **Prompt ingest** | ~12.7 tok/s |
| **Generation** | ~9.0 tok/s |
| **Context** | 262K tokens |
| **RAM use** | ~4 GB model + ~1 GB runtime overhead |
| **Startup** | Build ~5 min (first time), then instant |

> CPU inference at ~9 tok/s is usable for testing, light chat, and quick code queries. For interactive use, any GPU (even an iGPU) or the DGX Spark machines will give dramatically better throughput.

---

## Why Bonsai?

Bonsai 27B is a Qwen3.6‑27B derivative with **hybrid attention** (~75% linear / ~25% full) and **aggressive low‑bit quantization** designed from the ground up for local deployment:

- **262K token context** — full‑repo code analysis, long documents
- **3.9–7.2 GB footprints** — runs on laptops and single GPUs
- **2–3× faster than equivalently quantized models** — linear‑attention backbone
- **Apache 2.0** — free to use, modify, distribute

---

## Files

```
bonsai-runner/
├── start.sh          # One‑click runner
├── stop.sh           # Stop the server
├── AGENTS.md         # Instructions for AI agents
├── README.md         # You are here
└── test-start.sh     # Validation script (not shipped to end users)
```

Models and the llama.cpp build are cached under `~/.bonsai/` — re‑running is instant after the first download.

---

## Platform Support

`start.sh` auto-detects your hardware with a single script across all platforms:

| Platform | Detection | Backend | RAM source |
|---|---|---|---|
| **Linux** (any distro) | `uname` → `"Linux"` | CPU (or CUDA if GPU found) | `/proc/meminfo` |
| **macOS Apple Silicon** (M1–M5) | `uname -m` → `"arm64"` | **Metal** (GPU accelerated) | `sysctl hw.memsize` |
| **macOS Intel** | `uname -m` → `"x86_64"` | CPU | `sysctl hw.memsize` |
| **WSL2** (Windows) | `uname` → `"Linux"` | CPU (sees Windows CUDA drivers but ignores them) | `/proc/meminfo` |

**Notes:**
- **CUDA on Linux:** detected via `nvidia-smi`. If the tool exists but reports no GPU (WSL host drivers), falls back to CPU.
- **Metal on Mac:** uses the PrismML llama.cpp fork with `-DGGML_METAL=ON`. Only Apple Silicon (arm64) gets Metal; Intel Macs use CPU.
- **No Mac to test?** The detection logic uses standard POSIX commands (`uname`, `sysctl`) that behave identically across macOS versions. The build flags (`-DGGML_METAL=ON`) come from the upstream PrismML fork. If you hit issues on macOS, [open an issue](https://github.com/Mikehutu/bonsai-runner/issues) — the `uname`/`sysctl` paths are well-tested, but Metal compile issues are upstream.
