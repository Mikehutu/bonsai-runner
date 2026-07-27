# 🪢 Bonsai 27B Runner

Run a 27B‑class LLM locally with **one command**.

| Variant | Footprint | Quality | Hardware |
| --- | --- | --- | --- |
| **1-bit** (Q1_0) | **3.9 GB** | 89.5% of FP16 | CPU, or any GPU with ≥6 GB VRAM |
| **Ternary** (Q2_0) | **7.2 GB** | 94.6% of FP16 | CPU, or any GPU with ≥10 GB VRAM |
| **1-bit + DSpark** | 5.7 GB | Lossless speedup | CUDA GPU (speculative decoding) |
| **Ternary + DSpark** | 9.1 GB | Lossless speedup | CUDA GPU (speculative decoding) |

No cloud API. No subscription. Your data stays on your machine.

---

## Quickstart

```bash
git clone https://github.com/Mikehutu/bonsai-runner
cd bonsai-runner

# Default: 1-bit (3.9 GB, works on most GPUs)
bash start.sh

# Or pick a variant:
bash start.sh ternary          # 7.2 GB, higher quality
bash start.sh 1bit+dspark      # with speculative decoding
bash start.sh ternary+dspark   # quality + speed
```

**Tip:** Run `bash start.sh` with no arguments to auto-detect your hardware
and get a model recommendation. On interactive terminals, you'll be prompted
to choose. On SSH/CI, it defaults to `1bit`.

The script will:

1. ✅ Check prerequisites (cmake, make, python3)
2. ✅ Install `huggingface-hub` for model downloads
3. ✅ Clone and build the PrismML llama.cpp fork with CUDA/Metal/CPU support
4. ✅ Download your chosen model from HuggingFace (one‑time)
5. ✅ Start an OpenAI‑compatible server on `http://0.0.0.0:8080`

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

| Env var | Default | Description |
| --- | --- | --- |
| `PORT` | `8080` | Server port (auto‑fallbacks to 8081, 8082 if busy) |
| `HOST` | `0.0.0.0` | Bind address |
| `NGL` | `99` | GPU layers (Metal/CUDA — ignored on CPU) |

Example:

```bash
PORT=18080 NGL=60 bash start.sh ternary
```

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
On a 24-thread machine, it defaults to 4. To override, add `--parallel N`
to the `exec` line in `start.sh`. Each slot gets its own KV cache, so memory
scales linearly. For CPU-only, `--parallel 1` minimizes memory pressure.

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
| **Bonsai-27B-Q1_0** | [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) | 3.9 GB | 1-bit, 89.5% of FP16 |
| **Ternary-Bonsai-27B-Q2_0** | [`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) | 7.2 GB | Ternary, 94.6% of FP16 |

Both ship with optional DSpark speculative‑decoding drafters for ~1.35× CUDA decode speedup.

---

## Benchmark Results

| Variant | Full Suite (79 scenarios) | Finnish (10 FI) | Speed |
|---|---|---|---|---||
| **Ternary Q2_0** | **85/100** ⭐ | **90/100** | 5.7 s/turn, 30 t/s |
| **Q1_0** | **81/100** | **85 Quality** | 5.1 s/turn, 44 t/s |
| **Ternary + DSpark** | **77** | **85 Quality** | 16.2 s/turn (batch: 1.49× throughput) |

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
├── start.sh          # One‑click runner (this script)
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
