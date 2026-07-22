# 🪢 Bonsai 27B Runner

> [!NOTE]
> **Update (July 22, 2026): Optimized Speculative Decoding Recipe (`DSpark K=4`) & Memory Sizing**
> 
> - **Performance Impact:** Reduced single-turn interactive latency from **`16.2s` down to `5.73s` per turn** (**2.8× faster turn latency**) while recovering full multi-turn benchmark quality (**82/100 score** across 79 scenarios, 0 timeouts).
> - **Key Technical Edits:**
>   1. **Capped Speculative Depth (`--spec-draft-n-max 4`)**: Replaced uncapped draft depth to eliminate sequential draft generation stalls.
>   2. **Context & Kernel Alignment (`-c 4096 -fa auto`)**: Matched draft model context and enabled Flash Attention.
>   3. **Flexible Hardware Sizing**: Added environment controls (`NP`, `CTX_SIZE`, `CACHE_RAM`) allowing peak memory to scale from **~5.2 GB** (12GB GPUs / laptops) to **~19 GB** (multi-user servers) and **~5.5 GB** (pure CPU mode).

Run a 27B‑class LLM locally with **one command**.

| Variant | Footprint | Quality | Hardware |
|---|---|---|---|
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

## Configuration & Hardware Tuning

| Env Var | Default | Description |
|---|---|---|
| `PORT` | `8080` | Server port (auto‑fallbacks to 8081, 8082 if busy) |
| `HOST` | `0.0.0.0` | Bind address |
| `NGL` | `99` | GPU layers (Metal/CUDA — set `NGL=0` for pure CPU) |
| `NP` | `4` | Concurrent slots (set `NP=1` for single-user low memory) |
| `CTX_SIZE` | `4096` | Active context length per slot (up to 262,144) |
| `CACHE_RAM` | `8192` | Host RAM reserved for prompt caching in MB |

---

## 💾 Memory & VRAM Sizing Guide (12 GB to 128 GB+)

Why does Bonsai use ~19 GB VRAM in default server mode?
In `llama-server`, memory consists of **Model Weights + Multi-Slot KV Cache + Prompt Cache**:
1. **Model Weights**: ~3.9 GB (Q1_0) or ~7.2 GB (Q2_0) + ~1.8 GB for DSpark.
2. **KV Cache & Slots**: `llama-server` defaults to 4 parallel slots (`NP=4`). Each slot allocates a KV cache buffer for concurrent conversations.
3. **Prompt Cache & Graphs**: Up to 8 GB host RAM reserved for context caching and CUDA compute graphs.

### Recommended Settings by Hardware Tier

| Hardware / VRAM Tier | Recommended Command | Peak Memory | Notes |
|---|---|---|---|
| **12 GB VRAM / RAM**<br>*(RTX 3060/4060 12GB, 16GB Macs)* | `NP=1 CTX_SIZE=4096 CACHE_RAM=0 bash start.sh 1bit` | **~5.2 GB** | Fits comfortably on 12 GB GPUs. |
| **16 GB – 24 GB VRAM**<br>*(RTX 3090/4090 24GB, Apple M-series)* | `NP=1 CTX_SIZE=8192 bash start.sh ternary+dspark` | **~10.5 GB** | Single-user workhorse setup with DSpark speedup. |
| **32 GB – 48 GB RAM**<br>*(CPU-Only Workstation / Minisforum)* | `NGL=0 NP=1 CTX_SIZE=8192 bash start.sh 1bit` | **~5.5 GB** | Pure CPU mode (`NGL=0`). ~9.0 tok/s generation. |
| **64 GB – 128 GB VRAM**<br>*(NVIDIA GB10 / DGX Spark / Server)* | `NP=4 CTX_SIZE=16384 bash start.sh ternary+dspark` | **~19.0 GB** | Full multi-user concurrent server mode. |

Example for 12GB GPU:

```bash
NP=1 CTX_SIZE=4096 CACHE_RAM=0 bash start.sh 1bit
```

Example for CPU-only (no GPU):

```bash
NGL=0 NP=1 CTX_SIZE=8192 bash start.sh 1bit
```

---

## What You Get

An **OpenAI‑compatible API** endpoint at `http://localhost:8080/v1/chat/completions`. Works with any tool or library that speaks the OpenAI API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

---

## Models

| Model | HF repo | Size | Notes |
|---|---|---|---|
| **Bonsai-27B-Q1_0** | [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) | 3.9 GB | 1-bit, 89.5% of FP16 |
| **Ternary-Bonsai-27B-Q2_0** | [`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) | 7.2 GB | Ternary, 94.6% of FP16 |

Both ship with optional DSpark speculative‑decoding drafters for ~1.35× CUDA decode speedup.

---

## Benchmark Results

| Variant | Full Suite (79 scenarios) | Finnish (10 FI) | Speed |
|---|---|---|---|---|
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
└── README.md         # You are here
```

Models and the llama.cpp build are cached under `~/.bonsai/` — re‑running is instant after the first download.
