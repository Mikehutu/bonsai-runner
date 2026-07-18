# 🪢 Bonsai 27B Runner

Run a 27B‑class LLM locally with **one command**.

| Variant | Footprint | Quality | Hardware |
|---|---|---|---|
| **1-bit** (Q1_0) | **3.9 GB** | 89.5% of FP16 | Any GPU with ≥6 GB VRAM |
| **Ternary** (Q2_0) | **7.2 GB** | 94.6% of FP16 | Any GPU with ≥10 GB VRAM |
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
3. ✅ Clone and build the PrismML llama.cpp fork with CUDA/Metal support
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
|---|---|---|
| `PORT` | `8080` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `NGL` | `99` | GPU layers (Metal/CUDA) |

Example:

```bash
PORT=18080 NGL=60 bash start.sh ternary
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
└── README.md         # You are here
```

Models and the llama.cpp build are cached under `~/.bonsai/` — re‑running is instant after the first download.
