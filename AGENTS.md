# AGENTS.md — Bonsai 27B Runner

This file provides instructions for AI agents (Claude Code, Codex, OpenCode, etc.)
to help users set up, run, and modify the Bonsai 27B Runner.

## Project Overview

**bonsai-runner** is a one-click local runner for Bonsai 27B — a 27B-parameter
Qwen3.6 derivative with hybrid attention and aggressive low-bit quantization.

- **1-bit (Q1_0):** 3.9 GB, 89.5% of FP16 quality
- **Ternary (Q2_0):** 7.2 GB, 94.6% of FP16 quality
- **DSpark variants:** add speculative decoding for ~1.35× CUDA speedup

The `start.sh` script auto-builds the PrismML llama.cpp fork, downloads GGUF
weights from HuggingFace, and starts an OpenAI-compatible API server with a
built-in web UI.

## Project Structure

```
bonsai-runner/
├── start.sh              # One-click runner — builds, downloads, serves
├── AGENTS.md             # This file
├── README.md             # User-facing documentation
└── test-start.sh         # Validation script (not shipped to end users)
```

**Cache locations (not in repo):**
- `~/.bonsai/llama.cpp/` — PrismML llama.cpp build
- `~/.bonsai/models/<variant>/` — downloaded GGUF model files
- `~/.bonsai/venv-hf/` — huggingface-hub venv (fallback install)

## Conventions

1. **No machine-specific paths.** The `start.sh` script must work on any Linux/macOS
   machine. Never hardcode `/mnt/ugreen`, `gx10`, `mikehutu`, or any other
   machine-specific path. Use `~/.bonsai/` for all cache.

2. **Strict bash mode.** `start.sh` uses `set -euo pipefail`. All scripts in the
   repo should follow the same pattern.

3. **Generic cache paths.** When checking for existing model files, check these
   locations in order:
   - `~/.bonsai/models/<variant>/<file>` (script's own cache)
   - `~/.cache/huggingface/hub/<file>` (HF Hub standard cache)
   - `~/models/<file>` (user's manual models folder)
   - `./models/<file>` (repo-local models folder)

4. **Port fallback.** If the default port (8080) is busy, try 8081, then 8082.

5. **Backend detection.** `start.sh` auto-detects CUDA (`nvidia-smi`), Metal
   (Apple Silicon arm64), and CPU. **One script, all platforms.** See the
   Platform Support section in the README for details.

## Variant System

The `start.sh` script uses a `MODELS` associative array to define variants:

```bash
declare -A MODELS
MODELS[1bit]="prism-ml/Bonsai-27B-gguf|Bonsai-27B-Q1_0.gguf|Bonsai-27B-dspark-Q4_1.gguf|--spec-type draft-dspark --spec-draft-n-max 4|Bonsai-27B-mmproj-Q8_0.gguf"
MODELS[ternary]="prism-ml/Ternary-Bonsai-27B-gguf|Ternary-Bonsai-27B-Q2_0.gguf|Ternary-Bonsai-27B-dspark-Q4_1.gguf|--spec-type draft-dspark --spec-draft-n-max 4|Ternary-Bonsai-27B-mmproj-Q8_0.gguf"
```

Each entry is pipe-separated: `HF_REPO | MODEL_FILE | DSPARK_FILE | DSPARK_ARGS | MMPROJ_FILE`

Pipe-separation is used because `DSPARK_ARGS` contains spaces (e.g.
`--spec-type draft-dspark --spec-draft-n-max 4`). The `MMPROJ_FILE` is the
multimodal projector (vision tower) — it is downloaded from the same HF repo
and passed to `llama-server` via the `-mm` flag. This enables image input.
The projector is loaded only when an image arrives, so text-only inference is
unaffected.

**To add a new variant:**
1. Add a new entry to the `MODELS` array
2. Add a case branch in the variant parser
3. Add a help text line
4. Update the README variant table

### Finding the mmproj file for a new model

Bonsai-27B is a vision-language model. The mmproj file (multimodal projector)
is what enables image input. To find the correct mmproj filename for a model
variant, list the files in the HuggingFace repo:

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('prism-ml/Bonsai-27B-gguf')
for f in files:
    if 'mmproj' in f:
        print(f)
"
```

This will output something like:
```
Bonsai-27B-mmproj-BF16.gguf
Bonsai-27B-mmproj-Q8_0.gguf
```

Use the Q8_0 variant (smaller, faster) unless you need maximum precision.

### Vision / Image Settings

The `start.sh` script passes these flags to `llama-server` for vision support:

| Flag | Value | Purpose |
|---|---|---|
| `-mm` | `<path-to-mmproj>` | Loads the multimodal projector (vision tower) |
| `--image-max-tokens` | `1024` | Downscales images to 1024 vision tokens for speed |
| `--image-min-tokens` | *(not set)* | Uses model default (typically 256) |

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

### Concurrency (`n_parallel`)

llama-server auto-detects `n_parallel` based on CPU cores. On a 24-thread
machine, it defaults to 4 (n_cores / 2, capped). This means 4 concurrent
requests can be processed.

The `start.sh` script exposes this via the `PARALLEL` env var:

```bash
# Default: auto (llama-server picks based on CPU cores)
bash start.sh ternary

# Single-request mode (benchmark-safe, lower memory)
PARALLEL=1 bash start.sh ternary

# Explicit concurrency
PARALLEL=4 bash start.sh ternary
```

Each parallel slot gets its own copy of the KV cache, so memory usage scales
linearly with `n_parallel`. For CPU-only inference, `PARALLEL=1` is
recommended to minimize memory pressure.

**Benchmark note:** Always use `PARALLEL=1` when running tool-eval-bench or
other single-turn benchmarks. The default auto-detection may pick `n_parallel=4`,
which contaminates latency measurements.

### Model Variant Quick Reference

| Variant | Model File | mmproj File | DSPark | Size | Quality |
|---|---|---|---|---|---|
| `1bit` | `Bonsai-27B-Q1_0.gguf` | `Bonsai-27B-mmproj-Q8_0.gguf` | optional | 3.9 GB | 89.5% FP16 |
| `ternary` | `Ternary-Bonsai-27B-Q2_0.gguf` | `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` | optional | 7.2 GB | 94.6% FP16 |

Both variants share the same vision tower format (Q8_0 mmproj). The ternary
variant has higher image quality due to better overall model precision.

## Testing

Run the validation suite:

```bash
bash test-start.sh
```

This checks:
- Help text lists all variants
- Error handling for bad variants
- Model config resolution (HF repo, file names, dspark args)
- File structure (start.sh, README.md, AGENTS.md, executable bit)
- Script invariants (cmake, make, pip, llama-server, huggingface, git clone, strict mode)
- Stale build dir recovery (cmake configure failure handling)
- Prerequisite detection (cmake, nvidia-smi)
- gx10-b integration (if reachable via SSH)
- ShellCheck (if available)

## Common Tasks for Agents

### Help a user run the model

```bash
git clone https://github.com/Mikehutu/bonsai-runner
cd bonsai-runner
bash start.sh                # 1-bit (3.9 GB) — auto-detects hardware and recommends
bash start.sh ternary        # ternary (7.2 GB)
```

When run with no arguments, `start.sh` auto-detects the machine's hardware
(CUDA VRAM, Metal, or system RAM) and recommends the best-fitting model
variant. On interactive terminals, it prompts the user to choose. On
non-interactive (CI, SSH), it defaults to `1bit`.

To skip detection and use a specific variant, pass it explicitly:
```bash
bash start.sh ternary        # skip detection, use ternary
bash start.sh 1bit+dspark    # skip detection, use 1-bit with DSpark
```

### Help a user open the web UI

After `start.sh` finishes, the user just opens **`http://localhost:8080`**
in their browser. llama.cpp's built-in web UI is already running — no Docker,
no extra steps. The UI supports:
- Chat with the model
- File upload (PDF, images, text, code)
- Image input — click the `+` icon to upload a photo or screenshot
- Multi-turn conversation
- Copy/paste responses

### Debug a build failure

1. Check if cmake/make are installed: `cmake --version`, `make --version`
2. Check the llama.cpp build log in `~/.bonsai/llama.cpp/build/`
3. If cmake configure fails after `git pull`, the script auto-cleans the build
   dir and retries. If it still fails, manually run:
   ```bash
   rm -rf ~/.bonsai/llama.cpp/build
   cd ~/.bonsai/llama.cpp
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release -j $(nproc)
   ```
4. On DGX Spark, system pip may be blocked — the script falls back to a venv
5. Ensure `build-essential` (Linux) or Xcode CLI (macOS) is installed

### Debug a model download failure

1. Check HuggingFace CLI: `huggingface-cli whoami`
2. Check disk space: the 1-bit model is 3.9 GB, ternary is 7.2 GB
3. Check if the model already exists in cache paths (see Conventions #3)
4. The script will symlink from cache instead of re-downloading

### Modify the llama.cpp build flags

Edit the `CMAKE_FLAGS` section in `start.sh`:

```bash
CMAKE_FLAGS="-DGGML_CUDA=ON"       # CUDA
CMAKE_FLAGS="-DGGML_METAL=ON"      # macOS Metal
CMAKE_FLAGS="-DGGML_VULKAN=ON"     # Vulkan (Linux)
```

## API Usage

The server exposes an OpenAI-compatible endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Available endpoints:
- `POST /v1/chat/completions` — chat completions (streaming supported)
- `GET /v1/models` — list available models
- `POST /v1/embeddings` — embeddings
- `GET /health` — health check

## Troubleshooting

| Problem | Solution |
|---|---|
| `cmake not found` | `sudo apt install cmake build-essential` (Linux) or `brew install cmake` (macOS) |
| `pip install blocked` | Script auto-falls back to venv at `~/.bonsai/venv-hf/` |
| `Port 8080 busy` | Script tries 8081, 8082 automatically. Or set `PORT=18080` |
| `Model download fails` | Check `huggingface-cli whoami`, disk space, network |
| `CUDA not detected` | Install nvidia-smi + CUDA toolkit, or run with `NGL=0` for CPU |
| `llama-server not found after build` | Check `~/.bonsai/llama.cpp/build/bin/` — try `cmake --build . --config Release` |
| `Web UI not loading` | Make sure `start.sh` finished and the server is running. Check `http://localhost:8080/health` |
