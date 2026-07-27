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

5. **Backend detection.** `start.sh` auto-detects CUDA (nvidia-smi) and Metal
   (Apple Silicon). CPU is the default fallback.

## Variant System

The `start.sh` script uses a `MODELS` associative array to define variants:

```bash
declare -A MODELS
MODELS[1bit]="prism-ml/Bonsai-27B-gguf  Bonsai-27B-Q1_0.gguf  Bonsai-27B-dspark-Q4_1.gguf  --spec-type draft-dspark --spec-draft-n-max 4"
MODELS[ternary]="prism-ml/Ternary-Bonsai-27B-gguf  Ternary-Bonsai-27B-Q2_0.gguf  Ternary-Bonsai-27B-dspark-Q4_1.gguf  --spec-type draft-dspark --spec-draft-n-max 4"
```

Each entry is: `HF_REPO  MODEL_FILE  DSPARK_FILE  DSPARK_ARGS`

**To add a new variant:**
1. Add a new entry to the `MODELS` array
2. Add a case branch in the variant parser
3. Add a help text line
4. Update the README variant table

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
bash start.sh                # 1-bit (3.9 GB)
bash start.sh ternary        # ternary (7.2 GB)
```

The server will be at `http://localhost:8080`.

### Help a user open the web UI

After `start.sh` finishes, the user just opens **`http://localhost:8080`**
in their browser. llama.cpp's built-in web UI is already running — no Docker,
no extra steps. The UI supports:
- Chat with the model
- File upload (PDF, images, text, code)
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
