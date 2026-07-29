#!/usr/bin/env bash
# start.sh — Bonsai 27B one‑click runner
#   bash start.sh                # default: 1-bit (Q1_0, ~3.9 GB)
#   bash start.sh 1bit           # explicit 1-bit
#   bash start.sh ternary        # ternary (Q2_0, ~7.2 GB)
#   bash start.sh ternary+dspark # ternary + speculative decoding drafter
#   bash start.sh 1bit+dspark    # 1-bit + speculative decoding drafter
set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────
MODEL_VARIANT="${1:-1bit}"
LLAMA_CPP_REPO="https://github.com/PrismML-Eng/llama.cpp"
LLAMA_CPP_DIR="${HOME}/.bonsai/llama.cpp"
MODELS_DIR="${HOME}/.bonsai/models"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
NGL="${NGL:-99}"     # GPU layers (Metal/CUDA)
PARALLEL="${PARALLEL:-0}"   # 0 = auto (llama-server default), 1 = single-request (benchmark-safe)
CONTEXT_SIZE="${CONTEXT_SIZE:-0}"  # 0 = model default (typically 32K). Set to e.g. 262144 for 262K context.

# Model definitions:  HF_REPO | MODEL_FILE | DSPARK_FILE | DSPARK_ARGS | MMPROJ_FILE
# (pipe-separated because DSPARK_ARGS contains spaces)
declare -A MODELS
MODELS[1bit]="prism-ml/Bonsai-27B-gguf|Bonsai-27B-Q1_0.gguf|Bonsai-27B-dspark-Q4_1.gguf|--spec-type draft-dspark --spec-draft-n-max 4|Bonsai-27B-mmproj-Q8_0.gguf"
MODELS[ternary]="prism-ml/Ternary-Bonsai-27B-gguf|Ternary-Bonsai-27B-Q2_0.gguf|Ternary-Bonsai-27B-dspark-Q4_1.gguf|--spec-type draft-dspark --spec-draft-n-max 4|Ternary-Bonsai-27B-mmproj-Q8_0.gguf"

# ─── Help ─────────────────────────────────────────────────────────────
if [[ "${MODEL_VARIANT}" == "-h" || "${MODEL_VARIANT}" == "--help" ]]; then
  echo "Usage: bash start.sh [VARIANT]"
  echo ""
  echo "Variants:"
  echo "  1bit        (default) Bonsai-27B Q1_0      — 3.9 GB, 89.5% of FP16"
  echo "  ternary     Ternary-Bonsai-27B Q2_0        — 7.2 GB, 94.6% of FP16"
  echo "  1bit+dspark 1-bit + DSpark drafter          — 5.7 GB"
  echo "  ternary+dspark Ternary + DSpark drafter     — 9.1 GB"
  echo ""
  echo "Environment variables:"
echo "  PORT=8080   HOST=0.0.0.0   NGL=99   PARALLEL=0   CONTEXT_SIZE=0"
echo ""
echo "Context size:"
echo "  CONTEXT_SIZE=0       model default (~32K, lowest RAM)"
echo "  CONTEXT_SIZE=65536   64K context"
echo "  CONTEXT_SIZE=131072  128K context"
echo "  CONTEXT_SIZE=262144  262K context (max)"
  exit 0
fi

# Parse variant
DSPARK=false
case "${MODEL_VARIANT}" in
  1bit)          KEY="1bit"; DSPARK=false  ;;
  ternary)       KEY="ternary"; DSPARK=false ;;
  1bit+dspark)   KEY="1bit"; DSPARK=true   ;;
  ternary+dspark) KEY="ternary"; DSPARK=true ;;
  *)
    echo "❌ Unknown variant '${MODEL_VARIANT}'"
    echo "   Valid: 1bit, ternary, 1bit+dspark, ternary+dspark"
    exit 1
    ;;
esac

IFS='|' read -r HF_REPO MODEL_FILE DSPARK_FILE DSPARK_ARGS MMPROJ_FILE <<< "${MODELS[$KEY]}"

echo "══════════════════════════════════════════════"
echo "  Bonsai 27B Runner"
echo "  Variant:    ${MODEL_VARIANT}"
echo "  HF repo:    ${HF_REPO}"
echo "  Model:      ${MODEL_FILE}"
if $DSPARK; then echo "  Drafter:    ${DSPARK_FILE}"; fi
if [[ -n "${MMPROJ_FILE}" ]]; then echo "  Vision:     ${MMPROJ_FILE}"; fi
echo "══════════════════════════════════════════════"

# ─── Step 1: Check prerequisites ─────────────────────────────────────
PREREQ_FAIL=false

if ! command -v cmake &>/dev/null; then
  echo "⚠  cmake not found. Install:  sudo apt install cmake build-essential  (or brew install cmake)"
  PREREQ_FAIL=true
fi

if ! command -v make &>/dev/null; then
  echo "⚠  make not found. Install:  sudo apt install build-essential"
  PREREQ_FAIL=true
fi

if ! command -v python3 &>/dev/null; then
  echo "⚠  python3 not found. Needed for huggingface-hub downloader."
  PREREQ_FAIL=true
fi

# Check for CUDA or Metal
BACKEND="CPU"
if command -v nvidia-smi &>/dev/null; then
  # Verify there's actually a usable GPU (not just Windows CUDA drivers visible from WSL)
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q . 2>/dev/null; then
    BACKEND="CUDA"
    echo "✔  CUDA GPU detected"
  else
    echo "   nvidia-smi found but no GPU reported (WSL sees Windows drivers only)"
  fi
elif [[ "$(uname)" == "Darwin" ]]; then
  if [[ "$(uname -m)" == "arm64" ]]; then
    BACKEND="Metal"
    echo "✔  Apple Silicon (Metal) detected"
  fi
fi
echo "   Backend: ${BACKEND}"

# ─── Hardware detection & model recommendation ────────────────────────
# If no variant specified (default), detect available VRAM/RAM and
# recommend the best-fitting model.
if [[ "${1:-}" == "" ]]; then
  TOTAL_VRAM_MB=0
  TOTAL_RAM_MB=0

  # Get total system RAM (errors suppressed for set -e strict mode)
  OS_NAME="$(uname 2>/dev/null)" || true
  if [[ "${OS_NAME}" == "Linux" ]]; then
    TOTAL_RAM_MB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' 2>/dev/null || echo 0)
  elif [[ "${OS_NAME}" == "Darwin" ]]; then
    TOTAL_RAM_MB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1048576)}' 2>/dev/null || echo 0)
  fi

  # Get GPU VRAM if CUDA (errors suppressed for set -e strict mode)
  if [[ "${BACKEND}" == "CUDA" ]]; then
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' 2>/dev/null || echo 0)
    # On unified-memory systems (e.g. DGX Spark GB10), nvidia-smi reports
    # [N/A] for memory.total. Keep CUDA backend — the GPU is usable, VRAM
    # is just not separately queryable.
    if [[ -z "${TOTAL_VRAM_MB}" ]] || ! [[ "${TOTAL_VRAM_MB}" =~ ^[0-9]+$ ]] 2>/dev/null; then
      TOTAL_VRAM_MB=0
      echo "   (GPU VRAM not queryable — unified memory system, using CUDA)"
    fi
  fi

  echo ""
  echo "── Hardware Detection ──"
  echo "   System RAM: ${TOTAL_RAM_MB} MB"
  if [[ "${BACKEND}" == "CUDA" ]]; then
    if [[ "${TOTAL_VRAM_MB}" -gt 0 ]]; then
      echo "   GPU VRAM:   ${TOTAL_VRAM_MB} MB"
    else
      echo "   GPU:        NVIDIA (unified memory — VRAM not separately queryable)"
    fi
  fi

  echo ""
  echo "── Model Recommendations ──"
  echo "   Available variants:"
  echo "     1bit        3.9 GB model  — fits in 6 GB VRAM, 5 GB RAM"
  echo "     ternary     7.2 GB model  — fits in 10 GB VRAM, 9 GB RAM"
  echo "     1bit+dspark 5.7 GB model  — CUDA only, needs 8 GB VRAM"
  echo "     ternary+dspark 9.1 GB model — CUDA only, needs 12 GB VRAM"
  echo ""

  # Context size estimation
  # KV cache for a 27B model: ~6 MB per 1K tokens (Q1_0) or ~8 MB (ternary)
  # Formula: available_ram_for_kv = total_ram - model_size - 1 GB overhead
  if [[ "${TOTAL_RAM_MB}" -gt 0 ]]; then
    if [[ "${KEY}" == "ternary" ]]; then
      MODEL_SIZE_MB=7200
      KV_PER_1K_MB=8
    else
      MODEL_SIZE_MB=3900
      KV_PER_1K_MB=6
    fi
    AVAIL_RAM_MB=$((TOTAL_RAM_MB - MODEL_SIZE_MB - 1024))  # subtract model + 1 GB overhead
    if [[ "${AVAIL_RAM_MB}" -gt 0 ]]; then
      MAX_CTX_K=$((AVAIL_RAM_MB / KV_PER_1K_MB))
      # Cap at 262K (model's practical limit)
      if [[ "${MAX_CTX_K}" -gt 262 ]]; then
        MAX_CTX_K=262
      fi
      echo "   Context estimate: ~${MAX_CTX_K}K tokens (based on ${TOTAL_RAM_MB} MB RAM)"
      echo "   Set CONTEXT_SIZE=$((MAX_CTX_K * 1024)) for max context, or"
      echo "   CONTEXT_SIZE=0 for model default (~32K)"
    else
      echo "   ⚠  Tight on RAM — model may not fit with extra context"
      echo "   Set CONTEXT_SIZE=0 (model default) or try a smaller variant"
    fi
  fi
  echo ""

  # Recommend based on hardware
  if [[ "${BACKEND}" == "CUDA" ]] && [[ "${TOTAL_VRAM_MB}" -gt 0 ]]; then
    if [[ "${TOTAL_VRAM_MB}" -ge 12000 ]]; then
      echo "   ✅ Recommended: ternary+dspark (you have ${TOTAL_VRAM_MB} MB VRAM)"
    elif [[ "${TOTAL_VRAM_MB}" -ge 8000 ]]; then
      echo "   ✅ Recommended: 1bit+dspark (you have ${TOTAL_VRAM_MB} MB VRAM)"
    elif [[ "${TOTAL_VRAM_MB}" -ge 6000 ]]; then
      echo "   ✅ Recommended: 1bit (you have ${TOTAL_VRAM_MB} MB VRAM)"
    else
      echo "   ⚠  Your GPU (${TOTAL_VRAM_MB} MB) may not have enough VRAM for 27B models"
      echo "   ✅ Recommended: 1bit (smallest footprint)"
    fi
  elif [[ "${BACKEND}" == "CUDA" ]]; then
    # Unified memory (VRAM not queryable) — use system RAM as proxy
    if [[ "${TOTAL_RAM_MB}" -ge 10000 ]]; then
      echo "   ✅ Recommended: ternary+dspark (CUDA + ${TOTAL_RAM_MB} MB unified RAM)"
    elif [[ "${TOTAL_RAM_MB}" -ge 5000 ]]; then
      echo "   ✅ Recommended: 1bit+dspark (CUDA + ${TOTAL_RAM_MB} MB unified RAM)"
    else
      echo "   ✅ Recommended: 1bit (smallest footprint)"
    fi
  elif [[ "${BACKEND}" == "Metal" ]]; then
    echo "   ✅ Recommended: 1bit (Metal, 3.9 GB)"
  else
    # CPU — check RAM
    if [[ "${TOTAL_RAM_MB}" -ge 10000 ]]; then
      echo "   ✅ Recommended: ternary (you have ${TOTAL_RAM_MB} MB RAM)"
    elif [[ "${TOTAL_RAM_MB}" -ge 5000 ]]; then
      echo "   ✅ Recommended: 1bit (you have ${TOTAL_RAM_MB} MB RAM)"
    else
      echo "   ⚠  Your system (${TOTAL_RAM_MB} MB RAM) may be tight for 27B models"
      echo "   ✅ Recommended: 1bit (smallest footprint)"
    fi
  fi
  echo ""
  echo "   To use a specific variant: bash start.sh <variant>"
  echo "   To skip this and use default (1bit): press Enter or set MODEL_VARIANT=1bit"
  echo ""

  # Interactive prompt if running in a TTY
  if [[ -t 0 ]]; then
    read -p "Choose variant [1bit/ternary/1bit+dspark/ternary+dspark] (default: 1bit): " CHOSEN
    if [[ -n "${CHOSEN}" ]]; then
      MODEL_VARIANT="${CHOSEN}"
      # Re-parse variant after interactive selection
      case "${MODEL_VARIANT}" in
        1bit)          KEY="1bit"; DSPARK=false  ;;
        ternary)       KEY="ternary"; DSPARK=false ;;
        1bit+dspark)   KEY="1bit"; DSPARK=true   ;;
        ternary+dspark) KEY="ternary"; DSPARK=true ;;
        *)
          echo "❌ Unknown variant '${MODEL_VARIANT}'"
          echo "   Valid: 1bit, ternary, 1bit+dspark, ternary+dspark"
          exit 1
          ;;
      esac
      IFS='|' read -r HF_REPO MODEL_FILE DSPARK_FILE DSPARK_ARGS MMPROJ_FILE <<< "${MODELS[$KEY]}"
      echo ""
      echo "══════════════════════════════════════════════"
      echo "  Bonsai 27B Runner"
      echo "  Variant:    ${MODEL_VARIANT}"
      echo "  HF repo:    ${HF_REPO}"
      echo "  Model:      ${MODEL_FILE}"
      if $DSPARK; then echo "  Drafter:    ${DSPARK_FILE}"; fi
      if [[ -n "${MMPROJ_FILE}" ]]; then echo "  Vision:     ${MMPROJ_FILE}"; fi
      echo "══════════════════════════════════════════════"
    fi

    # ── Context size prompt ──────────────────────────────
    # Recalculate context estimate for the chosen variant
    if [[ "${KEY}" == "ternary" ]]; then
      MODEL_SIZE_MB=7200
      KV_PER_1K_MB=8
    else
      MODEL_SIZE_MB=3900
      KV_PER_1K_MB=6
    fi
    AVAIL_RAM_MB=$((TOTAL_RAM_MB - MODEL_SIZE_MB - 1024))
    MAX_CTX_K=0
    if [[ "${AVAIL_RAM_MB}" -gt 0 ]]; then
      MAX_CTX_K=$((AVAIL_RAM_MB / KV_PER_1K_MB))
      [[ "${MAX_CTX_K}" -gt 262 ]] && MAX_CTX_K=262
    fi

    echo ""
    echo "── Context Size ──"
    echo "   Your machine can handle up to ~${MAX_CTX_K}K tokens of context."
    echo "   Options:"
    echo "     0        = model default (~32K, lowest RAM)"
    echo "     65536    = 64K context"
    echo "     131072   = 128K context"
    echo "     $((MAX_CTX_K * 1024)) = ${MAX_CTX_K}K (max for your hardware)"
    echo ""
    read -p "Context size [0/65536/131072/$((MAX_CTX_K * 1024))] (default: 0): " CTX_CHOSEN
    if [[ -n "${CTX_CHOSEN}" ]]; then
      CONTEXT_SIZE="${CTX_CHOSEN}"
      echo "   → Context: ${CONTEXT_SIZE} tokens"
    fi
  else
    echo "   (Non-interactive mode — using default: 1bit)"
  fi
fi

if $PREREQ_FAIL; then
  echo ""
  echo "❌ Install missing prerequisites and re-run."
  exit 1
fi

# ─── Step 2: Install huggingface-hub CLI ──────────────────────────────
# Prefer 'hf' (new CLI) over 'huggingface-cli' (deprecated, prints warnings, may fail)
HF_CMD=""
HF_VENV=""
if command -v hf &>/dev/null; then
  HF_CMD="hf"
  echo "✔  hf CLI already available"
elif command -v huggingface-cli &>/dev/null; then
  HF_CMD="huggingface-cli"
  echo "✔  huggingface-cli already available (deprecated — consider upgrading to 'hf')"
else
  echo ""
  echo "── Step 2: Installing huggingface-hub CLI ──"
  # Try pip install — fall back to venv if system pip is blocked (DGX Spark)
  if pip3 install -q huggingface-hub 2>/dev/null || pip install -q huggingface-hub 2>/dev/null; then
    echo "✔  huggingface-hub installed via pip"
  else
    echo "   System pip blocked — trying venv..."
    HF_VENV="${HOME}/.bonsai/venv-hf"
    python3 -m venv "${HF_VENV}" 2>/dev/null
    "${HF_VENV}/bin/pip" install -q huggingface-hub 2>/dev/null
    if [[ -f "${HF_VENV}/bin/hf" ]]; then
      HF_CMD="${HF_VENV}/bin/hf"
      echo "✔  huggingface-hub installed in venv at ${HF_VENV}"
    elif [[ -f "${HF_VENV}/bin/huggingface-cli" ]]; then
      HF_CMD="${HF_VENV}/bin/huggingface-cli"
      echo "✔  huggingface-hub installed in venv at ${HF_VENV} (deprecated — consider upgrading to 'hf')"
    else
      echo "⚠  Could not install huggingface-hub. Install manually:"
      echo "   python3 -m venv ~/.bonsai/venv-hf"
      echo "   ~/.bonsai/venv-hf/bin/pip install huggingface-hub"
      exit 1
    fi
  fi
  # Detect command after install — prefer 'hf' over deprecated 'huggingface-cli'
  if [[ -z "${HF_CMD}" ]]; then
    if command -v hf &>/dev/null; then
      HF_CMD="hf"
    elif command -v huggingface-cli &>/dev/null; then
      HF_CMD="huggingface-cli"
    else
      echo "⚠  huggingface-hub CLI not in PATH after install."
      HF_CMD="python3 -m huggingface_hub.huggingface_cli"
    fi
  fi
fi

# ─── Helper: safe download with fallback ──────────────────────────────
# Downloads a file from HuggingFace. If the primary HF_CMD fails (e.g.
# deprecated 'huggingface-cli' prints warnings and exits non-zero),
# falls back to 'hf' if available.
# Temporarily unsets HF_HUB_OFFLINE so downloads work even if the user's
# shell profile sets it (common on airgapped machines with pre-cached models).
download_model() {
  local repo="$1" file="$2" dest="$3"
  # Unset HF_HUB_OFFLINE for the download — we want to reach HuggingFace
  # Also set HF_HUB_DISABLE_XET=1 to avoid xet permission errors on machines
  # where the xet cache path is not writable (e.g. DGX Spark with /mnt/storage)
  ( unset HF_HUB_OFFLINE; export HF_HUB_DISABLE_XET=1; ${HF_CMD} download "${repo}" "${file}" --local-dir "${dest}" ) 2>&1
  if [[ $? -ne 0 ]]; then
    echo "   ⚠  '${HF_CMD}' failed — trying 'hf' as fallback..."
    # Check system PATH first, then venv
    if command -v hf &>/dev/null; then
      ( unset HF_HUB_OFFLINE; export HF_HUB_DISABLE_XET=1; hf download "${repo}" "${file}" --local-dir "${dest}" ) 2>&1
    elif [[ -n "${HF_VENV}" && -x "${HF_VENV}/bin/hf" ]]; then
      ( unset HF_HUB_OFFLINE; export HF_HUB_DISABLE_XET=1; "${HF_VENV}/bin/hf" download "${repo}" "${file}" --local-dir "${dest}" ) 2>&1
    else
      echo "❌  Download failed and no 'hf' fallback available."
      return 1
    fi
    if [[ $? -ne 0 ]]; then
      echo "❌  Download failed."
      return 1
    fi
  fi
}

# ─── Step 3: Build or update llama.cpp (PrismML fork) ────────────────
echo ""
echo "── Step 3: Building llama.cpp (PrismML fork) �─"
mkdir -p "${HOME}/.bonsai"

if [[ -d "${LLAMA_CPP_DIR}" ]]; then
  echo "   Repository exists at ${LLAMA_CPP_DIR}"
  echo "   Updating..."
  (cd "${LLAMA_CPP_DIR}" && git pull --ff-only 2>/dev/null) || echo "   (could not update, using existing)"
else
  echo "   Cloning PrismML fork..."
  git clone --depth 1 -b prism "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
fi

BUILD_DIR="${LLAMA_CPP_DIR}/build"
CMAKE_FLAGS=""
if [[ "${BACKEND}" == "CUDA" ]]; then
  CMAKE_FLAGS="-DGGML_CUDA=ON"
  echo "   CUDA build enabled"
elif [[ "${BACKEND}" == "Metal" ]]; then
  CMAKE_FLAGS="-DGGML_METAL=ON"
  echo "   Metal build enabled"
fi

mkdir -p "${BUILD_DIR}"
echo "   Configuring..."
if ! (cd "${BUILD_DIR}" && cmake .. ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3); then
  echo "   CMake configure failed — stale build dir after git pull. Cleaning and retrying..."
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
  (cd "${BUILD_DIR}" && cmake .. ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3)
fi
echo "   Building..."
cmake --build "${BUILD_DIR}" --config Release -j "$(nproc)" 2>&1 | tail -5
echo "✔  llama.cpp built"
echo "   Binaries: ${BUILD_DIR}/bin/"

SERVER_BIN="${BUILD_DIR}/bin/llama-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
  # Try alternate location
  SERVER_BIN="${BUILD_DIR}/examples/server/llama-server"
fi
if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "❌  Could not find llama-server binary after build."
  echo "   Looked in: ${BUILD_DIR}/bin/llama-server"
  echo "   Looked in: ${BUILD_DIR}/examples/server/llama-server"
  exit 1
fi

# ─── Step 4: Download model ──────────────────────────────────────────
echo ""
echo "── Step 4: Downloading model weights ──"
mkdir -p "${MODELS_DIR}/${KEY}"

MODEL_PATH="${MODELS_DIR}/${KEY}/${MODEL_FILE}"

# Smart cache: check common local paths before downloading
if [[ -f "${MODEL_PATH}" ]]; then
  echo "✔  Model already downloaded: $(du -h "${MODEL_PATH}" | cut -f1)"
else
  # Check common local model cache paths
  for CACHE in \
    "${MODELS_DIR}/${MODEL_FILE}" \
    "${HOME}/.cache/huggingface/hub/${MODEL_FILE}" \
    "${HOME}/models/${MODEL_FILE}" \
    "./models/${MODEL_FILE}"
  do
    if [[ -f "$CACHE" ]]; then
      echo "✔  Found cached model at $CACHE — symlinking..."
      ln -sf "$CACHE" "${MODEL_PATH}"
      break
    fi
  done
fi

# Download if still not present
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "   Downloading ${HF_REPO}/${MODEL_FILE} from HuggingFace ..."
  echo "   (This may take a while — file is $( [[ $KEY == "ternary" ]] && echo "~7.2 GB" || echo "~3.9 GB"))"
  download_model "${HF_REPO}" "${MODEL_FILE}" "${MODELS_DIR}/${KEY}"
  echo "✔  Downloaded: ${MODEL_PATH}"
fi

# Download drafter if requested
DSPARK_MODEL_PATH=""
DSPARK_SERVER_ARGS=""
MMPROJ_PATH=""
MMPROJ_SERVER_ARGS=""
if $DSPARK; then
  DSPARK_MODEL_PATH="${MODELS_DIR}/${KEY}/${DSPARK_FILE}"
  if [[ -f "${DSPARK_MODEL_PATH}" ]]; then
    echo "✔  Drafter already downloaded: $(du -h "${DSPARK_MODEL_PATH}" | cut -f1)"
  else
    # Check local cache paths for drafter
    for CACHE in \
      "${MODELS_DIR}/${DSPARK_FILE}" \
      "${HOME}/.cache/huggingface/hub/${DSPARK_FILE}" \
      "${HOME}/models/${DSPARK_FILE}"
    do
      if [[ -f "$CACHE" ]]; then
        echo "✔  Found cached drafter at $CACHE — symlinking..."
        ln -sf "$CACHE" "${DSPARK_MODEL_PATH}"
        break
      fi
    done
  fi
  if [[ ! -f "${DSPARK_MODEL_PATH}" ]]; then
    echo "   Downloading drafter ${HF_REPO}/${DSPARK_FILE} from HuggingFace ..."
    download_model "${HF_REPO}" "${DSPARK_FILE}" "${MODELS_DIR}/${KEY}"
    echo "✔  Drafter downloaded"
  fi
  DSPARK_SERVER_ARGS="-md ${DSPARK_MODEL_PATH} ${DSPARK_ARGS}"
fi

# ─── Step 4b: Download multimodal projector (vision tower) ──────────
# Bonsai-27B is a vision-language model: the mmproj file enables image input.
# It is loaded only when an image arrives, so text-only inference is unaffected.
if [[ -n "${MMPROJ_FILE}" ]]; then
  MMPROJ_PATH="${MODELS_DIR}/${KEY}/${MMPROJ_FILE}"
  if [[ -f "${MMPROJ_PATH}" ]]; then
    echo "✔  Multimodal projector already downloaded: $(du -h "${MMPROJ_PATH}" | cut -f1)"
  else
    # Check common local cache paths
    for CACHE in \
      "${MODELS_DIR}/${MMPROJ_FILE}" \
      "${HOME}/.cache/huggingface/hub/${MMPROJ_FILE}" \
      "${HOME}/models/${MMPROJ_FILE}" \
      "./models/${MMPROJ_FILE}"
    do
      if [[ -f "$CACHE" ]]; then
        echo "✔  Found cached mmproj at $CACHE — symlinking..."
        ln -sf "$CACHE" "${MMPROJ_PATH}"
        break
      fi
    done
  fi
  # Download if still not present
  if [[ ! -f "${MMPROJ_PATH}" ]]; then
    echo "   Downloading multimodal projector ${HF_REPO}/${MMPROJ_FILE} from HuggingFace ..."
    echo "   (This enables image input — ~0.6 GB)"
    download_model "${HF_REPO}" "${MMPROJ_FILE}" "${MODELS_DIR}/${KEY}"
    echo "✔  Multimodal projector downloaded"
  fi
  MMPROJ_SERVER_ARGS="-mm ${MMPROJ_PATH}"
fi

# ─── Port fallback (try PORT, PORT+1, PORT+2) ─────────────────────────
FINAL_PORT="${PORT}"
if command -v ss &>/dev/null; then
  for try_port in "${PORT}" $((PORT + 1)) $((PORT + 2)); do
    if ! ss -tlnp 2>/dev/null | grep -qE "[: ]${try_port}\b"; then
      FINAL_PORT="${try_port}"
      break
    fi
  done
elif command -v lsof &>/dev/null; then
  for try_port in "${PORT}" $((PORT + 1)) $((PORT + 2)); do
    if ! lsof -i :"${try_port}" &>/dev/null; then
      FINAL_PORT="${try_port}"
      break
    fi
  done
fi
if [[ "${FINAL_PORT}" != "${PORT}" ]]; then
  echo "   Port ${PORT} busy → using port ${FINAL_PORT}"
fi

# ─── Step 5: Start server ────────────────────────────────────────────
echo ""
echo "── Step 5: Starting llama-server ──"
echo "   Model:     ${MODEL_PATH}"
if [[ -n "${MMPROJ_PATH}" ]]; then echo "   Vision:    ${MMPROJ_PATH}"; fi
echo "   Endpoint:  http://${HOST}:${FINAL_PORT}"
echo "   GPU layers: ${NGL}"
echo "   Context:   ${CONTEXT_SIZE:-0} (0 = model default)"
echo "   Parallel:   ${PARALLEL:-auto}"
echo ""
echo "   Press Ctrl+C to stop."
echo ""

# Build parallel args — PARALLEL=0 means "auto" (llama-server default)
PARALLEL_ARGS=""
if [[ "${PARALLEL}" != "0" && -n "${PARALLEL}" ]]; then
  PARALLEL_ARGS="--parallel ${PARALLEL}"
fi

# Build context size args
# CONTEXT_SIZE=0 → model default (~32K)
# CONTEXT_SIZE=262144 → 262K context (tested on CPU + DDR5, ~4 GB model + ~1.6 GB KV cache)
CTX_ARGS="-c ${CONTEXT_SIZE}"

exec "${SERVER_BIN}" \
  -m "${MODEL_PATH}" \
  ${MMPROJ_SERVER_ARGS} \
  ${DSPARK_SERVER_ARGS} \
  --host "${HOST}" \
  --port "${FINAL_PORT}" \
  -ngl "${NGL}" \
  ${CTX_ARGS} \
  --temp 0.7 \
  --top-p 0.95 \
  --top-k 40 \
  --image-max-tokens 1024 \
  ${PARALLEL_ARGS}
