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

# Model definitions:  HF_REPO  MODEL_FILE  DSPARK_FILE  DSPARK_FLAG
declare -A MODELS
MODELS[1bit]="prism-ml/Bonsai-27B-gguf  Bonsai-27B-Q1_0.gguf  Bonsai-27B-dspark-Q4_1.gguf  --spec-type draft-dspark --spec-draft-n-max 4"
MODELS[ternary]="prism-ml/Ternary-Bonsai-27B-gguf  Ternary-Bonsai-27B-Q2_0.gguf  Ternary-Bonsai-27B-dspark-Q4_1.gguf  --spec-type draft-dspark --spec-draft-n-max 4"

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
  echo "  PORT=8080   HOST=0.0.0.0   NGL=99"
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

IFS=' ' read -r HF_REPO MODEL_FILE DSPARK_FILE DSPARK_ARGS <<< "${MODELS[$KEY]}"

echo "══════════════════════════════════════════════"
echo "  Bonsai 27B Runner"
echo "  Variant:    ${MODEL_VARIANT}"
echo "  HF repo:    ${HF_REPO}"
echo "  Model:      ${MODEL_FILE}"
if $DSPARK; then echo "  Drafter:    ${DSPARK_FILE}"; fi
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
  BACKEND="CUDA"
  echo "✔  CUDA GPU detected"
elif [[ "$(uname)" == "Darwin" ]] && command -v metal &>/dev/null 2>&1 || true; then
  # macOS with Apple Silicon likely has Metal
  if [[ "$(uname -m)" == "arm64" ]]; then
    BACKEND="Metal"
    echo "✔  Apple Silicon (Metal) detected"
  fi
fi
echo "   Backend: ${BACKEND}"

if $PREREQ_FAIL; then
  echo ""
  echo "❌ Install missing prerequisites and re-run."
  exit 1
fi

# ─── Step 2: Install huggingface-hub CLI ──────────────────────────────
HF_CMD=""
HF_VENV=""
if command -v huggingface-cli &>/dev/null; then
  HF_CMD="huggingface-cli"
  echo "✔  huggingface-cli already available"
elif command -v hf &>/dev/null; then
  HF_CMD="hf"
  echo "✔  hf CLI already available"
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
    if [[ -f "${HF_VENV}/bin/huggingface-cli" ]]; then
      HF_CMD="${HF_VENV}/bin/huggingface-cli"
      echo "✔  huggingface-hub installed in venv at ${HF_VENV}"
    else
      echo "⚠  Could not install huggingface-hub. Install manually:"
      echo "   python3 -m venv ~/.bonsai/venv-hf"
      echo "   ~/.bonsai/venv-hf/bin/pip install huggingface-hub"
      exit 1
    fi
  fi
  # Detect command after install
  if [[ -z "${HF_CMD}" ]]; then
    if command -v huggingface-cli &>/dev/null; then
      HF_CMD="huggingface-cli"
    elif command -v hf &>/dev/null; then
      HF_CMD="hf"
    else
      echo "⚠  huggingface-hub CLI not in PATH after install."
      HF_CMD="python3 -m huggingface_hub.huggingface_cli"
    fi
  fi
fi

# ─── Step 3: Build or update llama.cpp (PrismML fork) �───────────────
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
(cd "${BUILD_DIR}" && cmake .. ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3)
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
  ${HF_CMD} download "${HF_REPO}" "${MODEL_FILE}" --local-dir "${MODELS_DIR}/${KEY}" 2>&1
  echo "✔  Downloaded: ${MODEL_PATH}"
fi

# Download drafter if requested
DSPARK_MODEL_PATH=""
DSPARK_SERVER_ARGS=""
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
    ${HF_CMD} download "${HF_REPO}" "${DSPARK_FILE}" --local-dir "${MODELS_DIR}/${KEY}" 2>&1
    echo "✔  Drafter downloaded"
  fi
  DSPARK_SERVER_ARGS="-md ${DSPARK_MODEL_PATH} ${DSPARK_ARGS}"
fi

# ─── Step 5: Start server ────────────────────────────────────────────
echo ""
echo "── Step 5: Starting llama-server ──"
echo "   Model:     ${MODEL_PATH}"
echo "   Endpoint:  http://${HOST}:${PORT}"
echo "   GPU layers: ${NGL}"
echo ""
echo "   Press Ctrl+C to stop."
echo ""

exec "${SERVER_BIN}" \
  -m "${MODEL_PATH}" \
  ${DSPARK_SERVER_ARGS} \
  --host "${HOST}" \
  --port "${PORT}" \
  -ngl "${NGL}" \
  -c 0 \
  --temp 0.7 \
  --top-p 0.95 \
  --top-k 40
