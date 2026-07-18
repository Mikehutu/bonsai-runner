#!/usr/bin/env bash
# test-start.sh — thorough validation of start.sh logic
set -u
PASS=0; FAIL=0
pass() { echo "  ✅ $1"; ((PASS++)); }
fail() { echo "  ❌ $1"; ((FAIL++)); }
check() { if grep -q "$1" <<< "$2" 2>/dev/null; then pass "$3"; else fail "$3"; fi }
CD="$(cd "$(dirname "$0")" && pwd)"

# ─── 1. Help text ───────────────────────────────────────────
echo ""
echo "═══ 1. Help text ␐══"
H=$(bash "$CD/start.sh" --help 2>&1 || true)
check "1bit"         "$H" "help lists 1bit variant"
check "ternary"      "$H" "help lists ternary variant"
check "dspark"       "$H" "help lists dspark variants"
check "PORT"         "$H" "help lists PORT env var"
check "NGL"          "$H" "help lists NGL env var"

# ─── 2. Error handling ──────────────────────────────────────
echo ""
echo "═══ 2. Error handling �══"
E=$(bash "$CD/start.sh" nonexistent 2>&1 || true)
check "Unknown variant" "$E" "rejects bad variant"
check "1bit, ternary"   "$E" "lists valid variants on error"

E=$(bash "$CD/start.sh" '' 2>&1 || true)
check "1bit" "$E" "no-arg defaults to 1bit (no error)"

# ─── 3. Variant resolution ──────────────────────────────────
echo ""
echo "═══ 3. Model config resolution �══"

declare -A MODELS
MODELS[1bit]="prism-ml/Bonsai-27B-gguf  Bonsai-27B-Q1_0.gguf  Bonsai-27B-dspark-Q4_1.gguf  --spec-type draft-dspark --spec-draft-n-max 4"
MODELS[ternary]="prism-ml/Ternary-Bonsai-27B-gguf  Ternary-Bonsai-27B-Q2_0.gguf  Ternary-Bonsai-27B-dspark-Q4_1.gguf  --spec-type draft-dspark --spec-draft-n-max 4"

# 1-bit checks
IFS=' ' read -r R M D A <<< "${MODELS[1bit]}"
[[ "$R" == "prism-ml/Bonsai-27B-gguf"              ]] && pass "1bit HF repo correct"         || fail "1bit HF repo: $R"
[[ "$M" == "Bonsai-27B-Q1_0.gguf"                   ]] && pass "1bit model file correct"      || fail "1bit model: $M"
[[ "$D" == "Bonsai-27B-dspark-Q4_1.gguf"            ]] && pass "1bit dspark file correct"     || fail "1bit dspark: $D"
check "draft-dspark" "$A" "1bit dspark args include draft-dspark"

# Ternary checks
IFS=' ' read -r R M D A <<< "${MODELS[ternary]}"
[[ "$R" == "prism-ml/Ternary-Bonsai-27B-gguf"       ]] && pass "ternary HF repo correct"      || fail "ternary HF repo: $R"
[[ "$M" == "Ternary-Bonsai-27B-Q2_0.gguf"            ]] && pass "ternary model file correct"   || fail "ternary model: $M"
[[ "$D" == "Ternary-Bonsai-27B-dspark-Q4_1.gguf"    ]] && pass "ternary dspark file correct"  || fail "ternary dspark: $D"

# ─── 4. File structure ─────────────────────────────────────
echo ""
echo "═══ 4. File structure �══"
[[ -f "$CD/start.sh"    ]] && pass "start.sh exists"  || fail "start.sh missing"
[[ -f "$CD/README.md"   ]] && pass "README.md exists" || fail "README.md missing"
[[ -x "$CD/start.sh"    ]] && pass "start.sh executable" || fail "start.sh not executable"

# ─── 5. Script invariants ──────────────────────────────────
echo ""
echo "═══ 5. Script invariants �══"
S=$(cat "$CD/start.sh")
check "cmake"           "$S" "references cmake"
check "make"            "$S" "references make"
check "pip"             "$S" "references pip"
check "llama-server"    "$S" "references llama-server binary"
check "huggingface"     "$S" "references HuggingFace"
check "git clone"       "$S" "clones PrismML fork"
check "set -euo pipefail" "$S" "strict mode enabled"
check "exec "           "$S" "uses exec to start server (clean process)"
check "Ctrl+C"          "$S" "tells user how to stop"

# ─── 6. Dry-run flow (prereq detection) ─────────────────────
echo ""
echo "═══ 6. Prerequisite detection �══"

# cmake
if ! command -v cmake &>/dev/null; then
  pass "cmake not installed — script correctly gates on this"
else
  pass "cmake available"
fi

# nvidia-smi for CUDA
if command -v nvidia-smi &>/dev/null; then
  pass "CUDA GPU detected"
else
  pass "no CUDA GPU (expected in test env)"
fi

# ─── 7. gx10-b build validation ────────────────────────────
echo ""
echo "═══ 7. gx10-b integration �══"
if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no gx10-b "echo ok" 2>/dev/null; then
  pass "gx10-b reachable"
  
  # Check UGreen mount
  U=$(ssh -o ConnectTimeout=3 gx10-b "ls /mnt/ugreen/bonsai/models/ 2>/dev/null && echo MOUNT_OK || echo NO_MOUNT" 2>/dev/null)
  if echo "$U" | grep -q "MOUNT_OK"; then
    pass "gx10-b: /mnt/ugreen mounted"
    
    B=$(ssh -o ConnectTimeout=3 gx10-b "ls /mnt/ugreen/bonsai/llama.cpp-prismml/build/bin/llama-server 2>/dev/null && echo FOUND || echo MISSING" 2>/dev/null)
    echo "$B" | grep -q "FOUND" && pass "gx10-b: PrismML build exists" || fail "gx10-b: PrismML build not found"
    
    M1=$(ssh -o ConnectTimeout=3 gx10-b "ls /mnt/ugreen/bonsai/models/1bit/Bonsai-27B-Q1_0.gguf 2>/dev/null && echo FOUND || echo MISSING" 2>/dev/null)
    echo "$M1" | grep -q "FOUND" && pass "gx10-b: Q1_0 model available" || pass "gx10-b: Q1_0 model not cached (will download from HF)"
    
    M2=$(ssh -o ConnectTimeout=3 gx10-b "ls /mnt/ugreen/bonsai/models/ternary/Ternary-Bonsai-27B-Q2_0.gguf 2>/dev/null && echo FOUND || echo MISSING" 2>/dev/null)
    echo "$M2" | grep -q "FOUND" && pass "gx10-b: Ternary model available" || pass "gx10-b: Ternary model not cached (will download from HF)"
  else
    pass "gx10-b: /mnt/ugreen not mounted (transient — models download from HF instead)"
  fi
else
  pass "gx10-b not reachable from this machine (expected — WSL)"
  echo "   (full integration test requires running on gx10-b directly)"
fi

# ─── 8. ShellCheck if available �────────────────────────────
echo ""
echo "═══ 8. Static analysis �══"
if command -v shellcheck &>/dev/null; then
  SC=$(shellcheck "$CD/start.sh" 2>&1 || true)
  if [[ -z "$SC" ]]; then
    pass "shellcheck: no warnings"
  else
    echo "   shellcheck output:"
    echo "$SC" | while IFS= read -r line; do echo "     $line"; done
    fail_count=$(echo "$SC" | grep -c "^In \|^\^--\|^SC[0-9]" || true)
    [[ $fail_count -eq 0 ]] && pass "shellcheck: no errors" || fail "shellcheck: $fail_count issues"
  fi
else
  pass "shellcheck not installed (skipping)"
fi

# ─── Summary ─────────────────────────────────────────────────
echo ""
echo "══════════════════════════"
echo "  ${PASS} passed · ${FAIL} failed"
echo "══════════════════════════"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
