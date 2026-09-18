# start.ps1 - Bonsai one-click runner for Windows (native, no WSL needed)
#   powershell -ExecutionPolicy Bypass -File start.ps1                # default: bonsai2 (PTQ1_0)
#   powershell -ExecutionPolicy Bypass -File start.ps1 -Variant bonsai2-pq2
#   powershell -ExecutionPolicy Bypass -File start.ps1 -Variant 1bit
#
# Uses the official PrismML llama.cpp fork prebuilt Windows binaries (stock
# llama.cpp cannot load PTQ1_0/PQ2_0). Backend auto-detect: NVIDIA => CUDA,
# AMD/Intel GPU => Vulkan, no GPU => CPU. Override with -Backend.
#
# Requires: Windows 10/11 with curl.exe (built-in) and internet on first run.

param(
    [string]$Variant = "bonsai2",
    [string]$Backend = "",          # auto | cuda | vulkan | cpu
    [int]$Port = 0,                 # 0 = 8080 with 8081/8082 fallback
    [string]$HostAddr = "",         # "" = 0.0.0.0
    [int]$Ngl = 99,
    [int]$ContextSize = 0,          # 0 = safe default (32K for Bonsai 2)
    [int]$Parallel = 0,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Env overrides (same names as start.sh)
if ($env:PORT -and $Port -eq 0)             { $Port = [int]$env:PORT }
if ($env:HOST -and $HostAddr -eq "")        { $HostAddr = $env:HOST }
if ($env:NGL -and $Ngl -eq 99)              { $Ngl = [int]$env:NGL }
if ($env:CONTEXT_SIZE -and $ContextSize -eq 0) { $ContextSize = [int]$env:CONTEXT_SIZE }
if ($env:PARALLEL -and $Parallel -eq 0)     { $Parallel = [int]$env:PARALLEL }
if ($env:BONSAI_BACKEND -and $Backend -eq "") { $Backend = $env:BONSAI_BACKEND }

# Variants: "HF_REPO|MODEL_FILE|MMPROJ_FILE"
$MODELS = @{
    "1bit"       = "prism-ml/Bonsai-27B-gguf|Bonsai-27B-Q1_0.gguf|Bonsai-27B-mmproj-Q8_0.gguf"
    "ternary"    = "prism-ml/Ternary-Bonsai-27B-gguf|Ternary-Bonsai-27B-Q2_g64.gguf|Ternary-Bonsai-27B-mmproj-Q8_0.gguf"
    "bonsai2"    = "prism-ml/Ternary-Bonsai-2-27B-gguf|Ternary-Bonsai-2-27B-PTQ1_0.gguf|Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
    "bonsai2-pq2"= "prism-ml/Ternary-Bonsai-2-27B-gguf|Ternary-Bonsai-2-27B-PQ2_0.gguf|Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
}

if ($Help) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File start.ps1 [options]"
    Write-Host ""
    Write-Host "Variants:"
    Write-Host "  bonsai2       Bonsai 2 PTQ1_0  - 6.0 GB, 98.2% of FP16 (default)"
    Write-Host "  bonsai2-pq2   Bonsai 2 PQ2_0  - 7.2 GB, fastest on NVIDIA (uses --no-repack, see llama.cpp#180)"
    Write-Host "  ternary       Bonsai 27B Q2_g64 - 7.2 GB"
    Write-Host "  1bit          Bonsai 27B Q1_0   - 3.9 GB"
    Write-Host ""
    Write-Host "Options: -Backend cuda|vulkan|cpu  -Port N  -HostAddr X  -Ngl N  -ContextSize N  -Parallel N"
    Write-Host "Env: PORT HOST NGL CONTEXT_SIZE PARALLEL BONSAI2_REPACK BONSAI_BACKEND"
    Write-Host "Stop: powershell -ExecutionPolicy Bypass -File stop.ps1"
    exit 0
}

if (-not $MODELS.ContainsKey($Variant)) {
    Write-Warning "Unknown variant '$Variant'. Valid: $($MODELS.Keys -join ', ')"
    exit 1
}

$fields = $MODELS[$Variant] -split "\|"
$hfRepo = $fields[0]; $modelFile = $fields[1]; $mmprojFile = $fields[2]

# --- Paths -----------------------------------------------------------
$base = Join-Path $HOME ".bonsai"
$modelsDir = Join-Path $base "models"
$binRoot   = Join-Path $base "llama.cpp-bin"
$dlDir     = Join-Path $base "downloads"
$pidFile   = Join-Path $base "llama-server.pid"
New-Item -ItemType Directory -Force -Path $modelsDir, $binRoot, $dlDir | Out-Null

# --- Backend detection -----------------------------------------------
if ($Backend -eq "") { $Backend = "auto" }
if ($Backend -eq "auto") {
    $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    $gpuStr = $gpus -join "; "
    Write-Host "   GPUs: $gpuStr"
    if ($gpuStr -match "NVIDIA") { $Backend = "cuda" }
    elseif ($gpuStr -match "AMD|Radeon|Intel|Arc") { $Backend = "vulkan" }
    else { $Backend = "cpu" }
}
Write-Host "   Backend: $Backend"

# Map backend -> asset suffix (x64; ARM64 devices only have cpu/cuda-13.4)
$arch = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
$archSuffix = if ($arch.ToString() -eq "Arm64") { "arm64" } else { "x64" }
switch ($Backend) {
    "cuda"   { $assetSub = "win-cuda-12.4-$archSuffix" }
    "vulkan" { $assetSub = "win-vulkan-$archSuffix" }
    "cpu"    { $assetSub = "win-cpu-$archSuffix" }
    default  { Write-Host "Unknown backend '$Backend'"; exit 1 }
}

# --- Get latest PrismML fork release ---------------------------------
$relTag = ""; $assetName = ""; $assetUrl = ""
try {
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest" -Headers @{ "User-Agent" = "bonsai-runner" }
    $relTag = $rel.tag_name
    $asset = $rel.assets | Where-Object { $_.name -like "*bin-win*" -and $_.name -like "*$assetSub*" -and $_.name -like "*.zip" -and $_.name -notlike "cudart*" } | Select-Object -First 1
    if ($asset) { $assetName = $asset.name; $assetUrl = $asset.browser_download_url }
} catch { Write-Host "   (GitHub API unreachable - falling back to known release)" }
if (-not $assetUrl) {
    # Fallback: known-good release
    $relTag = "prism-b10660-e311ed3"
    $assetName = "llama-prism-b10660-e311ed3-bin-$assetSub.zip"
    $assetUrl = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$relTag/$assetName"
}
Write-Host "   Release: $relTag ($assetName)"

$binDir = Join-Path $binRoot $relTag
$serverExe = Join-Path $binDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    $zip = Join-Path $dlDir $assetName
    if (-not (Test-Path $zip)) {
        Write-Host "   Downloading prebuilt binary... (large for CUDA builds)"
        curl.exe -L --retry 3 --fail -o "$zip" "$assetUrl"
        if ($LASTEXITCODE -ne 0) { Write-Host "[ERR] Binary download failed: $assetUrl"; exit 1 }
    }
    Write-Host "   Extracting..."
    Expand-Archive -Path $zip -DestinationPath $binDir -Force
    if (-not (Test-Path $serverExe)) {
        # handle nested folder (bin/ inside archive)
        $nested = Get-ChildItem $binDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if ($nested) { $serverExe = $nested.FullName } else { Write-Host "[ERR] llama-server.exe not found in $assetName"; exit 1 }
    }
}

# --- Download model + mmproj (resumable) ----------------------------
function Download-File($repo, $file, $destDir) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $dest = Join-Path $destDir $file
    if (Test-Path $dest) { Write-Host "   [OK] $file already present"; return $dest }
    $url = "https://huggingface.co/$repo/resolve/main/$file"
    Write-Host "   Downloading $file ..."
    curl.exe -L --retry 3 -C - -o "$dest" "$url"
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERR] Download failed: $url"; exit 1 }
    return $dest
}

$variantDir = Join-Path $modelsDir $Variant
$modelPath = Download-File $hfRepo $modelFile $variantDir
$mmprojPath = Download-File $hfRepo $mmprojFile $variantDir

# --- Port fallback ---------------------------------------------------
function Test-PortFree($p) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $ar = $c.BeginConnect("127.0.0.1", $p, $null, $null)
        $ok = $ar.AsyncWaitHandle.WaitOne(400)
        if ($ok -and $c.Connected) { return $false }
        return $true
    } catch { return $true }
    finally { $c.Close() }
}
$finalPort = 8080
if ($Port -gt 0) { $finalPort = $Port }
while (-not (Test-PortFree $finalPort)) {
    if ($Port -gt 0) { Write-Host "   Port $finalPort busy"; exit 1 }
    $finalPort++
    if ($finalPort -gt 8082) { Write-Host "[ERR] 8080-8082 busy"; exit 1 }
    Write-Host "   Port busy -> trying $finalPort"
}
$hostBind = if ($HostAddr -eq "") { "0.0.0.0" } else { $HostAddr }

# --- Per-variant sampling / extras (mirrors start.sh) ---------------
$sampling = "--temp 0.7 --top-p 0.95 --top-k 40"
$extra = ""
if ($Variant -eq "bonsai2" -or $Variant -eq "bonsai2-pq2") {
    $sampling = "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
    $extra = "-fa on --jinja"
    # PrismML-Eng/llama.cpp#180: PQ2_0 repack segfaults (Hadamard F32 helpers);
    # PTQ1_0 unaffected. BONSAI2_REPACK=1 re-enables repacking for PQ2_0.
    if ($Variant -eq "bonsai2-pq2" -and $env:BONSAI2_REPACK -ne "1") {
        $extra = "$extra --no-repack"
    }
}

# CONTEXT_SIZE=0 -> safe 32K for Bonsai 2 (GGUF default is 262K)
$ctx = $ContextSize
if ($ctx -eq 0 -and ($Variant -eq "bonsai2" -or $Variant -eq "bonsai2-pq2")) { $ctx = 32768 }

$args = @(
    "-m", "`"$modelPath`""
    "--mmproj", "`"$mmprojPath`""
    "--host", $hostBind
    "--port", "$finalPort"
    "-ngl", "$Ngl"
    "-c", "$ctx"
) + (@($sampling -split '\s+')) + (@($extra -split '\s+')) + @(
    "--image-max-tokens", "1024"
)
if ($Parallel -gt 0) { $args += @("--parallel", "$Parallel") }
$serverArgs = $args

Write-Host ""
Write-Host "-- Starting llama-server --"
Write-Host "   Model:    $modelPath"
Write-Host "   Vision:   $mmprojPath"
Write-Host "   Endpoint: http://$hostBind`:$finalPort"
Write-Host "   Backend:  $Backend  Context: $ctx tokens"
Write-Host "   Stop:     powershell -ExecutionPolicy Bypass -File stop.ps1"
Write-Host ""

$p = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -WindowStyle Hidden -PassThru
Set-Content -Path $pidFile -Value $p.Id
Write-Host "   PID: $($p.Id) (saved to $pidFile)"
Write-Host "   Waiting for server... (open http://localhost:$finalPort in your browser)"
$p.WaitForExit()
