# Jeveration on AMD GPUs (Windows), ROCm 7.13 wheels from repo.amd.com.
#   .\setup_amd.ps1                       # Strix Halo (Radeon 8060S, gfx1151)
#   .\setup_amd.ps1 -Gfx gfx120X-all      # Radeon RX 9070 / 9070 XT (gfx1201)
#   .\setup_amd.ps1 -Gfx gfx110X-all      # Radeon Pro W7900 / RX 7900 (gfx1100)
# Linux uses the same index ending in linux_x86_64 wheels; there, also install
# flash-linear-attention==0.5.2 (Triton exists for ROCm on Linux only) for the fused linear-attention kernels.
param([string]$Gfx = "gfx1151")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$idx = "https://repo.amd.com/rocm/whl/$Gfx/"
"torch==2.11.0+rocm7.13.0`ntorchvision==0.26.0+rocm7.13.0" | Set-Content -Encoding ascii constraints-rocm.txt
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^(torch|flash-linear-attention)' } | Set-Content -Encoding ascii requirements-amd.txt

uv venv -p 3.12 .venv
uv pip install -p .venv --index-url $idx --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match `
    -c constraints-rocm.txt torch torchvision -r requirements-amd.txt
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

if (-not (Test-Path weights\jeveration.pt)) {
    .venv\Scripts\hf download linlinlinlinsdada/Jeveration --local-dir weights
}
"Once per GPU (~10 min, GEMM tuning): .venv\Scripts\python tools\pretune.py --batch-sizes 1 4"
