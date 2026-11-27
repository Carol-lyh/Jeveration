# Jeveration on AMD GPUs (ROCm)

Tested: Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), Windows 11, ROCm 7.13, torch 2.11.

## Setup

```powershell
.\setup_amd.ps1                     # Strix Halo; -Gfx gfx120X-all (RX 9070) / gfx110X-all (W7900, RX 7900)
.venv\Scripts\python tools\pretune.py --batch-sizes 1 4      # once per GPU, ~10 min: GEMM tuning for all prompt lengths
```

Linux: the same index (`https://repo.amd.com/rocm/whl/<gfx>/`) has linux_x86_64 wheels. There, also install
`flash-linear-attention==0.5.2` (Triton exists for ROCm on Linux only): fused linear-attention kernels instead of the
PyTorch fallback.

## Speed (512 px text-to-image, Radeon 8060S)

| | per image, batch 1 | per image, batch 4 | first image of a process |
|---|---|---|---|
| as released (`--exact`) | 674 ms | 393 ms | ~39 s |
| `fast` (default of the scripts) | 231 ms | 184 ms | ~3 s |

## What `fast=True` does (`jeveration/fast.py`)

| change | effect on results |
|---|---|
| AMD flash attention (AOTriton) on, `jeveration/__init__.py` (always, also for `--exact`) | rounding only |
| LoRA adapters, image head and decoder weights stored in bfloat16 (autocast casts them anyway) | bit-identical |
| linear attention: the 63-step Python loop of transformers' fallback -> block-doubling triangular inverse | rounding only (97 % identical codes, as for any rounding change) |
| small convolutions without MIOpen (it compiles 13 s per process on Windows) | rounding only |
| prompts left-padded to multiples of 16 tokens, so that few GEMM shapes occur | same quality, other details (~83 % identical codes) |
| PyTorch TunableOp: fastest GEMM kernel per shape, kept in `tuning/tunableop_<arch>0.csv` | rounding only |

Check: `python tools/compare_fast.py` (each step against the reference), `python bench_t2i.py`, `python tools/profile_t2i.py --fast`.

Not usable: merging LoRA into the bfloat16 weights (the small updates drown in the rounding: visibly worse images),
`torch.linalg.solve_triangular` (~30x slower on ROCm), a power-series inverse (unstable).
