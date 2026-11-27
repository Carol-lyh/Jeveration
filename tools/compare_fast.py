"""fast=True against the reference computation: same seeds -> share of identical codes, pixel difference, side-by-side PNGs.

    python tools/compare_fast.py --out compare/
"""
import argparse
import gc
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS = ["A stone cottage with a thatched roof and a flower garden in the countryside.",
           "A lone pine tree on a small island in a calm lake at dawn.",
           "An oil painting of sunflowers in a blue ceramic vase.",
           "A red fox sitting in fresh snow, golden hour light."]


def codes_and_images(pipe):
    out = [pipe.text_to_image([p], pipe.generator(i), return_codes=True) for i, p in enumerate(PROMPTS)]
    return [(im[0], c[0].cpu()) for im, c in out]


def report(name, ref, new, out=None):
    same = np.mean([(rc == nc).float().mean().item() for (_, rc), (_, nc) in zip(ref, new)])
    diff = np.mean([np.abs(ri.astype(np.int16) - ni.astype(np.int16)).mean() for (ri, _), (ni, _) in zip(ref, new)])
    print(f"{name:>36}: identical codes {same:6.1%}, mean pixel difference {diff:5.1f} / 255", flush=True)
    if out:
        for i, ((ri, _), (ni, _)) in enumerate(zip(ref, new)):
            Image.fromarray(np.concatenate([ri, ni], 1)).save(os.path.join(out, f"{i}_exact_vs_fast.png"))


def main():
    """Every speed-up of fast=True applied one after the other to a reference pipeline, each compared with the reference."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from jeveration import JeverationPipeline, fast
    pipe = JeverationPipeline("weights/jeveration.pt", decoder="weights/decoder512.pt")
    ref = codes_and_images(pipe)
    report("reference, second run", ref, codes_and_images(pipe))
    steps = [("+ LoRA in bfloat16", lambda: fast.lora_to(pipe.model.lm)),
             ("+ head / decoder weights in bf16", lambda: (fast.store_autocast_weights(pipe.model.head),
                                                           fast.store_autocast_weights(pipe.codec.decoder))),
             ("+ linear attention w/o loop", lambda: fast.patch_linear_attention(pipe.model.lm)),
             ("+ convs without MIOpen", lambda: (fast.replace_slow_convs(pipe.model.lm), fast.replace_slow_convs(pipe.codec.decoder))),
             ("+ prompts padded to 16", lambda: setattr(pipe.ptok, "pad_multiple", 16)),
             ("+ tuned GEMMs", lambda: fast.enable_gemm_tuning(pipe.device))]
    for name, step in steps:
        step()
        new = codes_and_images(pipe)
        report(name, ref, new, a.out if name == steps[-1][0] else None)


if __name__ == "__main__":
    main()
