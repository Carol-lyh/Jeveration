"""Load the pipeline once, warm up, then time text-to-image requests (end to end, incl. decoding).

    python bench_t2i.py --runs 5 --out bench/
"""
import argparse
import os
import time

import torch
from PIL import Image

PROMPTS = [
    "A stone cottage with a thatched roof and a flower garden in the countryside.",
    "A lone pine tree on a small island in a calm lake at dawn.",
    "An oil painting of sunflowers in a blue ceramic vase.",
    "A red fox sitting in fresh snow, golden hour light.",
]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/jeveration.pt")
    ap.add_argument("--decoder", default="weights/decoder512.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default="bench")
    ap.add_argument("--exact", action="store_true", help="fast=False: the reference computation (slower)")
    a = ap.parse_args()

    from jeveration import JeverationPipeline
    t = time.perf_counter()
    pipe = JeverationPipeline(a.weights, decoder=a.decoder, device=a.device, fast=not a.exact)
    print(f"load: {time.perf_counter() - t:.1f} s on {torch.cuda.get_device_name(0) if pipe.device.type == 'cuda' else 'CPU'}")

    os.makedirs(a.out, exist_ok=True)
    times = []
    for r in range(a.runs + 1):                     # run 0 = warm-up (kernel compilation, allocator)
        batch = [PROMPTS[(r * a.batch_size + j) % len(PROMPTS)] for j in range(a.batch_size)]
        sync(pipe.device)
        t = time.perf_counter()
        images = pipe.text_to_image(batch, pipe.generator(r))
        sync(pipe.device)
        dt = time.perf_counter() - t
        print(f"{'warm-up' if r == 0 else f'run {r}'}: {dt * 1000:.0f} ms for {a.batch_size} image(s)")
        if r:
            times.append(dt)
        for j, im in enumerate(images):
            Image.fromarray(im).save(os.path.join(a.out, f"{r:02d}_{j}.png"))
    times.sort()
    print(f"median: {times[len(times) // 2] * 1000:.0f} ms / batch, {times[len(times) // 2] * 1000 / a.batch_size:.0f} ms / image")
    if pipe.device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
