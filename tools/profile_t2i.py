"""Split one text-to-image request into its stages and list the heaviest GPU ops.

    python tools/profile_t2i.py [--merge-lora] [--trace]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT = "A lone pine tree on a small island in a calm lake at dawn."


def timed(name, fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    print(f"{name:>14}: {(time.perf_counter() - t) / n * 1000:7.1f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true", help="print the top GPU ops of one full request")
    ap.add_argument("--fast", action="store_true", help="pipeline with fast=True")
    a = ap.parse_args()
    from jeveration import JeverationPipeline
    kw = {"fast": True} if a.fast else {}
    pipe = JeverationPipeline("weights/jeveration.pt", decoder="weights/decoder512.pt", **kw)
    ids, mask = pipe.ptok([PROMPT])
    ids, mask = ids.cuda(), mask.cuda()
    with torch.no_grad():
        codes = timed("sample_codes", lambda: pipe.model.sample_codes(ids, mask, pipe.generator(0), heads_in_autocast=True))
        timed("clip text", lambda: pipe.codec.text_guidance([PROMPT]))
        timed("decode (all)", lambda: pipe.codec.decode(codes, [PROMPT]))
        timed("end to end", lambda: pipe.text_to_image([PROMPT], pipe.generator(0)))
        if a.trace:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                pipe.text_to_image([PROMPT], pipe.generator(0))
                torch.cuda.synchronize()
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))


if __name__ == "__main__":
    main()
