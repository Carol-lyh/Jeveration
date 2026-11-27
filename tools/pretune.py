"""Tune the GEMMs of text-to-image once for every prompt-length bucket (fast mode, ROCm), so that no request pays for it.

    python tools/pretune.py [--batch-sizes 1 4]      # ~10 s per bucket and batch size; the result: tuning/*.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/jeveration.pt")
    ap.add_argument("--decoder", default="weights/decoder512.pt")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    from jeveration import JeverationPipeline, config
    pipe = JeverationPipeline(a.weights, decoder=a.decoder, fast=True)
    word = pipe.ptok.tok(" stone", add_special_tokens=False)["input_ids"]
    assert len(word) == 1
    step, fixed = pipe.ptok.pad_multiple, len(pipe.ptok.head_ids) + len(pipe.ptok.tail_ids)
    for b in a.batch_sizes:
        for n in range(step, fixed + config.MAX_PROMPT_TOKENS + step, step):      # one prompt per padded length
            k = max(1, min(config.MAX_PROMPT_TOKENS, n - fixed))
            prompt = "A" + " stone" * (k - 1)
            t = time.perf_counter()
            pipe.text_to_image([prompt] * b, pipe.generator(0))
            print(f"batch {b}, length {len(pipe.ptok([prompt])[0][0])}: {time.perf_counter() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
