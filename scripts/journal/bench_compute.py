"""Latency and memory per POPE question for every method (compute-aware evaluation).

    python -m scripts.journal.bench_compute --model llava-1.5-7b --precision fp16 int8 int4 --n 200

Methods, all scoring the first answer token of one question:
  baseline     1 forward pass
  probe        1 forward pass + one dot product on the last-token hidden state(s)
  vcd          real + VCD-noised image, one batched pass of 2
  viewlayer    5 views (orig, blur 1/2/3, hflip), one batched pass of 5
  dae+probe    DAE restoration (U-Net, fp32) + original and restored image, batched pass of 2
Times include preprocessing (image transforms + processor), which a deployed system also pays.
Writes results/journal/compute_<model>.csv (median and p90 ms, peak GPU MiB, precision).
"""
import argparse
import csv
import json
import time

import numpy as np
import torch
from PIL import Image

from src.paths import POPE_SOURCE_IMAGES, REPO_ROOT, pope_question_file

VIEWS = ("orig", "blur1", "blur2", "blur3", "hflip")


def load_quantized(name, precision):
    from src.vlm import load_vlm
    return load_vlm(name, quant=None if precision == "fp16" else precision)


def timed(fn, n_warm=5):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(fn.n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return np.median(ts), np.percentile(ts, 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--precision", nargs="+", default=["fp16"])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    from scripts.run_layers import make_view, load_view_dae
    qs = [json.loads(l) for l in open(pope_question_file("random"))][: a.n]
    imgs = {q["image"]: Image.open(POPE_SOURCE_IMAGES / q["image"]).convert("RGB") for q in qs}
    load_view_dae()
    out = REPO_ROOT / "results" / "journal" / f"compute_{a.model}.csv"
    rows = []
    for prec in a.precision:
        vlm = load_quantized(a.model, prec)
        w = torch.randn(vlm.model.config.get_text_config().hidden_size, device=vlm.device)
        it = iter(())

        def nextq():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(qs)
                return next(it)

        def baseline():
            q = nextq()
            vlm.forward(vlm.encode(imgs[q["image"]], q["text"]), hidden=False)

        def probe():
            q = nextq()
            _, hs = vlm.forward(vlm.encode(imgs[q["image"]], q["text"]))
            float(hs[-8].to(w.dtype) @ w)

        def vcd():
            q = nextq()
            x = vlm.encode(imgs[q["image"]], q["text"])
            vlm.forward_batch([x, vlm.add_vcd_noise(x, 999)], hidden=False)

        def viewlayer():
            q = nextq()
            vlm.forward_batch([vlm.encode(make_view(imgs[q["image"]], v), q["text"]) for v in VIEWS])

        def dae_probe():
            q = nextq()
            im = imgs[q["image"]]
            _, hs = vlm.forward_batch([vlm.encode(im, q["text"]), vlm.encode(make_view(im, "dae"), q["text"])])

        for name, fn in [("baseline", baseline), ("probe", probe), ("vcd", vcd), ("viewlayer", viewlayer),
                         ("dae+probe", dae_probe)]:
            fn.n = a.n
            torch.cuda.reset_peak_memory_stats()
            med, p90 = timed(fn)
            row = dict(model=a.model, precision=prec, method=name, ms_median=round(med, 1), ms_p90=round(p90, 1),
                       peak_mib=round(torch.cuda.max_memory_allocated() / 2**20), gpu=torch.cuda.get_device_name())
            print(row, flush=True)
            rows.append(row)
        del vlm
        torch.cuda.empty_cache()
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
