"""Train the DAE (Phase 2, fix B) on COCO train2014 crops degraded on the fly.

    python -m scripts.train_dae --name dae_v1                 # full run (on ASPIRE: scripts/pbs/train_dae.pbs)
    python -m scripts.train_dae --name smoke --steps 50 --crop 128 --batch 4 --val-every 25 --val-limit 2

Training pairs (degraded crop, clean crop):
- One blind model for every kind and level. Each sample is clean with probability --p-clean
  (so the DAE learns to leave good images alone), otherwise blur or noise with equal odds, at a
  parameter drawn uniformly from --blur-range / --noise-range. The ranges cover the test LEVELS
  with slack, so the model never relies on seeing the exact test parameters.
- Degradation uses src.degradations.apply at native resolution, the same code and uint8
  rounding as the POPE images, so train and test inputs come from one distribution.

Validation: held-out train2014 images (never POPE) at the exact test LEVELS plus clean, at full
resolution. PSNR of the degraded input and of the DAE output, both against the clean image.
best.pt = highest mean output PSNR over the 8 degraded conditions; clean PSNR is logged so a DAE
that damages clean images is visible.

Outputs in $VCD_CHECKPOINT_ROOT/dae/<name>/: config.yaml, last.pt (resumable), best.pt,
train_log.csv, val_log.csv. Rerunning with the same --name resumes from last.pt.
"""
import argparse
import csv
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader, Dataset

from src.degradations import KINDS, LEVELS, apply, degrade
from src.fixes.dae import DAE, save_dae, to_tensor, to_uint8
from src.model import pick_device
from src.paths import CHECKPOINT_ROOT, COCO_TRAIN_IMAGES, coco_train_split

RESUME_IGNORES = {"workers"}  # args that may change when resuming


def read_split(split: str) -> list[str]:
    names = coco_train_split(split).read_text().split()
    assert names and not any("val2014" in n for n in names), "DAE data must come from train2014 only"
    return names


def degrade_crop(img: np.ndarray, y: int, x: int, c: int, kind: str, param: float, seed: int) -> np.ndarray:
    """img[y:y+c, x:x+c] degraded exactly as `apply` would degrade the whole image, at the cost of
    only a margin-padded region. gaussian_filter reads at most int(4*sigma + 0.5) px away
    (truncate=4), so that margin makes the crop bit-identical to blurring the full image; where
    the region meets the image edge, its reflect padding is the full image's. Noise is per-pixel."""
    h, w = img.shape[:2]
    m = int(4 * param + 0.5) if kind == "blur" else 0
    y0, x0 = max(0, y - m), max(0, x - m)
    region = img[y0 : min(h, y + c + m), x0 : min(w, x + c + m)]
    out = np.asarray(apply(Image.fromarray(region), kind, param, seed))
    return out[y - y0 : y - y0 + c, x - x0 : x - x0 + c]


class DegradedCrops(Dataset):
    def __init__(self, names, crop, p_clean, ranges):
        self.names, self.crop, self.p_clean, self.ranges = names, crop, p_clean, ranges

    def __len__(self):
        return len(self.names)

    def sample_degradation(self, rng) -> tuple[str, float]:
        if rng.random() < self.p_clean:
            return "clean", 0.0
        kind = KINDS[rng.integers(len(KINDS))]
        return kind, float(rng.uniform(*self.ranges[kind]))

    def __getitem__(self, i):
        # torch's RNG is seeded per worker from the main seed, so this is reproducible
        rng = np.random.default_rng(torch.randint(0, 2**62, (1,)).item())
        img = np.asarray(Image.open(COCO_TRAIN_IMAGES / self.names[i]).convert("RGB"))
        h, w = img.shape[:2]
        c = min(self.crop, h, w)
        y, x = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        clean = img[y : y + c, x : x + c]
        kind, param = self.sample_degradation(rng)
        deg = clean if kind == "clean" else degrade_crop(img, y, x, c, kind, param, int(rng.integers(2**32)))
        if rng.random() < 0.5:
            clean, deg = clean[:, ::-1], deg[:, ::-1]
        return to_tensor(deg), to_tensor(clean)


VAL_CONDITIONS = [("clean", 0)] + [(k, lvl) for k in KINDS for lvl in range(1, len(LEVELS[k]) + 1)]


def build_val(names: list[str]) -> list[tuple[np.ndarray, dict]]:
    """(clean array, {condition: degraded array}) per image, at the exact test LEVELS."""
    val = []
    for name in names:
        img = Image.open(COCO_TRAIN_IMAGES / name).convert("RGB")
        clean = np.asarray(img)
        val.append((clean, {(k, lvl): clean if k == "clean" else np.asarray(degrade(img, k, lvl, name))
                            for k, lvl in VAL_CONDITIONS}))
    return val


@torch.inference_mode()
def validate(model, val, device) -> dict:
    """Mean PSNR (dB) of degraded input and DAE output vs clean, per condition. fp32 forward."""
    model.eval()
    psnr_in = {c: [] for c in VAL_CONDITIONS}
    psnr_out = {c: [] for c in VAL_CONDITIONS}
    for clean, degraded in val:
        for cond, deg in degraded.items():
            # an unchanged clean image gives PSNR = inf, which is the ideal result, not an error
            out = to_uint8(model(to_tensor(deg)[None].to(device))[0])
            with np.errstate(divide="ignore"):
                if cond[0] != "clean":
                    psnr_in[cond].append(peak_signal_noise_ratio(clean, deg, data_range=255))
                psnr_out[cond].append(peak_signal_noise_ratio(clean, out, data_range=255))
    model.train()
    return {c: (np.mean(psnr_in[c]) if psnr_in[c] else float("nan"), np.mean(psnr_out[c])) for c in VAL_CONDITIONS}


def append_csv(path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def prepare_run_dir(args) -> tuple:
    run_dir = CHECKPOINT_ROOT / "dae" / args.name
    cfg = {k: v for k, v in vars(args).items()}
    saved = run_dir / "config.yaml"
    if saved.exists():
        previous = yaml.safe_load(saved.read_text())
        strip = lambda d: {k: v for k, v in d.items() if k not in RESUME_IGNORES}
        if strip(previous) != strip(cfg):
            raise SystemExit(f"{run_dir} exists with a different config; use a new --name")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        saved.write_text(yaml.safe_dump(cfg))
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--p-clean", type=float, default=0.2)
    p.add_argument("--blur-range", type=float, nargs=2, default=[2.0, 20.0])
    p.add_argument("--noise-range", type=float, nargs=2, default=[0.1, 1.1])
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-limit", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = prepare_run_dir(args)
    device = pick_device()
    use_amp = device.type == "cuda"  # bf16 autocast on A100; fp32 elsewhere

    model = DAE(args.base, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    lr_at = lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, best = 0, -float("inf")
    if (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed from step {step}")
    print(f"{run_dir} on {device}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    # seeding by (seed, step) keeps a resumed run reproducible, though not identical to an unbroken one
    random.seed(args.seed + step)
    np.random.seed(args.seed + step)
    torch.manual_seed(args.seed + step)

    ranges = {"blur": tuple(args.blur_range), "noise": tuple(args.noise_range)}
    data = DegradedCrops(read_split("train"), args.crop, args.p_clean, ranges)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    t0 = time.perf_counter()
    val = build_val(read_split("val")[: args.val_limit])
    print(f"train {len(data)} images, val {len(val)} images x {len(VAL_CONDITIONS)} conditions "
          f"({time.perf_counter() - t0:.0f}s to build)")

    def evaluate_and_save():
        nonlocal best
        res = validate(model, val, device)
        for (kind, level), (pin, pout) in res.items():
            append_csv(run_dir / "val_log.csv", dict(step=step, kind=kind, level=level, psnr_in=round(pin, 3), psnr_out=round(pout, 3)))
        score = np.mean([pout for (kind, _), (_, pout) in res.items() if kind != "clean"])
        print(f"val step {step}: degraded mean PSNR out {score:.2f} dB, clean {res[('clean', 0)][1]:.2f} dB | "
              + " ".join(f"{k[0]}{lvl}:{pin:.1f}->{pout:.1f}" for (k, lvl), (pin, pout) in res.items() if k != "clean"))
        if score > best:
            best = score
            save_dae(model, run_dir / "best.pt", step=step, val_score=score)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best}, run_dir / "last.pt")

    model.train()
    running, t_log = [], time.perf_counter()
    while step < args.steps:
        for deg, clean in loader:
            deg, clean = deg.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(deg)
            loss = F.l1_loss(out.float(), clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            if step % args.log_every == 0:
                dt = (time.perf_counter() - t_log) / args.log_every
                row = dict(step=step, loss=round(float(np.mean(running)), 5), lr=sched.get_last_lr()[0], sec_per_step=round(dt, 4))
                append_csv(run_dir / "train_log.csv", row)
                print(row, flush=True)
                running, t_log = [], time.perf_counter()
            if step % args.val_every == 0 or step == args.steps:
                evaluate_and_save()
            if step >= args.steps:
                break
    print(f"done: best degraded mean PSNR {best:.2f} dB, weights {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
