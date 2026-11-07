"""Figure: what each contrast branch sees, at LLaVA's 336 px input (for the report).

Rows are images; for each condition (clean, blur s4, noise s4) the columns are the real input,
VCD's T=999 diffusion-noise contrast, the mean null and the phase-scrambled null. CPU only, no model.

    python -m model_fix_2.show_nulls [--n 3] [--out results/fix2_nulls_examples.png]
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

from model_fix_2.nulls import CLIP_MEAN, CLIP_STD, make_null
from src.model import MODEL_ID
from src.paths import REPO_ROOT, degraded_image_path, pope_question_file
from src.vcd import add_diffusion_noise

TILE, LABEL_H = 224, 18


def to_pil(pv: torch.Tensor) -> Image.Image:
    x = (pv[0] * CLIP_STD[0] + CLIP_MEAN[0]).clamp(0, 1)
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((TILE, TILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "fix2_nulls_examples.png"))
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    with open(pope_question_file("random")) as f:
        names = list(dict.fromkeys(json.loads(line)["image"] for line in f))[: args.n]
    conditions = [("clean", 0), ("blur", 4), ("noise", 4)]
    columns = ["input", "VCD T=999", "mean null", "phase null"]
    sheet = Image.new("RGB", (TILE * len(columns) * len(conditions), LABEL_H + TILE * len(names)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, (kind, level) in enumerate(conditions):
        for j, col in enumerate(columns):
            draw.text(((c * len(columns) + j) * TILE + 4, 3), f"{kind}{level or ''}: {col}", fill="black")
        for r, name in enumerate(names):
            img = Image.open(degraded_image_path(name, kind, level)).convert("RGB")
            pv = processor.image_processor(img, return_tensors="pt")["pixel_values"]
            tiles = [pv, add_diffusion_noise(pv, 999, torch.Generator().manual_seed(0)),
                     make_null("mean", pv, name), make_null("phase", pv, name)]
            for j, t in enumerate(tiles):
                sheet.paste(to_pil(t), ((c * len(columns) + j) * TILE, LABEL_H + r * TILE))
    sheet.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
