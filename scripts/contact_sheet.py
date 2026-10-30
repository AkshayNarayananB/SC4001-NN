"""Grid of sample images x degradation levels, shown at LLaVA's 336px input view.

    python -m scripts.contact_sheet --n 4 --out results/contact_sheet.png
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.download_pope import load_image_names
from src.degradations import KINDS, LEVELS
from src.paths import degraded_image_path

TILE = 336  # LLaVA-1.5 processor: resize shortest side to 336, then center crop
LABEL_H = 20


def model_view(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = TILE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - TILE) // 2, (h - TILE) // 2
    return img.crop((left, top, left + TILE, top + TILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="number of sample images (rows)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/contact_sheet.png")
    args = parser.parse_args()

    available = [n for n in load_image_names() if degraded_image_path(n, "clean").exists()]
    names = random.Random(args.seed).sample(available, min(args.n, len(available)))
    columns = [("clean", 0)] + [(k, lvl) for k in KINDS for lvl in range(1, len(LEVELS[k]) + 1)]

    sheet = Image.new("RGB", (TILE * len(columns), LABEL_H + TILE * len(names)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (kind, level) in enumerate(columns):
        label = "clean" if kind == "clean" else f"{kind} s{level} ({LEVELS[kind][level - 1]})"
        draw.text((col * TILE + 6, 4), label, fill="black")
        for row, name in enumerate(names):
            tile = model_view(Image.open(degraded_image_path(name, kind, level)).convert("RGB"))
            sheet.paste(tile, (col * TILE, LABEL_H + row * TILE))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
