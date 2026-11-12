"""First-token contrast logits for the ablation VCD variants (zero / shuffle hooks), all conditions.

    python -m scripts.run_contrast --modes zero shuffle

For each condition, runs the contrast branch (src/hooks.py) once per unique (image, question) that
scripts/run_layers.py already processed, and records the final-layer Yes/No logits. The real-branch
logits are not recomputed: they are read from the layer grid's cache (layer 32).

Output: $VCD_OUTPUT_ROOT/contrast_<mode>_<kind>_s<level>/_cache.jsonl, one row per unique question:
    {image, text, hidx, cd_yes, cd_no}
Resumable: finished rows are kept and only missing ones are run.
The shuffle hook is random, so each question is seeded by crc32(image|text|mode).
"""
import argparse
import json
import zlib

import torch
from PIL import Image
from tqdm import tqdm

from scripts.run_layers import CONDITIONS, cond_name, make_view
from scripts.run_pope import OUTPUT_ROOT
from src.hooks import visual_contrast_hook
from src.model import answer_token_ids, encode, load_model
from src.paths import degraded_image_path


def contrast_dir(mode: str, kind: str, level: int):
    return OUTPUT_ROOT / f"contrast_{mode}_{cond_name(kind, level)}"


def layer_cache_rows(kind: str, level: int) -> list[dict]:
    path = OUTPUT_ROOT / f"layers_{cond_name(kind, level)}" / "_cache.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f]


@torch.inference_mode()
def run_condition(model, processor, mode, kind, level, ids):
    run_dir = contrast_dir(mode, kind, level)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "_cache.jsonl"
    done = set()
    if cache_path.exists():
        with open(cache_path) as f:
            done = {(r["image"], r["text"]) for r in map(json.loads, f)}
    todo = [r for r in layer_cache_rows(kind, level) if (r["image"], r["text"]) not in done]
    print(f"{run_dir.name}: {len(done)} done, {len(todo)} to run", flush=True)

    current, image = None, None
    with open(cache_path, "a") as out:
        for r in tqdm(todo, desc=run_dir.name, mininterval=30):
            if r["image"] != current:
                image = Image.open(degraded_image_path(r["image"], kind, level)).convert("RGB")
                current = r["image"]
            input_ids, pixel_values = encode(processor, image, r["text"])
            torch.manual_seed(zlib.crc32(f"{r['image']}|{r['text']}|{mode}".encode()))
            with visual_contrast_hook(model, mode=mode):
                logits = model(input_ids=input_ids.to(model.device),
                               pixel_values=pixel_values.to(model.device, model.dtype)).logits[:, -1].float()
            out.write(json.dumps({"image": r["image"], "text": r["text"], "hidx": r["hidx"],
                                  "cd_yes": round(logits[0, ids["yes"]].item(), 4),
                                  "cd_no": round(logits[0, ids["no"]].item(), 4)}) + "\n")
            out.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["zero", "shuffle"], choices=["zero", "shuffle"])
    args = parser.parse_args()
    model, processor = load_model()
    ids = answer_token_ids(processor)
    for mode in args.modes:
        for kind, level in CONDITIONS:
            run_condition(model, processor, mode, kind, level, ids)


if __name__ == "__main__":
    main()
