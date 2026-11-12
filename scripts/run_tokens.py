"""Extra single-pass features for the readout probe: the queried object's own hidden state, and the image
tokens pooled by their similarity to it. Same forward pass that answers the question; no extra passes.

    python -m scripts.run_tokens                                   # all 9 conditions
    python -m scripts.run_tokens --conditions noise_s4 --limit 30  # smoke test

For each unique (image, question) already processed by scripts/run_layers.py (same order, same hidx), and for
each layer l in LAYERS, saves (float16, row hidx of _tokens.f16):
    obj_l   mean hidden state over the object name's tokens in the question
    pool_l  image tokens (576) pooled with softmax(cos(obj_l, img_tokens_l) / TAU): question-conditioned
            visual evidence, read at the place where the question meets the image
    mean_l  plain mean of the 576 image tokens (question-independent control)
Output: $VCD_OUTPUT_ROOT/layers_<kind>_s<level>/_tokens.f16 with rows (len(LAYERS), 3, 4096); resumable.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from scripts.run_layers import CONDITIONS, cond_name
from scripts.run_pope import OUTPUT_ROOT
from src.model import encode, load_model
from src.paths import degraded_image_path

LAYERS = (16, 20, 24)
KINDS = ("obj", "pool", "mean")
DIM = 4096
TAU = 0.05


def find_sublist(seq, sub):
    for s in range(len(seq) - len(sub), -1, -1):
        if seq[s:s + len(sub)] == sub:
            return s
    return None


@torch.inference_mode()
def token_features(model, input_ids, pixel_values, obj_ids, image_token_id):
    out = model(input_ids=input_ids.to(model.device), pixel_values=pixel_values.to(model.device, model.dtype),
                output_hidden_states=True, use_cache=False)
    ids = input_ids[0].tolist()
    img = [i for i, t in enumerate(ids) if t == image_token_id]
    start = find_sublist(ids, obj_ids)
    assert len(img) == 576 and start is not None and start > img[-1]
    obj_pos = list(range(start, start + len(obj_ids)))
    res = np.empty((len(LAYERS), len(KINDS), DIM), np.float16)
    for li, l in enumerate(LAYERS):
        h = out.hidden_states[l][0].float()
        obj = h[obj_pos].mean(0)
        tok = h[img]
        w = torch.softmax(F.cosine_similarity(tok, obj[None], dim=-1) / TAU, dim=0)
        res[li, 0] = obj.cpu().numpy()
        res[li, 1] = (w[:, None] * tok).sum(0).cpu().numpy()
        res[li, 2] = tok.mean(0).cpu().numpy()
    return res


def run_condition(model, processor, kind, level, limit, image_token_id):
    run_dir = OUTPUT_ROOT / f"layers_{cond_name(kind, level)}"
    with open(run_dir / "_cache.jsonl") as f:
        rows = sorted((json.loads(l) for l in f), key=lambda r: r["hidx"])
    if limit:
        rows = rows[:limit]
    path = run_dir / "_tokens.f16"
    row_bytes = len(LAYERS) * len(KINDS) * DIM * 2
    n_done = path.stat().st_size // row_bytes if path.exists() else 0
    if path.exists():
        with open(path, "r+b") as f:
            f.truncate(n_done * row_bytes)
    print(f"{run_dir.name}/_tokens.f16: {n_done} done, {len(rows) - n_done} to run", flush=True)
    current, image = None, None
    with open(path, "ab") as out:
        for r in tqdm(rows[n_done:], desc=f"tokens_{cond_name(kind, level)}", mininterval=30):
            assert r["hidx"] == n_done
            if r["image"] != current:
                image = Image.open(degraded_image_path(r["image"], kind, level)).convert("RGB")
                current = r["image"]
            obj = r["text"].split("Is there a")[-1].split(" in the im")[0].strip()
            obj = obj[2:] if obj.startswith("n ") else obj   # "an apple"
            obj_ids = processor.tokenizer(obj, add_special_tokens=False).input_ids
            input_ids, pixel_values = encode(processor, image, r["text"])
            out.write(token_features(model, input_ids, pixel_values, obj_ids, image_token_id).tobytes())
            out.flush()
            n_done += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=[cond_name(*c) for c in CONDITIONS])
    parser.add_argument("--limit", type=int, default=None, help="smoke test: first N questions (written to the real file; a later full run resumes after them)")
    args = parser.parse_args()
    model, processor = load_model()
    image_token_id = getattr(model.config, "image_token_id", None) or model.config.image_token_index
    for kind, level in CONDITIONS:
        if cond_name(kind, level) in args.conditions:
            run_condition(model, processor, kind, level, args.limit, image_token_id)


if __name__ == "__main__":
    main()
