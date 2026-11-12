"""One forward pass per POPE question and condition, logging every layer's first-token readout.

    python -m scripts.run_layers                      # all 9 conditions, all 3 splits
    python -m scripts.run_layers --conditions blur_s4 noise_s4 --limit 30   # smoke test

For every question this logs, for layers 1..32 (read out through the final norm and the LM head,
as DoLa does), the Yes logit, the No logit and the full-vocabulary entropy. Layer 32 is the
baseline. Because every entropy-routed variant (hard min-entropy, soft routing, top-k, gradient
early exit, a fixed layer) is a function of these numbers only, all of them, and any of their
hyperparameters, can be scored offline by scripts/analyze_routing.py with no further GPU work.

Questions are deduplicated per condition by (image, text): the "yes" questions are shared by all
three splits, so this saves about a third of the passes. Output, one row per question:
    $VCD_OUTPUT_ROOT/layers_<kind>_s<level>[__<view>]/<split>.jsonl
With --views, the same readout is logged for test-time views of the image (blur / flip), one
forward pass per view, for the multi-view methods in scripts/analyze_layers.py.
Resumable: unique-question results are appended to _cache.jsonl as they finish.
"""
import argparse
import json

import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from scripts.run_pope import OUTPUT_ROOT, select_questions
from src.model import answer_token_ids, encode, load_model
from src.paths import POPE_SPLITS, degraded_image_path, pope_question_file

HIDDEN_LAYERS = (8, 12, 16, 20, 24, 28, 32)   # last-token hidden states saved for probe experiments
HIDDEN_DIM = 4096
CONDITIONS = [("clean", 0)] + [(k, l) for k in ("blur", "noise") for l in (1, 2, 3, 4)]


VIEWS = ("orig", "blur1", "blur2", "blur3", "hflip")   # ViewLayer's views (origin/main ViewLayer/run_viewlayer.py)


def cond_name(kind: str, level: int) -> str:
    return f"{kind}_s{level}"


def run_dir_name(kind: str, level: int, view: str = "orig") -> str:
    return f"layers_{cond_name(kind, level)}" + ("" if view == "orig" else f"__{view}")


def make_view(img: Image.Image, view: str) -> Image.Image:
    """Deterministic test-time view of the (degraded) image at native resolution; identical to
    ViewLayer's make_view (torchvision gaussian_blur with kernel 2*ceil(3 sigma)+1)."""
    if view == "orig":
        return img
    import math
    import torchvision.transforms.functional as TF
    if view == "hflip":
        return TF.hflip(img)
    if view.startswith("blur"):
        sigma = float(view[4:])
        return TF.gaussian_blur(img, kernel_size=2 * math.ceil(3 * sigma) + 1, sigma=sigma)
    raise ValueError(f"unknown view {view!r}")


def final_norm(model):
    """The language model's final RMSNorm, wherever this transformers version puts it
    (4.46: language_model.model.norm; 5.x: model.language_model.norm)."""
    found = [(n, m) for n, m in model.named_modules()
             if n.endswith("norm") and "language_model" in n and ".layers." not in n]
    if len(found) != 1:
        raise AttributeError(f"final norm not unique: {[n for n, _ in found]}")
    return found[0][1]


@torch.inference_mode()
def layer_readout(model, norm, lm_head, input_ids, pixel_values, yes_id: int, no_id: int) -> dict:
    out = model(input_ids=input_ids.to(model.device), pixel_values=pixel_values.to(model.device, model.dtype),
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # (embeddings, layer 1, ..., layer 32); the last one is already normed
    mid = torch.stack([h[0, -1] for h in hs[1:-1]])               # layers 1..31, pre-norm
    logits = torch.cat([lm_head(norm(mid)), out.logits[:, -1]]).float()   # (32, vocab)
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    r = lambda t: [round(x, 4) for x in t.tolist()]
    hidden = torch.stack([hs[l][0, -1] for l in HIDDEN_LAYERS]).to(torch.float16).cpu().numpy()
    return {"layer_yes": r(logits[:, yes_id]), "layer_no": r(logits[:, no_id]), "layer_ent": r(ent),
            "top1_final": int(logits[-1].argmax())}, hidden


def run_condition(model, processor, kind, level, limit, norm, lm_head, ids, view="orig"):
    run_dir = OUTPUT_ROOT / run_dir_name(kind, level, view)
    run_dir.mkdir(parents=True, exist_ok=True)
    questions = {}
    for split in POPE_SPLITS:
        with open(pope_question_file(split)) as f:
            questions[split] = select_questions([json.loads(l) for l in f if l.strip()], limit)
    if all((run_dir / f"{s}.jsonl").exists() for s in POPE_SPLITS):
        print(f"{run_dir.name}: done, skipping")
        return

    # _hidden.f16: raw float16 rows (len(HIDDEN_LAYERS), HIDDEN_DIM), row i <-> cache entry with hidx i
    cache_path, hidden_path = run_dir / "_cache.jsonl", run_dir / "_hidden.f16"
    row_bytes = len(HIDDEN_LAYERS) * HIDDEN_DIM * 2
    n_hidden = hidden_path.stat().st_size // row_bytes if hidden_path.exists() else 0
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                if row["hidx"] < n_hidden:     # drop a row whose hidden state was cut off mid-write
                    cache[(row["image"], row["text"])] = row
    if hidden_path.exists():
        with open(hidden_path, "r+b") as f:
            f.truncate(len(cache) * row_bytes)
    todo = sorted({(q["image"], q["text"]) for qs in questions.values() for q in qs} - set(cache))
    print(f"{run_dir.name}: {len(cache)} cached, {len(todo)} to run")

    current_image, image = None, None
    with open(cache_path, "a") as out, open(hidden_path, "ab") as hout:
        for image_name, text in tqdm(todo, desc=run_dir.name, mininterval=30):
            if image_name != current_image:
                image = make_view(Image.open(degraded_image_path(image_name, kind, level)).convert("RGB"), view)
                current_image = image_name
            input_ids, pixel_values = encode(processor, image, text)
            readout, hidden = layer_readout(model, norm, lm_head, input_ids, pixel_values, ids["yes"], ids["no"])
            row = {"image": image_name, "text": text, "hidx": len(cache), **readout}
            hout.write(hidden.tobytes())
            hout.flush()
            out.write(json.dumps(row) + "\n")
            out.flush()
            cache[(image_name, text)] = row

    for split, qs in questions.items():
        with open(run_dir / f"{split}.jsonl", "w") as f:
            for q in qs:
                c = cache[(q["image"], q["text"])]
                f.write(json.dumps({"question_id": q["question_id"], "image": q["image"], "label": q["label"],
                                    **{k: v for k, v in c.items() if k not in ("image", "text")}}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=[cond_name(*c) for c in CONDITIONS])
    parser.add_argument("--limit", type=int, default=None, help="questions per split (smoke tests)")
    parser.add_argument("--views", nargs="+", default=["orig"], choices=VIEWS,
                        help="extra test-time views (one forward pass each) for multi-view methods")
    args = parser.parse_args()
    if args.limit is not None:
        global OUTPUT_ROOT
        OUTPUT_ROOT = OUTPUT_ROOT / f"smoke_limit{args.limit}"

    model, processor = load_model()
    ids = answer_token_ids(processor)
    norm = final_norm(model)
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.language_model.lm_head
    for view in args.views:
        for kind, level in CONDITIONS:
            if cond_name(kind, level) in args.conditions:
                run_condition(model, processor, kind, level, args.limit, norm, lm_head, ids, view)


if __name__ == "__main__":
    main()
