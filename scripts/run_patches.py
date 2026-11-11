"""Patch-level grounding readout: how strongly each object name is decoded from the image tokens.

    python -m scripts.run_patches                       # all 9 conditions, all 500 POPE images
    python -m scripts.run_patches --conditions blur_s4 --limit-images 3   # smoke test

LLaVA puts the 576 image tokens before the question and attends causally, so the image-token hidden
states do not depend on the question: one forward pass per image scores every object at once.

For each image, layer l in LAYERS and image patch p, the patch state is read out through the final norm
and the LM head (logit lens on image tokens), giving log p_l,p(token). For each POPE object name we
record, over the 576 patches:
    max, mean of top-5, mean     of the log-prob of the name's first sub-token  ("first")
    max, mean of top-5, mean     of the mean log-prob over all the name's sub-tokens  ("all")
Output: $VCD_OUTPUT_ROOT/patches_<kind>_s<level>.npz with
    images (n,), objects (k,), layers (L,), stats float16 (n, L, k, 2 [first/all], 3 [max/top5/mean])
Resumable per image via a sidecar _partial.npz written every 50 images.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.run_layers import CONDITIONS, cond_name, final_norm
from scripts.run_pope import OUTPUT_ROOT
from src.model import encode, load_model
from src.paths import POPE_SPLITS, degraded_image_path, pope_question_file
import re

LAYERS = (8, 12, 16, 20, 24, 28, 32)
TOPK = 5


def pope_images_and_objects():
    images, objects = set(), set()
    for split in POPE_SPLITS:
        with open(pope_question_file(split)) as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    images.add(q["image"])
                    objects.add(re.sub(r"^Is there an? (.*) in the im.*$", r"\1", q["text"]).strip().lower())
    return sorted(images), sorted(objects)


def object_token_ids(tok, objects):
    """Sub-token ids of each object name as it would appear mid-sentence (SentencePiece word start)."""
    ids = []
    for o in objects:
        t = tok(o, add_special_tokens=False).input_ids
        assert t, o
        ids.append(t)
    return ids


@torch.inference_mode()
def image_stats(model, norm, lm_head, input_ids, pixel_values, image_token_id, obj_ids):
    out = model(input_ids=input_ids.to(model.device), pixel_values=pixel_values.to(model.device, model.dtype),
                output_hidden_states=True, use_cache=False)
    pos = (input_ids[0] == image_token_id).nonzero().squeeze(1).to(model.device)
    assert len(pos) == 576, len(pos)
    first = torch.tensor([t[0] for t in obj_ids], device=model.device)
    flat = torch.tensor([i for t in obj_ids for i in t], device=model.device)
    owner = torch.tensor([j for j, t in enumerate(obj_ids) for _ in t], device=model.device)
    counts = torch.bincount(owner).float()
    res = np.empty((len(LAYERS), len(obj_ids), 2, 3), np.float32)
    n_layers = len(out.hidden_states) - 1
    for li, l in enumerate(LAYERS):
        h = out.hidden_states[l][0, pos]
        if l < n_layers:
            h = norm(h)
        logp = torch.log_softmax(lm_head(h).float(), dim=-1)          # (576, vocab)
        s_first = logp[:, first]                                       # (576, k)
        s_all = torch.zeros(576, len(obj_ids), device=model.device).index_add_(1, owner, logp[:, flat]) / counts
        for si, s in enumerate((s_first, s_all)):
            res[li, :, si, 0] = s.max(0).values.cpu().numpy()
            res[li, :, si, 1] = s.topk(TOPK, dim=0).values.mean(0).cpu().numpy()
            res[li, :, si, 2] = s.mean(0).cpu().numpy()
    return res


def run_condition(model, processor, kind, level, images, objects, obj_ids, norm, lm_head, image_token_id, suffix=""):
    path = OUTPUT_ROOT / f"patches_{cond_name(kind, level)}{suffix}.npz"
    if path.exists():
        print(f"{path.name}: done, skipping", flush=True)
        return
    partial = path.with_name(path.stem + "_partial.npz")
    done = {}
    if partial.exists():
        z = np.load(partial, allow_pickle=True)
        done = dict(zip(z["images"].tolist(), z["stats"]))
    print(f"{path.name}: {len(done)} done, {len(images) - len(done)} to run", flush=True)

    def save(target):
        keys = [i for i in images if i in done]
        np.savez(target, images=np.array(keys), objects=np.array(objects), layers=np.array(LAYERS),
                 stats=np.stack([done[i] for i in keys]).astype(np.float16))

    for n, img in enumerate(tqdm([i for i in images if i not in done], desc=path.stem, mininterval=30)):
        image = Image.open(degraded_image_path(img, kind, level)).convert("RGB")
        # the question text is irrelevant to the image-token states (causal attention, image comes first)
        input_ids, pixel_values = encode(processor, image, "Is there an object in the image?")
        done[img] = image_stats(model, norm, lm_head, input_ids, pixel_values, image_token_id, obj_ids)
        if (n + 1) % 50 == 0:
            save(partial)
    save(path)
    partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=[cond_name(*c) for c in CONDITIONS])
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    images, objects = pope_images_and_objects()
    suffix = ""
    if args.limit_images:
        images, suffix = images[:args.limit_images], f"_smoke{args.limit_images}"
    model, processor = load_model()
    obj_ids = object_token_ids(processor.tokenizer, objects)
    image_token_id = getattr(model.config, "image_token_id", None) or model.config.image_token_index
    norm = final_norm(model)
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.language_model.lm_head
    print(f"{len(images)} images, {len(objects)} objects, layers {LAYERS}", flush=True)
    for kind, level in CONDITIONS:
        if cond_name(kind, level) in args.conditions:
            run_condition(model, processor, kind, level, images, objects, obj_ids, norm, lm_head, image_token_id, suffix)


if __name__ == "__main__":
    main()
