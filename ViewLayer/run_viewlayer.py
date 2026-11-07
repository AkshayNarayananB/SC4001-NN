"""View x layer readout for POPE: every layer's Yes-No margin, for the degraded image and a few views.

Motivation (our Phase 2 results):
  - RITUAL ablation: a slightly blurred copy of a noisy image raises AUROC (noise s4 0.737 -> 0.751)
    and averaging several views gives the most consistent AUROC gain of any method.
  - DeCo layer table: layer ~20 separates present/absent objects better than the final layer, and the
    gap grows with degradation (blur s4: 0.779 vs 0.750).
Neither paper combines the two. This runner logs everything needed to choose views and a layer on DEV
images only (ViewLayer/analyze.py), so the combination is selected without touching TEST labels.

Views (deterministic; applied to the degraded image at native resolution, like RITUAL):
    orig, blur1, blur2, blur3 (Gaussian sigma 1/2/3 px), hflip
One batched forward per question with hidden states; for each view and each decoder layer l we store
    margin_l = (W_yes - W_no) . norm(h_l)        (layer 32 is already normed by HF Llama)
i.e. the Yes-minus-No logit that layer would give if read out through the LM head (as in DeCo/DoLa).

    python -m ViewLayer.run_viewlayer --condition-index 8 [--limit 30]   # 0..8 = clean, blur s1-4, noise s1-4
Writes $VCD_OUTPUT_ROOT/viewlayer_<kind>_s<level>/<split>.jsonl (resumable).
"""
import argparse
import json
import math
import time

import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from fix_eval.common import CONDITIONS, OUTPUT_ROOT, POPE_SPLITS
from src.model import answer_token_ids, encode, load_model
from src.paths import degraded_image_path, pope_question_file

VIEWS = ("orig", "blur1", "blur2", "blur3", "hflip")


def make_view(img: Image.Image, view: str) -> Image.Image:
    if view == "orig":
        return img
    if view == "hflip":
        return F.hflip(img)
    if view.startswith("blur"):
        sigma = float(view[4:])
        return F.gaussian_blur(img, kernel_size=2 * math.ceil(3 * sigma) + 1, sigma=sigma)
    raise ValueError(f"unknown view {view!r}")


def layer_margins(model, hidden_states, yes: int, no: int) -> torch.Tensor:
    """(batch, n_layers) Yes-minus-No logit of every decoder layer at the last position."""
    lm = model.language_model
    w = (lm.lm_head.weight[yes] - lm.lm_head.weight[no]).float()
    n_layers = len(hidden_states) - 1
    out = []
    for l in range(1, n_layers + 1):
        h = hidden_states[l][:, -1]
        h = h if l == n_layers else lm.model.norm(h)
        out.append(h.float() @ w)
    return torch.stack(out, dim=1)


@torch.inference_mode()
def score_question(model, processor, q: dict, kind: str, level: int, answer_ids: dict[str, int]) -> dict:
    image = Image.open(degraded_image_path(q["image"], kind, level)).convert("RGB")
    input_ids, pv_orig = encode(processor, image, q["text"])
    pvs = [pv_orig] + [processor.image_processor(make_view(image, v), return_tensors="pt")["pixel_values"]
                       for v in VIEWS[1:]]
    pv = torch.cat(pvs).to(model.device, model.dtype)
    ids = input_ids.expand(len(VIEWS), -1).to(model.device)
    out = model(input_ids=ids, pixel_values=pv, output_hidden_states=True, num_logits_to_keep=1)
    margins = layer_margins(model, out.hidden_states, answer_ids["yes"], answer_ids["no"])
    rec = {"question_id": q["question_id"], "image": q["image"], "label": q["label"]}
    for i, v in enumerate(VIEWS):
        rec[f"m_{v}"] = [round(x, 4) for x in margins[i].tolist()]
    return rec


def run_condition(model, processor, kind, level, splits, limit, answer_ids):
    run_dir = OUTPUT_ROOT / (f"viewlayer_{kind}_s{level}" + (f"_limit{limit}" if limit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": kind, "level": level, "views": list(VIEWS), "limit": limit}
    saved = run_dir / "config.yaml"
    if saved.exists() and yaml.safe_load(saved.read_text()) != cfg:
        raise SystemExit(f"{run_dir} exists with a different config")
    saved.write_text(yaml.safe_dump(cfg))
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        done = {json.loads(line)["question_id"] for line in open(path)} if path.exists() else set()
        with open(pope_question_file(split)) as f:
            qs = [json.loads(line) for line in f if line.strip()]
        todo = [q for q in (qs[:limit] if limit else qs) if q["question_id"] not in done]
        t0 = time.time()
        with open(path, "a") as out:
            for q in tqdm(todo, desc=f"viewlayer {kind} s{level} {split}", mininterval=60):
                out.write(json.dumps(score_question(model, processor, q, kind, level, answer_ids)) + "\n")
        print(f"{run_dir.name}/{split}: {len(todo)} new, {len(done)} resumed, "
              f"{(time.time() - t0) / max(len(todo), 1):.3f} s/question", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-index", type=int, required=True, help="0..8 = clean, blur s1-4, noise s1-4")
    parser.add_argument("--splits", nargs="+", default=list(POPE_SPLITS))
    parser.add_argument("--limit", type=int, default=None, help="first N questions per split (smoke test)")
    args = parser.parse_args()
    kind, level = CONDITIONS[args.condition_index]
    model, processor = load_model()
    run_condition(model, processor, kind, level, args.splits, args.limit, answer_token_ids(processor))


if __name__ == "__main__":
    main()
