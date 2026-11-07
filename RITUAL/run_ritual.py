"""RITUAL on POPE: first-token scores for the degraded image and all six transformed views.

RITUAL (Woo et al. 2024) decodes from
    p_RITUAL(y) = p(y | v, x) + alpha * p(y | T(v), x)
with T one random transformation per image, alpha = 3, and VCD's plausibility constraint on the
ORIGINAL distribution (tokens with p(y|v) < beta * max p(.|v) are removed, beta = 0.1).
The paper samples; we decode greedily, like the Stage 2 baseline/VCD runs we compare against.
POPE answers are one word, so the greedy first token decides the answer.

One batched forward per question runs [original, hflip, vflip, rotate, color_jitter,
gaussian_blur, crop]. We log Yes/No probabilities per view, plus the exact full-vocabulary greedy
first token for
    ritual_a<alpha>    RITUAL as published: the image's randomly chosen view
    ritual6_a<alpha>   all six views averaged (the paper's multi-view ablation, taken to 6)
for every alpha in ALPHAS, so alpha can be chosen on DEV images only.

    python -m RITUAL.run_ritual --condition-index 8 [--limit 30]   # 0..8 = clean, blur s1-4, noise s1-4
Writes $VCD_OUTPUT_ROOT/ritual_<kind>_s<level>/<split>.jsonl (resumable).
"""
import argparse
import json
import time

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from fix_eval.common import CONDITIONS, OUTPUT_ROOT, POPE_SPLITS
from RITUAL.transforms import TRANSFORMS, apply_transform, choose_transform
from src.model import answer_token_ids, encode, load_model
from src.paths import degraded_image_path, pope_question_file

ALPHAS = (0.5, 1.0, 2.0, 3.0, 5.0)
PAPER_ALPHA = 3.0
BETA = 0.1
SEED = 0


def ritual_scores(p_real: torch.Tensor, p_view: torch.Tensor, alpha: float, beta: float = BETA) -> torch.Tensor:
    """p(y|v) + alpha * p(y|T(v)), with tokens implausible under p(y|v) set to -inf."""
    scores = p_real + alpha * p_view
    return scores.masked_fill(p_real < beta * p_real.max(dim=-1, keepdim=True).values, float("-inf"))


def first_token_answer(token_id: int, answer_ids: dict[str, int]) -> str:
    return {v: k for k, v in answer_ids.items()}.get(token_id, "other")


@torch.inference_mode()
def score_question(model, processor, q: dict, kind: str, level: int, answer_ids: dict[str, int]) -> dict:
    image = Image.open(degraded_image_path(q["image"], kind, level)).convert("RGB")
    input_ids, pv_real = encode(processor, image, q["text"])
    views = [processor.image_processor(apply_transform(image, t, q["image"], SEED), return_tensors="pt")["pixel_values"]
             for t in TRANSFORMS]
    pv = torch.cat([pv_real, *views]).to(model.device, model.dtype)
    ids = input_ids.expand(pv.shape[0], -1).to(model.device)
    logits = model(input_ids=ids, pixel_values=pv, num_logits_to_keep=1).logits[:, -1].float()
    probs = torch.softmax(logits, -1)
    p_real, p_views = probs[0], dict(zip(TRANSFORMS, probs[1:]))
    chosen = choose_transform(q["image"], SEED)

    rec = {"question_id": q["question_id"], "image": q["image"], "label": q["label"], "transform": chosen,
           "maxprob_real": p_real.max().item(), "top1_real": int(p_real.argmax()),
           "logit_yes_real": round(logits[0, answer_ids["yes"]].item(), 4),
           "logit_no_real": round(logits[0, answer_ids["no"]].item(), 4)}
    for name, p in [("real", p_real), *p_views.items()]:
        # full precision: confident answers have tiny probabilities that rounding would zero out
        rec[f"p_yes_{name}"] = p[answer_ids["yes"]].item()
        rec[f"p_no_{name}"] = p[answer_ids["no"]].item()
    p_mean = torch.stack(list(p_views.values())).mean(0)
    for a in ALPHAS:
        rec[f"ans_ritual_a{a}"] = first_token_answer(int(ritual_scores(p_real, p_views[chosen], a).argmax()), answer_ids)
        rec[f"ans_ritual6_a{a}"] = first_token_answer(int(ritual_scores(p_real, p_mean, a).argmax()), answer_ids)
    return rec


def run_condition(model, processor, kind, level, splits, limit, answer_ids):
    run_dir = OUTPUT_ROOT / (f"ritual_{kind}_s{level}" + (f"_limit{limit}" if limit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": kind, "level": level, "transforms": list(TRANSFORMS), "alphas": list(ALPHAS), "beta": BETA,
           "seed": SEED, "limit": limit}
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
            for q in tqdm(todo, desc=f"ritual {kind} s{level} {split}", mininterval=60):
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
