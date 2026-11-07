"""First-token logits of the null contrast branches for every POPE question in one condition.

For each question, one batched forward runs [real, phase-scrambled null, mean null] on the same
prompt (src.model.encode, identical to Stages 0-2). We log Yes/No logits and log-probs per branch,
JS(real || null), and, for each null and each alpha in ALPHAS, the exact greedy first token under
the VCD rule over the full vocabulary (with the plausibility mask). POPE answers are decided by the
first token, so this is everything model_fix_2/analyze.py needs; VCD and baseline come from the
Stage 2 records.

    python -m model_fix_2.run_nulls --kind blur --level 4 [--limit 30]
    python -m model_fix_2.run_nulls --condition-index 8          # 0..8 = clean, blur s1-4, noise s1-4

Writes $VCD_OUTPUT_ROOT/fix2_nulls_<kind>_s<level>/<split>.jsonl (resumable).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from model_fix_2.nulls import NULLS, make_null
from src.decode import js_divergence
from src.model import answer_token_ids, encode, load_model
from src.paths import POPE_SPLITS, REPO_ROOT, degraded_image_path, pope_question_file
from src.vcd import vcd_logits

OUTPUT_ROOT = Path(os.environ.get("VCD_OUTPUT_ROOT", REPO_ROOT / "results" / "raw"))
CONDITIONS = [("clean", 0)] + [(k, l) for k in ("blur", "noise") for l in (1, 2, 3, 4)]
ALPHAS = (0.25, 0.5, 1.0, 2.0, 3.0)
BETA = 0.1
SEED = 0


def run_name(kind: str, level: int) -> str:
    return f"fix2_nulls_{kind}_s{level}"


def load_questions(split: str, limit: int | None) -> list[dict]:
    with open(pope_question_file(split)) as f:
        qs = [json.loads(line) for line in f if line.strip()]
    return qs[:limit] if limit else qs


def first_token_answer(token_id: int, answer_ids: dict[str, int]) -> str:
    return {v: k for k, v in answer_ids.items()}.get(token_id, "other")


@torch.inference_mode()
def score_question(model, processor, q: dict, kind: str, level: int, answer_ids: dict[str, int]) -> dict:
    image = Image.open(degraded_image_path(q["image"], kind, level)).convert("RGB")
    input_ids, pixel_values = encode(processor, image, q["text"])
    branches = {"real": pixel_values, **{n: make_null(n, pixel_values, q["image"], SEED) for n in NULLS}}
    pv = torch.cat(list(branches.values())).to(model.device, model.dtype)
    ids = input_ids.expand(len(branches), -1).to(model.device)
    logits = model(input_ids=ids, pixel_values=pv, num_logits_to_keep=1).logits[:, -1].float()
    by_name = dict(zip(branches, logits))

    rec = {"question_id": q["question_id"], "image": q["image"], "label": q["label"]}
    for name, lg in by_name.items():
        lp = torch.log_softmax(lg, -1)
        rec[f"top1_{name}"] = int(lg.argmax())
        for word, idx in answer_ids.items():
            rec[f"logit_{word}_{name}"] = round(lg[idx].item(), 4)
            rec[f"logprob_{word}_{name}"] = round(lp[idx].item(), 4)
        if name != "real":
            rec[f"js_{name}"] = round(js_divergence(by_name["real"][None], lg[None]).item(), 6)
            for a in ALPHAS:
                tok = int(vcd_logits(by_name["real"][None], lg[None], a, BETA).argmax())
                rec[f"ans_{name}_a{a}"] = first_token_answer(tok, answer_ids)
    rec["ans_real"] = first_token_answer(rec["top1_real"], answer_ids)
    return rec


def run_condition(model, processor, kind: str, level: int, splits, limit, answer_ids):
    # smoke tests (limit set) get their own dir so they never block or pollute the full run
    run_dir = OUTPUT_ROOT / (run_name(kind, level) + (f"_limit{limit}" if limit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": kind, "level": level, "nulls": list(NULLS), "alphas": list(ALPHAS), "beta": BETA,
           "seed": SEED, "limit": limit}
    saved = run_dir / "config.yaml"
    if saved.exists() and yaml.safe_load(saved.read_text()) != cfg:
        raise SystemExit(f"{run_dir} exists with a different config")
    saved.write_text(yaml.safe_dump(cfg))

    for split in splits:
        path = run_dir / f"{split}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["question_id"] for line in open(path)}
        todo = [q for q in load_questions(split, limit) if q["question_id"] not in done]
        t0 = time.time()
        with open(path, "a") as out:
            for q in tqdm(todo, desc=f"{kind} s{level} {split}", mininterval=60):
                out.write(json.dumps(score_question(model, processor, q, kind, level, answer_ids)) + "\n")
        print(f"{run_dir.name}/{split}: {len(todo)} new, {len(done)} resumed, "
              f"{(time.time() - t0) / max(len(todo), 1):.3f} s/question", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["clean", "blur", "noise"])
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--condition-index", type=int, help="0..8 = clean, blur s1-4, noise s1-4")
    parser.add_argument("--splits", nargs="+", default=list(POPE_SPLITS))
    parser.add_argument("--limit", type=int, default=None, help="first N questions per split (smoke test)")
    args = parser.parse_args()
    kind, level = CONDITIONS[args.condition_index] if args.condition_index is not None else (args.kind, args.level)

    model, processor = load_model()
    answer_ids = answer_token_ids(processor)
    run_condition(model, processor, kind, level, args.splits, args.limit, answer_ids)


if __name__ == "__main__":
    main()
