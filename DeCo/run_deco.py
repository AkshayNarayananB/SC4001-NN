"""DeCo on POPE: one forward per question with hidden states, first-token DeCo decisions.

Logged per question:
    - final-layer Yes/No logits (the baseline), anchor layer, p_max, candidate-set size, and the
      anchor layer's Yes/No logits, so AUROC can be recomputed for any alpha
    - the exact full-vocabulary greedy first token under DeCo for every alpha in ALPHAS
    - Yes/No logits of EVERY layer (lm_head(norm(h_l))), for the layer-wise analysis: does the
      object evidence survive in the middle layers when the final answer gets it wrong?
Prompt is the Stage 0-2 prompt (src.model.encode), so the final layer reproduces the baseline. (The
official DeCo POPE script drops the "one word" suffix; we keep it for comparability with Stage 2.)

    python -m DeCo.run_deco --condition-index 8 [--limit 30]     # 0..8 = clean, blur s1-4, noise s1-4
Writes $VCD_OUTPUT_ROOT/deco_<kind>_s<level>/<split>.jsonl (resumable).
"""
import argparse
import json
import time

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from DeCo.deco import ALPHA, EARLY_EXIT_LAYERS, TOP_K, TOP_P, candidate_tokens, deco_scores, layer_logits, select_anchor
from fix_eval.common import CONDITIONS, OUTPUT_ROOT, POPE_SPLITS
from src.model import answer_token_ids, encode, load_model
from src.paths import degraded_image_path, pope_question_file

ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5)


def first_token_answer(token_id: int, answer_ids: dict[str, int]) -> str:
    return {v: k for k, v in answer_ids.items()}.get(token_id, "other")


@torch.inference_mode()
def score_question(model, processor, q: dict, kind: str, level: int, answer_ids: dict[str, int]) -> dict:
    image = Image.open(degraded_image_path(q["image"], kind, level)).convert("RGB")
    input_ids, pv = encode(processor, image, q["text"])
    out = model(input_ids=input_ids.to(model.device), pixel_values=pv.to(model.device, model.dtype),
                output_hidden_states=True)
    hs = out.hidden_states
    n_layers = len(hs) - 1
    final = out.logits[0, -1].float()
    all_layers = layer_logits(model, hs, range(1, n_layers + 1))      # (n_layers, V)
    early = all_layers[[l - 1 for l in EARLY_EXIT_LAYERS]]
    cand = candidate_tokens(final)
    idx, p_max = select_anchor(early, cand)
    anchor = early[idx]
    y, n = answer_ids["yes"], answer_ids["no"]

    rec = {"question_id": q["question_id"], "image": q["image"], "label": q["label"],
           "logit_yes_final": round(final[y].item(), 4), "logit_no_final": round(final[n].item(), 4),
           "top1_final": int(final.argmax()), "anchor_layer": EARLY_EXIT_LAYERS[idx], "p_max": p_max,
           "n_cand": int(cand.numel()), "yes_in_cand": bool((cand == y).any()), "no_in_cand": bool((cand == n).any()),
           "logit_yes_anchor": round(anchor[y].item(), 4), "logit_no_anchor": round(anchor[n].item(), 4),
           "layer_yes": [round(v, 3) for v in all_layers[:, y].tolist()],
           "layer_no": [round(v, 3) for v in all_layers[:, n].tolist()]}
    for a in ALPHAS:
        rec[f"ans_deco_a{a}"] = first_token_answer(int(deco_scores(final, anchor, p_max, cand, a).argmax()), answer_ids)
    return rec


def run_condition(model, processor, kind, level, splits, limit, answer_ids):
    run_dir = OUTPUT_ROOT / (f"deco_{kind}_s{level}" + (f"_limit{limit}" if limit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": kind, "level": level, "alphas": list(ALPHAS), "top_p": TOP_P, "top_k": TOP_K,
           "early_exit_layers": list(EARLY_EXIT_LAYERS), "limit": limit}
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
            for q in tqdm(todo, desc=f"deco {kind} s{level} {split}", mininterval=60):
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
