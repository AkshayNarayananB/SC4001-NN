"""Extra views for ViewLayer+DAE: every layer's Yes-No margin for the DAE-restored image and for VCD's
diffusion-noised contrast image.

ViewLayer+DAE (ViewLayerDAE/analyze.py) scores a question as
    score = (1 + a) * mean_{v in V} m_L(v)  -  a * m_L(v_cd)
ViewLayer/run_viewlayer.py already logged m_l for orig, blur1-3 and hflip, so this runner only adds
    dae   the degraded image restored by the Fix B DAE (always on: the DAE is run on every
          condition, clean and blur included, so the method never needs to know the degradation)
    cd    the degraded image + diffusion noise at T=999, the same eps as the Stage 2 VCD runs
          (seed 55, keyed by split and question), so layer 32 reproduces Stage 2's contrast branch
One batched forward (2 images) per question; margins are read out as in ViewLayer.

    python -m ViewLayerDAE.run_viewlayer_dae --condition-index 8 [--limit 30]   # 0..8 = clean, blur s1-4, noise s1-4
Writes $VCD_OUTPUT_ROOT/viewlayer_dae_<kind>_s<level>/<split>.jsonl (resumable).
"""
import argparse
import json
import time

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from fix_eval.common import CONDITIONS, OUTPUT_ROOT, POPE_SPLITS
from scripts.run_pope import seeded_generator
from src.fixes.dae import load_dae, restore
from src.model import answer_token_ids, encode, load_model
from src.paths import CHECKPOINT_ROOT, degraded_image_path, pope_question_file
from src.vcd import add_diffusion_noise
from ViewLayer.run_viewlayer import layer_margins

EXTRA_VIEWS = ("dae", "cd")
STAGE2_SEED = 55     # seed of the Stage 2 greedy VCD runs (scripts/make_configs.py)
NOISE_STEP = 999


def contrast_pixel_values(pixel_values: torch.Tensor, split: str, question_id) -> torch.Tensor:
    """VCD's contrast input, with exactly the noise the Stage 2 VCD run used for this question."""
    return add_diffusion_noise(pixel_values, NOISE_STEP, seeded_generator(STAGE2_SEED, split, question_id, "noise"))


@torch.inference_mode()
def score_question(model, processor, dae, q: dict, kind: str, level: int, split: str,
                   answer_ids: dict[str, int], cache: dict) -> dict:
    image = Image.open(degraded_image_path(q["image"], kind, level)).convert("RGB")
    input_ids, pv_orig = encode(processor, image, q["text"])
    if q["image"] not in cache:      # one restored image in memory; POPE lists an image's questions together
        cache.clear()
        cache[q["image"]] = processor.image_processor(restore(dae, image), return_tensors="pt")["pixel_values"]
    pv = torch.cat([cache[q["image"]], contrast_pixel_values(pv_orig, split, q["question_id"])])
    out = model(input_ids=input_ids.expand(len(EXTRA_VIEWS), -1).to(model.device),
                pixel_values=pv.to(model.device, model.dtype), output_hidden_states=True, num_logits_to_keep=1)
    margins = layer_margins(model, out.hidden_states, answer_ids["yes"], answer_ids["no"])
    rec = {"question_id": q["question_id"], "image": q["image"], "label": q["label"]}
    for i, v in enumerate(EXTRA_VIEWS):
        rec[f"m_{v}"] = [round(x, 4) for x in margins[i].tolist()]
    return rec


def run_condition(model, processor, dae, dae_name, kind, level, splits, limit, answer_ids):
    run_dir = OUTPUT_ROOT / (f"viewlayer_dae_{kind}_s{level}" + (f"_limit{limit}" if limit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": kind, "level": level, "views": list(EXTRA_VIEWS), "limit": limit, "dae": dae_name,
           "noise_step": NOISE_STEP, "seed": STAGE2_SEED}
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
        t0, cache = time.time(), {}
        with open(path, "a") as out:
            for q in tqdm(todo, desc=f"viewlayer_dae {kind} s{level} {split}", mininterval=60):
                rec = score_question(model, processor, dae, q, kind, level, split, answer_ids, cache)
                out.write(json.dumps(rec) + "\n")
        print(f"{run_dir.name}/{split}: {len(todo)} new, {len(done)} resumed, "
              f"{(time.time() - t0) / max(len(todo), 1):.3f} s/question", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-index", type=int, required=True, help="0..8 = clean, blur s1-4, noise s1-4")
    parser.add_argument("--splits", nargs="+", default=list(POPE_SPLITS))
    parser.add_argument("--limit", type=int, default=None, help="first N questions per split (smoke test)")
    parser.add_argument("--dae", default="dae_v1", help="DAE run name under $VCD_CHECKPOINT_ROOT/dae/")
    args = parser.parse_args()
    kind, level = CONDITIONS[args.condition_index]
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model, processor = load_model()
    dae = load_dae(CHECKPOINT_ROOT / "dae" / args.dae / "best.pt", model.device)
    run_condition(model, processor, dae, args.dae, kind, level, args.splits, args.limit, answer_token_ids(processor))


if __name__ == "__main__":
    main()
