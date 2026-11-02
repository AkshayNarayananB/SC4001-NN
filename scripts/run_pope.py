"""Run one POPE experiment config.

    python -m scripts.run_pope --config configs/stage0/vcd_sample_seed55.yaml

Per-question records go to $VCD_OUTPUT_ROOT/<name>/<split>.jsonl (resumable: finished question
ids are skipped). Per-split metrics go to results/metrics/<name>.csv.
"""
import argparse
import csv
import json
import os
import random
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.decode import METHODS, generate
from src.eval_pope import pope_metrics
from src.model import answer_token_ids, encode, load_model
from src.degradations import LEVELS, apply
from src.paths import REPO_ROOT, degraded_image_path, pope_question_file
from src.vcd import add_diffusion_noise

OUTPUT_ROOT = Path(os.environ.get("VCD_OUTPUT_ROOT", REPO_ROOT / "results" / "raw"))
METRICS_DIR = REPO_ROOT / "results" / "metrics"
SUBSET_SEED = 0

DEFAULTS = dict(
    method="baseline", alpha=1.0, beta=0.1, noise_step=999,
    decoding="sample", temperature=1.0, seed=55,
    splits=["random", "popular", "adversarial"],
    kind="clean", level=0, limit=None, max_new_tokens=16,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = {**DEFAULTS, **yaml.safe_load(f)}
    assert "name" in cfg, "config needs a unique 'name'"
    if cfg.get("param") is None and cfg["kind"] != "clean":
        # Record the cached level's actual parameter, so a run dir made under different LEVELS
        # fails the config check in prepare_run_dir instead of being silently resumed.
        cfg["level_param"] = LEVELS[cfg["kind"]][cfg["level"] - 1]
    return cfg


def seeded_generator(*parts) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32("|".join(map(str, parts)).encode()))


def build_method(cfg):
    if cfg["method"] == "baseline":
        return METHODS["baseline"]()
    return METHODS[cfg["method"]](alpha=cfg["alpha"], beta=cfg["beta"])


def prepare_run_dir(cfg) -> Path:
    """Create the run dir, or verify an existing one was produced by the same config (resume)."""
    run_dir = OUTPUT_ROOT / cfg["name"]
    saved = run_dir / "config.yaml"
    if saved.exists():
        with open(saved) as f:
            previous = yaml.safe_load(f)
        if previous != cfg:
            raise SystemExit(f"{run_dir} exists with a different config; use a new 'name'")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(saved, "w") as f:
            yaml.safe_dump(cfg, f)
    return run_dir


def load_image(cfg, image_name: str) -> Image.Image:
    """The cached degraded PNG for (kind, level), or, if the config sets an explicit `param`
    (used for calibration sweeps), the clean image degraded on the fly with that parameter."""
    if cfg.get("param") is None:
        return Image.open(degraded_image_path(image_name, cfg["kind"], cfg["level"])).convert("RGB")
    clean = Image.open(degraded_image_path(image_name, "clean")).convert("RGB")
    return apply(clean, cfg["kind"], cfg["param"], zlib.crc32(f"{image_name}|{cfg['kind']}|{cfg['param']}".encode()))


def select_questions(questions: list[dict], limit: int | None) -> list[dict]:
    """All questions, or a fixed random subset of `limit` of them. The subset seed is constant,
    not the run seed, so every method and degradation level sees the same questions. A random
    subset rather than the first N because the files are ordered by image (6 questions each)."""
    if limit is None or limit >= len(questions):
        return questions
    subset = random.Random(SUBSET_SEED).sample(questions, limit)
    return sorted(subset, key=lambda q: q["question_id"])


def run_split(cfg, split, model, processor, method, answer_ids, run_dir) -> list[dict]:
    with open(pope_question_file(split)) as f:
        questions = select_questions([json.loads(line) for line in f if line.strip()], cfg["limit"])

    out_path = run_dir / f"{split}.jsonl"
    done = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["question_id"]] = r

    eos = processor.tokenizer.eos_token_id
    with open(out_path, "a") as out:
        for q in tqdm(questions, desc=split, mininterval=30):
            if q["question_id"] in done:
                continue
            image = load_image(cfg, q["image"])
            input_ids, pixel_values = encode(processor, image, q["text"])
            # Noise depends only on (seed, split, question), so every degradation level of the
            # same question gets the same eps: paired comparisons across conditions.
            pixel_values_cd = (
                add_diffusion_noise(pixel_values, cfg["noise_step"], seeded_generator(cfg["seed"], split, q["question_id"], "noise"))
                if method.needs_contrast else None
            )
            t0 = time.perf_counter()
            tokens, diag = generate(
                model, input_ids, pixel_values, pixel_values_cd, method,
                answer_ids=answer_ids,
                max_new_tokens=cfg["max_new_tokens"],
                sample=cfg["decoding"] == "sample",
                temperature=cfg["temperature"],
                generator=seeded_generator(cfg["seed"], split, q["question_id"], "sample"),
                eos_token_id=eos,
            )
            record = {
                "question_id": q["question_id"], "image": q["image"], "label": q["label"],
                "answer": processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "seconds": round(time.perf_counter() - t0, 4),
                **diag,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[q["question_id"]] = record
    return [done[q["question_id"]] for q in questions]


def run_config(cfg, model, processor, answer_ids):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    run_dir = prepare_run_dir(cfg)
    method = build_method(cfg)
    print(f"run {cfg['name']} on {model.device}, answer ids {answer_ids}, output {run_dir}")

    rows = []
    for split in cfg["splits"]:
        records = run_split(cfg, split, model, processor, method, answer_ids, run_dir)
        metrics = pope_metrics(records)
        row = {
            "name": cfg["name"], "method": cfg["method"], "decoding": cfg["decoding"], "seed": cfg["seed"],
            "alpha": cfg["alpha"] if cfg["method"] != "baseline" else "",
            "beta": cfg["beta"] if cfg["method"] != "baseline" else "",
            "noise_step": cfg["noise_step"] if cfg["method"] != "baseline" else "",
            "kind": cfg["kind"], "level": cfg["level"], "param": cfg.get("param", cfg.get("level_param", "")), "split": split,
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()},
            "sec_per_q": round(np.mean([r["seconds"] for r in records]), 4),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        rows.append(row)
        print(json.dumps(row))

    with open(METRICS_DIR / f"{cfg['name']}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="one or more config files, run in order")
    args = parser.parse_args()
    configs = [load_config(path) for path in args.config]  # validate all before loading the model

    model, processor = load_model()
    answer_ids = answer_token_ids(processor)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        run_config(cfg, model, processor, answer_ids)


if __name__ == "__main__":
    main()
