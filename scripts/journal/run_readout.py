"""Unified single-pass logger for the journal extension (any model in src/vlm.py, any POPE set).

    POPE_SET=pope_fresh python -m scripts.journal.run_readout --model qwen2.5-vl-7b \
        --conditions clean_s0 blur_s4 lowlight_s2 --views orig --vcd

For every unique (image, question) of the POPE set and every condition/view it logs:
  m        first-token Yes-No margin (the baseline score; answer "yes" iff m > 0)
  yes, no, top   logsumexp of the Yes / No token variants and the max logit (for VCD's mask)
  lm       logit-lens margin of every layer 1..L (final norm + LM head); lm[-1] == m
  hidden   last-token hidden states at HIDDEN_FRACS of the depth (float16, _hidden.f16)
With --vcd, a second pass on the VCD-noised image (T=999) adds cd_yes, cd_no (contrast branch).
Degradations are applied on the fly (deterministic, bit-identical to the stored PNGs).
Output: $VCD_OUTPUT_ROOT/journal/<POPE_SET>/<model>/<kind>_s<level>[__<view>]/{rows.jsonl,_hidden.f16,meta.json}
Resumable: rows are appended as they finish; a cut-off hidden row is dropped on restart.
"""
import argparse
import json
import zlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.run_pope import OUTPUT_ROOT
from src.degradations import ALL_LEVELS, degrade
from src.paths import POPE_SET, POPE_SOURCE_IMAGES, POPE_SPLITS, pope_question_file
from src.vlm import load_vlm

HIDDEN_FRACS = (0.5, 0.625, 0.75, 1.0)
VCD_STEP = 999
ROOT = OUTPUT_ROOT / "journal" / POPE_SET


def all_conditions() -> list[str]:
    return ["clean_s0"] + [f"{k}_s{l}" for k in ALL_LEVELS for l in (1, 2, 3, 4)]


def parse_cond(c: str):
    kind, lvl = c.rsplit("_s", 1)
    return kind, int(lvl)


def unique_questions(limit=None):
    seen = {}
    for split in POPE_SPLITS:
        for line in open(pope_question_file(split)):
            if line.strip():
                q = json.loads(line)
                seen.setdefault((q["image"], q["text"]), q)
    keys = sorted(seen)
    if limit:
        imgs = sorted({k[0] for k in keys})[:limit]
        keys = [k for k in keys if k[0] in set(imgs)]
    return keys


def make_view(img, view, dae_name="dae_v1"):
    if view == "orig":
        return img
    from scripts.run_layers import _dae, load_view_dae, make_view as mv
    if view == "dae":
        load_view_dae(dae_name)
    return mv(img, view)


def run(vlm, cond, view, keys, do_vcd, out_root):
    kind, level = parse_cond(cond)
    d = out_root / vlm.name / (cond + ("" if view == "orig" else f"__{view}"))
    d.mkdir(parents=True, exist_ok=True)
    L = vlm.n_layers
    layers = sorted({max(1, round(f * L)) for f in HIDDEN_FRACS})
    meta = dict(model=vlm.name, pope_set=POPE_SET, kind=kind, level=level, view=view,
                param=None if kind == "clean" else ALL_LEVELS[kind][level - 1], vcd=do_vcd,
                vcd_step=VCD_STEP, hidden_layers=layers, n_layers=L, yes_ids=vlm.yes_ids, no_ids=vlm.no_ids)
    mp = d / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if {k: old.get(k) for k in meta} != meta:
            raise SystemExit(f"{d}: meta mismatch, refusing to resume\n{old}\n{meta}")
    mp.write_text(json.dumps(meta, indent=1))
    if (d / "DONE").exists():
        print(f"{d.name}: done")
        return

    hid_dim = vlm.model.config.get_text_config().hidden_size
    row_bytes = len(layers) * hid_dim * 2
    rows_p, hid_p = d / "rows.jsonl", d / "_hidden.f16"
    n_hid = hid_p.stat().st_size // row_bytes if hid_p.exists() else 0
    done = []
    if rows_p.exists():
        done = [json.loads(l) for l in open(rows_p)][:n_hid]
        with open(rows_p, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in done)
    if hid_p.exists():
        with open(hid_p, "r+b") as f:
            f.truncate(len(done) * row_bytes)
    have = {(r["image"], r["text"]) for r in done}
    todo = [k for k in keys if k not in have]
    print(f"{vlm.name}/{d.name}: {len(have)} done, {len(todo)} to run", flush=True)

    cur, img = None, None
    with open(rows_p, "a") as fr, open(hid_p, "ab") as fh:
        for name, text in tqdm(todo, desc=d.name, mininterval=60):
            if name != cur:
                cur = name
                img = Image.open(POPE_SOURCE_IMAGES / name).convert("RGB")
                if kind != "clean":
                    img = degrade(img, kind, level, name)
                img = make_view(img, view)
            inputs = vlm.encode(img, text)
            if do_vcd:   # real and contrast branch in one batched pass
                g = torch.Generator().manual_seed(zlib.crc32(f"{name}|{text}|{cond}|{view}".encode()))
                lg, hb = vlm.forward_batch([inputs, vlm.add_vcd_noise(inputs, VCD_STEP, g)])
                logits, cd_logits, hs = lg[0], lg[1], tuple(h[0] for h in hb)
            else:
                logits, hs = vlm.forward(inputs)
            ll = vlm.layer_logits(hs)
            lse = lambda t, ids: torch.logsumexp(t[..., ids], -1)
            lm = (lse(ll, vlm.yes_ids) - lse(ll, vlm.no_ids)).tolist() + [vlm.margin(logits)]
            row = dict(image=name, text=text, m=round(lm[-1], 4),
                       yes=round(float(lse(logits, vlm.yes_ids)), 4), no=round(float(lse(logits, vlm.no_ids)), 4),
                       top=round(float(logits.max()), 4), lm=[round(x, 3) for x in lm])
            if do_vcd:
                row.update(cd_yes=round(float(lse(cd_logits, vlm.yes_ids)), 4),
                           cd_no=round(float(lse(cd_logits, vlm.no_ids)), 4))
            fh.write(torch.stack([hs[l] for l in layers]).to(torch.float16).cpu().numpy().tobytes())
            fr.write(json.dumps(row) + "\n")
            fh.flush()
            fr.flush()
    (d / "DONE").write_text("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--conditions", nargs="+", default=all_conditions())
    ap.add_argument("--views", nargs="+", default=["orig"])
    ap.add_argument("--vcd", action="store_true")
    ap.add_argument("--limit-images", type=int, default=None, help="smoke test: first N images")
    ap.add_argument("--tag", default="", help="suffix for the output root (e.g. separate smoke variants)")
    args = ap.parse_args()
    out_root = ROOT if args.limit_images is None else ROOT / f"smoke{args.limit_images}"
    if args.tag:
        out_root = out_root.with_name(out_root.name + "_" + args.tag)
    keys = unique_questions(args.limit_images)
    vlm = load_vlm(args.model)
    for view in args.views:
        for cond in args.conditions:
            run(vlm, cond, view, keys, args.vcd, out_root)


if __name__ == "__main__":
    main()
