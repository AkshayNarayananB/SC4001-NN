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


class Sink:
    """Output directory of one (condition, view): meta check, resume state, append-only writers."""

    def __init__(self, vlm, cond, view, do_vcd, out_root):
        kind, level = parse_cond(cond)
        self.d = d = out_root / vlm.name / (cond + ("" if view == "orig" else f"__{view}"))
        d.mkdir(parents=True, exist_ok=True)
        L = vlm.n_layers
        self.layers = sorted({max(1, round(f * L)) for f in HIDDEN_FRACS})
        meta = dict(model=vlm.name, pope_set=POPE_SET, kind=kind, level=level, view=view,
                    param=None if kind == "clean" else ALL_LEVELS[kind][level - 1], vcd=do_vcd,
                    vcd_step=VCD_STEP, hidden_layers=self.layers, n_layers=L, yes_ids=vlm.yes_ids, no_ids=vlm.no_ids)
        mp = d / "meta.json"
        if mp.exists():
            old = json.loads(mp.read_text())
            if {k: old.get(k) for k in meta} != meta:
                raise SystemExit(f"{d}: meta mismatch, refusing to resume\n{old}\n{meta}")
        mp.write_text(json.dumps(meta, indent=1))
        self.finished = (d / "DONE").exists()
        self.have = set()
        if self.finished:
            return
        row_bytes = len(self.layers) * vlm.model.config.get_text_config().hidden_size * 2
        self.rows_p, self.hid_p = d / "rows.jsonl", d / "_hidden.f16"
        n_hid = self.hid_p.stat().st_size // row_bytes if self.hid_p.exists() else 0
        done = []
        if self.rows_p.exists():
            done = [json.loads(l) for l in open(self.rows_p)][:n_hid]
            with open(self.rows_p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in done)
        if self.hid_p.exists():
            with open(self.hid_p, "r+b") as f:
                f.truncate(len(done) * row_bytes)
        self.have = {(r["image"], r["text"]) for r in done}

    def open(self):
        self.fr, self.fh = open(self.rows_p, "a"), open(self.hid_p, "ab")

    def write(self, row, hs):
        self.fh.write(torch.stack([hs[l] for l in self.layers]).to(torch.float16).cpu().numpy().tobytes())
        self.fr.write(json.dumps(row) + "\n")
        self.fh.flush()
        self.fr.flush()
        self.have.add((row["image"], row["text"]))

    def close(self, keys):
        self.fr.close()
        self.fh.close()
        if all(k in self.have for k in keys):
            (self.d / "DONE").write_text("")


def make_row(vlm, name, text, logits, hs, cd_logits=None):
    ll = vlm.layer_logits(hs)
    lse = lambda t, ids: torch.logsumexp(t[..., ids], -1)
    lm = (lse(ll, vlm.yes_ids) - lse(ll, vlm.no_ids)).tolist() + [vlm.margin(logits)]
    row = dict(image=name, text=text, m=round(lm[-1], 4),
               yes=round(float(lse(logits, vlm.yes_ids)), 4), no=round(float(lse(logits, vlm.no_ids)), 4),
               top=round(float(logits.max()), 4), lm=[round(x, 3) for x in lm])
    if cd_logits is not None:
        row.update(cd_yes=round(float(lse(cd_logits, vlm.yes_ids)), 4),
                   cd_no=round(float(lse(cd_logits, vlm.no_ids)), 4))
    return row


def run(vlm, cond, views, keys, do_vcd, out_root):
    """Log one condition for one or several views. Several views of a question share one batched forward
    pass (same image size -> same token length); each view keeps its own run directory. VCD needs one view."""
    assert not (do_vcd and len(views) > 1), "VCD is logged for single-view runs only"
    kind, level = parse_cond(cond)
    sinks = {v: Sink(vlm, cond, v, do_vcd, out_root) for v in views}
    sinks = {v: k for v, k in sinks.items() if not k.finished}
    if not sinks:
        print(f"{vlm.name}/{cond} {views}: done")
        return
    todo = [k for k in keys if any(k not in sk.have for sk in sinks.values())]
    print(f"{vlm.name}/{cond} views {list(sinks)}: {len(todo)} questions to run", flush=True)
    for sk in sinks.values():
        sk.open()
    cur, imgs = None, {}
    for name, text in tqdm(todo, desc=f"{cond}[{','.join(sinks)}]", mininterval=60):
        if name != cur:
            cur = name
            base = Image.open(POPE_SOURCE_IMAGES / name).convert("RGB")
            if kind != "clean":
                base = degrade(base, kind, level, name)
            imgs = {v: make_view(base, v) for v in sinks}
        need = [v for v, sk in sinks.items() if (name, text) not in sk.have]
        enc = [vlm.encode(imgs[v], text) for v in need]
        if do_vcd:   # real and contrast branch in one batched pass
            g = torch.Generator().manual_seed(zlib.crc32(f"{name}|{text}|{cond}|{need[0]}".encode()))
            lg, hb = vlm.forward_batch([enc[0], vlm.add_vcd_noise(enc[0], VCD_STEP, g)])
            sinks[need[0]].write(make_row(vlm, name, text, lg[0], tuple(h[0] for h in hb), lg[1]),
                                 tuple(h[0] for h in hb))
            continue
        if len(enc) == 1:   # unbatched pass: keeps single-view runs bit-identical to earlier logs
            logits, hs1 = vlm.forward(enc[0])
            lg, hb = logits[None], tuple(h[None] for h in hs1)
        else:
            lg, hb = vlm.forward_batch(enc)
        for i, v in enumerate(need):
            hs = tuple(h[i] for h in hb)
            sinks[v].write(make_row(vlm, name, text, lg[i], hs), hs)
    for sk in sinks.values():
        sk.close(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--conditions", nargs="+", default=all_conditions())
    ap.add_argument("--views", nargs="+", default=["orig"])
    ap.add_argument("--vcd", action="store_true")
    ap.add_argument("--batch-views", action="store_true", help="all --views of a question in one batched pass")
    ap.add_argument("--limit-images", type=int, default=None, help="smoke test: first N images")
    ap.add_argument("--tag", default="", help="suffix for the output root (e.g. separate smoke variants)")
    args = ap.parse_args()
    out_root = ROOT if args.limit_images is None else ROOT / f"smoke{args.limit_images}"
    if args.tag:
        out_root = out_root.with_name(out_root.name + "_" + args.tag)
    keys = unique_questions(args.limit_images)
    vlm = load_vlm(args.model)
    if args.batch_views:
        for cond in args.conditions:
            run(vlm, cond, args.views, keys, args.vcd, out_root)
    else:
        for view in args.views:
            for cond in args.conditions:
                run(vlm, cond, [view], keys, args.vcd, out_root)


if __name__ == "__main__":
    main()
