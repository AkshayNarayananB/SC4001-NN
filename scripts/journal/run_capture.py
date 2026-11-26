"""Sensing sweep: POPE first-token margins under simulated captures (src/capture.py).

    POPE_SET=pope python -m scripts.journal.run_capture --model llava-1.5-7b --grid calib --limit-images 100

For every image, all capture settings of the grid are rendered from ONE set of frames per (light, speed, t)
(burst prefixes share frames), then every unique POPE question is asked on every setting; settings of one question
share a batched forward pass (same image size -> same token length). Logged per (image, question, setting):
m, yes, no, top (as run_readout). No hidden states (the sweep is large).
Output: $VCD_OUTPUT_ROOT/journal/<POPE_SET>/<model>/capture_<grid>[/shardIofN]/{rows.jsonl,images.done,meta.json,DONE}
Resumable at image granularity (an image is listed in images.done only after all its rows are written).
"""
import argparse
import json
import zlib
from itertools import product

import torch
from PIL import Image
from tqdm import tqdm

from scripts.journal.run_readout import ROOT, unique_questions
from src import capture as C
from src.paths import POPE_SET, POPE_SOURCE_IMAGES
from src.vlm import load_vlm

T = C.T_REF_MS
EXPOSURES = [round(T * 2 ** k, 1) for k in range(-3, 4)]       # 4.2 ... 266.4 ms
VIEW_TS = (8.3, 33.3, 133.2)      # single-shot exposures that also get the 6-pass multi-view compute arm
MV_VIEWS = ("blur1", "blur2", "blur3", "hflip")   # + orig + dae = ViewLayer's six views
GRIDS = {
    # quick response-surface calibration on DEV
    "calib": dict(lights=(0.03, 0.06, 0.15, 0.6), speeds=(0.0, 0.3, 1.0),
                  settings=[(round(T / 8, 1), 1, "single"), (round(T, 1), 1, "single"), (round(T * 8, 1), 1, "single"),
                            (round(T / 2, 1), 16, "aligned"), (round(T, 1), 8, "aligned")]),
    "smoke": dict(lights=(0.15,), speeds=(0.0, 0.3), settings=None, compute=True),
}
# CHAIR captions: a representative subset (short / video-rate / long single shots, and the two burst shapes)
GRIDS["capsel"] = dict(lights=(0.06, 0.15, 0.3, 0.6), speeds=(0.0, 0.1, 0.3, 0.6),
                       settings=[(8.3, 1, "single"), (33.3, 1, "single"), (266.4, 1, "single"),
                                 (16.6, 16, "aligned"), (33.3, 8, "aligned")])
GRIDS["full"] = dict(lights=(0.06, 0.15, 0.3, 0.6), speeds=(0.0, 0.1, 0.3, 0.6), settings=None, compute=True)


def full_settings(max_total=T * 8 + 1, ns=(1, 2, 4, 8, 16), fusions=("naive", "aligned")):
    out = []
    for t in EXPOSURES:
        for n in ns:
            if t * n <= max_total:
                out += [(t, 1, "single")] if n == 1 else [(t, n, f) for f in fusions]
    return out


def grid_conditions(g):
    """Capture settings of a grid (settings=None -> full_settings())."""
    caps = []
    for light, speed in product(g["lights"], g["speeds"]):
        for t, n, f in (g["settings"] or full_settings()):
            if speed == 0 and f == "aligned":       # static camera: alignment can only add error -> skip
                f = "naive"
            caps.append(C.Capture(light, speed, t, n, f))
    return list(dict.fromkeys(caps))


def compute_extras(caps):
    """Compute-arm variants scored alongside the captures: for every single-frame capture, VCD's contrast image
    (__vcdneg) and the DAE-restored image (__dae); for single-frame captures at VIEW_TS, the multi-view views."""
    ex = []
    for c in caps:
        if c.n == 1:
            ex += [(c.name, "vcdneg"), (c.name, "dae")]
            if c.t_ms in VIEW_TS:
                ex += [(c.name, v) for v in MV_VIEWS]
    return ex


def render_all(img, name, caps, extras=()):
    out = {}
    groups = {}
    for c in caps:
        groups.setdefault((c.light, c.speed, c.t_ms), []).append(c)
    for (light, speed, t), cs in groups.items():
        ns = tuple(sorted({c.n for c in cs}))
        fus = tuple(sorted({c.fusion for c in cs if c.n > 1})) or ("naive",)
        ims = C.capture_prefixes(img, name, light, speed, t, ns, fus)
        for c in cs:
            out[c.name] = ims[(c.n, c.fusion)]
    if extras:
        from scripts.run_layers import make_view
        for base, v in extras:
            if v in MV_VIEWS:
                out[f"{base}__{v}"] = make_view(out[base], v)
    return out


_CAPS, _EXTRAS = None, ()


def _render_worker(name):
    img = Image.open(POPE_SOURCE_IMAGES / name).convert("RGB")
    return name, render_all(img, name, _CAPS, _EXTRAS)


def _init(caps, extras):
    global _CAPS, _EXTRAS
    _CAPS, _EXTRAS = caps, extras


VCD_STEP = 999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--grid", default="calib")
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--caption", action="store_true", help="CHAIR mode: greedy captions instead of POPE margins")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6, help="CPU processes rendering captures ahead of the GPU")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    caps = grid_conditions(GRIDS[a.grid])
    extras = compute_extras(caps) if GRIDS[a.grid].get("compute") and not a.caption else []
    keys = unique_questions(a.limit_images)
    if a.nshards > 1:
        keys = [k for k in keys if zlib.crc32(k[0].encode()) % a.nshards == a.shard]
    kind = "caption" if a.caption else "capture"
    out = ROOT / a.model / f"{kind}_{a.grid}" / (f"shard{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    if a.limit_images:
        out = ROOT / a.model / f"{kind}_{a.grid}_lim{a.limit_images}" / (f"shard{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(model=a.model, pope_set=POPE_SET, grid=a.grid, conditions=[c.name for c in caps],
                extras=[f"{b}__{v}" for b, v in extras], vcd_step=VCD_STEP,
                caption=a.caption, max_new_tokens=a.max_new_tokens if a.caption else None, read_noise=C.READ_NOISE_E, align_down=C.ALIGN_DOWN, align_max_px=C.ALIGN_MAX_PX)
    mp = out / "meta.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == meta, f"meta mismatch in {out}"
    else:
        mp.write_text(json.dumps(meta, indent=1))
    if (out / "DONE").exists():
        print("done"); return
    done = set((out / "images.done").read_text().split()) if (out / "images.done").exists() else set()
    by_img = {}
    for name, text in keys:
        by_img.setdefault(name, []).append(text)
    todo = [n for n in sorted(by_img) if n not in done]
    # drop rows of a half-finished image
    rp = out / "rows.jsonl"
    if rp.exists():
        good = [l for l in rp.read_text().splitlines() if l and json.loads(l)["image"] in done]
        rp.write_text("".join(x + "\n" for x in good))
    print(f"{a.model} {POPE_SET} {a.grid}: {len(caps)} settings, {len(todo)}/{len(by_img)} images to run", flush=True)
    vlm = load_vlm(a.model)
    from scripts.run_layers import make_view

    def vcd_tf(name, cond):     # VCD contrast branch: forward-diffusion noise on the normalised pixels (T=999)
        g = torch.Generator().manual_seed(zlib.crc32(f"{name}|{cond}".encode()))
        return lambda enc: vlm.add_vcd_noise(enc, VCD_STEP, g)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with open(rp, "a") as fr, open(out / "images.done", "a") as fd, \
            ctx.Pool(a.workers, initializer=_init, initargs=(caps, extras)) as pool:
        for name, ims in tqdm(pool.imap(_render_worker, todo), total=len(todo), mininterval=60):
            names, texts = list(ims), by_img[name]
            if a.caption:
                caps_ = vlm.caption([ims[c] for c in names], max_new_tokens=a.max_new_tokens, batch=a.batch)
                for c, t in zip(names, caps_):
                    fr.write(json.dumps(dict(image=name, cond=c, caption=t)) + "\n")
                fr.flush(); fd.write(name + "\n"); fd.flush()
                continue
            if extras:     # DAE runs on the GPU in this process; VCD noise is applied to the encoding
                for b, v in extras:
                    if v == "dae":
                        ims[f"{b}__dae"] = make_view(ims[b], "dae")
                    elif v == "vcdneg":
                        ims[f"{b}__vcdneg"] = ims[b]
                names = list(ims)
            tfs = [vcd_tf(name, c) if c.endswith("__vcdneg") else None for c in names]
            S = vlm.scores_shared_prefix([ims[c] for c in names], texts, a.batch, tfs)
            for i, c in enumerate(names):
                for j, text in enumerate(texts):
                    m, y, n, tp = S[i, j].tolist()
                    fr.write(json.dumps(dict(image=name, text=text, cond=c, m=round(m, 4), yes=round(y, 4),
                                             no=round(n, 4), top=round(tp, 4))) + "\n")
            fr.flush()
            fd.write(name + "\n"); fd.flush()
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
