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

from PIL import Image
from tqdm import tqdm

from scripts.journal.run_readout import ROOT, unique_questions
from src import capture as C
from src.paths import POPE_SET, POPE_SOURCE_IMAGES
from src.vlm import load_vlm

T = C.T_REF_MS
EXPOSURES = [round(T * 2 ** k, 1) for k in range(-3, 4)]       # 4.2 ... 266.4 ms
GRIDS = {
    # quick response-surface calibration on DEV
    "calib": dict(lights=(0.03, 0.06, 0.15, 0.6), speeds=(0.0, 0.3, 1.0),
                  settings=[(round(T / 8, 1), 1, "single"), (round(T, 1), 1, "single"), (round(T * 8, 1), 1, "single"),
                            (round(T / 2, 1), 16, "aligned"), (round(T, 1), 8, "aligned")]),
}


def full_settings(max_total=T * 8 + 1, ns=(1, 2, 4, 8, 16), fusions=("naive", "aligned")):
    out = []
    for t in EXPOSURES:
        for n in ns:
            if t * n <= max_total:
                out += [(t, 1, "single")] if n == 1 else [(t, n, f) for f in fusions]
    return out


def grid_conditions(g):
    """[(Capture, group_key)] where group_key = (light, speed, t) shares one frame set."""
    caps = []
    for light, speed in product(g["lights"], g["speeds"]):
        for t, n, f in g["settings"]:
            if speed == 0 and f == "aligned":       # static camera: alignment can only add error -> skip
                f = "naive"
            caps.append(C.Capture(light, speed, t, n, f))
    return list(dict.fromkeys(caps))


def render_all(img, name, caps):
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
    return out


_CAPS = None


def _render_worker(name):
    img = Image.open(POPE_SOURCE_IMAGES / name).convert("RGB")
    return name, render_all(img, name, _CAPS)


def _init(caps):
    global _CAPS
    _CAPS = caps


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
    keys = unique_questions(a.limit_images)
    if a.nshards > 1:
        keys = [k for k in keys if zlib.crc32(k[0].encode()) % a.nshards == a.shard]
    kind = "caption" if a.caption else "capture"
    out = ROOT / a.model / f"{kind}_{a.grid}" / (f"shard{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    if a.limit_images:
        out = ROOT / a.model / f"{kind}_{a.grid}_lim{a.limit_images}" / (f"shard{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(model=a.model, pope_set=POPE_SET, grid=a.grid, conditions=[c.name for c in caps],
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
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with open(rp, "a") as fr, open(out / "images.done", "a") as fd, \
            ctx.Pool(a.workers, initializer=_init, initargs=(caps,)) as pool:
        for name, ims in tqdm(pool.imap(_render_worker, todo), total=len(todo), mininterval=60):
            names, texts = list(ims), by_img[name]
            if a.caption:
                caps_ = vlm.caption([ims[c] for c in names], max_new_tokens=a.max_new_tokens, batch=a.batch)
                for c, t in zip(names, caps_):
                    fr.write(json.dumps(dict(image=name, cond=c, caption=t)) + "\n")
                fr.flush(); fd.write(name + "\n"); fd.flush()
                continue
            S = vlm.scores_shared_prefix([ims[c] for c in names], texts, a.batch)
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
