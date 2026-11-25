"""POPE-style questions on ExDark (real low-light photos, 12 annotated classes): a pure transfer test set.

    python -m scripts.journal.make_exdark_pope

ExDark (Loh & Chan, CVIU 2019; 7363 images from the HF mirror fuhao7i/ExDark, train+test merged) annotates 12
classes, so absence can only be guaranteed within those 12. Per image, k = min(#present classes, 3) present classes
are "yes" questions (shared by all splits) and k absent classes are "no" questions:
  random       uniform among the absent annotated classes
  popular      the most frequent annotated classes (over ExDark) that are absent
  adversarial  the classes that co-occur most often (over ExDark) with the image's classes and are absent
Class names follow COCO wording (People -> person, Motorbike -> motorcycle, Table -> dining table).
Also records each image's mean linear luminance so results can be stratified by how dark the photo really is.
Output: $VCD_DATA_ROOT/exdark/{questions/coco_pope_{split}.json, coco_val2014 -> images symlink, luminance.csv}
Nothing is tuned on this set.
"""
import collections
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.degradations import srgb_to_linear
from src.paths import DATA_ROOT

SRC = Path("/mnt/Data/akshay/exdark/JPEGImages")
OUT = DATA_ROOT / "exdark"
NAMES = {"Bicycle": "bicycle", "Boat": "boat", "Bottle": "bottle", "Bus": "bus", "Car": "car", "Cat": "cat",
         "Chair": "chair", "Cup": "cup", "Dog": "dog", "Motorbike": "motorcycle", "People": "person", "Table": "dining table"}
SEED = 2026


def question(name):
    return f"Is there {'an' if name[0] in 'aeiou' else 'a'} {name} in the image?"


def main():
    objs = {}
    for sp in ("train", "test"):
        d = json.load(open(SRC / f"{sp}.json"))
        cat = {c["id"]: NAMES[c["name"]] for c in d["categories"]}
        files = {im["id"]: im["file_name"] for im in d["images"]}
        per = collections.defaultdict(set)
        for a in d["annotations"]:
            per[a["image_id"]].add(cat[a["category_id"]])
        for i, f in files.items():
            if per[i]:
                assert f not in objs, f
                objs[f] = per[i]
    classes = sorted(NAMES.values())
    freq = collections.Counter(c for s in objs.values() for c in s)
    by_freq = [c for c, _ in freq.most_common()]
    cooc = collections.defaultdict(collections.Counter)
    for s in objs.values():
        for a, b in itertools.permutations(s, 2):
            cooc[a][b] += 1

    rng = random.Random(SEED)
    rows = {s: [] for s in ("random", "popular", "adversarial")}
    qid = 0
    for f in sorted(objs):
        present = objs[f]
        k = min(len(present), 3, len(classes) - len(present))
        absent = [c for c in classes if c not in present]
        pos = rng.sample(sorted(present), k)
        co = sum((cooc[p] for p in present), collections.Counter())
        adv = [c for c, _ in co.most_common() if c not in present]
        adv += [c for c in by_freq if c not in present and c not in adv]          # pad if few co-occurrences
        neg = {"random": rng.sample(absent, k),
               "popular": [c for c in by_freq if c not in present][:k],
               "adversarial": adv[:k]}
        for split in rows:
            for name, label in [(p, "yes") for p in pos] + [(n, "no") for n in neg[split]]:
                qid += 1
                rows[split].append(dict(question_id=qid, image=f, text=question(name), label=label, object=name))

    (OUT / "questions").mkdir(parents=True, exist_ok=True)
    for split, qs in rows.items():
        with open(OUT / "questions" / f"coco_pope_{split}.json", "w") as fh:
            fh.writelines(json.dumps(q) + "\n" for q in qs)
    link = OUT / "coco_val2014"          # the name run_readout expects for source images
    if not link.exists():
        link.symlink_to(SRC / "IMGS")
    with open(OUT / "luminance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "mean_linear_luminance", "mean_srgb"])
        for f in sorted(objs):
            a = np.asarray(Image.open(SRC / "IMGS" / f).convert("RGB"), dtype=np.float64) / 255.0
            lin = srgb_to_linear(a) @ np.array([0.2126, 0.7152, 0.0722])
            w.writerow([f, round(float(lin.mean()), 5), round(float(a.mean()), 4)])
    n_unique = len({(q["image"], q["text"]) for qs in rows.values() for q in qs})
    print({s: len(q) for s, q in rows.items()}, "images", len(objs), "unique questions", n_unique)
    print("class frequency:", freq.most_common())


if __name__ == "__main__":
    main()
