"""Build POPE-style questions on fresh COCO val2014 images, disjoint from POPE's 500.

    python -m scripts.journal.make_fresh_pope --n-images 2000          # questions + image download

This follows POPE's recipe (Li et al., EMNLP 2023): each image has >= 3 annotated object
categories; 3 of them, sampled at random, become "yes" questions shared by all splits. Each split
then adds 3 "no" questions about absent categories:
  random       uniformly sampled absent categories
  popular      the most frequent categories in the dataset that are absent, in frequency order
  adversarial  the categories that co-occur most often with the image's objects and are absent
Category frequency and co-occurrence are counted over all val2014 instance annotations.

These images form the held-out TEST set for the journal extension. The 500 original POPE images
(inspected during the course project) are the DEV set. Nothing is tuned on this set.
Output: $VCD_DATA_ROOT/pope_fresh/{questions/coco_pope_{split}.json, coco_val2014/, images.txt}
"""
import argparse
import collections
import concurrent.futures as cf
import itertools
import json
import random
import urllib.request
from pathlib import Path

from src.paths import DATA_ROOT, POPE_SPLITS, pope_question_file

ANN = Path("/mnt/Data/akshay/coco/annotations/instances_val2014.json")
OUT = DATA_ROOT / "pope_fresh"
SEED = 2026
N_POS = N_NEG = 3


def article(name: str) -> str:
    return "an" if name[0] in "aeiou" else "a"


def question(name: str) -> str:
    return f"Is there {article(name)} {name} in the image?"


def build(n_images: int):
    ann = json.loads(ANN.read_text())
    cat_name = {c["id"]: c["name"] for c in ann["categories"]}
    file_of = {im["id"]: im["file_name"] for im in ann["images"]}
    objs = collections.defaultdict(set)
    for a in ann["annotations"]:
        objs[a["image_id"]].add(cat_name[a["category_id"]])

    freq = collections.Counter(c for s in objs.values() for c in s)          # images containing c
    cooc = collections.defaultdict(collections.Counter)
    for s in objs.values():
        for a, b in itertools.permutations(s, 2):
            cooc[a][b] += 1
    by_freq = [c for c, _ in freq.most_common()]

    pope_images = set()
    for split in POPE_SPLITS:
        pope_images |= {json.loads(l)["image"] for l in open(pope_question_file(split)) if l.strip()}
    pool = sorted(i for i, s in objs.items() if len(s) >= N_POS and file_of[i] not in pope_images)
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(pool, n_images))
    all_cats = sorted(cat_name.values())

    rows = {s: [] for s in POPE_SPLITS}
    qid = 0
    for img_id in chosen:
        present = objs[img_id]
        absent = [c for c in all_cats if c not in present]
        pos = rng.sample(sorted(present), N_POS)
        neg = {
            "random": rng.sample(absent, N_NEG),
            "popular": [c for c in by_freq if c not in present][:N_NEG],
            "adversarial": [c for c, _ in sum((cooc[p] for p in present), collections.Counter()).most_common()
                            if c not in present][:N_NEG],
        }
        for split in POPE_SPLITS:
            assert len(neg[split]) == N_NEG, (img_id, split)
            for name, label in [(p, "yes") for p in pos] + [(n, "no") for n in neg[split]]:
                qid += 1
                rows[split].append({"question_id": qid, "image": file_of[img_id], "text": question(name),
                                    "label": label, "object": name})
    return rows, [file_of[i] for i in chosen]


def download(names, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)

    def get(name):
        p = dest / name
        if not p.exists():
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2014/{name}", tmp)
            tmp.rename(p)
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(get, names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    rows, names = build(args.n_images)
    (OUT / "questions").mkdir(parents=True, exist_ok=True)
    for split, qs in rows.items():
        with open(OUT / "questions" / f"coco_pope_{split}.json", "w") as f:
            f.writelines(json.dumps(q) + "\n" for q in qs)
    (OUT / "images.txt").write_text("\n".join(names) + "\n")
    print({s: len(q) for s, q in rows.items()}, "images:", len(names))
    if not args.no_download:
        download(names, OUT / "coco_val2014")
        print("downloaded", len(list((OUT / "coco_val2014").glob("*.jpg"))))


if __name__ == "__main__":
    main()
