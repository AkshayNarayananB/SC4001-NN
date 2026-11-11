"""Download a fixed random subset of COCO train2014 images for training the DAE (Phase 2, fix B).

train2014 is disjoint from val2014, where all 500 POPE images come from, so the DAE never sees a
test image. The split is a seeded shuffle of all eligible images: val = the first --n-val, train =
the next --n-train. Smaller runs therefore pick a prefix of larger ones.

Needs internet, so on ASPIRE run it on the login node:
    VCD_DATA_ROOT=/scratch/users/ntu/c230202/data python -m scripts.download_coco_train
"""
import argparse
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from scripts.download_pope import fetch
from src.paths import COCO_TRAIN_IMAGES, COCO_TRAIN_ROOT, coco_train_split

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
IMAGE_URL = "http://images.cocodataset.org/train2014/{name}"
MIN_SIDE = 320  # training crops are 256 px; skip the few small images


def eligible_images() -> list[str]:
    archive = COCO_TRAIN_ROOT / "annotations_trainval2014.zip"
    if not archive.exists():
        print(f"downloading {ANNOTATIONS_URL} (~240 MB)")
        fetch(ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as z, z.open("annotations/captions_train2014.json") as f:
        images = json.load(f)["images"]
    names = sorted(im["file_name"] for im in images if min(im["width"], im["height"]) >= MIN_SIDE)
    assert all("train2014" in n for n in names)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    COCO_TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    names = eligible_images()
    random.Random(args.seed).shuffle(names)
    splits = {"val": sorted(names[: args.n_val]), "train": sorted(names[args.n_val : args.n_val + args.n_train])}
    for split, split_names in splits.items():
        coco_train_split(split).write_text("\n".join(split_names) + "\n")
    print(f"{len(names)} eligible images; train {len(splits['train'])}, val {len(splits['val'])}")

    wanted = splits["train"] + splits["val"]
    todo = [n for n in wanted if not (COCO_TRAIN_IMAGES / n).exists()]
    print(f"{len(todo)} to download")
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fetch, IMAGE_URL.format(name=n), COCO_TRAIN_IMAGES / n) for n in todo]
        for job in tqdm(jobs, mininterval=10):
            job.result()

    missing = [n for n in wanted if not (COCO_TRAIN_IMAGES / n).exists()]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    print(f"done: {len(wanted)} images in {COCO_TRAIN_IMAGES}")


if __name__ == "__main__":
    main()
