"""Download POPE COCO question files and the 500 COCO val2014 images they reference.

Needs internet, so on ASPIRE run it on the login node:
    VCD_DATA_ROOT=/scratch/users/ntu/c230202/data python -m scripts.download_pope
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from src.paths import POPE_QUESTIONS, POPE_SOURCE_IMAGES, POPE_SPLITS, pope_question_file

QUESTIONS_URL = "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_{split}.json"
COCO_URL = "http://images.cocodataset.org/val2014/{name}"


def fetch(url: str, dest, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                f.write(resp.read())
            tmp.rename(dest)
            return
        except OSError:
            if attempt == retries - 1:
                raise


def load_image_names() -> list[str]:
    names = set()
    for split in POPE_SPLITS:
        with open(pope_question_file(split)) as f:
            names.update(json.loads(line)["image"] for line in f if line.strip())
    return sorted(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    POPE_QUESTIONS.mkdir(parents=True, exist_ok=True)
    POPE_SOURCE_IMAGES.mkdir(parents=True, exist_ok=True)

    for split in POPE_SPLITS:
        dest = pope_question_file(split)
        if not dest.exists():
            fetch(QUESTIONS_URL.format(split=split), dest)

    names = load_image_names()
    todo = [n for n in names if not (POPE_SOURCE_IMAGES / n).exists()]
    print(f"{len(names)} images referenced, {len(todo)} to download")

    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fetch, COCO_URL.format(name=n), POPE_SOURCE_IMAGES / n) for n in todo]
        for job in tqdm(jobs):
            job.result()

    missing = [n for n in names if not (POPE_SOURCE_IMAGES / n).exists()]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    print(f"done: {len(names)} images in {POPE_SOURCE_IMAGES}")


if __name__ == "__main__":
    main()
