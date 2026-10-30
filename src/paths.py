import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("VCD_DATA_ROOT", REPO_ROOT / "data"))

POPE_ROOT = DATA_ROOT / "pope"
POPE_QUESTIONS = POPE_ROOT / "questions"
POPE_SOURCE_IMAGES = POPE_ROOT / "coco_val2014"
POPE_IMAGES = POPE_ROOT / "images"
POPE_MANIFEST = POPE_ROOT / "manifest.csv"

POPE_SPLITS = ("random", "popular", "adversarial")


def pope_question_file(split: str) -> Path:
    return POPE_QUESTIONS / f"coco_pope_{split}.json"


def degraded_image_path(image_name: str, kind: str, level: int = 0) -> Path:
    """Path of one degraded version; kind='clean' ignores level."""
    stem = Path(image_name).stem
    if kind == "clean":
        return POPE_IMAGES / "clean" / f"{stem}.png"
    return POPE_IMAGES / kind / f"s{level}" / f"{stem}.png"
