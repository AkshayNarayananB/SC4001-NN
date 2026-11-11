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

# DAE training data (Fix B): a subset of COCO train2014, disjoint from POPE's val2014 images
COCO_TRAIN_ROOT = DATA_ROOT / "coco_train2014"
COCO_TRAIN_IMAGES = COCO_TRAIN_ROOT / "images"

CHECKPOINT_ROOT = Path(os.environ.get("VCD_CHECKPOINT_ROOT", REPO_ROOT / "checkpoints"))


def pope_question_file(split: str) -> Path:
    return POPE_QUESTIONS / f"coco_pope_{split}.json"


def degraded_image_path(image_name: str, kind: str, level: int = 0) -> Path:
    """Path of one degraded version; kind='clean' ignores level."""
    stem = Path(image_name).stem
    if kind == "clean":
        return POPE_IMAGES / "clean" / f"{stem}.png"
    return POPE_IMAGES / kind / f"s{level}" / f"{stem}.png"


def coco_train_split(split: str) -> Path:
    """List of image file names for the DAE's 'train' or 'val' split."""
    return COCO_TRAIN_ROOT / f"{split}.txt"


def restored_image_path(image_name: str, restorer: str, kind: str, level: int) -> Path:
    """Path of a degraded image after restoration by `restorer` (e.g. a DAE run name)."""
    return POPE_IMAGES / "restored" / restorer / kind / f"s{level}" / f"{Path(image_name).stem}.png"
