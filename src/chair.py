"""CHAIR (Rohrbach et al., EMNLP 2018) for free-form captions on COCO val2014: a port of the reference
implementation (github.com/LisaAnne/Hallucination, utils/chair.py) with the same synonym list
(configs/journal/chair_synonyms.txt), double-word rules and ground truth (instance annotations UNION objects
mentioned in the 5 reference captions). Differences: Python 3, `inflect` instead of `pattern.en` for singularising.

    ev = Chair(coco_annotations_dir)      # dir with instances_val2014.json + captions_val2014.json
    ev.score({image_id: caption, ...}) -> dict(chair_s, chair_i, recall, n_words, per_image)
"""
import json
from functools import lru_cache
from pathlib import Path

import inflect
import nltk

from src.paths import REPO_ROOT

SYN = REPO_ROOT / "configs" / "journal" / "chair_synonyms.txt"
DOUBLE = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light',
          'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove',
          'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
          'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'home plate', 'train track']
ANIMALS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
_inf = inflect.engine()


@lru_cache(maxsize=None)
def singularize(w: str) -> str:
    s = _inf.singular_noun(w)
    return s if s else w


class Chair:
    def __init__(self, ann_dir, image_ids=None):
        syn = [[s.strip() for s in line.strip().split(",") if s.strip()] for line in open(SYN) if line.strip()]
        self.objects, self.inv = set(), {}
        for group in syn:
            self.objects.update(group)
            for s in group:
                self.inv[s] = group[0]
        self.double = {d: d for d in DOUBLE}
        for a in ANIMALS:
            self.double[f"baby {a}"] = a
            self.double[f"adult {a}"] = a
        for v in ("jet", "train"):
            self.double[f"passenger {v}"] = v
        self.double.update({"bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass"})
        ann_dir = Path(ann_dir)
        inst = json.load(open(ann_dir / "instances_val2014.json"))
        caps = json.load(open(ann_dir / "captions_val2014.json"))
        keep = set(image_ids) if image_ids is not None else None
        names = {c["id"]: c["name"] for c in inst["categories"]}
        self.gt = {}
        for a in inst["annotations"]:
            if keep is None or a["image_id"] in keep:
                self.gt.setdefault(a["image_id"], set()).add(self.inv[names[a["category_id"]]])
        for a in caps["annotations"]:
            if keep is None or a["image_id"] in keep:
                self.gt.setdefault(a["image_id"], set()).update(self.words(a["caption"])[1])

    def words(self, caption: str):
        ws = [singularize(w) for w in nltk.word_tokenize(caption.lower())]
        out, i = [], 0
        while i < len(ws):
            dw = " ".join(ws[i:i + 2])
            if dw in self.double:
                out.append(self.double[dw]); i += 2
            else:
                out.append(ws[i]); i += 1
        if "toilet" in out and "seat" in out:
            out = [w for w in out if w != "seat"]
        found = [w for w in out if w in self.objects]
        return found, [self.inv[w] for w in found]

    def score(self, caps: dict) -> dict:
        n_cap_h = n_obj = n_obj_h = 0
        rec_hit = rec_tot = 0
        per = {}
        for imid, c in caps.items():
            gt = self.gt.get(imid, set())
            _, nodes = self.words(c)
            hall = [w for w in nodes if w not in gt]
            n_obj += len(nodes); n_obj_h += len(hall); n_cap_h += bool(hall)
            rec_hit += len(gt & set(nodes)); rec_tot += len(gt)
            per[imid] = dict(mentioned=sorted(set(nodes)), hallucinated=sorted(set(hall)))
        n = max(len(caps), 1)
        return dict(chair_s=n_cap_h / n, chair_i=n_obj_h / max(n_obj, 1), recall=rec_hit / max(rec_tot, 1),
                    n_caps=len(caps), per_image=per)
