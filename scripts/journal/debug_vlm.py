"""Debug one model end to end: shapes, generated text, top first tokens, margin variation.
    python -m scripts.journal.debug_vlm --model instructblip-7b"""
import argparse
import json

import torch
from PIL import Image

from src.paths import POPE_SOURCE_IMAGES, pope_question_file
from src.vlm import load_vlm

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
vlm = load_vlm(a.model)
qs = [json.loads(l) for l in open(pope_question_file("random"))][:6]
tok = vlm.processor.tokenizer
print("yes ids", vlm.yes_ids, tok.convert_ids_to_tokens(vlm.yes_ids), "no ids", vlm.no_ids, tok.convert_ids_to_tokens(vlm.no_ids))
for q in qs:
    img = Image.open(POPE_SOURCE_IMAGES / q["image"]).convert("RGB")
    x = vlm.encode(img, q["text"])
    print({k: tuple(v.shape) for k, v in x.items() if torch.is_tensor(v)})
    logits, hs = vlm.forward(x)
    top = logits.topk(5)
    print(q["label"], q["text"], "| m =", round(vlm.margin(logits), 3), "| top:", tok.convert_ids_to_tokens(top.indices.tolist()),
          [round(v, 2) for v in top.values.tolist()], "| nan:", bool(torch.isnan(logits).any()))
    with torch.inference_mode():
        out = vlm.model.generate(**vlm._to_device(x), max_new_tokens=8, do_sample=False)
    print("   generate:", repr(tok.decode(out[0][-8:], skip_special_tokens=True)), "| full len", out.shape)
print("prompt sample:", repr(tok.decode(x["input_ids"][0][-30:])))
