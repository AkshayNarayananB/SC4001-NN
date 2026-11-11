import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.model import load_model, encode, answer_token_ids
from src.hooks import visual_contrast_hook
from src.decode import first_step_diagnostics
from PIL import Image

from src.paths import REPO_ROOT

def test_ablations(image_path, question, label="yes"):
    model, processor = load_model()
    ans_ids = answer_token_ids(processor)
    
    # Load and encode clean image
    img = Image.open(image_path).convert("RGB")
    input_ids, pixel_values = encode(processor, img, question)
    
    with torch.inference_mode():
        # 1. Real branch (Baseline)
        out = model(input_ids=input_ids.to(model.device), 
                    pixel_values=pixel_values.to(model.device, model.dtype))
        logits = out.logits[:, -1].float()
        
        # 2. KV-Cache Zeroing
        with visual_contrast_hook(model, mode="zero"):
            out_zero = model(input_ids=input_ids.to(model.device), 
                             pixel_values=pixel_values.to(model.device, model.dtype))
            logits_zero = out_zero.logits[:, -1].float()
            
        # 3. Positional Shuffling
        with visual_contrast_hook(model, mode="shuffle"):
            out_shuffle = model(input_ids=input_ids.to(model.device), 
                                pixel_values=pixel_values.to(model.device, model.dtype))
            logits_shuffle = out_shuffle.logits[:, -1].float()

    # Get Yes/No Margins (logit_Yes - logit_No)
    diag_base = first_step_diagnostics(logits, None, ans_ids)
    diag_zero = first_step_diagnostics(logits, logits_zero, ans_ids)
    diag_shuf = first_step_diagnostics(logits, logits_shuffle, ans_ids)
    
    margin_base = diag_base["logit_yes"] - diag_base["logit_no"]
    margin_zero = diag_zero["logit_yes_cd"] - diag_zero["logit_no_cd"]
    margin_shuf = diag_shuf["logit_yes_cd"] - diag_shuf["logit_no_cd"]
    
    print(f"Label: {label} | Baseline Margin: {margin_base:.3f} | Zero Margin: {margin_zero:.3f} | Shuffle Margin: {margin_shuf:.3f}")
    print(f"JS Divergence (Base || Zero): {diag_zero['js']}")
    print(f"JS Divergence (Base || Shuffle): {diag_shuf['js']}\n")

if __name__ == "__main__":
    # Point this to a few sample images from POPE_IMAGES to immediately verify the JS divergence 
    # and contrast margin behavior before queueing the 9000-question SLURM jobs.
    print("Testing Ablations...")
    test_ablations(f"{REPO_ROOT}/pope/images/clean/COCO_val2014_000000156704.png", "Is there a surfboard in the image?", "yes")
    test_ablations(f"{REPO_ROOT}/pope/images/clean/COCO_val2014_000000156704.png", "Is there a dog in the image?", "yes")