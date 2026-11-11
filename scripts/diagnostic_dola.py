import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import argparse
import os

from src.model import answer_token_ids, encode, load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, required=True, help="Path to a clean POPE image to blur in-memory")
    parser.add_argument("--question", type=str, default="Is there a bull in the image?")
    parser.add_argument("--out", type=str, default="results/dola_layers_blur_forced.png")
    args = parser.parse_args()

    print("Loading local LLaVA model via src.model...")
    model, processor = load_model()
    model.config.output_hidden_states = True
    model.eval()

    answer_ids = answer_token_ids(processor)
    YES_ID = answer_ids['yes']
    NO_ID = answer_ids['no']

    print(f"Processing image: {args.image_file}")
    
    # Force the severe blur in memory to bypass disk cache silently failing
    image = Image.open(args.image_file).convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=8))
    
    # Pass the raw question string so processor handles formatting correctly
    input_ids, pixel_values = encode(processor, image, args.question)
    
    input_ids = input_ids.to(model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device)

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=True
        )

    hidden_states = outputs.hidden_states
    
    yes_probs, no_probs = [], []
    yes_logits, no_logits = [], []
    layers = list(range(len(hidden_states)))

    print("Projecting hidden states through LM Head...")
    for i, h in enumerate(hidden_states):
        last_token_hidden = h[0, -1, :]
        
        if hasattr(model, 'language_model'):
            layer_logits = model.language_model.lm_head(last_token_hidden)
        else:
            layer_logits = model.lm_head(last_token_hidden)
        
        layer_probs = torch.softmax(layer_logits.float(), dim=-1)
        
        yes_probs.append(layer_probs[YES_ID].item())
        no_probs.append(layer_probs[NO_ID].item())
        yes_logits.append(layer_logits[YES_ID].item())
        no_logits.append(layer_logits[NO_ID].item())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(layers, yes_probs, label="Yes Probability", color="green", marker="o")
    ax1.plot(layers, no_probs, label="No Probability", color="red", marker="x")
    ax1.set_title(f"DoLa Latent Evolution (Forced Blur 8): '{args.question}'")
    ax1.set_ylabel("Probability")
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(layers, yes_logits, label="Yes Logit", color="green", linestyle="--")
    ax2.plot(layers, no_logits, label="No Logit", color="red", linestyle="--")
    ax2.set_title("Raw Logits across Transformer Depth")
    ax2.set_xlabel("Transformer Layer (0 = Embedding, 32 = Final)")
    ax2.set_ylabel("Logit Value")
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"\nPlot saved to {args.out}")
    
    print("\n--- Layer Divergence Summary ---")
    print(f"Layer 16 (Mid): Yes Probs = {yes_probs[16]:.4f} | No Probs = {no_probs[16]:.4f}")
    print(f"Layer 24 (Late): Yes Probs = {yes_probs[24]:.4f} | No Probs = {no_probs[24]:.4f}")
    print(f"Layer 32 (Final): Yes Probs = {yes_probs[-1]:.4f} | No Probs = {no_probs[-1]:.4f}")

if __name__ == "__main__":
    main()