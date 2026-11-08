import contextlib
import torch

def _get_module(model, target_type):
    """Dynamically finds modules to bypass Hugging Face version path changes."""
    for name, module in model.named_modules():
        if target_type == "projector" and "projector" in name:
            return module
        if target_type == "vision_embeddings" and "vision" in name and name.endswith("embeddings"):
            return module
            
    # If it still fails, this will print the exact tree so we can see what HF named it
    all_names = [n for n, _ in model.named_modules()]
    raise AttributeError(f"Could not find '{target_type}'. Available modules:\n" + "\n".join(all_names))

@contextlib.contextmanager
def visual_contrast_hook(model, mode="zero"):
    """
    Temporarily patches the LLaVA-1.5 model for the contrast forward pass.
    Zeroes out projector outputs or shuffles vision patches entirely on-GPU.
    """
    handles = []
    
    if mode == "zero":
        projector = _get_module(model, "projector")
        
        def zero_hook(module, args, output):
            return torch.zeros_like(output)
            
        handles.append(projector.register_forward_hook(zero_hook))
        
    elif mode == "shuffle":
        embeddings = _get_module(model, "vision_embeddings")
        
        def shuffle_hook(module, args, output):
            # output shape: [batch, seq_len, hidden_size]
            cls_token = output[:, :1, :]
            patches = output[:, 1:, :]
            
            # GPU-optimized shuffle using argsort on random noise to avoid CPU syncs
            noise = torch.rand(patches.shape[:2], device=output.device, dtype=output.dtype)
            shuffle_idx = noise.argsort(dim=1).unsqueeze(-1).expand_as(patches)
            
            shuffled_patches = torch.gather(patches, dim=1, index=shuffle_idx)
            return torch.cat([cls_token, shuffled_patches], dim=1)
            
        handles.append(embeddings.register_forward_hook(shuffle_hook))
        
    try:
        yield
    finally:
        for h in handles:
            h.remove()