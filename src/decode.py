"""Shared generation loop. Decoding methods differ only in how next-token logits are formed
from the real-image logits and (optionally) the contrast-image logits."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.vcd import vcd_logits
from src.hooks import visual_contrast_hook

class Baseline:
    needs_contrast = False
    is_dola = False

    def combine(self, logits, logits_cd):
        return logits

@dataclass
class DoLaAnchor:
    """Dynamically routes the highest-JSD intermediate layer to the output.
    Does not apply a subtractive penalty. Proves if the visual anchor survives."""
    needs_contrast = False
    is_dola = True
    alpha: float = 1.0  # Dummy argument for run_pope.py compatibility
    beta: float = 0.1   # Dummy argument for run_pope.py compatibility

    def combine(self, logits, logits_cd):
        # We override the logits inside the generation loop dynamically, 
        # so combine just passes them through.
        return logits



@dataclass
class VCDAblated:
    alpha: float = 1.0
    beta: float = 0.1
    ablation_mode: str = "zero"  # "zero" or "shuffle"
    needs_contrast = True
    is_dola = False

    def combine(self, logits, logits_cd):
        return vcd_logits(logits, logits_cd, self.alpha, self.beta)

@dataclass
class VCD:
    alpha: float = 1.0
    beta: float = 0.1
    needs_contrast = True
    is_dola = False

    def combine(self, logits, logits_cd):
        return vcd_logits(logits, logits_cd, self.alpha, self.beta)


# Update METHODS mapping
METHODS = {
    "baseline": Baseline, 
    "vcd": VCD,
    "vcd_zero": lambda alpha, beta: VCDAblated(alpha=alpha, beta=beta, ablation_mode="zero"),
    "vcd_shuffle": lambda alpha, beta: VCDAblated(alpha=alpha, beta=beta, ablation_mode="shuffle"),
    "dola_anchor": DoLaAnchor,
}


def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    logp, logq = F.log_softmax(logits_p, dim=-1), F.log_softmax(logits_q, dim=-1)
    logm = torch.logsumexp(torch.stack([logp, logq]), dim=0) - math.log(2)
    return 0.5 * ((logp.exp() * (logp - logm)).sum(-1) + (logq.exp() * (logq - logm)).sum(-1))


def get_entropy(logits: torch.Tensor) -> float:
    """Calculates Shannon Entropy of the logit distribution."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean().item()

# def select_dola_anchor(hidden_states: tuple, model) -> tuple[torch.Tensor, int, float]:
#     """Scans candidate mid-layers and selects the one with the highest confidence (Lowest Entropy)."""
#     CANDIDATE_LAYERS = [18, 20, 22, 24, 26, 28, 30]
    
#     # 1. Safely locate the LM Head
#     if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
#         lm_head = model.language_model.lm_head
#     elif hasattr(model, 'lm_head'):
#         lm_head = model.lm_head
#     else:
#         raise ValueError("Could not locate lm_head in the model architecture.")

#     # 2. Safely locate the final RMSNorm (Handles both HF Transformers and Original LLaVA)
#     norm = None
#     paths = [
#         lambda m: m.language_model.model.norm,  # HF Transformers standard
#         lambda m: m.model.norm,                 # Original LLaVA / LLaMA
#         lambda m: m.text_model.norm,            # Alternate HF structure
#         lambda m: m.model.text_model.norm,      # Alternate HF structure
#     ]
#     for p in paths:
#         try:
#             n = p(model)
#             if n is not None:
#                 norm = n
#                 break
#         except AttributeError:
#             continue
            
#     # Ultimate fallback: scan modules for the last normalization layer
#     if norm is None:
#         norms = [m for m in model.modules() if 'norm' in m.__class__.__name__.lower()]
#         norm = norms[-1] if norms else lambda x: x

#     best_entropy = float('inf')
#     best_logits = None
#     best_layer = -1

#     for layer_idx in CANDIDATE_LAYERS:
#         if layer_idx >= len(hidden_states) - 1:
#             continue
            
#         layer_hidden = hidden_states[layer_idx][:, -1, :]
#         # Apply the dynamically found norm
#         layer_logits = lm_head(norm(layer_hidden)).float()
        
#         layer_entropy = get_entropy(layer_logits)
        
#         # We want the layer that is the MOST certain (lowest entropy)
#         if layer_entropy < best_entropy:
#             best_entropy = layer_entropy
#             best_logits = layer_logits
#             best_layer = layer_idx

#     # Fallback to final layer if something goes wrong
#     if best_logits is None:
#         final_hidden = hidden_states[-1][:, -1, :]
#         best_logits = lm_head(norm(final_hidden)).float()
#         best_layer = len(hidden_states) - 1
#         best_entropy = get_entropy(best_logits)

#     return best_logits, best_layer, best_entropy

def select_dola_anchor(hidden_states: tuple, model, temperature: float = 0.5) -> tuple[torch.Tensor, float, float]:
    """Soft-routes by blending candidate mid-layers weighted by their confidence (Lowest Entropy)."""
    CANDIDATE_LAYERS = [18, 20, 22, 24, 26, 28, 30]
    
    # 1. Safely locate the LM Head and Norm
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        lm_head = model.language_model.lm_head
    elif hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    else:
        raise ValueError("Could not locate lm_head in the model architecture.")

    norm = None
    paths = [
        lambda m: m.language_model.model.norm,
        lambda m: m.model.norm,
        lambda m: m.text_model.norm,
        lambda m: m.model.text_model.norm,
    ]
    for p in paths:
        try:
            n = p(model)
            if n is not None:
                norm = n
                break
        except AttributeError:
            continue
            
    if norm is None:
        norms = [m for m in model.modules() if 'norm' in m.__class__.__name__.lower()]
        norm = norms[-1] if norms else lambda x: x

    # 2. Gather logits and entropies for the candidate zone
    candidate_logits = []
    entropies = []
    
    for layer_idx in CANDIDATE_LAYERS:
        if layer_idx >= len(hidden_states) - 1:
            continue
            
        layer_hidden = hidden_states[layer_idx][:, -1, :]
        logits = lm_head(norm(layer_hidden)).float()
        
        entropy = get_entropy(logits)
        candidate_logits.append(logits)
        entropies.append(entropy)

    # Fallback if no valid layers (shouldn't happen, but just in case)
    if not candidate_logits:
        final_hidden = hidden_states[-1][:, -1, :]
        return lm_head(norm(final_hidden)).float(), len(hidden_states) - 1, get_entropy(lm_head(norm(final_hidden)).float())

    # 3. Softmax Weighting: Convert Entropies to Weights
    # We want LOW entropy to have HIGH weight, so we negate the entropies
    entropies_tensor = torch.tensor(entropies, device=candidate_logits[0].device)
    weights = torch.nn.functional.softmax(-entropies_tensor / temperature, dim=0)

    # 4. Blend the logits
    stacked_logits = torch.stack(candidate_logits, dim=0)  # [num_candidates, batch, vocab]
    weights = weights.view(-1, 1, 1)                       # Broadcast to match logits
    
    blended_logits = (stacked_logits * weights).sum(dim=0)
    
    # Calculate the "effective layer" (weighted average of indices) for diagnostics
    effective_layer = sum(w.item() * l for w, l in zip(weights.flatten(), CANDIDATE_LAYERS[:len(candidate_logits)]))
    mean_entropy = sum(w.item() * e for w, e in zip(weights.flatten(), entropies))

    return blended_logits, effective_layer, mean_entropy


def first_step_diagnostics(logits, logits_cd, answer_ids: dict[str, int]) -> dict:
    """Yes/No logits and log-probs at the first answer token for each branch, plus JS(real || contrast).
    The real-vs-contrast Yes/No margin is the main diagnostic for the degraded-input hypothesis."""
    diag = {"top1": logits.argmax(-1).item()}
    branches = {"": logits} if logits_cd is None else {"": logits, "_cd": logits_cd}
    for suffix, lg in branches.items():
        logprobs = F.log_softmax(lg, dim=-1)
        for word, idx in answer_ids.items():
            diag[f"logit_{word}{suffix}"] = round(lg[0, idx].item(), 4)
            diag[f"logprob_{word}{suffix}"] = round(logprobs[0, idx].item(), 4)
    if logits_cd is not None:
        diag["js"] = round(js_divergence(logits, logits_cd).item(), 6)
    return diag


@torch.inference_mode()
def generate(
    model,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_values_cd: torch.Tensor | None,
    method,
    *,
    answer_ids: dict[str, int],
    max_new_tokens: int = 16,
    sample: bool = True,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    eos_token_id: int,
) -> tuple[list[int], dict]:
    """Batch-of-1 decoding with separate KV caches for the real and contrast branches.
    Sampling draws on CPU with `generator` so results don't depend on device or job order."""
    device, dtype = model.device, model.dtype
    input_ids = input_ids.to(device)
    
    is_dola = getattr(method, "is_dola", False)
    lm_head = getattr(model, "language_model", model).lm_head if is_dola else None

    def prefill(pv):
        out = model(
            input_ids=input_ids, 
            pixel_values=pv.to(device, dtype), 
            use_cache=True, 
            output_hidden_states=is_dola
        )
        return out.logits[:, -1].float(), out.past_key_values, getattr(out, "hidden_states", None)

    def step(token, past):
        # Pass embeddings, not ids: with a 1-token input_ids, transformers 4.46 Llava misdetects
        # "legacy" image processing (no image tokens present) and requires an attention_mask.
        embeds = model.get_input_embeddings()(token)
        out = model(
            inputs_embeds=embeds, 
            past_key_values=past, 
            use_cache=True, 
            output_hidden_states=is_dola
        )
        return out.logits[:, -1].float(), out.past_key_values, getattr(out, "hidden_states", None)

    logits, past, hidden = prefill(pixel_values)
    
    if method.needs_contrast:
        ablation = getattr(method, "ablation_mode", None)
        if ablation:
            # Use the clean pixel_values, the hook handles the degradation dynamically
            with visual_contrast_hook(model, mode=ablation):
                logits_cd, past_cd, _ = prefill(pixel_values)
        else:
            # Standard VCD using explicitly degraded pixel_values_cd
            logits_cd, past_cd, _ = prefill(pixel_values_cd)
    else:
        logits_cd, past_cd = None, None

    # Handle DoLa routing for the crucial first step (Yes/No choice)
    diag_dola = {}
    if is_dola and hidden is not None:
        anchor_logits, layer_idx, entropy_val = select_dola_anchor(hidden, model)
        logits = anchor_logits
        diag_dola = {"dola_layer": layer_idx, "dola_entropy": round(entropy_val, 4)}

    diag = first_step_diagnostics(logits, logits_cd, answer_ids)
    diag.update(diag_dola)

    # tokens = []
    # for _ in range(max_new_tokens):
    #     scores = method.combine(logits, logits_cd)
    #     if sample:
    #         probs = F.softmax(scores / temperature, dim=-1).cpu()
    #         next_token = torch.multinomial(probs, 1, generator=generator)
    #     else:
    #         next_token = scores.argmax(dim=-1, keepdim=True).cpu()
            
    #     if next_token.item() == eos_token_id:
    #         break
            
    #     tokens.append(next_token.item())
    #     next_token = next_token.to(device)
        
    #     logits, past, hidden = step(next_token, past)
    #     if method.needs_contrast:
    #         logits_cd, past_cd, _ = step(next_token, past_cd)
            
    #     if is_dola and hidden is not None:
    #         anchor_logits, _, _ = select_dola_anchor(hidden, lm_head)
    #         logits = anchor_logits

    tokens = []
    for _ in range(max_new_tokens):
        scores = method.combine(logits, logits_cd)
        if sample:
            probs = F.softmax(scores / temperature, dim=-1).cpu()
            next_token = torch.multinomial(probs, 1, generator=generator)
        else:
            next_token = scores.argmax(dim=-1, keepdim=True).cpu()
            
        if next_token.item() == eos_token_id:
            break
            
        tokens.append(next_token.item())
        next_token = next_token.to(device)
        
        logits, past, hidden = step(next_token, past)
        if method.needs_contrast:
            logits_cd, past_cd, _ = step(next_token, past_cd)
            
        # DO NOT call select_dola_anchor here! 
        # We let Layer 32 (logits) take over naturally to finish the sentence and output <eos>.
            
    return tokens, diag