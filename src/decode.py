"""Shared generation loop. Decoding methods differ only in how next-token logits are formed:
from the real-image logits and (optionally) the contrast-image logits, or, for the entropy-routing
methods, from an intermediate layer read out through the final norm and the LM head.

Every entropy-routing rule here is a function of the per-layer Yes/No logits and entropies, so
scripts/run_layers.py + scripts/analyze_routing.py score all of them (and any hyperparameter) from
one set of forward passes; these classes are the online versions used by scripts/run_pope.py."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.hooks import visual_contrast_hook
from src.vcd import vcd_logits

SOFT_LAYERS = (18, 20, 22, 24, 26, 28, 30)   # candidate zone of the soft / top-k routers
GRAD_LAYERS = tuple(range(16, 31))           # candidate zone of the gradient early exit


# ---------------------------------------------------------------- methods
class Baseline:
    needs_contrast = False
    is_dola = False

    def combine(self, logits, logits_cd):
        return logits


@dataclass
class VCD:
    alpha: float = 1.0
    beta: float = 0.1
    needs_contrast = True
    is_dola = False

    def combine(self, logits, logits_cd):
        return vcd_logits(logits, logits_cd, self.alpha, self.beta)


@dataclass
class VCDAblated(VCD):
    """VCD whose contrast branch is the real image with the projector output zeroed or the vision
    patches shuffled (src/hooks.py) instead of a diffusion-noised image."""
    ablation_mode: str = "zero"  # "zero" or "shuffle"


@dataclass
class _Router:
    """First-token entropy routing: replaces the final-layer logits of the first answer token
    (the Yes/No decision) by `route(...)`; later tokens come from layer 32 as usual."""
    needs_contrast = False
    is_dola = True

    def combine(self, logits, logits_cd):
        return logits


@dataclass
class DoLaTopK(_Router):
    """Blend the top_k lowest-entropy candidate layers with softmax(-entropy / temperature).
    top_k = 7 (all candidates) is the soft router; top_k = 1 is hard min-entropy routing."""
    top_k: int = 7
    temperature: float = 0.5

    def route(self, layer_logits):
        return route_top_k(layer_logits, self.top_k, self.temperature)


@dataclass
class DoLaGradient(_Router):
    """EMA-smoothed entropy early exit over layers 16..30."""
    patience: int = 2
    epsilon: float = 0.05
    ema: float = 0.7

    def route(self, layer_logits):
        return route_gradient(layer_logits, self.patience, self.epsilon, self.ema)


METHODS = {
    "baseline": Baseline,
    "vcd": VCD,
    "vcd_zero": lambda **kw: VCDAblated(ablation_mode="zero", **kw),
    "vcd_shuffle": lambda **kw: VCDAblated(ablation_mode="shuffle", **kw),
    "dola_anchor": lambda **kw: DoLaTopK(top_k=7, **kw),    # soft router (all 7 candidates)
    "dola_topk": DoLaTopK,
    "dola_gradient": DoLaGradient,
}


# ---------------------------------------------------------------- layer routing
def get_entropy(logits: torch.Tensor) -> float:
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1).mean().item()


def readout(model, hidden_states: tuple, layer: int) -> torch.Tensor:
    """Logits that decoder layer `layer` (1..32) would give at the last position (logit lens)."""
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.language_model.lm_head
    h = hidden_states[layer][:, -1, :]
    if layer < len(hidden_states) - 1:  # the last hidden state is already normed by HF Llama
        h = _final_norm(model)(h)
    return lm_head(h).float()


def _final_norm(model):
    if not hasattr(model, "_final_norm_cache"):
        found = [m for n, m in model.named_modules()
                 if n.endswith("norm") and "language_model" in n and ".layers." not in n]
        assert len(found) == 1, "final norm not unique"
        model._final_norm_cache = found[0]
    return model._final_norm_cache


def route_top_k(layer_logits: dict, top_k: int, temperature: float):
    layers = sorted(layer_logits, key=lambda l: layer_logits[l][1])[:top_k]   # lowest entropy first
    ent = torch.tensor([layer_logits[l][1] for l in layers])
    w = F.softmax(-ent / temperature, dim=0)
    blended = sum(wi.item() * layer_logits[l][0] for wi, l in zip(w, layers))
    return blended, sum(wi.item() * l for wi, l in zip(w, layers)), sum(wi.item() * e for wi, e in zip(w, ent.tolist()))


def route_gradient(layer_logits: dict, patience: int, epsilon: float, ema: float):
    smooth, best, best_l, strikes = None, math.inf, None, 0
    for l in sorted(layer_logits):
        h = layer_logits[l][1]
        smooth = h if smooth is None else ema * h + (1 - ema) * smooth
        if smooth < best:
            best, best_l = smooth, l
        strikes = strikes + 1 if smooth > best + epsilon else 0
        if strikes >= patience:
            break
    return layer_logits[best_l][0], best_l, best


def candidate_layers(method) -> tuple:
    return GRAD_LAYERS if isinstance(method, DoLaGradient) else SOFT_LAYERS


# ---------------------------------------------------------------- diagnostics
def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    logp, logq = F.log_softmax(logits_p, dim=-1), F.log_softmax(logits_q, dim=-1)
    logm = torch.logsumexp(torch.stack([logp, logq]), dim=0) - math.log(2)
    return 0.5 * ((logp.exp() * (logp - logm)).sum(-1) + (logq.exp() * (logq - logm)).sum(-1))


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


# ---------------------------------------------------------------- generation
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
    is_dola = method.is_dola

    def prefill(pv):
        out = model(input_ids=input_ids, pixel_values=pv.to(device, dtype), use_cache=True,
                    output_hidden_states=is_dola)
        return out.logits[:, -1].float(), out.past_key_values, getattr(out, "hidden_states", None)

    def step(token, past):
        # Pass embeddings, not ids: with a 1-token input_ids, transformers 4.46 Llava misdetects
        # "legacy" image processing (no image tokens present) and requires an attention_mask.
        embeds = model.get_input_embeddings()(token)
        out = model(inputs_embeds=embeds, past_key_values=past, use_cache=True)
        return out.logits[:, -1].float(), out.past_key_values

    logits, past, hidden = prefill(pixel_values)

    logits_cd = past_cd = None
    if method.needs_contrast:
        ablation = getattr(method, "ablation_mode", None)
        if ablation:  # contrast = the real image with the hook applied
            with visual_contrast_hook(model, mode=ablation):
                logits_cd, past_cd, _ = prefill(pixel_values)
        else:
            logits_cd, past_cd, _ = prefill(pixel_values_cd)

    diag_dola = {}
    if is_dola:
        layer_logits = {}
        for l in candidate_layers(method):
            lg = readout(model, hidden, l)
            layer_logits[l] = (lg, get_entropy(lg))
        logits, layer, entropy = method.route(layer_logits)
        diag_dola = {"dola_layer": layer, "dola_entropy": round(entropy, 4)}

    diag = first_step_diagnostics(logits, logits_cd, answer_ids)
    diag.update(diag_dola)

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
        # Routing applies to the first token only; layer 32 finishes the answer and emits <eos>.
        logits, past = step(next_token, past)
        if method.needs_contrast:
            logits_cd, past_cd = step(next_token, past_cd)
    return tokens, diag
