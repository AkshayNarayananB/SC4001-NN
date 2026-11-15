"""DeCo: Dynamic Correction Decoding (Wang et al., "MLLM Can See?", ICLR 2025; github.com/zjunlp/DeCo).

Finding behind it: the MLLM often recognises the object in its *preceding* (middle) layers, but the
language prior in the final layers overrides that. DeCo mixes a middle "anchor" layer back into the
final logits at every decoding step. Ported from the official deco_greedy_search:

    1. candidates C: smallest top-k prefix of the final-layer softmax whose cumulative probability
       reaches top_p (capped at top_k)
    2. early logits for each layer l in early_exit_layers: lm_head(norm(hidden_states[l]))
    3. anchor layer = the layer holding the largest softmax probability on any candidate token;
       p_max = that probability
    4. score = final_logits + alpha * p_max * anchor_logits, tokens outside C set to -inf
Defaults (official pope_llava.py / README): alpha 0.6, top_p 0.9, top_k 20, layers range(20, 29).
hidden_states indexing follows HF Llama: [l] = output of decoder layer l (pre-norm) for l < 32.
"""
import torch

ALPHA = 0.6
TOP_P = 0.9
TOP_K = 20
EARLY_EXIT_LAYERS = tuple(range(20, 29))


def candidate_tokens(final_logits: torch.Tensor, top_p: float = TOP_P, top_k: int = TOP_K) -> torch.Tensor:
    """final_logits: (V,). Returns ids of the candidate set C."""
    probs = torch.softmax(final_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k)
    cut = torch.searchsorted(top_probs.cumsum(-1), torch.tensor(top_p, dtype=top_probs.dtype, device=top_probs.device),
                             right=False)
    return top_ids[: min(int(cut) + 1, top_k)]


def select_anchor(early_logits: torch.Tensor, cand: torch.Tensor) -> tuple[int, float]:
    """early_logits: (L, V) for the early-exit layers. Returns (index into the layer list, p_max)."""
    cand_probs = torch.softmax(early_logits, dim=-1)[:, cand]          # (L, |C|)
    flat = int(cand_probs.argmax())
    return flat // cand_probs.shape[1], cand_probs.max().item()


def deco_scores(final_logits: torch.Tensor, anchor_logits: torch.Tensor, p_max: float, cand: torch.Tensor,
                alpha: float = ALPHA) -> torch.Tensor:
    scores = final_logits + alpha * p_max * anchor_logits
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[cand] = True
    return scores.masked_fill(~keep, float("-inf"))


def layer_logits(model, hidden_states, layers) -> torch.Tensor:
    """lm_head(norm(h_l)) at the last position for each l in `layers` -> (len(layers), V) float32.
    Layer 32 (the final entry) is already normed by HF Llama, so it only gets lm_head."""
    lm = model.language_model
    n_layers = len(hidden_states) - 1
    out = []
    for l in layers:
        h = hidden_states[l][:, -1]
        h = h if l == n_layers else lm.model.norm(h)
        out.append(lm.lm_head(h)[0].float())
    return torch.stack(out)
