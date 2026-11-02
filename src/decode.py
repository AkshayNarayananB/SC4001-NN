"""Shared generation loop. Decoding methods differ only in how next-token logits are formed
from the real-image logits and (optionally) the contrast-image logits."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.vcd import vcd_logits


class Baseline:
    needs_contrast = False

    def combine(self, logits, logits_cd):
        return logits


@dataclass
class VCD:
    alpha: float = 1.0
    beta: float = 0.1
    needs_contrast = True

    def combine(self, logits, logits_cd):
        return vcd_logits(logits, logits_cd, self.alpha, self.beta)


METHODS = {"baseline": Baseline, "vcd": VCD}


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

    def prefill(pv):
        out = model(input_ids=input_ids, pixel_values=pv.to(device, dtype), use_cache=True)
        return out.logits[:, -1].float(), out.past_key_values

    def step(token, past):
        # Pass embeddings, not ids: with a 1-token input_ids, transformers 4.46 Llava misdetects
        # "legacy" image processing (no image tokens present) and requires an attention_mask.
        embeds = model.get_input_embeddings()(token)
        out = model(inputs_embeds=embeds, past_key_values=past, use_cache=True)
        return out.logits[:, -1].float(), out.past_key_values

    logits, past = prefill(pixel_values)
    logits_cd, past_cd = prefill(pixel_values_cd) if method.needs_contrast else (None, None)
    diag = first_step_diagnostics(logits, logits_cd, answer_ids)

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
        logits, past = step(next_token, past)
        if method.needs_contrast:
            logits_cd, past_cd = step(next_token, past_cd)
    return tokens, diag
