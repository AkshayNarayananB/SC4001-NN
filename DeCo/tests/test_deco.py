"""CPU tests for DeCo (no weights needed).   python -m pytest DeCo/tests -q"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from DeCo.analyze import deco_margin
from DeCo.deco import EARLY_EXIT_LAYERS, candidate_tokens, deco_scores, layer_logits, select_anchor


def official_deco_step(next_token_logits, dict_outputs, early_exit_layers, alpha, top_p, top_k):
    """Verbatim logic of zjunlp/DeCo transformers/generation/utils.py::deco_greedy_search (batch 1)."""
    last_layer_tokens_logits = next_token_logits[None]
    last_layer_tokens_probs = F.softmax(last_layer_tokens_logits, dim=-1).squeeze(dim=0)
    candidate_tokens_probs, candidate_tokens_ids = torch.topk(last_layer_tokens_probs, dim=-1, k=top_k)
    candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
    candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, top_p, right=False)
    candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(top_k))
    candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]
    stacked_early_exit_layers = torch.stack([dict_outputs[i][None] for i in early_exit_layers], dim=0)
    softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
    candidate_tokens_early_exit_probs = softmax_early_exit_layers[:, :, candidate_tokens_ids].squeeze(dim=1)
    max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
    premature_max_probs = candidate_tokens_early_exit_probs.max().item()
    target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1)
    selected = dict_outputs[early_exit_layers[target_layers.item()]][None]
    indices_to_remove = torch.ones_like(selected)
    indices_to_remove[:, candidate_tokens_ids] = 0
    final = last_layer_tokens_logits + alpha * premature_max_probs * selected
    final = final.masked_fill(indices_to_remove.bool(), -float("inf"))
    return int(final.argmax()), early_exit_layers[target_layers.item()], premature_max_probs


def test_matches_official_implementation():
    g = torch.Generator().manual_seed(0)
    for trial in range(200):
        V = 50
        final = torch.randn(V, generator=g) * (1 + trial % 5)
        dict_outputs = {l: torch.randn(V, generator=g) * 3 for l in EARLY_EXIT_LAYERS}
        ref_tok, ref_layer, ref_p = official_deco_step(final, dict_outputs, list(EARLY_EXIT_LAYERS), 0.6, 0.9, 20)
        early = torch.stack([dict_outputs[l] for l in EARLY_EXIT_LAYERS])
        cand = candidate_tokens(final, 0.9, 20)
        idx, p_max = select_anchor(early, cand)
        tok = int(deco_scores(final, early[idx], p_max, cand, 0.6).argmax())
        assert (tok, EARLY_EXIT_LAYERS[idx]) == (ref_tok, ref_layer) and abs(p_max - ref_p) < 1e-6


def test_candidate_set_is_top_p_prefix():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    assert candidate_tokens(logits, top_p=0.9, top_k=4).tolist() == [0, 1, 2]
    assert candidate_tokens(logits, top_p=0.4, top_k=4).tolist() == [0]
    assert candidate_tokens(logits, top_p=0.99, top_k=2).tolist() == [0, 1]


def test_margin_handles_missing_candidates():
    df = pd.DataFrame(dict(logit_yes_final=[2.0, 0, 0, 0], logit_no_final=[1.0, 0, 0, 0], p_max=[0.5] * 4,
                           logit_yes_anchor=[0.0, 0, 0, 0], logit_no_anchor=[2.0, 0, 0, 0],
                           yes_in_cand=[True, True, False, False], no_in_cand=[True, False, True, False]))
    m = deco_margin(df, 1.0)
    assert m[0] == pytest.approx(1.0 - 1.0) and m[1] == np.inf and m[2] == -np.inf and m[3] == 0.0


def test_last_layer_logits_equal_model_output():
    torch.manual_seed(0)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                       image_size=56, patch_size=14),
        text_config=LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=32,
                                num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256),
        image_token_index=120, image_seq_length=16, vision_feature_layer=-2, vision_feature_select_strategy="default")
    model = LlavaForConditionalGeneration(cfg).eval()
    ids = torch.cat([torch.tensor([1, 5]), torch.full((16,), 120), torch.tensor([7, 3])])[None]
    out = model(input_ids=ids, pixel_values=torch.randn(1, 3, 56, 56), output_hidden_states=True)
    assert len(out.hidden_states) == 33
    lg = layer_logits(model, out.hidden_states, range(1, 33))
    torch.testing.assert_close(lg[-1], out.logits[0, -1].float(), atol=1e-5, rtol=1e-5)
    # an intermediate layer goes through the final norm before lm_head
    h = out.hidden_states[20][:, -1]
    torch.testing.assert_close(lg[19], model.language_model.lm_head(model.language_model.model.norm(h))[0])
