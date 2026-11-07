"""CPU tests for ViewLayer (no weights needed).   python -m pytest ViewLayer/tests -q"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from DeCo.deco import layer_logits
from ViewLayer.analyze import best_threshold, score
from ViewLayer.run_viewlayer import VIEWS, layer_margins, make_view


def tiny_llava():
    torch.manual_seed(0)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                       image_size=56, patch_size=14),
        text_config=LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=32,
                                num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256),
        image_token_index=120, image_seq_length=16, vision_feature_layer=-2, vision_feature_select_strategy="default")
    return LlavaForConditionalGeneration(cfg).eval()


def test_layer_margins_match_full_readout_and_final_logits():
    model = tiny_llava()
    ids = torch.cat([torch.tensor([1, 5]), torch.full((16,), 120), torch.tensor([7, 3])])[None].expand(2, -1)
    out = model(input_ids=ids, pixel_values=torch.randn(2, 3, 56, 56), output_hidden_states=True)
    yes, no = 9, 17
    m = layer_margins(model, out.hidden_states, yes, no)
    assert m.shape == (2, 32)
    torch.testing.assert_close(m[:, -1], (out.logits[:, -1, yes] - out.logits[:, -1, no]).float(), atol=1e-4, rtol=1e-4)
    full = layer_logits(model, tuple(h[:1] for h in out.hidden_states), range(1, 33))   # DeCo's readout, row 0
    torch.testing.assert_close(m[0], full[:, yes] - full[:, no], atol=1e-4, rtol=1e-4)


def test_views():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8))
    assert make_view(img, "orig") is img
    assert np.array_equal(np.asarray(make_view(img, "hflip")), np.asarray(img)[:, ::-1])
    stds = [np.asarray(make_view(img, v), float).std() for v in ("orig", "blur1", "blur2", "blur3")]
    assert stds == sorted(stds, reverse=True)  # more blur, less pixel variance
    assert set(VIEWS) == {"orig", "blur1", "blur2", "blur3", "hflip"}


def test_score_and_threshold():
    df = pd.DataFrame({"label": ["yes", "no", "yes", "no"],
                       **{f"m_{v}": [[float(i + j) for j in range(32)] for i in (3, -1, 2, 0)] for v in VIEWS}})
    np.testing.assert_allclose(score(df, ("orig", "blur1"), 1), [3, -1, 2, 0])
    t = best_threshold([df], ("orig",), 1)
    assert 0 <= t < 2  # any threshold in [0, 2) separates perfectly
