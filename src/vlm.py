"""Model-agnostic wrapper for the journal extension: one interface for every VLM we evaluate.

    vlm = load_vlm("llava-1.5-7b")        # or "qwen2.5-vl-7b", "instructblip-7b"
    inputs = vlm.encode(image, question)   # dict of tensors on CPU (pixel values fp32)
    out = vlm.forward(inputs)              # final logits + per-layer last-token hidden states

Every model gets the same POPE prompt content (question + " Please answer this question with one
word.") in its own chat format, and the same first-token score: m = logsumexp(Yes variants) -
logsumexp(No variants). For LLaVA-1.5 the variant sets are {"▁Yes"} / {"▁No"}, which reproduces
the course-project margin exactly (src/model.py). `add_vcd_noise` adds VCD's forward-diffusion
noise to the normalised pixel tensor the model sees (for Qwen's flattened patches, one noise draw
is shared by the two temporal copies of each patch, so it equals noising the image itself).
"""
from dataclasses import dataclass, field

import torch

from src.model import POPE_SUFFIX, SYSTEM, pick_device

MODELS = {
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",   # edge-sized model (Jetson-class deployment)
    "instructblip-7b": "Salesforce/instructblip-vicuna-7b",
}
QWEN_MAX_PIXELS = 640 * 640   # COCO images (<= 640 px) are fed at native resolution


@dataclass
class VLM:
    name: str
    model: torch.nn.Module
    processor: object
    yes_ids: list[int]
    no_ids: list[int]
    norm: torch.nn.Module = field(repr=False)
    lm_head: torch.nn.Module = field(repr=False)

    @property
    def device(self):
        return self.model.device

    @property
    def n_layers(self) -> int:
        return self.model.config.get_text_config().num_hidden_layers

    def prompt(self, question: str) -> str:
        q = question + POPE_SUFFIX
        if self.name.startswith("llava"):
            return f"{SYSTEM} USER: <image>\n{q} ASSISTANT:"
        if self.name.startswith("qwen"):
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if self.name.startswith("instructblip"):
            return q
        raise ValueError(self.name)

    def encode(self, image, question: str) -> dict:
        kw = {}
        if self.name.startswith("qwen"):
            kw = {"images_kwargs": {"max_pixels": QWEN_MAX_PIXELS}}
        inputs = dict(self.processor(images=image, text=self.prompt(question), return_tensors="pt", **kw))
        pv = inputs.get("pixel_values")
        if self.name.startswith("instructblip") and pv is not None and pv.ndim == 5:
            inputs["pixel_values"] = pv[:, 0]   # transformers 5.x processor nests images: (1, 1, 3, H, W)
        return inputs

    def _to_device(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(self.model.dtype)
            out[k] = v
        return out

    @torch.inference_mode()
    def forward(self, inputs: dict, hidden: bool = True):
        """Returns (final-position logits [vocab] fp32, tuple of last-token hidden states per layer
        (index 0 = embeddings, index n_layers = final layer, already normed) or None)."""
        logits, hs = self.forward_batch([inputs], hidden)
        return logits[0], (tuple(h[0] for h in hs) if hidden else None)

    @torch.inference_mode()
    def forward_batch(self, batch: list[dict], hidden: bool = True):
        """One forward pass over several encodings of the SAME prompt (e.g. real + VCD-noised image,
        or several views of one image), which share a token length, so no padding is involved.
        Returns (logits [B, vocab] fp32, tuple over layers of [B, dim] last-token hidden states)."""
        if len(batch) == 1:
            inputs = batch[0]
        else:
            n = batch[0]["input_ids"].shape[1]
            assert all(b["input_ids"].shape[1] == n for b in batch), "batched encodings must share length"
            inputs = {k: torch.cat([b[k] for b in batch], 0) if torch.is_tensor(batch[0][k]) else batch[0][k]
                      for k in batch[0]}
        out = self.model(**self._to_device(inputs), output_hidden_states=hidden, use_cache=False)
        hs = None
        if hidden:
            hs = getattr(out, "hidden_states", None)
            if hs is None:   # InstructBLIP: the LM's outputs are nested
                hs = out.language_model_outputs.hidden_states
            hs = tuple(h[:, -1] for h in hs)
        return out.logits[:, -1].float(), hs

    def margin(self, logits: torch.Tensor) -> float:
        return float(torch.logsumexp(logits[self.yes_ids], 0) - torch.logsumexp(logits[self.no_ids], 0))

    @torch.inference_mode()
    def layer_logits(self, hs) -> torch.Tensor:
        """Logit-lens readout of layers 1..n_layers-1 through the final norm + LM head ([L-1, vocab])."""
        mid = torch.stack(hs[1:-1])
        return self.lm_head(self.norm(mid)).float()

    def add_vcd_noise(self, inputs: dict, noise_step: int, generator=None) -> dict:
        from src.vcd import add_diffusion_noise
        key = "pixel_values"
        pv = inputs[key]
        if self.name.startswith("qwen"):
            cfg = self.model.config.vision_config
            t, p = cfg.temporal_patch_size, cfg.patch_size
            x = pv.view(pv.shape[0], 3, t, p, p)[:, :, :1]           # one temporal copy
            noisy = add_diffusion_noise(x, noise_step, generator=generator).expand(-1, -1, t, -1, -1)
            new = noisy.reshape(pv.shape)
        else:
            new = add_diffusion_noise(pv, noise_step, generator=generator)
        return {**inputs, key: new}


def _token_ids(tok, words) -> list[int]:
    ids = []
    for w in words:
        i = tok.convert_tokens_to_ids(w)
        if i is not None and i != tok.unk_token_id and i not in ids:
            ids.append(i)
    assert ids, words
    return ids


def _final_norm(model):
    found = [m for n, m in model.named_modules()
             if n.endswith("norm") and "language_model" in n and ".layers." not in n]
    if len(found) != 1:
        raise AttributeError("final norm not unique")
    return found[0]


def load_vlm(name: str, device=None, dtype=None, quant: str | None = None) -> VLM:
    """quant: None (fp16/bf16), "int8" (LLM.int8) or "int4" (NF4 weights, fp16 compute) via bitsandbytes,
    which quantizes the Linear layers of the whole model (vision tower included) except the LM head."""
    import transformers as T
    device = device or pick_device()
    dtype = dtype or (torch.float32 if device.type == "cpu" else torch.float16)
    repo = MODELS[name]
    if name.startswith("llava"):
        cls, yes, no = T.LlavaForConditionalGeneration, ["▁Yes"], ["▁No"]
    elif name.startswith("qwen"):
        cls, yes, no = T.Qwen2_5_VLForConditionalGeneration, ["Yes", "yes"], ["No", "no"]
        dtype = torch.bfloat16 if device.type == "cuda" else dtype   # Qwen2.5-VL overflows in fp16
    elif name.startswith("instructblip"):
        cls, yes, no = T.InstructBlipForConditionalGeneration, ["▁Yes", "▁yes"], ["▁No", "▁no"]
    else:
        raise ValueError(name)
    if quant is None:
        model = cls.from_pretrained(repo, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    else:
        qc = T.BitsAndBytesConfig(load_in_8bit=True) if quant == "int8" else T.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        model = cls.from_pretrained(repo, dtype=dtype, quantization_config=qc, device_map={"": 0}).eval()
    processor = T.AutoProcessor.from_pretrained(repo)
    tok = processor.tokenizer
    return VLM(name, model, processor, _token_ids(tok, yes), _token_ids(tok, no),
               _final_norm(model), model.get_output_embeddings())
