"""LLaVA-1.5 loading and POPE prompt construction.

The prompt reproduces the VCD reference (llava_v1 conversation template plus
"Please answer this question with one word."), rendered as a plain string for the HF processor.
"""
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
POPE_SUFFIX = " Please answer this question with one word."


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str = MODEL_ID, device: torch.device | None = None):
    device = device or pick_device()
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def pope_prompt(question: str) -> str:
    return f"{SYSTEM} USER: <image>\n{question}{POPE_SUFFIX} ASSISTANT:"


def encode(processor, image, question: str):
    """Returns (input_ids, pixel_values). pixel_values stays fp32 on CPU so VCD noise is added
    to the exact normalized tensor the model sees, before the cast to the model dtype."""
    inputs = processor(images=image, text=pope_prompt(question), return_tensors="pt")
    return inputs["input_ids"], inputs["pixel_values"]


def answer_token_ids(processor) -> dict[str, int]:
    """Token ids of the first answer token for 'Yes' / 'No' (SentencePiece word-start form)."""
    tok = processor.tokenizer
    ids = {"yes": tok.convert_tokens_to_ids("▁Yes"), "no": tok.convert_tokens_to_ids("▁No")}
    assert tok.unk_token_id not in ids.values(), ids
    return ids
