"""Shared model loading utilities for evaluation.

All experiment-facing evaluation code should create model callables through
this module so prompt rendering, chat template handling, and decoding are
consistent across baselines and trained checkpoints.
"""

from typing import Callable


def create_model_fn(
    model_path: str,
    *,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Create a model callable from a HuggingFace model or local checkpoint.

    Args:
        model_path: HuggingFace model ID or local checkpoint path.
        max_input_tokens: Prompt truncation length.
        max_new_tokens: Maximum generation length.
        temperature: Sampling temperature. `0.0` means greedy decoding.

    Returns:
        Callable that accepts a plain-text prompt and returns only the model's
        generated completion.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("WARNING: No GPU available. Model inference will be slow on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    has_chat_template = hasattr(tokenizer, "apply_chat_template")

    def render_prompt(prompt: str) -> str:
        if has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def model_fn(prompt: str) -> str:
        rendered = render_prompt(prompt)
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        completion_tokens = outputs[0][prompt_len:]
        return tokenizer.decode(completion_tokens, skip_special_tokens=True)

    return model_fn
