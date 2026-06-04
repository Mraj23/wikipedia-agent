"""Shared Tinker helpers used by generators and eval."""

import asyncio
import inspect
from typing import Any


async def resolve_async(value: Any) -> Any:
    """Awaits coroutines and resolves Tinker futures (`.result_async`).

    Mirrors the bridge used in
    connect4_opponent_modeling/experiments/opponent_next_move/tinker_train.py
    so the same code works across Tinker SDK versions.
    """
    if inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if callable(result_async):
        return await result_async()
    return value


async def build_sampling_client(
    base_model: str,
    renderer_name: str,
    lora_rank: int = 1,
    name: str = "caveman-rltf",
):
    """Base-model sampling client via the rank-1 LoRA trick."""
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer

    service_client = tinker.ServiceClient()
    training_client = await resolve_async(
        service_client.create_lora_training_client_async(
            base_model=base_model, rank=lora_rank
        )
    )
    save_result = await resolve_async(
        training_client.save_weights_for_sampler_async(
            name=name, ttl_seconds=3600
        )
    )
    sampling_client = await resolve_async(
        service_client.create_sampling_client_async(model_path=save_result.path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)
    return sampling_client, renderer, tokenizer


async def build_sampling_client_from_path(
    sampler_path: str, base_model: str, renderer_name: str
):
    """Sampling client for a trained checkpoint (tinker:// URI)."""
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer

    service_client = tinker.ServiceClient()
    sampling_client = await resolve_async(
        service_client.create_sampling_client_async(model_path=sampler_path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)
    return sampling_client, renderer, tokenizer


_CONTROL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def clean_generation(tokenizer, tokens) -> str:
    """Decode generated tokens to text, KEEPING <think>/</think> markers but
    dropping chat control tokens. We decode with special tokens visible (so the
    </think> boundary survives) then trim at the end-of-message marker.

    For thinking models the prompt already opened "<think>\\n", so the
    generated text looks like:  <thinking>...\\n</think>\\n\\n<answer>
    """
    text = tokenizer.decode(list(tokens), skip_special_tokens=False)
    for ctrl in ("<|im_end|>", "<|endoftext|>"):
        if ctrl in text:
            text = text.split(ctrl)[0]
    # Drop any stray role headers that leaked in.
    text = text.replace("<|im_start|>assistant", "").replace("<|im_start|>", "")
    return text.strip()


async def sample_many(
    sampling_client,
    renderer,
    tokenizer,
    user_msg: str,
    sampling_params,
    n_samples: int = 1,
    system_msg: str | None = None,
):
    """Sample `n_samples` completions for a single user message.

    Returns a list of dicts: {"text": <full output incl think block>,
    "n_tokens": <generated token count>}. Token count is exact (len of the
    sampled tokens), avoiding any dependency on re-tokenizing parsed text.
    """
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    prompt = renderer.build_generation_prompt(messages)
    result = await resolve_async(
        sampling_client.sample_async(
            prompt=prompt,
            num_samples=n_samples,
            sampling_params=sampling_params,
        )
    )
    return [
        {"text": clean_generation(tokenizer, seq.tokens), "n_tokens": len(seq.tokens)}
        for seq in result.sequences
    ]
