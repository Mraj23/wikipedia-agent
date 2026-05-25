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


async def sample_many(
    sampling_client,
    renderer,
    tokenizer,
    user_msg: str,
    sampling_params,
    n_samples: int = 1,
    system_msg: str | None = None,
):
    """Sample `n_samples` completions for a single user message."""
    from tinker_cookbook.renderers import get_text_content

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
    out = []
    for seq in result.sequences:
        tokens = list(seq.tokens)
        try:
            msg, ok = renderer.parse_response(tokens)
            if ok:
                out.append(str(get_text_content(msg)))
                continue
        except Exception:
            pass
        out.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return out
