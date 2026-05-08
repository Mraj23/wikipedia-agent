"""Renderer that bridges chat-template messages to Tinker tensor inputs.

Mirrors the renderer pattern in tinker-cookbook (recipes/rl_loop.py). The
exact Tinker wire types are imported lazily so this module can be imported
in environments without the SDK (tests, eval-only paths).

Responsibilities:
    1. Apply the model's chat template to a list of messages with
       add_generation_prompt=True, returning the prompt token IDs and the
       (start) length used by the trainer to pad logprobs/advantages.
    2. Decode the generation tokens back to text for reward scoring.
    3. Build a Tinker `Datum` with `loss_fn_inputs` containing
       target_tokens, logprobs, and advantages, all padded to the full
       model_input length.

The exact Tinker tensor shape contract has evolved across SDK versions;
this module isolates that detail so the trainer doesn't have to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass
class RenderedPrompt:
    text: str
    token_ids: List[int]
    stop_sequences: List[str]
    tinker_prompt: Optional[Any] = None


@dataclass
class RolloutTokens:
    prompt_token_ids: List[int]
    generation_token_ids: List[int]
    sample_logprobs: List[float]
    completion_text: str


class FaithfulnessRenderer:
    """Wraps a HuggingFace tokenizer for the chat-template rendering Tinker
    expects, and provides helpers for decoding sampled tokens.

    Parameters
    ----------
    tokenizer : transformers tokenizer
        Must support apply_chat_template and decode().
    stop_sequences : list[str], optional
        Stop tokens to pass into Tinker's SamplingParams.
    """

    def __init__(self, tokenizer, stop_sequences: Optional[Sequence[str]] = None) -> None:
        self.tokenizer = tokenizer
        self._stop_sequences = list(stop_sequences) if stop_sequences else []

    def get_stop_sequences(self) -> List[str]:
        return list(self._stop_sequences)

    def render_prompt(self, messages: List[dict]) -> RenderedPrompt:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return RenderedPrompt(
            text=text,
            token_ids=list(token_ids),
            stop_sequences=self.get_stop_sequences(),
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


def build_datum(
    *,
    rendered: RenderedPrompt,
    rollout: RolloutTokens,
    advantage: float,
    tinker_module: Optional[Any] = None,
    tensor_data_cls: Optional[Any] = None,
):
    """Construct a Tinker `Datum` for one rollout.

    Imports `tinker.types` lazily; errors here indicate Tinker is not
    installed or the wire format has shifted (see SDK release notes).

    The contract from tinker-cookbook recipes/rl_loop.py:
        loss_fn_inputs = {
            "target_tokens":  full_sequence_token_ids (prompt + generation),
            "logprobs":        zeros over prompt, sample logprobs over generation,
            "advantages":      zeros over prompt, advantage scalar over generation,
        }
    All three are padded to the full model_input.length.
    """
    import torch

    if tinker_module is None or tensor_data_cls is None:
        import tinker as tinker_module  # type: ignore[import-not-found,no-redef]
        from tinker import TensorData as tensor_data_cls  # type: ignore[import-not-found,no-redef]

    prompt_ids = rendered.token_ids
    gen_ids = rollout.generation_token_ids

    if len(gen_ids) < 2:
        return None

    if rendered.tinker_prompt is not None:
        ob_len = rendered.tinker_prompt.length - 1
        model_input = rendered.tinker_prompt.append(
            tinker_module.EncodedTextChunk(tokens=list(gen_ids[:-1]))
        )
        target_tokens = [0] * ob_len + list(gen_ids)
    else:
        ob_len = len(prompt_ids)
        model_input = tinker_module.ModelInput.from_token_ids(
            prompt_ids + list(gen_ids[:-1])
        )
        target_tokens = [0] * ob_len + list(gen_ids)

    padded_logprobs = [0.0] * ob_len + [float(x) for x in rollout.sample_logprobs]
    padded_advantages = [0.0] * ob_len + [float(advantage)] * (model_input.length - ob_len)

    return tinker_module.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tensor_data_cls.from_torch(torch.tensor(target_tokens)),
            "logprobs": tensor_data_cls.from_torch(torch.tensor(padded_logprobs)),
            "advantages": tensor_data_cls.from_torch(torch.tensor(padded_advantages)),
        },
    )
