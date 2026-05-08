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
from typing import List, Optional, Sequence


@dataclass
class RenderedPrompt:
    text: str
    token_ids: List[int]
    stop_sequences: List[str]


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
    from tinker import types  # type: ignore[import-not-found]
    from tinker.types import TensorData  # type: ignore[import-not-found]

    prompt_ids = rendered.token_ids
    gen_ids = rollout.generation_token_ids
    full_ids = prompt_ids + gen_ids
    seq_len = len(full_ids)

    pad_prompt = len(prompt_ids)
    target_tokens = torch.tensor(full_ids, dtype=torch.long)

    logprobs_full = [0.0] * pad_prompt + list(rollout.sample_logprobs)
    if len(logprobs_full) < seq_len:
        logprobs_full.extend([0.0] * (seq_len - len(logprobs_full)))
    logprobs_t = torch.tensor(logprobs_full[:seq_len], dtype=torch.float32)

    advantages_full = [0.0] * pad_prompt + [advantage] * len(gen_ids)
    advantages_t = torch.tensor(advantages_full, dtype=torch.float32)

    model_input = types.ModelInput.from_token_ids(prompt_ids)  # type: ignore[attr-defined]

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(target_tokens),
            "logprobs": TensorData.from_torch(logprobs_t),
            "advantages": TensorData.from_torch(advantages_t),
        },
    )
