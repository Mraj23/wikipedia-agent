"""Tests for Tinker datum alignment without importing the real SDK."""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.rl.tinker_renderer import RenderedPrompt, RolloutTokens, build_datum


@dataclass
class FakeEncodedTextChunk:
    tokens: list


class FakePrompt:
    def __init__(self, length):
        self.length = length
        self.appended = None

    def append(self, chunk):
        self.appended = chunk
        return FakeModelInput(self.length + len(chunk.tokens))


class FakeModelInput:
    def __init__(self, length):
        self.length = length


class FakeTensorData:
    @staticmethod
    def from_torch(tensor):
        return tensor


class FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class FakeTinker:
    EncodedTextChunk = FakeEncodedTextChunk
    Datum = FakeDatum


def test_build_datum_aligns_prompt_generation_targets():
    prompt = FakePrompt(length=5)
    rendered = RenderedPrompt(
        text="",
        token_ids=[],
        stop_sequences=[],
        tinker_prompt=prompt,
    )
    rollout = RolloutTokens(
        prompt_token_ids=[],
        generation_token_ids=[10, 11, 12, 13],
        sample_logprobs=[-0.1, -0.2, -0.3, -0.4],
        completion_text="{}",
    )

    datum = build_datum(
        rendered=rendered,
        rollout=rollout,
        advantage=0.75,
        tinker_module=FakeTinker,
        tensor_data_cls=FakeTensorData,
    )

    assert datum is not None
    assert prompt.appended.tokens == [10, 11, 12]
    assert datum.model_input.length == 8
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [0, 0, 0, 0, 10, 11, 12, 13]
    assert len(datum.loss_fn_inputs["logprobs"]) == datum.model_input.length
    assert len(datum.loss_fn_inputs["advantages"]) == datum.model_input.length
