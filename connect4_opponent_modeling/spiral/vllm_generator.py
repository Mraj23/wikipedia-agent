"""Fast generation using vLLM with CPU offloading.

During generation: training model moves to CPU, vLLM uses full GPU.
During training: vLLM is destroyed, training model moves back to GPU.

This avoids the GPU memory sharing problem that limited vLLM to 40%
utilization and caused fragile weight sync via internal APIs.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


class VLLMGenerator:
    """Manages vLLM lifecycle with CPU offloading for training models."""

    def __init__(self, model_path: str, dtype: str = "bfloat16"):
        """Initialize the generator.

        Args:
            model_path: HuggingFace model ID or local path.
            dtype: Model dtype for vLLM ("bfloat16" or "float16").
        """
        self._model_path = model_path
        self._dtype = dtype
        self._weights_dir = tempfile.mkdtemp(prefix="vllm_weights_")
        self._engine = None
        self._step = -1  # Track which step's weights are loaded

    def generate(
        self,
        prompt: str,
        n: int,
        max_tokens: int = 256,
        min_tokens: int = 10,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate n completions for a prompt using vLLM.

        Args:
            prompt: Input prompt string.
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            min_tokens: Minimum tokens per completion.
            temperature: Sampling temperature.

        Returns:
            List of completion strings.
        """
        from vllm import LLM, SamplingParams

        if self._engine is None:
            logger.info("Initializing vLLM engine...")
            self._engine = LLM(
                model=self._weights_dir if self._step >= 0 else self._model_path,
                dtype=self._dtype,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                max_model_len=1024,
            )
            logger.info("vLLM engine ready.")

        params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
        )
        outputs = self._engine.generate([prompt], params)
        return [out.text for out in outputs[0].outputs]

    def update_weights(self, model, tokenizer, step: int) -> None:
        """Save training model weights for vLLM to load.

        Args:
            model: The training model (on CPU or GPU).
            tokenizer: The tokenizer.
            step: Current training step.
        """
        start = time.time()
        model.save_pretrained(self._weights_dir)
        tokenizer.save_pretrained(self._weights_dir)
        self._step = step
        # Destroy current engine so next generate() loads new weights
        self._destroy_engine()
        logger.debug("Weights saved in %.1fs", time.time() - start)

    def _destroy_engine(self) -> None:
        """Destroy the vLLM engine to free GPU memory."""
        if self._engine is not None:
            del self._engine
            self._engine = None
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        """Clean up temporary files and engine."""
        self._destroy_engine()
        import shutil
        if os.path.exists(self._weights_dir):
            shutil.rmtree(self._weights_dir, ignore_errors=True)
