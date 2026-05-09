"""CLI: generate a base-model-entropy-filtered training pool.

This script can spend Tinker sampling credits when run with
`--backend tinker`. Unit tests exercise the same filtering logic with a fake
sampler, so do not run this script against Tinker until the local checks pass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from env.pons_wrapper import PonsSolver
from faithfulness.eval.entropy_pool import generate_entropy_pool, write_entropy_pool

logger = logging.getLogger(__name__)


def _resolve_tinker_value(value):
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


def _build_tinker_sample_fn(
    *,
    model: str,
    renderer_name: str,
    max_tokens: int,
    temperature: float,
    lora_rank: int,
):
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer, get_text_content
    from transformers import AutoTokenizer

    service_client = tinker.ServiceClient()
    logger.info("Creating LoRA training client for entropy scorer")
    training_client = _resolve_tinker_value(
        service_client.create_lora_training_client(
            base_model=model,
            rank=lora_rank,
        )
    )
    logger.info("Saving initial sampler weights for entropy scorer")
    save_result = _resolve_tinker_value(
        training_client.save_weights_for_sampler(
            name="entropy-pool-base-sampler",
            ttl_seconds=3600,
        )
    )
    logger.info("Creating sampling client from %s", save_result.path)
    sampling_client = _resolve_tinker_value(
        service_client.create_sampling_client(model_path=save_result.path)
    )
    logger.info("Loading tokenizer and renderer")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    renderer = get_renderer(renderer_name, tokenizer, model_name=model)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    def sample_fn(messages: List[dict], num_samples: int) -> List[str]:
        prompt = renderer.build_generation_prompt(messages)
        result = _resolve_tinker_value(
            sampling_client.sample(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
        )
        out = []
        for seq in result.sequences:
            try:
                parsed_message, ok = renderer.parse_response(list(seq.tokens))
                if ok:
                    out.append(str(get_text_content(parsed_message)))
                    continue
            except Exception:
                pass
            out.append(tokenizer.decode(list(seq.tokens), skip_special_tokens=True))
        return out

    return sample_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="faithfulness/data/entropy_training_positions.jsonl",
    )
    parser.add_argument("--n-positions", type=int, default=5000)
    parser.add_argument("--candidate-games", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-board", type=int, default=8)
    parser.add_argument("--backend", choices=("tinker",), default="tinker")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--renderer", default="qwen3_instruct")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--include-tactical-candidates", action="store_true")
    parser.add_argument("--max-most-common-pct", type=float, default=0.625)
    parser.add_argument("--min-valid-rate", type=float, default=0.75)
    parser.add_argument("--min-legal-rate", type=float, default=0.75)
    parser.add_argument("--min-score-spread", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.backend == "tinker":
        sample_fn = _build_tinker_sample_fn(
            model=args.model,
            renderer_name=args.renderer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            lora_rank=args.lora_rank,
        )
    else:  # pragma: no cover - argparse choices prevent this.
        parser.error(f"unsupported backend: {args.backend}")

    records = generate_entropy_pool(
        sample_fn=sample_fn,
        solver=PonsSolver(strict=True),
        n_positions=args.n_positions,
        seed=args.seed,
        candidate_games=args.candidate_games,
        samples_per_board=args.samples_per_board,
        include_tactical_candidates=args.include_tactical_candidates,
        max_candidates=args.max_candidates,
        min_valid_rate=args.min_valid_rate,
        min_legal_rate=args.min_legal_rate,
        max_most_common_pct=args.max_most_common_pct,
        min_score_spread=args.min_score_spread,
        progress_every=args.progress_every,
    )
    write_entropy_pool(records, args.output)
    logging.info("Wrote %d entropy-filtered positions to %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
