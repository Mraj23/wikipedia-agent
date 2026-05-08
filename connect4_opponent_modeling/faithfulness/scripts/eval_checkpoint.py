"""CLI: run the full faithfulness evaluation on a model checkpoint.

Two backends:
    --mode local   : transformers via eval/model_loader.create_model_fn.
                     Slow but works without external services. Good for
                     baseline / sanity check on the un-RL'd base model.
    --mode tinker  : Tinker sampling_client. Adapt to your service URL and
                     checkpoint path.

Output: a JSON file with per-category and overall summaries plus a JSONL
of per-board records for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from env.pons_wrapper import PonsSolver
from faithfulness.eval.evaluator import evaluate_checkpoint
from faithfulness.prompt import SYSTEM_PROMPT


def _build_local_sample_fn(model_path: str, max_new_tokens: int, temperature: float):
    """Adapt eval.model_loader.create_model_fn (single-prompt) into the
    sample_fn(messages, n) -> list[str] contract this module expects.
    """
    from eval.model_loader import create_model_fn

    raw = create_model_fn(
        model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    def sample_fn(messages: List[dict], num_samples: int) -> List[str]:
        # Concatenate system+user content into a single prompt; the loader
        # itself applies the chat template if available.
        # We pass only the user content through the loader's render path,
        # so prepend the system content explicitly to preserve instructions.
        # NOTE: this is a pragmatic adapter for the existing loader API,
        # which expects a single prompt string.
        user_block = "\n\n".join(
            m["content"] for m in messages if m["role"] != "system"
        )
        sys_block = next((m["content"] for m in messages if m["role"] == "system"), "")
        prompt = f"{sys_block}\n\n{user_block}" if sys_block else user_block
        return [raw(prompt) for _ in range(num_samples)]

    return sample_fn


def _build_tinker_sample_fn(checkpoint_path: str, base_url: str, max_new_tokens: int):
    """Adapt a Tinker sampling client.

    Imports tinker lazily so the module loads without the dependency.
    """
    import tinker  # type: ignore[import-not-found]

    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client_from_state(
        state_path=checkpoint_path,
    )
    sampling_params = tinker.types.SamplingParams(max_tokens=max_new_tokens)

    def sample_fn(messages: List[dict], num_samples: int) -> List[str]:
        # Render via the service's renderer; here we keep it simple by
        # encoding messages as a JSON-ish string. Real production code should
        # use the same Renderer used by the trainer (see rl/tinker_renderer.py).
        prompt = "\n\n".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        future = sampling_client.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
        result = future.result()
        return [seq.text for seq in result.sequences]

    return sample_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="faithfulness/data/eval_boards.jsonl")
    parser.add_argument("--mode", choices=("local", "tinker"), default="local")
    parser.add_argument("--model-path", required=False, help="local model id or path")
    parser.add_argument("--tinker-checkpoint", required=False)
    parser.add_argument(
        "--tinker-base-url", default=None, help="Tinker ServiceClient base_url"
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--output", required=True, help="Output JSON path for summary"
    )
    parser.add_argument(
        "--records-output",
        default=None,
        help="Optional JSONL path for per-board records",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "local":
        if not args.model_path:
            parser.error("--model-path is required for --mode local")
        sample_fn = _build_local_sample_fn(
            args.model_path, args.max_new_tokens, args.temperature
        )
    else:
        if not args.tinker_checkpoint:
            parser.error("--tinker-checkpoint is required for --mode tinker")
        sample_fn = _build_tinker_sample_fn(
            args.tinker_checkpoint, args.tinker_base_url, args.max_new_tokens
        )

    solver = PonsSolver(strict=True)
    result = evaluate_checkpoint(
        sample_fn,
        args.eval_set,
        solver,
        run_causal=not args.no_causal,
        n_resamples=args.n_resamples,
        threshold=args.threshold,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.summary(), indent=2))

    if args.records_output:
        rec_path = Path(args.records_output)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with rec_path.open("w") as f:
            for r in result.records:
                f.write(
                    json.dumps(
                        {
                            "moves": r.moves,
                            "category": r.category,
                            "chosen_move": r.chosen_move,
                            "valid_json": r.valid_json,
                            "legal": r.legal,
                            "optimal": r.optimal,
                            "regret": r.regret,
                            "truth_labels": r.truth_labels,
                            "causal_labels": r.causal_labels,
                            "max_change_rates": r.max_change_rates,
                            "parse_error": r.parse_error,
                        }
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
