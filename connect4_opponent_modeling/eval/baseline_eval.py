"""Canonical experiment evaluation suite.

Despite the historical filename, this is the source-of-truth runner for
experiment evaluation after the May 1, 2026 cleanup. It uses the same model
loading behavior and ladder harness across baselines, prompt-only F, and
trained checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from eval.game_ladder import (
    PROMPT_STYLE_BASE,
    PROMPT_STYLE_OPPONENT_AWARE,
    run_difficulty_ladder,
    summarize_game_results,
)
from eval.model_loader import create_model_fn


def _default_prompt_style(condition_label: str) -> str:
    if condition_label == "F":
        return PROMPT_STYLE_OPPONENT_AWARE
    return PROMPT_STYLE_BASE


def run_all_evals(
    model_path: str,
    condition_label: str,
    output_dir: str,
    *,
    skip: Optional[List[str]] = None,
    prompt_style: Optional[str] = None,
    transfer_games: Sequence[str] = ("connect_four", "breakthrough", "nim"),
    num_games: int = 50,
    seed: int = 42,
) -> Dict:
    """Run the canonical evaluation suite."""
    if skip is None:
        skip = ["math"]

    resolved_prompt_style = prompt_style or _default_prompt_style(condition_label)
    model_fn = create_model_fn(model_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: Dict = {
        "condition": condition_label,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "prompt_style": resolved_prompt_style,
        "num_games": num_games,
        "seed": seed,
        "transfer_games": list(transfer_games),
        "skip": skip,
    }

    if "pons_benchmark" not in skip:
        print("\n=== Running Pons Benchmark ===")
        from eval.pons_benchmark import run_pons_benchmark

        start = time.time()
        results["pons_benchmark"] = run_pons_benchmark(
            model_fn,
            condition_label=condition_label,
            seed=seed,
        )
        results["pons_benchmark"]["elapsed_s"] = time.time() - start
        print(f"  Optimal move %: {results['pons_benchmark']['overall_pct_optimal']:.3f}")
    else:
        print("Skipping pons_benchmark")

    if "probe" not in skip:
        print("\n=== Running Neutral Opponent Probe ===")
        from eval.probe import run_consequence_probe, run_probe

        start = time.time()
        results["probe"] = run_probe(model_fn)
        results["probe"]["elapsed_s"] = time.time() - start
        print(f"  Probe accuracy: {results['probe']['overall_accuracy']:.3f}")

        results["consequence_probe"] = run_consequence_probe(model_fn)
        print(
            "  Consequence accuracy: "
            f"{results['consequence_probe']['overall_cell_accuracy']:.3f}"
        )
    else:
        print("Skipping probe")

    if "difficulty_ladder" not in skip:
        print("\n=== Running Difficulty Ladder ===")
        ladder_dir = output_path / f"{condition_label.lower()}_ladder_logs"
        start = time.time()
        ladder = run_difficulty_ladder(
            model_fn,
            games=transfer_games,
            num_games=num_games,
            prompt_style=resolved_prompt_style,
            output_dir=str(ladder_dir),
            seed=seed,
        )
        ladder["elapsed_s"] = time.time() - start
        results["difficulty_ladder"] = ladder
        results["transfer_summary"] = summarize_game_results(ladder)
        bt_score = results["transfer_summary"].get("breakthrough_transfer_score")
        if bt_score is not None:
            print(f"  Breakthrough transfer score: {bt_score:.3f}")
        clean_bt = results["transfer_summary"].get("breakthrough_clean_transfer_score")
        if clean_bt is not None:
            print(f"  Breakthrough clean-game transfer score: {clean_bt:.3f}")
    else:
        print("Skipping difficulty_ladder")

    if "math" not in skip:
        print("\n=== Running Math Evaluations ===")
        from eval.math_eval import run_gsm8k, run_math500

        start = time.time()
        results["gsm8k"] = run_gsm8k(model_fn)
        results["math500"] = run_math500(model_fn)
        results["math_elapsed_s"] = time.time() - start
        print(f"  GSM8K accuracy: {results['gsm8k'].get('accuracy', -1):.3f}")
        print(f"  MATH-500 accuracy: {results['math500'].get('accuracy', -1):.3f}")
    else:
        print("Skipping math")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"{condition_label}_eval_{timestamp}.json"
    with open(results_file, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    _print_summary(results)
    return results


def _print_summary(results: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY — Condition {results.get('condition', '?')}")
    print("=" * 60)
    print(f"  Prompt Style:    {results.get('prompt_style', 'unknown')}")
    print(f"  Eval Seed:       {results.get('seed', 'unknown')}")

    if "pons_benchmark" in results and "error" not in results["pons_benchmark"]:
        pb = results["pons_benchmark"]
        print(f"  Pons Benchmark:  {pb['overall_pct_optimal']:.1%} optimal moves")

    if "probe" in results and "error" not in results["probe"]:
        print(f"  Probe Accuracy:  {results['probe']['overall_accuracy']:.1%}")

    if "consequence_probe" in results:
        cp = results["consequence_probe"]
        print(f"  Consequence:     {cp.get('overall_cell_accuracy', -1):.1%}")

    transfer_summary = results.get("transfer_summary", {})
    if transfer_summary:
        bt_score = transfer_summary.get("breakthrough_transfer_score")
        if bt_score is not None:
            print(f"  Breakthrough:    {bt_score:.1%} hard-opponent transfer score")
        clean_bt = transfer_summary.get("breakthrough_clean_transfer_score")
        if clean_bt is not None:
            print(f"  Breakthrough*:   {clean_bt:.1%} clean-game transfer score")
        nim_score = transfer_summary.get("nim_transfer_score")
        if nim_score is not None:
            print(f"  Nim Transfer:    {nim_score:.1%} hard-opponent transfer score")

    if "gsm8k" in results and "error" not in results.get("gsm8k", {}):
        print(f"  GSM8K:           {results['gsm8k'].get('accuracy', -1):.1%}")
    if "math500" in results and "error" not in results.get("math500", {}):
        print(f"  MATH-500:        {results['math500'].get('accuracy', -1):.1%}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the canonical experiment evaluation suite")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path or HF model ID")
    parser.add_argument("--condition", type=str, required=True, help="Condition label (A-F)")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument(
        "--prompt_style",
        choices=[PROMPT_STYLE_BASE, PROMPT_STYLE_OPPONENT_AWARE],
        default=None,
        help="Override the default prompt style for this condition",
    )
    parser.add_argument(
        "--games",
        nargs="*",
        default=["connect_four", "breakthrough", "nim"],
        help="Transfer games to run in the difficulty ladder",
    )
    parser.add_argument("--num_games", type=int, default=50, help="Games per ladder opponent")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation RNG seed")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=["math"],
        help="Suite components to skip: pons_benchmark probe difficulty_ladder math",
    )
    args = parser.parse_args()

    run_all_evals(
        model_path=args.model,
        condition_label=args.condition,
        output_dir=args.output,
        skip=args.skip,
        prompt_style=args.prompt_style,
        transfer_games=args.games,
        num_games=args.num_games,
        seed=args.seed,
    )
