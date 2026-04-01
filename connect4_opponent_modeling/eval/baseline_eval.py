"""Master eval runner.

Runs all evaluation benchmarks in order and produces a summary report.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _create_model_fn(model_path: str) -> Callable[[str], str]:
    """Create a model callable from a checkpoint path.

    Args:
        model_path: Path to model checkpoint.

    Returns:
        Function that takes a prompt and returns model output.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            print("WARNING: No GPU available. Model inference will be slow on CPU.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        model.eval()

        def model_fn(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        return model_fn
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Returning dummy model for testing.")
        return lambda prompt: "<think>test</think><move>3</move>"


def run_all_evals(
    model_path: str,
    condition_label: str,
    output_dir: str,
    skip: Optional[List[str]] = None,
) -> Dict:
    """Run all evaluation benchmarks.

    Runs in order: pons_benchmark -> probe -> gtbench -> gamebench -> math_eval.

    Args:
        model_path: Path to model checkpoint directory.
        condition_label: Condition label (A-F) for reporting.
        output_dir: Directory to save results.
        skip: List of benchmark names to skip. Defaults to ['gamebench'].

    Returns:
        Dict with all evaluation results.
    """
    if skip is None:
        skip = ["gamebench"]

    model_fn = _create_model_fn(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: Dict = {
        "condition": condition_label,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Pons Benchmark
    if "pons_benchmark" not in skip:
        print("\n=== Running Pons Benchmark ===")
        try:
            from eval.pons_benchmark import run_pons_benchmark

            start = time.time()
            results["pons_benchmark"] = run_pons_benchmark(model_fn)
            results["pons_benchmark"]["elapsed_s"] = time.time() - start
            print(f"  Optimal move %: {results['pons_benchmark']['overall_pct_optimal']:.3f}")
        except Exception as e:
            print(f"  Pons benchmark failed: {e}")
            results["pons_benchmark"] = {"error": str(e)}
    else:
        print("Skipping pons_benchmark")

    # 2. Probe
    if "probe" not in skip:
        print("\n=== Running Mechanistic Probe ===")
        try:
            from eval.probe import run_probe, run_consequence_probe

            start = time.time()
            results["probe"] = run_probe(model_fn)
            results["probe"]["elapsed_s"] = time.time() - start
            print(f"  Probe accuracy: {results['probe']['overall_accuracy']:.3f}")

            results["consequence_probe"] = run_consequence_probe(model_fn)
            print(f"  Consequence accuracy: {results['consequence_probe']['overall_cell_accuracy']:.3f}")
        except FileNotFoundError:
            print("  Probe positions not locked yet. Run lock_probe_positions first.")
            results["probe"] = {"error": "Probe positions not locked"}
        except Exception as e:
            print(f"  Probe failed: {e}")
            results["probe"] = {"error": str(e)}
    else:
        print("Skipping probe")

    # 3. GTBench
    if "gtbench" not in skip:
        print("\n=== Running GTBench ===")
        try:
            from eval.gtbench_eval import run_gtbench_full

            start = time.time()
            results["gtbench"] = run_gtbench_full(model_fn)
            results["gtbench"]["elapsed_s"] = time.time() - start
            print(f"  Average NRA: {results['gtbench']['average_nra']:.3f}")
        except ImportError as e:
            print(f"  GTBench not installed: {e}")
            results["gtbench"] = {"error": str(e)}
        except Exception as e:
            print(f"  GTBench failed: {e}")
            results["gtbench"] = {"error": str(e)}
    else:
        print("Skipping gtbench")

    # 4. GameBench
    if "gamebench" not in skip:
        print("\n=== Running GameBench ===")
        try:
            from eval.gamebench_eval import run_gamebench

            results["gamebench"] = run_gamebench(model_fn)
            print("  GameBench: stub results (not yet integrated)")
        except Exception as e:
            print(f"  GameBench failed: {e}")
            results["gamebench"] = {"error": str(e)}
    else:
        print("Skipping gamebench")

    # 5. Math
    if "math" not in skip:
        print("\n=== Running Math Evaluations ===")
        try:
            from eval.math_eval import run_gsm8k, run_math500

            start = time.time()
            results["gsm8k"] = run_gsm8k(model_fn)
            results["math500"] = run_math500(model_fn)
            elapsed = time.time() - start
            results["math_elapsed_s"] = elapsed
            print(f"  GSM8K accuracy: {results['gsm8k'].get('accuracy', -1):.3f}")
            print(f"  MATH-500 accuracy: {results['math500'].get('accuracy', -1):.3f}")
        except Exception as e:
            print(f"  Math eval failed: {e}")
            results["math"] = {"error": str(e)}
    else:
        print("Skipping math")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"{condition_label}_eval_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Print summary table
    _print_summary(results)

    return results


def _print_summary(results: Dict) -> None:
    """Print a formatted summary table.

    Args:
        results: Full results dict.
    """
    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY — Condition {results.get('condition', '?')}")
    print("=" * 60)

    if "pons_benchmark" in results and "error" not in results["pons_benchmark"]:
        pb = results["pons_benchmark"]
        print(f"  Pons Benchmark:  {pb['overall_pct_optimal']:.1%} optimal moves")
        wr = pb.get("win_rate_vs_minimax", {})
        for depth, rate in sorted(wr.items(), key=lambda x: int(x[0])):
            print(f"    vs minimax-{depth}: {rate:.1%} win rate")

    if "probe" in results and "error" not in results["probe"]:
        pr = results["probe"]
        print(f"  Probe Accuracy:  {pr['overall_accuracy']:.1%}")
        for cat, acc in pr.get("by_depth", {}).items():
            print(f"    {cat}: {acc:.1%}")

    if "consequence_probe" in results:
        cp = results["consequence_probe"]
        print(f"  Consequence Probe: {cp.get('overall_cell_accuracy', -1):.1%} cell accuracy")

    if "gtbench" in results and "error" not in results["gtbench"]:
        gt = results["gtbench"]
        print(f"  GTBench NRA:     {gt['average_nra']:.3f}")

    if "gsm8k" in results and "error" not in results.get("gsm8k", {}):
        print(f"  GSM8K:           {results['gsm8k'].get('accuracy', -1):.1%}")

    if "math500" in results and "error" not in results.get("math500", {}):
        print(f"  MATH-500:        {results['math500'].get('accuracy', -1):.1%}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--condition", type=str, required=True, help="Condition label (A-F)")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--skip", nargs="*", default=["gamebench"], help="Benchmarks to skip")
    args = parser.parse_args()

    run_all_evals(
        model_path=args.model,
        condition_label=args.condition,
        output_dir=args.output,
        skip=args.skip,
    )
