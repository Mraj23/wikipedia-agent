"""Periodic evaluation callback during GRPO training.

Runs pons_benchmark + probe (fast evals) at regular intervals.
GTBench and math evals are skipped during training for speed.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from env.pons_wrapper import PonsSolver

logger = logging.getLogger(__name__)


class EvalCallback:
    """Runs lightweight evaluation during training and tracks best checkpoint."""

    def __init__(
        self,
        model_fn_factory: Callable[[], Callable[[str], str]],
        condition: str,
        log_dir: str,
        checkpoint_dir: str,
        save_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize the eval callback.

        Args:
            model_fn_factory: Callable that returns a fresh model_fn for eval.
                This allows the eval to use the latest model weights.
            condition: Condition label (B-E).
            log_dir: Directory for eval result JSON files.
            checkpoint_dir: Base directory for saving best checkpoints.
            save_fn: Function to save model checkpoint to a given path.
        """
        self._model_fn_factory = model_fn_factory
        self._condition = condition
        self._log_dir = Path(log_dir)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._save_fn = save_fn
        self._best_pons_score = -1.0
        self._solver = PonsSolver()
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, step: int) -> Dict:
        """Run evaluation at the given training step.

        Runs pons_benchmark (if benchmark files exist) and probe (if locked).
        Saves results and updates best checkpoint.

        Args:
            step: Current training step.

        Returns:
            Dict of evaluation results.
        """
        model_fn = self._model_fn_factory()
        results: Dict = {
            "step": step,
            "condition": self._condition,
            "timestamp": datetime.now().isoformat(),
        }

        # Pons benchmark (in-domain)
        try:
            from eval.pons_benchmark import run_pons_benchmark
            pons_results = run_pons_benchmark(
                model_fn, solver=self._solver
            )
            results["pons_benchmark"] = pons_results
            pons_score = pons_results.get("overall_pct_optimal", 0.0)
            logger.info(
                "Step %d: Pons optimal=%.3f", step, pons_score
            )
        except Exception as e:
            logger.warning("Pons benchmark failed at step %d: %s", step, e)
            pons_score = 0.0
            results["pons_benchmark"] = {"error": str(e)}

        # Probe (opponent prediction)
        try:
            from eval.probe import run_probe
            probe_results = run_probe(model_fn, solver=self._solver)
            results["probe"] = probe_results
            logger.info(
                "Step %d: Probe accuracy=%.3f",
                step, probe_results.get("overall_accuracy", 0.0),
            )
        except FileNotFoundError:
            logger.info("Probe positions not locked yet, skipping probe.")
            results["probe"] = {"error": "not_locked"}
        except Exception as e:
            logger.warning("Probe failed at step %d: %s", step, e)
            results["probe"] = {"error": str(e)}

        # Save results
        result_file = self._log_dir / f"eval_step_{step}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Track best checkpoint
        if pons_score > self._best_pons_score and self._save_fn is not None:
            self._best_pons_score = pons_score
            best_path = str(self._checkpoint_dir / "best")
            self._save_fn(best_path)
            logger.info(
                "New best checkpoint at step %d (pons=%.3f), saved to %s",
                step, pons_score, best_path,
            )

        return results
