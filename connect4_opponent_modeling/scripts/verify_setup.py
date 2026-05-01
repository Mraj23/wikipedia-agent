"""Verify that a machine is ready for the active experiment protocol."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, f"import ok: {module_name}"
    except Exception as exc:  # pragma: no cover - exercised by CLI
        return False, f"import failed: {module_name} ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify experiment bootstrap state.")
    parser.add_argument("--expect-gpu", action="store_true", help="Require CUDA availability.")
    parser.add_argument("--expect-vllm", action="store_true", help="Require vLLM import.")
    parser.add_argument("--expect-wandb", action="store_true", help="Require wandb import.")
    args = parser.parse_args()

    checks: list[tuple[bool, str]] = []

    for module_name in ["torch", "transformers", "pyspiel", "numpy"]:
        checks.append(_check_import(module_name))
    if args.expect_vllm:
        checks.append(_check_import("vllm"))
    if args.expect_wandb:
        checks.append(_check_import("wandb"))

    import torch

    if args.expect_gpu:
        checks.append((torch.cuda.is_available(), "CUDA available"))

    solver_path = PROJECT_ROOT / "connect4_solver"
    checks.append((solver_path.is_file(), f"solver exists: {solver_path}"))
    checks.append((os.access(solver_path, os.X_OK), f"solver executable: {solver_path}"))

    book_path = PROJECT_ROOT / "7x6.book"
    checks.append((book_path.is_file(), f"opening book exists: {book_path}"))

    probe_path = PROJECT_ROOT / "data" / "probe_positions_locked.jsonl"
    checks.append((probe_path.is_file(), f"probe positions exist: {probe_path}"))

    from env.connect_four_env import ConnectFourEnv
    from env.pons_wrapper import PonsSolver

    solver = PonsSolver()
    checks.append((solver.is_available(), "Pons solver + opening book available"))

    # Exercise the solver path to catch missing opening-book cwd issues.
    try:
        best_move = solver.best_move(ConnectFourEnv())
        checks.append((0 <= best_move <= 6, f"solver returned legal opening move: {best_move}"))
    except Exception as exc:  # pragma: no cover - exercised by CLI
        checks.append((False, f"solver execution failed: {exc}"))

    print("=== Setup Verification ===")
    failed = False
    for ok, message in checks:
        prefix = "OK" if ok else "FAIL"
        print(f"[{prefix}] {message}")
        failed = failed or not ok

    if not failed:
        print(f"VLLM_USE_DEEP_GEMM={os.environ.get('VLLM_USE_DEEP_GEMM', '<unset>')}")
        print("Environment looks ready for training and evaluation.")
        return 0

    print("Setup verification failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
