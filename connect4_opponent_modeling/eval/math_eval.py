"""GSM8K and MATH-500 evaluation.

Measures whether RL training for adversarial reasoning affects
mathematical reasoning capabilities (capability preservation check).
"""

import re
import random
from typing import Callable, Dict, List, Optional


def _extract_answer(text: str) -> Optional[str]:
    r"""Extract the final numerical answer from model output.

    Looks for patterns like "The answer is X" or \\boxed{X}.

    Args:
        text: Model output string.

    Returns:
        Extracted answer string, or None.
    """
    # Try \boxed{...} first
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Try "The answer is ..."
    answer_match = re.search(r"[Tt]he answer is\s*[:\s]*([^\.\n]+)", text)
    if answer_match:
        return answer_match.group(1).strip()

    # Try "#### ..." (GSM8K format)
    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return hash_match.group(1).strip()

    # Last resort: last number in text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Args:
        answer: Raw answer string.

    Returns:
        Normalized answer.
    """
    if answer is None:
        return ""
    # Remove commas, whitespace, dollar signs
    ans = answer.strip().replace(",", "").replace("$", "").replace("%", "")
    # Try to parse as float for numerical comparison
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans.lower().strip()


def run_gsm8k(
    model_fn: Callable[[str], str],
    n: int = 500,
    seed: int = 42,
) -> Dict:
    """Evaluate on GSM8K test set.

    Tries lm_eval (EleutherAI harness) first; falls back to direct
    HuggingFace datasets load if not installed.

    Args:
        model_fn: Function taking a prompt string and returning model output.
        n: Number of test problems to sample.
        seed: Random seed for sampling.

    Returns:
        Dict with accuracy, n_correct, and n_total.
    """
    problems = _load_gsm8k(n=n, seed=seed)
    if not problems:
        return {"accuracy": -1.0, "n_correct": 0, "n_total": 0, "error": "Could not load GSM8K"}

    correct = 0
    total = 0

    for problem in problems:
        prompt = (
            "Solve the following math problem step by step. "
            "Show your work and give the final answer after 'The answer is'.\n\n"
            f"Problem: {problem['question']}\n\nSolution:"
        )
        response = model_fn(prompt)
        predicted = _extract_answer(response)
        expected = problem["answer"]

        if _normalize_answer(predicted) == _normalize_answer(expected):
            correct += 1
        total += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "n_correct": correct,
        "n_total": total,
    }


def run_math500(
    model_fn: Callable[[str], str],
) -> Dict:
    """Evaluate on MATH-500.

    Tries lm_eval first; falls back to direct HuggingFace datasets load.

    Args:
        model_fn: Function taking a prompt string and returning model output.

    Returns:
        Dict with accuracy and by_subject breakdown.
    """
    problems = _load_math500()
    if not problems:
        return {"accuracy": -1.0, "by_subject": {}, "error": "Could not load MATH-500"}

    by_subject: Dict[str, List[int]] = {}
    correct = 0
    total = 0

    for problem in problems:
        prompt = (
            "Solve the following math problem. Show your work. "
            "Put your final answer in \\boxed{}.\n\n"
            f"Problem: {problem['problem']}\n\nSolution:"
        )
        response = model_fn(prompt)
        predicted = _extract_answer(response)
        expected = problem.get("answer", "")

        is_correct = int(_normalize_answer(predicted) == _normalize_answer(expected))
        correct += is_correct
        total += 1

        subject = problem.get("subject", "unknown")
        if subject not in by_subject:
            by_subject[subject] = []
        by_subject[subject].append(is_correct)

    subject_accuracy = {
        subj: sum(vals) / len(vals) if vals else 0.0
        for subj, vals in by_subject.items()
    }

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "n_correct": correct,
        "n_total": total,
        "by_subject": subject_accuracy,
    }


def _load_gsm8k(n: int = 500, seed: int = 42) -> List[Dict]:
    """Load GSM8K test problems.

    Args:
        n: Number of problems to sample.
        seed: Random seed.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(n, len(ds)))

        problems = []
        for idx in indices:
            item = ds[idx]
            # GSM8K answer format: "...#### <number>"
            answer_text = item["answer"]
            final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
            problems.append({
                "question": item["question"],
                "answer": final_answer,
            })
        return problems
    except Exception as e:
        print(f"Could not load GSM8K: {e}")
        print("Install with: pip install datasets")
        return []


def _load_math500() -> List[Dict]:
    """Load MATH-500 problems.

    Returns:
        List of dicts with 'problem', 'answer', and 'subject' keys.
    """
    try:
        from datasets import load_dataset

        # MATH dataset
        ds = load_dataset("hendrycks/competition_math", split="test")

        # Sample 500 or use all if less
        rng = random.Random(42)
        indices = rng.sample(range(len(ds)), min(500, len(ds)))

        problems = []
        for idx in indices:
            item = ds[idx]
            # Extract answer from \boxed{} in solution
            solution = item.get("solution", "")
            boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
            answer = boxed[-1] if boxed else ""

            problems.append({
                "problem": item["problem"],
                "answer": answer,
                "subject": item.get("type", "unknown"),
            })
        return problems
    except Exception as e:
        print(f"Could not load MATH-500: {e}")
        print("Install with: pip install datasets")
        return []


if __name__ == "__main__":
    print("=== Math Evaluation Demo ===\n")
    print("Answer extraction tests:")

    test_cases = [
        ("The answer is 42.", "42"),
        ("Therefore, \\boxed{7/3}", "7/3"),
        ("Work work work\n#### 156", "156"),
        ("The result is approximately 3.14", "3.14"),
    ]

    for text, expected in test_cases:
        extracted = _extract_answer(text)
        norm = _normalize_answer(extracted)
        status = "OK" if norm == _normalize_answer(expected) else "FAIL"
        print(f"  [{status}] '{text[:50]}...' -> '{extracted}' (expected '{expected}')")
