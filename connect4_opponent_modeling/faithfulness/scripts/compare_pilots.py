"""Side-by-side comparison of trained checkpoints + base on the eval set.

For each named run, this samples one completion per held-out board with the
training-time prompt condition, then reports for that run:

- chosen-move quality (legal_rate, optimal_rate, mean_clipped_regret)
- schema validity rate
- per-tactical-field exact-match, false-negative-given-gt, false-positive
- aggregate exact-match rate

Includes a base-model row sampled directly from the LoRA-rank-1 base
trick used in preflight. Output is a single JSON file plus a printed
table for quick reading.

Usage example (after both pilots finish):

    TINKER_API_KEY=... WANDB_API_KEY=... \
    /tmp/tinker_venv/bin/python -m faithfulness.scripts.compare_pilots \
        --eval-set faithfulness/data/eval_boards.jsonl --n-boards 100 \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
        --run move_only:tactical_claims:tinker://...path... \
        --run tactical_lambda0:tactical_claims:tinker://...path... \
        --output faithfulness/data/runs/compare_pilots_20260509.json

Each --run argument is "<label>:<condition>:<checkpoint_path_or_BASE>".
Use BASE as the third field to evaluate the un-trained base model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.claims import CLAIM_TYPE_TO_TACTICAL_FIELD, ClaimType
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import make_messages
from faithfulness.verifier.claim_verifier import ground_truth_tactical_claims
from faithfulness.verifier.move_evaluator import evaluate_move


FIELDS = (
    "self_immediate_win_columns",
    "opponent_immediate_win_columns",
    "unsafe_moves",
    "self_double_threat_moves",
    "self_single_threat_moves",
)


def _resolve_tinker_value(value):
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


def _build_sample_fn(checkpoint_path: str, base_model: str, base_url, max_tokens, temperature):
    """Build a sample_fn(messages) -> str for either a trained checkpoint or
    the base model (when checkpoint_path == 'BASE').
    """
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer, get_text_content

    service_kwargs = {}
    if base_url is not None:
        service_kwargs["base_url"] = base_url
    service_client = tinker.ServiceClient(**service_kwargs)

    if checkpoint_path == "BASE":
        training_client = _resolve_tinker_value(
            service_client.create_lora_training_client(base_model=base_model, rank=1)
        )
        save_result = _resolve_tinker_value(
            training_client.save_weights_for_sampler(
                name="compare-base", ttl_seconds=3600
            )
        )
        path = save_result.path
    else:
        path = checkpoint_path

    sampling_client = _resolve_tinker_value(
        service_client.create_sampling_client(model_path=path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer, model_name=base_model)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    def sample_fn(messages):
        prompt = renderer.build_generation_prompt(messages)
        result = _resolve_tinker_value(
            sampling_client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )
        )
        seq = result.sequences[0]
        try:
            parsed_message, ok = renderer.parse_response(list(seq.tokens))
            if ok:
                return str(get_text_content(parsed_message))
        except Exception:
            pass
        return tokenizer.decode(list(seq.tokens), skip_special_tokens=True)

    return sample_fn


def _env_from_moves(moves):
    env = ConnectFourEnv()
    for m in moves:
        env.make_move(int(m))
    return env


def _model_sets_from_parsed(parsed):
    out = {f: None for f in FIELDS}
    for claim in parsed.claims:
        field_name = CLAIM_TYPE_TO_TACTICAL_FIELD.get(claim.type)
        if field_name is None:
            continue
        if claim.type is ClaimType.SET_UNSAFE_MOVES:
            out[field_name] = list(claim.fields.get("entries", []))
        else:
            out[field_name] = sorted(claim.fields.get("values", []))
    return out


def _norm_unsafe(seq):
    out = set()
    for entry in seq:
        out.add((entry["move"], tuple(sorted(entry.get("opponent_replies", [])))))
    return out


def evaluate_run(label, condition, sample_fn, boards, solver):
    n = len(boards)
    n_valid = 0
    n_schema_valid = 0
    n_legal = 0
    n_optimal = 0
    regret_sum = 0.0
    n_exact_match = 0
    field_stats = {f: defaultdict(int) for f in FIELDS}

    for i, row in enumerate(boards):
        env = _env_from_moves(row["moves"])
        completion = sample_fn(make_messages(env, condition=condition))
        parsed = parse_structured_response(completion, condition=condition)

        if parsed.valid_json:
            n_valid += 1
        if parsed.schema_valid:
            n_schema_valid += 1

        if parsed.chosen_move is not None and parsed.chosen_move in env.legal_moves():
            n_legal += 1
            ev = evaluate_move(env, parsed.chosen_move, solver)
            regret_sum += ev.clipped_regret
            if ev.is_optimal:
                n_optimal += 1
        else:
            regret_sum += 2.0  # match the trainer's invalid-move regret floor

        if condition == "tactical_claims":
            gt = ground_truth_tactical_claims(env)
            model_sets = _model_sets_from_parsed(parsed)
            all_match = True
            for f in FIELDS:
                gt_v = gt[f]
                mv = model_sets[f]
                gt_nonempty = bool(gt_v)
                mv_nonempty = (mv is not None) and bool(mv)
                if gt_nonempty:
                    field_stats[f]["gt_nonempty"] += 1
                if mv_nonempty:
                    field_stats[f]["model_nonempty"] += 1
                if f == "unsafe_moves":
                    exact = _norm_unsafe(gt_v) == _norm_unsafe(mv or [])
                else:
                    exact = sorted(gt_v) == (mv or [])
                if exact:
                    field_stats[f]["exact"] += 1
                else:
                    all_match = False
                if gt_nonempty and not mv_nonempty:
                    field_stats[f]["fn"] += 1
                if mv_nonempty and not gt_nonempty:
                    field_stats[f]["fp"] += 1
            if all_match:
                n_exact_match += 1

        if (i + 1) % 25 == 0:
            logging.info("[%s] %d/%d", label, i + 1, n)

    summary = {
        "label": label,
        "condition": condition,
        "n_boards": n,
        "valid_json_rate": n_valid / n if n else 0.0,
        "schema_valid_rate": n_schema_valid / n if n else 0.0,
        "legal_rate": n_legal / n if n else 0.0,
        "optimal_rate": n_optimal / n if n else 0.0,
        "mean_regret": regret_sum / n if n else 0.0,
    }
    if condition == "tactical_claims":
        per_field = {}
        for f in FIELDS:
            s = field_stats[f]
            gt_nz = s["gt_nonempty"]
            gt_z = n - gt_nz
            per_field[f] = {
                "gt_nonempty_rate": gt_nz / n if n else 0.0,
                "model_nonempty_rate": s["model_nonempty"] / n if n else 0.0,
                "exact_rate": s["exact"] / n if n else 0.0,
                "false_negative_rate_given_gt": (s["fn"] / gt_nz) if gt_nz else None,
                "false_positive_rate_given_gt_empty": (s["fp"] / gt_z) if gt_z else None,
            }
        summary["exact_match_rate"] = n_exact_match / n if n else 0.0
        summary["by_field"] = per_field
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="faithfulness/data/eval_boards.jsonl")
    parser.add_argument("--n-boards", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help='Triplet "<label>:<condition>:<checkpoint_or_BASE>" (repeatable).',
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.run:
        print("--run required at least once", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in Path(args.eval_set).read_text().splitlines() if l.strip()]
    import random

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    boards = rows[: args.n_boards]
    logging.info("Loaded %d boards", len(boards))

    solver = PonsSolver(strict=True)
    summaries = []
    for spec in args.run:
        label, condition, ckpt = spec.split(":", 2)
        logging.info("Evaluating %s (condition=%s checkpoint=%s)", label, condition, ckpt)
        sample_fn = _build_sample_fn(
            ckpt, args.base_model, args.base_url, args.max_tokens, args.temperature
        )
        summary = evaluate_run(label, condition, sample_fn, boards, solver)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    out = {"summaries": summaries, "n_boards": len(boards)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logging.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
