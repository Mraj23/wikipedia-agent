"""Pre-flight: measure base-model schema-validity rate under a given condition.

Cheap check before any RL spend. Samples N boards from the eval set, generates
one completion per board with the chosen prompt condition, parses with the
matching condition, and reports the fraction of completions that pass strict
schema validation.

For tactical_claims this answers: can base Qwen3-4B-Instruct-2507 produce the
required exhaustive tactical schema often enough that GRPO has any non-penalty
gradient signal to learn from in the first hundred steps?

Usage:
    TINKER_API_KEY=... /tmp/tinker_venv/bin/python -m \
        faithfulness.scripts.preflight_schema_validity \
        --condition tactical_claims --n-boards 50 \
        --output faithfulness/data/runs/preflight_tactical.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from env.connect_four_env import ConnectFourEnv
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import make_messages


def _resolve_tinker_value(value):
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


def _build_tinker_sample_fn(base_model: str, base_url, max_tokens: int, temperature: float):
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer, get_text_content

    service_kwargs = {}
    if base_url is not None:
        service_kwargs["base_url"] = base_url
    service_client = tinker.ServiceClient(**service_kwargs)
    # Tinker requires a "tinker://"-style path for create_sampling_client; the
    # cheapest way to sample the *base* model is to spin up a LoRA training
    # client and immediately materialize a sampling checkpoint at zero training
    # steps. The resulting weights equal the base model.
    training_client = _resolve_tinker_value(
        service_client.create_lora_training_client(base_model=base_model, rank=1)
    )
    save_result = _resolve_tinker_value(
        training_client.save_weights_for_sampler(
            name="preflight-base", ttl_seconds=3600
        )
    )
    sampling_client = _resolve_tinker_value(
        service_client.create_sampling_client(model_path=save_result.path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer, model_name=base_model)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    def sample_fn(messages, num_samples=1):
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


def _env_from_moves(moves):
    env = ConnectFourEnv()
    for m in moves:
        env.make_move(int(m))
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--condition",
        choices=("claims_rationale", "move_only", "tactical_claims"),
        default="tactical_claims",
    )
    parser.add_argument(
        "--eval-set", default="faithfulness/data/eval_boards.jsonl"
    )
    parser.add_argument("--n-boards", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--records-output",
        default=None,
        help="Optional JSONL with raw completions and per-board parse outcome.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    boards_path = Path(args.eval_set)
    if not boards_path.exists():
        print(f"eval set not found: {boards_path}", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in boards_path.read_text().splitlines() if line.strip()]
    import random

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_boards]
    logging.info("Loaded %d boards (cap %d)", len(rows), args.n_boards)

    sample_fn = _build_tinker_sample_fn(
        args.base_model, args.base_url, args.max_tokens, args.temperature
    )

    n_total = 0
    n_valid_json = 0
    n_schema_valid = 0
    n_legal_move = 0
    parse_error_counter: Counter[str] = Counter()
    records = []

    for i, row in enumerate(rows):
        moves = row.get("moves", [])
        env = _env_from_moves(moves)
        messages = make_messages(env, condition=args.condition)
        completions = sample_fn(messages, num_samples=1)
        completion = completions[0] if completions else ""
        parsed = parse_structured_response(completion, condition=args.condition)

        n_total += 1
        if parsed.valid_json:
            n_valid_json += 1
        if parsed.schema_valid:
            n_schema_valid += 1
        if parsed.chosen_move is not None and parsed.chosen_move in env.legal_moves():
            n_legal_move += 1

        err = parsed.parse_error or ("" if parsed.schema_valid else "schema_invalid")
        parse_error_counter[err] += 1

        records.append(
            {
                "moves": moves,
                "completion": completion,
                "valid_json": parsed.valid_json,
                "schema_valid": parsed.schema_valid,
                "chosen_move": parsed.chosen_move,
                "parse_error": parsed.parse_error,
                "n_claims": len(parsed.claims),
            }
        )

        if (i + 1) % 10 == 0:
            logging.info(
                "%d/%d  valid_json=%d  schema_valid=%d  legal_move=%d",
                i + 1,
                n_total,
                n_valid_json,
                n_schema_valid,
                n_legal_move,
            )

    summary = {
        "condition": args.condition,
        "base_model": args.base_model,
        "n_boards": n_total,
        "valid_json_rate": n_valid_json / n_total if n_total else 0.0,
        "schema_valid_rate": n_schema_valid / n_total if n_total else 0.0,
        "legal_move_rate": n_legal_move / n_total if n_total else 0.0,
        "parse_error_breakdown": dict(parse_error_counter),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote summary to %s", out)
    print(json.dumps(summary, indent=2))

    if args.records_output:
        rec_path = Path(args.records_output)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with rec_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logging.info("Wrote %d records to %s", len(records), rec_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
