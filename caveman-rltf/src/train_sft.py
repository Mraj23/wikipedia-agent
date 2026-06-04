"""LoRA SFT on (prompt, completion) pairs via Tinker.

Same script for every SFT arm (datasets built by src/build_sft_dataset.py):
  - rltf_sft:     data/train/rltf_sft.jsonl    (correct + shorter y1)
  - sft_y1:       data/train/sft_y1.jsonl      (correct y1, no length gate)
  - sft_caveman:  data/train/sft_caveman.jsonl (model's own shortest correct y0)

This is upstream RLTF's *SFT* distillation mode, not RLTF-SD (see
../FAITHFULNESS.md). Pattern matches faithfulness_v2/train_move_only.py:
create LoRA training
client, build supervised datums via tinker_cookbook.supervised.data with
TrainOnWhat.LAST_ASSISTANT_MESSAGE so loss is masked to assistant tokens
only, then forward_backward + optim_step. Final weights are saved with
save_weights_for_sampler_async; the resulting `tinker://...` path goes
into manifest.json so src/eval.py can pick it up.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from _tinker import resolve_async


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


async def amain(args):
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.supervised.data import (
        TrainOnWhat,
        conversation_to_datum,
    )

    service_client = tinker.ServiceClient()
    training_client = await resolve_async(
        service_client.create_lora_training_client_async(
            base_model=args.model, rank=args.lora_rank
        )
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer, model_name=args.model)

    rows = load_jsonl(args.train)
    print(f"loaded {len(rows)} training rows from {args.train}")

    datums = []
    skipped = 0
    for r in rows:
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]
        try:
            datum = conversation_to_datum(
                messages,
                renderer,
                max_length=args.max_seq_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
        except Exception as e:
            skipped += 1
            continue
        datums.append(datum)
    if skipped:
        print(f"  skipped {skipped} rows that failed datum construction")

    n = len(datums)
    bs = args.batch_size
    steps_per_epoch = n // bs
    total_steps = steps_per_epoch * args.epochs
    if total_steps == 0:
        raise SystemExit(
            f"not enough data for batch_size={bs} (have {n} datums, "
            f"need >= {bs})"
        )

    adam = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
    )

    print(
        f"training: {args.epochs} epochs x {steps_per_epoch} steps/epoch "
        f"= {total_steps} total steps, batch_size={bs}, lr={args.learning_rate}"
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    losses = []
    intermediate_samplers = []
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        idxs = list(range(n))
        rng.shuffle(idxs)
        for start in range(0, n - bs + 1, bs):
            batch = [datums[idxs[i]] for i in range(start, start + bs)]
            fwdbwd_future = await resolve_async(
                training_client.forward_backward_async(
                    batch, loss_fn="cross_entropy"
                )
            )
            optim_future = await resolve_async(
                training_client.optim_step_async(adam)
            )
            loss = getattr(fwdbwd_future, "loss", None)
            losses.append(float(loss) if loss is not None else None)
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(f"  step {step}/{total_steps} loss={loss}")

            if (
                args.save_sampler_every_steps
                and step % args.save_sampler_every_steps == 0
                and step < total_steps
            ):
                save = await resolve_async(
                    training_client.save_weights_for_sampler_async(
                        name=f"{args.run_name}-step{step}",
                        ttl_seconds=3600 * 24,
                    )
                )
                intermediate_samplers.append({"step": step, "path": save.path})
                print(f"  step {step}: saved sampler at {save.path}")

    final = await resolve_async(
        training_client.save_weights_for_sampler_async(
            name=f"{args.run_name}-final", ttl_seconds=3600 * 24 * 7
        )
    )

    manifest = {
        "run_name": args.run_name,
        "base_model": args.model,
        "renderer": args.renderer,
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n_train_rows": len(rows),
        "n_train_datums": n,
        "total_steps": total_steps,
        "train_file": args.train,
        "final_sampler_path": final.path,
        "intermediate_samplers": intermediate_samplers,
        "loss_curve": losses,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"final sampler path: {final.path}")
    print(f"manifest -> {out_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--renderer", default="qwen2_5_instruct")
    parser.add_argument("--train", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="dir for manifest.json with the tinker:// final sampler path",
    )
    parser.add_argument("--run-name", default="rltf-sd")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-sampler-every-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
