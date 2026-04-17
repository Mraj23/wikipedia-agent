"""Gradient flow verification for GRPO training.

Confirms that the GRPO loss produces non-zero gradients, the importance
ratio starts near 1.0, and KL divergence starts near 0.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.grpo_config import get_config
from spiral.grpo_trainer import GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("validate_gradients")


def main():
    print("=" * 60)
    print("GRADIENT FLOW VERIFICATION")
    print("=" * 60)
    print()

    model_path = "checkpoints/placeholder"
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; ...")
        sys.exit(1)

    # Use condition C (dense reward, simpler than D/E)
    config = get_config("C", model_path)
    config.game_steps = 1
    config.group_size = 2
    config.use_vllm = False
    config.use_wandb = False

    print("Initializing trainer (condition C, 1 step, group_size=2)...")
    trainer = GRPOTrainer(config=config, log_dir="logs/gradient_check")

    # Run one training step manually to inspect internals
    print("Running one training step...")
    envs = trainer.position_buffer.sample(batch_size=1)
    env = envs[0]

    from training.prompts import format_prompt
    prompt = format_prompt("C", env)

    completions, gen_log_probs = trainer._generate_group(prompt)
    print(f"  Generated {len(completions)} completions")
    for i, c in enumerate(completions):
        print(f"  Completion {i}: {c[:100]}...")

    rewards, reward_components = trainer._compute_rewards_batch(env, completions)
    print(f"  Rewards: {rewards}")
    print(f"  Components: {reward_components}")

    from spiral.rae import compute_advantages
    advantages = compute_advantages(
        rewards=rewards,
        reward_components=reward_components,
        reward_weights=config.reward_weights or {},
        use_rae=config.use_rae,
    )
    print(f"  Advantages: {advantages.tolist()}")

    # Run GRPO step
    loss, kl = trainer._grpo_step(prompt, completions, gen_log_probs, advantages)
    print(f"\n  Loss: {loss}")
    print(f"  KL: {kl}")

    # Check 1: Loss is finite
    print("\n--- Check 1: Loss finiteness ---")
    if not (abs(loss) < float("inf") and loss == loss):  # not inf, not nan
        print("  FAIL: Loss is NaN or Inf!")
    else:
        print(f"  PASS: Loss = {loss:.6f}")

    # Check 2: KL near zero at step 0
    print("\n--- Check 2: KL near zero at step 0 ---")
    if abs(kl) < 1.0:
        print(f"  PASS: KL = {kl:.6f} (< 1.0)")
    else:
        print(f"  WARNING: KL = {kl:.6f} (expected < 1.0 at step 0)")

    # Check 3: Non-zero gradients
    print("\n--- Check 3: Non-zero gradients ---")
    params_with_grad = 0
    total_params = 0
    max_grad_norm = 0.0
    for name, p in trainer.model.named_parameters():
        total_params += 1
        if p.grad is not None:
            grad_norm = p.grad.abs().max().item()
            if grad_norm > 0:
                params_with_grad += 1
                max_grad_norm = max(max_grad_norm, grad_norm)

    if params_with_grad > 0:
        print(f"  PASS: {params_with_grad}/{total_params} parameters have non-zero gradients")
        print(f"  Max gradient element: {max_grad_norm:.6e}")
    else:
        print(f"  FAIL: No parameters have non-zero gradients!")

    # Check 4: Importance ratio near 1.0 at step 0
    print("\n--- Check 4: Importance ratio at step 0 ---")
    prompt_ids = trainer.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )["input_ids"][0].to(trainer.device)

    ratios = []
    for i, completion in enumerate(completions):
        gen_ids = trainer.tokenizer(
            completion, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(trainer.device)
        if gen_ids.numel() == 0:
            continue

        with torch.no_grad():
            current_lp = trainer._compute_log_probs(prompt_ids, gen_ids)
        old_lp = gen_log_probs[i].detach()

        min_len = min(current_lp.shape[0], old_lp.shape[0])
        ratio = torch.exp(current_lp[:min_len] - old_lp[:min_len])
        ratios.append(ratio)
        print(f"  Completion {i}: mean_ratio={ratio.mean().item():.4f}, "
              f"max_ratio={ratio.max().item():.4f}, min_ratio={ratio.min().item():.4f}")

    if ratios:
        all_ratios = torch.cat(ratios)
        mean_r = all_ratios.mean().item()
        if 0.8 <= mean_r <= 1.2:
            print(f"  PASS: Mean ratio = {mean_r:.4f} (in [0.8, 1.2])")
        else:
            print(f"  WARNING: Mean ratio = {mean_r:.4f} (expected ~1.0 at step 0)")

    # Summary
    print("\n" + "=" * 60)
    checks_passed = sum([
        abs(loss) < float("inf") and loss == loss,
        abs(kl) < 1.0,
        params_with_grad > 0,
        len(ratios) > 0 and 0.8 <= mean_r <= 1.2,
    ])
    print(f"RESULTS: {checks_passed}/4 checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
