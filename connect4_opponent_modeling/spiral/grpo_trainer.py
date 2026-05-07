"""Core GRPO training loop for Connect Four RL conditions B-E.

Implements Group Relative Policy Optimization with optional Role-conditioned
Advantage Estimation (RAE/SPIRAL). Each training step:
  1. Samples a board position
  2. Generates group_size completions
  3. Computes condition-specific rewards
  4. Computes advantages (standard or RAE)
  5. Applies clipped surrogate loss + KL penalty
"""

import json
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.grpo_config import GRPOConfig
from training.minimax import MinimaxSolver
from training.prompts import format_prompt, parse_response, validate_response
from training.reward import RewardCalculator

from spiral.eval_callback import EvalCallback
from spiral.game_playout import play_to_completion
from spiral.position_buffer import PositionBuffer
from spiral.rae import compute_advantages

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO trainer for Connect Four RL conditions."""

    def __init__(self, config: GRPOConfig, log_dir: str) -> None:
        """Initialize the trainer.

        Args:
            config: GRPO configuration with condition-specific settings.
            log_dir: Directory for training logs.
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Seed for reproducibility
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Device setup (CPU-compatible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use bf16 on GPU (better compatibility with GH200/H100), fp32 on CPU
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info("Loading model from %s (device=%s)", config.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Check if tokenizer supports chat template (Qwen3, etc.)
        self._has_chat_template = hasattr(self.tokenizer, "apply_chat_template")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path, trust_remote_code=True, dtype=self.dtype
        ).to(self.device)

        # Enable gradient checkpointing to reduce activation memory during backward.
        # This sets use_cache=False on the model config, but we override it
        # during generation by calling model.generate(use_cache=True).
        if self.device.type == "cuda":
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (for backward pass only).")

        # Frozen reference model for KL computation
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Use 8-bit AdamW on GPU to reduce optimizer memory from 32GB to 8GB
        if self.device.type == "cuda":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=config.lr)
                logger.info("Using 8-bit AdamW (bitsandbytes).")
            except ImportError:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
                logger.warning("bitsandbytes not installed, using standard AdamW.")
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        # vLLM generator for fast generation (GPU only).
        # Uses CPU offloading: training model goes to CPU during generation,
        # vLLM gets full GPU. After generation, vLLM is destroyed and
        # training model moves back to GPU.
        self._vllm_gen = None
        if config.use_vllm and self.device.type == "cuda":
            try:
                from spiral.vllm_generator import VLLMGenerator
                vllm_dtype = "bfloat16" if self.dtype == torch.bfloat16 else "float16"
                self._vllm_gen = VLLMGenerator(config.model_path, dtype=vllm_dtype)
                logger.info("vLLM generator initialized (CPU offload mode).")
            except ImportError:
                logger.warning("vLLM not installed. Using HF generate() (slower).")
            except Exception as e:
                logger.warning("vLLM init failed: %s. Using HF generate().", e)

        # Game infrastructure
        require_pons = os.environ.get("C4_REQUIRE_PONS_SOLVER", "1") != "0"
        self.solver = PonsSolver(fallback_depth=config.opponent_depth, strict=require_pons)
        if require_pons and not self.solver.is_available():
            raise RuntimeError(
                "Pons solver unavailable. Training is intentionally blocked because "
                "minimax fallback is too slow for the active experiment. "
                "Run scripts/bootstrap_gpu.sh or set C4_REQUIRE_PONS_SOLVER=0 to override."
            )
        self.reward_calc = RewardCalculator(self.solver)
        self.minimax = MinimaxSolver(depth=config.opponent_depth)

        # Position buffer — load from disk if available, else generate
        buffer_path = Path("data/position_buffer.json")
        if buffer_path.exists():
            logger.info("Loading position buffer from %s...", buffer_path)
            self.position_buffer = PositionBuffer.load(str(buffer_path))
        else:
            logger.info("Building position buffer (no cached file at %s)...", buffer_path)
            self.position_buffer = PositionBuffer(
                pool_size=min(10000, max(50, config.game_steps * 2)),
                min_moves_remaining=6 if config.condition == "B" else 2,
            )
            # Save for next run
            self.position_buffer.save(str(buffer_path))
            logger.info("Saved position buffer to %s", buffer_path)
        logger.info("Position buffer ready (%d positions)", len(self.position_buffer))

        # Eval callback
        self.checkpoint_dir = Path(
            config.checkpoint_dir or f"checkpoints/condition_{config.condition.lower()}"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_callback = EvalCallback(
            model_fn_factory=self._make_model_fn,
            condition=config.condition,
            log_dir=str(self.log_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            save_fn=self._save_checkpoint,
        )

        # Training state
        self._step = 0
        self._train_log: List[Dict] = []

        # W&B logging
        self._wandb_run = None
        if config.use_wandb and _WANDB_AVAILABLE:
            run_name = config.wandb_run_name or f"condition_{config.condition.lower()}"
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=vars(config),
                tags=[f"condition_{config.condition}", "grpo", "connect4"],
                reinit=True,
            )
            self._wandb_run.define_metric("train/*", step_metric="trainer_step")
            self._wandb_run.define_metric("reward/*", step_metric="trainer_step")
            self._wandb_run.define_metric("eval/*", step_metric="trainer_step")
            self._wandb_run.define_metric("final/*", step_metric="trainer_step")
            logger.info("W&B run initialized: %s", self._wandb_run.url)
        elif config.use_wandb and not _WANDB_AVAILABLE:
            logger.warning("use_wandb=True but wandb not installed. pip install wandb")

        # Dynamic temperature — increases when move diversity collapses.
        # Start slightly lower to improve format validity early in training.
        self._temperature = 0.5

    def train(self) -> None:
        """Run the full GRPO training loop."""
        logger.info(
            "Starting GRPO training: condition=%s, steps=%d, group_size=%d",
            self.config.condition, self.config.game_steps, self.config.group_size,
        )

        # Save config
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)

        for step in range(self.config.game_steps):
            self._step = step
            step_start = time.time()

            # 1. Sample a position
            envs = self.position_buffer.sample(batch_size=1)
            env = envs[0]

            # 2. Format prompt for this condition
            prompt = format_prompt(self.config.condition, env)

            # 3. Generate group of completions
            completions, gen_log_probs = self._generate_group(prompt)

            # 4. Compute rewards and per-component breakdowns
            rewards, reward_components = self._compute_rewards_batch(
                env, completions
            )

            # 4b. Move diversity + validity + thinking length monitoring
            from collections import Counter
            import re as _re
            moves_played = []
            thinking_token_counts = []
            valid_count = 0
            for resp in completions:
                parsed = parse_response(resp, self.config.condition)
                is_valid, _ = validate_response(parsed, self.config.condition, env.legal_moves())
                move = parsed.get("move")
                moves_played.append(move)
                if is_valid:
                    valid_count += 1
                # Count tokens before <answer> (approximate thinking length)
                answer_pos = resp.find("<answer>")
                if answer_pos > 0:
                    thinking_text = resp[:answer_pos]
                    thinking_tokens = len(self.tokenizer.encode(thinking_text))
                    thinking_token_counts.append(thinking_tokens)
                elif len(resp) > 0:
                    # No answer tag — all tokens were thinking (truncated)
                    thinking_token_counts.append(self.config.max_tokens)
            valid_moves = [m for m in moves_played if m is not None]
            move_counts = Counter(valid_moves)
            unique_moves = len(move_counts)
            valid_pct = valid_count / max(len(completions), 1)
            invalid_pct = 1.0 - valid_pct
            avg_thinking_tokens = sum(thinking_token_counts) / max(len(thinking_token_counts), 1)
            zero_reward_count = sum(1 for r in rewards if r == 0.0)
            zero_reward_pct = zero_reward_count / max(len(rewards), 1)
            if move_counts:
                most_common_col, most_common_count = move_counts.most_common(1)[0]
                most_common_pct = most_common_count / max(len(completions), 1)
            else:
                most_common_col, most_common_pct = -1, 0.0

            # Collapse detection: if >80% play the same column, increase temperature
            if most_common_pct > 0.8:
                self._temperature = min(1.5, self._temperature + 0.1)
                logger.warning(
                    "Step %d: Move collapse — %d/%d play col %d. Temp -> %.2f",
                    step, most_common_count, len(completions),
                    most_common_col, self._temperature,
                )
            else:
                self._temperature = max(0.5, self._temperature - 0.02)

            # 5. Compute advantages
            advantages = compute_advantages(
                rewards=rewards,
                reward_components=reward_components,
                reward_weights=self.config.reward_weights or {},
                use_rae=self.config.use_rae,
            )

            # 6. Compute loss and update
            loss, kl = self._grpo_step(prompt, completions, gen_log_probs, advantages)

            # 7. Sync weights to vLLM engine if using it
            vllm_synced = self._sync_vllm_weights()

            step_time = time.time() - step_start

            # 8. Logging
            log_entry = {
                "step": step,
                "loss": loss,
                "kl": kl,
                "mean_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "unique_moves": unique_moves,
                "most_common_move": most_common_col,
                "most_common_pct": most_common_pct,
                "valid_completions": valid_count,
                "valid_pct": valid_pct,
                "invalid_pct": invalid_pct,
                "avg_thinking_tokens": avg_thinking_tokens,
                "temperature": self._temperature,
                "step_time_s": step_time,
                "zero_reward_pct": zero_reward_pct,
                "vllm_synced": float(vllm_synced),
            }
            self._train_log.append(log_entry)

            log_every = min(10, max(1, self.config.game_steps // 30))
            if step % log_every == 0:
                logger.info(
                    "Step %d/%d: loss=%.4f kl=%.4f reward=%.3f [%.3f, %.3f] "
                    "valid=%d/%d (%d%%) think=%.0f moves=%d temp=%.2f (%.1fs)",
                    step, self.config.game_steps, loss, kl,
                    log_entry["mean_reward"], log_entry["min_reward"],
                    log_entry["max_reward"], valid_count, len(completions),
                    valid_pct * 100, avg_thinking_tokens, unique_moves,
                    self._temperature, step_time,
                )

            # Log sample completion every step (truncated)
            if completions:
                # Find a valid completion to show
                sample = completions[0][:300]
                for resp in completions:
                    if "<answer>" in resp:
                        sample = resp[:300]
                        break
                logger.info("Step %d sample: %s", step, sample)

            # W&B per-step logging
            if self._wandb_run is not None:
                wandb_log = {
                    "trainer_step": step,
                    "train/loss": loss,
                    "train/kl": kl,
                    "train/reward_mean": log_entry["mean_reward"],
                    "train/reward_max": log_entry["max_reward"],
                    "train/reward_min": log_entry["min_reward"],
                    "train/step_time_s": step_time,
                    "train/unique_moves": unique_moves,
                    "train/most_common_pct": most_common_pct,
                    "train/most_common_move": most_common_col,
                    "train/valid_completions": valid_count,
                    "train/valid_pct": valid_pct,
                    "train/invalid_pct": invalid_pct,
                    "train/avg_thinking_tokens": avg_thinking_tokens,
                    "train/temperature": self._temperature,
                    "train/zero_reward_pct": zero_reward_pct,
                    "train/vllm_synced": log_entry["vllm_synced"],
                    "train/mode_collapse_flag": float(most_common_pct > 0.8),
                }
                # Log per-component reward means across the group
                if reward_components and reward_components[0]:
                    for comp_name in reward_components[0]:
                        comp_vals = [rc.get(comp_name, 0.0) for rc in reward_components]
                        wandb_log[f"reward/{comp_name}_mean"] = sum(comp_vals) / len(comp_vals)
                        if len(comp_vals) > 1:
                            wandb_log[f"reward/{comp_name}_std"] = (
                                torch.tensor(comp_vals, dtype=torch.float32).std().item()
                            )
                # Log advantage stats
                if advantages is not None and advantages.numel() > 0:
                    wandb_log["train/advantage_mean"] = advantages.mean().item()
                    wandb_log["train/advantage_std"] = advantages.std().item()
                if step == 0 or step % 25 == 0:
                    sample_response = ""
                    sample_valid = 0.0
                    sample_move = -1
                    for resp in completions:
                        parsed = parse_response(resp, self.config.condition)
                        is_valid, _ = validate_response(
                            parsed, self.config.condition, env.legal_moves()
                        )
                        if is_valid:
                            sample_response = resp[:500]
                            sample_valid = 1.0
                            sample_move = parsed.get("move", -1)
                            break
                    if not sample_response and completions:
                        sample_response = completions[0][:500]
                    wandb_log["samples/sample_completion"] = sample_response
                    wandb_log["samples/sample_valid"] = sample_valid
                    wandb_log["samples/sample_move"] = sample_move
                self._wandb_run.log(wandb_log, step=step)

            # 8. Divergence detection
            if math.isnan(loss) or math.isinf(loss):
                logger.error("Step %d: Loss is NaN/Inf! Stopping training.", step)
                self._save_train_log()
                break

            if kl > 10.0:
                logger.warning(
                    "Step %d: KL divergence is %.2f (threshold=10.0). "
                    "Policy may be diverging from reference.", step, kl,
                )

            # Check for reward collapse (all zero for 20+ consecutive steps)
            if len(self._train_log) >= 20:
                recent_rewards = [e["max_reward"] for e in self._train_log[-20:]]
                if all(r == 0.0 for r in recent_rewards):
                    logger.error(
                        "Step %d: Reward collapse — all rewards zero for 20 "
                        "consecutive steps. Model may have degenerated.", step,
                    )

            # 9. Periodic eval
            if step > 0 and step % self.config.eval_every == 0:
                eval_results = self.eval_callback.run(step)
                if self._wandb_run is not None:
                    wandb_eval = {"trainer_step": step}
                    pons = eval_results.get("pons_benchmark", {})
                    if "overall_pct_optimal" in pons:
                        wandb_eval["eval/pons_pct_optimal"] = pons["overall_pct_optimal"]
                        wandb_eval["eval/pons_invalid_output_rate"] = pons.get(
                            "invalid_output_rate", 0.0
                        )
                        for depth, stats in pons.get("win_rate_vs_minimax", {}).items():
                            wandb_eval[f"eval/minimax_{depth}_win_rate"] = stats.get("win_rate", 0.0)
                            wandb_eval[f"eval/minimax_{depth}_invalid_game_rate"] = stats.get(
                                "invalid_game_rate", 0.0
                            )
                    probe = eval_results.get("probe", {})
                    if "overall_accuracy" in probe:
                        wandb_eval["eval/probe_accuracy"] = probe["overall_accuracy"]
                        for depth_name, acc in probe.get("by_depth", {}).items():
                            wandb_eval[f"eval/probe_{depth_name}_accuracy"] = acc
                    if wandb_eval:
                        self._wandb_run.log(wandb_eval, step=step)

            # 10. Periodic log save
            if step % 500 == 0 and step > 0:
                self._save_train_log()

        # Final eval and checkpoint
        logger.info("Training complete. Running final eval...")
        final_eval = self.eval_callback.run(self.config.game_steps)
        self._save_checkpoint(str(self.checkpoint_dir / "final"))
        self._save_train_log()

        # W&B final logging and cleanup
        if self._wandb_run is not None:
            wandb_final = {
                "trainer_step": self.config.game_steps,
                "final/total_steps": self.config.game_steps,
            }
            pons = final_eval.get("pons_benchmark", {})
            if "overall_pct_optimal" in pons:
                wandb_final["final/pons_pct_optimal"] = pons["overall_pct_optimal"]
                wandb_final["final/pons_invalid_output_rate"] = pons.get(
                    "invalid_output_rate", 0.0
                )
            probe = final_eval.get("probe", {})
            if "overall_accuracy" in probe:
                wandb_final["final/probe_accuracy"] = probe["overall_accuracy"]
            self._wandb_run.log(wandb_final, step=self.config.game_steps)
            # Save training log as artifact
            artifact = wandb.Artifact(
                f"train_log_{self.config.condition.lower()}", type="training_log"
            )
            artifact.add_file(str(self.log_dir / "train_log.json"))
            for eval_file in sorted(self.log_dir.glob("eval_step_*.json")):
                artifact.add_file(str(eval_file))
            config_path = self.log_dir / "config.json"
            if config_path.exists():
                artifact.add_file(str(config_path))
            self._wandb_run.log_artifact(artifact)
            self._wandb_run.finish()
    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to raw prompt for instruct models.

        This is critical for /no_think to work — without the chat template,
        /no_think is just literal text and the model ignores it.
        """
        if self._has_chat_template:
            try:
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return prompt

    def _generate_group(
        self, prompt: str
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate group_size completions for a prompt.

        Uses vLLM if available (much faster on GPU), falls back to HF generate().

        Args:
            prompt: The formatted prompt string (raw, before chat template).

        Returns:
            Tuple of (decoded_texts, per_completion_log_probs).
        """
        if self._vllm_gen is not None:
            return self._generate_group_vllm(prompt)
        return self._generate_group_hf(prompt)

    def _generate_group_vllm(
        self, prompt: str
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate completions using vLLM with CPU offloading.

        Flow:
        1. Move training model + ref model to CPU
        2. vLLM generates on GPU (full memory)
        3. Destroy vLLM engine
        4. Move models back to GPU
        5. Compute log-probs under training model
        """
        import time
        gen_start = time.time()

        # 1. Offload training and ref models to CPU to free GPU for vLLM
        self.model.to("cpu")
        self.ref_model.to("cpu")
        torch.cuda.empty_cache()

        # Apply chat template for /no_think to work
        chat_prompt = self._apply_chat_template(prompt)

        # 2. Generate with vLLM (full GPU)
        completions = self._vllm_gen.generate(
            prompt=chat_prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_tokens,
            min_tokens=10,
            temperature=self._temperature,
        )

        # 3. Destroy vLLM to free GPU
        self._vllm_gen._destroy_engine()

        # 4. Move models back to GPU
        self.model.to(self.device)
        self.ref_model.to(self.device)

        gen_time = time.time() - gen_start
        logger.debug("vLLM generation: %d completions in %.1fs", len(completions), gen_time)

        # 5. Compute log-probs under training model (on GPU)
        log_probs_list = []
        prompt_ids = self.tokenizer(
            chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"][0].to(self.device)

        for text in completions:
            gen_ids = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(self.device)
            if gen_ids.numel() > 0:
                lp = self._compute_log_probs(prompt_ids, gen_ids)
            else:
                lp = torch.zeros(1, device=self.device)
            log_probs_list.append(lp)

        return completions, log_probs_list

    def _generate_group_hf(
        self, prompt: str
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate completions using HuggingFace model.generate()."""
        self.model.eval()

        # Disable gradient checkpointing for generation so KV cache works.
        # Re-enable after generation for the backward pass.
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True

        # Apply chat template so /no_think is processed correctly
        chat_prompt = self._apply_chat_template(prompt)
        inputs = self.tokenizer(
            chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        completions = []
        log_probs_list = []

        # Generate in mini-batches to manage memory
        remaining = self.config.group_size
        while remaining > 0:
            batch = min(remaining, 4)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    min_new_tokens=10,
                    num_return_sequences=batch,
                    do_sample=True,
                    temperature=self._temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for seq in outputs:
                gen_tokens = seq[prompt_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                completions.append(text)
                lp = self._compute_log_probs(inputs["input_ids"][0], gen_tokens)
                log_probs_list.append(lp)

            remaining -= batch

        # Re-enable gradient checkpointing for backward pass
        if self.device.type == "cuda" and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.model.train()
        return completions, log_probs_list

    def _sync_vllm_weights(self) -> bool:
        """Save updated weights so vLLM loads them on next generation."""
        if self._vllm_gen is None:
            return False
        sync_every = max(1, int(getattr(self.config, "vllm_sync_every", 1)))
        should_sync = (
            self._step == 0
            or (self._step + 1) % sync_every == 0
            or self._step == self.config.game_steps - 1
        )
        if not should_sync:
            logger.debug(
                "Skipping vLLM weight sync at step %d (sync_every=%d).",
                self._step,
                sync_every,
            )
            return False
        self._vllm_gen.update_weights(self.model, self.tokenizer, self._step)
        return True

    def _compute_log_probs(
        self, prompt_ids: torch.Tensor, gen_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for generated tokens.

        Args:
            prompt_ids: Prompt token IDs, shape (prompt_len,).
            gen_ids: Generated token IDs, shape (gen_len,).

        Returns:
            Log-probs tensor, shape (gen_len,).
        """
        full_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(full_ids).logits[0]  # (seq_len, vocab)

        # Log-probs at positions [prompt_len-1 ... prompt_len+gen_len-2]
        # predict tokens at [prompt_len ... prompt_len+gen_len-1]
        prompt_len = prompt_ids.shape[0]
        gen_len = gen_ids.shape[0]
        relevant_logits = logits[prompt_len - 1 : prompt_len + gen_len - 1]
        log_probs = F.log_softmax(relevant_logits, dim=-1)

        # Gather the log-prob of each actual generated token
        token_log_probs = log_probs.gather(
            1, gen_ids.unsqueeze(1).to(self.device)
        ).squeeze(1)
        return token_log_probs

    def _compute_ref_log_probs(
        self, prompt_ids: torch.Tensor, gen_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log-probs under the reference model.

        Args:
            prompt_ids: Prompt token IDs, shape (prompt_len,).
            gen_ids: Generated token IDs, shape (gen_len,).

        Returns:
            Log-probs tensor, shape (gen_len,).
        """
        full_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.ref_model(full_ids).logits[0]

        prompt_len = prompt_ids.shape[0]
        gen_len = gen_ids.shape[0]
        relevant_logits = logits[prompt_len - 1 : prompt_len + gen_len - 1]
        log_probs = F.log_softmax(relevant_logits, dim=-1)
        token_log_probs = log_probs.gather(
            1, gen_ids.unsqueeze(1).to(self.device)
        ).squeeze(1)
        return token_log_probs

    def _compute_rewards_batch(
        self, env: ConnectFourEnv, completions: List[str]
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Compute rewards for all completions in a group.

        Caches solver analysis since all completions share the same position.

        Args:
            env: The board position (BEFORE any move).
            completions: List of model response strings.

        Returns:
            Tuple of (total_rewards, per_completion_component_dicts).
        """
        rewards = []
        components = []
        condition = self.config.condition

        for response in completions:
            # With /no_think + <reasoning>/<answer> format, the model's
            # response contains the tags directly. No prepending needed.
            r, comp = self._compute_single_reward(env, response)
            rewards.append(r)
            components.append(comp)

        return rewards, components

    def _compute_single_reward(
        self, env: ConnectFourEnv, response: str
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward for a single completion.

        Args:
            env: Board position BEFORE the move.
            response: Model response string.

        Returns:
            Tuple of (total_reward, component_dict).
        """
        condition = self.config.condition
        parsed = parse_response(response, condition)
        is_valid, _ = validate_response(parsed, condition, env.legal_moves())

        if not is_valid or parsed["move"] is None:
            if condition == "B":
                return 0.0, {}
            return 0.0, {k: 0.0 for k in (self.config.reward_weights or {})}

        played_col = parsed["move"]

        if condition == "B":
            game_result = play_to_completion(
                env, played_col, self.minimax, env.current_player()
            )
            reward = self.reward_calc.condition_b_reward(game_result)
            return reward, {}

        # Determine game_result for C/D/E
        test_env = env.copy()
        test_env.make_move(played_col)
        if test_env.is_terminal():
            winner = test_env.winner()
            if winner == env.current_player():
                game_result = "win"
            elif winner is None:
                game_result = "draw"
            else:
                game_result = "loss"
        else:
            game_result = "ongoing"

        # Compute per-component rewards for RAE
        # Format is a binary gate (handled above), not a weighted component.
        move_quality = self.solver.normalize_reward(env, played_col)
        terminal = RewardCalculator._terminal_reward(game_result)

        if condition == "C":
            reward = self.reward_calc.condition_c_reward(
                env, played_col, game_result, response
            )
            comp = {"move": move_quality, "terminal": terminal}

        elif condition == "Value":
            reward = move_quality
            comp = {"move": move_quality}

        elif condition == "D":
            reward = self.reward_calc.condition_d_reward(
                env, played_col, game_result, response
            )
            future_state = parsed.get("future_state", "")
            fs_acc = self.reward_calc._future_state_accuracy(env, played_col, future_state)
            comp = {
                "move": move_quality, "future": fs_acc,
                "terminal": terminal,
            }
            # D and E now share the same required output schema. D scores only
            # the future-state auxiliary field; the opponent prediction is
            # collected but left unscored in this condition.

        elif condition == "E":
            predicted_opp = parsed.get("opponent_prediction")
            if predicted_opp is None:
                predicted_opp = -1
            reward = self.reward_calc.condition_e_reward(
                env, played_col, predicted_opp, game_result, response
            )
            pred_acc = self.reward_calc._prediction_accuracy(env, played_col, predicted_opp)
            comp = {
                "move": move_quality, "pred": pred_acc,
                "terminal": terminal,
            }

        elif condition == "OpponentNextMove":
            predicted_opp = parsed.get("opponent_prediction")
            if predicted_opp is None:
                predicted_opp = -1
            pred_acc = self.reward_calc._prediction_accuracy(env, played_col, predicted_opp)
            weights = self.config.reward_weights or {"move": 0.8, "pred": 0.2}
            comp = {"move": move_quality, "pred": pred_acc}
            reward = sum(weights.get(name, 0.0) * value for name, value in comp.items())

        elif condition == "G":
            predicted_count = parsed.get("piece_count")
            if predicted_count is None:
                predicted_count = -1
            reward = self.reward_calc.condition_g_reward(
                env, played_col, predicted_count, game_result, response
            )
            count_acc = RewardCalculator._piece_count_accuracy(env, predicted_count)
            comp = {
                "move": move_quality, "count": count_acc,
                "terminal": terminal,
            }
        else:
            reward = 0.0
            comp = {}

        return reward, comp

    def _grpo_step(
        self,
        prompt: str,
        completions: List[str],
        gen_log_probs: List[torch.Tensor],
        advantages: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute and apply the GRPO loss.

        Args:
            prompt: The prompt string.
            completions: List of completion strings.
            gen_log_probs: Log-probs from generation time (old policy).
            advantages: Per-completion advantage values.

        Returns:
            Tuple of (loss_value, mean_kl).
        """
        self.model.train()
        chat_prompt = self._apply_chat_template(prompt)
        prompt_ids = self.tokenizer(
            chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"][0].to(self.device)

        valid_items = []
        for i, completion in enumerate(completions):
            gen_ids = self.tokenizer(
                completion, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(self.device)
            if gen_ids.numel() > 0:
                valid_items.append((i, gen_ids))

        if not valid_items:
            return 0.0, 0.0

        self.optimizer.zero_grad()
        total_loss_value = 0.0
        total_kl = 0.0
        n_valid = len(valid_items)

        for i, gen_ids in valid_items:
            # Current policy log-probs (with grad)
            full_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)
            logits = self.model(full_ids).logits[0]
            prompt_len = prompt_ids.shape[0]
            gen_len = gen_ids.shape[0]
            relevant_logits = logits[prompt_len - 1 : prompt_len + gen_len - 1]
            current_log_probs = F.log_softmax(relevant_logits, dim=-1)
            current_token_lp = current_log_probs.gather(
                1, gen_ids.unsqueeze(1)
            ).squeeze(1)

            # Old policy log-probs (from generation, detached)
            old_token_lp = gen_log_probs[i].detach().to(self.device)

            # Truncate to matching length
            min_len = min(current_token_lp.shape[0], old_token_lp.shape[0])
            current_token_lp = current_token_lp[:min_len]
            old_token_lp = old_token_lp[:min_len]

            # Reference model log-probs (for KL)
            ref_token_lp = self._compute_ref_log_probs(prompt_ids, gen_ids[:min_len])

            # Per-token importance ratio
            ratio = torch.exp(current_token_lp - old_token_lp)

            # Advantage for this completion
            adv = advantages[i].to(self.device)

            # Clipped surrogate
            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
            ) * adv
            surrogate = torch.min(surr1, surr2).mean()

            # KL penalty
            kl = (current_token_lp - ref_token_lp).mean()
            total_kl += kl.item()

            # Entropy bonus — prevents mode collapse by rewarding diverse outputs.
            # When all completions converge to the same tokens, entropy drops
            # and this term provides gradient to push toward more diversity.
            probs = torch.exp(current_log_probs[:min_len])
            token_entropy = -torch.sum(probs * current_log_probs[:min_len], dim=-1)
            entropy = token_entropy.mean()

            # Loss = -surrogate + kl_coef * kl - entropy_coef * entropy
            completion_loss = (
                -surrogate
                + self.config.kl_coef * kl
                - self.config.entropy_coef * entropy
            )
            total_loss_value += completion_loss.detach().item()
            (completion_loss / n_valid).backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss_value / n_valid, total_kl / n_valid

    def _make_model_fn(self) -> Callable[[str], str]:
        """Create a model inference callable for evaluation.

        Returns:
            Function that takes a prompt and returns model output.
        """
        self.model.eval()
        device = self.device
        tokenizer = self.tokenizer
        model = self.model

        def model_fn(prompt: str) -> str:
            checkpointing_was_disabled = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
                checkpointing_was_disabled = True
            if hasattr(model, "config"):
                model.config.use_cache = True

            rendered = self._apply_chat_template(prompt)
            inputs = tokenizer(
                rendered, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            finally:
                if self.device.type == "cuda" and checkpointing_was_disabled:
                    model.gradient_checkpointing_enable()
            completion_tokens = outputs[0][prompt_len:]
            return tokenizer.decode(completion_tokens, skip_special_tokens=True)

        return model_fn

    def _save_checkpoint(self, path: str) -> None:
        """Save model and tokenizer to a directory.

        Args:
            path: Directory path.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved to %s", path)

    def _save_train_log(self) -> None:
        """Save training log to JSON."""
        log_path = self.log_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(self._train_log, f, indent=2)
