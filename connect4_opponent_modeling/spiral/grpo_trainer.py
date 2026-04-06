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
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        logger.info("Loading model from %s (device=%s)", config.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)

        # Frozen reference model for KL computation
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        # Game infrastructure
        self.solver = PonsSolver(fallback_depth=config.opponent_depth)
        self.reward_calc = RewardCalculator(self.solver)
        self.minimax = MinimaxSolver(depth=config.opponent_depth)

        # Position buffer
        logger.info("Building position buffer...")
        self.position_buffer = PositionBuffer(
            pool_size=min(10000, max(50, config.game_steps * 2)),
            min_moves_remaining=6 if config.condition == "B" else 2,
        )
        logger.info("Position buffer ready (%d positions)", len(self.position_buffer))

        # Eval callback
        self.checkpoint_dir = Path(f"checkpoints/condition_{config.condition.lower()}")
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

            # 5. Compute advantages
            advantages = compute_advantages(
                rewards=rewards,
                reward_components=reward_components,
                reward_weights=self.config.reward_weights or {},
                use_rae=self.config.use_rae,
            )

            # 6. Compute loss and update
            loss, kl = self._grpo_step(prompt, completions, gen_log_probs, advantages)

            step_time = time.time() - step_start

            # 7. Logging
            log_entry = {
                "step": step,
                "loss": loss,
                "kl": kl,
                "mean_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "step_time_s": step_time,
            }
            self._train_log.append(log_entry)

            log_every = 100 if self.config.game_steps > 100 else 1
            if step % log_every == 0:
                logger.info(
                    "Step %d/%d: loss=%.4f kl=%.4f reward=%.3f [%.3f, %.3f] (%.1fs)",
                    step, self.config.game_steps, loss, kl,
                    log_entry["mean_reward"], log_entry["min_reward"],
                    log_entry["max_reward"], step_time,
                )

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
                self.eval_callback.run(step)

            # 10. Periodic log save
            if step % 500 == 0 and step > 0:
                self._save_train_log()

        # Final eval and checkpoint
        logger.info("Training complete. Running final eval...")
        self.eval_callback.run(self.config.game_steps)
        self._save_checkpoint(str(self.checkpoint_dir / "final"))
        self._save_train_log()

    def _generate_group(
        self, prompt: str
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate group_size completions for a prompt.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Tuple of (decoded_texts, per_completion_log_probs).
        """
        self.model.eval()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        completions = []
        log_probs_list = []

        # Generate in mini-batches to manage memory
        remaining = self.config.group_size
        while remaining > 0:
            batch = min(remaining, 4)
            with torch.no_grad():
                # Force minimum generation length to prevent immediate EOS
                # (the SFT model learns to emit EOS after </move>, and the
                # prompt examples contain </move> which triggers early stopping)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    min_new_tokens=10,
                    num_return_sequences=batch,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for seq in outputs:
                # Decode only the generated part
                gen_tokens = seq[prompt_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                completions.append(text)

                # Compute log-probs for the generated tokens
                lp = self._compute_log_probs(inputs["input_ids"][0], gen_tokens)
                log_probs_list.append(lp)

            remaining -= batch

        self.model.train()
        return completions, log_probs_list

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
        move_quality = self.solver.normalize_reward(env, played_col)
        terminal = RewardCalculator._terminal_reward(game_result)
        format_r = RewardCalculator._format_reward(response, condition)

        if condition == "C":
            reward = self.reward_calc.condition_c_reward(
                env, played_col, game_result, response
            )
            comp = {"move": move_quality, "terminal": terminal, "format": format_r}

        elif condition == "D":
            reward = self.reward_calc.condition_d_reward(
                env, played_col, game_result, response
            )
            future_state = parsed.get("future_state", "")
            fs_acc = self.reward_calc._future_state_accuracy(env, played_col, future_state)
            comp = {
                "move": move_quality, "future": fs_acc,
                "terminal": terminal, "format": format_r,
            }
            # D also extracts opponent_prediction (same format as E) but
            # does NOT reward it — the auxiliary signal comes from future_state only.
            # This equalizes the output format between D and E.

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
                "terminal": terminal, "format": format_r,
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
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"][0].to(self.device)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl = 0.0
        n_valid = 0

        for i, completion in enumerate(completions):
            gen_ids = self.tokenizer(
                completion, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(self.device)

            if gen_ids.numel() == 0:
                continue

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

            # Loss = -surrogate + kl_coef * kl
            completion_loss = -surrogate + self.config.kl_coef * kl
            total_loss = total_loss + completion_loss
            n_valid += 1

        if n_valid > 0:
            avg_loss = total_loss / n_valid
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return avg_loss.item(), total_kl / n_valid
        else:
            return 0.0, 0.0

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
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
