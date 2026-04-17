"""Modal deployment for Connect Four Opponent Modeling experiment.

Runs SFT warmup and GRPO RL training on Modal's GPU cloud.

Setup (one-time, local):
    pip install modal
    python3 -m modal setup          # authenticates via browser

Usage:
    # Run SFT warmup (condition A) — do this first
    modal run modal_train.py --command sft

    # Run preliminary RL (conditions B, C, E at 1000 steps each)
    modal run modal_train.py --command preliminary

    # Run a single RL condition
    modal run modal_train.py --command rl --condition E --steps 1000

    # Run eval on trained checkpoints
    modal run modal_train.py --command eval
"""

import modal

# ---------------------------------------------------------------------------
# Modal image: defines the container environment (cached after first build)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "cmake")
    # Core ML deps
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.51.0",
        "datasets",
        "accelerate",
        "peft",
        "open_spiel",
        "tqdm",
        "matplotlib",
        "pandas",
        "scipy",
        "tensorboard",
        "wandb",
        "bitsandbytes",
        "numpy",
        "pytest",
    )
    # Build Pons Connect Four solver
    .run_commands(
        "git clone https://github.com/PascalPons/connect4 /tmp/pons",
        "cd /tmp/pons && make -j$(nproc) || true",
        # Copy whatever binary was built (name varies by repo version)
        "cp /tmp/pons/c4solver /opt/connect4_solver 2>/dev/null "
        "|| cp /tmp/pons/connect4_solver /opt/connect4_solver 2>/dev/null "
        "|| echo 'Solver build failed, will use minimax fallback'",
        "chmod +x /opt/connect4_solver 2>/dev/null || true",
    )
)

app = modal.App("connect4-opponent-modeling", image=image)

# Persistent volume for checkpoints, data, and logs across runs
vol = modal.Volume.from_name("connect4-vol", create_if_missing=True)
VOL_PATH = "/data"

# W&B API key (set with: modal secret create wandb-secret WANDB_API_KEY=<your-key>)
wandb_secret = modal.Secret.from_name("wandb-secret", required_hint="Run: modal secret create wandb-secret WANDB_API_KEY=<key>")


def _setup_project():
    """Clone repo and symlink solver binary. Called inside each Modal function."""
    import subprocess
    import os
    from pathlib import Path

    project = Path(f"{VOL_PATH}/wikipedia-agent/connect4_opponent_modeling")

    # Clone or update repo
    if not project.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/Mraj23/wikipedia-agent.git",
             f"{VOL_PATH}/wikipedia-agent"],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", f"{VOL_PATH}/wikipedia-agent", "pull"],
                       check=False)

    os.chdir(str(project))

    # Symlink Pons solver into project dir
    solver_src = Path("/opt/connect4_solver")
    solver_dst = project / "connect4_solver"
    if solver_src.exists() and not solver_dst.exists():
        solver_dst.symlink_to(solver_src)

    # Add project to Python path
    import sys
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))

    return project


# ---------------------------------------------------------------------------
# SFT Warmup (Condition A)
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",
    timeout=7200,  # 2 hours
    volumes={VOL_PATH: vol},
    secrets=[wandb_secret],
)
def run_sft():
    """Run SFT warmup training (Condition A)."""
    project = _setup_project()

    from training.sft_train import train

    print("=" * 60)
    print("SFT WARMUP (Condition A)")
    print("=" * 60)

    train(
        model_name="Qwen/Qwen3-4B-Base",
        data_path="data/sft_warmup.jsonl",
        output_dir=f"{VOL_PATH}/checkpoints/condition_a",
        epochs=3,
        batch_size=8,
        lr=2e-5,
        device="cuda",
    )

    # Persist to volume
    vol.commit()
    print("\nSFT warmup complete! Checkpoint saved to volume.")


# ---------------------------------------------------------------------------
# RL Training (single condition)
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",
    timeout=14400,  # 4 hours
    volumes={VOL_PATH: vol},
    secrets=[wandb_secret],
)
def run_rl(condition: str = "C", game_steps: int = 1000, group_size: int = 8, seed: int = 42):
    """Run GRPO RL training for a single condition."""
    import os
    project = _setup_project()

    # Verify SFT checkpoint
    from pathlib import Path
    sft_path = Path(f"{VOL_PATH}/checkpoints/condition_a/best")
    if not sft_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_path}. Run 'modal run modal_train.py --command sft' first."
        )

    from training.grpo_config import get_config
    from spiral.grpo_trainer import GRPOTrainer

    config = get_config(condition, str(sft_path))
    config.game_steps = game_steps
    config.group_size = group_size
    config.seed = seed
    config.use_vllm = False

    # Enable W&B if key is available
    if os.environ.get("WANDB_API_KEY"):
        config.use_wandb = True
        config.wandb_project = "connect4-opponent-modeling"
        config.wandb_run_name = f"prelim_{condition}_{game_steps}steps"

    log_dir = f"{VOL_PATH}/logs/prelim_{condition.lower()}"

    print("=" * 60)
    print(f"GRPO RL TRAINING — Condition {condition}")
    print(f"  Steps: {game_steps}, Group: {group_size}, Seed: {seed}")
    print(f"  Model: {sft_path}")
    print(f"  Log dir: {log_dir}")
    print(f"  W&B: {'enabled' if config.use_wandb else 'disabled'}")
    print("=" * 60)

    trainer = GRPOTrainer(config=config, log_dir=log_dir)
    trainer.train()

    vol.commit()
    print(f"\nCondition {condition} training complete! Checkpoint saved to volume.")


# ---------------------------------------------------------------------------
# Preliminary run (B, C, E sequential)
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",
    timeout=36000,  # 10 hours
    volumes={VOL_PATH: vol},
    secrets=[wandb_secret],
)
def run_preliminary():
    """Run conditions B, C, E at 1000 steps each (sequential on 1 GPU)."""
    import os
    project = _setup_project()

    from pathlib import Path
    sft_path = Path(f"{VOL_PATH}/checkpoints/condition_a/best")
    if not sft_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_path}. Run 'modal run modal_train.py --command sft' first."
        )

    from training.grpo_config import get_config
    from spiral.grpo_trainer import GRPOTrainer
    import time

    conditions = ["B", "C", "E"]
    start = time.time()

    for condition in conditions:
        print("\n" + "=" * 60)
        print(f"STARTING CONDITION {condition}")
        print("=" * 60)

        config = get_config(condition, str(sft_path))
        config.game_steps = 1000
        config.group_size = 8
        config.seed = 42
        config.use_vllm = False

        if os.environ.get("WANDB_API_KEY"):
            config.use_wandb = True
            config.wandb_project = "connect4-opponent-modeling"
            config.wandb_run_name = f"prelim_{condition}_1k"

        log_dir = f"{VOL_PATH}/logs/prelim_{condition.lower()}"
        trainer = GRPOTrainer(config=config, log_dir=log_dir)
        trainer.train()

        vol.commit()
        print(f"Condition {condition} complete!")

    elapsed = (time.time() - start) / 60
    print("\n" + "=" * 60)
    print(f"ALL PRELIMINARY CONDITIONS COMPLETE in {elapsed:.0f} minutes")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",
    timeout=3600,
    volumes={VOL_PATH: vol},
)
def run_eval():
    """Evaluate trained checkpoints and compare conditions."""
    import json
    project = _setup_project()
    from pathlib import Path

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Compare training logs
    for cond in ["b", "c", "e"]:
        log_path = Path(f"{VOL_PATH}/logs/prelim_{cond}/train_log.json")
        if log_path.exists():
            log = json.loads(log_path.read_text())
            final = log[-1] if log else {}
            print(f"\nCondition {cond.upper()}:")
            print(f"  Final reward: {final.get('mean_reward', 'N/A')}")
            print(f"  Final loss:   {final.get('loss', 'N/A')}")
            print(f"  Final KL:     {final.get('kl', 'N/A')}")
            print(f"  Steps:        {len(log)}")
        else:
            print(f"\nCondition {cond.upper()}: No training log found")

    # Run pons benchmark if checkpoints exist
    try:
        from eval.pons_benchmark import run_pons_benchmark
        from env.pons_wrapper import PonsSolver

        solver = PonsSolver()
        for cond in ["b", "c", "e"]:
            ckpt = Path(f"{VOL_PATH}/checkpoints/condition_{cond}/best")
            if not ckpt.exists():
                ckpt = Path(f"checkpoints/condition_{cond}/best")
            if ckpt.exists():
                print(f"\nPons benchmark — Condition {cond.upper()}:")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    str(ckpt), trust_remote_code=True, dtype=torch.float16
                ).to("cuda")
                model.eval()

                def model_fn(prompt):
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id)
                    return tokenizer.decode(out[0], skip_special_tokens=True)

                results = run_pons_benchmark(model_fn, solver=solver)
                print(f"  Optimal moves: {results.get('overall_pct_optimal', 'N/A')}")

                del model
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nBenchmark failed: {e}")

    vol.commit()


# ---------------------------------------------------------------------------
# Download results from volume to local machine
# ---------------------------------------------------------------------------
@app.function(volumes={VOL_PATH: vol})
def download_results():
    """List and print training results for download."""
    import json
    from pathlib import Path

    results = {}
    for cond in ["b", "c", "e"]:
        log_path = Path(f"{VOL_PATH}/logs/prelim_{cond}/train_log.json")
        if log_path.exists():
            results[cond] = json.loads(log_path.read_text())

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    command: str = "preliminary",
    condition: str = "C",
    steps: int = 1000,
):
    """Main entry point.

    Args:
        command: One of 'sft', 'rl', 'preliminary', 'eval', 'download'
        condition: RL condition (B, C, D, E, G) — used with 'rl' command
        steps: Number of RL steps — used with 'rl' command
    """
    if command == "sft":
        print("Starting SFT warmup on Modal (H100)...")
        run_sft.remote()

    elif command == "rl":
        print(f"Starting RL condition {condition} ({steps} steps) on Modal (H100)...")
        run_rl.remote(condition=condition, game_steps=steps)

    elif command == "preliminary":
        print("Starting preliminary run (B, C, E × 1000 steps) on Modal (H100)...")
        run_preliminary.remote()

    elif command == "eval":
        print("Running evaluation on Modal (H100)...")
        run_eval.remote()

    elif command == "download":
        print("Downloading results from Modal volume...")
        results = download_results.remote()
        print(results)
        with open("preliminary_results.json", "w") as f:
            f.write(results)
        print("\nSaved to preliminary_results.json")

    else:
        print(f"Unknown command: {command}")
        print("Valid commands: sft, rl, preliminary, eval, download")
