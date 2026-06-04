# Faithfulness: how this experiment relates to the upstream repos

Both references are vendored in this repo: `../upstream-caveman/` (the Caveman
skill) and `../upstream-rltf/` (the RLTF tinker_cookbook). This document is the
honest accounting of where we follow them and where we deliberately diverge, so
the blog post does not overclaim.

## 1. Caveman (`upstream-caveman`)

**What Caveman is.** A prose style for coding agents:
> "Respond terse like smart caveman. All technical substance stay. Only fluff
> die." — `upstream-caveman/skills/caveman/SKILL.md:11`

Rules (SKILL.md:21): drop articles / filler / pleasantries / hedging; fragments
OK; short synonyms; **technical terms, names, code, error strings exact**.
Arrows `X → Y` are an `ultra`-level device, not the default.

**Its explicit scope — output, not reasoning:**
> "Caveman only affects output tokens — thinking/reasoning tokens untouched.
> Caveman no make brain smaller. Caveman make *mouth* smaller."
> — `upstream-caveman/README.md:163-164`

**What we do (and the divergence).** We borrow the *style* and apply it to the
hidden **thinking trace** — exactly the place Caveman leaves alone. This is a
deliberate research extension, not a reproduction. Consequences we keep visible:

- Our caveman prompt (`src/prompts.py`) mirrors the canonical rules (drop
  articles/filler/pleasantries/hedging; keep names/numbers/constraints exact;
  fragments OK) and *adds* reasoning-specific guidance ("only the necessary
  dependency chain"; arrows allowed). Faithful to the style, extended in scope.
- We use **thinking models** (`Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-9B`, renderer
  `qwen3`, native `<think>...</think>`), precisely so the brain/mouth split is
  real. We measure **thinking tokens** (brain) and **answer tokens** (mouth)
  separately; the compression target and primary Pareto axis is thinking
  tokens. This directly tests whether feedback can shrink the brain — the thing
  Caveman explicitly does not attempt.

## 2. RLTF (`upstream-rltf`)

Upstream defines distinct distillation modes (`tinker_cookbook/rl/
data_processing.py`). The two relevant ones:

| | Upstream **SD** (`rl_reweight_mask`) | Upstream **SFT** (`sft`) |
|---|---|---|
| Objective | advantage-weighted **importance sampling** on synthetic `(x→y1)` | **cross-entropy** on `(x→y1)` |
| Trained with | jointly with multi-turn GRPO rollout | standalone |
| Correctness filter | none (advantage sign + per-token logprob mask @ −5.0) | **yes, correct revisions only** |
| Logprobs | recomputed under single-turn context | n/a |

**What we built = upstream's SFT mode**, plus a length-compression gate:
correctness-filter the revisions, then cross-entropy on `x0 → y1` with the
feedback removed (`src/build_sft_dataset.py`, `src/train_sft.py`). We therefore
call our arm **RLTF-SFT**, never RLTF-SD.

What we keep faithful to RLTF generally:
- Two-turn structure: y0 → external-judge critique → y1 (`horizon=2`).
- Feedback removed at train time so inference is single-turn (`x0 → y1`).
- An **external, stronger judge** (Claude), not self-critique. We use the
  ground-truth-grounded judge variant (the judge sees the gold answer but is
  told to give a critique, not the solution) — upstream's
  `judge_with_ground_truth` family.

What we do **not** implement (and why it's fine for a blog MVP):
- The advantage-weighted IS objective, the joint GRPO rollout, the single-turn
  logprob recompute, and the −5.0 per-token mask. Implementing those would make
  it genuine RLTF-SD; we instead provide a separate **GRPO length-reward
  control** (`src/train_grpo_length.py`) so there is at least one real RL arm to
  compare against.

## 3. Controls that make the claim falsifiable

Because RLTF-SFT is essentially rejection-sampling distillation, gains could
come from (a) just sampling a shorter correct attempt, or (b) the length
filter, rather than from caveman feedback. The ablations isolate each:

- `sft_caveman` — shortest correct **y0**, no revision at all.
- `sft_y1` — correct y1 with **no length gate**.
- `--no-feedback` — revise-shorter with **no critique**.
- `--shuffle-feedback` — revise against a **mismatched** critique (content vs
  pressure).
- `FEEDBACK_MODE=generic` — concise critique **without** caveman styling.
- `grpo_length` — RL that rewards short-if-correct **directly**.

If `rltf_sft` does not beat these, the text-feedback story does not hold, and
the post should say so.
