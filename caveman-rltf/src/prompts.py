"""Prompt templates.

Two families:

1. EVAL PROMPT CONDITIONS (`PROMPTS` / `build_prompt`) — the inference-time
   comparison arms: plain baseline, concise, Chain-of-Draft, and caveman.
   These let eval.py measure each prompt's effect, and (critically) let us
   evaluate a TRAINED model under the PLAIN prompt to test whether terseness
   was internalized rather than merely prompted.

2. RLTF-SFT PIPELINE PROMPTS — x0 (first turn, = caveman), the judge
   critique (caveman-strict or generic-concise), and x1 (revision).

Faithfulness note: the canonical Caveman skill (see ../FAITHFULNESS.md and
upstream-caveman/skills/caveman/SKILL.md) is a *prose* style — "Respond
terse like smart caveman. All technical substance stay. Only fluff die." It
explicitly does NOT touch hidden reasoning ("make mouth smaller, not brain").
This experiment deliberately applies that style to the visible reasoning
trace, which is an extension beyond Caveman's stated scope. The caveman
prompt below mirrors the canonical rules (drop articles/filler/pleasantries/
hedging, keep technical terms/names exact, fragments OK) and adds reasoning-
specific guidance (show only the necessary dependency chain; arrows OK — note
arrows are an `ultra`-level device upstream, used here by choice).
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# 1. Eval prompt conditions
# --------------------------------------------------------------------------

_PLAIN = """Solve the task. Think step by step, then give the final answer.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ..."""

_CONCISE = """Solve the task. Use brief reasoning. Avoid unnecessary words.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ..."""

# Chain of Draft (Xu et al. 2025): dense, minimal intermediate steps.
_CHAIN_OF_DRAFT = """Solve the task using concise draft reasoning.
Keep only the key intermediate steps; each step a few words at most.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ..."""

# Caveman — mirrors the canonical SKILL.md rules, applied to reasoning.
_CAVEMAN = """Solve task. Respond terse like smart caveman. All logic stay. Only fluff die.
Rules:
- Drop articles, filler, pleasantries, hedging.
- Fragments OK. Short words.
- Names, objects, constraints, numbers: exact.
- Show only necessary dependency chain. Arrows OK (X -> Y).
- No restating problem. No "let me" / "let's".
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ..."""

PROMPTS = {
    "plain": _PLAIN,
    "concise": _CONCISE,
    "chain_of_draft": _CHAIN_OF_DRAFT,
    "caveman": _CAVEMAN,
}


def build_prompt(condition: str, question: str) -> str:
    if condition not in PROMPTS:
        raise KeyError(f"unknown prompt condition: {condition!r} (have {list(PROMPTS)})")
    return PROMPTS[condition].format(question=question)


# x0 (the first-turn RLTF prompt) IS the caveman prompt.
def build_x0(question: str) -> str:
    return build_prompt("caveman", question)


# --------------------------------------------------------------------------
# 2. Judge critique (caveman-strict vs generic-concise)
# --------------------------------------------------------------------------

_JUDGE_SYSTEM_CAVEMAN = """You are a strict reasoning-compression critic.
Given a question, gold answer, and model answer, provide feedback that helps
the model revise into a shorter, correct caveman-style reasoning trace.
Do not write the revised solution.
Do not solve the task from scratch unless needed to explain the error.
Focus on:
- correctness
- necessary state transitions / constraints
- redundant reasoning
- missing or unsafe cuts
- caveman-style compression (drop articles/filler/narration; arrows; fragments)
Return exactly:
Verdict: CORRECT or INCORRECT
Correctness:
- If incorrect, identify the first reasoning error.
- If correct, say "No correctness issue."
Keep:
- List reasoning steps or facts that must stay.
Cut:
- List filler, repeated, or unnecessary steps.
Merge:
- List steps that can be merged into shorter caveman notation.
Unsafe Cuts:
- List any facts that look verbose but are necessary.
Revision Instruction:
- One sentence telling the model how to rewrite."""

# Generic-concise critic — same correctness discipline, no caveman styling.
# Used for the "does the caveman FEEDBACK content matter?" ablation.
_JUDGE_SYSTEM_GENERIC = """You are a reasoning-compression critic.
Given a question, gold answer, and model answer, provide feedback that helps
the model revise into shorter, correct reasoning.
Do not write the revised solution.
Do not solve the task from scratch unless needed to explain the error.
Focus on:
- correctness
- redundant or repeated reasoning
- filler phrases and restating the prompt
- unnecessary self-talk and verbose transitions
Return exactly:
Verdict: CORRECT or INCORRECT
Correctness:
- If incorrect, identify the first reasoning error.
- If correct, say "No correctness issue."
Keep:
- List reasoning steps or facts that must stay.
Cut:
- List filler, repeated, or unnecessary steps.
Revision Instruction:
- One sentence telling the model how to rewrite more concisely."""

JUDGE_SYSTEMS = {
    "caveman": _JUDGE_SYSTEM_CAVEMAN,
    "generic": _JUDGE_SYSTEM_GENERIC,
}

JUDGE_USER = """Question:
{question}

Gold answer:
{gold}

Model first answer:
{y0}"""


def build_judge_messages(question: str, gold: str, y0: str, mode: str = "caveman"):
    if mode not in JUDGE_SYSTEMS:
        raise KeyError(f"unknown feedback mode: {mode!r} (have {list(JUDGE_SYSTEMS)})")
    return (
        JUDGE_SYSTEMS[mode],
        JUDGE_USER.format(question=question, gold=gold, y0=y0),
    )


# --------------------------------------------------------------------------
# 3. Revision prompt x1 (with feedback) and the no-feedback variant
# --------------------------------------------------------------------------

_X1_WITH_FEEDBACK = """Original Question:
{question}
Your first answer:
{y0}
Feedback:
{c0}
Revise your answer using the feedback.
Rules:
- Use caveman reasoning.
- Preserve all necessary state transitions or constraints.
- Remove filler and redundant explanation.
- Use arrows where useful.
- Return only Reasoning and Answer.
Return exactly:
Reasoning: ...
Answer: ..."""

# No-feedback control: same revise-shorter pressure, but NO critique. Tests
# how much of the gain is just rejection-sampling a shorter correct attempt.
_X1_NO_FEEDBACK = """Original Question:
{question}
Your first answer:
{y0}
Revise your answer to be much shorter while keeping it correct.
Rules:
- Use caveman reasoning.
- Preserve all necessary state transitions or constraints.
- Remove filler and redundant explanation.
- Return only Reasoning and Answer.
Return exactly:
Reasoning: ...
Answer: ..."""


def build_x1(question: str, y0: str, c0: str) -> str:
    return _X1_WITH_FEEDBACK.format(question=question, y0=y0, c0=c0)


def build_x1_no_feedback(question: str, y0: str) -> str:
    return _X1_NO_FEEDBACK.format(question=question, y0=y0)
