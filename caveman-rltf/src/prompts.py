"""Prompt templates for a THINKING model (Qwen3.x hybrid).

The model thinks natively inside <think>...</think>, then writes a visible
answer. We do NOT ask for a "Reasoning:" section (the <think> block IS the
reasoning); we only require the response to END with a line "Answer: <x>" so
the final answer is parseable. The compression conditions push on the
*thinking* itself — the real test of whether feedback shrinks the "brain".

Two families:
1. EVAL PROMPT CONDITIONS (`build_prompt`): plain / concise / chain_of_draft /
   caveman. Evaluating a TRAINED model under `plain` is the internalization
   test (did it stay terse without the terse prompt?).
2. RLTF-SFT PIPELINE: x0 (=caveman), judge critique (caveman-strict or
   generic-concise), and x1 revision (+ a no-feedback variant).

Faithfulness: see ../FAITHFULNESS.md. The canonical Caveman skill targets
visible OUTPUT prose, not hidden reasoning ("mouth, not brain"). Here we
deliberately apply the terse style to the thinking trace, which is an
extension beyond Caveman's stated scope.
"""

from __future__ import annotations


_ANSWER_TAIL = 'End your response with a line exactly: "Answer: <answer>".'

_PLAIN = """Solve the task. Think it through, then give the final answer.
Question:
{question}
""" + _ANSWER_TAIL

_CONCISE = """Solve the task. Keep your thinking brief — avoid unnecessary words.
Question:
{question}
""" + _ANSWER_TAIL

# Chain of Draft (Xu et al. 2025): dense, minimal intermediate steps.
_CHAIN_OF_DRAFT = """Solve the task. Think in minimal draft steps — each step a few words at most.
Question:
{question}
""" + _ANSWER_TAIL

# Caveman — canonical SKILL.md rules, applied to the THINKING.
_CAVEMAN = """Solve task. Think caveman style. All logic stay. Only fluff die.
Thinking rules:
- Drop articles, filler, pleasantries, hedging.
- Fragments OK. Short words.
- Names, objects, constraints, numbers: exact.
- Only necessary dependency chain. Arrows OK (X -> Y).
- No restating problem. No "let me" / "let's".
Question:
{question}
""" + _ANSWER_TAIL

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


def build_x0(question: str) -> str:
    return build_prompt("caveman", question)


# --------------------------------------------------------------------------
# Judge critique (caveman-strict vs generic-concise)
# --------------------------------------------------------------------------

_JUDGE_SYSTEM_CAVEMAN = """You are a strict reasoning-compression critic.
Given a question, gold answer, and the model's THINKING + answer, give feedback
that helps the model re-think in far fewer tokens, caveman-style, while staying
correct.
Do not write the revised solution. Do not solve from scratch unless needed to
explain an error.
Focus on:
- correctness
- necessary state transitions / constraints
- redundant or repeated thinking
- missing or unsafe cuts
- caveman-style compression (drop articles/filler/narration; arrows; fragments)
Return exactly:
Verdict: CORRECT or INCORRECT
Correctness:
- If incorrect, identify the first reasoning error.
- If correct, say "No correctness issue."
Keep:
- Thinking steps or facts that must stay.
Cut:
- Filler, repeated, or unnecessary thinking.
Merge:
- Steps that can be merged into shorter caveman notation.
Unsafe Cuts:
- Facts that look verbose but are necessary.
Revision Instruction:
- One sentence telling the model how to re-think more tersely."""

_JUDGE_SYSTEM_GENERIC = """You are a reasoning-compression critic.
Given a question, gold answer, and the model's thinking + answer, give feedback
that helps the model re-think in fewer tokens while staying correct.
Do not write the revised solution. Do not solve from scratch unless needed to
explain an error.
Focus on: correctness; redundant or repeated thinking; filler and restating the
prompt; unnecessary self-talk and verbose transitions.
Return exactly:
Verdict: CORRECT or INCORRECT
Correctness:
- If incorrect, identify the first reasoning error.
- If correct, say "No correctness issue."
Keep:
- Thinking steps or facts that must stay.
Cut:
- Filler, repeated, or unnecessary thinking.
Revision Instruction:
- One sentence telling the model how to re-think more concisely."""

JUDGE_SYSTEMS = {
    "caveman": _JUDGE_SYSTEM_CAVEMAN,
    "generic": _JUDGE_SYSTEM_GENERIC,
}

JUDGE_USER = """Question:
{question}

Gold answer:
{gold}

Model first attempt (thinking + answer):
{y0}"""


def build_judge_messages(question: str, gold: str, y0: str, mode: str = "caveman"):
    if mode not in JUDGE_SYSTEMS:
        raise KeyError(f"unknown feedback mode: {mode!r} (have {list(JUDGE_SYSTEMS)})")
    return (JUDGE_SYSTEMS[mode], JUDGE_USER.format(question=question, gold=gold, y0=y0))


# --------------------------------------------------------------------------
# Revision prompt x1 (with feedback) and the no-feedback variant
# --------------------------------------------------------------------------

_X1_WITH_FEEDBACK = """Original Question:
{question}
Your first attempt:
{y0}
Feedback:
{c0}
Re-solve using the feedback. Think caveman style in far fewer thinking tokens,
preserve all necessary transitions/constraints, drop filler.
""" + _ANSWER_TAIL

_X1_NO_FEEDBACK = """Original Question:
{question}
Your first attempt:
{y0}
Re-solve with much shorter thinking while keeping it correct. Think caveman
style, preserve all necessary transitions/constraints, drop filler.
""" + _ANSWER_TAIL


def build_x1(question: str, y0: str, c0: str) -> str:
    return _X1_WITH_FEEDBACK.format(question=question, y0=y0, c0=c0)


def build_x1_no_feedback(question: str, y0: str) -> str:
    return _X1_NO_FEEDBACK.format(question=question, y0=y0)
