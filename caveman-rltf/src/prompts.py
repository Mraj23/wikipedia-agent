"""Prompt templates: x0 (first-turn), judge feedback, x1 (revision)."""


X0_CAVEMAN = """Solve task using caveman reasoning.
Rules:
- No filler.
- No articles unless needed.
- Use fragments.
- Use arrows for transitions.
- Keep exact names, objects, constraints.
- Show only necessary dependency chain.
- Final answer must be concise.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ..."""


JUDGE_SYSTEM = """You are a strict reasoning-compression critic.
Given a question, gold answer, and model answer, provide feedback that helps the model revise into a shorter, correct caveman-style reasoning trace.
Do not write the revised solution.
Do not solve the task from scratch unless needed to explain the error.
Focus on:
- correctness
- necessary state transitions / constraints
- redundant reasoning
- missing or unsafe cuts
- caveman-style compression
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


JUDGE_USER = """Question:
{question}

Gold answer:
{gold}

Model first answer:
{y0}"""


X1_REVISION = """Original Question:
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


def build_x0(question: str) -> str:
    return X0_CAVEMAN.format(question=question)


def build_judge_messages(question: str, gold: str, y0: str):
    return (
        JUDGE_SYSTEM,
        JUDGE_USER.format(question=question, gold=gold, y0=y0),
    )


def build_x1(question: str, y0: str, c0: str) -> str:
    return X1_REVISION.format(question=question, y0=y0, c0=c0)
