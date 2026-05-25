"""Prompt templates per condition."""

PROMPTS = {
    "answer_only": """Solve the task. Return only the final answer.
Question:
{question}
Answer:""",

    "normal_cot": """Solve the task. Think step by step, then give the final answer.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ...""",

    "concise_cot": """Solve the task. Use concise reasoning. Avoid unnecessary explanation.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ...""",

    "chain_of_draft": """Solve the task using minimal draft notes.
Each reasoning step should be very short, around 5 words or fewer.
Do not write full explanatory sentences unless necessary.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ...""",

    "caveman_full": """Solve task using caveman reasoning.
Rules:
- No filler.
- No articles unless needed.
- Use fragments.
- Use arrows for transitions.
- Keep exact names, objects, constraints.
- Show only necessary dependency chain.
Question:
{question}
Return exactly:
Reasoning: ...
Answer: ...""",
}


BUDGET_LINE = "Use at most {budget} reasoning tokens."

MATCHED_BUDGET_CONDITIONS = {
    "concise_cot_matched_budget": "concise_cot",
    "chain_of_draft_matched_budget": "chain_of_draft",
}


def build_prompt(condition: str, question: str, budget: int | None = None) -> str:
    if condition in MATCHED_BUDGET_CONDITIONS:
        if budget is None:
            raise ValueError(f"condition {condition!r} requires --budget")
        base = PROMPTS[MATCHED_BUDGET_CONDITIONS[condition]]
        template = base.replace(
            "Question:\n{question}",
            BUDGET_LINE.format(budget=budget) + "\nQuestion:\n{question}",
        )
    elif condition in PROMPTS:
        template = PROMPTS[condition]
    else:
        raise KeyError(f"unknown condition: {condition}")
    return template.format(question=question)
