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

    # Mirrors the canonical Caveman rules (upstream-caveman/skills/caveman/
    # SKILL.md), applied to the reasoning trace. NOTE: arrows are an
    # `ultra`-level device upstream and Caveman targets output, not reasoning,
    # tokens — see ../caveman-rltf/FAITHFULNESS.md.
    "caveman_full": """Solve task. Respond terse like smart caveman. All logic stay. Only fluff die.
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
