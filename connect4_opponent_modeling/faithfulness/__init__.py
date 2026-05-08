"""RL faithfulness experiment for Connect Four.

Measures whether tactical claims in LLM reasoning traces are (a) tactically
true (Pons oracle) and (b) causally influential on the chosen move
(intervention + resampling). Yields a 2x2 categorization:

    | truth \\ causal | causal              | non-causal           |
    | true            | faithful            | decorative truth     |
    | false           | load-bearing-false  | hallucinated         |

Headline metric: rate of false-but-causal claims as a function of
solver regret over RL training steps.

This experiment is orthogonal to the active opponent-modeling experiment
described in connect4_opponent_modeling/CLAUDE.md. Faithfulness results do
not extend, and are not extended by, the E > D claim.
"""
