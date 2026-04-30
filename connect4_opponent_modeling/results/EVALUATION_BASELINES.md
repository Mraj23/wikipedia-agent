# Evaluation Baselines: Qwen3-4B Instruct (Untrained)

**Date:** April 29-30, 2026
**Model:** Qwen3-4B (instruct), no RL training
**Hardware:** Lambda Labs A100-SXM4-40GB
**Format:** `/no_think` + `<reasoning>/<answer>` tags
**Games per level:** 10 (alternating first/second player)

---

## 1. Difficulty Ladder Results

### Connect Four

| Opponent | Win Rate | Valid Moves |
|---|---|---|
| Random | 70% (7/10) | 92% |
| Minimax-1 | 10% (1/10) | 96% |
| Minimax-2 | 0% (0/10) | 98% |
| Minimax-4 | 0% (0/10) | 100% |
| MCTS-100 | 0% (0/10) | 100% |

### Breakthrough (6x6)

| Opponent | Win Rate | Valid Moves |
|---|---|---|
| Random | **90%** (9/10) | 87% |
| Minimax-1 | **60%** (6/10) | 89% |
| Minimax-2 | **50%** (5/10) | 86% |
| Minimax-4 | **40%** (4/10) | 86% |
| MCTS-100 | **10%** (1/10) | 87% |

### Nim (piles: 1, 3, 5, 7)

| Opponent | Win Rate | Valid Moves |
|---|---|---|
| Random | 40% (4/10) | 91% |
| Minimax-1 | 20% (2/10) | 96% |
| Minimax-2 | 10% (1/10) | 84% |
| Minimax-4 | 0% (0/10) | 82% |
| MCTS-100 | 0% (0/10) | 85% |

### Tic-Tac-Toe

| Opponent | Win Rate | Valid Moves |
|---|---|---|
| Random | 40% (4/10) | 81% |
| Minimax-1 | 20% (2/10) | 59% |
| Minimax-2 | 0% (0/10) | 52% |
| Minimax-4 | 0% (0/10) | 55% |
| MCTS-100 | 10% (1/10) | 62% |

---

## 2. Key Findings

### Breakthrough is the best transfer evaluation target
- Smoothest difficulty gradient: 90% → 60% → 50% → 40% → 10%
- Model beats MCTS-100 10% of the time — genuine strategic ability
- Valid move rates 86-89% across all difficulty levels
- Most room for measurable improvement from training

### Connect Four baseline is moderate
- Beats random 70%, struggles at minimax-1 (10%), fails at minimax-2+
- Valid moves 92-100% — excellent format compliance
- Clear training target: push minimax-1 win rate from 10% to 40%+

### Nim and TicTacToe are weaker targets
- Nim: low valid rates (82-96%), barely beats random (40%)
- TicTacToe: lowest valid rates (52-81%), solved game limits ceiling
- Both have less room for meaningful improvement

### Valid move rates improved significantly with lenient parsing
- Nim: 53% → 91% (fixed missing semicolons)
- TicTacToe: 48% → 81% (fixed missing player prefix)
- Breakthrough: 75% → 87% (fixed missing capture asterisk)
- Connect Four: was already 90%+ (column numbers are simple)

---

## 3. Prompt Comparison: Base vs Opponent-Modeling

**Game:** Connect Four vs Minimax-1, 20 games per prompt style

| Prompt | Wins | Win Rate | Valid Moves |
|---|---|---|---|
| Base | 4/20 | 20% | 92% |
| Opponent-Modeling | 3/20 | 15% | 100% |

**Finding:** No benefit from opponent-modeling prompt in `/no_think` mode. The model doesn't actually reason about opponents when thinking is suppressed. This establishes the condition F baseline — prompting alone doesn't help.

---

## 4. Thinking Mode Observations

### Qwen3-4B's default thinking is problematic
- Internal `<think>` chain-of-thought consumes 2000+ tokens per move
- Most tokens spent parsing ASCII board representation, not strategizing
- Often never finishes thinking within token budget
- Previous valid rates of 12-50% were caused by this, not model inability

### `/no_think` + `<reasoning>` tags is the solution
- Suppresses Qwen's verbose internal CoT
- Our `<reasoning>` tags produce concise strategic analysis (2-3 sentences, ~50-80 tokens)
- Example output:
  ```
  <reasoning>The opponent has two in a row at columns 4 and 5. To block this,
  dropping in column 5 would prevent completion. Column 4 also extends my own
  line.</reasoning>
  <answer>5</answer>
  ```
- Works across all games without game-specific templates
- Based on SPIRAL paper format: structured tags, free-form reasoning content

### Board representation matters
- Raw OpenSpiel ASCII (`..xx...`) confuses the model → wastes tokens parsing
- Clean spaced format with column labels → model immediately reads positions
  ```
    . . . . . . .
    . . x x . . .
    . . o x o o .
    0 1 2 3 4 5 6
  ```

---

## 5. Methodology

### Evaluation setup
- Each difficulty level: 10 games with alternating first/second player
- Invalid moves fall back to random play
- Opponents:
  - **Random:** uniform random over legal moves
  - **Minimax-1/2/4:** OpenSpiel alpha-beta with heuristic value function at depth 1/2/4
  - **MCTS-100:** OpenSpiel Monte Carlo Tree Search with 100 simulations, random rollout evaluator

### Prompt format
```
{game_description}

Board:
{clean_board_state}

Legal moves: {legal_action_strings}

Respond in this exact format:
<reasoning>Analyze threats and best move in 2-3 sentences</reasoning>
<answer>your_move</answer>

/no_think
```

### Move parsing (lenient)
- Primary: exact match against OpenSpiel legal action strings inside `<answer>` tag
- Fallbacks per game:
  - Connect Four: bare column digits 0-6
  - TicTacToe: add missing player prefix (`(0,1)` → `x(0,1)`)
  - Nim: flexible `pile:X, take:Y` without requiring semicolon
  - Breakthrough: match without capture asterisk (`d3c2` → `d3c2*`)

### Scripts
- Calibration: `scripts/calibrate_transfer.py`
- Detailed logs saved to `results/calibration_v4/` (JSON with per-move model responses)

---

## 6. Recommended Evaluation Hierarchy for the Experiment

| Level | Game | Opponent | Expected signal |
|---|---|---|---|
| **In-domain** | Connect Four | Pons solver | High — direct training target |
| **Near-transfer** | Connect Four | Minimax-1/2/4 ladder | High — same game, different eval |
| **Primary transfer** | Breakthrough | Minimax-1/2/4 + MCTS-100 | Moderate — best gradient |
| **Secondary transfer** | Nim | Minimax-1/2/4 | Low — fewer valid moves |
| **Control** | TicTacToe | Minimax-2 | None expected — game is solved |
| **Non-adversarial** | GSM8K / MATH-500 | N/A | None expected — adversarial specificity control |
