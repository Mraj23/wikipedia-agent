# Evaluation Baselines: Qwen3-4B Instruct (Untrained)

**Date:** April 29-30, 2026
**Model:** Qwen3-4B (instruct), no RL training
**Hardware:** Lambda Labs A100-SXM4-40GB
**Format:** `/no_think` + `<reasoning>/<answer>` tags
**Games per level:** 10 (alternating first/second player)
**Detailed logs:** `results/calibration_v4/` (JSON with per-move model responses)

---

## 1. Final Difficulty Ladder Results

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
| Random | 40% (4/10) | 90% |
| Minimax-1 | 20% (2/10) | 63% |
| Minimax-2 | 0% (0/10) | 59% |
| Minimax-4 | 0% (0/10) | 61% |
| MCTS-100 | 0% (0/10) | 61% |

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
- Clear training target: push minimax-1 win rate up

### Nim and TicTacToe are weaker targets
- Nim: valid rates 82-96%, barely beats random (40%)
- TicTacToe: valid rates 59-90%, solved game limits ceiling
- TicTacToe's `x(row,col)` format is inherently hard for the model

---

## 3. Prompt Comparison: Base vs Opponent-Modeling

**Game:** Connect Four vs Minimax-1, 20 games per prompt style

| Prompt | Wins | Win Rate | Valid Moves |
|---|---|---|---|
| Base | 4/20 | 20% | 92% |
| Opponent-Modeling | 3/20 | 15% | 100% |

**Finding:** No benefit from opponent-modeling prompt in `/no_think` mode. This establishes the condition F baseline — prompting alone doesn't help.

---

## 4. Thinking Mode Analysis

### Qwen3-4B's default thinking wastes tokens
- Internal `<think>` CoT consumes 2000+ tokens parsing ASCII board representation
- Never reaches strategic analysis within reasonable token budget
- Previous 12-50% valid rates were caused by this

### Solution: `/no_think` + `<reasoning>` tags
- Suppresses Qwen's verbose internal CoT
- `<reasoning>` tags produce strategic analysis in 2-3 sentences (~50-80 tokens)
- Game-agnostic — same format works for all games
- Example:
  ```
  <reasoning>The opponent has two in a row at columns 4 and 5. Dropping in
  column 5 blocks their threat. Column 4 also extends my line.</reasoning>
  <answer>5</answer>
  ```

### Board representation matters
- Raw OpenSpiel ASCII confuses the model
- Clean spaced format with column labels works well:
  ```
    . . . . . . .
    . . x x . . .
    . . o x o o .
    0 1 2 3 4 5 6
  ```

---

## 5. Move Parsing Improvements

Invalid move patterns identified and fixed:
- **TicTacToe:** Model outputs `(1,1)` without player prefix → infer from legal actions
- **Nim:** Model omits trailing semicolon → flexible `pile:X, take:Y` parsing
- **Breakthrough:** Model omits capture asterisk → match `d3c2` against `d3c2*`
- **All games:** Fall back to searching full response when `<answer>` tag missing

---

## 6. Methodology

### Evaluation setup
- 10 games per difficulty level, alternating first/second player
- Invalid moves fall back to random play
- Opponents: Random, Minimax depth 1/2/4, MCTS 100 simulations

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

### Scripts and logs
- Calibration: `scripts/calibrate_transfer.py`
- Logs with per-move model responses: `results/calibration_v4/`

---

## 7. Evaluation Hierarchy for the Experiment

| Level | Game | Opponent | Expected signal |
|---|---|---|---|
| **In-domain** | Connect Four | Pons solver | High — direct training target |
| **Near-transfer** | Connect Four | Minimax ladder | High — same game, different eval |
| **Primary transfer** | Breakthrough | Minimax ladder + MCTS | Moderate — best gradient |
| **Secondary transfer** | Nim | Minimax ladder | Low — fewer valid moves |
| **Control** | TicTacToe | Minimax-2 | None expected — game is solved |
| **Non-adversarial** | GSM8K / MATH-500 | N/A | None expected — specificity control |
