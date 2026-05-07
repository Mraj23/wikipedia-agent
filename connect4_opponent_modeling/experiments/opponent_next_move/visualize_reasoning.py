"""Build a static HTML report for inspecting saved reasoning traces."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROWS = 6
COLS = 7


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0


def _load_banks(path: Path) -> Dict[Tuple[str, str], Dict]:
    banks = {}
    with path.open() as handle:
        for line in handle:
            rec = json.loads(line)
            banks[(rec.get("split", "unknown"), str(rec["moves"]))] = rec
    return banks


def _board_from_moves(moves: str) -> Tuple[List[List[int]], int]:
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    heights = [ROWS - 1] * COLS
    for idx, ch in enumerate(moves):
        col = int(ch)
        row = heights[col]
        player = 1 if idx % 2 == 0 else 2
        board[row][col] = player
        heights[col] -= 1
    current_player = 1 if len(moves) % 2 == 0 else 2
    return board, current_player


def _render_board(moves: str) -> str:
    board, current_player = _board_from_moves(moves)
    other_player = 2 if current_player == 1 else 1
    lines = ['<div class="board">']
    for row in board:
        for cell in row:
            if cell == current_player:
                cls, text = "piece you", "X"
            elif cell == other_player:
                cls, text = "piece opp", "O"
            else:
                cls, text = "piece empty", "."
            lines.append(f'<div class="{cls}">{text}</div>')
    for col in range(COLS):
        lines.append(f'<div class="col-label">{col}</div>')
    lines.append("</div>")
    return "\n".join(lines)


def _render_scores(bank_rec: Dict, chosen: object) -> str:
    scores = {int(col): int(score) for col, score in bank_rec.get("scores", {}).items()}
    if not scores:
        return '<div class="scores missing">No bank scores found.</div>'

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    best = {col for col, score in scores.items() if score == max_score}
    chosen_col = chosen if isinstance(chosen, int) else None
    parts = ['<div class="score-grid">']
    for col in range(COLS):
        if col not in scores:
            parts.append(f'<div class="score-card illegal"><b>{col}</b><span>illegal</span></div>')
            continue
        score = scores[col]
        norm = 1.0 if max_score == min_score else (score - min_score) / (max_score - min_score)
        classes = ["score-card"]
        if col == chosen_col:
            classes.append("chosen")
        if col in best:
            classes.append("best")
        parts.append(
            f'<div class="{" ".join(classes)}">'
            f"<b>{col}</b>"
            f'<div class="bar"><span style="width:{norm * 100:.1f}%"></span></div>'
            f"<span>{score}</span>"
            "</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _format_number(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _bucket_records(records: List[Dict], max_per_bucket: int) -> List[Tuple[str, List[Dict]]]:
    answer_valid = [rec for rec in records if rec.get("answer_valid")]
    invalid = [rec for rec in records if not rec.get("answer_valid")]
    capped = [rec for rec in records if len(rec.get("response", "")) >= 1000]

    high = sorted(answer_valid, key=lambda rec: rec.get("move_quality", 0.0), reverse=True)
    low = sorted(answer_valid, key=lambda rec: rec.get("move_quality", 0.0))

    if records:
        stride = max(1, len(records) // max_per_bucket)
        spread = records[::stride][:max_per_bucket]
    else:
        spread = []

    return [
        ("High Move Quality", high[:max_per_bucket]),
        ("Low Move Quality", low[:max_per_bucket]),
        ("Invalid Or Missing Answer", invalid[:max_per_bucket]),
        ("Stored Trace Capped At 1000 Chars", capped[:max_per_bucket]),
        ("Spread Sample", spread),
    ]


def _record_card(rec: Dict, bank_rec: Dict, response_chars: int) -> str:
    response = rec.get("response", "")
    capped_note = ""
    if len(response) >= 1000:
        capped_note = '<span class="warn">stored response hit 1000-char cap</span>'

    meta = {
        "idx": rec.get("index"),
        "split": rec.get("split"),
        "moves": rec.get("moves"),
        "move_count": rec.get("move_count"),
        "model_move": rec.get("model_move"),
        "move_quality": _format_number(rec.get("move_quality")),
        "optimal": rec.get("optimal"),
        "opponent_prediction": rec.get("opponent_prediction"),
        "opponent_reply_quality": _format_number(rec.get("opponent_reply_quality")),
        "answer_valid": rec.get("answer_valid"),
        "schema_valid": rec.get("schema_valid"),
        "score_spread": rec.get("score_spread"),
    }
    chips = "\n".join(
        f'<span class="chip"><b>{html.escape(str(key))}</b> {html.escape(str(value))}</span>'
        for key, value in meta.items()
    )

    response_text = response[:response_chars]
    if len(response) > response_chars:
        response_text += "\n...[display truncated by report]"

    return f"""
<article class="card">
  <div class="card-head">
    <div class="chips">{chips}</div>
    {capped_note}
  </div>
  <div class="viz-row">
    {_render_board(str(rec.get("moves", "")))}
    {_render_scores(bank_rec, rec.get("model_move"))}
  </div>
  <details open>
    <summary>Saved model response</summary>
    <pre>{html.escape(response_text)}</pre>
  </details>
</article>
"""


def build_report(
    *,
    result_path: Path,
    banks_path: Path,
    output_path: Path,
    max_per_bucket: int,
    response_chars: int,
) -> None:
    data = json.loads(result_path.read_text())
    banks = _load_banks(banks_path)
    records = data.get("records", [])
    summary = data.get("summary", {})

    cards = []
    for bucket_name, bucket in _bucket_records(records, max_per_bucket):
        cards.append(f"<h2>{html.escape(bucket_name)}</h2>")
        if not bucket:
            cards.append('<p class="empty">No records in this bucket.</p>')
            continue
        for rec in bucket:
            bank_rec = banks.get((rec.get("split", "unknown"), str(rec.get("moves", ""))), {})
            cards.append(_record_card(rec, bank_rec, response_chars))

    by_split = summary.get("by_split", {})
    split_rows = []
    for split, row in sorted(by_split.items()):
        split_rows.append(
            "<tr>"
            f"<td>{html.escape(split)}</td>"
            f"<td>{row.get('n', 0)}</td>"
            f"<td>{_format_number(row.get('mean_move_quality'))}</td>"
            f"<td>{_format_number(row.get('pct_optimal'))}</td>"
            f"<td>{_format_number(row.get('answer_valid_rate'))}</td>"
            f"<td>{_format_number(row.get('schema_valid_rate'))}</td>"
            f"<td>{_format_number(row.get('position_valid_rate'))}</td>"
            f"<td>{row.get('position_invalid_count', 0)}</td>"
            "</tr>"
        )

    reasoning_tag_rate = _mean(
        1.0 if "<reasoning>" in rec.get("response", "") else 0.0 for rec in records
    )
    answer_tag_rate = _mean(
        1.0 if "<answer>" in rec.get("response", "") else 0.0 for rec in records
    )
    capped_count = sum(1 for rec in records if len(rec.get("response", "")) >= 1000)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Reasoning Report - {html.escape(summary.get('model_label', result_path.stem))}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f7f4;
  --panel: #ffffff;
  --ink: #1d2428;
  --muted: #667078;
  --line: #d9dedb;
  --you: #2674a6;
  --opp: #c44d40;
  --best: #2f8f5b;
  --chosen: #111827;
  --warn: #a15c00;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{ max-width: 1240px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 6px; font-size: 28px; }}
h2 {{ margin: 34px 0 12px; font-size: 20px; }}
.subtle {{ color: var(--muted); margin-top: 0; }}
.summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0;
}}
.stat {{ background: var(--panel); border: 1px solid var(--line); padding: 12px; border-radius: 8px; }}
.stat b {{ display: block; font-size: 18px; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
th {{ color: var(--muted); font-weight: 600; font-size: 13px; }}
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  margin: 12px 0;
}}
.card-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; font-size: 12px; color: var(--muted); }}
.chip b {{ color: var(--ink); }}
.warn {{ color: var(--warn); font-weight: 700; font-size: 12px; white-space: nowrap; }}
.viz-row {{ display: grid; grid-template-columns: 286px 1fr; gap: 18px; margin: 14px 0; align-items: start; }}
.board {{ display: grid; grid-template-columns: repeat(7, 34px); gap: 4px; }}
.piece, .col-label {{
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  font-weight: 800;
  font-size: 15px;
}}
.empty {{ background: #eef1ef; color: #a0a8a4; }}
.you {{ background: var(--you); color: white; }}
.opp {{ background: var(--opp); color: white; }}
.col-label {{ border-radius: 4px; height: 22px; color: var(--muted); font-size: 12px; }}
.score-grid {{ display: grid; grid-template-columns: repeat(7, minmax(68px, 1fr)); gap: 8px; }}
.score-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px; min-height: 76px; }}
.score-card b {{ display: block; margin-bottom: 6px; }}
.score-card.best {{ border-color: var(--best); box-shadow: inset 0 0 0 2px rgba(47, 143, 91, 0.15); }}
.score-card.chosen {{ border-color: var(--chosen); box-shadow: inset 0 0 0 2px rgba(17, 24, 39, 0.18); }}
.score-card.illegal {{ opacity: 0.38; }}
.bar {{ height: 8px; background: #edf0ee; border-radius: 999px; overflow: hidden; margin: 8px 0; }}
.bar span {{ display: block; height: 100%; background: var(--best); }}
details {{ border-top: 1px solid var(--line); padding-top: 8px; }}
summary {{ cursor: pointer; color: var(--muted); font-weight: 700; }}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f4f5f2;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  line-height: 1.45;
  font-size: 13px;
}}
@media (max-width: 900px) {{
  .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .viz-row {{ grid-template-columns: 1fr; }}
  .score-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<main>
<h1>Reasoning Report: {html.escape(summary.get('model_label', result_path.stem))}</h1>
<p class="subtle">Source: {html.escape(str(result_path))}</p>
<section class="summary">
  <div class="stat">Primary move quality<b>{_format_number(summary.get('primary_mean_move_quality'))}</b></div>
  <div class="stat">Position validity<b>{_format_number(summary.get('position_valid_rate'))}</b></div>
  <div class="stat">Responses with reasoning tag<b>{reasoning_tag_rate:.3f}</b></div>
  <div class="stat">Capped traces<b>{capped_count}/{len(records)}</b></div>
  <div class="stat">Responses with answer tag<b>{answer_tag_rate:.3f}</b></div>
  <div class="stat">Total records<b>{len(records)}</b></div>
  <div class="stat">Condition<b>{html.escape(str(summary.get('condition')))}</b></div>
  <div class="stat">Backend<b>{html.escape(str(summary.get('generation_backend')))}</b></div>
</section>
<table>
<thead>
<tr><th>split</th><th>n</th><th>move_quality</th><th>pct_optimal</th><th>answer_valid</th><th>schema_valid</th><th>position_valid</th><th>position_invalid</th></tr>
</thead>
<tbody>
{''.join(split_rows)}
</tbody>
</table>
{''.join(cards)}
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved reasoning traces.")
    parser.add_argument(
        "--result",
        default="experiments/opponent_next_move/results/base_scaffold.json",
        help="Result JSON produced by eval_move_quality.py.",
    )
    parser.add_argument(
        "--banks",
        default="experiments/opponent_next_move/data/connect4_eval_banks.jsonl",
        help="Locked bank JSONL with oracle scores.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML output path. Defaults to RESULT stem plus _reasoning_report.html.",
    )
    parser.add_argument("--max_per_bucket", type=int, default=8)
    parser.add_argument("--response_chars", type=int, default=3000)
    args = parser.parse_args()

    result_path = Path(args.result)
    if args.output is None:
        output_path = result_path.with_name(f"{result_path.stem}_reasoning_report.html")
    else:
        output_path = Path(args.output)

    build_report(
        result_path=result_path,
        banks_path=Path(args.banks),
        output_path=output_path,
        max_per_bucket=args.max_per_bucket,
        response_chars=args.response_chars,
    )
    print(output_path)


if __name__ == "__main__":
    main()
