"""Generate outputs/RESULTS.md from summary.csv (plan section 17).

Auto-derives a proceed / do-not-proceed verdict per the plan-section-15
pass criteria:
  caveman_full accuracy >= concise_cot
  caveman_full accuracy >= chain_of_draft
  caveman_full uses fewer reasoning tokens than at least one of them
  caveman_full beats the matched-budget control on accuracy
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _fmt(value, kind):
    if pd.isna(value):
        return "-"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{value:.0f}"
    return f"{value:.3f}"


def section_per_task(df: pd.DataFrame, task: str) -> str:
    sub = df[df["task"] == task].sort_values("condition")
    lines = [f"### {task}", ""]
    lines.append(
        "| condition | accuracy | mean reasoning tokens | median | parse success | invalid answer |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['condition']} | "
            f"{_fmt(r['accuracy'], 'pct')} | "
            f"{_fmt(r['mean_reasoning_tokens'], 'int')} | "
            f"{_fmt(r['median_reasoning_tokens'], 'int')} | "
            f"{_fmt(r['parse_success_rate'], 'pct')} | "
            f"{_fmt(r['invalid_answer_rate'], 'pct')} |"
        )
    lines.append("")
    return "\n".join(lines)


def pick_main_finding(df: pd.DataFrame):
    lines = []
    strong_tasks = 0
    total_tasks = 0
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].set_index("condition")
        if "caveman_full" not in sub.index:
            continue
        total_tasks += 1
        cav = sub.loc["caveman_full"]

        def get(name):
            return sub.loc[name] if name in sub.index else None

        concise = get("concise_cot")
        cod = get("chain_of_draft")
        cod_mb = get("chain_of_draft_matched_budget")
        concise_mb = get("concise_cot_matched_budget")

        beats_concise = concise is not None and cav["accuracy"] >= concise["accuracy"]
        beats_cod = cod is not None and cav["accuracy"] >= cod["accuracy"]
        fewer_tokens = any(
            ref is not None
            and cav["mean_reasoning_tokens"] < ref["mean_reasoning_tokens"]
            for ref in (concise, cod)
        )
        beats_mb = any(
            ref is not None and cav["accuracy"] > ref["accuracy"]
            for ref in (cod_mb, concise_mb)
        )

        strong = beats_concise and beats_cod and fewer_tokens and beats_mb
        if strong:
            strong_tasks += 1
        lines.append(
            f"- **{task}**: caveman_full acc="
            f"{cav['accuracy']:.3f}, tokens={cav['mean_reasoning_tokens']:.0f}. "
            f"beats_concise={beats_concise}, beats_cod={beats_cod}, "
            f"fewer_tokens={fewer_tokens}, beats_matched_budget={beats_mb} "
            f"-> {'STRONG POSITIVE' if strong else 'not strong'}"
        )

    decision = (
        "proceed to larger benchmark / training stage"
        if strong_tasks >= 1
        else "do not proceed; reframe as 'structured symbolic scratchpads'"
    )
    return "\n".join(lines), decision, strong_tasks, total_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/metrics/summary.csv")
    parser.add_argument("--output", default="outputs/RESULTS.md")
    parser.add_argument("--budgets", default="outputs/metrics/budgets.json")
    parser.add_argument(
        "--pareto-plot",
        default="plots/pareto.png",
        help="path embedded in the report, relative to the report file",
    )
    parser.add_argument("--error-analysis-dir", default="error_analysis")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    budgets = {}
    bpath = Path(args.budgets)
    if bpath.exists():
        budgets = json.loads(bpath.read_text())

    findings, decision, strong, total = pick_main_finding(df)
    model = df["model"].iloc[0] if len(df) else "?"

    lines = []
    lines.append("# First-pass Results")
    lines.append("")
    lines.append(f"**Model:** {model}")
    lines.append("")
    lines.append("**Tasks:**")
    for task in sorted(df["task"].unique()):
        n = int(df[df["task"] == task]["n_examples"].max())
        lines.append(f"- {task}, n={n}")
    lines.append("")

    if budgets:
        lines.append(
            "**Per-task caveman_full reasoning-token budgets** "
            "(used to constrain matched-budget controls):"
        )
        for t, b in sorted(budgets.items()):
            lines.append(f"- {t}: {b}")
        lines.append("")

    lines.append("## Main Finding")
    lines.append("")
    lines.append(findings)
    lines.append("")

    lines.append("## Key Table")
    lines.append("")
    for task in sorted(df["task"].unique()):
        lines.append(section_per_task(df, task))

    lines.append("## Pareto Plot")
    lines.append("")
    lines.append(f"![pareto]({args.pareto_plot})")
    lines.append("")

    err_dir = Path(args.output).parent / args.error_analysis_dir
    if err_dir.exists():
        lines.append("## Error Analysis")
        lines.append("")
        lines.append(
            f"Sampled error cells (heuristic labels; manual review expected): "
            f"`{err_dir}/`"
        )
        summary_json = err_dir / "summary.json"
        if summary_json.exists():
            data = json.loads(summary_json.read_text())
            lines.append("")
            lines.append("| task | condition | n sampled | label counts |")
            lines.append("|---|---|---:|---|")
            for r in data:
                labels = ", ".join(f"{k}={v}" for k, v in sorted(r["labels"].items()))
                lines.append(
                    f"| {r['task']} | {r['condition']} | {r['n_sampled']} | {labels} |"
                )
            lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"Strong-positive tasks: {strong}/{total}.")
    lines.append("")
    lines.append(f"**Decision:** {decision}.")
    lines.append("")
    lines.append(
        "Pass criteria (plan §15): on at least one task, caveman_full must "
        "match or beat concise_cot and chain_of_draft on accuracy, use fewer "
        "reasoning tokens than at least one of them, and beat the matched-"
        "budget control on accuracy."
    )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
