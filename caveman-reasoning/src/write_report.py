"""Generate outputs/RESULTS.md from summary.csv (plan section 17).

Reports two distinct verdicts that the plan distinguishes:

  §15 Strong positive (per-task signal quality): caveman_full accuracy
       >= concise_cot AND >= chain_of_draft, uses fewer reasoning tokens
       than at least one, AND beats the matched-budget control.

  §18 Proceed gate (decision for next stage): caveman_full beats
       chain_of_draft_matched_budget on at least one task AND does not
       significantly hurt the other task. "Significantly hurt" is
       operationalized as accuracy >= 5pp below normal_cot.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


HURT_THRESHOLD_PP = 5.0


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


def strong_positive_per_task(df: pd.DataFrame):
    """Plan §15 per-task signal-quality check."""
    findings = []
    strong_count = 0
    total = 0
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].set_index("condition")
        if "caveman_full" not in sub.index:
            continue
        total += 1
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
            strong_count += 1
        findings.append(
            f"- **{task}**: caveman_full acc="
            f"{cav['accuracy']:.3f}, tokens={cav['mean_reasoning_tokens']:.0f}. "
            f"beats_concise={beats_concise}, beats_cod={beats_cod}, "
            f"fewer_tokens={fewer_tokens}, beats_matched_budget={beats_mb} "
            f"-> {'STRONG POSITIVE' if strong else 'not strong'}"
        )
    return "\n".join(findings), strong_count, total


def proceed_decision(df: pd.DataFrame, hurt_pp: float = HURT_THRESHOLD_PP):
    """Plan §18 next-stage gate."""
    per_task = {}
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].set_index("condition")

        def get(name):
            return sub.loc[name] if name in sub.index else None

        cav = get("caveman_full")
        cod_mb = get("chain_of_draft_matched_budget")
        normal = get("normal_cot")

        if cav is None:
            per_task[task] = None
            continue

        beats_cod_mb = (
            cod_mb is not None and cav["accuracy"] > cod_mb["accuracy"]
        )
        drop_pp = (
            (normal["accuracy"] - cav["accuracy"]) * 100
            if normal is not None
            else None
        )
        per_task[task] = {
            "beats_chain_of_draft_matched_budget": beats_cod_mb,
            "accuracy_drop_vs_normal_cot_pp": drop_pp,
            "significantly_hurts": drop_pp is not None and drop_pp > hurt_pp,
        }

    valid = [v for v in per_task.values() if v is not None]
    beats_any = any(v["beats_chain_of_draft_matched_budget"] for v in valid)
    hurts_any = any(v["significantly_hurts"] for v in valid)
    proceed = beats_any and not hurts_any
    return proceed, per_task


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
    parser.add_argument(
        "--hurt-threshold-pp",
        type=float,
        default=HURT_THRESHOLD_PP,
        help="accuracy-drop threshold (pp) for §18 'significantly hurt' check",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    budgets = {}
    bpath = Path(args.budgets)
    if bpath.exists():
        budgets = json.loads(bpath.read_text())

    strong_findings, strong, total = strong_positive_per_task(df)
    proceed, per_task = proceed_decision(df, hurt_pp=args.hurt_threshold_pp)
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

    lines.append("## Main Finding (§15 strong-positive per task)")
    lines.append("")
    lines.append(strong_findings)
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
    lines.append(f"§15 strong-positive tasks: {strong}/{total}.")
    lines.append("")
    lines.append("§18 proceed-gate per task:")
    for task, v in per_task.items():
        if v is None:
            lines.append(f"- {task}: no caveman_full data")
            continue
        drop = v["accuracy_drop_vs_normal_cot_pp"]
        drop_s = f"{drop:+.1f}pp" if drop is not None else "n/a"
        lines.append(
            f"- {task}: beats_cod_mb={v['beats_chain_of_draft_matched_budget']}, "
            f"drop_vs_normal_cot={drop_s}, "
            f"significantly_hurts={v['significantly_hurts']}"
        )
    lines.append("")
    decision = (
        "**Decision (§18): proceed** to PrOntoQA / ProofWriter / 32B / "
        "caveman_lite / caveman_ultra."
        if proceed
        else "**Decision (§18): do not proceed** to next stage; reframe as "
        "'structured symbolic scratchpads' rather than caveman reasoning."
    )
    lines.append(decision)
    lines.append("")
    lines.append(
        f"§18 rule: caveman_full beats chain_of_draft_matched_budget on at "
        f"least one task AND does not significantly hurt the other task "
        f"(operationalized as accuracy drop vs normal_cot > "
        f"{args.hurt_threshold_pp:.0f}pp)."
    )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
