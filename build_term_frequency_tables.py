"""Build per-arm (DAX vs Human) term-frequency tables from N independent
v2.3 runs on the same cohort.

A flag is included if it appears in >= MAJORITY_THRESHOLD of N runs at the
(source_note_id, normalized_term, category, bucket) granularity. Singleton
flags (in only 1 run) are excluded as run-to-run noise.

Outputs (to BIAS_NOTES_OUTPUT_DIR or --output-dir):
  - <prefix>_term_frequency_dax.csv         per-DAX-arm term counts
  - <prefix>_term_frequency_human.csv       per-Human-arm term counts
  - <prefix>_term_frequency_combined.csv    side-by-side counts
  - <prefix>_term_frequency_combined.md     reviewer-friendly markdown

Usage:
  uv run python build_term_frequency_tables.py \
      --output-prefix n500 \
      --majority 2 \
      <reviewer_adj_run1.csv> <reviewer_adj_run2.csv> <reviewer_adj_run3.csv>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

from bias_review_utils import normalize_term


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="Reviewer-adjudication CSV paths for each run (>= 2 required)",
    )
    parser.add_argument(
        "--majority",
        type=int,
        default=2,
        help="Minimum number of runs a flag must appear in (default 2)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs (default: $BIAS_NOTES_OUTPUT_DIR or ./output)",
    )
    parser.add_argument(
        "--output-prefix",
        default="term_frequency",
        help="Output filename prefix (default: term_frequency)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Number of rows to include in the markdown's combined view (default 60)",
    )
    return parser


def load_runs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for i, path in enumerate(paths, 1):
        df = pd.read_csv(path).fillna("")
        df["normalized_term"] = df["model_term"].astype(str).map(normalize_term)
        df["run"] = i
        frames.append(
            df[
                [
                    "source_note_id",
                    "Dax_or_Human",
                    "model_bucket",
                    "model_term",
                    "normalized_term",
                    "model_category",
                    "run",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def per_arm_table(majority_df: pd.DataFrame, arm: str) -> pd.DataFrame:
    sub = majority_df[majority_df["Dax_or_Human"] == arm]
    return (
        sub.groupby(["normalized_term", "model_category"])
        .agg(
            n_notes=("source_note_id", "nunique"),
            n_flags=("source_note_id", "size"),
            buckets=("model_bucket", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values(["n_notes", "n_flags"], ascending=False)
    )


def render_markdown(
    combined: pd.DataFrame,
    dax_tbl: pd.DataFrame,
    human_tbl: pd.DataFrame,
    majority_df: pd.DataFrame,
    n_runs: int,
    majority_threshold: int,
    top_n: int,
) -> str:
    lines = []
    lines.append(
        f"# Term Frequency Tables (N={n_runs} runs, majority threshold = {majority_threshold})\n"
    )
    lines.append(
        f"Methodology: {n_runs} independent v2.3 runs on the same cohort. A flag is "
        f"included here if it appeared in ≥ {majority_threshold} of {n_runs} runs at "
        f"the (note_id, normalized_term, category, bucket) granularity. Singleton "
        f"flags (in only 1 of {n_runs} runs) are excluded as run-to-run noise.\n"
    )

    n_dax_notes_flagged = (
        majority_df[majority_df["Dax_or_Human"] == "Dax"]
        .groupby("source_note_id")
        .size()
        .gt(0)
        .sum()
    )
    n_hum_notes_flagged = (
        majority_df[majority_df["Dax_or_Human"] == "Human"]
        .groupby("source_note_id")
        .size()
        .gt(0)
        .sum()
    )
    lines.append(f"- Total majority-confidence flags: **{len(majority_df)}**")
    lines.append(
        f"- DAX notes with at least one majority-confidence flag: **{n_dax_notes_flagged}**"
    )
    lines.append(
        f"- Human notes with at least one majority-confidence flag: **{n_hum_notes_flagged}**"
    )
    lines.append("\n---\n")

    lines.append(
        f"## Combined side-by-side (top {top_n}, sorted by total notes flagged)"
    )
    lines.append("")
    lines.append(
        "| Term | Category | DAX notes | Human notes | DAX flags | Human flags | DAX − Human |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in combined.head(top_n).iterrows():
        lines.append(
            f"| `{row['normalized_term']}` | {row['model_category']} | "
            f"{int(row['dax_notes'])} | {int(row['human_notes'])} | "
            f"{int(row['dax_flags'])} | {int(row['human_flags'])} | "
            f"{int(row['dax_minus_human_notes']):+d} |"
        )
    lines.append("\n---")

    lines.append("\n## DAX term frequency (top 40)")
    lines.append("")
    lines.append("| Term | Category | DAX notes | DAX flags | Buckets |")
    lines.append("|---|---|---:|---:|---|")
    for _, row in dax_tbl.head(40).iterrows():
        lines.append(
            f"| `{row['normalized_term']}` | {row['model_category']} | "
            f"{int(row['n_notes'])} | {int(row['n_flags'])} | {row['buckets']} |"
        )

    lines.append("\n## Human term frequency (top 40)")
    lines.append("")
    lines.append("| Term | Category | Human notes | Human flags | Buckets |")
    lines.append("|---|---|---:|---:|---|")
    for _, row in human_tbl.head(40).iterrows():
        lines.append(
            f"| `{row['normalized_term']}` | {row['model_category']} | "
            f"{int(row['n_notes'])} | {int(row['n_flags'])} | {row['buckets']} |"
        )

    lines.append("\n---\n")
    lines.append("## Asymmetric terms (in only one arm)")
    lines.append("")
    lines.append(
        "These are terms reviewers may want to inspect because they appeared exclusively in one arm."
    )
    lines.append("")
    dax_only = (
        combined[combined["human_notes"] == 0]
        .sort_values("dax_notes", ascending=False)
        .head(25)
    )
    hum_only = (
        combined[combined["dax_notes"] == 0]
        .sort_values("human_notes", ascending=False)
        .head(25)
    )

    lines.append("### DAX-only (top 25)")
    lines.append("")
    lines.append("| Term | Category | DAX notes |")
    lines.append("|---|---|---:|")
    for _, row in dax_only.iterrows():
        lines.append(
            f"| `{row['normalized_term']}` | {row['model_category']} | {int(row['dax_notes'])} |"
        )

    lines.append("\n### Human-only (top 25)")
    lines.append("")
    lines.append("| Term | Category | Human notes |")
    lines.append("|---|---|---:|")
    for _, row in hum_only.iterrows():
        lines.append(
            f"| `{row['normalized_term']}` | {row['model_category']} | {int(row['human_notes'])} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    if len(args.runs) < 2:
        print("Need at least 2 run CSVs.")
        return 2

    paths = [Path(p) for p in args.runs]
    for p in paths:
        if not p.exists():
            print(f"File not found: {p}")
            return 2

    output_dir = Path(
        args.output_dir or os.getenv("BIAS_NOTES_OUTPUT_DIR") or "./output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_raw = load_runs(paths)
    print(f"Loaded {len(combined_raw)} flag rows across {len(paths)} runs")

    agg = (
        combined_raw.groupby(
            [
                "source_note_id",
                "Dax_or_Human",
                "normalized_term",
                "model_category",
                "model_bucket",
            ]
        )
        .size()
        .reset_index(name="n_runs_flagged")
    )
    majority_df = agg[agg["n_runs_flagged"] >= args.majority].copy()
    print(
        f"Majority-confidence flags (>= {args.majority} of {len(paths)} runs): "
        f"{len(majority_df)} ({(agg['n_runs_flagged'] == 1).sum()} singletons excluded, "
        f"{(agg['n_runs_flagged'] == len(paths)).sum()} consensus across all runs)"
    )

    dax_tbl = per_arm_table(majority_df, "Dax")
    human_tbl = per_arm_table(majority_df, "Human")

    combined_view = pd.merge(
        dax_tbl[["normalized_term", "model_category", "n_notes", "n_flags"]].rename(
            columns={"n_notes": "dax_notes", "n_flags": "dax_flags"}
        ),
        human_tbl[["normalized_term", "model_category", "n_notes", "n_flags"]].rename(
            columns={"n_notes": "human_notes", "n_flags": "human_flags"}
        ),
        on=["normalized_term", "model_category"],
        how="outer",
    ).fillna(0)
    for col in ("dax_notes", "dax_flags", "human_notes", "human_flags"):
        combined_view[col] = combined_view[col].astype(int)
    combined_view["dax_minus_human_notes"] = (
        combined_view["dax_notes"] - combined_view["human_notes"]
    )
    combined_view["total_notes"] = (
        combined_view["dax_notes"] + combined_view["human_notes"]
    )
    combined_view = combined_view.sort_values(
        ["total_notes", "dax_minus_human_notes"], ascending=[False, False]
    )

    prefix = args.output_prefix
    dax_csv = output_dir / f"{prefix}_term_frequency_dax.csv"
    human_csv = output_dir / f"{prefix}_term_frequency_human.csv"
    combined_csv = output_dir / f"{prefix}_term_frequency_combined.csv"
    combined_md = output_dir / f"{prefix}_term_frequency_combined.md"

    dax_tbl.to_csv(dax_csv, index=False)
    human_tbl.to_csv(human_csv, index=False)
    combined_view.to_csv(combined_csv, index=False)
    md_text = render_markdown(
        combined_view,
        dax_tbl,
        human_tbl,
        majority_df,
        len(paths),
        args.majority,
        args.top_n,
    )
    combined_md.write_text(md_text)

    for f in (dax_csv, human_csv, combined_csv, combined_md):
        print(f"Wrote: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
