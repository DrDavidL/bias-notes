"""Create a reviewer-friendly adjudication CSV from an existing results file."""

import argparse
from pathlib import Path

import pandas as pd

from bias_pipeline import build_reviewer_adjudication_dataframe


def default_output_path(results_path: Path) -> Path:
    if results_path.suffix.lower() == ".csv":
        return results_path.with_name(results_path.stem + "_reviewer_adjudication.csv")
    return Path(str(results_path) + "_reviewer_adjudication.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_file", help="Path to an existing bias results CSV.")
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Defaults to <results>_reviewer_adjudication.csv.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    output_path = Path(args.output) if args.output else default_output_path(results_path)

    df = pd.read_csv(results_path)
    adjudication_df = build_reviewer_adjudication_dataframe(df)
    adjudication_df.to_csv(output_path, index=False)

    print(f"Loaded results: {results_path}")
    print(f"Wrote reviewer adjudication CSV: {output_path}")
    print(f"Rows written: {len(adjudication_df)}")


if __name__ == "__main__":
    main()
