"""Run the shared Azure bias pipeline from a script without touching notebooks."""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

from bias_pipeline import (
    AzureBiasPipeline,
    AzureBiasPipelineConfig,
    generate_output_path,
    write_output_bundle,
)
from notebook_hygiene import format_issues, inspect_repo_notebooks
from project_config import (
    BIAS_NOTES_CACHE_PATH,
    BIAS_NOTES_CLEANED_CSV,
    BIAS_NOTES_OUTPUT_DIR,
    BIAS_NOTES_PROMPT_PATH,
    PROJECT_ROOT,
)

load_dotenv()

DEFAULT_OUTPUT_PREFIX = "bias_flagged_updated"
DEFAULT_CHART_COUNT = 100
DEFAULT_RANDOM_SEED = 42
DEFAULT_CHUNK_CHAR_LIMIT = 2800
DEFAULT_MAX_RETRIES = 5
DEFAULT_SLEEP_BASE_SEC = 1.4
DEFAULT_TEMPERATURE = 0
DEFAULT_PARALLEL_CALLS = 4
DEFAULT_GLOBAL_MAX_CALLS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--charts",
        type=int,
        default=DEFAULT_CHART_COUNT,
        help="Number of charts/notes to process when selection is not 'all'. Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible chart selection. Default: 42.",
    )
    parser.add_argument(
        "--selection",
        choices=("random", "head", "all"),
        default="random",
        help="How to select charts from the input CSV. Default: random.",
    )
    parser.add_argument(
        "--skip-notebook-check",
        action="store_true",
        help="Skip the notebook hygiene guard. By default, the runner verifies that repo notebooks are clean before and after the run.",
    )
    parser.add_argument(
        "--input-csv",
        default=BIAS_NOTES_CLEANED_CSV,
        help="Input CSV path. Defaults to BIAS_NOTES_CLEANED_CSV from .env.",
    )
    parser.add_argument(
        "--output-dir",
        default=BIAS_NOTES_OUTPUT_DIR,
        help="Output directory. Defaults to BIAS_NOTES_OUTPUT_DIR from .env.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output filename prefix. Default: bias_flagged_updated.",
    )
    parser.add_argument(
        "--prompt-path",
        default=BIAS_NOTES_PROMPT_PATH,
        help="Prompt file path. Defaults to BIAS_NOTES_PROMPT_PATH from .env.",
    )
    parser.add_argument(
        "--cache-path",
        default=BIAS_NOTES_CACHE_PATH,
        help="Cache CSV path. Defaults to BIAS_NOTES_CACHE_PATH from .env.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Model temperature. Default: 0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate notebook hygiene, load the CSV, and preview chart selection without calling Azure or writing outputs.",
    )
    return parser


def require_setting(value: str | None, message: str) -> str:
    if value is None or not str(value).strip():
        raise RuntimeError(message)
    return str(value).strip()


def get_model_for_api() -> str:
    model_for_api = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_MODEL_NAME")
    return require_setting(
        model_for_api,
        "No Azure deployment/model specified. Set AZURE_DEPLOYMENT or AZURE_MODEL_NAME in .env.",
    )


def create_client() -> AzureOpenAI:
    api_version = require_setting(
        os.getenv("AZURE_API_VERSION"), "Missing AZURE_API_VERSION in .env."
    )
    azure_endpoint = require_setting(
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        "Missing AZURE_OPENAI_ENDPOINT in .env.",
    )
    api_key = require_setting(
        os.getenv("AZURE_SUBSCRIPTION_KEY"),
        "Missing AZURE_SUBSCRIPTION_KEY in .env.",
    )
    return AzureOpenAI(
        api_version=api_version, azure_endpoint=azure_endpoint, api_key=api_key
    )


def select_rows(
    df_full: pd.DataFrame,
    charts_to_process: int,
    selection: str,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    total_rows = len(df_full)
    if selection == "all":
        return df_full.reset_index(drop=True), "all rows"

    n_to_process = min(charts_to_process, total_rows)
    if selection == "random" and n_to_process < total_rows:
        selected = df_full.sample(n=n_to_process, random_state=seed).reset_index(
            drop=True
        )
        return selected, f"random sample (seed={seed})"

    selected = df_full.head(n_to_process).reset_index(drop=True)
    if n_to_process < total_rows:
        return selected, "first N rows"
    return selected, "all rows"


def verify_notebooks_clean() -> None:
    issues = inspect_repo_notebooks(PROJECT_ROOT)
    if issues:
        raise RuntimeError(format_issues(issues, root=PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.charts < 1:
        parser.error("--charts must be at least 1.")

    input_csv = require_setting(
        args.input_csv,
        "No input CSV configured. Set BIAS_NOTES_CLEANED_CSV in .env or pass --input-csv.",
    )
    output_dir = require_setting(
        args.output_dir,
        "No output directory configured. Set BIAS_NOTES_OUTPUT_DIR in .env or pass --output-dir.",
    )
    prompt_path = require_setting(
        args.prompt_path,
        "No prompt path configured. Set BIAS_NOTES_PROMPT_PATH in .env or pass --prompt-path.",
    )
    if not args.skip_notebook_check:
        verify_notebooks_clean()

    df_full = pd.read_csv(input_csv)
    if "note_text" not in df_full.columns:
        raise ValueError("Input CSV must contain a 'note_text' column.")

    df, selection_mode = select_rows(df_full, args.charts, args.selection, args.seed)
    model_for_api = get_model_for_api() if not args.dry_run else "dry-run"

    if args.dry_run:
        print(f"{'=' * 60}")
        print("BIAS DETECTION — DRY RUN")
        print(f"{'=' * 60}")
        print(f"Input file: {input_csv}")
        print(f"Total rows in file: {len(df_full)}")
        print(f"Rows selected: {len(df)} ({selection_mode})")
        print(f"Selection mode: {args.selection}")
        print(f"Random seed: {args.seed}")
        if not args.skip_notebook_check:
            print("Notebook hygiene check: passed")
        preview_cols = [
            column
            for column in ("unique_id", "patient_study_id", "Dax_or_Human")
            if column in df.columns
        ]
        if preview_cols:
            print("\nSelected row preview:")
            print(df[preview_cols].head(min(len(df), 10)).to_string(index=False))
        else:
            print(
                "\nSelected row preview unavailable because no preview columns are present."
            )
        return 0

    output_csv = generate_output_path(output_dir, args.output_prefix)
    chunk_failure_log_path = (
        output_csv[:-4] + "_chunk_failures.jsonl"
        if output_csv.lower().endswith(".csv")
        else output_csv + "_chunk_failures.jsonl"
    )
    pipeline = AzureBiasPipeline(
        create_client(),
        AzureBiasPipelineConfig(
            prompt_path=prompt_path,
            model_for_api=model_for_api,
            temperature=args.temperature,
            chunk_char_limit=DEFAULT_CHUNK_CHAR_LIMIT,
            max_retries=DEFAULT_MAX_RETRIES,
            sleep_base_sec=DEFAULT_SLEEP_BASE_SEC,
            parallel_calls=DEFAULT_PARALLEL_CALLS,
            global_max_calls=DEFAULT_GLOBAL_MAX_CALLS,
            cache_path=args.cache_path,
            chunk_failure_log_path=chunk_failure_log_path,
            enable_second_pass_adjudication=True,
        ),
    )

    print(f"{'=' * 60}")
    print("BIAS DETECTION — SCRIPT RUNNER")
    print(f"{'=' * 60}")
    print(f"Input file: {input_csv}")
    print(f"Total rows in file: {len(df_full)}")
    print(f"Rows to process: {len(df)} ({selection_mode})")
    print(f"Selection mode: {args.selection}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {output_csv}")
    print(f"Chunk failure log: {chunk_failure_log_path}")
    print(f"Model: {model_for_api}")
    print(f"PARALLEL_CALLS={DEFAULT_PARALLEL_CALLS}")
    if args.skip_notebook_check:
        print("Notebook hygiene check: skipped")
    else:
        print("Notebook hygiene check: passed before run")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    processed = pipeline.process_dataframe(df)
    output_paths = write_output_bundle(processed, output_csv)
    total_time = time.time() - start_time

    if not args.skip_notebook_check:
        verify_notebooks_clean()

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Reviewer output: {output_paths['reviewer_path']}")
    print(f"Analysis-ready output: {output_paths['analysis_path']}")
    print(f"Reviewer adjudication output: {output_paths['adjudication_path']}")
    print(f"Chunk failure log: {chunk_failure_log_path}")
    print(f"Total notes: {len(processed)}")
    print(f"Possible bias flags kept: {int(processed['Possible_Bias_Count'].sum())}")
    print(f"Likely bias flags kept: {int(processed['Likely_Bias_Count'].sum())}")
    print(
        f"Notes with chunk failures: {int((processed['Chunk_Failure_Count'] > 0).sum())}"
    )
    print(
        f"Total time: {total_time:.1f}s ({total_time / len(processed):.1f}s per note)"
    )
    if not args.skip_notebook_check:
        print("Notebook hygiene check: passed after run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
