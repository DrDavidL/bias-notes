"""Run bias detection on 100 randomly selected notes using the shared Azure pipeline."""

import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

from project_config import (
    BIAS_NOTES_CACHE_PATH,
    BIAS_NOTES_CLEANED_CSV,
    BIAS_NOTES_OUTPUT_DIR,
    BIAS_NOTES_PROMPT_PATH,
)
from bias_pipeline import (
    AzureBiasPipeline,
    AzureBiasPipelineConfig,
    generate_output_path,
    write_output_bundle,
)

load_dotenv()

# ---------- AZURE OPENAI CLIENT ----------
client = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
)

# ---------- CONFIG ----------
INPUT_CSV = BIAS_NOTES_CLEANED_CSV
OUTPUT_DIR = BIAS_NOTES_OUTPUT_DIR
OUTPUT_PREFIX = "bias_flagged_updated"
PROMPT_PATH = BIAS_NOTES_PROMPT_PATH
CACHE_PATH = BIAS_NOTES_CACHE_PATH

N_ROWS_TO_PROCESS = 100
RANDOM_SELECTION = True
RANDOM_SEED = 42

CHUNK_CHAR_LIMIT = 2800
MAX_RETRIES = 5
SLEEP_BASE_SEC = 1.4
TEMPERATURE = 0
PARALLEL_CALLS = 4
GLOBAL_MAX_CALLS = 8

MODEL_FOR_API = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_MODEL_NAME")
if not MODEL_FOR_API:
    raise RuntimeError("No Azure deployment/model specified.")
if not INPUT_CSV:
    raise RuntimeError("No input CSV configured. Set BIAS_NOTES_CLEANED_CSV in .env.")
if not OUTPUT_DIR:
    raise RuntimeError("No output directory configured. Set BIAS_NOTES_OUTPUT_DIR in .env.")
if not PROMPT_PATH:
    raise RuntimeError("No prompt path configured. Set BIAS_NOTES_PROMPT_PATH in .env.")

OUTPUT_CSV = generate_output_path(OUTPUT_DIR, OUTPUT_PREFIX)


def select_rows(df_full: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    total_rows = len(df_full)
    n_to_process = min(N_ROWS_TO_PROCESS, total_rows)
    if RANDOM_SELECTION and n_to_process < total_rows:
        selected = df_full.sample(n=n_to_process, random_state=RANDOM_SEED).reset_index(drop=True)
        return selected, f"random sample (seed={RANDOM_SEED})"
    selected = df_full.head(n_to_process).reset_index(drop=True)
    selection_mode = "first N rows" if n_to_process < total_rows else "all rows"
    return selected, selection_mode


if __name__ == "__main__":
    df_full = pd.read_csv(INPUT_CSV)
    if "note_text" not in df_full.columns:
        raise ValueError("Input CSV must contain a 'note_text' column.")

    df, selection_mode = select_rows(df_full)
    pipeline = AzureBiasPipeline(
        client,
        AzureBiasPipelineConfig(
            prompt_path=PROMPT_PATH,
            model_for_api=MODEL_FOR_API,
            temperature=TEMPERATURE,
            chunk_char_limit=CHUNK_CHAR_LIMIT,
            max_retries=MAX_RETRIES,
            sleep_base_sec=SLEEP_BASE_SEC,
            parallel_calls=PARALLEL_CALLS,
            global_max_calls=GLOBAL_MAX_CALLS,
            cache_path=CACHE_PATH,
            enable_second_pass_adjudication=True,
        ),
    )

    print(f"{'=' * 60}")
    print("BIAS DETECTION — SHARED AZURE PIPELINE")
    print(f"{'=' * 60}")
    print(f"Input file: {INPUT_CSV}")
    print(f"Total rows in file: {len(df_full)}")
    print(f"Rows to process: {len(df)} ({selection_mode})")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Model: {MODEL_FOR_API}")
    print(f"PARALLEL_CALLS={PARALLEL_CALLS}")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    processed = pipeline.process_dataframe(df)
    output_paths = write_output_bundle(processed, OUTPUT_CSV)
    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Reviewer output: {output_paths['reviewer_path']}")
    print(f"Analysis-ready output: {output_paths['analysis_path']}")
    print(f"Reviewer adjudication output: {output_paths['adjudication_path']}")
    print(f"Total notes: {len(processed)}")
    print(f"Possible bias flags kept: {int(processed['Possible_Bias_Count'].sum())}")
    print(f"Likely bias flags kept: {int(processed['Likely_Bias_Count'].sum())}")
    print(f"Total time: {total_time:.1f}s ({total_time / len(processed):.1f}s per note)")
