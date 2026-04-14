"""Analyze bias-detection output files without displaying note text."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_json_list(value) -> List:
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def get_series_list(df: pd.DataFrame, column: str) -> List[List]:
    if column not in df.columns:
        return [[] for _ in range(len(df))]
    return [parse_json_list(value) for value in df[column]]


def ensure_word_count(df: pd.DataFrame) -> pd.Series:
    if "Note_Word_Count" in df.columns:
        counts = pd.to_numeric(df["Note_Word_Count"], errors="coerce").fillna(0).astype(int)
        if counts.sum() > 0:
            return counts
    if "note_text" in df.columns:
        return df["note_text"].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    return pd.Series([0] * len(df), index=df.index, dtype=int)


def print_rate_block(label: str, subset: pd.DataFrame) -> None:
    total_notes = len(subset)
    if total_notes == 0:
        print(f"\n{label}: no rows")
        return

    possible_note_rate = 100 * subset["has_possible"].mean()
    likely_note_rate = 100 * subset["has_likely"].mean()
    any_note_rate = 100 * subset["has_any_bias"].mean()
    possible_flags = int(subset["Possible_Bias_Count"].sum())
    likely_flags = int(subset["Likely_Bias_Count"].sum())
    total_words = int(subset["Note_Word_Count"].sum())
    words_per_1k = total_words / 1000 if total_words else 0
    possible_per_1k = possible_flags / words_per_1k if words_per_1k else 0
    likely_per_1k = likely_flags / words_per_1k if words_per_1k else 0

    print(f"\n{label}")
    print("-" * len(label))
    print(f"Notes: {total_notes}")
    print(f"Possible note rate: {possible_note_rate:.1f}%")
    print(f"Likely note rate:   {likely_note_rate:.1f}%")
    print(f"Any-bias note rate: {any_note_rate:.1f}%")
    print(f"Avg possible flags/note: {possible_flags / total_notes:.2f}")
    print(f"Avg likely flags/note:   {likely_flags / total_notes:.2f}")
    print(f"Possible flags / 1k words: {possible_per_1k:.2f}")
    print(f"Likely flags / 1k words:   {likely_per_1k:.2f}")


def count_terms(series_of_lists: Iterable[List], top_n: int) -> List[tuple]:
    counter = Counter()
    for items in series_of_lists:
        counter.update(items)
    return counter.most_common(top_n)


def count_note_prevalence(series_of_lists: Iterable[List], top_n: int) -> List[tuple]:
    counter = Counter()
    for items in series_of_lists:
        counter.update(set(items))
    return counter.most_common(top_n)


def print_top_counter(title: str, rows: List[tuple]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("No data")
        return
    for term, count in rows:
        print(f"{count:4d}x  {term}")


def print_category_prevalence_block(title: str, series_of_lists: Iterable[List], total_notes: int, top_n: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if total_notes == 0:
        print("No data")
        return

    rows = count_note_prevalence(series_of_lists, top_n)
    if not rows:
        print("No data")
        return

    for category, note_count in rows:
        rate = 100 * note_count / total_notes
        print(f"{note_count:4d} notes  ({rate:5.1f}%)  {category}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_file", help="Path to a results CSV produced by the bias pipeline.")
    parser.add_argument("--top-n", type=int, default=15, help="How many top terms/categories to show.")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    df = pd.read_csv(results_path)

    df["Possible_Bias_Count"] = (
        pd.to_numeric(df.get("Possible_Bias_Count", 0), errors="coerce").fillna(0).astype(int)
    )
    df["Likely_Bias_Count"] = (
        pd.to_numeric(df.get("Likely_Bias_Count", 0), errors="coerce").fillna(0).astype(int)
    )
    df["Note_Word_Count"] = ensure_word_count(df)
    df["has_possible"] = df["Possible_Bias_Count"] > 0
    df["has_likely"] = df["Likely_Bias_Count"] > 0
    df["has_any_bias"] = df["has_possible"] | df["has_likely"]

    possible_terms = get_series_list(df, "Possible_Biased_Terms_Normalized")
    if not any(possible_terms):
        possible_terms = get_series_list(df, "Possible_Biased_Terms")
    likely_terms = get_series_list(df, "Likely_Biased_Terms_Normalized")
    if not any(likely_terms):
        likely_terms = get_series_list(df, "Likely_Biased_Terms")

    possible_categories = get_series_list(df, "Possible_Bias_Categories")
    likely_categories = get_series_list(df, "Likely_Bias_Categories")

    print(f"Results file: {results_path}")
    print(f"Rows loaded: {len(df)}")
    if "Prompt_Version" in df.columns:
        versions = sorted({str(value) for value in df["Prompt_Version"].dropna().unique()})
        print(f"Prompt version(s): {', '.join(versions)}")
    if "Pipeline_Version" in df.columns:
        versions = sorted({str(value) for value in df["Pipeline_Version"].dropna().unique()})
        print(f"Pipeline version(s): {', '.join(versions)}")
    if "Model_Used" in df.columns:
        versions = sorted({str(value) for value in df["Model_Used"].dropna().unique()})
        print(f"Model(s): {', '.join(versions)}")

    print_rate_block("Overall", df)

    if "Dax_or_Human" in df.columns:
        for note_type in sorted(df["Dax_or_Human"].dropna().unique()):
            print_rate_block(f"{note_type} notes", df[df["Dax_or_Human"] == note_type])

    print_top_counter(
        f"Top {args.top_n} likely terms",
        count_terms(likely_terms, args.top_n),
    )
    print_top_counter(
        f"Top {args.top_n} possible terms",
        count_terms(possible_terms, args.top_n),
    )
    print_top_counter(
        f"Top {args.top_n} likely categories",
        count_terms(likely_categories, args.top_n),
    )
    print_top_counter(
        f"Top {args.top_n} possible categories",
        count_terms(possible_categories, args.top_n),
    )
    print_category_prevalence_block(
        f"Top {args.top_n} likely category note prevalence",
        likely_categories,
        len(df),
        args.top_n,
    )
    print_category_prevalence_block(
        f"Top {args.top_n} possible category note prevalence",
        possible_categories,
        len(df),
        args.top_n,
    )

    if "Dax_or_Human" in df.columns:
        for note_type in sorted(df["Dax_or_Human"].dropna().unique()):
            subset = df[df["Dax_or_Human"] == note_type]
            print_category_prevalence_block(
                f"{note_type} likely category note prevalence",
                get_series_list(subset, "Likely_Bias_Categories"),
                len(subset),
                args.top_n,
            )
            print_category_prevalence_block(
                f"{note_type} possible category note prevalence",
                get_series_list(subset, "Possible_Bias_Categories"),
                len(subset),
                args.top_n,
            )

    print("\nSample flagged rows")
    print("------------------")
    display_cols = []
    for column in (
        "unique_id",
        "Dax_or_Human",
        "Likely_Bias_Categories",
        "Likely_Biased_Terms",
        "Possible_Bias_Categories",
        "Possible_Biased_Terms",
    ):
        if column in df.columns:
            display_cols.append(column)
    flagged = df[df["has_any_bias"]]
    if flagged.empty:
        print("No flagged rows")
    else:
        print(flagged[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
