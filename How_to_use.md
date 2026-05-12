# How To Use Bias-Notes

This guide explains how to (1) run the bias-detection pipeline on your own clinical notes and (2) generate a reviewer-evaluation harness that scores any future model run against a gold standard.

It is written for a researcher who wants to reuse this repo on their own data.

---

## 0. What this repo does

Given a CSV of clinical notes, the pipeline:

1. Sends each note (chunked by sentence boundaries) to an LLM with a curated prompt that defines stigmatizing-language categories and exception rules.
2. Returns JSON of `possible` and `likely` flagged phrases, each with one or more canonical bias-category labels.
3. Discards LLM hallucinations: any term the model emits that does not appear verbatim in its source chunk is dropped automatically.
4. Writes three CSVs per run:
   - `bias_flagged_<prefix>_<timestamp>.csv` — wide-format with all flagged terms per note (raw JSON columns).
   - `..._analysis_ready.csv` — same minus note text and detail JSON.
   - `..._reviewer_adjudication.csv` — long-format, one row per flagged term, with empty reviewer columns ready for manual review.

Two models are supported out of the box: any Azure-deployed model (production path) and any local Ollama model (offline experimentation).

---

## 1. Quick start

### Setup

```bash
uv sync                                  # install pinned dependencies
cp .env.example .env                     # then edit values
```

Required `.env` keys for Azure:

| Key | Purpose |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint |
| `AZURE_API_VERSION` | e.g. `2024-12-01-preview` |
| `AZURE_SUBSCRIPTION_KEY` | API key |
| `AZURE_DEPLOYMENT` | Deployment name (e.g. `gpt-4.1`) |
| `BIAS_NOTES_RAW_CSV` | Path to the raw, unprocessed note CSV |
| `BIAS_NOTES_CLEANED_CSV` | Where to write/read the cleaned note CSV |
| `BIAS_NOTES_OUTPUT_DIR` | Where flagged outputs land |
| `BIAS_NOTES_CACHE_PATH` *(optional)* | Note-hash cache for re-runs |

### Run the pipeline on 100 random notes

```bash
uv run python run_bias_batch.py \
    --selection random --charts 100 --seed 42 \
    --output-prefix my_run
```

Selection modes:
- `--selection random --seed N` — reproducible random sample.
- `--selection head` — first N rows (deterministic).
- `--selection all` — every row in the input CSV.

Useful flags:
- `--dry-run` — validate setup and preview the selection without calling Azure.
- `--temperature 0` — default; raises non-determinism if you change it.
- `--skip-notebook-check` — bypass the notebook-hygiene precondition.
- `--cache-path /path/to/cache.csv` — reuse prior chunk responses keyed by `(note_hash, prompt_version, model)`.

### Required CSV schema

The input CSV must contain at minimum:

- `unique_id` — integer or string, unique per note (used as the join key everywhere).
- `note_text` — the de-identified clinical note.
- `Dax_or_Human` — `"Dax"` or `"Human"` if you want the H-vs-DAX comparison; any string otherwise. Optional.

All other columns are passed through to the outputs unchanged.

---

## 2. Adapting to your own data

### A. Make a cleaned CSV

Open `pre_process_notes.ipynb`. It:
1. Loads `BIAS_NOTES_RAW_CSV`.
2. Strips template scaffolding (DAX attestations in our case; edit the regex list for your templates).
3. Repairs UTF-8 mojibake (`kg/mÃ¢Â²` → `kg/m²`) via `ftfy.fix_text` plus a targeted post-pass for clinical artifacts.
4. Writes `BIAS_NOTES_CLEANED_CSV`.

If your templates differ, edit only the `removal_patterns` list in the "Clean Notes and Save" cell.

### B. Adapt the prompt

`bias_detection_prompt.py` is the single source of truth for what counts as bias. It is a Python module whose `__doc__` is loaded at runtime — edit the docstring, save, re-run. No other code change required.

The prompt is organized as:

| Section | Purpose |
|---|---|
| TASK | LLM contract: input/output shape |
| BIAS CATEGORIES (1–9) | Each category with examples, exceptions, suggested replacements |
| ADDITIONAL SCOPE — LIKELY BIAS | 17 additional likely-bias patterns |
| SCOPE — POSSIBLE BIAS | When to flag as "possible" instead of "likely" |
| **MANDATORY FLAGS** | Hard list of phrases that *always* flag, overriding any exception |
| EXCLUSIONS | What to never flag (SDOH screening, screening-instrument items, neutral declines, etc.) |
| MATCHING RULES | Casing, plurals, identity-vs-adjective handling |
| OUTPUT | Strict JSON schema |
| TERM_BANK — LIKELY / POSSIBLE | Allowlists for category routing |

Recommended order of edits when adapting:
1. **Categories first** — define which bias categories matter for your domain. Edit the canonical category list at the top.
2. **TERM_BANK second** — add domain-specific stigma terms (case-insensitive variants are auto-matched).
3. **Exceptions third** — list patterns the LLM tends to over-flag in your data; phrase them as positive carve-outs.
4. **MANDATORY FLAGS last** — if an exception begins over-applying and dropping real stigma, list those stigma phrases here. Mandatory always wins ties.

After any edit, bump `PROMPT_VERSION` in `bias_pipeline.py` so the cache invalidates and the new prompt is recorded in every output row.

### C. Pick a model

- **For real studies, use a flagship model** (gpt-4.1, gpt-4o, Claude Sonnet 4.x). On this project, gpt-4o-mini produced ~40% hallucinated terms; gpt-4.1 produced ~0%.
- **For development/iteration**, gpt-4o-mini is fine — fast and cheap. Just don't trust its outputs as gold.
- The hallucination guard runs unconditionally and protects against the worst failure mode regardless of model.

### D. Run

```bash
uv run python run_bias_batch.py --selection random --charts 100 --seed 42 --output-prefix my_v1
```

Outputs go to `$BIAS_NOTES_OUTPUT_DIR/bias_flagged_my_v1_<timestamp>.csv` (+ analysis-ready and reviewer-adjudication variants).

---

## 3. Setting up an evaluation harness

The same data structure that powers the reviewer-adjudication export also lets you score future runs against a gold standard. This is how you measure prompt and model changes objectively.

### A. Build the gold standard

1. Run the pipeline once and open the `..._reviewer_adjudication.csv`. Each row is one flagged term per note.
2. Have one or more domain experts fill in:
   - **`false_positive_term`** — write `x` if the model was wrong to flag this term.
   - **`reviewer_notes`** — free-text rationale, especially for edge cases.
   - **`missed_term`** — write the verbatim term if the reviewer spotted stigma the model missed in the same note (false negative). One row per missed term.
3. Save the reviewed file as an `.xlsx` (or stay in CSV).

That file is now your gold standard.

### B. Run the eval notebook

`eval_against_gold.ipynb` already does the scoring. To reuse it:

1. Set `GOLD_PATH` in cell 1 to your reviewed file.
2. Set `CANDIDATE_CSV` in cell 1 to the flagged-output CSV you want to evaluate. Set to `None` to run a sanity check against the gold set itself.
3. Run all cells. You get:
   - Overall precision, FP rate, and recall on missed terms.
   - Per-category precision (which categories the model gets right vs. wrong).
   - Per-(term, category) FP rate — useful for pruning the TERM_BANK.
   - Dax-vs-Human (or any binary stratifier) FP-rate parity.
   - A one-line `summary` dict suitable for a run log.

The match key is `(source_note_id, normalized_term, model_category)`. If your CSV uses a different note-ID column, change `_get_note_id_column` in `bias_pipeline.py` or pass the column name explicitly.

### C. Verifiability filter (recommended)

Older gold standards in this project contained reviewer judgments on terms the original model had hallucinated. To get an honest baseline, filter the gold set down to rows where the `model_term` actually appears verbatim in its source note. The `eval_against_gold.ipynb` notebook does this implicitly because new candidate runs are now hallucination-guarded; for older gold sets, add this cell before scoring:

```python
notes = pd.read_csv(BIAS_NOTES_CLEANED_CSV).set_index('unique_id')
def in_note(row):
    txt = str(notes.loc[row['source_note_id'], 'note_text']).lower()
    return str(row['model_term']).lower() in txt or str(row['normalized_term']).lower() in txt
gold = gold[gold.apply(in_note, axis=1)]
```

Any new run produced by this version of the pipeline is already verifiable by construction.

### D. Recommended run log

For every model/prompt combination you try, keep one line in a CSV like:

```
date,prompt_version,model,n_notes,total_flags,matched,tp,fp,precision,fp_rate,dax_fp,human_fp,gap,admits_recall
2026-05-11,v2.1,gpt-4.1,100,150,56,48,8,0.857,0.143,0.133,0.154,-0.021,5/5
```

Lines come straight from the `summary` dict at the end of `eval_against_gold.ipynb`.

---

## 4. Architecture reference

| File | Purpose |
|---|---|
| `bias_detection_prompt.py` | The LLM prompt as a Python module docstring. Single source of truth for bias categories, terms, exceptions, and mandatory flags. |
| `bias_pipeline.py` | Core engine: chunking, parallel Azure calls, JSON parsing, hallucination guard, second-pass adjudication, output formatting. |
| `run_bias_batch.py` | CLI entry point. Loads `.env`, selects notes, instantiates `AzureBiasPipeline`, writes outputs. |
| `pre_process_notes.ipynb` | DAX-attestation stripping and mojibake repair → cleaned CSV. |
| `prepare_csv.ipynb` / `bias_assessment.ipynb` | Notebook variants of the script pipeline (for demos). |
| `ollama_bias_assessment.ipynb` | Offline Ollama variant. |
| `eval_against_gold.ipynb` | Scores any flagged-output CSV against a reviewer-adjudicated gold set. |
| `analyze_results.ipynb` / `analyze_results.py` | Term-frequency and rate-by-source analyses. |
| `main.py` | Streamlit app for interactive review. |
| `bias_review_utils.py` | Category canonicalization and per-term enrichment helpers. |
| `tests/` | Unit tests for the pipeline, including the hallucination guard. |

### The hallucination guard

Implemented in `bias_pipeline.py` as `filter_hallucinated_terms`. Runs after every LLM chunk response, before caching. Drops any flagged term whose text does not appear in the chunk after case- and whitespace-normalization (plus a parenthetical-head fallback). Counts are reported at the end of each run; set `BIAS_HALLUCINATION_GUARD_VERBOSE=1` to log each drop.

### What `PROMPT_VERSION` does

Embedded into every output row. Used as part of the cache key, so changing the prompt forces a re-run on the same notes. Also used by `analyze_results.py` to filter outputs to a specific prompt revision when comparing runs.

---

## 5. Reproducing the canonical run

The canonical 100-note study uses:

- `--selection random --seed 42 --charts 100`
- Model: gpt-4.1 (via `AZURE_DEPLOYMENT=gpt-4.1`)
- Prompt: `2026-05-11-reviewer-feedback-v2.1`

To reproduce:

```bash
uv run python run_bias_batch.py \
    --skip-notebook-check \
    --output-prefix bias_flagged_v21_gpt41 \
    --selection random --seed 42 --charts 100
```

Then open `eval_against_gold.ipynb`, set `CANDIDATE_CSV` to the new output, run all cells.
