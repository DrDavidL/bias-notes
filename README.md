# Bias Notes

Clinical note bias-detection workflow for comparing physician-written vs. AI-generated notes for stigmatizing language.

## Purpose

This repo contains:

- preprocessing for note cleanup
- Azure-based bias detection workflows
- an optional Ollama workflow for local experiments
- result analysis utilities
- reviewer-friendly adjudication exports
- a Streamlit app for interactive review

The core detection logic lives in `bias_detection_prompt.py` and the shared execution engine lives in `bias_pipeline.py`.

## Security Model

- Do not store note CSVs, XLSX files, or secrets in the repo.
- Keep local paths in `.env`, not in notebooks or committed Python files.
- `.env` is ignored by git.
- Notebooks are committed in a clean state with no outputs, execution counts, or local directory paths.
- Use the analysis-ready and reviewer-adjudication exports when you want to remove confidential note text from downstream review files.

## Setup

1. Create a local `.env` from `.env.example`.
2. Set Azure OpenAI credentials:
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_API_VERSION`
   - `AZURE_SUBSCRIPTION_KEY`
   - `AZURE_DEPLOYMENT`
3. Set local secure file locations:
   - `BIAS_NOTES_RAW_CSV`
   - `BIAS_NOTES_CLEANED_CSV`
   - `BIAS_NOTES_OUTPUT_DIR`
   - optional: `BIAS_NOTES_PROMPT_PATH`
   - optional: `BIAS_NOTES_CACHE_PATH`
   - optional: `BIAS_NOTES_RESULTS_FILE`
4. Install dependencies with `uv`:

```bash
uv sync
```

Optional:

- Create or refresh the lockfile with `uv lock`
- Run scripts with `uv run ...`
- If you still need a `requirements.txt` export for another environment, generate it from the lockfile instead of editing it manually

```bash
uv export --format requirements-txt -o requirements.txt
```

## Changing The Azure Hosted Model

The shared Azure pipeline reads the hosted model from env:

- primary: `AZURE_DEPLOYMENT`
- fallback: `AZURE_MODEL_NAME`

To switch to a different hosted Azure model or deployment:

1. Update `AZURE_DEPLOYMENT` in your local `.env`
2. Rerun the script, notebook, or app
3. Review the output metadata columns:
   - `Model_Used`
   - `Prompt_Version`
   - `Pipeline_Version`

Notes:

- For GPT-5-family deployments, some workflows may require `temperature=1`
- The Streamlit app reads its Azure deployment from `st.secrets["azure_deployment"]`
- Persistent cache reuse is version- and model-aware, so changing the deployment will not silently reuse stale cached note results from a different model

## Recommended Workflow

1. Preprocess notes:
   - Use `pre_process_notes.ipynb` to remove DAX attestation language from the raw CSV.
2. Run Azure bias detection:
   - Use `uv run python run_bias_batch.py` for the shared batch pipeline that runs fully outside notebooks.
   - Set `--charts` to the number of charts you want to process and `--seed` to keep random selection reproducible.
   - The legacy `run_100_notes.py` entry point now forwards to the same script-first runner.
   - Or use `prepare_csv.ipynb` / `bias_assessment.ipynb` if notebook execution is preferred.
3. Review outputs:
   - reviewer-facing CSV: full result file with note text
   - analysis-ready CSV: drops note text and detailed context
   - reviewer-adjudication CSV: one row per flagged phrase for manual review
4. Analyze:
   - Use `analyze_results.py` or `analyze_results.ipynb`
5. Adjudicate:
   - Review the long-format adjudication CSV
   - Record keep/drop/move/missed-term decisions

## Main Entry Points

### Batch runner

`run_bias_batch.py`

- uses the shared Azure pipeline
- does not modify or execute `.ipynb` files
- checks that repo notebooks are blank before and after the run unless `--skip-notebook-check` is used
- lets you choose how many charts/notes to sample
- lets you set the random seed explicitly for reproducible chart selection
- supports persistent cache reuse across runs
- writes:
  - reviewer CSV
  - analysis-ready CSV
  - reviewer adjudication CSV

Example:

```bash
uv run python run_bias_batch.py --charts 10 --seed 42
```

Preview the exact 10-chart selection without calling Azure:

```bash
uv run python run_bias_batch.py --charts 10 --seed 42 --dry-run
```

Compatibility alias:

```bash
uv run python run_100_notes.py --charts 10 --seed 42
```

### Analysis CLI

`analyze_results.py`

- computes note-level rates
- reports flags per 1,000 words
- summarizes terms and categories
- compares Human vs Dax when `Dax_or_Human` is present

Example:

```bash
uv run python analyze_results.py /secure/path/to/results_analysis_ready.csv
```

### Reviewer export converter

`export_reviewer_adjudication.py`

Use this when you already have a historical results CSV and want the newer reviewer-friendly long-format export without rerunning the model.

```bash
uv run python export_reviewer_adjudication.py /secure/path/to/results.csv
```

### Streamlit app

`main.py`

- uses the shared Azure pipeline for LLM review
- supports manual review plus comparison against model output
- uses a local cache file under `output/`

Example:

```bash
uv run streamlit run main.py
```

### Ollama local workflow

`ollama_bias_assessment.ipynb`

Use this notebook only when local hosting is appropriate under your data-handling rules.

To switch from Azure to Ollama/local hosting:

1. Install and start Ollama locally
2. Pull a supported local model, for example:

```bash
ollama pull llama3.1:8b
```

3. Open `ollama_bias_assessment.ipynb`
4. Set or confirm:
   - `OLLAMA_BASE_URL`
   - `OLLAMA_MODEL`
   - `INPUT_CSV`
   - `OUTPUT_DIR`
   - `PROMPT_PATH`
5. Run the notebook

Important differences:

- The Ollama workflow is separate from the shared Azure pipeline
- It does not use Azure credentials
- It is intended for local experiments, not the default protected-data research path
- If you want full parity with the shared Azure pipeline over time, keep validating Ollama outputs against the adjudication workflow

## Shared Configuration

`project_config.py` is the single place where env-backed local paths are loaded.

Defaults:

- `BIAS_NOTES_OUTPUT_DIR` defaults to `./output`
- `BIAS_NOTES_PROMPT_PATH` defaults to `./bias_detection_prompt.py`
- `BIAS_NOTES_CACHE_PATH` defaults to `./output/bias_detection_cache.csv`

Required for most real runs:

- `BIAS_NOTES_CLEANED_CSV`

Required for preprocessing:

- `BIAS_NOTES_RAW_CSV`

## Outputs

The shared pipeline writes these columns into the reviewer-facing results:

- `Possible_Biased_Terms`
- `Likely_Biased_Terms`
- `Possible_Biased_Terms_Normalized`
- `Likely_Biased_Terms_Normalized`
- `Possible_Bias_Categories`
- `Likely_Bias_Categories`
- `Possible_Bias_Details`
- `Likely_Bias_Details`
- `Possible_Bias_Count`
- `Likely_Bias_Count`
- `Note_Word_Count`
- `Prompt_Version`
- `Pipeline_Version`
- `Model_Used`
- `Note_Hash`

The reviewer-adjudication export flattens phrase-level review into columns such as:

- `source_note_id`
- `model_bucket`
- `model_term`
- `normalized_term`
- `model_category`
- `term_context`
- `review_decision`
- `reviewed_term`
- `reviewed_category`
- `reviewer_notes`
- `false_positive_term`
- `missed_term`
- `needs_prompt_update`
- `needs_rule_update`

## Testing

Run:

```bash
uv run python -m unittest discover -s tests
```

The tests cover:

- sentence chunking
- false-positive filtering
- reviewer export generation
- analysis summaries
- persistent cache reuse

## Git Hygiene

Safe to commit:

- cleaned notebooks
- Python code
- tests
- `.env.example`

Do not commit:

- `.env`
- note data files
- exported result CSVs
- notebook outputs with note content

Before pushing, confirm:

- notebooks have no outputs or execution counts
- `uv run python check_notebooks_clean.py` passes
- no absolute local paths appear in committed files
- no confidential CSV/XLSX files are staged

## Git Hooks

The repo includes a versioned pre-push hook at `.githooks/pre-push` that blocks pushes when notebooks are dirty.

Enable it once per clone:

```bash
uv run python scripts/install_git_hooks.py
```

After that, every `git push` will run the notebook hygiene check automatically.

## Notes

- Azure-based workflows are the intended research path for protected note content.
- The Ollama notebook is retained for local experimentation and should only be used when appropriate under your data handling rules.
