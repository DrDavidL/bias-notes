# Bias Notes - Clinical Note Bias Detection Pipeline

## Project Overview
Research project (Landsburg Society Research Award) comparing physician-written vs. AI-generated (Nuance DAX Copilot) clinical notes for stigmatizing language. Led by Drs. Asantewaa Ture & David Liebovitz at Northwestern University.

## Architecture & Pipeline

### Pipeline stages (in order):
1. **`pre_process_notes.ipynb`** - Loads raw CSV (`all_500_raw.csv`), removes DAX attestation phrases, outputs `all_500_cleaned.csv`
2. **`prepare_csv.ipynb`** - Azure OpenAI client setup, row selection, runs bias detection prompt against notes via Azure API. Outputs timestamped `bias_flagged_*.csv` with `Possible_Biased_Terms` and `Likely_Biased_Terms` columns
3. **`ollama_bias_assessment.ipynb`** - Same pipeline but using local Ollama models instead of Azure
4. **`bias_assessment.ipynb`** - Earlier/alternative version of the assessment notebook (Azure-based)
5. **`analyze_results.ipynb`** - Loads flagged output CSVs, computes detection rates by note type (Human vs DAX), shows term frequency breakdowns
6. **`main.py`** - Streamlit web app for interactive upload, manual review, LLM review, and comparison

### Key files:
- **`bias_detection_prompt.py`** - The core LLM prompt defining bias categories, term banks, matching rules, and expected JSON output format (`{"possible": [...], "likely": [...]}`)
- **`.env`** - Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT, AZURE_SUBSCRIPTION_KEY, AZURE_API_VERSION)

## Data
- Input data stored in OneDrive (not in repo): `all_500_raw.csv`, `all_500_cleaned.csv`
- Output CSVs go to `output/` directory
- CSV columns: `patient_study_id`, `age`, `author_type`, `author_title`, `note_text`, `Dax_or_Human`, `unique_id`
- All data must be de-identified per IRB requirements

## Development Notes
- `.gitignore` excludes `.ipynb`, `.csv`, `.xlsx` files from version control
- The `bias_detection_prompt.py` is the single source of truth for what constitutes biased language - both Azure and Ollama notebooks load it
- LLM output is JSON: `{"possible": ["term1", ...], "likely": ["term1", ...]}`
- Notes are chunked by sentence boundaries (default 2800 chars) before sending to the LLM
- Temperature should be 0 for deterministic extraction (except GPT-5 which requires 1)
- Use `uv` as the default Python workflow for this repo: `uv sync` for environment setup, `uv lock` to refresh locked versions, and `uv run ...` for scripts/tests/apps
- Prefer the `uv` workflow over ad hoc `pip install ...` or bare `python ...` commands because it gives better reproducibility, safer dependency management, and lower maintenance overhead across local runs and agent sessions
