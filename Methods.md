# Methods

This document describes the end-to-end methodology used to detect stigmatizing language in clinical notes for the Landsburg Society Research Award study comparing physician-written and AI-generated (Nuance DAX Copilot) notes. It is intended to be detailed enough for an independent group to reproduce the analysis.

Every step below is annotated with the exact file and function (or notebook cell) in this repository that implements it.

---

## 1. Study cohort and data preparation

### 1.1 Input data

- **Source corpus:** `all_500_raw.csv` — 500 de-identified ambulatory clinical notes, paired by encounter, with one physician-authored note and one DAX-generated note per patient.
- **Required columns:** `patient_study_id`, `age`, `author_type`, `author_title`, `note_text`, `Dax_or_Human`, `unique_id`.
- **PHI handling:** all data is de-identified per the IRB protocol. Raw CSVs are stored in institutional OneDrive and never committed to the repository; paths are read from environment variables (`BIAS_NOTES_RAW_CSV`, `BIAS_NOTES_CLEANED_CSV`, `BIAS_NOTES_OUTPUT_DIR`) declared in `project_config.py`.

### 1.2 Pre-processing

Implemented in `pre_process_notes.ipynb`. Steps in order:

1. **Load raw CSV** from `$BIAS_NOTES_RAW_CSV`.
2. **Strip vendor scaffolding** from DAX notes via the `removal_patterns` regex list — primarily the DAX attestation phrase ("This note was generated using AI...") and section labels reserved for software stamping. Physician notes are passed through unchanged.
3. **Repair UTF-8 mojibake.** A `ftfy.fix_text` pass corrects double-encoded glyphs (e.g., `kg/mÃ¢Â²` → `kg/m²`). A targeted post-pass regex `re.compile(r'kg\s*/\s*m[âÂ]+\s*²')` handles a clinical-specific doubly-corrupted artifact that `ftfy` alone misses.
4. **Write cleaned CSV** to `$BIAS_NOTES_CLEANED_CSV`.

### 1.3 Sampling

For the model-development cohort (n=100), the pipeline draws a fixed random sample of 100 notes from the cleaned CSV using `random.Random(42).sample(...)` in `run_bias_batch.py` (`--selection random --seed 42 --charts 100`). The full-corpus run uses `--selection all`. Selection is deterministic given the seed, so the same seed reproduces the same 100 notes.

---

## 2. Bias-detection prompt

### 2.1 Single source of truth

`bias_detection_prompt.py` is a Python module whose top-level docstring **is** the prompt. The pipeline loads it at runtime by reading `__doc__` from the imported module (`load_prompt_text` in `bias_pipeline.py:189`). All notebooks and scripts share the same prompt; there is no separate copy.

### 2.2 Prompt structure

The prompt is organized into the following sections, in the order the model sees them:

| Section | Purpose |
|---|---|
| TASK | I/O contract: input is a single note chunk, output is a JSON object with `possible` and `likely` arrays. |
| CANONICAL CATEGORY LABELS | Closed list of 25 bias-category strings. The model must use one of these in each term's `categories` array. |
| BIAS CATEGORIES (9 detailed) | Substance identity, condition identity, weight-based identity, moralizing lifestyle, effort-based, outcome-judging, substance-use judgment, attitude/behavior, plus 17 additional likely-bias patterns (ageism, classism, credibility-doubting, stereotyping, etc.). Each category lists positive examples, negative exceptions, and a suggested non-stigmatizing alternative. |
| MANDATORY FLAGS | Phrases that ALWAYS flag regardless of any exception elsewhere in the prompt. Added in v2.1 to correct over-application of exceptions by gpt-4.1 (e.g., `elderly`, `non-compliant`, `uncontrolled diabetic`, identity-form `obese`/`diabetic`/`smoker`, `admits to`, `misuse of tobacco`). The hierarchy is explicit: mandatory > exceptions > general rules. |
| MANDATORY NEVER-FLAG | **Top-priority exclusions** (added in v2.3) that override every other rule including MANDATORY FLAGS. Currently lists EHR-template / auto-calculator artifacts: `failed to calculate`, `diabetic: yes`/`diabetic: no` (any spacing), `unable to calculate`, `not calculated`. These are explicitly skipped under any category or bucket. A complementary deterministic deny-list filter in `bias_pipeline.py` (§4b) enforces this rule even when the model fails to comply. |
| EXCLUSIONS | What never to flag. Includes neutral charting verbs, SDOH screening taxonomy, validated screening-instrument items (PHQ, GAD, AUDIT items), bare diagnosis names in problem lists, objective weight/BMI measurements, neutral substance-use type/quantity descriptors, neutral vaccine refusal. |
| MATCHING RULES | Case-insensitive matching, plural/hyphen variants, identity-vs-adjective handling, conservative-by-default ("when ambiguous, prefer `possible` over `likely`"), do-not-flag-`normal` rule. |
| OUTPUT | Strict JSON schema repeated for emphasis. |
| TERM_BANK — LIKELY | ~80 stigmatizing phrases pre-classified as likely. |
| TERM_BANK — POSSIBLE | ~25 context-dependent phrases pre-classified as possible. |

### 2.3 Versioning

A constant `PROMPT_VERSION` in `bias_pipeline.py:21` is embedded into every output row and is part of the disk-cache key. Any prompt change requires bumping this string so that cached chunk responses from older prompt versions cannot be reused.

The version used for the publication-ready run is `2026-05-12-reviewer-feedback-v2.3-template-exclusions-filter`.

### 2.4 Iteration history (summarized)

| Version | Key change | Outcome |
|---|---|---|
| v1 | Initial term-bank + category prompt | gpt-4o-mini precision 0.413 on verifiable subset; 40% of model-emitted terms were hallucinated. |
| v2 | Reviewer-driven exception additions; switched model to gpt-4.1 | Eliminated nearly all hallucinations, but new model over-applied exceptions and missed canonical stigma (e.g., 0/29 recall on `elderly`). |
| v2.1 | Added MANDATORY FLAGS hierarchy | Restored recall on `elderly`, `admits to`, identity-form `obese`/`diabetic`. Precision 0.855. |
| v2.1-guarded | Added in-pipeline hallucination guard | Zero hallucinations in output. |
| v2.2 | Added EHR-template / auto-calculator exclusions (`failed to calculate`, `diabetic: yes/no` as checkbox answers) inside the EXCLUSIONS section | Spot-check showed gpt-4.1 still emitted 8/11 template strings, sometimes re-categorizing under `other / review needed`. Insufficient. |
| **v2.3** (current) | Promoted the EHR-template carve-out to a new top-of-hierarchy MANDATORY NEVER-FLAG section; added complementary deterministic deny-list filter in `bias_pipeline.py`; added OpenAI `seed=42` parameter to the API call | 0 template-artifact rows in output across three 100-note runs. Multi-run noise floor at 88.6% term stability (seeded vs seeded), 0.80 Jaccard. |

---

## 3. Model invocation

Implemented in the `AzureBiasPipeline` class (`bias_pipeline.py:406`).

### 3.1 Model and decoding parameters

| Parameter | Value | Where |
|---|---|---|
| Provider | Azure OpenAI | `AzureOpenAI` client constructed in `run_bias_batch.py` |
| Deployment | `gpt-4.1` (configured via `AZURE_DEPLOYMENT` env var) | `AzureBiasPipelineConfig.model_for_api` |
| Temperature | 0 (deterministic) | `AzureBiasPipelineConfig.temperature` (default 0) |
| Reproducibility seed | 42 (OpenAI "best effort" `seed` parameter) | `AzureBiasPipelineConfig.seed` (default 42) |
| Max retries on transient errors | 5 | `AzureBiasPipelineConfig.max_retries` |
| Backoff base | 1.4 s, exponential | `_sleep_backoff` |
| Per-request parallelism | 4 concurrent chunks per note | `AzureBiasPipelineConfig.parallel_calls` |
| Global concurrency cap | 8 in-flight requests across the pool | `AzureBiasPipelineConfig.global_max_calls` (via `Semaphore`) |

### 3.2 Structured outputs

When the deployment supports it, the pipeline issues a JSON Schema–constrained response (`use_structured_outputs=True`). The schema enforces the `{"possible": [...], "likely": [...]}` shape with `term: string`, `categories: array[string]`. If the deployment or API version does not support structured outputs, the pipeline automatically falls back to prompt-guided JSON parsing (`_parse_model_response_text` in `bias_pipeline.py:318`). The fallback path also strips markdown code fences and tolerates trailing prose.

### 3.3 Chunking

`chunk_by_sentences` (`bias_pipeline.py:203`) splits each note into sentence-aligned chunks of ≤ 2,800 characters (`AzureBiasPipelineConfig.chunk_char_limit`). Sentence boundaries are detected by the regex `_SENT_SPLIT_RE`. A single sentence longer than 2,800 characters is hard-wrapped on character count as a last resort. Per-chunk results are merged in `_merge_chunk_results` (`bias_pipeline.py:755`), preserving first-appearance order and deduplicating identical phrases per bucket.

### 3.4 Caching

Per-note disk cache at `$BIAS_NOTES_CACHE_PATH` (default `output/bias_detection_cache.csv`). The cache key is `(note_hash, prompt_version, pipeline_version, model_used)`. Cache reuse is opt-in via `cache_chunk_responses=True` (default) and is fully invalidated when `PROMPT_VERSION` changes. This makes reruns cheap for unchanged notes while preventing stale results across prompt versions.

`note_hash` is SHA-256 over the normalized note text (`AzureBiasPipeline._note_hash`).

### 3.5 Failure isolation

Per-chunk failures (parse errors, transient HTTP errors past max retries) are logged to a sidecar JSONL file (`*_chunk_failures.jsonl`) containing `note_hash`, `chunk_index`, `error`, and a 500-character preview of the malformed response. Failures do not terminate the run; the failing chunk contributes no terms but the note continues through the other chunks. The output row records `Chunk_Failure_Count` and `Chunk_Failure_Details` for downstream filtering.

---

## 4. Hallucination guard

Implemented in `filter_hallucinated_terms` and `term_present_in_chunk` (`bias_pipeline.py:351` and `:369`). It runs **after every LLM chunk response and before caching**, on every model output regardless of which model is used.

### 4.1 Motivation

In the v1 run, gpt-4o-mini emitted canonical stigma terms (e.g., `elderly`, `non-compliant`, `uncontrolled diabetic`) that did not appear anywhere in the source note text. These look correct to a reviewer and would have inflated apparent precision. 40% of the original reviewer-adjudicated gold set was contaminated by such hallucinations.

### 4.2 Algorithm

For each `(bucket, term)` returned by the model on chunk `c`:

1. Lowercase both `term` and `c`.
2. Collapse all whitespace runs to single spaces (`_WS_RE = re.compile(r"\s+")`).
3. Strip leading/trailing punctuation (` \t\n\r.,;:!?"'()[]{}<>`).
4. Substring-match the normalized term in the normalized chunk.
5. If the term contains a parenthetical qualifier (e.g., `obesity (BMI 30–39.9)`), also try matching just the head (`obesity`).
6. If neither matches, drop the term and increment the run's hallucination counter.

### 4.3 Observability

A run-level counter is reported at completion (`Hallucination guard dropped: N model-emitted term(s) absent from source`). Per-drop verbose logging is enabled with `BIAS_HALLUCINATION_GUARD_VERBOSE=1`.

### 4.4 Empirical effect

In the canonical 100-note run with gpt-4.1 + v2.1-guarded prompt, the guard dropped 8 model emissions during execution; the final output contained 0 unverifiable terms. In the v2.3 multi-run series, the guard dropped 7–10 emissions per 100-note run, again with 0 unverifiable terms in the final output.

---

## 4b. Template-artifact filter

Implemented in `filter_template_artifacts` (`bias_pipeline.py`). Runs immediately after the hallucination guard, before caching. Drops any flagged term whose normalized form (lowercased, whitespace-collapsed, colon-spacing normalized) is in the fixed deny-list `TEMPLATE_ARTIFACT_DENYLIST`:

```
failed to calculate
diabetic: yes
diabetic: no
diabetic:yes
diabetic:no
unable to calculate
not calculated
```

These are EHR-template / auto-calculator artifacts (BMI / GFR / ASCVD risk / fall-risk calculator output, screening checkboxes) — software-generated text, not clinician voice. The prompt also lists them at the top of its hierarchy under MANDATORY NEVER-FLAG, but model compliance was empirically incomplete: a v2.2 spot-check (prompt-only exclusion) showed gpt-4.1 still emitted 8 of 11 template strings, sometimes re-categorizing them under `other / review needed` rather than dropping them. The deny-list filter is the deterministic enforcement layer that backs up the prompt rule.

Run-level counter reported at completion (`Template-artifact guard dropped: N EHR-template term(s) on the deny-list`). Verbose logging via `BIAS_TEMPLATE_ARTIFACT_GUARD_VERBOSE=1`.

**Empirical effect (v2.3, 100-note runs):** filter dropped 0–1 emissions per run, with 0 template-artifact rows in any final output. The prompt change handles most cases; the filter catches the remainder.

---

## 4c. Run-to-run reproducibility (noise floor)

Even at temperature 0 with structured outputs, Azure-side serving introduces non-determinism (floating-point non-associativity across batched GPU inference, speculative decoding variation, MoE routing). To bound this, we re-ran the same 100-note sample three times under the v2.3 prompt: one run without the `seed` parameter, two consecutive runs with `seed=42`.

| Comparison | Conditions | Jaccard | Stability (kept / |A|) |
|---|---|---|---|
| v2.2 → v2.3 | Different prompts (real prompt drift + model noise) | 0.65 | 70.7% |
| v2.3 unseeded → v2.3 seeded run 1 | Same prompt; seed only in run 2 | 0.70 | 82.1% |
| v2.3 unseeded → v2.3 seeded run 2 | Same prompt; seed only in run 2 | 0.70 | 82.1% |
| **v2.3 seeded run 1 → seeded run 2** | **Same prompt, both seeded** | **0.80** | **88.6%** |

At the same `(note_id, normalized_term)` granularity, **62.5% of all distinct flagged pairs across the three runs were consensus** (present in every run); **20.4% were singletons** (present in only one run). Singletons skewed toward low-frequency, context-dependent terms (`admits`, `declined`, `alcohol cravings`, `she declined the covid-19 booster`).

At the note level (where the paper's primary metrics live) reproducibility is substantially higher: **74 of 100 notes had identical flag counts across all three runs**; only 1 note varied by ≥3 flags; the mean per-note range (max − min flag count across runs) was 0.49.

**Implication for reporting**: aggregate metrics (note-level any-flag prevalence, flags-per-1,000-words, per-category counts) inherit the ~74-of-100 note-level stability and are highly reproducible. Phrase-by-phrase claims should be hedged or reported as consensus across N ≥ 3 runs.

**Recommended for the full-cohort analysis**: run the n=500 pipeline three times under `seed=42` and report each metric as the median of the three runs with a 95% bootstrap CI from the run-level distribution. Cached outputs (one per run, keyed by `Note_Hash + Prompt_Version + Model_Used`) make this affordable for re-runs.

---

## 5. Second-pass adjudication

`adjudicate_ambiguous_terms` (`bias_pipeline.py:649`) re-queries the model only for terms in a small allowlist of context-sensitive words (`_AMBIGUOUS_POSSIBLE_TERMS`: `obese`, `diabetic`, `hypertensive`, `schizophrenic`, `bipolar`, `psychotic`, `elderly`, `refused`, `controlled`, `borderline control`, `overweight`, `underweight`, `failure`, `failed treatment`, `homeless`, `dependent`, `relapse`). It sends the term plus a short context window and asks the model to choose `keep_possible | move_likely | drop`. Only up to `ambiguous_term_limit=8` terms per note are adjudicated.

This pass primarily corrects mis-bucketing (an obvious identity-form `obese` accidentally placed in `possible`) and removes neutral usages (e.g., `refused` in a polite-decline context).

---

## 6. Output schema

`write_output_bundle` (`bias_pipeline.py:993`) writes three files per run, sharing a timestamped prefix:

### 6.1 `bias_flagged_<prefix>_<timestamp>.csv` — reviewer-facing

All input columns plus:

- `Possible_Biased_Terms`, `Likely_Biased_Terms` — JSON arrays of `{term, categories}` objects.
- `Possible_Biased_Terms_Normalized`, `Likely_Biased_Terms_Normalized` — lowercased / canonicalized term strings.
- `Possible_Bias_Categories`, `Likely_Bias_Categories` — flat lists of category labels in first-appearance order.
- `Possible_Bias_Details`, `Likely_Bias_Details` — JSON arrays with per-term `term`, `normalized_term`, `categories`, and surrounding `context` snippet (used by reviewers).
- `Possible_Bias_Count`, `Likely_Bias_Count` — integer flag counts.
- `Note_Word_Count` — for per-1,000-word normalization.
- `Chunk_Failure_Count`, `Chunk_Failure_Details` — diagnostic.
- `Prompt_Version`, `Pipeline_Version`, `Model_Used`, `Note_Hash` — provenance.

### 6.2 `..._analysis_ready.csv`

Same shape as the reviewer CSV, minus `note_text` and `Possible_Bias_Details` / `Likely_Bias_Details`. Suitable for de-identified sharing with statisticians.

### 6.3 `..._reviewer_adjudication.csv`

Long format: one row per flagged term, written by `build_reviewer_adjudication_dataframe` (`bias_pipeline.py:909`). Columns include `source_note_id`, `model_bucket`, `model_term`, `normalized_term`, `model_category`, `term_context`, and **blank reviewer columns**: `review_decision`, `reviewed_term`, `reviewed_category`, `reviewer_notes`, `false_positive_term`, `missed_term`, `needs_prompt_update`, `needs_rule_update`. Reviewers fill these in directly to produce the gold-standard file.

---

## 7. Reviewer adjudication and gold-standard construction

### 7.1 Process

Two independent clinical reviewers (Drs. Ture and Liebovitz) review the reviewer-adjudication CSV. For each row:

- **`false_positive_term`** — mark `x` if the model was wrong to flag this term.
- **`reviewer_notes`** — free-text rationale, especially for edge cases.
- **`missed_term`** — verbatim term if the reviewer spotted stigma the model missed in the same note (one new row per missed term).
- Disagreements are resolved by consensus discussion; a third reviewer adjudicates persistent disagreements.

The reviewed file is exported as `.xlsx` and stored alongside the cleaned corpus in OneDrive.

### 7.2 Verifiability filter

Because the v1 gold standard contained reviewer judgments on terms the original model had hallucinated, the evaluation harness restricts scoring to the **verifiable subset** of gold rows — those where `model_term` actually appears verbatim in its source note after the same normalization used by the hallucination guard. For the original gold set, 393 of 650 rows met this criterion. Going forward, every output produced by v2.1-guarded or later is verifiable by construction.

The filter recipe is documented in `How_to_use.md` Section 3.C and reproduced verbatim in the eval notebook.

---

## 8. Evaluation

Implemented in `eval_against_gold.ipynb`. The notebook is parameterized by two paths set in cell 1:

- `GOLD_PATH` — the reviewer-adjudicated `.xlsx`.
- `CANDIDATE_CSV` — the run-output CSV to score.

### 8.1 Match key

Candidate flags are joined to gold rows on `(source_note_id, normalized_term, model_category)`. The notebook critically casts `matched['is_fp'] = matched['is_fp'].astype(bool)` after merge to repair the dtype upcast that pandas performs when merging boolean columns through a non-matching key.

### 8.2 Metrics reported

| Metric | Definition |
|---|---|
| Precision | TP / (TP + FP) where FP = `false_positive_term == 'x'`. |
| FP rate | FP / matched. |
| Per-category precision | Precision restricted to each `model_category`. |
| Per-(term, category) FP rate | For pruning the TERM_BANK. |
| Dax-vs-Human FP-rate parity | FP rate stratified by `Dax_or_Human`; reports the gap. |
| Recall on key missed terms | Specifically `admits` (the most-flagged missed term in v1 review), counted across `missed_term` entries. |
| Hallucination count | Terms in candidate output that do not appear verbatim in source. Should be 0 for runs produced by v2.1-guarded or later. |
| Summary dict | One-line export suitable for the run log. |

### 8.3 Run log

Each model × prompt combination is appended to `output/run_log.csv` with columns: `date, prompt_version, model, n_notes, total_flags, matched, tp, fp, precision, fp_rate, dax_fp, human_fp, gap, admits_recall, hallucinations`. The log is the single source of truth for comparing iterations.

---

## 9. Statistical analysis

Implemented in `analyze_results.py` (CLI) and `analyze_results.ipynb` (interactive).

### 9.1 Note-level rates

- **Any-flag prevalence** — proportion of notes with at least one flag, overall and stratified by `Dax_or_Human`.
- **Likely-flag prevalence** — same, restricted to the `likely` bucket.
- **Flags per 1,000 words** — `Likely_Bias_Count + Possible_Bias_Count` divided by `Note_Word_Count`, scaled. Reported as median (IQR) and compared between Dax and Human notes via Mann–Whitney U.

### 9.2 Term and category breakdowns

- **Top-K most frequent terms** — overall, likely-only, possible-only.
- **Category distribution** — count and share of total flags by canonical category label.
- **Source-level differences** — per-category counts in Dax vs Human notes; report categories with the largest absolute difference and absolute rate difference.

### 9.3 Significance tests

For the publication run (n=500):
- Note-level **any-flag** prevalence: Fisher's exact test, two-sided.
- **Flags-per-1,000-words** distribution: Mann–Whitney U, two-sided. Report median (IQR) per arm and the rank-biserial correlation as effect size.
- **Per-category counts**: chi-square test of independence with Holm correction across the 9 primary categories. Categories with expected counts < 5 use Fisher's exact.

All tests are computed in `analyze_results.ipynb` using scipy. Bonferroni-controlled alpha is reported alongside Holm-adjusted p-values.

---

## 10. Reproducibility

### 10.1 Software stack

- **Python** 3.11 (constrained in `pyproject.toml`: `requires-python = ">=3.11,<3.12"`).
- **Dependency manager:** `uv` with `uv.lock` committed.
- **Pinned versions** (from `pyproject.toml`): `openai==1.78.0`, `pandas==2.2.3`, `numpy==2.2.5`, `ftfy==6.3.1`, `python-dotenv==1.1.1`, `openpyxl==3.1.5`, `requests==2.32.3`, `streamlit==1.45.0`, `ipykernel==6.30.1`.
- **Install:** `uv sync --frozen`.

### 10.2 Commands to reproduce the publication run

```bash
# 1. Pre-process the raw corpus (one-time).
uv run jupyter nbconvert --to notebook --execute pre_process_notes.ipynb

# 2. Full-corpus bias detection.
uv run python run_bias_batch.py \
    --selection all \
    --output-prefix bias_flagged_n500_v22 \
    --skip-notebook-check

# 3. Evaluate against the reviewer-adjudicated gold standard.
#    Set GOLD_PATH and CANDIDATE_CSV in cell 1 of eval_against_gold.ipynb,
#    then run all cells.
uv run jupyter notebook eval_against_gold.ipynb

# 4. Statistical analysis.
uv run python analyze_results.py \
    "$BIAS_NOTES_OUTPUT_DIR/bias_flagged_n500_v22_<timestamp>_analysis_ready.csv"
```

For the development sample (n=100):

```bash
uv run python run_bias_batch.py \
    --selection random --seed 42 --charts 100 \
    --output-prefix bias_flagged_n100_v22
```

### 10.3 Determinism

- Sampling is seeded (`--seed 42`).
- Temperature is 0; structured outputs further constrain the response shape.
- Cache reuse is controlled by `(note_hash, prompt_version, pipeline_version, model_used)`.

Two caveats on determinism:
1. Azure-side model serving introduces non-determinism that temperature alone does not eliminate. We mitigate by caching first-pass results; once a note is cached, all downstream metrics depend only on the cached row.
2. The hallucination guard is purely post-hoc and deterministic given the model output.

### 10.4 Code and data availability

- **Code:** this repository (https://github.com/<org>/bias-notes) at commit `<commit-hash>`.
- **De-identified analysis-ready CSV:** released alongside the publication, omitting `note_text`.
- **Reviewer-adjudication file:** available on request under data-use agreement, omitting note text.
- **Raw notes:** retained on institutional storage under IRB; not released.

---

## 11. Limitations

1. **Single institution.** All 500 notes come from one academic health system. Generalizability to community settings or non-English notes is untested.
2. **Reviewer pool size.** Two primary reviewers with adjudication. Inter-rater reliability statistics (Cohen's κ on the overlap set) should be reported in the manuscript.
3. **LLM as instrument.** Model outputs depend on a vendor-controlled deployment. We record `Model_Used` and `Prompt_Version` in every row to make the instrument version explicit, and we maintain a hallucination guard to bound the worst failure mode, but model behavior may drift across Azure deployments.
4. **EHR-template artifacts.** v2.3's MANDATORY NEVER-FLAG + deterministic deny-list filter eliminates the three known template strings (`failed to calculate`, `diabetic: yes`, `diabetic: no`) deterministically, and they appear in 0 of 300 flagged rows across the three v2.3 100-note runs. Other auto-text patterns may exist in the n=500 cohort that are not yet on the deny-list; we recommend an end-of-run sensitivity audit identifying any high-frequency new term whose context window matches a templated yes/no or calculator-output signature, and adding it to `TEMPLATE_ARTIFACT_DENYLIST` for a follow-up run.
5. **Possible vs. likely bucketing** carries reviewer subjectivity at the margins. We report the two buckets separately in all primary results.

---

## 12. File / function index for replication

| Step | File | Function / cell |
|---|---|---|
| Pre-processing | `pre_process_notes.ipynb` | "Clean Notes and Save" cell; "Repair Mojibake" cell |
| Prompt definition | `bias_detection_prompt.py` | module docstring |
| Prompt loading | `bias_pipeline.py:189` | `load_prompt_text` |
| Sentence chunking | `bias_pipeline.py:203` | `chunk_by_sentences` |
| Azure call | `bias_pipeline.py:556` | `call_model_on_chunk` |
| Structured-output fallback | `bias_pipeline.py:318` | `_parse_model_response_text` |
| Hallucination guard | `bias_pipeline.py` | `term_present_in_chunk`, `filter_hallucinated_terms` |
| Template-artifact filter | `bias_pipeline.py` | `filter_template_artifacts`, `TEMPLATE_ARTIFACT_DENYLIST`, `_normalize_for_denylist` |
| Reproducibility seed wiring | `bias_pipeline.py` | `AzureBiasPipelineConfig.seed`, `_create_completion`, `adjudicate_ambiguous_terms` |
| Per-note merge | `bias_pipeline.py:755` | `_merge_chunk_results` |
| Second-pass adjudication | `bias_pipeline.py:649` | `adjudicate_ambiguous_terms` |
| Disk cache | `bias_pipeline.py:446`, `:474` | `_load_note_cache`, `_write_note_cache` |
| Output bundle | `bias_pipeline.py:993` | `write_output_bundle` |
| Reviewer adjudication export | `bias_pipeline.py:909` | `build_reviewer_adjudication_dataframe` |
| CLI entry point | `run_bias_batch.py` | `main` |
| Gold-standard evaluation | `eval_against_gold.ipynb` | all cells |
| Statistical analysis | `analyze_results.py`, `analyze_results.ipynb` | top-level |
| Unit tests | `tests/test_bias_pipeline.py` | `HallucinationGuardTests`, chunking, postprocessing, cache, reviewer-export, analysis-summary tests |

---

## 13. Change log against earlier runs

| Date | Prompt version | Notable change |
|---|---|---|
| 2026-04-14 | v1 | Initial prompt; gpt-4o-mini. |
| 2026-05-08 | v2 | Reviewer-driven exceptions; switched to gpt-4.1. |
| 2026-05-11 | v2.1 | Added MANDATORY FLAGS section. |
| 2026-05-11 | v2.1-guarded | Hallucination guard live in pipeline. |
| 2026-05-12 | v2.2 | Added EHR-template/auto-calculator exclusions inside EXCLUSIONS section. Spot-check showed model bypassing the rule by re-categorizing the targets. |
| 2026-05-12 | **v2.3** | Promoted EHR-template carve-out to top-of-hierarchy MANDATORY NEVER-FLAG; added deterministic deny-list filter in `bias_pipeline.py`; added OpenAI `seed=42` parameter. **This is the version used for the n=500 publication run.** Multi-run noise floor (3 × 100-note runs): 88.6% term stability, 0 template-artifact leakage. |
