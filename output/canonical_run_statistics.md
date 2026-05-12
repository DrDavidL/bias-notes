# Canonical Run Statistics

**Run**: `bias_flagged_final_20260511_121241.csv`
**Date**: 2026-05-11
**Prompt version**: `2026-05-11-reviewer-feedback-v2.1-guarded`
**Model**: gpt-4.1 (Azure OpenAI)
**Selection**: `random --seed 42 --charts 100`
**Hallucination guard**: live (dropped 8 model-emitted terms during the run; 0 remain)

---

## 1. Cohort

| Metric | All | Dax | Human |
|---|---|---|---|
| Notes | 100 | 46 | 54 |
| Mean length (chars) | 6 247 | **8 079** | 4 686 |

Dax notes are ~70% longer on average — relevant when interpreting absolute flag counts.

## 2. Flag prevalence

| | Any flag | Likely flag | Possible flag |
|---|---|---|---|
| All notes | **68%** | 55% | 36% |
| **Dax** (n=46) | 74% | 63% | 33% |
| **Human** (n=54) | 63% | 48% | 39% |

Dax notes are slightly more likely to contain stigmatizing language (63% with at least one *likely* flag vs 48% for Human).

## 3. Total flag counts

| Bucket | Total | Dax | Human | Per Dax note | Per Human note |
|---|---|---|---|---|---|
| **Likely** | 96 | 46 | 50 | 1.00 | 0.93 |
| **Possible** | 53 | 27 | 26 | 0.59 | 0.48 |
| **All** | 149 | 73 | 76 | **1.59** | **1.41** |

Per-note rates are essentially flat (1.59 vs 1.41). The headline comparison is **not** dominated by a volume effect.

## 4. Top 10 most frequent terms (overall)

| Rank | Term | Count |
|---|---|---|
| 1 | misuse of tobacco | 9 |
| 2 | obesity | 7 |
| 3 | overweight | 5 |
| 4 | diabetic: no | 5 |
| 5 | failed to calculate | 4 |
| 6 | smoker | 3 |
| 7 | well controlled | 3 |
| 8 | obesity (BMI 30–39.9) | 2 |
| 9 | severe obesity (BMI ≥40) (CMS-HCC) | 2 |
| 10 | bipolar disorder | 2 |

### 4a. Top likely-only terms

| Term | Count |
|---|---|
| misuse of tobacco | 9 |
| obesity | 5 |
| diabetic: no | 4 |
| smoker | 2 |
| admits to | 2 |
| illicit substances | 2 |
| obesity (BMI 30–39.9) | 2 |
| she is obese | 2 |

### 4b. Top possible-only terms

| Term | Count |
|---|---|
| overweight | 4 |
| failed to calculate | 4 |
| well controlled | 2 |
| bipolar disorder | 2 |
| obesity | 2 |
| weight gain | 2 |

## 5. Flag distribution by bias category

| Category | Total | Likely | Possible | % of total |
|---|---|---|---|---|
| weight-based identity label | 29 | 19 | 10 | 19.5% |
| effort-based language | 22 | 11 | 11 | 14.8% |
| substance-use judgment | 22 | 19 | 3 | 14.8% |
| outcome-judging language | 21 | 12 | 9 | 14.1% |
| substance identity label | 13 | 9 | 4 | 8.7% |
| condition identity label | 10 | 5 | 5 | 6.7% |
| credibility-doubting language | 8 | 7 | 1 | 5.4% |
| moralizing lifestyle language | 8 | 4 | 4 | 5.4% |
| disapproval / moralizing tone | 4 | 3 | 1 | 2.7% |
| behavior description | 4 | 3 | 1 | 2.7% |
| judgmental social risk framing | 3 | 1 | 2 | 2.0% |
| ageism / appearance-based / autonomy-undermining / difficult-patient / paternalistic | 5 | 3 | 2 | 3.4% |

## 6. Source-level differences (Dax vs Human, raw counts)

### 6a. More common in Dax notes

| Category | Dax | Human |
|---|---|---|
| substance-use judgment | **15** | 7 |
| effort-based language | **13** | 9 |
| credibility-doubting language ("admits to") | **7** | 1 |
| moralizing lifestyle language | **6** | 2 |

### 6b. More common in Human notes

| Category | Human | Dax |
|---|---|---|
| outcome-judging language | **19** | 2 |
| weight-based identity label | **16** | 13 |
| substance identity label | **8** | 5 |
| disapproval / moralizing tone | **4** | 0 |

The Human-vs-Dax difference in outcome-judging language (19 vs 2) is the largest single category gap.

## 7. Flag-count distribution per note

| Flags per note | # notes |
|---|---|
| 0 | 32 |
| 1 | 25 |
| 2 | 22 |
| 3 | 11 |
| 4–5 | 8 |
| 6–10 | 2 |
| >10 | 0 |

- Median across all 100 notes: **1 flag**
- Median across the 68 flagged notes: **2 flags**
- Max: 7 flags in a single note

## 8. EHR-template artifacts worth investigating

Two patterns appear concentrated in Human notes and look like EHR templating, not physician authorship:

- **"failed to calculate"** — 4 occurrences in Human notes (0 in Dax). Likely an auto-text artifact when a calculated field (e.g., BMI) cannot be derived from missing source values.
- **"diabetic: yes" / "diabetic: no"** — 7 occurrences across both groups, all appearing as screening-checkbox patterns inside templated sections.

If confirmed as template-generated, both should be added to the prompt's EXCLUSIONS section before the final analysis, which would further narrow the (currently inverted) outcome-judging difference between Dax and Human.

## 9. Evaluation against the reviewer-adjudicated gold set

Restricted to the verifiable subset of the gold set (393 of 650 rows where the model_term actually appears in its source note):

| Metric | Value |
|---|---|
| Candidate flags matched to gold | 55 |
| TP | 47 |
| FP | 8 |
| **Precision** | **0.855** |
| FP rate | 0.145 |
| Dax FP rate | 0.115 |
| Human FP rate | 0.172 |
| "admits" recall (reviewer-flagged missed term) | **5/5** |
| Hallucinations remaining in output | **0** |

## 10. New flags surfaced by this run (not in the prior gold set)

94 candidate flags fall outside the prior reviewer-adjudicated rows — these are unreviewed-but-likely-valid flags surfaced by the v2.1 prompt + gpt-4.1 + guard combination, including: `misuse of tobacco` (8 new), `diabetic: no` (5), `failed to calculate` (4 — see EHR-template note above), `she is obese` (2), `admits to` (2), `severe obesity (BMI ≥40) (CMS-HCC)` (2), and 70+ singleton flags spanning weight-based identity, effort, and outcome-judging categories. These are the rows reviewers should adjudicate next to extend the gold set.
