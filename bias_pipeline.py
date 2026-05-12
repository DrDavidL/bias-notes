import concurrent.futures
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Lock, Semaphore
from typing import Dict, List, Optional

import pandas as pd
from openai import AzureOpenAI

from bias_review_utils import (
    ALLOWED_BIAS_CATEGORIES,
    canonicalize_bias_categories,
    enrich_bias_result,
)


PIPELINE_VERSION = "2026-04-14-structured-categories-v3"
PROMPT_VERSION = "2026-05-12-reviewer-feedback-v2.2-template-exclusions"
_AMBIGUOUS_POSSIBLE_TERMS = {
    "obese",
    "diabetic",
    "uncontrolled diabetic",
    "hypertensive",
    "schizophrenic",
    "bipolar",
    "psychotic",
    "elderly",
    "refused",
    "controlled",
    "borderline control",
    "overweight",
    "underweight",
    "failure",
    "failed treatment",
    "homeless",
    "dependent",
    "relapse",
    "non-english speaking",
}
_OUTPUT_COLUMNS = [
    "Possible_Biased_Terms",
    "Likely_Biased_Terms",
    "Possible_Biased_Terms_Normalized",
    "Likely_Biased_Terms_Normalized",
    "Possible_Bias_Categories",
    "Likely_Bias_Categories",
    "Possible_Bias_Details",
    "Likely_Bias_Details",
    "Possible_Bias_Count",
    "Likely_Bias_Count",
    "Note_Word_Count",
    "Chunk_Failure_Count",
    "Chunk_Failure_Details",
    "Prompt_Version",
    "Pipeline_Version",
    "Model_Used",
    "Note_Hash",
]

_SENT_SPLIT_RE = re.compile(r"(?<=\S[.!?])\s+(?=[\"'([{]*[A-Z0-9])", re.VERBOSE)
_PLACEHOLDER_PATTERNS = [
    r"<<<\s*\{PASTE THE NOTE TEXT/CHUNK HERE\}\s*>>>",
    r"\{CHUNK\}",
    r"\{\{CHUNK\}\}",
    r"<<<\s*\{CHUNK\}\s*>>>",
]

_PHYSIOLOGIC_NORMAL_PATTERNS = [
    r"(?:normal\s+)?(?:heart\s+)?rate(?:\s+(?:and|is|:))?\s*(?:normal|regular)",
    r"normal\s+(?:heart\s+sounds|pulses|rate)",
    r"(?:heart\s+sounds|pulses)\s*(?::|are|is)?\s*normal",
    r"regular\s+rhythm",
    r"(?:normal\s+)?(?:bilateral\s+)?breath\s+sounds\s*(?::|are|is)?\s*normal",
    r"(?:normal\s+)?(?:bilateral\s+)?breath\s+sounds",
    r"(?:lung|lungs)\s*(?::|exam|were|are|is)?\s*(?:normal|clear)",
    r"pulmonary\s+effort\s+(?:is\s+)?normal",
    r"air\s+movement\s+(?:is\s+)?normal",
    r"respirations?\s*(?::|are|is)?\s*normal",
    r"normal\s+breath\s+sounds",
    r"(?:normal\s+)?bowel\s+sounds\s*(?::|are|is)?\s*normal",
    r"bowel\s+sounds\s+(?:are\s+)?normal",
    r"normal\s+bowel\s+sounds",
    r"normal\s+(?:range\s+of\s+motion|ROM)",
    r"normal\s+ROM",
    r"coordination\s*(?::|is)?\s*normal",
    r"normal\s+coordination",
    r"conjunctivae?\s*(?::|are|is)?\s*normal",
    r"normal\s+conjunctivae?",
    r"(?:nose|mouth|ears?|throat|pharynx|tympanic)\s*(?::|are|is)?\s*normal",
    r"normal\s+(?:nose|mouth|ears?|throat|pharynx|tympanic)",
    r"normal\s+(?:labia|vaginal\s+canal|cervix|vulva|uterus|adnexa)",
    r"(?:labia|vaginal\s+canal|cervix|vulva|uterus|adnexa)\s*(?::|are|is)?\s*normal",
    r"blood\s+pressure\s+(?:is\s+)?normal",
    r"(?:heart\s+rate|HR)\s*(?::|is)?\s*normal",
    r"normal\s+exam",
    r"(?:lung|cardiac|abdominal|pelvic|neurological|musculoskeletal)\s+exam\s+normal",
    r"effort\s*(?::|is)?\s*(?:pulmonary\s+effort\s+is\s+)?normal",
    r"normal\s+range\b",
    r"low\s+range\s+of\s+normal",
    r"nutritional\s+status\s*(?::|is)?\s*normal",
    r"normal\s+(?:thyroid|liver|renal|kidney|cardiac|pulmonary)\s+function",
    r"normal\s+(?:labs?|imaging|x-ray|ct|mri|ultrasound|echo)",
    r"normal\s+(?:vaccine|medication|drug)\s+reactions?",
    r"(?:skin|nails?|hair)\s*(?::|are|is)?\s*normal",
    r"normal\s+(?:skin|nails?|hair)",
]
_PHYSIOLOGIC_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PHYSIOLOGIC_NORMAL_PATTERNS), re.IGNORECASE
)
_KEEP_PATTERNS = re.compile(
    r"(?:mood|behavior|behaviour|judgment|thought\s+content|thought\s+process|mental\s+status|attention|perception|affect|appearance|gait|weight)",
    re.IGNORECASE,
)
_DOCUMENTATION_ARTIFACTS = re.compile(
    r"(?:tobacco\s+smoker\s*:\s*no|smoking\s+status\s*:\s*|mood\s+and\s+affect\s+congruent)",
    re.IGNORECASE,
)
_STRUCTURED_OUTPUT_SCHEMA = {
    "name": "bias_detection_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "possible": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ALLOWED_BIAS_CATEGORIES,
                            },
                        },
                    },
                    "required": ["term", "categories"],
                    "additionalProperties": False,
                },
            },
            "likely": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ALLOWED_BIAS_CATEGORIES,
                            },
                        },
                    },
                    "required": ["term", "categories"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["possible", "likely"],
        "additionalProperties": False,
    },
}


@dataclass
class AzureBiasPipelineConfig:
    prompt_path: str
    model_for_api: str
    temperature: float = 0
    chunk_char_limit: int = 2800
    max_retries: int = 5
    sleep_base_sec: float = 1.4
    parallel_calls: int = 4
    global_max_calls: int = 8
    cache_chunk_responses: bool = True
    cache_path: Optional[str] = None
    chunk_failure_log_path: Optional[str] = None
    use_structured_outputs: bool = True
    structured_outputs_fallback_to_prompt_json: bool = True
    enable_second_pass_adjudication: bool = True
    ambiguous_term_limit: int = 8
    prompt_version: str = PROMPT_VERSION
    pipeline_version: str = PIPELINE_VERSION


def generate_output_path(output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{prefix}_{timestamp}.csv")


def load_prompt_text(prompt_path: str) -> str:
    try:
        spec = importlib.util.spec_from_file_location("bias_prompt_mod", prompt_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        text = getattr(mod, "PROMPT", None) or getattr(mod, "prompt", None)
        if isinstance(text, str) and text.strip():
            return text
    except Exception:
        pass
    with open(prompt_path, "r", encoding="utf-8") as handle:
        return handle.read()


def chunk_by_sentences(text: str, max_chars: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = [
        part.strip() for part in _SENT_SPLIT_RE.split(normalized) if part.strip()
    ]
    if not sentences:
        return []

    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current_len + add_len <= max_chars:
            current.append(sentence)
            current_len += add_len
            continue

        if current:
            chunks.append(" ".join(current).strip())
        if len(sentence) > max_chars:
            piece = sentence
            while len(piece) > max_chars:
                chunks.append(piece[:max_chars])
                piece = piece[max_chars:]
            current, current_len = ([piece], len(piece)) if piece else ([], 0)
        else:
            current, current_len = [sentence], len(sentence)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def inject_chunk_into_prompt(prompt_text: str, chunk_text: str) -> str:
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, prompt_text):

            def _replace(_match):
                if "<<<" in pattern and ">>>" in pattern:
                    return f"<<<\n{chunk_text}\n>>>"
                return chunk_text

            return re.sub(pattern, _replace, prompt_text, count=1)

    block_re = re.compile(
        r"(INPUT\s+NOTE\s+CHUNK.*?<<<)([\s\S]*?)(>>>)([\s\S]*)", re.IGNORECASE
    )
    match = block_re.search(prompt_text)
    if match:
        return (
            match.group(1) + "\n" + chunk_text + "\n" + match.group(3) + match.group(4)
        )
    return f"{prompt_text.rstrip()}\n\nINPUT NOTE CHUNK\n<<<\n{chunk_text}\n>>>"


def is_physiologic_false_positive(term: str) -> bool:
    stripped = term.strip()
    if _DOCUMENTATION_ARTIFACTS.search(stripped):
        return True
    if _KEEP_PATTERNS.search(stripped):
        return False
    if _PHYSIOLOGIC_RE.search(stripped):
        return True
    return bool(re.fullmatch(r"normal\.?", stripped, re.IGNORECASE))


def _extract_result_term(item) -> str:
    if isinstance(item, dict):
        for key in ("term", "phrase", "text"):
            value = str(item.get(key, "")).strip()
            if value:
                return re.sub(r"\s+", " ", value)
        return ""
    return re.sub(r"\s+", " ", str(item).strip())


def _extract_result_categories(item) -> List[str]:
    if not isinstance(item, dict):
        return []
    categories = item.get("categories")
    if categories is None and item.get("category"):
        categories = [item.get("category")]
    elif isinstance(categories, str):
        categories = [categories]
    elif not isinstance(categories, list):
        categories = []
    return canonicalize_bias_categories(categories)


def _normalize_result_entry(item):
    term = _extract_result_term(item)
    if not term:
        return None
    categories = _extract_result_categories(item)
    if categories:
        return {"term": term, "categories": categories}
    return term


def _canonical_term_key(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip()).strip(" ,.;:!?()[]{}\"'").lower()


def _collect_detail_categories(detail: Dict[str, object]) -> List[str]:
    raw_categories = detail.get("categories", [])
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    categories = canonicalize_bias_categories(
        raw_categories if isinstance(raw_categories, list) else []
    )
    if categories:
        return categories

    category = detail.get("category", "")
    if category:
        categories = canonicalize_bias_categories([category])
        if categories:
            return categories
        return [str(category).strip()]
    return []


def _parse_model_response_text(text: str):
    decoder = json.JSONDecoder()
    try:
        return json.loads(text)
    except json.JSONDecodeError as initial_error:
        first_openers = [
            idx
            for idx in (
                text.find("{"),
                text.find("["),
            )
            if idx != -1
        ]
        for start_idx in sorted(set(first_openers)):
            try:
                parsed, _ = decoder.raw_decode(text[start_idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        raise initial_error


_WS_RE = re.compile(r"\s+")


def _normalize_for_presence_check(text: str) -> str:
    """Lowercase + collapse whitespace + strip ASCII punctuation for substring matching."""
    lowered = text.lower()
    collapsed = _WS_RE.sub(" ", lowered).strip()
    # Strip common trailing punctuation that the LLM sometimes appends/omits inconsistently
    return collapsed.strip(" \t\n\r.,;:!?\"'()[]{}<>")


def term_present_in_chunk(term: str, chunk_text: str) -> bool:
    """True iff `term` appears in `chunk_text` after whitespace/case/punctuation normalization."""
    if not term or not chunk_text:
        return False
    norm_term = _normalize_for_presence_check(term)
    if not norm_term:
        return False
    norm_chunk = _normalize_for_presence_check(chunk_text)
    if norm_term in norm_chunk:
        return True
    # Also try the term without its trailing parenthetical (e.g. "obesity (BMI 30-39.9" -> "obesity")
    if "(" in norm_term:
        head = norm_term.split("(", 1)[0].strip()
        if head and head in norm_chunk:
            return True
    return False


def filter_hallucinated_terms(
    result: Dict[str, List[object]], chunk_text: str
) -> tuple[Dict[str, List[object]], List[Dict[str, str]]]:
    """Drop any flagged term whose text does not appear in `chunk_text`.

    Returns (filtered_result, dropped_log). `dropped_log` lists the discarded terms
    so callers can record them for visibility.
    """
    dropped: List[Dict[str, str]] = []
    filtered: Dict[str, List[object]] = {"possible": [], "likely": []}
    for bucket in ("possible", "likely"):
        for item in result.get(bucket, []):
            term = _extract_result_term(item)
            if term_present_in_chunk(term, chunk_text):
                filtered[bucket].append(item)
            else:
                dropped.append({"bucket": bucket, "term": term})
    return filtered, dropped


def postprocess_results(result: Dict[str, List[object]]) -> Dict[str, List[object]]:
    return {
        "possible": [
            normalized_item
            for item in result["possible"]
            for normalized_item in [_normalize_result_entry(item)]
            if normalized_item
            and not is_physiologic_false_positive(_extract_result_term(normalized_item))
        ],
        "likely": [
            normalized_item
            for item in result["likely"]
            for normalized_item in [_normalize_result_entry(item)]
            if normalized_item
            and not is_physiologic_false_positive(_extract_result_term(normalized_item))
        ],
    }


class AzureBiasPipeline:
    def __init__(self, client: AzureOpenAI, config: AzureBiasPipelineConfig):
        self.client = client
        self.config = config
        self.prompt_text = load_prompt_text(config.prompt_path)
        self._semaphore = Semaphore(config.global_max_calls)
        self._cache_lock = Lock()
        self._chunk_cache: Dict[str, Dict[str, List[object]]] = {}
        self._hallucinations_dropped_count: int = 0
        self._hallucination_guard_verbose: bool = os.getenv(
            "BIAS_HALLUCINATION_GUARD_VERBOSE", ""
        ).lower() in ("1", "true", "yes")
        self._note_cache = self._load_note_cache(config.cache_path)
        self._current_note_hash = ""
        self._last_chunk_failures: List[Dict[str, object]] = []
        self._structured_outputs_enabled = config.use_structured_outputs
        self._structured_outputs_disabled_reason = ""

    def _sleep_backoff(self, attempt: int) -> None:
        delay = (
            self.config.sleep_base_sec * (2 ** (attempt - 1)) * random.uniform(0.7, 1.3)
        )
        time.sleep(delay)

    @staticmethod
    def _note_hash(note_text: str) -> str:
        return hashlib.sha256(str(note_text).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _serialize_output_value(value):
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def _deserialize_cache_value(column: str, value):
        if pd.isna(value):
            if column.endswith("_Count") or column == "Note_Word_Count":
                return 0
            return "[]"
        if column.endswith("_Count") or column == "Note_Word_Count":
            return int(value)
        return value

    def _load_note_cache(
        self, cache_path: Optional[str]
    ) -> Dict[str, Dict[str, object]]:
        if not cache_path or not os.path.exists(cache_path):
            return {}
        try:
            cache_df = pd.read_csv(cache_path)
        except Exception:
            return {}

        note_cache: Dict[str, Dict[str, object]] = {}
        for _, row in cache_df.iterrows():
            row_model = str(row.get("Model_Used", ""))
            row_prompt = str(row.get("Prompt_Version", ""))
            row_pipeline = str(row.get("Pipeline_Version", ""))
            row_hash = str(row.get("Note_Hash", ""))
            if (
                not row_hash
                or row_model != self.config.model_for_api
                or row_prompt != self.config.prompt_version
                or row_pipeline != self.config.pipeline_version
            ):
                continue
            note_cache[row_hash] = {
                column: self._deserialize_cache_value(column, row.get(column))
                for column in _OUTPUT_COLUMNS
                if column in row.index
            }
        return note_cache

    def _write_note_cache(self) -> None:
        if not self.config.cache_path:
            return
        cache_dir = os.path.dirname(self.config.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cache_rows = []
        for _, payload in sorted(self._note_cache.items()):
            cache_rows.append(
                {
                    column: self._serialize_output_value(payload.get(column, "[]"))
                    for column in _OUTPUT_COLUMNS
                }
            )
        pd.DataFrame(cache_rows).to_csv(self.config.cache_path, index=False)

    def _cached_note_payload(self, note_hash: str) -> Optional[Dict[str, object]]:
        payload = self._note_cache.get(note_hash)
        if not payload:
            return None
        return dict(payload)

    def _record_chunk_failure(
        self,
        chunk_index: int,
        chunk_text: str,
        error: Exception,
        response_text: str,
    ) -> None:
        failure = {
            "note_hash": self._current_note_hash,
            "chunk_index": chunk_index,
            "chunk_char_count": len(chunk_text),
            "error": str(error),
            "response_preview": response_text[:500],
        }
        self._last_chunk_failures.append(failure)

        if not self.config.chunk_failure_log_path:
            return

        log_dir = os.path.dirname(self.config.chunk_failure_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.config.chunk_failure_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    @staticmethod
    def _should_fallback_from_structured_outputs(error: Exception) -> bool:
        if isinstance(error, TypeError):
            return True

        message = str(error).lower()
        return (
            "response_format" in message
            or "json_schema" in message
            or "structured output" in message
            or "structured outputs" in message
            or ("schema" in message and "unsupported" in message)
            or ("invalid parameter" in message and "response_format" in message)
        )

    def _create_completion(
        self, user_content: str, response_format: Optional[Dict[str, object]] = None
    ):
        request_kwargs = {
            "model": self.config.model_for_api,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.config.temperature,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        try:
            return self.client.chat.completions.create(
                max_completion_tokens=1024,
                **request_kwargs,
            )
        except TypeError:
            return self.client.chat.completions.create(
                max_tokens=1024,
                **request_kwargs,
            )

    def call_model_on_chunk(
        self, chunk_text: str, chunk_index: int = -1
    ) -> Dict[str, List[object]]:
        cache_key = chunk_text.strip()
        if self.config.cache_chunk_responses and cache_key:
            with self._cache_lock:
                cached = self._chunk_cache.get(cache_key)
            if cached is not None:
                return {
                    "possible": list(cached["possible"]),
                    "likely": list(cached["likely"]),
                }

        user_content = inject_chunk_into_prompt(self.prompt_text, chunk_text)
        last_err: Optional[Exception] = None
        last_response_text = ""

        for attempt in range(1, self.config.max_retries + 1):
            try:
                with self._semaphore:
                    try:
                        if self._structured_outputs_enabled:
                            response = self._create_completion(
                                user_content,
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": _STRUCTURED_OUTPUT_SCHEMA,
                                },
                            )
                        else:
                            response = self._create_completion(user_content)
                    except Exception as structured_error:
                        if (
                            self._structured_outputs_enabled
                            and self.config.structured_outputs_fallback_to_prompt_json
                            and self._should_fallback_from_structured_outputs(
                                structured_error
                            )
                        ):
                            self._structured_outputs_enabled = False
                            if not self._structured_outputs_disabled_reason:
                                self._structured_outputs_disabled_reason = str(
                                    structured_error
                                )
                                print(
                                    "[INFO] Structured outputs unavailable for this deployment/session; "
                                    "falling back to prompt-guided JSON parsing."
                                )
                            response = self._create_completion(user_content)
                        else:
                            raise

                text = (response.choices[0].message.content or "").strip()
                last_response_text = text
                data = _parse_model_response_text(text)

                result: Dict[str, List[object]] = {"possible": [], "likely": []}
                if isinstance(data, dict):
                    for key in ("possible", "likely"):
                        if key in data and isinstance(data[key], list):
                            result[key] = [
                                normalized_item
                                for item in data[key]
                                for normalized_item in [_normalize_result_entry(item)]
                                if normalized_item
                            ]
                elif isinstance(data, list):
                    result["likely"] = [
                        normalized_item
                        for item in data
                        for normalized_item in [_normalize_result_entry(item)]
                        if normalized_item
                    ]
                # Hallucination guard: drop any term the LLM emitted that does not
                # actually appear in this chunk. Required because gpt-4o-mini was
                # observed inventing canonical stigma terms (e.g., "elderly",
                # "non-compliant", "uncontrolled diabetic") that never appeared in
                # the source notes. The guard is model-agnostic and is the final
                # filter before the chunk result is cached/returned.
                result, dropped = filter_hallucinated_terms(result, chunk_text)
                if dropped and self._hallucination_guard_verbose:
                    sample = ", ".join(
                        f'{d["bucket"]}:"{d["term"]}"' for d in dropped[:5]
                    )
                    extra = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
                    print(
                        f"[GUARD] Chunk {chunk_index}: dropped {len(dropped)} hallucinated term(s): {sample}{extra}"
                    )
                self._hallucinations_dropped_count += len(dropped)
                if self.config.cache_chunk_responses and cache_key:
                    with self._cache_lock:
                        self._chunk_cache[cache_key] = {
                            "possible": list(result["possible"]),
                            "likely": list(result["likely"]),
                        }
                return result
            except Exception as exc:
                last_err = exc
                if attempt < self.config.max_retries:
                    self._sleep_backoff(attempt)

        print(
            f"[WARN] Failed chunk after {self.config.max_retries} attempts: {last_err}"
        )
        if last_err is not None:
            self._record_chunk_failure(
                chunk_index, chunk_text, last_err, last_response_text
            )
        return {"possible": [], "likely": []}

    def adjudicate_ambiguous_terms(
        self, note_text: str, enriched: Dict[str, object]
    ) -> Dict[str, object]:
        if not self.config.enable_second_pass_adjudication:
            return enriched

        possible_details = list(enriched.get("possible_details", []))
        ambiguous = [
            detail
            for detail in possible_details
            if detail.get("normalized_term", "").lower() in _AMBIGUOUS_POSSIBLE_TERMS
        ]
        if not ambiguous:
            return enriched

        ambiguous = ambiguous[: self.config.ambiguous_term_limit]
        adjudication_prompt = {
            "task": "Review ambiguous potential bias terms from a clinical note. For each term, decide one of: keep_possible, move_likely, drop.",
            "rules": [
                "Return only JSON.",
                "Drop neutral clinical descriptors or neutral informed-refusal language.",
                "Move to likely only when the context clearly stigmatizes, moralizes, or identity-labels the patient.",
                "Keep as possible when context remains ambiguous.",
            ],
            "items": [
                {
                    "term": detail["term"],
                    "normalized_term": detail["normalized_term"],
                    "context": detail.get("context", ""),
                }
                for detail in ambiguous
            ],
            "output_schema": {
                "decisions": [
                    {"term": "<term>", "decision": "keep_possible|move_likely|drop"}
                ]
            },
        }

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_for_api,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(adjudication_prompt, ensure_ascii=False),
                    }
                ],
                max_completion_tokens=512,
                temperature=0,
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else []
        except Exception:
            return enriched

        decision_by_term = {
            str(item.get("term", "")).strip(): str(item.get("decision", "")).strip()
            for item in decisions
            if isinstance(item, dict)
        }
        if not decision_by_term:
            return enriched

        possible_details_kept = []
        likely_details = list(enriched.get("likely_details", []))
        for detail in possible_details:
            decision = decision_by_term.get(detail["term"], "keep_possible")
            if decision == "drop":
                continue
            if decision == "move_likely":
                likely_details.append(detail)
                continue
            possible_details_kept.append(detail)

        enriched["possible_details"] = possible_details_kept
        enriched["likely_details"] = likely_details
        enriched["possible_terms"] = [
            detail["term"] for detail in possible_details_kept
        ]
        enriched["likely_terms"] = [detail["term"] for detail in likely_details]
        enriched["possible_normalized_terms"] = list(
            dict.fromkeys(detail["normalized_term"] for detail in possible_details_kept)
        )
        enriched["likely_normalized_terms"] = list(
            dict.fromkeys(detail["normalized_term"] for detail in likely_details)
        )
        enriched["possible_categories"] = list(
            dict.fromkeys(
                category
                for detail in possible_details_kept
                for category in _collect_detail_categories(detail)
            )
        )
        enriched["likely_categories"] = list(
            dict.fromkeys(
                category
                for detail in likely_details
                for category in _collect_detail_categories(detail)
            )
        )
        return enriched

    def analyze_note_text(self, full_text: str) -> Dict[str, List[object]]:
        if not isinstance(full_text, str) or not full_text.strip():
            return {"possible": [], "likely": []}

        self._last_chunk_failures = []
        chunks = chunk_by_sentences(full_text, self.config.chunk_char_limit)
        if not chunks:
            return {"possible": [], "likely": []}

        if self.config.parallel_calls > 1 and len(chunks) > 1:
            results_by_idx: Dict[int, Dict[str, List[object]]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.parallel_calls
            ) as executor:
                future_map = {
                    executor.submit(self.call_model_on_chunk, chunk, idx): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in concurrent.futures.as_completed(future_map):
                    idx = future_map[future]
                    try:
                        results_by_idx[idx] = future.result() or {
                            "possible": [],
                            "likely": [],
                        }
                    except Exception:
                        results_by_idx[idx] = {"possible": [], "likely": []}
            return self._merge_chunk_results(chunks, results_by_idx)

        sequential_results = {
            idx: self.call_model_on_chunk(chunk, idx)
            for idx, chunk in enumerate(chunks)
        }
        return self._merge_chunk_results(chunks, sequential_results)

    @staticmethod
    def _merge_chunk_results(
        chunks: List[str], results_by_idx: Dict[int, Dict[str, List[object]]]
    ) -> Dict[str, List[object]]:
        aggregated: Dict[str, List[object]] = {"possible": [], "likely": []}

        for bucket in ("possible", "likely"):
            merged_by_key: Dict[str, Dict[str, object]] = {}
            ordered_keys: List[str] = []

            for idx in range(len(chunks)):
                chunk_result = results_by_idx.get(idx, {"possible": [], "likely": []})
                for item in chunk_result.get(bucket, []):
                    normalized_item = _normalize_result_entry(item)
                    if not normalized_item:
                        continue

                    term = _extract_result_term(normalized_item)
                    key = _canonical_term_key(term)
                    if not key:
                        continue

                    categories = _extract_result_categories(normalized_item)
                    if key not in merged_by_key:
                        merged_by_key[key] = {
                            "term": term,
                            "categories": list(categories),
                        }
                        ordered_keys.append(key)
                        continue

                    existing_categories = merged_by_key[key]["categories"]
                    for category in categories:
                        if category not in existing_categories:
                            existing_categories.append(category)

            aggregated[bucket] = [
                {
                    "term": merged_by_key[key]["term"],
                    "categories": list(merged_by_key[key]["categories"]),
                }
                if merged_by_key[key]["categories"]
                else str(merged_by_key[key]["term"])
                for key in ordered_keys
            ]

        return aggregated

    def process_dataframe(
        self, df: pd.DataFrame, text_column: str = "note_text"
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"Input data must contain a '{text_column}' column.")

        processed = df.copy()
        processed["Possible_Biased_Terms"] = "[]"
        processed["Likely_Biased_Terms"] = "[]"
        processed["Possible_Biased_Terms_Normalized"] = "[]"
        processed["Likely_Biased_Terms_Normalized"] = "[]"
        processed["Possible_Bias_Categories"] = "[]"
        processed["Likely_Bias_Categories"] = "[]"
        processed["Possible_Bias_Details"] = "[]"
        processed["Likely_Bias_Details"] = "[]"
        processed["Possible_Bias_Count"] = 0
        processed["Likely_Bias_Count"] = 0
        processed["Note_Word_Count"] = 0
        processed["Chunk_Failure_Count"] = 0
        processed["Chunk_Failure_Details"] = "[]"
        processed["Prompt_Version"] = self.config.prompt_version
        processed["Pipeline_Version"] = self.config.pipeline_version
        processed["Model_Used"] = self.config.model_for_api
        processed["Note_Hash"] = ""

        start_time = time.time()
        cache_hits = 0
        total_rows = len(processed)
        for processed_idx, (row_idx, note_text) in enumerate(
            processed[text_column].items(), start=1
        ):
            note_hash = self._note_hash(note_text)
            cached_payload = self._cached_note_payload(note_hash)
            if cached_payload is not None:
                cache_hits += 1
                for column, value in cached_payload.items():
                    processed.at[row_idx, column] = value
            else:
                self._current_note_hash = note_hash
                result = postprocess_results(self.analyze_note_text(note_text))
                chunk_failures = list(self._last_chunk_failures)
                enriched = enrich_bias_result(note_text, result)
                enriched = self.adjudicate_ambiguous_terms(note_text, enriched)
                payload = {
                    "Possible_Biased_Terms": json.dumps(
                        enriched["possible_terms"], ensure_ascii=False
                    ),
                    "Likely_Biased_Terms": json.dumps(
                        enriched["likely_terms"], ensure_ascii=False
                    ),
                    "Possible_Biased_Terms_Normalized": json.dumps(
                        enriched["possible_normalized_terms"], ensure_ascii=False
                    ),
                    "Likely_Biased_Terms_Normalized": json.dumps(
                        enriched["likely_normalized_terms"], ensure_ascii=False
                    ),
                    "Possible_Bias_Categories": json.dumps(
                        enriched["possible_categories"], ensure_ascii=False
                    ),
                    "Likely_Bias_Categories": json.dumps(
                        enriched["likely_categories"], ensure_ascii=False
                    ),
                    "Possible_Bias_Details": json.dumps(
                        enriched["possible_details"], ensure_ascii=False
                    ),
                    "Likely_Bias_Details": json.dumps(
                        enriched["likely_details"], ensure_ascii=False
                    ),
                    "Possible_Bias_Count": len(enriched["possible_terms"]),
                    "Likely_Bias_Count": len(enriched["likely_terms"]),
                    "Note_Word_Count": enriched["note_word_count"],
                    "Chunk_Failure_Count": len(chunk_failures),
                    "Chunk_Failure_Details": json.dumps(
                        chunk_failures, ensure_ascii=False
                    ),
                    "Prompt_Version": self.config.prompt_version,
                    "Pipeline_Version": self.config.pipeline_version,
                    "Model_Used": self.config.model_for_api,
                    "Note_Hash": note_hash,
                }
                for column, value in payload.items():
                    processed.at[row_idx, column] = value
                if not chunk_failures:
                    self._note_cache[note_hash] = dict(payload)

            if processed_idx % 10 == 0 or processed_idx == total_rows:
                elapsed = time.time() - start_time
                rate = processed_idx / elapsed if elapsed > 0 else 0
                remaining = (total_rows - processed_idx) / rate if rate > 0 else 0
                print(
                    f"  Processed {processed_idx}/{total_rows} notes... "
                    f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)"
                )

        self._write_note_cache()
        print(f"  Cache hits: {cache_hits}/{total_rows}")
        print(
            f"  Hallucination guard dropped: {self._hallucinations_dropped_count} model-emitted term(s) absent from source"
        )
        return processed


def build_analysis_ready_dataframe(
    df: pd.DataFrame, text_column: str = "note_text"
) -> pd.DataFrame:
    analysis_ready = df.copy()
    drop_columns = [
        column
        for column in (
            text_column,
            "Possible_Bias_Details",
            "Likely_Bias_Details",
        )
        if column in analysis_ready.columns
    ]
    return analysis_ready.drop(columns=drop_columns)


def parse_json_list(value) -> List:
    if pd.isna(value) or value in ("", "[]"):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _get_note_id_column(df: pd.DataFrame) -> str:
    for candidate in ("unique_id", "note_id", "patient_study_id", "__row_id__"):
        if candidate in df.columns:
            return candidate
    return "__row_index__"


def build_reviewer_adjudication_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    note_id_column = _get_note_id_column(df)
    rows: List[Dict[str, object]] = []

    for row_index, row in df.iterrows():
        source_note_id = (
            row[note_id_column] if note_id_column in row.index else row_index
        )
        shared_fields = {
            "source_id_column": note_id_column,
            "source_note_id": source_note_id,
            "row_index": row_index,
            "Note_Hash": row.get("Note_Hash", ""),
            "Dax_or_Human": row.get("Dax_or_Human", ""),
            "Prompt_Version": row.get("Prompt_Version", ""),
            "Pipeline_Version": row.get("Pipeline_Version", ""),
            "Model_Used": row.get("Model_Used", ""),
            "adjudication_status": "pending",
            "review_decision": "",
            "reviewed_term": "",
            "reviewed_category": "",
            "reviewer_notes": "",
            "false_positive_term": "",
            "missed_term": "",
            "needs_prompt_update": "",
            "needs_rule_update": "",
        }

        for bucket in ("likely", "possible"):
            details = parse_json_list(row.get(f"{bucket.title()}_Bias_Details", "[]"))
            if not details:
                terms = parse_json_list(row.get(f"{bucket.title()}_Biased_Terms", "[]"))
                normalized_terms = parse_json_list(
                    row.get(f"{bucket.title()}_Biased_Terms_Normalized", "[]")
                )
                categories = parse_json_list(
                    row.get(f"{bucket.title()}_Bias_Categories", "[]")
                )
                details = [
                    {
                        "term": term,
                        "normalized_term": normalized_terms[idx]
                        if idx < len(normalized_terms)
                        else str(term).lower(),
                        "category": categories[idx] if idx < len(categories) else "",
                        "context": "",
                    }
                    for idx, term in enumerate(terms)
                ]

            for detail in details:
                detail_categories = (
                    _collect_detail_categories(detail)
                    if isinstance(detail, dict)
                    else []
                )
                rows.append(
                    {
                        **shared_fields,
                        "model_bucket": bucket,
                        "model_term": detail.get("term", ""),
                        "normalized_term": detail.get("normalized_term", ""),
                        "model_category": detail.get("category", "")
                        or " | ".join(detail_categories),
                        "term_context": detail.get("context", ""),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "source_id_column",
            "source_note_id",
            "row_index",
            "Note_Hash",
            "Dax_or_Human",
            "Prompt_Version",
            "Pipeline_Version",
            "Model_Used",
            "model_bucket",
            "model_term",
            "normalized_term",
            "model_category",
            "term_context",
            "adjudication_status",
            "review_decision",
            "reviewed_term",
            "reviewed_category",
            "reviewer_notes",
            "false_positive_term",
            "missed_term",
            "needs_prompt_update",
            "needs_rule_update",
        ],
    )


def write_output_bundle(
    df: pd.DataFrame, output_csv: str, text_column: str = "note_text"
) -> Dict[str, str]:
    reviewer_path = output_csv
    analysis_path = (
        output_csv[:-4] + "_analysis_ready.csv"
        if output_csv.lower().endswith(".csv")
        else output_csv + "_analysis_ready.csv"
    )
    adjudication_path = (
        output_csv[:-4] + "_reviewer_adjudication.csv"
        if output_csv.lower().endswith(".csv")
        else output_csv + "_reviewer_adjudication.csv"
    )

    df.to_csv(reviewer_path, index=False)
    build_analysis_ready_dataframe(df, text_column=text_column).to_csv(
        analysis_path, index=False
    )
    build_reviewer_adjudication_dataframe(df).to_csv(adjudication_path, index=False)

    return {
        "reviewer_path": reviewer_path,
        "analysis_path": analysis_path,
        "adjudication_path": adjudication_path,
    }
