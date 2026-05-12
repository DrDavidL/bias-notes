import re
from typing import Dict, Iterable, List


_GENERIC_FALSE_POSITIVE_TERMS = {
    "chief complaint",
    "chief complaints",
    "denies",
    "denied",
    "the patient",
    "patient",
    "difficult airway",
    "difficult intubation",
}

_NORMALIZATION_ALIASES = {
    "morbidly obese": "morbid obesity",
    "the diabetic": "diabetic",
    "diabetic patient": "diabetic",
    "sickle cell patient": "sickle cell patient",
    # Smoker variants collapse to the canonical identity noun "smoker" so that
    # diff/consensus analyses count them as the same term across runs. Prior
    # to this fix, these mapped to the literal category name "smoker identity
    # label", which leaked the category into the normalized-term field and
    # inflated cross-run agreement on a value that was not actually a term.
    "current smoker": "smoker",
    "former smoker": "smoker",
    "social smoker": "smoker",
    "tobacco smoker": "smoker",
}

ALLOWED_BIAS_CATEGORIES = [
    "substance identity label",
    "condition identity label",
    "weight-based identity label",
    "moralizing lifestyle language",
    "effort-based language",
    "outcome-judging language",
    "substance-use judgment",
    "behavior description",
    "credibility-doubting language",
    "disapproval / moralizing tone",
    "stereotyping",
    "difficult-patient framing",
    "paternalistic framing",
    "judgmental social risk framing",
    "power/privilege judgment",
    "appearance-based assumption",
    "protected-class stereotyping",
    "autonomy-undermining language",
    "disrespectful / condescending tone",
    "cultural insensitivity",
    "ageism",
    "gender bias",
    "classism",
    "language-bias framing",
    "other / review needed",
]
_CANONICAL_CATEGORY_BY_KEY = {
    re.sub(r"\s+", " ", label.strip()).lower(): label
    for label in ALLOWED_BIAS_CATEGORIES
}
_CATEGORY_SYNONYMS = {
    "substance-related identity-based labels": "substance identity label",
    "condition-based identity labels": "condition identity label",
    "weight-based identity labels": "weight-based identity label",
    "moralizing lifestyle labels": "moralizing lifestyle language",
    "attitude/behavior descriptions": "behavior description",
    "substance-use judgmental language": "substance-use judgment",
    "questioning patient credibility": "credibility-doubting language",
    "difficult patient framing": "difficult-patient framing",
    '"difficult patient" framing': "difficult-patient framing",
    "unilateral, authority-centered framing": "paternalistic framing",
    "social/behavioral risk assessments framed judgmentally": "judgmental social risk framing",
    "power/privilege descriptors used without clinical relevance": "power/privilege judgment",
    "assumptions based on appearance": "appearance-based assumption",
    "language reinforcing stereotypes about protected classes": "protected-class stereotyping",
    "undermining autonomy": "autonomy-undermining language",
    "disrespectful/condescending tone": "disrespectful / condescending tone",
    "language barriers framed pejoratively": "language-bias framing",
}

_CATEGORY_PATTERNS = [
    (
        "substance identity label",
        re.compile(r"\b(smoker|alcoholic|drug user)\b", re.IGNORECASE),
    ),
    (
        "condition identity label",
        re.compile(
            r"\b(diabetic|schizophrenic|bipolar|psychotic|hypertensive|sickle cell patient)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weight-based identity label",
        re.compile(
            r"\b(morbid(?:ly)? obese|morbid obesity|class [23] obesity|obesity|overweight|underweight|obese)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "moralizing lifestyle language",
        re.compile(r"\b(unhealthy diet|sedentary|not motivated)\b", re.IGNORECASE),
    ),
    (
        "effort-based language",
        re.compile(
            r"\b(struggles|suffers|trying to stay active|trying to limit sugar|trying to lose weight|tries to get in exercise|tries to adhere)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "outcome-judging language",
        re.compile(
            r"\b(poorly controlled|well controlled|good control|borderline control|controlled)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "substance-use judgment",
        re.compile(
            r"\b(alcohol abuse|alcohol misuse|substance abuse|tobacco abuse|misuse of tobacco|heavy smoking|binge drinking|excessive alcohol|illicit drug use|heavy marijuana use)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "behavior description",
        re.compile(
            r"\b(aggressive|belligerent|combative|aggression|aggressively|combativeness)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credibility-doubting language",
        re.compile(r"\b(claims|alleges|insists|truthfully)\b", re.IGNORECASE),
    ),
    (
        "difficult-patient framing",
        re.compile(
            r"\b(difficult patient|drug-seeking|malingerer|manipulative|complainer|frequent fl(?:y|i)er)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "paternalistic framing",
        re.compile(
            r"\b(was instructed to|was told to|not allowed to)\b", re.IGNORECASE
        ),
    ),
    ("ageism", re.compile(r"\b(the elderly|elderly|senile|demented)\b", re.IGNORECASE)),
    ("language-bias framing", re.compile(r"\b(non-english speaking)\b", re.IGNORECASE)),
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=\S[.!?])\s+(?=[\"'([{]*[A-Z0-9])", re.VERBOSE)


def normalize_term(term: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(term).strip()).strip(" ,.;:!?()[]{}\"'")
    lowered = cleaned.lower()
    return _NORMALIZATION_ALIASES.get(lowered, lowered)


def is_generic_false_positive(term: str) -> bool:
    return normalize_term(term) in _GENERIC_FALSE_POSITIVE_TERMS


def infer_bias_category(term: str) -> str:
    for label, pattern in _CATEGORY_PATTERNS:
        if pattern.search(term):
            return label
    return "other / review needed"


def canonicalize_bias_category(category: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(category).strip().strip(" ,.;:!?()[]{}\"'"))
    if not cleaned:
        return ""

    lowered = cleaned.lower().replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    canonical = _CANONICAL_CATEGORY_BY_KEY.get(lowered)
    if canonical:
        return canonical

    mapped = _CATEGORY_SYNONYMS.get(lowered, "")
    if mapped:
        return mapped
    return ""


def canonicalize_bias_categories(categories: Iterable[str]) -> List[str]:
    canonical: List[str] = []
    seen = set()
    for category in categories:
        normalized = canonicalize_bias_category(category)
        if normalized and normalized not in seen:
            canonical.append(normalized)
            seen.add(normalized)
    return canonical


def _split_sentences(note_text: str) -> List[str]:
    if not isinstance(note_text, str):
        return []
    normalized = re.sub(r"\s+", " ", note_text).strip()
    if not normalized:
        return []
    return [
        segment.strip()
        for segment in _SENTENCE_SPLIT_RE.split(normalized)
        if segment.strip()
    ]


def extract_term_context(note_text: str, term: str, fallback_radius: int = 120) -> str:
    if not isinstance(note_text, str) or not note_text.strip():
        return ""

    pattern = re.compile(re.escape(term), re.IGNORECASE)
    for sentence in _split_sentences(note_text):
        if pattern.search(sentence):
            return sentence

    match = pattern.search(note_text)
    if not match:
        return ""

    start = max(0, match.start() - fallback_radius)
    end = min(len(note_text), match.end() + fallback_radius)
    snippet = note_text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(note_text):
        snippet = f"{snippet}..."
    return re.sub(r"\s+", " ", snippet)


def count_note_words(note_text: str) -> int:
    if not isinstance(note_text, str):
        return 0
    return len(re.findall(r"\b\w+\b", note_text))


def enrich_bias_result(
    note_text: str, result: Dict[str, List[object]]
) -> Dict[str, object]:
    enriched: Dict[str, object] = {
        "possible_terms": [],
        "likely_terms": [],
        "possible_normalized_terms": [],
        "likely_normalized_terms": [],
        "possible_categories": [],
        "likely_categories": [],
        "possible_details": [],
        "likely_details": [],
        "note_word_count": count_note_words(note_text),
    }

    for bucket in ("possible", "likely"):
        seen_terms = set()
        seen_normalized = set()
        seen_categories = set()
        details: List[Dict[str, object]] = []
        kept_terms: List[str] = []
        normalized_terms: List[str] = []
        bucket_categories: List[str] = []

        for item in result.get(bucket, []):
            if isinstance(item, dict):
                raw_term = next(
                    (
                        re.sub(r"\s+", " ", str(item.get(key, "")).strip())
                        for key in ("term", "phrase", "text")
                        if str(item.get(key, "")).strip()
                    ),
                    "",
                )
                raw_categories = item.get("categories")
                if raw_categories is None and item.get("category"):
                    raw_categories = [item.get("category")]
                elif isinstance(raw_categories, str):
                    raw_categories = [raw_categories]
            else:
                raw_term = re.sub(r"\s+", " ", str(item).strip())
                raw_categories = []
            if not raw_term or is_generic_false_positive(raw_term):
                continue

            normalized = normalize_term(raw_term)
            if normalized in seen_terms:
                continue
            seen_terms.add(normalized)

            detail_categories = canonicalize_bias_categories(raw_categories or [])
            if not detail_categories:
                detail_categories = [infer_bias_category(raw_term)]
            category = detail_categories[0]
            context = extract_term_context(note_text, raw_term)

            kept_terms.append(raw_term)
            if normalized not in seen_normalized:
                normalized_terms.append(normalized)
                seen_normalized.add(normalized)
            for category_label in detail_categories:
                if category_label not in seen_categories:
                    bucket_categories.append(category_label)
                    seen_categories.add(category_label)

            details.append(
                {
                    "term": raw_term,
                    "normalized_term": normalized,
                    "category": category,
                    "categories": detail_categories,
                    "context": context,
                }
            )

        enriched[f"{bucket}_terms"] = kept_terms
        enriched[f"{bucket}_normalized_terms"] = normalized_terms
        enriched[f"{bucket}_categories"] = bucket_categories
        enriched[f"{bucket}_details"] = details

    return enriched
