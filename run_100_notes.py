"""Run bias detection on 100 randomly selected notes using the updated prompt."""

import os, re, json, time, random, importlib.util
from datetime import datetime
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import concurrent.futures
from threading import Semaphore

load_dotenv()

# ---------- AZURE OPENAI CLIENT ----------
client = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
)

# ---------- CONFIG ----------
INPUT_CSV          = "/Users/dli989/Library/CloudStorage/OneDrive-NorthwesternUniversity/all_500_cleaned.csv"
OUTPUT_DIR         = "/Users/dli989/Documents/GitHub/bias-notes/output"
OUTPUT_PREFIX      = "bias_flagged_updated"
PROMPT_PATH        = "/Users/dli989/Documents/GitHub/bias-notes/bias_detection_prompt.py"

N_ROWS_TO_PROCESS  = 100
RANDOM_SELECTION   = True
RANDOM_SEED        = 42          # reproducible selection

CHUNK_CHAR_LIMIT   = 2800
MAX_RETRIES        = 5
SLEEP_BASE_SEC     = 1.4
TEMPERATURE        = 0
PARALLEL_CALLS     = 4
GLOBAL_MAX_CALLS   = 8

MODEL_FOR_API = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_MODEL_NAME")
if not MODEL_FOR_API:
    raise RuntimeError("No Azure deployment/model specified.")

_LLM_SEMAPHORE = Semaphore(GLOBAL_MAX_CALLS)

# ---------- OUTPUT PATH ----------
def generate_output_path(output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{prefix}_{timestamp}.csv")

OUTPUT_CSV = generate_output_path(OUTPUT_DIR, OUTPUT_PREFIX)

# ---------- PROMPT LOADING ----------
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
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

PROMPT_TEXT = load_prompt_text(PROMPT_PATH)

# ---------- SENTENCE CHUNKING ----------
_SENT_SPLIT_RE = re.compile(
    r"(?<=\S[.!?])\s+(?=[\"'([{]*[A-Z0-9])",
    re.VERBOSE,
)

def chunk_by_sentences(text: str, max_chars: int = CHUNK_CHAR_LIMIT) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    t = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
    if not sents:
        return []
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        add_len = len(s) + (1 if cur else 0)
        if cur_len + add_len <= max_chars:
            cur.append(s)
            cur_len += add_len
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            if len(s) > max_chars:
                piece = s
                while len(piece) > max_chars:
                    chunks.append(piece[:max_chars])
                    piece = piece[max_chars:]
                cur, cur_len = ([piece], len(piece)) if piece else ([], 0)
            else:
                cur, cur_len = [s], len(s)
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

# ---------- PROMPT INJECTION ----------
_PLACEHOLDER_PATTERNS = [
    r"<<<\s*\{PASTE THE NOTE TEXT/CHUNK HERE\}\s*>>>",
    r"\{CHUNK\}",
    r"\{\{CHUNK\}\}",
    r"<<<\s*\{CHUNK\}\s*>>>",
]

def inject_chunk_into_prompt(prompt_text: str, chunk_text: str) -> str:
    for pat in _PLACEHOLDER_PATTERNS:
        if re.search(pat, prompt_text):
            def _repl(_m):
                if "<<<" in pat and ">>>" in pat:
                    return f"<<<\n{chunk_text}\n>>>"
                return chunk_text
            return re.sub(pat, _repl, prompt_text, count=1)
    block_re = re.compile(r"(INPUT\s+NOTE\s+CHUNK.*?<<<)([\s\S]*?)(>>>)([\s\S]*)", re.IGNORECASE)
    m = block_re.search(prompt_text)
    if m:
        return m.group(1) + "\n" + chunk_text + "\n" + m.group(3) + m.group(4)
    return f"{prompt_text.rstrip()}\n\nINPUT NOTE CHUNK\n<<<\n{chunk_text}\n>>>"

# ---------- LLM CALL ----------
def _sleep_backoff(attempt: int) -> None:
    delay = SLEEP_BASE_SEC * (2 ** (attempt - 1)) * random.uniform(0.7, 1.3)
    time.sleep(delay)

def call_model_on_chunk(chunk_text: str) -> Dict[str, List[str]]:
    user_content = inject_chunk_into_prompt(PROMPT_TEXT, chunk_text)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _LLM_SEMAPHORE:
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_FOR_API,
                        messages=[{"role": "user", "content": user_content}],
                        max_completion_tokens=1024,
                        temperature=TEMPERATURE,
                    )
                except TypeError:
                    resp = client.chat.completions.create(
                        model=MODEL_FOR_API,
                        messages=[{"role": "user", "content": user_content}],
                        max_tokens=1024,
                        temperature=TEMPERATURE,
                    )
            text = (resp.choices[0].message.content or "").strip()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{[\s\S]*\}", text)
                if not m:
                    return {"possible": [], "likely": []}
                data = json.loads(m.group(0))

            result = {"possible": [], "likely": []}
            if isinstance(data, dict):
                for key in ["possible", "likely"]:
                    if key in data and isinstance(data[key], list):
                        result[key] = [
                            x.strip() if isinstance(x, str) else str(x).strip()
                            for x in data[key] if x
                        ]
            elif isinstance(data, list):
                result["likely"] = [
                    x.strip() if isinstance(x, str) else str(x).strip()
                    for x in data if x
                ]
            return result
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)
    print(f"[WARN] Failed chunk after {MAX_RETRIES} attempts: {last_err}")
    return {"possible": [], "likely": []}

# ---------- PER-NOTE AGGREGATION ----------
def analyze_note_text(full_text: str) -> Dict[str, List[str]]:
    if not isinstance(full_text, str) or not full_text.strip():
        return {"possible": [], "likely": []}
    chunks = chunk_by_sentences(full_text, max_chars=CHUNK_CHAR_LIMIT)
    if not chunks:
        return {"possible": [], "likely": []}

    if PARALLEL_CALLS > 1 and len(chunks) > 1:
        results_by_idx: Dict[int, Dict[str, List[str]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as ex:
            future_map = {ex.submit(call_model_on_chunk, ch): idx for idx, ch in enumerate(chunks)}
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                try:
                    results_by_idx[idx] = fut.result() or {"possible": [], "likely": []}
                except Exception:
                    results_by_idx[idx] = {"possible": [], "likely": []}
        seen_possible, seen_likely = set(), set()
        aggregated = {"possible": [], "likely": []}
        for idx in range(len(chunks)):
            chunk_result = results_by_idx.get(idx, {"possible": [], "likely": []})
            for p in chunk_result.get("possible", []):
                if p and p not in seen_possible:
                    seen_possible.add(p)
                    aggregated["possible"].append(p)
            for p in chunk_result.get("likely", []):
                if p and p not in seen_likely:
                    seen_likely.add(p)
                    aggregated["likely"].append(p)
        return aggregated

    seen_possible, seen_likely = set(), set()
    aggregated = {"possible": [], "likely": []}
    for ch in chunks:
        chunk_result = call_model_on_chunk(ch)
        for p in chunk_result.get("possible", []):
            if p and p not in seen_possible:
                seen_possible.add(p)
                aggregated["possible"].append(p)
        for p in chunk_result.get("likely", []):
            if p and p not in seen_likely:
                seen_likely.add(p)
                aggregated["likely"].append(p)
    return aggregated

# ---------- POST-PROCESSING: PHYSIOLOGIC FALSE POSITIVE FILTER ----------
# These patterns match objective physical exam / physiologic findings that should
# NOT be flagged as bias. They use "normal" to describe organ-system exam results,
# not normative judgments about the patient's appearance, behavior, or mental status.
_PHYSIOLOGIC_NORMAL_PATTERNS = [
    # Cardiovascular
    r"(?:normal\s+)?(?:heart\s+)?rate(?:\s+(?:and|is|:))?\s*(?:normal|regular)",
    r"normal\s+(?:heart\s+sounds|pulses|rate)",
    r"(?:heart\s+sounds|pulses)\s*(?::|are|is)?\s*normal",
    r"regular\s+rhythm",
    # Pulmonary
    r"(?:normal\s+)?(?:bilateral\s+)?breath\s+sounds\s*(?::|are|is)?\s*normal",
    r"(?:normal\s+)?(?:bilateral\s+)?breath\s+sounds",
    r"(?:lung|lungs)\s*(?::|exam|were|are|is)?\s*(?:normal|clear)",
    r"pulmonary\s+effort\s+(?:is\s+)?normal",
    r"air\s+movement\s+(?:is\s+)?normal",
    r"respirations?\s*(?::|are|is)?\s*normal",
    r"normal\s+breath\s+sounds",
    # GI
    r"(?:normal\s+)?bowel\s+sounds\s*(?::|are|is)?\s*normal",
    r"bowel\s+sounds\s+(?:are\s+)?normal",
    r"normal\s+bowel\s+sounds",
    # Musculoskeletal
    r"normal\s+(?:range\s+of\s+motion|ROM)",
    r"normal\s+ROM",
    # Neurological
    r"coordination\s*(?::|is)?\s*normal",
    r"normal\s+coordination",
    # Ophthalmologic
    r"conjunctivae?\s*(?::|are|is)?\s*normal",
    r"normal\s+conjunctivae?",
    # ENT
    r"(?:nose|mouth|ears?|throat|pharynx|tympanic)\s*(?::|are|is)?\s*normal",
    r"normal\s+(?:nose|mouth|ears?|throat|pharynx|tympanic)",
    # GYN
    r"normal\s+(?:labia|vaginal\s+canal|cervix|vulva|uterus|adnexa)",
    r"(?:labia|vaginal\s+canal|cervix|vulva|uterus|adnexa)\s*(?::|are|is)?\s*normal",
    # Vitals
    r"blood\s+pressure\s+(?:is\s+)?normal",
    r"(?:heart\s+rate|HR)\s*(?::|is)?\s*normal",
    # General exam
    r"normal\s+exam",
    r"(?:lung|cardiac|abdominal|pelvic|neurological|musculoskeletal)\s+exam\s+normal",
    # Effort (pulmonary context)
    r"effort\s*(?::|is)?\s*(?:pulmonary\s+effort\s+is\s+)?normal",
    # Generic organ-system findings
    r"normal\s+range\b",
    r"low\s+range\s+of\s+normal",
    r"nutritional\s+status\s*(?::|is)?\s*normal",
    # Lab / endocrine / imaging findings
    r"normal\s+(?:thyroid|liver|renal|kidney|cardiac|pulmonary)\s+function",
    r"normal\s+(?:labs?|imaging|x-ray|ct|mri|ultrasound|echo)",
    # Vaccine / medication reactions
    r"normal\s+(?:vaccine|medication|drug)\s+reactions?",
    # Skin / dermatologic
    r"(?:skin|nails?|hair)\s*(?::|are|is)?\s*normal",
    r"normal\s+(?:skin|nails?|hair)",
]

_PHYSIOLOGIC_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PHYSIOLOGIC_NORMAL_PATTERNS),
    re.IGNORECASE,
)

# Terms that should ALWAYS be kept (mental status, appearance, body norms)
_KEEP_PATTERNS = re.compile(
    r"(?:mood|behavior|behaviour|judgment|thought\s+content|thought\s+process|"
    r"mental\s+status|attention|perception|affect|appearance|gait|weight)",
    re.IGNORECASE,
)

# Patterns that are documentation artifacts, not bias
_DOCUMENTATION_ARTIFACTS = re.compile(
    r"(?:"
    r"tobacco\s+smoker\s*:\s*no"        # screening field showing negative
    r"|smoking\s+status\s*:\s*"          # screening field label
    r"|mood\s+and\s+affect\s+congruent"  # clinical observation, not normative
    r")",
    re.IGNORECASE,
)

def is_physiologic_false_positive(term: str) -> bool:
    """Return True if the term is an objective physiologic finding, not a bias flag."""
    t = term.strip()
    # Filter documentation artifacts
    if _DOCUMENTATION_ARTIFACTS.search(t):
        return True
    # If it matches a mental status / appearance / body norm keyword, keep it
    if _KEEP_PATTERNS.search(t):
        return False
    # If it matches a physiologic pattern, filter it out
    if _PHYSIOLOGIC_RE.search(t):
        return True
    # Catch remaining bare "normal" or "Normal" without mental status context
    if re.fullmatch(r"normal\.?", t, re.IGNORECASE):
        return True
    return False

def postprocess_results(result: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove physiologic false positives from LLM results."""
    return {
        "possible": [t for t in result["possible"] if not is_physiologic_false_positive(t)],
        "likely": [t for t in result["likely"] if not is_physiologic_false_positive(t)],
    }

# ---------- MAIN ----------
if __name__ == "__main__":
    df_full = pd.read_csv(INPUT_CSV)
    if "note_text" not in df_full.columns:
        raise ValueError("Input CSV must contain a 'note_text' column.")

    total_rows = len(df_full)
    n_to_process = min(N_ROWS_TO_PROCESS, total_rows)

    if RANDOM_SELECTION and n_to_process < total_rows:
        df = df_full.sample(n=n_to_process, random_state=RANDOM_SEED).reset_index(drop=True)
        selection_mode = f"random sample (seed={RANDOM_SEED})"
    else:
        df = df_full.head(n_to_process).reset_index(drop=True)
        selection_mode = "first N rows" if n_to_process < total_rows else "all rows"

    print(f"{'='*60}")
    print(f"BIAS DETECTION — UPDATED PROMPT")
    print(f"{'='*60}")
    print(f"Input file: {INPUT_CSV}")
    print(f"Total rows in file: {total_rows}")
    print(f"Rows to process: {n_to_process} ({selection_mode})")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Model: {MODEL_FOR_API}")
    print(f"PARALLEL_CALLS={PARALLEL_CALLS}")
    print(f"{'='*60}\n")

    df["Possible_Biased_Terms"] = "[]"
    df["Likely_Biased_Terms"] = "[]"

    start_time = time.time()
    for i, (row_idx, txt) in enumerate(df["note_text"].items(), start=1):
        result = postprocess_results(analyze_note_text(txt))
        df.at[row_idx, "Possible_Biased_Terms"] = json.dumps(result["possible"], ensure_ascii=False)
        df.at[row_idx, "Likely_Biased_Terms"] = json.dumps(result["likely"], ensure_ascii=False)
        if i % 10 == 0 or i == n_to_process:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (n_to_process - i) / rate if rate > 0 else 0
            print(f"  Processed {i}/{n_to_process} notes... ({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")

    df.to_csv(OUTPUT_CSV, index=False)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total notes: {len(df)}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(df):.1f}s per note)")
