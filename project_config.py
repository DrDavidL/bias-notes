import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "bias_detection_prompt.py"
DEFAULT_CACHE_PATH = DEFAULT_OUTPUT_DIR / "bias_detection_cache.csv"


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


BIAS_NOTES_RAW_CSV = get_env("BIAS_NOTES_RAW_CSV")
BIAS_NOTES_CLEANED_CSV = get_env("BIAS_NOTES_CLEANED_CSV")
BIAS_NOTES_OUTPUT_DIR = get_env("BIAS_NOTES_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
BIAS_NOTES_PROMPT_PATH = get_env("BIAS_NOTES_PROMPT_PATH", str(DEFAULT_PROMPT_PATH))
BIAS_NOTES_CACHE_PATH = get_env("BIAS_NOTES_CACHE_PATH", str(DEFAULT_CACHE_PATH))
BIAS_NOTES_RESULTS_FILE = get_env("BIAS_NOTES_RESULTS_FILE")
