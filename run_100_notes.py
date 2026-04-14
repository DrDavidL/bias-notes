"""Backward-compatible wrapper for the script-first batch runner."""

from run_bias_batch import main


if __name__ == "__main__":
    raise SystemExit(main())
