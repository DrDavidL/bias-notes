#!/usr/bin/env python3
"""Ask Claude on OpenRouter for a bounded second opinion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Claude on OpenRouter for a second opinion."
    )
    parser.add_argument("--prompt", help="Prompt text to send.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt text from stdin and append it after any --prompt text.",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="File whose contents should be embedded into the prompt. Repeatable.",
    )
    parser.add_argument(
        "--context-note",
        action="append",
        default=[],
        help="Additional context note to append before the main question. Repeatable.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model slug. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        dest="max_tokens",
        help="Optional maximum output tokens.",
    )
    parser.add_argument(
        "--site-url",
        default=os.environ.get("OPENROUTER_SITE_URL"),
        help="Optional HTTP-Referer header value. Defaults to OPENROUTER_SITE_URL.",
    )
    parser.add_argument(
        "--site-name",
        default=os.environ.get("OPENROUTER_SITE_NAME"),
        help="Optional X-OpenRouter-Title header value. Defaults to OPENROUTER_SITE_NAME.",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print the raw OpenRouter JSON response.",
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.prompt:
        parts.append(args.prompt.strip())
    if args.stdin:
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            parts.append(stdin_text)
    if not parts:
        raise SystemExit("Provide --prompt, --stdin, or both.")
    return "\n\n".join(parts)


def build_context(args: argparse.Namespace) -> str:
    sections: list[str] = []
    for note in args.context_note:
        note = note.strip()
        if note:
            sections.append(f"Context note:\n{note}")
    for raw_path in args.context_file:
        path = Path(raw_path).expanduser().resolve()
        try:
            text = path.read_text()
        except OSError as exc:
            raise SystemExit(f"Failed to read context file {path}: {exc}") from exc
        sections.append(f"Context file: {path}\n```text\n{text}\n```")
    return "\n\n".join(sections)


def build_final_prompt(args: argparse.Namespace) -> str:
    body = read_prompt(args)
    context = build_context(args)
    if context:
        return (
            "Provide an independent opinion based on the supplied context. "
            "Be concrete, concise, and explicit about tradeoffs.\n\n"
            f"{context}\n\nQuestion:\n{body}"
        )
    return body


def build_payload(args: argparse.Namespace, prompt: str) -> dict:
    payload: dict = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    return payload


def build_request(args: argparse.Namespace, payload: dict) -> urllib.request.Request:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running this script.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if args.site_url:
        headers["HTTP-Referer"] = args.site_url
    if args.site_name:
        headers["X-OpenRouter-Title"] = args.site_name

    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        OPENROUTER_URL, data=data, headers=headers, method="POST"
    )


def extract_text(response_data: dict) -> str:
    choices = response_data.get("choices") or []
    if not choices:
        raise SystemExit("OpenRouter response did not contain any choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()
    raise SystemExit("OpenRouter response did not contain text content.")


def main() -> int:
    args = parse_args()
    prompt = build_final_prompt(args)
    payload = build_payload(args, prompt)
    request = build_request(args, payload)

    try:
        with urllib.request.urlopen(request) as response:
            response_data = json.load(response)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(error_body, file=sys.stderr)
        return exc.code or 1
    except urllib.error.URLError as exc:
        print(f"OpenRouter request failed: {exc}", file=sys.stderr)
        return 1

    if args.raw_json:
        print(json.dumps(response_data, indent=2))
        return 0

    print(extract_text(response_data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
