---
name: claude-openrouter-opinion
description: Ask Claude through OpenRouter for an independent opinion, critique, or alternative approach. Use when Codex should get a second model's view before making a decision, compare implementation strategies, sanity-check a design, review a draft prompt or plan, or summarize another model's answer with a clear recommendation.
---

# Claude OpenRouter Opinion

Use the bundled script to ask Claude on OpenRouter for a bounded second opinion with low side effects.

## Configure Authentication

Set the API key in an environment variable before running the script:

```bash
export OPENROUTER_API_KEY='...'
```

Prefer an environment variable over hardcoding the key into the skill or committing it to a repo. If you want a persistent local setup, put the export in your shell profile or in a local secrets file that is not committed.

Optional headers:

- `OPENROUTER_SITE_URL` for `HTTP-Referer`
- `OPENROUTER_SITE_NAME` for `X-OpenRouter-Title`

## Workflow

1. Decide whether Claude needs raw context or just a focused question.
2. If files matter, pass them with `--context-file` so the exact text is embedded in the request.
3. Ask for a specific output shape: recommendation, risks, tradeoffs, ranking, or critique.
4. Treat Claude's answer as an input to judgment, not as authority.

## Run The Wrapper

Use `scripts/ask_openrouter_claude.py`. The wrapper:

- calls OpenRouter directly
- defaults to `anthropic/claude-sonnet-4.6`
- accepts context files and inline notes
- returns plain text by default
- can emit the raw OpenRouter JSON with `--raw-json`

Example:

```bash
python3 scripts/ask_openrouter_claude.py \
  --prompt "Review these two migration options and recommend one." \
  --context-file /abs/path/plan.md \
  --context-file /abs/path/schema.sql
```

Use stdin for longer prompts:

```bash
cat question.txt | python3 scripts/ask_openrouter_claude.py --stdin
```

## Prompting Guidance

Ask narrow questions. Good patterns:

- "Choose between approach A and B. Give one recommendation and three risks."
- "Critique this plan. Focus on hidden failure modes and missing validation."
- "Summarize the strongest counterargument to this design."
- "Given the attached diff, identify likely regressions."

If you need a different model, pass `--model`.

## Constraints

- Keep secrets out of prompts unless the user explicitly accepts that risk.
- Prefer passing relevant file contents with `--context-file` over granting any remote tool access.
- If OpenRouter returns an auth, quota, or upstream-model error, surface the response clearly and continue without blocking on the skill.
