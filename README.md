# SABench Artifact

This repository contains the anonymous artifact for SABench, a benchmark for
side-effect-aware evaluation of tool-using language model agents.

The canonical dataset metadata and tabular task dataset are hosted separately:

- Dataset URL: https://huggingface.co/datasets/anon-sabench-2026/sabench
- Croissant file for OpenReview: `croissant.json`

## Contents

- `sabench_re/`: task loading, tool schema, state replay, and deterministic oracle.
- `scripts/run_pilot_streaming.py`: OpenAI-compatible agent runner.
- `scripts/oracle_verify.py`: verify one task/trace pair.
- `traces/`: saved tool-call traces for the paper experiments.

The benchmark tasks are hosted in the Hugging Face dataset repo as `tasks.jsonl`.
This artifact repository contains the code and saved model traces.

## Installation

```bash
python -m pip install -e .
python -m pip install openai
```

The core oracle has no external runtime dependency beyond Python 3.11+.
The runner requires the `openai` package and an OpenAI-compatible chat-completions
endpoint.

## Verify One Trace

Download or materialize a task JSON from the Hugging Face dataset, then run:

```bash
python scripts/oracle_verify.py \
  /path/to/I-OR-1-001.json \
  traces/gpt-5.4/I/OR/I-OR-1-001.json
```

## Run Agents

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=... \
python scripts/run_pilot_streaming.py \
  --tasks-dir /path/to/tasks \
  --results-dir results/new_run \
  --agents gpt-5.4 \
  --max-steps 12
```

Multiple agents can be supplied as a comma-separated list. The runner writes raw
tool-call traces, oracle reports, and CSV/JSON summaries under the selected
results directory.
