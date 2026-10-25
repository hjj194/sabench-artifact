# SABench

Code and evaluation traces for **SABench** (Side-effect-Aware Benchmark), a
benchmark for side-effect-aware evaluation of tool-using language model agents.

> Jiaju Huang, Shaobin Chen, Xinglong Liang, Xinyu Ma, Hao Yang, Wei Ke, Tao Tan, Yue Sun.
> *Task Success Is Not Enough: Side-effect-Aware Evaluation of Tool-Using Language Model Agents.*
> NeurIPS 2026, Track on Evaluations and Datasets.

- Dataset (201 tasks, `tasks.jsonl`) and Croissant metadata: https://huggingface.co/datasets/Phaedrus/sabench
- Code and traces: https://github.com/hjj194/sabench-artifact

## Contents

- `sabench_re/`: task loading, tool schema, state replay, and deterministic oracle.
- `scripts/run_pilot_streaming.py`: OpenAI-compatible agent runner.
- `scripts/oracle_verify.py`: verify one task/trace pair.
- `traces/`: saved tool-call traces for the paper experiments
  (11 models x 201 tasks = 2,211 traces).

The benchmark tasks are hosted in the Hugging Face dataset repo as `tasks.jsonl`.
This repository contains the code and saved model traces.

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

## License

- Code (`sabench_re/`, `scripts/`): [MIT](LICENSE).
- Traces (`traces/`) and the Hugging Face dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

All names, email addresses, organizations, and workspace records in the tasks
and traces are synthetic.

## Citation

```bibtex
@inproceedings{huang2026sabench,
  title     = {Task Success Is Not Enough: Side-effect-Aware Evaluation of Tool-Using Language Model Agents},
  author    = {Huang, Jiaju and Chen, Shaobin and Liang, Xinglong and Ma, Xinyu and Yang, Hao and Ke, Wei and Tan, Tao and Sun, Yue},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026}
}
```

## Contact

Jiaju Huang (hjj194@gmail.com). Corresponding authors: Tao Tan (taotan@mpu.edu.mo)
and Yue Sun (yuesun@mpu.edu.mo).
