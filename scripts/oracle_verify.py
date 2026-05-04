#!/usr/bin/env python3
"""Run the deterministic oracle for one task and one trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sabench_re.io import load_task, load_trace
from sabench_re.oracle import oracle_verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify one SABENCH_RE trace")
    parser.add_argument("task")
    parser.add_argument("trace")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = oracle_verify(load_task(args.task), load_trace(args.trace))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
