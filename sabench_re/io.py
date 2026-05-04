"""JSON IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import Task, Trace


def load_task(path: str | Path) -> Task:
    with Path(path).open() as f:
        return Task.from_dict(json.load(f))


def save_task(task: Task, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(task.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def load_trace(path: str | Path) -> Trace:
    with Path(path).open() as f:
        return Trace.from_dict(json.load(f))


def load_tasks(root: str | Path) -> list[Task]:
    root = Path(root)
    if root.is_file():
        with root.open() as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return [Task.from_dict(item) for item in raw]
        return [Task.from_dict(raw)]
    return [load_task(path) for path in sorted(root.glob("**/*.json"))]


def save_tasks(tasks: Iterable[Task], root: str | Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        path = root / task.state_type / task.trap_type / f"{task.task_id}.json"
        save_task(task, path)
