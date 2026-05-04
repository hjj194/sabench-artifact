"""Data models for SABENCH_RE.

The benchmark oracle compares successful write operations from a trace against
the task's authorized write set. These models keep that surface explicit and
serializable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


def _freeze_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_calendar_create_title(value: str) -> str:
    title = " ".join(str(value).strip().split()).lower()
    title = re.sub(r"\s+-\s+[a-z][a-z .'-]+$", "", title)
    title = re.sub(r"\s+with\s+[a-z][a-z .'-]+$", "", title)
    return title


def _comparable_write_parts(
    tool: str,
    action: str,
    entity_id: str,
    fields: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    comparable_fields = dict(fields)
    comparable_entity_id = entity_id
    if tool == "calendar" and action == "create":
        comparable_fields["title"] = _normalize_calendar_create_title(str(comparable_fields.get("title", "")))
        comparable_entity_id = f"calendar_create:{comparable_fields.get('title', '')}:{comparable_fields.get('time', '')}"
    return comparable_entity_id, comparable_fields


@dataclass(frozen=True)
class WriteOperation:
    tool: str
    action: str
    entity_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    is_external: bool = False
    step: int = -1

    def comparable(self) -> tuple[str, str, str, str, bool]:
        """Key used for set equality; step is provenance, not authorization."""
        entity_id, fields = _comparable_write_parts(self.tool, self.action, self.entity_id, self.fields)
        return (
            self.tool,
            self.action,
            entity_id,
            _freeze_json(fields),
            self.is_external,
        )

    def with_step(self, step: int) -> "WriteOperation":
        return WriteOperation(
            tool=self.tool,
            action=self.action,
            entity_id=self.entity_id,
            fields=dict(self.fields),
            is_external=self.is_external,
            step=step,
        )

    def __hash__(self) -> int:
        return hash(self.comparable())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WriteOperation):
            return NotImplemented
        return self.comparable() == other.comparable()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "action": self.action,
            "entity_id": self.entity_id,
            "fields": self.fields,
            "is_external": self.is_external,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WriteOperation":
        return cls(
            tool=data["tool"],
            action=data["action"],
            entity_id=data["entity_id"],
            fields=dict(data.get("fields") or {}),
            is_external=bool(data.get("is_external", False)),
            step=int(data.get("step", -1)),
        )


@dataclass
class Task:
    task_id: str
    template_id: str
    state_type: str
    trap_type: str
    s0: dict[str, Any]
    x: str
    G: list[str]
    A_explicit: frozenset[WriteOperation]
    is_ambiguous: bool = False
    candidate_entities: list[str] = field(default_factory=list)
    decoy_entity_id: str | None = None
    dependency_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "template_id": self.template_id,
            "state_type": self.state_type,
            "trap_type": self.trap_type,
            "s0": self.s0,
            "x": self.x,
            "G": list(self.G),
            "A_explicit": sorted(
                (op.to_dict() for op in self.A_explicit),
                key=lambda op: (op["tool"], op["action"], op["entity_id"], _freeze_json(op["fields"])),
            ),
            "is_ambiguous": self.is_ambiguous,
            "candidate_entities": list(self.candidate_entities),
            "decoy_entity_id": self.decoy_entity_id,
            "dependency_type": self.dependency_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"],
            template_id=data["template_id"],
            state_type=data["state_type"],
            trap_type=data["trap_type"],
            s0=dict(data["s0"]),
            x=data["x"],
            G=list(data.get("G") or []),
            A_explicit=frozenset(WriteOperation.from_dict(op) for op in data.get("A_explicit", [])),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            candidate_entities=list(data.get("candidate_entities") or []),
            decoy_entity_id=data.get("decoy_entity_id"),
            dependency_type=data.get("dependency_type"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class APICall:
    tool: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    step: int = 0
    agent_message: str = ""

    @property
    def api_name(self) -> str:
        return f"{self.tool}_{self.action}"

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_step: int = 0) -> "APICall":
        tool = data.get("tool")
        action = data.get("action")
        api_name = data.get("api_name") or data.get("name")
        if (not tool or not action) and api_name:
            normalized = str(api_name).replace(".", "_")
            tool, action = normalized.split("_", 1)
        return cls(
            tool=str(tool),
            action=str(action),
            args=dict(data.get("args") or data.get("arguments") or {}),
            success=bool(data.get("success", True)),
            step=int(data.get("step", fallback_step)),
            agent_message=str(data.get("agent_message", "")),
        )


@dataclass
class Trace:
    calls: list[APICall] = field(default_factory=list)
    final_response: str = ""

    def all_agent_text(self) -> str:
        parts = [call.agent_message for call in self.calls if call.agent_message]
        if self.final_response:
            parts.append(self.final_response)
        return "\n".join(parts)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[dict[str, Any]]) -> "Trace":
        if isinstance(data, list):
            return cls(calls=[APICall.from_dict(call, i) for i, call in enumerate(data)])
        raw_calls = data.get("tool_calls") or data.get("calls") or data.get("trace") or []
        return cls(
            calls=[APICall.from_dict(call, i) for i, call in enumerate(raw_calls)],
            final_response=str(data.get("final_response", "")),
        )


@dataclass
class CDReport:
    task_id: str
    W: frozenset[WriteOperation]
    A_explicit: frozenset[WriteOperation]
    A_implicit: frozenset[WriteOperation]
    A: frozenset[WriteOperation]
    cd: frozenset[WriteOperation]
    goal_satisfied: bool
    external_cd: frozenset[WriteOperation]
    severity: str
    ab_violation: bool | None = None
    clarification_valid: bool | None = None
    clarification_before_write: bool | None = None
    scope_amplification: int | None = None

    def to_dict(self) -> dict[str, Any]:
        def ops(values: frozenset[WriteOperation]) -> list[dict[str, Any]]:
            return sorted(
                (op.to_dict() for op in values),
                key=lambda op: (op["step"], op["tool"], op["action"], op["entity_id"]),
            )

        return {
            "task_id": self.task_id,
            "W": ops(self.W),
            "A_explicit": ops(self.A_explicit),
            "A_implicit": ops(self.A_implicit),
            "A": ops(self.A),
            "cd": ops(self.cd),
            "goal_satisfied": self.goal_satisfied,
            "external_cd": ops(self.external_cd),
            "severity": self.severity,
            "ab_violation": self.ab_violation,
            "clarification_valid": self.clarification_valid,
            "clarification_before_write": self.clarification_before_write,
            "scope_amplification": self.scope_amplification,
        }
