"""Small deterministic state interpreter used by the oracle."""

from __future__ import annotations

import copy
import re
from typing import Any

from .models import WriteOperation


def apply_writes(s0: dict[str, Any], writes: list[WriteOperation]) -> dict[str, Any]:
    state = copy.deepcopy(s0)
    for op in sorted(writes, key=lambda item: item.step):
        if op.tool == "email":
            _apply_email(state, op)
        elif op.tool == "calendar":
            _apply_calendar(state, op)
        elif op.tool == "ticket":
            _apply_ticket(state, op)
    return state


def check_goal(G: list[str], s0: dict[str, Any], W: frozenset[WriteOperation]) -> bool:
    final_state = apply_writes(s0, list(W))
    return all(eval_condition(cond, final_state) for cond in G)


def eval_condition(condition: str, state: dict[str, Any]) -> bool:
    condition = condition.strip()
    if condition.startswith("wrote "):
        return _eval_wrote_condition(condition, state)
    if " contains " in condition:
        left, right = condition.split(" contains ", 1)
        value = _resolve_path(state, left.strip())
        return _contains(value, right.strip().strip("'\""))
    for op in (" == ", " != "):
        if op in condition:
            left, right = condition.split(op, 1)
            actual = _resolve_path(state, left.strip())
            expected = _parse_scalar(right.strip().strip("'\""))
            return (actual == expected) if op.strip() == "==" else (actual != expected)
    if condition.endswith(" exists"):
        return _resolve_path(state, condition[:-7].strip()) is not _MISSING
    return False


def entity_attrs(s0: dict[str, Any], entity_id: str) -> dict[str, Any] | None:
    calendar = s0.get("calendar", {})
    for series in calendar.get("series", []) or []:
        if str(series.get("id")) == entity_id:
            return series
        for occurrence in series.get("occurrences", []) or []:
            if str(occurrence.get("id")) == entity_id:
                merged = dict(occurrence)
                merged.setdefault("title", series.get("title", ""))
                merged.setdefault("series_id", series.get("id", ""))
                return merged
    for entity in iter_entities(s0):
        if str(entity.get("id")) == entity_id:
            return entity
    return None


def iter_entities(s0: dict[str, Any]):
    email = s0.get("email", {})
    for folder in ("inbox", "sent", "drafts", "archived"):
        for item in email.get(folder, []) or []:
            yield item
    for thread in email.get("threads", []) or []:
        for item in thread.get("emails", []) or []:
            yield item

    calendar = s0.get("calendar", {})
    for item in calendar.get("events", []) or []:
        yield item
    for series in calendar.get("series", []) or []:
        yield series
        for item in series.get("occurrences", []) or []:
            yield item

    ticket = s0.get("ticket", {})
    for item in ticket.get("tickets", []) or []:
        yield item
        for child in item.get("children", []) or []:
            yield child

    for item in s0.get("contacts", []) or []:
        if isinstance(item, dict):
            yield item
    for item in s0.get("distribution_lists", []) or []:
        if isinstance(item, dict):
            yield item


def _apply_email(state: dict[str, Any], op: WriteOperation) -> None:
    email = state.setdefault("email", {})
    for key in ("inbox", "sent", "drafts", "archived"):
        email.setdefault(key, [])
    if op.action == "reply":
        email["sent"].append({"id": f"sent_{op.step}", "reply_to": op.entity_id, "to": op.fields.get("to")})
    elif op.action == "forward":
        email["sent"].append({"id": f"sent_{op.step}", "forward_of": op.entity_id, "to": op.fields.get("to", [])})
    elif op.action == "send":
        email["sent"].append({"id": f"sent_{op.step}", "to": op.fields.get("to", [])})
    elif op.action == "archive":
        entity = _find_in_lists(email, op.entity_id, ["inbox", "sent", "drafts"])
        if entity is not None:
            entity["archived"] = True
            if not any(item.get("id") == op.entity_id for item in email["archived"]):
                email["archived"].append(entity)
    elif op.action == "delete":
        entity = _find_entity(state, op.entity_id)
        if entity is not None:
            entity["deleted"] = True
    elif op.action == "mark_read":
        entity = _find_entity(state, op.entity_id)
        if entity is not None:
            entity["read"] = True
    elif op.action == "label":
        entity = _find_entity(state, op.entity_id)
        if entity is not None:
            labels = entity.setdefault("labels", [])
            if op.fields.get("label") not in labels:
                labels.append(op.fields.get("label"))
    elif op.action == "draft_save":
        email["drafts"].append({"id": f"draft_{op.step}", "to": op.fields.get("to", "")})


def _apply_calendar(state: dict[str, Any], op: WriteOperation) -> None:
    calendar = state.setdefault("calendar", {})
    calendar.setdefault("events", [])
    if op.action == "create":
        calendar["events"].append({"id": op.entity_id, **op.fields})
    elif op.action in {"modify_occurrence", "modify_series"}:
        entity = _find_entity(state, op.entity_id)
        if entity is not None:
            entity.update(op.fields)
    elif op.action in {"cancel_occurrence", "cancel_series"}:
        entity = _find_entity(state, op.entity_id)
        if entity is not None:
            entity["status"] = "cancelled"
    elif op.action == "invite":
        entity = _find_entity(state, op.entity_id)
        if entity is not None:
            entity.setdefault("invited_attendees", []).extend(op.fields.get("attendees", []))


def _apply_ticket(state: dict[str, Any], op: WriteOperation) -> None:
    ticket = state.setdefault("ticket", {})
    ticket.setdefault("tickets", [])
    if op.action == "create":
        ticket["tickets"].append({"id": op.entity_id, **op.fields})
        return
    entity = _find_entity(state, op.entity_id)
    if entity is None:
        return
    if op.action == "update_status":
        entity["status"] = op.fields.get("status")
    elif op.action == "set_priority":
        entity["priority"] = op.fields.get("priority")
    elif op.action == "assign":
        entity["assignee"] = op.fields.get("user")
    elif op.action == "add_comment":
        entity.setdefault("comments", []).append({"id": f"comment_{op.step}"})


def _find_entity(state: dict[str, Any], entity_id: str) -> dict[str, Any] | None:
    for entity in iter_entities(state):
        if str(entity.get("id")) == entity_id:
            return entity
    return None


def _find_in_lists(root: dict[str, Any], entity_id: str, list_names: list[str]) -> dict[str, Any] | None:
    for name in list_names:
        for entity in root.get(name, []) or []:
            if str(entity.get("id")) == entity_id:
                return entity
    return None


def _resolve_path(state: dict[str, Any], path: str) -> Any:
    match = re.match(r"^(email|ticket|calendar)\[([^\]]+)\]\.(.+)$", path)
    if match:
        _, entity_id, attr = match.groups()
        entity = _find_entity(state, entity_id)
        return _MISSING if entity is None else entity.get(attr, _MISSING)

    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, _MISSING)
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _contains(value: Any, needle: str) -> bool:
    if value is _MISSING:
        return False
    if isinstance(value, list):
        return any(_contains(item, needle) for item in value)
    if isinstance(value, dict):
        return any(_contains(item, needle) for item in value.values())
    return needle.lower() in str(value).lower()


def _eval_wrote_condition(condition: str, state: dict[str, Any]) -> bool:
    # Goal sugar over the final synthetic state, e.g. "wrote email.reply e_001".
    _, api, entity_id = condition.split(" ", 2)
    tool, action = api.split(".", 1)
    if tool == "email" and action == "reply":
        return any(item.get("reply_to") == entity_id for item in state.get("email", {}).get("sent", []))
    if tool == "email" and action == "forward":
        return any(item.get("forward_of") == entity_id for item in state.get("email", {}).get("sent", []))
    return _find_entity(state, entity_id) is not None


def _parse_scalar(raw: str) -> Any:
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        return raw


class _Missing:
    pass


_MISSING = _Missing()
