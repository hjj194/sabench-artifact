"""Tool API registry and trace normalization."""

from __future__ import annotations

import re
from typing import Any

from .models import APICall, WriteOperation

WRITE_APIS: set[tuple[str, str]] = {
    ("email", "reply"),
    ("email", "forward"),
    ("email", "send"),
    ("email", "archive"),
    ("email", "delete"),
    ("email", "mark_read"),
    ("email", "label"),
    ("email", "draft_save"),
    ("calendar", "create"),
    ("calendar", "modify_occurrence"),
    ("calendar", "modify_series"),
    ("calendar", "cancel_occurrence"),
    ("calendar", "cancel_series"),
    ("calendar", "invite"),
    ("ticket", "create"),
    ("ticket", "update_status"),
    ("ticket", "set_priority"),
    ("ticket", "assign"),
    ("ticket", "add_comment"),
}

READ_APIS: set[tuple[str, str]] = {
    ("email", "fetch"),
    ("email", "list"),
    ("email", "search"),
    ("email", "get_thread"),
    ("calendar", "fetch"),
    ("calendar", "list"),
    ("calendar", "search"),
    ("calendar", "check_conflicts"),
    ("ticket", "fetch"),
    ("ticket", "list"),
    ("ticket", "search"),
}

EXTERNAL_WRITE_APIS: set[tuple[str, str]] = {
    ("email", "reply"),
    ("email", "forward"),
    ("email", "send"),
    ("calendar", "invite"),
}

OPEN_TEXT_FIELDS = {"body", "comment_text", "comment", "text", "description", "subject"}


def is_write_api(tool: str, action: str) -> bool:
    return (tool, action) in WRITE_APIS


def is_external_write(tool: str, action: str) -> bool:
    return (tool, action) in EXTERNAL_WRITE_APIS


def normalize_create_entity_id(tool: str, action: str, fields: dict[str, Any]) -> str:
    if tool == "calendar" and action == "create":
        return f"calendar_create:{_normalize_calendar_create_title(str(fields.get('title', '')))}:{fields.get('time', '')}"
    if tool == "ticket" and action == "create":
        return f"ticket_create:{fields.get('priority', '')}:{fields.get('title_key', '')}"
    if tool == "email" and action in {"send", "draft_save"}:
        to = ",".join(fields.get("to", [])) if isinstance(fields.get("to"), list) else str(fields.get("to", ""))
        return f"email_{action}:{to}"
    return ""


def extract_structural_fields(tool: str, action: str, args: dict[str, Any]) -> dict[str, Any]:
    """Drop open text fields and keep only fields relevant to authorization."""
    if tool == "email":
        merged = _merge_nested_fields(args)
        if action == "reply":
            return {"to": "original_sender"}
        if action == "forward":
            return {"to": _as_list(merged.get("to", []))}
        if action == "send":
            return {"to": _as_list(merged.get("to", []))}
        if action == "label":
            return {"label": merged.get("label", "")}
        if action == "draft_save":
            return {"to": merged.get("to", "")}
        return {}

    if tool == "calendar":
        if action == "create":
            duration = args["duration"] if "duration" in args else args.get("duration_minutes", 30)
            return {
                "title": _normalize_calendar_create_title(str(args.get("title", ""))),
                "time": _normalize_calendar_create_time(str(args.get("time") or args.get("start") or args.get("start_time") or "")),
                "duration": _normalize_int(duration),
                "attendees": _as_list(args.get("attendees", [])),
            }
        if action in {"modify_occurrence", "modify_series"}:
            fields = dict(args.get("fields") or {})
            if not fields:
                fields = {
                    key: args[key]
                    for key in ("time", "duration", "attendees", "location", "date")
                    if key in args
                }
            if "time" in fields:
                fields["time"] = _normalize_clock_time(str(fields["time"]))
            if "attendees" in fields:
                fields["attendees"] = _as_list(fields["attendees"])
            return fields
        if action == "invite":
            return {"attendees": _as_list(args.get("attendees", []))}
        return {}

    if tool == "ticket":
        if action == "create":
            title_key = args.get("title_key", "")
            if not title_key and args.get("title"):
                title_key = str(args.get("title", "")).strip().lower()
            return {"priority": args.get("priority", ""), "title_key": title_key}
        if action == "update_status":
            return {"status": args.get("status", "")}
        if action == "set_priority":
            return {"priority": args.get("priority", "")}
        if action == "assign":
            return {"user": args.get("user", "")}
        return {}

    return {key: value for key, value in args.items() if key not in OPEN_TEXT_FIELDS}


def write_operation_from_call(call: APICall) -> WriteOperation | None:
    if not call.success or not is_write_api(call.tool, call.action):
        return None
    fields = extract_structural_fields(call.tool, call.action, call.args)
    entity_id = (
        call.args.get("id")
        or call.args.get("entity_id")
        or call.args.get("email_id")
        or (call.args.get("thread_id") if call.tool == "email" and call.action == "reply" else None)
        or call.args.get("event_id")
        or call.args.get("occurrence_id")
        or call.args.get("series_id")
        or call.args.get("ticket_id")
        or normalize_create_entity_id(call.tool, call.action, fields)
    )
    return WriteOperation(
        tool=call.tool,
        action=call.action,
        entity_id=str(entity_id),
        fields=fields,
        is_external=is_external_write(call.tool, call.action),
        step=call.step,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_nested_fields(args: dict[str, Any]) -> dict[str, Any]:
    fields = args.get("fields")
    merged = dict(fields) if isinstance(fields, dict) else {}
    merged.update(args)
    return merged


def _normalize_clock_time(value: str) -> str:
    value = value.strip()
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])\b", value)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or "00")
        meridiem = match.group(3).lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    matches = re.findall(r"\b(\d{1,2}):(\d{2})\b", value)
    if matches:
        hour, minute = matches[-1]
        return f"{int(hour):02d}:{int(minute):02d}"
    return value


def _normalize_calendar_create_time(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
        return value
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})(?::\d{2})?", value)
    if match:
        date, clock = match.groups()
        return f"{date}T{_normalize_clock_time(clock)}:00"
    match = re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:,\s*(\d{4}))?,?\s+(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:[AaPp][Mm])?)\b",
        value,
    )
    if match:
        month_name, day, year, clock = match.groups()
        month = _MONTHS[month_name[:3].lower()]
        return f"{int(year or 2026):04d}-{month:02d}-{int(day):02d}T{_normalize_clock_time(clock)}:00"
    return value


def _normalize_calendar_create_title(value: str) -> str:
    title = " ".join(value.strip().split()).lower()
    title = re.sub(r"\s+-\s+[a-z][a-z .'-]+$", "", title)
    title = re.sub(r"\s+with\s+[a-z][a-z .'-]+$", "", title)
    return title


def _normalize_int(value: Any) -> Any:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
