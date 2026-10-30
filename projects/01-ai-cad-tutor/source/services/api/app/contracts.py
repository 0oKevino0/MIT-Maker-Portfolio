"""Offline runtime validation against the shared JSON contracts.

Loads the six schemas from ``packages/contracts`` into a local referencing
registry once (lazily, never at import time) and validates assembled public
payloads before they leave the server. The absolute ``$id`` URIs are resolved
only through this local registry; no network access is ever attempted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel
from referencing import Registry, Resource

_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "packages" / "contracts"

CONTRACT_NAMES = (
    "course_question",
    "course_answer",
    "drawing_analysis",
    "cad_model",
    "grading_request",
    "grading_result",
)

_FINITE_NUMBER_ERROR = "contract payload numbers must be finite"


class ContractViolation(Exception):
    """An assembled payload failed public-contract validation."""


def require_finite_numbers(value: object) -> object:
    """Reject non-finite floats anywhere in one contract-shaped value.

    JSON Schema treats Python's non-standard NaN and infinity float values as
    numbers, so this guard runs before schema validation. It traverses only
    value containers, never mutates the caller's object, and returns that same
    object so Pydantic ``before`` validators can share the boundary later.
    """

    active_containers: set[int] = set()

    def visit(current: object) -> None:
        if isinstance(current, bool):
            return
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError(_FINITE_NUMBER_ERROR)
            return

        nested: object = (
            current.model_dump(mode="python") if isinstance(current, BaseModel) else current
        )

        if isinstance(nested, Mapping):
            identity = id(nested)
            if identity in active_containers:
                raise ValueError(_FINITE_NUMBER_ERROR)
            active_containers.add(identity)
            try:
                for item in nested.values():
                    visit(item)
            finally:
                active_containers.remove(identity)
            return

        if isinstance(nested, (list, tuple)):
            identity = id(nested)
            if identity in active_containers:
                raise ValueError(_FINITE_NUMBER_ERROR)
            active_containers.add(identity)
            try:
                for item in nested:
                    visit(item)
            finally:
                active_containers.remove(identity)

    visit(value)
    return value


def _load_schema(name: str) -> dict[str, Any]:
    path = _CONTRACTS_DIR / f"{name}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources = [
        (schema["$id"], Resource.from_contents(schema))
        for schema in (_load_schema(name) for name in CONTRACT_NAMES)
    ]
    return Registry().with_resources(resources)


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = _load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())


def validate_public_contract(name: str, payload: dict[str, Any]) -> None:
    """Validate an assembled payload; raises ContractViolation on failure.

    The violation message contains the schema location and rule only — never
    payload content.
    """
    if name not in CONTRACT_NAMES:
        raise KeyError(f"unknown contract '{name}'")
    try:
        require_finite_numbers(payload)
    except ValueError:
        raise ContractViolation(
            f"contract '{name}' violated at '<root>' (rule: finite_number)"
        ) from None
    try:
        _validator(name).validate(payload)
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ContractViolation(
            f"contract '{name}' violated at '{location}' (rule: {exc.validator})"
        ) from None
