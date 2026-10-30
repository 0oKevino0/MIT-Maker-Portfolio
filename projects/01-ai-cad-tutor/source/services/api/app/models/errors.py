"""Unified API error object (ADR 0001 section 4.1).

Errors never include stack traces, temporary paths, provider responses,
course passages, or student data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_path: str
    reason: str


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    code: str
    message: str
    retryable: bool
    details: list[ErrorDetail] = Field(default_factory=list)
