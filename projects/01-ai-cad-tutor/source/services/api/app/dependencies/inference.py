"""Dependency injection for model backends.

Routes depend on these providers; they never construct or select a provider
themselves. The concrete backends are created once at application startup
from settings (see ``main.create_app``).
"""

from __future__ import annotations

from fastapi import Request

from services.inference.backends.base import ModelBackend


def get_text_backend(request: Request) -> ModelBackend:
    return request.app.state.text_backend


def get_vision_backend(request: Request) -> ModelBackend:
    return request.app.state.vision_backend
