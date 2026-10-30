"""Single sanitized command-line boundary for starting the local API."""

from __future__ import annotations

import sys

import uvicorn

from services.api.app.main import create_app
from services.api.app.security import RUNTIME_TOKEN_CONFIGURATION_ERROR
from services.inference.errors import BackendConfigurationError

PUBLIC_STARTUP_CONFIGURATION_ERROR = "API startup configuration is invalid"


def _public_startup_message(error: Exception) -> str:
    if str(error) == RUNTIME_TOKEN_CONFIGURATION_ERROR:
        return RUNTIME_TOKEN_CONFIGURATION_ERROR
    return PUBLIC_STARTUP_CONFIGURATION_ERROR


def main() -> int:
    """Build the app once, then hand that object to Uvicorn without a factory."""
    try:
        app = create_app()
    except (ValueError, RuntimeError, BackendConfigurationError) as error:
        print(_public_startup_message(error), file=sys.stderr)
        return 1

    # Bind where the configuration says. API_BIND_HOST and API_PORT already exist and are already
    # validated (the production guard refuses a non-loopback bind host), but this line used to
    # hard-code 127.0.0.1:8000 and silently ignore both — so a workstation running several projects
    # could not give each one its own port, and a configured port looked applied when it was not.
    # The defaults are unchanged, so a deployment that sets neither still binds 127.0.0.1:8000.
    settings = app.state.settings
    uvicorn.run(app, host=settings.api_bind_host, port=settings.api_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
