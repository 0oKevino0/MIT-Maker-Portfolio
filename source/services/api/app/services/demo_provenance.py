"""Internal provenance for the INTERNAL DEMO profile: one JSONL line per request, off by default.

The student-facing answer says nothing about matchers, corpora, retrieval queries or whether a model
ran. Those facts still have to be recoverable, so they are written here instead: when
``DEMO_PROVENANCE_LOG`` names a path, each demo request appends one line to it. The path is outside
the repository, like every other demo log.

What is written: the interaction type when a message was handled as one, whether retrieval ran,
whether the model was called, the retrieval query actually used (including when it was extended from
the previous user question), which corpora and record ids supplied evidence, and the coverage state.
What is never written: a credential, a token, or any course text — only record identifiers and the
user's own message.

A logging failure never fails a request: a demo that cannot write its log still answers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def record_demo_provenance(log_path: str | None, entry: dict[str, Any]) -> None:
    """Append one provenance line. Silent no-op when no path is configured."""
    if not log_path:
        return
    line = json.dumps(
        {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry},
        ensure_ascii=False,
        sort_keys=True,
    )
    try:
        path = Path(log_path)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        # Provenance is evidence, not a dependency: a demo that cannot write its log still answers.
        return
