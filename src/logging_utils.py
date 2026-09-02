"""Small helpers for readable, timestamped console output."""
from __future__ import annotations

import builtins
from datetime import datetime
from typing import Any


def enable_timestamped_prints() -> None:
    """Prefix every subsequent ``print`` line with the local timestamp.

    Calling this more than once is safe.  ``file``, ``sep``, ``end``, and
    ``flush`` retain their normal :func:`print` semantics.
    """
    if getattr(builtins.print, "_timestamps_enabled", False):
        return

    original_print = builtins.print

    def timestamped_print(*values: Any, **kwargs: Any) -> None:
        separator = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        message = separator.join(str(value) for value in values)
        prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        # Prefix embedded new lines too, so compact multi-line status blocks
        # remain easy to correlate with their execution time.
        original_print(prefix + message.replace("\n", "\n" + prefix), end=end, **kwargs)

    timestamped_print._timestamps_enabled = True  # type: ignore[attr-defined]
    builtins.print = timestamped_print
