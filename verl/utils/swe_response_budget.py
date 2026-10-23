"""Optional per-partition SWE response budgets within the shared rollout capacity.

This module only needs the standard library so the launcher can also execute it
directly to validate budgets before initializing training or inference engines.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

RESPONSE_LENGTH_ENV_NAMES = (
    "SWE_AGENT_ROLLOUT_TRAINING_RESPONSE_LENGTH",
    "SWE_AGENT_ROLLOUT_VALIDATION_RESPONSE_LENGTH",
)


def resolve_swe_response_lengths(capacity: int, environ: Mapping[str, str] | None = None) -> tuple[int, int]:
    """Return (training, validation) budgets; unset/empty settings inherit capacity.

    Budgets count retained assistant and observation tokens, excluding the
    initial prompt. They must fit the existing tensor/engine response capacity.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("SWE response capacity must be a positive integer")
    environ = os.environ if environ is None else environ
    lengths = []
    for name in RESPONSE_LENGTH_ENV_NAMES:
        raw = environ.get(name, "").strip()
        if not raw:
            lengths.append(capacity)
            continue
        if re.fullmatch(r"[0-9]+", raw) is None or int(raw) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}")
        length = int(raw)
        if length > capacity:
            raise ValueError(f"{name}={length} exceeds shared rollout response capacity {capacity}")
        lengths.append(length)
    return lengths[0], lengths[1]


if __name__ == "__main__":
    import sys

    try:
        capacity = int(sys.argv[1])
        training, validation = resolve_swe_response_lengths(capacity)
    except (IndexError, ValueError) as exc:
        sys.exit(str(exc))
    print(f"Effective SWE response lengths: training={training} validation={validation} capacity={capacity}")
