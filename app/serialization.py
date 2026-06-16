"""Convert `ml/` dataclass results into JSON-safe structures for `jobs.result_summary`.

`dataclasses.asdict()` already recurses into nested dataclasses/lists/dicts,
but leaves `date`/`datetime` objects and non-finite floats (`inf`/`nan`, which
can appear in basket `conviction` or empty-series stats) as-is. Both break
JSON/jsonb storage, so `to_jsonable` normalizes them.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value
