from __future__ import annotations
import json
from typing import Any, Mapping

class StructuredOutputError(ValueError):
    pass

def parse_structured_output(value: Any, schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        text = str(value).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"): text = text[4:].lstrip()
        try: result = json.loads(text)
        except json.JSONDecodeError as exc: raise StructuredOutputError("provider output is not valid JSON") from exc
    if not isinstance(result, dict): raise StructuredOutputError("structured output must be a JSON object")
    if schema:
        required = schema.get("required", [])
        missing = [key for key in required if key not in result]
        if missing: raise StructuredOutputError(f"missing required keys: {', '.join(missing)}")
    return result

