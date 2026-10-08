"""Strict host-owned JSON and digest helpers for the permission boundary."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError(f"Invalid JSON number: {value}")
    return json.loads(text, object_pairs_hook=_unique, parse_constant=invalid)


def read(path):
    return loads(Path(path).read_text())


def write(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def requests(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = loads(line)
        if (not isinstance(item, dict) or set(item) - {"task", "context"}
                or not isinstance(item.get("task"), str) or not item["task"].strip()
                or not isinstance(item.get("context", ""), str)):
            raise ValueError("Each request needs task and optional text context")
        result.append(item)
    if not result:
        raise ValueError("Request file is empty")
    return result
