"""Read and write the combined JSON produced by new experiments."""

import json
import math
import os
from pathlib import Path
import tempfile


BASE_FIELDS = {"question", "answer", "ground_truth", "time", "context", "source"}


def numeric(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def normalized_result(row):
    contexts = row.get("context") or []
    sources = row.get("source") or []
    if not isinstance(contexts, (list, tuple)) or not isinstance(sources, (list, tuple)):
        raise ValueError("Context and source must be lists")
    contexts = [item[0] if isinstance(item, (list, tuple)) and item else item for item in contexts]
    if any(not isinstance(item, str) for item in contexts):
        raise ValueError("Each context must contain text")
    return {
        "question": row["question"],
        "answer": row["answer"],
        "ground_truth": row["ground_truth"],
        "time": numeric(row.get("time")),
        "context": contexts,
        "source": list(sources),
        "scores": {},
        "metric_errors": {},
    }


def write_json(path, document):
    """Replace atomically, preserving Unicode and full numeric precision."""
    path = Path(path)
    text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def write_generation_results(frame, path, **metadata):
    path = Path(path).with_suffix(".json")
    document = dict(metadata, ragas={"status": "pending", "evaluated_questions": 0},
                    results=[normalized_result(row) for row in frame.to_dict("records")])
    write_json(path, document)
    return str(path)


def reject_constant(value):
    raise ValueError(f"Non-standard JSON number: {value}")


def load_results(path):
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"{path}: expected an experiment JSON")
    with path.open(encoding="utf-8-sig") as stream:
        document = json.load(stream, parse_constant=reject_constant)
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError(f"{path}: expected an object with a results list")
    for row in document["results"]:
        if not isinstance(row, dict) or not BASE_FIELDS.issubset(row):
            raise ValueError(f"{path}: incomplete question result")
    return document
