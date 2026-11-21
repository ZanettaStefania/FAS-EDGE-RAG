"""Read and write the combined JSON produced by new experiments."""

import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import tempfile


BASE_FIELDS = {"question", "answer", "ground_truth", "time", "context", "source"}
TIMING_FIELDS = ["model", "pipeline", "dataset", "question", "total_s", "ttft_s",
                 "retrieval_s", "reranking_s", "generation_s"]


class FirstTokenTimer:
    """transformers streamer (duck-typed) recording when the first new token is produced."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.first_token_at = None
        self._prompt_seen = False

    def put(self, value):
        # generate() first passes the prompt ids, then one call per new token.
        if not self._prompt_seen:
            self._prompt_seen = True
        elif self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def end(self):
        self._prompt_seen = False


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    _UtcFormatter(
        "[%(asctime)s.%(msecs)03dZ] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)
_project_logger = logging.getLogger("faser")
if not _project_logger.handlers:
    _project_logger.addHandler(_handler)
_project_logger.setLevel(logging.INFO)
_project_logger.propagate = False


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def get_logger(name):
    return _project_logger.getChild(name)


def log_event(message):
    _project_logger.info(message)


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


def timings_path():
    path = os.environ.get("THESIS_RAG_TIMINGS_FILE")
    if path:
        return Path(path)
    return Path(os.environ.get("THESIS_RAG_OUTPUT_DIR", "CSV_Result_med")) / "timings.csv"


def ensure_timings_file(path=None):
    """Create the timings CSV, or realign an existing one to the current columns."""
    path = Path(path) if path else timings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == TIMING_FIELDS:
                return path
            rows = list(reader)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_timing(row):
    """Append one question's timings; written per question so partial runs are kept."""
    path = ensure_timings_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=TIMING_FIELDS).writerow(row)


def write_generation_results(frame, path, **metadata):
    path = Path(path).with_suffix(".json")
    document = dict(metadata, ragas={"status": "pending", "evaluated_questions": 0},
                    results=[normalized_result(row) for row in frame.to_dict("records")])
    write_json(path, document)
    return str(path)


def mark_ragas_skipped(path):
    """Record an intentional evaluation skip in an existing result document."""
    document = load_results(path)
    document["ragas"] = {
        "status": "skipped",
        "evaluated_questions": 0,
        "skipped_at": utc_timestamp(),
    }
    write_json(path, document)


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
