"""Run the thesis pipelines sequentially and save one combined JSON per experiment."""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import config_med
from results_io import load_results, write_json


ROOT = Path(__file__).resolve().parent
MODELS = (
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Phi-3.5-mini-instruct",
    "Phi-3-mini-128k-instruct",
)
PIPELINES = {
    "main": (config.PERSIST_DIRECTORY_DB, ""),
    "main_med": (config_med.PERSIST_DIRECTORY_DB, "_med"),
    "new_config_main_med": (config_med.PERSIST_DIRECTORY_DB, "_med"),
}


def parse_args(argv=None):
    datasets = sorted(path.name for path in (ROOT / "EvalSet").glob("*.csv"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Short Qwen/Phi names; defaults to the five original models.")
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES,
                        default=tuple(PIPELINES), help="Defaults to all three pipelines.")
    parser.add_argument("--datasets", nargs="+", choices=datasets, default=datasets,
                        help="CSV filenames in EvalSet; defaults to every CSV.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the experiment plan without running models or producing results.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop at the first failed experiment instead of continuing.")
    args = parser.parse_args(argv)
    for model in args.models:
        if not re.fullmatch(r"(?:Qwen|Phi)[A-Za-z0-9_.-]+", model):
            parser.error(f"Use a short Qwen/Phi model name, not a path: {model}")
    if not args.datasets:
        parser.error("No question CSVs found in EvalSet.")
    for attribute in ("models", "pipelines", "datasets"):
        setattr(args, attribute, tuple(dict.fromkeys(getattr(args, attribute))))
    return args


def read_questions(dataset):
    with (ROOT / "EvalSet" / dataset).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not {"Question", "ground_truths"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{dataset} needs Question and ground_truths columns.")
        try:
            rows = list(reader)
        except csv.Error as exc:
            raise ValueError(
                f"{dataset}: malformed CSV near line {reader.line_num}: {exc}"
            ) from exc
    if not rows:
        raise ValueError(f"{dataset} needs at least one row.")
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{dataset}: line {line_number} has extra fields.")
        for field in ("Question", "ground_truths"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{dataset}: line {line_number} has an empty {field}.")
    return [row["Question"] for row in rows]


def validate_output(path, questions):
    document = load_results(path)
    rows = document["results"]
    if [row["question"] for row in rows] != questions:
        raise ValueError("Output questions or row order do not match the input dataset.")
    if document.get("ragas", {}).get("status") != "completed":
        raise ValueError("Ragas evaluation did not finish.")
    if any(not isinstance(row.get("scores"), dict) or not row["scores"]
           or not isinstance(row.get("metric_errors"), dict) for row in rows):
        raise ValueError("Output is missing Ragas scores or evaluation errors.")


def main(argv=None):
    args = parse_args(argv)
    questions = {dataset: read_questions(dataset) for dataset in args.datasets}
    for pipeline in args.pipelines:
        if not (ROOT / f"{pipeline}.py").is_file():
            raise FileNotFoundError(f"Pipeline script missing: {pipeline}.py")
        database, _ = PIPELINES[pipeline]
        if not (ROOT / database).is_dir():
            raise FileNotFoundError(f"Database directory missing: {database}")
    jobs = list(itertools.product(args.models, args.pipelines, args.datasets))
    print(f"{len(args.models)} models x {len(args.pipelines)} pipelines x "
          f"{len(args.datasets)} datasets = {len(jobs)} experiments", flush=True)
    if args.dry_run:
        for model, pipeline, dataset in jobs:
            print(f"{model} | {pipeline}.py | {dataset} | {len(questions[dataset])} questions")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "CSV_Result_med" / "experiments" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results: {run_dir}", flush=True)
    failures = 0
    run_file = run_dir / "run.json"
    manifest = {"started_at": stamp, "experiments": []}
    write_json(run_file, manifest)
    for index, (model, pipeline, dataset) in enumerate(jobs, 1):
        output_dir = run_dir / model / pipeline
        output_dir.mkdir(parents=True, exist_ok=True)
        database, suffix = PIPELINES[pipeline]
        result_file = output_dir / f"{Path(database).name}{model}{suffix}_{Path(dataset).stem}.json"
        log_file = output_dir / f"{Path(dataset).stem}.log"
        command = [sys.executable, "-u", str(ROOT / f"{pipeline}.py"),
                   "--model_name", model, "--test", str(ROOT / "EvalSet" / dataset)]
        env = dict(os.environ, THESIS_RAG_OUTPUT_DIR=str(output_dir), PYTHONIOENCODING="utf-8")
        row = dict(model=model, pipeline=pipeline, dataset=dataset, status="ok",
                   return_code=None, elapsed_seconds=None, results_json="",
                   log_file=str(log_file.relative_to(run_dir)), error="")
        print(f"[{index}/{len(jobs)}] {model} | {pipeline} | {dataset}", flush=True)
        start = time.perf_counter()
        try:
            with log_file.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            row["return_code"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"Pipeline exited with code {completed.returncode}; see log.")
            validate_output(result_file, questions[dataset])
            row["results_json"] = str(result_file.relative_to(run_dir))
        except KeyboardInterrupt:
            row.update(status="interrupted", error="Interrupted by user.")
        except (OSError, ValueError, RuntimeError) as exc:
            failures += 1
            row.update(status="error", error=str(exc))
        row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        manifest["experiments"].append(row)
        write_json(run_file, manifest)
        print(f"  {row['status']}: {row['error'] or row['results_json']}", flush=True)
        if row["status"] == "interrupted":
            return 130
        if failures and args.stop_on_error:
            break
    print(f"Finished. Failed experiments: {failures}. Run info: {run_file}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
