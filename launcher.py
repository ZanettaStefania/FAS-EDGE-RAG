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
from results_io import ensure_timings_file, get_logger, load_results, write_json


ROOT = Path(__file__).resolve().parent
logger = get_logger(__name__)
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
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Generate answers without running the Ragas evaluation.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-rerank", action="store_true",
                      help="Disable cross-encoder reranking in every selected pipeline.")
    mode.add_argument("--no-retrieval", action="store_true",
                      help="Skip retrieval (generation only, no RAG). Supported only by main_med.")
    args = parser.parse_args(argv)
    if args.no_retrieval and set(args.pipelines) - {"main_med"}:
        parser.error("--no-retrieval is supported only with --pipelines main_med")
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


def validate_output(path, questions, skip_ragas=False):
    document = load_results(path)
    rows = document["results"]
    if [row["question"] for row in rows] != questions:
        raise ValueError("Output questions or row order do not match the input dataset.")
    expected_status = "skipped" if skip_ragas else "completed"
    if document.get("ragas", {}).get("status") != expected_status:
        raise ValueError(f"Expected Ragas status {expected_status!r}.")
    if not skip_ragas and any(
        not isinstance(row.get("scores"), dict) or not row["scores"]
        or not isinstance(row.get("metric_errors"), dict) for row in rows
    ):
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
    logger.info(
        "%s models x %s pipelines x %s datasets = %s experiments",
        len(args.models), len(args.pipelines), len(args.datasets), len(jobs),
    )
    if args.dry_run:
        for model, pipeline, dataset in jobs:
            logger.info(
                "%s | %s.py | %s | %s questions",
                model, pipeline, dataset, len(questions[dataset]),
            )
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "CSV_Result_med" / "experiments" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Results: %s", run_dir)
    failures = 0
    run_file = run_dir / "run.json"
    timings_file = ensure_timings_file(run_dir / "timings.csv")
    manifest = {"started_at": stamp, "reranking": not (args.no_rerank or args.no_retrieval),
                "retrieval": not args.no_retrieval, "ragas": not args.skip_ragas,
                "timings_csv": timings_file.name,
                "experiments": []}
    write_json(run_file, manifest)
    for index, (model, pipeline, dataset) in enumerate(jobs, 1):
        output_dir = run_dir / model / pipeline
        output_dir.mkdir(parents=True, exist_ok=True)
        database, suffix = PIPELINES[pipeline]
        rerank_suffix = "_no_rerank" if args.no_rerank else "_no_retrieval" if args.no_retrieval else ""
        result_file = output_dir / f"{Path(database).name}{model}{suffix}{rerank_suffix}_{Path(dataset).stem}.json"
        log_file = output_dir / f"{Path(dataset).stem}{rerank_suffix}.log"
        command = [sys.executable, "-u", str(ROOT / f"{pipeline}.py"),
                   "--model_name", model, "--test", str(ROOT / "EvalSet" / dataset)]
        if args.no_rerank:
            command.append("--no-rerank")
        if args.no_retrieval:
            command.append("--no-retrieval")
        if args.skip_ragas:
            command.append("--skip-ragas")
        env = dict(os.environ, THESIS_RAG_OUTPUT_DIR=str(output_dir),
                   THESIS_RAG_TIMINGS_FILE=str(timings_file), PYTHONIOENCODING="utf-8")
        row = dict(model=model, pipeline=pipeline, dataset=dataset,
                   reranking=not (args.no_rerank or args.no_retrieval),
                   retrieval=not args.no_retrieval, ragas=not args.skip_ragas, status="ok",
                   return_code=None, elapsed_seconds=None, results_json="",
                   log_file=str(log_file.relative_to(run_dir)), error="")
        logger.info("[%s/%s] %s | %s | %s", index, len(jobs), model, pipeline, dataset)
        start = time.perf_counter()
        try:
            with log_file.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            row["return_code"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"Pipeline exited with code {completed.returncode}; see log.")
            validate_output(result_file, questions[dataset], skip_ragas=args.skip_ragas)
            row["results_json"] = str(result_file.relative_to(run_dir))
        except KeyboardInterrupt:
            row.update(status="interrupted", error="Interrupted by user.")
        except (OSError, ValueError, RuntimeError) as exc:
            failures += 1
            row.update(status="error", error=str(exc))
        row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        manifest["experiments"].append(row)
        write_json(run_file, manifest)
        logger.info("%s: %s", row["status"], row["error"] or row["results_json"])
        if row["status"] == "interrupted":
            return 130
        if failures and args.stop_on_error:
            break
    logger.info("Finished. Failed experiments: %s. Run info: %s. Timings: %s",
                failures, run_file, timings_file)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)
