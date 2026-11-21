import asyncio
import time
from pathlib import Path

from results_io import load_results, log_event, numeric, utc_timestamp, write_json
from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
    SemanticSimilarity,
)


JUDGE_MODELS = {
    "medgemma": "medgemma:27b",
}
OLLAMA_HOST = "http://localhost:11434"
JUDGE_MAX_TOKENS = 4096


def evaluate_ragas(results_file, embedding_model, use_context=True):
    """Add every question's Ragas scores to the same readable JSON file."""
    return asyncio.run(_evaluate_ragas(results_file, embedding_model, use_context))


async def _evaluate_ragas(results_file, embedding_model, use_context):
    async with AsyncOpenAI(base_url=OLLAMA_HOST + "/v1", api_key="ollama") as client:
        return await _evaluate_results(results_file, embedding_model, client, use_context)


def _create_metrics(judge, embeddings, use_context):
    metrics = (
        (AnswerCorrectness(llm=judge, embeddings=embeddings), ("user_input", "response", "reference")),
        (AnswerRelevancy(llm=judge, embeddings=embeddings), ("user_input", "response")),
        (ContextPrecision(llm=judge), ("user_input", "reference", "retrieved_contexts")),
        (ContextRecall(llm=judge), ("user_input", "reference", "retrieved_contexts")),
        (ContextEntityRecall(llm=judge), ("reference", "retrieved_contexts")),
        (FactualCorrectness(llm=judge, mode="f1"), ("response", "reference")),
        (FactualCorrectness(llm=judge, mode="precision", name="factual_precision"), ("response", "reference")),
        (FactualCorrectness(llm=judge, mode="recall", name="factual_recall"), ("response", "reference")),
        (Faithfulness(llm=judge), ("user_input", "response", "retrieved_contexts")),
        (NoiseSensitivity(llm=judge, mode="relevant"), ("user_input", "response", "reference", "retrieved_contexts")),
        (SemanticSimilarity(embeddings=embeddings), ("response", "reference")),
    )
    if use_context:
        return metrics
    # Without retrieval, context-based metrics are undefined.
    return tuple(m for m in metrics if "retrieved_contexts" not in m[1])


async def _evaluate_results(results_file, embedding_model, client, use_context):
    evaluation_started = time.perf_counter()
    evaluation_started_at = utc_timestamp()
    path = Path(results_file).with_suffix(".json")
    log_event(f"Ragas evaluation started: {path}")
    document = load_results(results_file)
    embeddings = embedding_factory("huggingface", model=embedding_model, device="cpu")
    metrics_by_judge = {}
    for judge_name, model_name in JUDGE_MODELS.items():
        judge = llm_factory(
            model_name, client=client, temperature=0, max_tokens=JUDGE_MAX_TOKENS
        )
        metrics_by_judge[judge_name] = _create_metrics(judge, embeddings, use_context)
    setup_elapsed = time.perf_counter() - evaluation_started

    document["ragas"] = {
        "status": "running",
        "judge_models": JUDGE_MODELS,
        "retrieval": use_context,
        "embedding_model": embedding_model,
        "started_at": evaluation_started_at,
        "completed_at": None,
        "setup_elapsed_seconds": round(setup_elapsed, 3),
        "elapsed_seconds": None,
        "judge_elapsed_seconds": {},
        "evaluated_questions": 0,
        "evaluated_questions_by_judge": {
            judge_name: 0 for judge_name in JUDGE_MODELS
        },
    }
    for row in document["results"]:
        row["scores"] = {judge_name: {} for judge_name in JUDGE_MODELS}
        row["metric_errors"] = {judge_name: {} for judge_name in JUDGE_MODELS}
        row["metric_durations_seconds"] = {
            judge_name: {} for judge_name in JUDGE_MODELS
        }
        row["evaluation_durations_seconds"] = {}
    write_json(path, document)
    log_event(
        f"Ragas setup completed in {setup_elapsed:.3f}s: "
        f"{len(document['results'])} questions"
    )
    for judge_name, metrics in metrics_by_judge.items():
        judge_started = time.perf_counter()
        log_event(f"Ragas {judge_name} ({JUDGE_MODELS[judge_name]}) started")
        for index, row in enumerate(document["results"], 1):
            question_started = time.perf_counter()
            sample = {
                "user_input": row["question"],
                "response": row["answer"],
                "reference": row["ground_truth"],
                "retrieved_contexts": row["context"],
            }
            for metric, required_fields in metrics:
                metric_started = time.perf_counter()
                try:
                    result = await metric.ascore(
                        **{field: sample[field] for field in required_fields}
                    )
                    score = numeric(result.value)
                    row["scores"][judge_name][metric.name] = score
                    if score is None:
                        row["metric_errors"][judge_name][metric.name] = (
                            "Metric returned a missing or non-finite value"
                        )
                except Exception as exc:
                    row["scores"][judge_name][metric.name] = None
                    row["metric_errors"][judge_name][metric.name] = str(exc)
                finally:
                    row["metric_durations_seconds"][judge_name][metric.name] = round(
                        time.perf_counter() - metric_started, 3
                    )
            question_elapsed = time.perf_counter() - question_started
            row["evaluation_durations_seconds"][judge_name] = round(
                question_elapsed, 3
            )
            document["ragas"]["evaluated_questions_by_judge"][judge_name] = index
            document["ragas"]["evaluated_questions"] = min(
                document["ragas"]["evaluated_questions_by_judge"].values()
            )
            write_json(path, document)
            log_event(
                f"Ragas {judge_name}: {index}/{len(document['results'])} "
                f"completed in {question_elapsed:.3f}s"
            )
        judge_elapsed = time.perf_counter() - judge_started
        document["ragas"]["judge_elapsed_seconds"][judge_name] = round(
            judge_elapsed, 3
        )
        write_json(path, document)
        log_event(f"Ragas {judge_name} completed in {judge_elapsed:.3f}s")
    document["ragas"]["status"] = "completed"
    document["ragas"]["completed_at"] = utc_timestamp()
    document["ragas"]["elapsed_seconds"] = round(
        time.perf_counter() - evaluation_started, 3
    )
    write_json(path, document)
    log_event(
        f"Ragas evaluation completed in "
        f"{document['ragas']['elapsed_seconds']:.3f}s: {path}"
    )
    return document
