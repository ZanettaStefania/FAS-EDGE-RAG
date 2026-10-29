import asyncio
from pathlib import Path

from results_io import load_results, numeric, write_json
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


JUDGE_MODEL = "medgemma:27b"
OLLAMA_HOST = "http://localhost:11434"
JUDGE_MAX_TOKENS = 4096


def evaluate_ragas(results_file, embedding_model):
    """Add every question's Ragas scores to the same readable JSON file."""
    return asyncio.run(_evaluate_ragas(results_file, embedding_model))


async def _evaluate_ragas(results_file, embedding_model):
    async with AsyncOpenAI(base_url=OLLAMA_HOST + "/v1", api_key="ollama") as client:
        return await _evaluate_results(results_file, embedding_model, client)


async def _evaluate_results(results_file, embedding_model, client):
    path = Path(results_file).with_suffix(".json")
    document = load_results(results_file)
    judge = llm_factory(JUDGE_MODEL, client=client, temperature=0, max_tokens=JUDGE_MAX_TOKENS)
    embeddings = embedding_factory("huggingface", model=embedding_model, device="cpu")

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

    document["ragas"] = {"status": "running", "judge_model": JUDGE_MODEL,
                         "embedding_model": embedding_model, "evaluated_questions": 0}
    write_json(path, document)
    for index, row in enumerate(document["results"], 1):
        sample = {
            "user_input": row["question"],
            "response": row["answer"],
            "reference": row["ground_truth"],
            "retrieved_contexts": row["context"],
        }
        row["scores"] = {}
        row["metric_errors"] = {}
        for metric, required_fields in metrics:
            try:
                result = await metric.ascore(
                    **{field: sample[field] for field in required_fields}
                )
                row["scores"][metric.name] = numeric(result.value)
                if row["scores"][metric.name] is None:
                    row["metric_errors"][metric.name] = "Metric returned a missing or non-finite value"
            except Exception as exc:
                row["scores"][metric.name] = None
                row["metric_errors"][metric.name] = str(exc)
        document["ragas"]["evaluated_questions"] = index
        write_json(path, document)
        print(f"Ragas: {index}/{len(document['results'])}", flush=True)
    document["ragas"]["status"] = "completed"
    write_json(path, document)
    print(f"Ragas results: {path}", flush=True)
    return document
