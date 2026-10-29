"""Log vector-search scores without repeating the search."""

from langchain_core.runnables import RunnableLambda


def logged_retriever(vectorstore, compressor, k, threshold):
    def retrieve(inputs):
        question = inputs["input"]
        scored_documents = vectorstore.similarity_search_with_relevance_scores(
            question, k=k
        )
        print(
            f"Retrieval: {len(scored_documents)} candidates "
            f"(k={k}, threshold={threshold})",
            flush=True,
        )

        selected = []
        for rank, (document, score) in enumerate(scored_documents, start=1):
            keep = score >= threshold
            if keep:
                selected.append(document)
            source = document.metadata.get("source", "unknown")
            excerpt = " ".join(document.page_content.split())[:100]
            print(
                f"  [{rank}] score={score:.3f} "
                f"{'kept' if keep else 'discarded'} | {source} | {excerpt}",
                flush=True,
            )
        print(f"  Passed threshold: {len(selected)}/{len(scored_documents)}", flush=True)
        documents = compressor.compress_documents(selected, question) if selected else []
        print(f"  After reranking: {len(documents)} chunks", flush=True)
        return documents

    return RunnableLambda(retrieve)
