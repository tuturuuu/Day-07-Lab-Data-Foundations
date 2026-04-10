from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.store.search(question, top_k=top_k)
        context_chunks = [item.get("content", "") for item in retrieved if item.get("content")]
        context = "\n".join(f"- {chunk}" for chunk in context_chunks)

        prompt = (
            "You are a helpful assistant that answers questions using only the provided context.\n"
            "If the context is insufficient, say you do not have enough information.\n\n"
            f"Context:\n{context if context else '- (no relevant context found)'}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
