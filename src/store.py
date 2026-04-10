from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = False
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": dict(doc.metadata or {}),
            "embedding": embedding,
            "store_id": f"{doc.id}::{self._next_index}",
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []

            for doc in docs:
                record = self._make_record(doc)
                self._next_index += 1
                ids.append(record["store_id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadata = dict(record["metadata"])
                metadata["doc_id"] = record["id"]
                metadatas.append(metadata)

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return

        for doc in docs:
            record = self._make_record(doc)
            self._next_index += 1
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            result = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            output: list[dict[str, Any]] = []
            for idx, content in enumerate(docs):
                metadata = metas[idx] if idx < len(metas) and metas[idx] is not None else {}
                distance = distances[idx] if idx < len(distances) else 0.0
                output.append(
                    {
                        "content": content,
                        "metadata": metadata,
                        "score": -float(distance),
                    }
                )
            return output

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if top_k <= 0:
            return []

        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            result = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            output: list[dict[str, Any]] = []
            for idx, content in enumerate(docs):
                metadata = metas[idx] if idx < len(metas) and metas[idx] is not None else {}
                distance = distances[idx] if idx < len(distances) else 0.0
                output.append(
                    {
                        "content": content,
                        "metadata": metadata,
                        "score": -float(distance),
                    }
                )
            return output

        filtered: list[dict[str, Any]] = []
        for record in self._store:
            metadata = record.get("metadata", {})
            if all(metadata.get(key) == value for key, value in metadata_filter.items()):
                filtered.append(record)
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            before = int(self._collection.count())
            self._collection.delete(where={"doc_id": doc_id})
            after = int(self._collection.count())
            return after < before

        before = len(self._store)
        self._store = [record for record in self._store if record.get("id") != doc_id]
        return len(self._store) < before
